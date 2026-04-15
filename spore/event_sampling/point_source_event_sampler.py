import numpy as np

from typing import List, Optional, Union

from ..conventions import SkyCoordinate, sky_to_local, ureg
from ..physics import neutrinos, Morphology
from ..source import PointSource
from ..detector import Detector

from .event import Event
from .event_sampler import EventSampler
from .utils import smear_truth_batch, build_adaptive_log_energy_grid, _effa_grid_steady_state


class PointSourceEventSampler(EventSampler):
    """Event sampler for a neutrino point source.

    Builds an inverse-CDF energy sampler from the source flux convolved with
    the detector effective area at the source zenith angle.

    Two modes are available depending on whether ``deltat`` is supplied at
    construction time:

    **Snapshot mode** (``steady_state=False``, default):
        The effective area is evaluated at a single reference epoch ``t``.
        Appropriate for short observations or for detectors at the geographic
        South Pole where the zenith angle is constant.

    **Steady-state mode** (``steady_state=True``):
        The effective area is averaged over a full diurnal cycle,

        .. math::

            \\langle A_\\mathrm{eff}(E) \\rangle = \\frac{1}{2\\pi}
            \\int_0^{2\\pi} A_\\mathrm{eff}(\\zeta(\\delta, H),\\, E)\\, dH,

        using ``n_time_samples`` uniformly spaced hour-angle samples.
        Correct for analyses spanning many sidereal days and for detectors
        at non-polar latitudes where the source transits through a range of
        zenith angles.

    Results are cached per morphology (steady-state) or per (epoch, morphology)
    (snapshot) so repeated pseudo-experiments are fast.

    Args:
        det: The detector to simulate.
        src: The point source to sample from.
        deltat: Observation livetime as a pint Quantity. When provided,
            enables steady-state hour-angle averaging of the effective area.
        n_time_samples: Number of hour-angle samples for the diurnal average.
            Passing this argument implies ``steady_state=True``.  Default 100
            when steady-state mode is active.
        e_min, e_max: Energy range in GeV for the CDF integration grid.
            If None (default), the bounds are read from the detector response.
            Explicit values override the auto-detected bounds.
    """

    def __init__(
        self,
        det: Union[Detector, List[Detector]],
        src: PointSource,
        steady_state: bool = False,
        n_time_samples: int = None,
        n_e: int = 50,
        e_min: float = None,
        e_max: float = None,
    ):
        if isinstance(det, list):
            self._multi = True
            self._samplers = [
                PointSourceEventSampler(
                    d, src,
                    steady_state=steady_state,
                    n_time_samples=n_time_samples,
                    n_e=n_e,
                    e_min=e_min,
                    e_max=e_max,
                )
                for d in det
            ]
            self._src = src
            self._steady_state = steady_state or (n_time_samples is not None)
            self._t0 = 60_355.83
            return

        self._multi = False
        super().__init__(det, src)
        if n_time_samples is not None:
            self._steady_state = True
        else:
            self._steady_state = steady_state
        self._n_time_samples = n_time_samples if n_time_samples is not None else 100

        from .utils import _effa_energy_bounds
        first_morph = sorted(det.response.effective_area)[0]
        first_effa  = det.response.effective_area[first_morph]
        _irf_lo, _irf_hi = _effa_energy_bounds(first_effa)
        _src_lo = getattr(getattr(src, 'flux', None), 'e_min_gev', None)
        _src_hi = getattr(getattr(src, 'flux', None), 'e_max_gev', None)
        _e_min = e_min if e_min is not None else (max(_irf_lo, _src_lo) if _src_lo is not None else _irf_lo)
        _e_max = e_max if e_max is not None else (min(_irf_hi, _src_hi) if _src_hi is not None else _irf_hi)
        log_es = build_adaptive_log_energy_grid(
            n_e, first_effa, src, first_morph, e_min=_e_min, e_max=_e_max,
        )
        self._es = np.clip(np.exp(log_es), _e_min, _e_max)
        self._cache = {}

    def _ha_averaged_effa(self, morphology: str) -> np.ndarray:
        """Return the hour-angle averaged effective area on self._es."""
        dec = self._src.location.declination
        effa_fn = self._det.response.effective_area[morphology]
        return _effa_grid_steady_state(
            np.array([dec]), self._det.location, effa_fn, self._es, self._n_time_samples
        )[0]

    def _build_cdf(self, morphology: str, effas: np.ndarray):
        """Build and cache the (spline, norm) pair for the given effective areas."""
        if Morphology.flavor_set(morphology) == "numu":
            fluxes = np.array([
                self._src(neutrinos[2], e) + self._src(neutrinos[3], e)
                for e in self._es
            ])
        else:
            fluxes = np.array([
                sum(self._src(nu, e) for nu in neutrinos)
                for e in self._es
            ])

        # Trim to energies where the effective area is non-zero.
        zeros = np.where(effas == 0)[0]
        idx = zeros[-1] if len(zeros) > 0 else -1
        es = self._es[idx + 1:]
        effas_trim = effas[idx + 1:]
        fluxes_trim = fluxes[idx + 1:]

        if len(es) == 0:
            return None, 0.0

        from scipy.integrate import quad
        from scipy.interpolate import PchipInterpolator

        integrand = lambda le: np.exp(le) * np.interp(
            np.exp(le), es, effas_trim * fluxes_trim
        )
        cdfs = np.concatenate([
            [0],
            [quad(integrand, np.log(es[0]), np.log(e))[0] for e in es[1:]],
        ])

        # Deduplicate to ensure strict monotonicity.
        cdfs_, es_, last = [], [], None
        for v, e in zip(cdfs, es):
            if last is not None and v <= last:
                continue
            cdfs_.append(v)
            es_.append(e)
            last = v

        cdfs = np.array(cdfs_)
        es = np.array(es_)

        norm = cdfs[-1]
        cdfs = cdfs / cdfs[-1]
        spl = PchipInterpolator(cdfs, np.log(es))
        return spl, norm

    def _ensure_cached(self, morphology: str, t: float):
        """Populate the energy CDF cache if not already done."""
        cache_key = morphology if self._steady_state else (t, morphology)
        if cache_key in self._cache:
            return

        if self._steady_state:
            effas = self._ha_averaged_effa(morphology)
        else:
            lc = sky_to_local(self._src.location, self._det.location, t)
            effa_fn = self._det.response.effective_area[morphology]
            effas = effa_fn(np.full(len(self._es), lc.zenith), self._es)

        self._cache[cache_key] = self._build_cdf(morphology, effas)

    def _get_cached(self, morphology: str, t: float):
        cache_key = morphology if self._steady_state else (t, morphology)
        return self._cache[cache_key]

    def _expected_events_single(
        self,
        morphology: str,
        deltat,
        t: Optional[float] = None,
    ):
        if t is None:
            t = self._t0
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )
        self._ensure_cached(morphology, t)
        _, norm = self._get_cached(morphology, t)
        return norm * deltat.to('s').magnitude

    def _sample_events_single(
        self,
        morphology: str,
        deltat=None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        seed=None,
        delta_clip: tuple = None,
    ):
        """Draw reconstructed events from the point source (single-detector).

        Parameters
        ----------
        morphology : str
        deltat : pint Quantity or None
            Observation livetime.  Sets the Poisson mean when ``nevent`` is
            None, and spreads event times over ``[t, t + deltat]`` whenever
            it is provided.
        nevent : int or None
            Fixed event count.  May be combined with ``deltat`` to fix the
            count while still spreading times over the observation window.
        t : float or None
            Reference epoch in MJD.  Defaults to ``t0``.
        seed : int or None
            Random seed.  Pass different values across pseudo-experiments;
            a fixed seed always produces identical events.
        delta_clip : tuple (lo, hi) or None
            Clips the log-energy smearing Delta = ln(E_reco / E_true) to
            [lo, hi].  Useful for suppressing tail artefacts in validation.

        .. note::
            **Eddington bias**: for steeply falling spectra the reconstructed-
            energy distribution will have more events at low energies and fewer
            at high energies than the unsmeared A_eff × flux integral predicts.
            This is a real physical effect, not a sampling artefact.  To
            quantify it, compare ``delta_clip=(0, 0)`` (no smearing) against
            the default.
        """
        if deltat is None and nevent is None:
            raise ValueError("Specify at least one of deltat or nevent.")

        self._check_steady_state(deltat)
        rng = np.random.default_rng(seed)
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )

        if t is None:
            t = self._t0

        self._ensure_cached(morphology, t)
        spl, norm = self._get_cached(morphology, t)

        if nevent is None:
            nevent = rng.poisson(norm * deltat.to('s').magnitude)

        if nevent == 0 or spl is None:
            return []

        true_energies  = np.exp(spl(rng.random(nevent)))
        true_direction = self._src.location
        true_decs = np.full(nevent, true_direction.declination)
        true_ras  = np.full(nevent, true_direction.right_ascension)
        reco_decs, reco_ras, reco_energies, ang_errs = smear_truth_batch(
            true_decs, true_ras, true_energies, self._det, morphology, rng=rng,
            delta_clip=delta_clip,
        )
        if deltat is not None:
            deltat_days = deltat.to('day').magnitude
            event_times = t + rng.uniform(0.0, deltat_days, nevent)
        else:
            event_times = np.full(nevent, t)
        return [
            Event(
                true_direction,
                SkyCoordinate(rd, rr),
                te,
                re,
                et,
                morphology,
                ang_err=ae,
            )
            for rd, rr, te, re, et, ae in zip(reco_decs, reco_ras, true_energies, reco_energies, event_times, ang_errs)
        ]
