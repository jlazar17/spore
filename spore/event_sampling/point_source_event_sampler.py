import numpy as np

from typing import List, Optional, Union

from ..conventions import SkyCoordinate, sky_to_local
from ..physics import neutrinos, Morphology
from ..source import PointSource
from ..detector import Detector

from .event import Event
from .event_sampler import EventSampler
from .utils import (
    smear_truth_batch,
    _assign_times_from_slices,
    _local_coords_from_times, _sample_times_in_slices,
    _grl_slice_weights, _grl_mean_rate, resolve_energy_bounds,
)
from ..detector.detector_response.utils import _largest_contiguous_nonzero


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

        _e_min, _e_max = resolve_energy_bounds(det, src, e_min, e_max)
        self._es = np.clip(
            np.exp(np.linspace(np.log(_e_min), np.log(_e_max), n_e)), _e_min, _e_max
        )
        self._cache = {}

    def _eval_effa(self, effa_fn, zen: float) -> np.ndarray:
        """Evaluate an effective area on the energy grid at fixed zenith.

        Returns a (n_e,) array for a single-species response, or a (6, n_e)
        array when the response file resolves the six neutrino species
        separately (as the HESE 7.5-yr release does).
        """
        zens = np.full(len(self._es), float(zen))
        if isinstance(effa_fn, list):
            return np.array([fn(zens, self._es) for fn in effa_fn])
        return effa_fn(zens, self._es)

    def _build_cdf(self, morphology: str, effas: np.ndarray):
        """Build and cache the (spline, norm) pair for the given effective areas.

        ``effas`` is either a 1-D array over the energy grid (single-species
        response) or a (6, n_e) array (per-species response), in which case
        each species is weighted by its own effective area rather than by the
        morphology-level flavour heuristic.
        """
        effas = np.asarray(effas)
        if effas.ndim == 2:
            # Per-species response: sum_species A_eff^(s)(E) * Phi^(s)(E).
            vals = np.zeros(len(self._es))
            for idx_sp, nu in enumerate(neutrinos):
                flux_sp = np.array([self._src(nu, e) for e in self._es])
                vals += effas[idx_sp] * flux_sp
            effas_1d = effas.sum(axis=0)
        else:
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
            vals = effas * fluxes
            effas_1d = effas

        # Restrict to the largest contiguous run of non-zero effective area.
        # Using the last zero index instead would discard the whole valid
        # range whenever the response also has trailing zeros above its
        # sensitivity ceiling.
        run = _largest_contiguous_nonzero(effas_1d)
        if run is None:
            return None, 0.0
        lo, hi = run
        es = self._es[lo:hi + 1]
        _vals_trim = vals[lo:hi + 1]

        if len(es) < 2:
            return None, 0.0

        from scipy.integrate import quad
        from scipy.interpolate import PchipInterpolator

        _log_es = np.log(es)
        _vals   = _vals_trim
        integrand = lambda le: np.exp(le) * np.interp(le, _log_es, _vals)
        cdfs = np.concatenate([
            [0],
            [quad(integrand, np.log(es[0]), np.log(e), limit=200)[0] for e in es[1:]],
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
            lat    = self._det.location.latitude
            dec    = self._src.location.declination
            ra_src = self._src.location.right_ascension
            effa_fn = self._det.response.effective_area[morphology]
            has     = np.linspace(0, 2 * np.pi, self._n_time_samples, endpoint=False)
            slice_cdfs, norms = [], []
            for ha in has:
                zen = float(np.arccos(np.clip(
                    np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha),
                    -1.0, 1.0,
                )))
                effas = self._eval_effa(effa_fn, zen)
                spl, norm = self._build_cdf(morphology, effas)
                slice_cdfs.append(spl)
                norms.append(norm)
            norms   = np.array(norms)
            total   = norms.sum()
            weights = norms / total if total > 0 else np.ones(len(norms)) / len(norms)
            self._cache[cache_key] = {
                "slice_cdfs": slice_cdfs,
                "norms":      norms,
                "weights":    weights,
                "norm":       float(norms.mean()),
                # RA of the source — used to convert LST to hour angle for GRL mode.
                "_ra_src":    ra_src,
            }
        else:
            lc = sky_to_local(self._src.location, self._det.location, t)
            effa_fn = self._det.response.effective_area[morphology]
            effas = self._eval_effa(effa_fn, lc.zenith)
            self._cache[cache_key] = self._build_cdf(morphology, effas)

    def _get_cached(self, morphology: str, t: float):
        cache_key = morphology if self._steady_state else (t, morphology)
        return self._cache[cache_key]

    def _expected_events_single(
        self,
        morphology: str,
        deltat,
        t: Optional[float] = None,
        grl=None,
    ):
        if t is None:
            t = self._t0
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )
        self._ensure_cached(morphology, t)
        cached = self._get_cached(morphology, t)
        norm = self._mean_rate(cached, grl)
        return norm * deltat.to('s').magnitude

    def _mean_rate(self, cached, grl):
        """Mean rate [s^-1]: exposure-weighted across slices when a GRL is given."""
        if not isinstance(cached, dict):
            return cached[1]
        if grl is None:
            return cached["norm"]
        return _grl_mean_rate(
            cached["norms"], grl, self._det.location.longitude,
            phase_ref=cached["_ra_src"],
        )

    def _sample_events_single(
        self,
        morphology: str,
        deltat=None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        seed=None,
        delta_clip: tuple = None,
        grl=None,
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
        cached = self._get_cached(morphology, t)

        if isinstance(cached, dict):
            # Steady-state: per-slice CDFs.
            slice_cdfs = cached["slice_cdfs"]
            weights    = cached["weights"]
            n_slices   = len(slice_cdfs)
            mean_norm  = cached["norm"]
            ra_src     = cached["_ra_src"]

            if nevent is None:
                rate = self._mean_rate(cached, grl)
                nevent = rng.poisson(rate * deltat.to('s').magnitude)
            if nevent == 0:
                return []

            lon = self._det.location.longitude
            if grl is not None:
                # Choose the hour-angle slice from rate x exposure, then draw a
                # time from the run list conditioned on that slice.
                k_arr = rng.choice(
                    n_slices, size=nevent,
                    p=_grl_slice_weights(cached["norms"], grl, lon,
                                         phase_ref=ra_src),
                )
                event_times = _sample_times_in_slices(
                    grl, k_arr, n_slices, rng, lon, phase_ref=ra_src,
                )
            else:
                k_arr = rng.choice(n_slices, size=nevent, p=weights)
                if deltat is not None:
                    event_times = _assign_times_from_slices(
                        t, deltat, k_arr, n_slices, rng,
                        longitude=lon, phase_ref=ra_src,
                    )
                else:
                    event_times = np.full(nevent, t)

            true_energies = np.empty(nevent)
            for k in range(n_slices):
                mask = k_arr == k
                n_k  = int(mask.sum())
                if n_k == 0:
                    continue
                spl_k = slice_cdfs[k]
                true_energies[mask] = (
                    np.exp(spl_k(rng.random(n_k))) if spl_k is not None
                    else np.full(n_k, self._es[0])
                )
        else:
            # Snapshot mode: single CDF.
            spl, norm = cached
            if nevent is None:
                nevent = rng.poisson(norm * deltat.to('s').magnitude)
            if nevent == 0 or spl is None:
                return []
            true_energies = np.exp(spl(rng.random(nevent)))
            if grl is not None:
                event_times = grl.sample_times(nevent, rng)
            elif deltat is not None:
                event_times = t + rng.uniform(0.0, deltat.to('day').magnitude, nevent)
            else:
                event_times = np.full(nevent, t)

        true_direction = self._src.location
        true_decs = np.full(nevent, true_direction.declination)
        true_ras  = np.full(nevent, true_direction.right_ascension)
        local_zeniths, local_azimuths = _local_coords_from_times(
            event_times, true_decs, true_ras,
            self._det.location.latitude, self._det.location.longitude,
        )
        reco_decs, reco_ras, reco_energies, ang_errs = smear_truth_batch(
            true_decs, true_ras, true_energies, self._det, morphology, rng=rng,
            delta_clip=delta_clip, local_zeniths=local_zeniths,
        )
        return [
            Event(
                true_direction,
                SkyCoordinate(rd, rr),
                te, re, et, morphology, ang_err=ae,
                zenith=ze, azimuth=az,
            )
            for rd, rr, te, re, et, ae, ze, az in zip(
                reco_decs, reco_ras, true_energies, reco_energies, event_times, ang_errs,
                local_zeniths, local_azimuths,
            )
        ]
