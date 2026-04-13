import logging
import numpy as np

from typing import List, Optional, Union

logger = logging.getLogger(__name__)

from ..conventions import SkyCoordinate, ureg
from ..physics import neutrinos, Morphology
from ..source import ExtendedSource
from ..detector import Detector

from .event_sampler import EventSampler
from .event import Event
from .utils import smear_truth_batch, zenith_grid, _build_sampling_data, sample_t_event, build_adaptive_log_energy_grid



def _effa_grid_steady_state(decs, earth_coord, effa_fn, es, n_ha_samples):
    """
    Compute the time-averaged effective area at each (dec, E) grid point
    by integrating analytically over hour angle.

        <A_eff(dec, E)> = (1/2pi) * integral_0^{2pi} A_eff(zeta(dec, HA), E) dHA

    cos(zeta) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(HA)

    Parameters
    ----------
    decs        : (n_dec,)  Declinations in radians.
    earth_coord : EarthCoordinate (only latitude is used).
    effa_fn     : callable  effa_fn(zenith_rad, energy_GeV) -> float
    es          : (n_e,)    Energies in internal units (GeV).
    n_ha_samples: int       Hour-angle samples over [0, 2pi].

    Returns
    -------
    effa_avg : ndarray, shape (n_dec, n_e)
    """
    lat = earth_coord.latitude
    has = np.linspace(0, 2 * np.pi, n_ha_samples, endpoint=False)
    effa_avg = np.zeros((len(decs), len(es)))
    for jdx, dec in enumerate(decs):
        cos_zens = (
            np.sin(lat) * np.sin(dec)
            + np.cos(lat) * np.cos(dec) * np.cos(has)
        )
        zens = np.arccos(np.clip(cos_zens, -1.0, 1.0))  # (n_ha,)
        zen_g, e_g = np.meshgrid(zens, es, indexing='ij')
        effa_avg[jdx] = effa_fn(zen_g.ravel(), e_g.ravel()).reshape(len(zens), len(es)).mean(axis=0)
    return effa_avg


class ExtendedSourceEventSampler(EventSampler):
    """
    Sample events from a spatially extended (or all-sky) astrophysical source.

    Uses hierarchical inverse-CDF sampling on a precomputed 3-D grid of
    A_eff × flux over (sin_dec, RA, log_E).  The joint distribution is
    factored as::

        p(sin_dec, RA, log_E) = p(sin_dec) * p(log_E | sin_dec) * p(RA | sin_dec, log_E)

    and each factor is sampled via its discrete inverse CDF.  This handles
    multimodal distributions (e.g. upgoing vs. downgoing tracks at different
    energies) exactly without burnin or mode-trapping, for any detector
    geometry.

    Grid resolution is controlled by ``n_dec``, ``n_ra``, and ``n_e``.
    Increasing these reduces discretisation error at the cost of memory and
    initialisation time.

    Two A_eff modes:

    **Transient** (``steady_state=False``, default)
        Effective area evaluated at the reference epoch ``t0`` via a single
        vectorised astropy transform.  Use for short observations (< 1 day).

    **Steady-state** (``steady_state=True``)
        Effective area averaged over a full diurnal cycle using
        ``n_time_samples`` hour-angle samples.  Correct for observations
        spanning many sidereal days.
    """

    def __init__(
        self,
        det: Union[Detector, List[Detector]],
        src,
        steady_state: bool = False,
        n_dec: int = 40,
        n_ra: int = 40,
        n_e: int = 40,
        n_time_samples: int = None,
        adaptive_energy_grid: bool = True,
        e_min: float = None,
        e_max: float = None,
    ):
        """
        Parameters
        ----------
        det : Detector or list of Detector
            A single detector or a list of detectors for joint multi-detector
            sampling.  When a list is passed, one per-detector sampler is
            built internally and ``sample_events`` returns a combined event list
            with each event tagged by its detector index.
        src : ExtendedSource
        steady_state : bool
            If True, average the effective area over a full diurnal cycle
            using ``n_time_samples`` hour-angle samples.  Appropriate for
            analyses spanning many sidereal days.  Default False.
        n_dec : int
            Number of sin(dec) grid points.  Default 40.
        n_ra : int
            Number of RA grid points.  Default 40.
        n_e : int
            Number of energy grid points.  Default 40.
        n_time_samples : int
            Hour-angle samples for the steady-state A_eff average.  Passing
            this argument implies ``steady_state=True``.  Default 100 when
            steady-state mode is active.
        adaptive_energy_grid : bool
            If True (default), place energy grid points at quantiles of the
            pilot weight ``A_eff * flux * E``, concentrating resolution near
            the detection threshold where the effective area rises steeply.
            If False, use uniform log(E) spacing.
        e_min, e_max : float or None
            Energy range in GeV for the sampling grid.  If None (default), the
            bounds are read from the detector response (the energy grid stored
            in the IRF).  Explicit values override the auto-detected bounds and
            can be used to restrict or extend the sampled range.
        """
        if isinstance(det, list):
            self._multi = True
            self._samplers = [
                ExtendedSourceEventSampler(
                    d, src,
                    steady_state=steady_state,
                    n_dec=n_dec, n_ra=n_ra, n_e=n_e,
                    n_time_samples=n_time_samples,
                    adaptive_energy_grid=adaptive_energy_grid,
                    e_min=e_min, e_max=e_max,
                )
                for d in det
            ]
            self._src = src
            self._steady_state = steady_state or (n_time_samples is not None)
            self._t0 = 60_355.83
            return

        self._multi = False
        _use_steady_state = steady_state or (n_time_samples is not None)
        n_time_samples = n_time_samples if n_time_samples is not None else 100
        super().__init__(det, src)
        # Override the base-class default after super().__init__ so it isn't clobbered.
        if _use_steady_state:
            self._steady_state = True

        _uses_ra = getattr(src, 'uses_ra', False)
        if _uses_ra and getattr(src, 'is_monochromatic', False):
            raise ValueError(
                "Monochromatic sources are not supported by ExtendedSourceEventSampler. "
                "Use a continuous spectrum (BoxSpectrum or SoftSpectrum) instead."
            )

        # --- resolve energy bounds from IRF ∩ flux if not provided ---
        from .utils import _effa_energy_bounds
        if e_min is None or e_max is None:
            first_morph = sorted(det.response.effective_area)[0]
            first_effa  = det.response.effective_area[first_morph]
            _irf_lo, _irf_hi = _effa_energy_bounds(first_effa)
            _src_lo = getattr(getattr(src, 'flux', None), 'e_min_gev', None)
            _src_hi = getattr(getattr(src, 'flux', None), 'e_max_gev', None)
            if e_min is None:
                e_min = max(_irf_lo, _src_lo) if _src_lo is not None else _irf_lo
            if e_max is None:
                e_max = min(_irf_hi, _src_hi) if _src_hi is not None else _irf_hi

        # --- coordinate grids (cell centres) ---
        sds  = np.linspace(-1.0, 1.0, n_dec)
        decs = np.arcsin(sds)
        ras  = np.linspace(0.0, 2.0 * np.pi, n_ra, endpoint=False)

        if adaptive_energy_grid:
            # Use the first available morphology's effective area to build the
            # pilot weight; threshold shape is similar across morphologies.
            first_morph = sorted(det.response.effective_area)[0]
            first_effa  = det.response.effective_area[first_morph]
            log_es = build_adaptive_log_energy_grid(
                n_e, first_effa, src, first_morph,
                e_min=e_min, e_max=e_max,
            )
        else:
            log_es = np.linspace(np.log(e_min), np.log(e_max), n_e)

        es = np.clip(np.exp(log_es), e_min, e_max)

        # --- cell edges for jittering ---
        dsd = (sds[-1] - sds[0]) / (n_dec - 1) if n_dec > 1 else 2.0
        sd_edges = np.clip(
            np.concatenate([[sds[0] - dsd / 2],
                            (sds[:-1] + sds[1:]) / 2,
                            [sds[-1] + dsd / 2]]),
            -1.0, 1.0,
        )
        ra_edges = np.linspace(0.0, 2.0 * np.pi, n_ra + 1)
        le_edges = np.concatenate([[log_es[0]],
                                   (log_es[:-1] + log_es[1:]) / 2,
                                   [log_es[-1]]])

        self._sds    = sds
        self._ras    = ras
        self._decs   = decs
        self._es     = es
        self._log_es = log_es

        # --- flux grid ---
        # For RA-independent sources: shape (6, n_dec, n_e).
        # For RA-dependent sources (e.g. galactic halo): shape (6, n_dec, n_ra, n_e).
        if _uses_ra:
            fluxes = np.zeros((6, n_dec, n_ra, n_e))
            logger.info("Building flux grid (%d dec × %d RA × %d energy points)", n_dec, n_ra, n_e)
            for jdx, dec in enumerate(decs):
                logger.debug("Flux grid: declination slice %d/%d (dec=%.3f rad)", jdx + 1, n_dec, dec)
                for kdx, ra in enumerate(ras):
                    for idx, nu in enumerate(neutrinos):
                        fluxes[idx, jdx, kdx, :] = src(nu, es, dec, ra)
        else:
            fluxes = np.zeros((6, n_dec, n_e))
            logger.info("Building flux grid (%d dec × %d energy points)", n_dec, n_e)
            for jdx, dec in enumerate(decs):
                logger.debug("Flux grid: declination slice %d/%d (dec=%.3f rad)", jdx + 1, n_dec, dec)
                for idx, nu in enumerate(neutrinos):
                    fluxes[idx, jdx, :] = src(nu, es, dec)

        # --- effective-area and sampling tables per available morphology ---
        # Compute the zenith grid once for transient mode — it's the same for
        # every morphology and every species, so there is no reason to repeat
        # the expensive astropy coordinate transform inside the loop.
        if not self._steady_state:
            _zeniths = zenith_grid(decs, ras, det.location, self._t0)
            _zen_flat = np.repeat(_zeniths.ravel(), n_e)
            _e_flat   = np.tile(es, n_dec * n_ra)
        else:
            _zen_flat = _e_flat = None

        self._d = {}
        for morph, effa in det.response.effective_area.items():
            if isinstance(effa, list):
                # Per-species A_effs: target = Σ_sp A_eff_sp × phi_sp
                # Correctly handles non-democratic fluxes (e.g. Honda+Gaisser)
                # where different species have very different detection efficiencies.
                target = np.zeros((n_dec, n_ra, n_e))
                for sp_idx, effa_sp in enumerate(effa):
                    if self._steady_state:
                        avg_sp = _effa_grid_steady_state(
                            decs, det.location, effa_sp, es, n_time_samples
                        )
                        grid_sp = avg_sp[:, np.newaxis, :]
                    else:
                        grid_sp = effa_sp(_zen_flat, _e_flat).reshape(n_dec, n_ra, n_e)
                    flux_slice = fluxes[sp_idx] if _uses_ra else fluxes[sp_idx, :, np.newaxis, :]
                    target += grid_sp * flux_slice
            else:
                # Legacy single A_eff: assumes species-independent detection.
                # Valid for democratic fluxes (astrophysical power law) but not
                # for atmospheric fluxes with unequal species contributions.
                grid = np.zeros((n_dec, n_ra, n_e))
                if self._steady_state:
                    avg = _effa_grid_steady_state(
                        decs, det.location, effa, es, n_time_samples
                    )
                    grid[:] = avg[:, np.newaxis, :]
                else:
                    grid = effa(_zen_flat, _e_flat).reshape(n_dec, n_ra, n_e)

                if Morphology.flavor_set(morph) == "numu":
                    flux_contrib = fluxes[2] + fluxes[3]
                else:
                    flux_contrib = fluxes.sum(axis=0)

                if _uses_ra:
                    target = grid * flux_contrib
                else:
                    target = grid * flux_contrib[:, np.newaxis, :]

            self._d[morph] = _build_sampling_data(
                target, es, log_es, sds, sd_edges, ra_edges, le_edges
            )

    def expected_events(
        self,
        morphology: str,
        deltat: float,
        t: Optional[float] = None,
    ):
        if self._multi:
            deltats = self._resolve_per_detector(deltat, "deltat")
            ts      = self._resolve_per_detector(t,      "t")
            return [s.expected_events(morphology, dt, t=t_) for s, dt, t_ in zip(self._samplers, deltats, ts)]
        if morphology not in self._d:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {list(self._d.keys())}."
            )
        return self._d[morphology]["norm"] * deltat.to('s').magnitude

    def sample_events(
        self,
        morphology: str,
        deltat: Optional[float] = None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        seed=None,
        delta_clip: tuple = None,
    ):
        """
        Draw a set of reconstructed events from the source.

        Parameters
        ----------
        morphology : {"track", "cascade"}
        t : float or None
            Reference epoch in MJD.  Used to rotate RA for Earth's rotation
            since ``t0``.  Defaults to ``t0``.
        nevent : int or None
            Fixed number of events.  Mutually exclusive with ``deltat``.
        deltat : pint Quantity or None
            Observation window with time units.  The number of events is drawn
            from Poisson(norm * deltat).
        delta_clip : tuple (lo, hi) or None
            If provided, the log-energy smearing variable
            Delta = ln(E_reco / E_true) is clipped to [lo, hi] before
            applying to true energies.  Useful for suppressing extreme tails
            of the energy resolution during validation.

        .. note::
            **Eddington bias**: the reconstructed-energy distribution can
            differ substantially from a naive A_eff × flux integral over true
            energies.  For steeply falling spectra (γ > 2) combined with the
            systematic downward shift of the HESE track energy resolution
            (median Δ ≈ −0.65, so E_reco ≈ 0.5 × E_true), events from
            higher true-energy bins are redistributed to lower reconstructed
            energies.  This is physically correct behavior, not a sampling
            artefact.  To isolate the effect, compare distributions from
            ``delta_clip=(0, 0)`` (no smearing, E_reco = E_true) against the
            default.

        Returns
        -------
        list of Event
        """
        if not ((nevent is None) ^ (deltat is None)):
            raise ValueError("Exactly one of nevent or deltat must be provided.")

        if self._multi:
            deltats     = self._resolve_per_detector(deltat,     "deltat")
            ts          = self._resolve_per_detector(t,          "t")
            delta_clips = self._resolve_per_detector(delta_clip, "delta_clip")
            rng = np.random.default_rng(seed)
            all_events = []
            for idx, (sampler, dt, t_, dc) in enumerate(zip(self._samplers, deltats, ts, delta_clips)):
                events = sampler.sample_events(
                    morphology, deltat=dt, nevent=nevent, t=t_,
                    seed=int(rng.integers(2**32)), delta_clip=dc,
                )
                for ev in events:
                    ev.detector_id = idx
                all_events.extend(events)
            return all_events

        self._check_steady_state(deltat)
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )

        rng = np.random.default_rng(seed)
        d = self._d[morphology]
        if nevent is None:
            nevent = rng.poisson(d["norm"] * deltat.to('s').magnitude)
        if t is None:
            t = self._t0

        n = int(nevent)
        if n == 0:
            return []

        cdf_dec            = d["cdf_dec"]
        cdf_e_given_dec    = d["cdf_e_given_dec"]
        cdf_ra_given_dec_e = d["cdf_ra_given_dec_e"]
        sd_edges           = d["sd_edges"]
        ra_edges           = d["ra_edges"]
        le_edges           = d["le_edges"]

        n_dec = len(cdf_dec)
        n_e   = cdf_e_given_dec.shape[1]
        n_ra  = cdf_ra_given_dec_e.shape[2]

        # --- Step 1: sample sin_dec bins from the marginal CDF ---
        i_dec = np.clip(
            np.searchsorted(cdf_dec, rng.random(n)),
            0, n_dec - 1,
        )

        # --- Step 2: sample log_E bins from the conditional CDF given sin_dec ---
        cdfs_e = cdf_e_given_dec[i_dec]          # (n, n_e)
        i_e = np.clip(
            np.argmax(cdfs_e >= rng.random(n)[:, None], axis=1),
            0, n_e - 1,
        )

        # --- Step 3: sample RA bins from the conditional CDF given (sin_dec, log_E) ---
        cdfs_ra = cdf_ra_given_dec_e[i_dec, i_e]  # (n, n_ra)
        i_ra = np.clip(
            np.argmax(cdfs_ra >= rng.random(n)[:, None], axis=1),
            0, n_ra - 1,
        )

        # --- Step 4: jitter uniformly within each selected cell ---
        sin_dec = rng.uniform(sd_edges[i_dec],     sd_edges[i_dec + 1])
        ra      = rng.uniform(ra_edges[i_ra],      ra_edges[i_ra + 1])
        log_e   = rng.uniform(le_edges[i_e],       le_edges[i_e + 1])

        # RA rotation for Earth's rotation since t0
        offset = 2.0 * np.pi * ((t - self._t0) % 1)
        ra = np.mod(ra + offset, 2.0 * np.pi)

        true_decs_arr     = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
        true_energies_arr = np.exp(log_e)
        reco_decs, reco_ras, reco_energies, ang_errs = smear_truth_batch(
            true_decs_arr, ra, true_energies_arr, self._det, morphology, rng=rng,
            delta_clip=delta_clip,
        )
        if deltat is not None:
            deltat_days = deltat.to('day').magnitude
            event_times = t + rng.uniform(-deltat_days / 2, deltat_days / 2, n)
        else:
            event_times = np.full(n, t)
        return [
            Event(
                SkyCoordinate(td, tr),
                SkyCoordinate(rd, rr),
                te,
                re,
                et,
                morphology,
                ang_err=ae,
            )
            for td, tr, rd, rr, te, re, et, ae in zip(
                true_decs_arr, ra, reco_decs, reco_ras, true_energies_arr, reco_energies, event_times, ang_errs
            )
        ]
