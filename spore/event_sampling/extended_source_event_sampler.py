import logging
import numpy as np

from typing import List, Optional, Union

logger = logging.getLogger(__name__)

from ..conventions import SkyCoordinate
from ..physics import neutrinos, Morphology
from ..detector import Detector

from .event_sampler import EventSampler
from .event import Event
from .utils import (
    smear_truth_batch, zenith_grid, _build_sampling_data,
    build_adaptive_log_energy_grid,
    _effa_grid_from_lst, _sample_from_slice, _assign_times_from_slices,
    _local_coords_from_times, _mjd_to_lst,
    _sample_times_in_slices, _grl_slice_weights, _grl_mean_rate,
    resolve_energy_bounds,
)


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
        uniform_fraction: float = 0.5,
        e_min: float = None,
        e_max: float = None,
    ):
        """
        Args:
            det: A single Detector or a list of Detectors for joint
                multi-detector sampling.  When a list is passed, one
                per-detector sampler is built internally and ``sample_events``
                returns a combined event list with each event tagged by its
                detector index.
            src: The extended source to sample from.
            steady_state: If True, average the effective area over a full
                diurnal cycle using ``n_time_samples`` hour-angle samples.
                Appropriate for analyses spanning many sidereal days.
                Default False.
            n_dec: Number of sin(dec) grid points.  Default 40.
            n_ra: Number of RA grid points.  Default 40.
            n_e: Number of energy grid points.  Default 40.
            n_time_samples: Hour-angle samples for the steady-state A_eff
                average.  Passing this argument implies ``steady_state=True``.
                Default 100 when steady-state mode is active.
            adaptive_energy_grid: If True (default), place energy grid points
                at quantiles of the pilot weight ``A_eff * flux * E``,
                concentrating resolution near the detection threshold where the
                effective area rises steeply.  If False, use uniform log(E)
                spacing.
            uniform_fraction: Weight of the uniform-in-log(E) CDF blended into
                the mass CDF before quantiles are taken, in [0, 1].  Only used
                when ``adaptive_energy_grid`` is True.  0 gives pure equal-mass
                placement, which can leave a single cell spanning more than a
                decade at the depleted end of a steeply falling spectrum; 1
                gives a uniform log grid.  Default 0.5, which bounds the widest
                cell at twice the uniform spacing.
            e_min: Minimum energy in GeV for the sampling grid.  If None, read
                from the detector response IRF.
            e_max: Maximum energy in GeV for the sampling grid.  If None, read
                from the detector response IRF.
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
                    uniform_fraction=uniform_fraction,
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
        e_min, e_max = resolve_energy_bounds(det, src, e_min, e_max)

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
                uniform_fraction=uniform_fraction,
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
            # When all species share the same distribution object (common for
            # isotropic/halo sources), compute the density grid once and scale
            # by per-species normalizations rather than evaluating n_species times.
            _dists = src.flux._distributions
            _norms = src.flux._normalizations
            _global_norm = src.flux._normalization
            _dist_ids = [id(_dists[nu]) for nu in neutrinos]
            if len(set(_dist_ids)) == 1:
                density_grid = _dists[neutrinos[0]].batch_density(es, decs, ras)
                for idx, nu in enumerate(neutrinos):
                    fluxes[idx] = _global_norm * _norms[nu] * density_grid
            else:
                for idx, nu in enumerate(neutrinos):
                    fluxes[idx] = _global_norm * _norms[nu] * _dists[nu].batch_density(es, decs, ras)
        else:
            fluxes = np.zeros((6, n_dec, n_e))
            logger.info("Building flux grid (%d dec × %d energy points)", n_dec, n_e)
            for jdx, dec in enumerate(decs):
                logger.debug("Flux grid: declination slice %d/%d (dec=%.3f rad)", jdx + 1, n_dec, dec)
                for idx, nu in enumerate(neutrinos):
                    fluxes[idx, jdx, :] = src(nu, es, dec, 0.0)

        # --- effective-area and sampling tables per available morphology ---
        # Snapshot mode: compute the zenith grid once (same for every morphology
        # and species) to avoid repeating the expensive astropy transform.
        if not self._steady_state:
            _zeniths  = zenith_grid(decs, ras, det.location, self._t0)
            _zen_flat = np.repeat(_zeniths.ravel(), n_e)
            _e_flat   = np.tile(es, n_dec * n_ra)

        lat = det.location.latitude

        self._d = {}
        for morph, effa in det.response.effective_area.items():
            if self._steady_state:
                # Build one sampling grid per LST slice so that each sampled
                # event is consistent with the instantaneous field of view at
                # the time it is assigned.
                lsts = np.linspace(0, 2 * np.pi, n_time_samples, endpoint=False)
                slice_data = []
                logger.info(
                    "Building steady-state grid for '%s': %d LST slices × "
                    "%d dec × %d RA × %d energy points",
                    morph, n_time_samples, n_dec, n_ra, n_e,
                )
                for lst in lsts:
                    if isinstance(effa, list):
                        target_k = np.zeros((n_dec, n_ra, n_e))
                        for sp_idx, effa_sp in enumerate(effa):
                            grid_sp   = _effa_grid_from_lst(decs, ras, lat, effa_sp, es, lst)
                            flux_slice = fluxes[sp_idx] if _uses_ra else fluxes[sp_idx, :, np.newaxis, :]
                            target_k  += grid_sp * flux_slice
                    else:
                        grid_k = _effa_grid_from_lst(decs, ras, lat, effa, es, lst)
                        if Morphology.flavor_set(morph) == "numu":
                            flux_contrib = fluxes[2] + fluxes[3]
                        else:
                            flux_contrib = fluxes.sum(axis=0)
                        target_k = grid_k * (flux_contrib if _uses_ra else flux_contrib[:, np.newaxis, :])
                    slice_data.append(
                        _build_sampling_data(target_k, es, log_es, sds, sd_edges, ra_edges, le_edges)
                    )
                norms  = np.array([d["norm"] for d in slice_data])
                total  = norms.sum()
                self._d[morph] = {
                    "slices":  slice_data,
                    "norms":   norms,
                    "weights": norms / total if total > 0 else np.ones(n_time_samples) / n_time_samples,
                    "norm":    float(norms.mean()),
                }
            else:
                # Snapshot mode: single grid at the reference epoch.
                if isinstance(effa, list):
                    target = np.zeros((n_dec, n_ra, n_e))
                    for sp_idx, effa_sp in enumerate(effa):
                        grid_sp    = effa_sp(_zen_flat, _e_flat).reshape(n_dec, n_ra, n_e)
                        flux_slice = fluxes[sp_idx] if _uses_ra else fluxes[sp_idx, :, np.newaxis, :]
                        target    += grid_sp * flux_slice
                else:
                    grid = effa(_zen_flat, _e_flat).reshape(n_dec, n_ra, n_e)
                    if Morphology.flavor_set(morph) == "numu":
                        flux_contrib = fluxes[2] + fluxes[3]
                    else:
                        flux_contrib = fluxes.sum(axis=0)
                    target = grid * (flux_contrib if _uses_ra else flux_contrib[:, np.newaxis, :])
                self._d[morph] = _build_sampling_data(
                    target, es, log_es, sds, sd_edges, ra_edges, le_edges
                )

    def _expected_events_single(
        self,
        morphology: str,
        deltat,
        t: Optional[float] = None,
        grl=None,
    ):
        if morphology not in self._d:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {list(self._d.keys())}."
            )
        return self._mean_rate(self._d[morphology], grl) * deltat.to('s').magnitude

    def _mean_rate(self, d, grl):
        """Mean rate [s^-1]: exposure-weighted across LST slices under a GRL."""
        if grl is None or "slices" not in d:
            return d["norm"]
        return _grl_mean_rate(d["norms"], grl, self._det.location.longitude)

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
        """Draw a set of reconstructed events from the source (single-detector).

        Args:
            morphology: Event morphology, e.g. "track" or "cascade".
            deltat: Observation window as a pint Quantity with time units.
                Controls the Poisson mean when ``nevent`` is None, and spreads
                event times over ``[t, t + deltat]`` whenever it is provided.
            nevent: Fixed number of events.  When provided alongside
                ``deltat``, the count is fixed but times are still spread over
                the window.  At least one of ``deltat`` or ``nevent`` must be
                given.
            t: Reference epoch in MJD.  Used to rotate RA for Earth's rotation
                since ``t0``.  Defaults to ``t0``.
            seed: Random seed.
            delta_clip: If provided, clip the log-energy smearing variable
                Delta = ln(E_reco / E_true) to ``(lo, hi)`` before applying to
                true energies.  Useful for suppressing extreme tails of the
                energy resolution during validation.

        Note:
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

        Returns:
            List of Event objects.
        """
        if nevent is None and deltat is None:
            raise ValueError("Specify at least one of deltat or nevent.")

        self._check_steady_state(deltat)
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )

        rng = np.random.default_rng(seed)
        d = self._d[morphology]
        if nevent is None:
            nevent = rng.poisson(self._mean_rate(d, grl) * deltat.to('s').magnitude)
        if t is None:
            t = self._t0

        n = int(nevent)
        if n == 0:
            return []

        if self._steady_state and "slices" in d:
            # --- Time-slice steady-state sampling ---
            # Each event is assigned to a sidereal-phase slice; position and
            # energy are drawn from that slice's instantaneous sensitivity grid
            # so the sky direction is consistent with the field of view at the
            # assigned time.
            slice_data = d["slices"]
            weights    = d["weights"]
            n_slices   = len(slice_data)

            lon = self._det.location.longitude
            if grl is not None:
                # Choose the LST slice from rate x exposure, then draw a time
                # from the run list conditioned on that slice.
                k_arr = rng.choice(
                    n_slices, size=n,
                    p=_grl_slice_weights(d["norms"], grl, lon),
                )
                event_times = _sample_times_in_slices(grl, k_arr, n_slices, rng, lon)
            else:
                k_arr = rng.choice(n_slices, size=n, p=weights)
                if deltat is not None:
                    event_times = _assign_times_from_slices(
                        t, deltat, k_arr, n_slices, rng, longitude=lon,
                    )
                else:
                    event_times = np.full(n, t)

            sin_dec_arr = np.empty(n)
            ra_arr      = np.empty(n)
            log_e_arr   = np.empty(n)
            for k in range(n_slices):
                mask = k_arr == k
                n_k  = int(mask.sum())
                if n_k == 0:
                    continue
                sd_k, ra_k, le_k = _sample_from_slice(slice_data[k], n_k, rng)
                sin_dec_arr[mask] = sd_k
                ra_arr[mask]      = ra_k
                log_e_arr[mask]   = le_k
        else:
            # --- Snapshot mode ---
            cdf_dec            = d["cdf_dec"]
            cdf_e_given_dec    = d["cdf_e_given_dec"]
            cdf_ra_given_dec_e = d["cdf_ra_given_dec_e"]
            sd_edges           = d["sd_edges"]
            ra_edges           = d["ra_edges"]
            le_edges           = d["le_edges"]

            n_dec = len(cdf_dec)
            n_e   = cdf_e_given_dec.shape[1]
            n_ra  = cdf_ra_given_dec_e.shape[2]

            i_dec = np.clip(np.searchsorted(cdf_dec, rng.random(n)), 0, n_dec - 1)
            cdfs_e = cdf_e_given_dec[i_dec]
            i_e = np.clip(np.argmax(cdfs_e >= rng.random(n)[:, None], axis=1), 0, n_e - 1)
            cdfs_ra = cdf_ra_given_dec_e[i_dec, i_e]
            i_ra = np.clip(np.argmax(cdfs_ra >= rng.random(n)[:, None], axis=1), 0, n_ra - 1)

            sin_dec_arr = rng.uniform(sd_edges[i_dec], sd_edges[i_dec + 1])
            ra_arr      = rng.uniform(ra_edges[i_ra],  ra_edges[i_ra  + 1])
            log_e_arr   = rng.uniform(le_edges[i_e],   le_edges[i_e   + 1])

            # The snapshot grid is built at t0, so rotate sampled RAs by the
            # Earth rotation between t0 and t.  That rotation is the change in
            # local sidereal time, not the fraction of a *solar* day elapsed:
            # the two drift apart by ~1 deg/day and are a half-turn out after
            # six months.
            lon     = self._det.location.longitude
            offset  = (_mjd_to_lst(t, lon) - _mjd_to_lst(self._t0, lon)) % (2.0 * np.pi)
            ra_arr  = np.mod(ra_arr + offset, 2.0 * np.pi)

            if grl is not None:
                event_times = grl.sample_times(n, rng)
            elif deltat is not None:
                event_times = t + rng.uniform(0.0, deltat.to('day').magnitude, n)
            else:
                event_times = np.full(n, t)

        true_decs_arr     = np.arcsin(np.clip(sin_dec_arr, -1.0, 1.0))
        true_energies_arr = np.exp(log_e_arr)
        local_zeniths, local_azimuths = _local_coords_from_times(
            event_times, true_decs_arr, ra_arr,
            self._det.location.latitude, self._det.location.longitude,
        )
        reco_decs, reco_ras, reco_energies, ang_errs = smear_truth_batch(
            true_decs_arr, ra_arr, true_energies_arr, self._det, morphology, rng=rng,
            delta_clip=delta_clip, local_zeniths=local_zeniths,
        )
        return [
            Event(
                SkyCoordinate(td, tr),
                SkyCoordinate(rd, rr),
                te, re, et, morphology, ang_err=ae,
                zenith=ze, azimuth=az,
            )
            for td, tr, rd, rr, te, re, et, ae, ze, az in zip(
                true_decs_arr, ra_arr, reco_decs, reco_ras,
                true_energies_arr, reco_energies, event_times, ang_errs,
                local_zeniths, local_azimuths,
            )
        ]
