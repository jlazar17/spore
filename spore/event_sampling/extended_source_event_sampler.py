import numpy as np

from tqdm import tqdm
from typing import Optional

from ..conventions import units, SkyCoordinate
from ..physics import neutrinos
from ..source import ExtendedSource
from ..detector import Detector

from .event_sampler import EventSampler
from .event import Event
from .utils import smear_truth, zenith_grid



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
    effa_fn     : callable  effa_fn(zenith_rad, energy_eV) -> float
    es          : (n_e,)    Energies in internal units (eV).
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
        zen_grid = np.repeat(zens[:, np.newaxis], len(es), axis=1)  # (n_ha, n_e)
        e_grid   = np.tile(es[np.newaxis, :], (len(zens), 1))       # (n_ha, n_e)
        effa_avg[jdx] = effa_fn(zen_grid.ravel(), e_grid.ravel()).reshape(len(zens), len(es)).mean(axis=0)
    return effa_avg


def _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges):
    """
    Precompute hierarchical inverse-CDF tables from a 3-D target grid.

    The joint distribution is factored as:

        p(sin_dec, RA, log_E) = p(sin_dec) * p(log_E | sin_dec) * p(RA | sin_dec, log_E)

    Each factor is stored as a discrete CDF so that sampling via
    ``np.searchsorted`` is exact (up to grid resolution) and handles any
    number of modes without burnin or mode-trapping.

    Parameters
    ----------
    target   : ndarray (n_dec, n_ra, n_e)   A_eff × flux on the grid.
    es       : (n_e,)   Energies in eV (cell centres).
    log_es   : (n_e,)   log(E) (cell centres).
    sds      : (n_dec,) sin(dec) cell centres.
    sd_edges : (n_dec+1,) sin(dec) cell edges.
    ra_edges : (n_ra+1,)  RA cell edges in radians.
    le_edges : (n_e+1,)   log(E) cell edges.

    Returns
    -------
    dict with keys:
        norm                  float      Total integrated rate [eV].
        cdf_dec               (n_dec,)   Marginal CDF over sin_dec.
        cdf_e_given_dec       (n_dec, n_e)   Conditional CDF over log_E | sin_dec.
        cdf_ra_given_dec_e    (n_dec, n_e, n_ra)  Conditional CDF over RA | (sin_dec, log_E).
        sd_edges, ra_edges, le_edges   passed through for use in jittering.
    """
    n_dec, n_ra, n_e = target.shape

    # Actual cell widths from edges — handles non-uniform boundary cells correctly.
    sd_widths = np.diff(sd_edges)   # (n_dec,)
    ra_widths = np.diff(ra_edges)   # (n_ra,)
    le_widths = np.diff(le_edges)   # (n_e,)

    # Per-cell weight: A_eff * flux * E  (integrand of the rate integral in log_E)
    w = target * es[np.newaxis, np.newaxis, :]   # (n_dec, n_ra, n_e)

    # Total rate: sum of (weight × cell volume) over all cells.
    cell_vols = (
        sd_widths[:, np.newaxis, np.newaxis]
        * ra_widths[np.newaxis, :, np.newaxis]
        * le_widths[np.newaxis, np.newaxis, :]
    )
    norm = float((w * cell_vols).sum())

    # --- Marginal CDF over sin_dec ---
    w_dec = w.sum(axis=(1, 2))                   # (n_dec,)
    total_dec = w_dec.sum()
    if total_dec > 0:
        cdf_dec = np.cumsum(w_dec / total_dec)
    else:
        cdf_dec = np.linspace(1.0 / n_dec, 1.0, n_dec)

    # --- Conditional CDF over log_E given sin_dec ---
    w_e = w.sum(axis=1)                          # (n_dec, n_e)
    row_totals = w_e.sum(axis=1, keepdims=True)
    safe = np.where(row_totals > 0, row_totals, 1.0)
    cdf_e_given_dec = np.cumsum(w_e / safe, axis=1)   # (n_dec, n_e)

    # --- Conditional CDF over RA given (sin_dec, log_E) ---
    w_ra = w.transpose(0, 2, 1)                  # (n_dec, n_e, n_ra)
    cell_totals = w_ra.sum(axis=2, keepdims=True)
    safe_c = np.where(cell_totals > 0, cell_totals, 1.0)
    cdf_ra_given_dec_e = np.cumsum(w_ra / safe_c, axis=2)  # (n_dec, n_e, n_ra)

    return {
        "norm":                norm,
        "cdf_dec":             cdf_dec,
        "cdf_e_given_dec":     cdf_e_given_dec,
        "cdf_ra_given_dec_e":  cdf_ra_given_dec_e,
        "sd_edges":            sd_edges,
        "ra_edges":            ra_edges,
        "le_edges":            le_edges,
    }


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

    **Transient** (``deltat=None``)
        Effective area evaluated at the reference epoch ``t0`` via a single
        vectorised astropy transform.  Use for short observations (< 1 day).

    **Steady-state** (``deltat`` provided)
        Effective area averaged over a full diurnal cycle using
        ``n_time_samples`` hour-angle samples.  Correct for observations
        spanning many sidereal days.
    """

    def __init__(
        self,
        det: Detector,
        src: ExtendedSource,
        deltat: Optional[float] = None,
        n_dec: int = 40,
        n_ra: int = 40,
        n_e: int = 40,
        n_time_samples: int = 100,
    ):
        """
        Parameters
        ----------
        det : Detector
        src : ExtendedSource
        deltat : float or None
            Pass a livetime to enable steady-state (HA-averaged) A_eff mode.
            The same value (or a fraction of it) should be passed to
            ``sample_events``.
        n_dec : int
            Number of sin(dec) grid points.  Default 40.
        n_ra : int
            Number of RA grid points.  Default 40.
        n_e : int
            Number of log(E) grid points spanning 100 GeV – 1 PeV.  Default 40.
        n_time_samples : int
            Hour-angle samples for the steady-state A_eff average.  Default 100.
        """
        self._steady_state = deltat is not None
        super().__init__(det, src)

        # --- coordinate grids (cell centres) ---
        sds    = np.linspace(-1.0, 1.0, n_dec)
        decs   = np.arcsin(sds)
        ras    = np.linspace(0.0, 2.0 * np.pi, n_ra, endpoint=False)
        log_es = np.linspace(np.log(1e2 * units.GeV), np.log(1e6 * units.GeV), n_e)
        es     = np.exp(log_es)

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

        track_effa   = det.response.effective_area.get("track")
        cascade_effa = det.response.effective_area.get("cascade")

        # --- flux grid (n_dec, n_e); RA-independent for current source model ---
        fluxes = np.zeros((6, n_dec, n_e))
        for jdx, dec in enumerate(tqdm(decs, desc="building flux grid")):
            for idx, nu in enumerate(neutrinos):
                fluxes[idx, jdx, :] = src(nu, es, dec)

        # --- effective-area grid (n_dec, n_ra, n_e) ---
        effa_grids = {}
        for name, effa in [("track", track_effa), ("cascade", cascade_effa)]:
            if effa is None:
                continue
            grid = np.zeros((n_dec, n_ra, n_e))
            if self._steady_state:
                avg = _effa_grid_steady_state(
                    decs, det.location, effa, es, n_time_samples
                )
                grid[:] = avg[:, np.newaxis, :]   # uniform over RA
            else:
                zeniths = zenith_grid(decs, ras, det.location, self._t0)
                # zeniths: (n_dec, n_ra) — broadcast with es to get (n_dec, n_ra, n_e)
                zen_flat = np.repeat(zeniths.ravel(), n_e)          # (n_dec*n_ra*n_e,)
                e_flat   = np.tile(es, n_dec * n_ra)                # (n_dec*n_ra*n_e,)
                grid = effa(zen_flat, e_flat).reshape(n_dec, n_ra, n_e)
            effa_grids[name] = grid

        # --- build per-morphology sampling tables ---
        flux_track = fluxes[2] + fluxes[3]   # NuMu + NuMuBar
        flux_all   = fluxes.sum(axis=0)       # all flavours

        self._d = {}
        for morph, flux_2d in [("track", flux_track), ("cascade", flux_all)]:
            if morph not in det.response.available_morphologies:
                continue
            effa_grid = effa_grids.get(morph, np.zeros((n_dec, n_ra, n_e)))
            # target[i, j, k] = A_eff(dec_i, RA_j, E_k) * flux(dec_i, E_k)
            target = effa_grid * flux_2d[:, np.newaxis, :]
            self._d[morph] = _build_sampling_data(
                target, es, log_es, sds, sd_edges, ra_edges, le_edges
            )

    def expected_events(
        self,
        morphology: str,
        deltat: float,
        t: Optional[float] = None,
    ) -> float:
        if morphology not in self._d:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {list(self._d.keys())}."
            )
        return self._d[morphology]["norm"] * deltat

    def sample_events(
        self,
        morphology: str,
        deltat: Optional[float] = None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        oversample: int = 1,
    ):
        """
        Draw a set of reconstructed events from the source.

        Parameters
        ----------
        morphology : {"track", "cascade"}
        oversample : int
            Draw ``nevent * oversample`` true-direction samples and keep
            every ``oversample``-th.  Unlike MCMC thinning this does not
            reduce autocorrelation (grid sampling has none), but it can
            reduce the variance of derived histograms at the cost of more
            smearing calls.  Default 1.
        t : float or None
            Reference epoch in MJD.  Used to rotate RA for Earth's rotation
            since ``t0``.  Defaults to ``t0``.
        nevent : int or None
            Fixed number of events.  Mutually exclusive with ``deltat``.
        deltat : float or None
            Observation window in internal time units (eV^{-1}).  The number
            of events is drawn from Poisson(norm * deltat).

        Returns
        -------
        list of Event
        """
        if not ((nevent is None) ^ (deltat is None)):
            raise ValueError("Exactly one of nevent or deltat must be provided.")
        if morphology not in ("track", "cascade"):
            raise ValueError(
                f"Invalid morphology '{morphology}': must be 'track' or 'cascade'."
            )
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )

        d = self._d[morphology]
        if nevent is None:
            nevent = np.random.poisson(d["norm"] * deltat)
        if t is None:
            t = self._t0

        n = int(nevent) * oversample
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
            np.searchsorted(cdf_dec, np.random.rand(n)),
            0, n_dec - 1,
        )

        # --- Step 2: sample log_E bins from the conditional CDF given sin_dec ---
        cdfs_e = cdf_e_given_dec[i_dec]          # (n, n_e)
        i_e = np.clip(
            (cdfs_e < np.random.rand(n)[:, None]).sum(axis=1),
            0, n_e - 1,
        )

        # --- Step 3: sample RA bins from the conditional CDF given (sin_dec, log_E) ---
        cdfs_ra = cdf_ra_given_dec_e[i_dec, i_e]  # (n, n_ra)
        i_ra = np.clip(
            (cdfs_ra < np.random.rand(n)[:, None]).sum(axis=1),
            0, n_ra - 1,
        )

        # --- Step 4: jitter uniformly within each selected cell ---
        sin_dec = np.random.uniform(sd_edges[i_dec],     sd_edges[i_dec + 1])
        ra      = np.random.uniform(ra_edges[i_ra],      ra_edges[i_ra + 1])
        log_e   = np.random.uniform(le_edges[i_e],       le_edges[i_e + 1])

        # RA rotation for Earth's rotation since t0
        offset = 2.0 * np.pi * ((t - self._t0) % 1)
        ra = np.mod(ra + offset, 2.0 * np.pi)

        # Thin by oversample
        sin_dec = sin_dec[::oversample]
        ra      = ra[::oversample]
        log_e   = log_e[::oversample]

        events = []
        for sd, ra_i, le in zip(sin_dec, ra, log_e):
            true_direction = SkyCoordinate(np.arcsin(sd), ra_i)
            true_energy    = np.exp(le)
            reco_direction, reco_energy, ang_err = smear_truth(
                true_direction, true_energy, self._det, morphology
            )
            if deltat is None:
                t_event = t
            else:
                t_event = t + np.random.uniform(
                    low=-deltat / 2, high=deltat / 2
                ) / (24 * 3600 * units.sec)
            morphology_id = 2 if morphology == "track" else 1
            events.append(Event(
                true_direction,
                reco_direction,
                true_energy,
                reco_energy,
                t_event,
                morphology_id,
                ang_err=ang_err,
            ))

        return events
