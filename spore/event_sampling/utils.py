import logging

import numpy as np

from ..conventions import SkyCoordinate, EarthCoordinate
from ..detector import Detector

logger = logging.getLogger(__name__)

_SIDEREAL_DAY = 0.99726958  # mean sidereal day in solar days
# The GMST polynomial below is expressed in days from the J2000 epoch,
# JD 2451545.0 = 2000-01-01 12:00, i.e. MJD 51544.5 (MJD = JD - 2400000.5).
# Using 51545.0 offsets every sidereal time by half a day, or 180.5 degrees.
_J2000_MJD    = 51544.5


def _effa_energy_bounds(effa_fn) -> tuple:
    """Return (e_min_gev, e_max_gev) from an effa callable or list of callables.

    Reads the ``e_min_gev`` / ``e_max_gev`` attributes attached by
    ``effa_helper``.  Falls back to (1e2, 1e7) if the attributes are absent.
    For a list of per-species callables, takes the union over those species
    that declare bounds; species without them are ignored rather than
    widening the range to the fallback, which would stretch the sampling grid
    over energies the response never covers.
    """
    fns = effa_fn if isinstance(effa_fn, list) else [effa_fn]
    los = [f.e_min_gev for f in fns if hasattr(f, 'e_min_gev')]
    his = [f.e_max_gev for f in fns if hasattr(f, 'e_max_gev')]
    return (min(los) if los else 1e2), (max(his) if his else 1e7)


def resolve_energy_bounds(det, src, e_min=None, e_max=None) -> tuple:
    """Sampling energy range: the IRF support intersected with the flux support.

    Explicit ``e_min`` / ``e_max`` win where given.  Both sampler classes need
    the same rule, so it lives here rather than being spelled out twice.

    Args:
        det: Detector whose first available morphology defines the IRF range.
        src: Source; its flux supplies ``e_min_gev`` / ``e_max_gev`` if present.
        e_min: Explicit lower bound in GeV, or None.
        e_max: Explicit upper bound in GeV, or None.

    Returns:
        Tuple ``(e_min_gev, e_max_gev)``.
    """
    if e_min is not None and e_max is not None:
        return e_min, e_max
    first_morph = sorted(det.response.effective_area)[0]
    irf_lo, irf_hi = _effa_energy_bounds(det.response.effective_area[first_morph])
    flux = getattr(src, 'flux', None)
    src_lo = getattr(flux, 'e_min_gev', None)
    src_hi = getattr(flux, 'e_max_gev', None)
    if e_min is None:
        e_min = max(irf_lo, src_lo) if src_lo is not None else irf_lo
    if e_max is None:
        e_max = min(irf_hi, src_hi) if src_hi is not None else irf_hi
    return e_min, e_max


def build_adaptive_log_energy_grid(
    n_e: int,
    effa_fn,
    src,
    morphology: str,
    n_pilot: int = 500,
    e_min: float = None,
    e_max: float = None,
    uniform_fraction: float = 0.5,
) -> np.ndarray:
    """Build a quantile-based log-energy grid for use in ExtendedSourceEventSampler.

    Bin edges are placed so that each cell contains roughly equal integrated
    probability mass ``A_eff(E) * flux(E) * E d(ln E)``, evaluated with a
    representative effective area (zenith = pi/2).  This concentrates bins
    near the detection threshold where A_eff rises steeply and the midpoint
    rule error is largest.

    Pure equal-mass placement leaves the depleted end of the range unresolved:
    for a steeply falling atmospheric spectrum essentially all the mass sits
    below a few tens of TeV, so the last quantile step can jump more than a
    decade in one cell.  Because the sampler draws log E uniformly within a
    cell and weights the cell by the density at its centre, such a cell
    smears its whole mass flat across that decade, producing a spurious
    plateau in the sampled spectrum.  Mixing the mass CDF with a uniform one
    bounds the cell width by ``(log range) / (uniform_fraction * (n_e - 1))``
    while keeping most of the resolution near threshold.

    Args:
        n_e: Number of grid points (cell centres).
        effa_fn: Effective area function(s) with signature
            ``effa(zenith_rad, energy_GeV) -> cm²``.  For per-species responses
            pass a list; the species are summed.
        src: Source object with a ``__call__(nu, E)`` interface.
        morphology: ``"track"`` sums nu_mu + anti-nu_mu; anything else sums all
            six species.
        n_pilot: Number of points in the fine pilot grid.  Default 500.
        e_min: Minimum energy in GeV.  If None, read from the ``e_min_gev``
            attribute of ``effa_fn`` (set by ``effa_helper`` at load time).
            Falls back to 100 GeV if the attribute is absent.
        e_max: Maximum energy in GeV.  If None, read from the ``e_max_gev``
            attribute of ``effa_fn``.  Falls back to 10 PeV if absent.
        uniform_fraction: Weight of the uniform-in-log component blended into
            the mass CDF before quantiles are taken, in [0, 1].  0 reproduces
            pure equal-mass placement; 1 gives a uniform log grid.  Default
            0.5, which caps the widest cell at twice the uniform spacing.

    Returns:
        ndarray of shape ``(n_e,)`` with cell-centre positions in ln(E/GeV),
        non-uniformly spaced.
    """
    from ..physics import neutrinos

    if e_min is None or e_max is None:
        _irf_lo, _irf_hi = _effa_energy_bounds(effa_fn)
        _src_lo = getattr(getattr(src, 'flux', None), 'e_min_gev', None)
        _src_hi = getattr(getattr(src, 'flux', None), 'e_max_gev', None)
        if e_min is None:
            e_min = max(_irf_lo, _src_lo) if _src_lo is not None else _irf_lo
        if e_max is None:
            e_max = min(_irf_hi, _src_hi) if _src_hi is not None else _irf_hi

    log_es_pilot = np.linspace(np.log(e_min), np.log(e_max), n_pilot)
    es_pilot     = np.clip(np.exp(log_es_pilot), e_min, e_max)
    zen_rep      = np.full(n_pilot, np.pi / 2)  # horizon: representative zenith

    if isinstance(effa_fn, list):
        effas = sum(fn(zen_rep, es_pilot) for fn in effa_fn)
    else:
        effas = effa_fn(zen_rep, es_pilot)

    # Use dec=0 (equator) as a representative sky position for the pilot flux.
    # The spectral shape is what matters; absolute normalisation cancels.
    # Pass the full es_pilot array rather than iterating over scalars to avoid
    # the scalar-indexing edge case in TabulatedEnergyDecFlux.density.
    # Pass dec=0.0, ra=0.0 as representative sky position for the pilot flux.
    # All distribution types accept both arguments (ignoring whichever they
    # don't use), so no branching on source dimensionality is needed.
    if "track" in morphology:
        fluxes = (src(neutrinos[2], es_pilot, 0.0, 0.0)
                  + src(neutrinos[3], es_pilot, 0.0, 0.0))
    else:
        fluxes = sum(src(nu, es_pilot, 0.0, 0.0) for nu in neutrinos)
    fluxes = np.asarray(fluxes, dtype=float)

    weights = effas * fluxes * es_pilot  # integrand in d(ln E) space

    if weights.sum() == 0:
        # No signal at this morphology: fall back to uniform spacing
        return np.linspace(np.log(e_min), np.log(e_max), n_e)

    dloge = np.gradient(log_es_pilot)
    cum   = np.cumsum(weights * dloge)
    cum  /= cum[-1]

    # Blend with a uniform-in-log CDF so no cell can span more than
    # (log range) / (uniform_fraction * (n_e - 1)); see the note above.
    if uniform_fraction > 0:
        span = log_es_pilot[-1] - log_es_pilot[0]
        u_lin = (log_es_pilot - log_es_pilot[0]) / span if span > 0 else cum
        cum = (1.0 - uniform_fraction) * cum + uniform_fraction * u_lin

    # Place n_e cell centres at quantiles 0, 1/(n_e-1), ..., 1
    quantiles = np.linspace(0.0, 1.0, n_e)
    return np.interp(quantiles, cum, log_es_pilot)


def _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges):
    """Precompute hierarchical inverse-CDF tables from a 3-D target grid.

    The joint distribution is factored as::

        p(sin_dec, RA, log_E) = p(sin_dec) * p(log_E | sin_dec) * p(RA | sin_dec, log_E)

    Args:
        target: ndarray of shape ``(n_dec, n_ra, n_e)`` with A_eff × flux on
            the grid.
        es: Array of shape ``(n_e,)`` with energies in GeV (cell centres).
        log_es: Array of shape ``(n_e,)`` with log(E) (cell centres).
        sds: Array of shape ``(n_dec,)`` with sin(dec) cell centres.
        sd_edges: Array of shape ``(n_dec+1,)`` with sin(dec) cell edges.
        ra_edges: Array of shape ``(n_ra+1,)`` with RA cell edges in radians.
        le_edges: Array of shape ``(n_e+1,)`` with log(E) cell edges.

    Returns:
        dict with keys ``norm``, ``cdf_dec``, ``cdf_e_given_dec``,
        ``cdf_ra_given_dec_e``, ``sd_edges``, ``ra_edges``, ``le_edges``.
    """
    n_dec, n_ra, n_e = target.shape

    sd_widths = np.diff(sd_edges)
    ra_widths = np.diff(ra_edges)
    le_widths = np.diff(le_edges)

    w = target * es[np.newaxis, np.newaxis, :]

    # Compute norm via broadcasting without materialising a separate cell_vols
    # array — saves one full (n_dec, n_ra, n_e) allocation.
    norm = float((
        w
        * sd_widths[:, np.newaxis, np.newaxis]
        * ra_widths[np.newaxis, :, np.newaxis]
        * le_widths[np.newaxis, np.newaxis, :]
    ).sum())

    # Weight by both log-E and sin(dec) cell widths.  The energy grid is
    # adaptive so le_widths are non-uniform; the boundary energy bins are
    # half-width.  The sin(dec) grid from linspace(-1,1,n_dec) gives boundary
    # cells (sin_dec = ±1) that are also half-width relative to interior cells.
    # Without the sd_widths correction those boundary bins get 2× the density
    # they should, producing a spike at the poles.
    le_w = le_widths[np.newaxis, np.newaxis, :]   # (1, 1, n_e)

    w_dec = (w * le_w).sum(axis=(1, 2))
    w_dec_vol = w_dec * sd_widths                 # correct for boundary half-cells
    total_dec = w_dec_vol.sum()
    if total_dec > 0:
        cdf_dec = np.cumsum(w_dec_vol / total_dec)
    else:
        cdf_dec = np.linspace(1.0 / n_dec, 1.0, n_dec)
    cdf_dec[-1] = 1.0  # pin to exactly 1 so argmax/searchsorted never fall off the end

    w_e = (w * le_w).sum(axis=1)
    row_totals = w_e.sum(axis=1, keepdims=True)
    safe = np.where(row_totals > 0, row_totals, 1.0)
    cdf_e_given_dec = np.cumsum(w_e / safe, axis=1)
    cdf_e_given_dec[:, -1] = 1.0

    w_ra = w.transpose(0, 2, 1)
    cell_totals = w_ra.sum(axis=2, keepdims=True)
    safe_c = np.where(cell_totals > 0, cell_totals, 1.0)
    cdf_ra_given_dec_e = np.cumsum(w_ra / safe_c, axis=2)
    cdf_ra_given_dec_e[:, :, -1] = 1.0

    return {
        "norm":                norm,
        "cdf_dec":             cdf_dec,
        "cdf_e_given_dec":     cdf_e_given_dec,
        "cdf_ra_given_dec_e":  cdf_ra_given_dec_e,
        "sd_edges":            sd_edges,
        "ra_edges":            ra_edges,
        "le_edges":            le_edges,
    }



def zenith_grid(
    decs: np.ndarray,
    ras: np.ndarray,
    earth_coord: EarthCoordinate,
    t: float,
) -> np.ndarray:
    """Compute the zenith angle at each (dec, RA) grid point for a single epoch.

    Uses a vectorised astropy coordinate transform.

    Args:
        decs: Array of shape ``(n_dec,)`` with declinations in radians.
        ras: Array of shape ``(n_ra,)`` with right ascensions in radians.
        earth_coord: EarthCoordinate of the detector.
        t: Observation epoch in MJD.

    Returns:
        ndarray of shape ``(n_dec, n_ra)`` with zenith angles in radians.
    """
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy import units as u

    sc = SkyCoord(ra=np.tile(ras, len(decs)) * u.rad,
                  dec=np.repeat(decs, len(ras)) * u.rad)
    ec = EarthLocation.from_geodetic(
        lon=earth_coord.longitude * u.rad,
        lat=earth_coord.latitude * u.rad,
    )
    altaz = sc.transform_to(AltAz(location=ec, obstime=Time(t, format='mjd')))
    return (np.pi / 2 - np.radians(altaz.alt.deg)).reshape(len(decs), len(ras))


def _sample_cone_batch(
    decs: np.ndarray,
    ras: np.ndarray,
    psis: np.ndarray,
    rng=None,
) -> tuple:
    """Vectorized cone sampling for N events.

    Args:
        decs: Array of shape ``(N,)`` with true declinations in radians.
        ras: Array of shape ``(N,)`` with true right ascensions in radians.
        psis: Array of shape ``(N,)`` with half-opening angles (PSF deflections)
            in radians.
        rng: numpy Generator or None.

    Returns:
        Tuple ``(reco_decs, reco_ras)`` each of shape ``(N,)``.
    """
    cos_dec = np.cos(decs)
    vs = np.stack([cos_dec * np.cos(ras), cos_dec * np.sin(ras), np.sin(decs)], axis=1)

    # Per-event orthonormal basis perpendicular to each direction vector
    mask = np.abs(vs[:, 0]) < 0.9
    v1 = np.where(
        mask[:, None],
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
    )
    us = np.cross(vs, v1)
    us /= np.linalg.norm(us, axis=1, keepdims=True)
    ws = np.cross(vs, us)
    ws /= np.linalg.norm(ws, axis=1, keepdims=True)

    _rng = rng if rng is not None else np.random.default_rng()
    phis = _rng.uniform(0.0, 2.0 * np.pi, len(decs))
    sin_psi = np.sin(psis)[:, None]
    cos_psi = np.cos(psis)[:, None]
    points = sin_psi * (np.cos(phis)[:, None] * us + np.sin(phis)[:, None] * ws) + cos_psi * vs

    reco_decs = np.arcsin(np.clip(points[:, 2], -1.0, 1.0))
    reco_ras  = np.arctan2(points[:, 1], points[:, 0]) % (2.0 * np.pi)
    return reco_decs, reco_ras


def _fallback_local_zeniths(detector, decs, ras, t, caller):
    """Local zenith angles when the caller did not supply them.

    The joint smearing IRF is indexed by local zenith, so an incorrect zenith
    silently selects the wrong smearing band.  With an epoch the exact
    hour-angle formula is used.  Without one, the zenith is well defined only
    at the geographic poles, where cos(zen) = sin(lat) sin(dec) exactly; for
    any other latitude we refuse to guess.
    """
    lat = detector.location.latitude
    decs = np.atleast_1d(np.asarray(decs, dtype=float))
    if t is not None:
        ras = np.atleast_1d(np.asarray(ras, dtype=float))
        zens, _ = _local_coords_from_times(
            np.full(decs.shape, float(t)) if np.ndim(t) == 0 else np.asarray(t, dtype=float),
            decs, ras, lat, detector.location.longitude,
        )
        return zens
    if not np.isclose(abs(lat), np.pi / 2, atol=1e-6):
        raise ValueError(
            f"{caller} needs the local zenith to select the joint smearing "
            f"band, but the detector latitude is {np.degrees(lat):.2f} deg, "
            "where zenith depends on hour angle.  Pass local_zenith(s), or an "
            "epoch t, or sample through a sampler class (which supplies them)."
        )
    return np.arccos(np.clip(np.sin(lat) * np.sin(decs), -1.0, 1.0))


def smear_truth_batch(
    true_decs: np.ndarray,
    true_ras: np.ndarray,
    true_energies: np.ndarray,
    detector: Detector,
    morphology: str,
    rng=None,
    delta_clip: tuple = None,
    local_zeniths: np.ndarray = None,
    t: float = None,
) -> tuple:
    """Vectorized smearing for N events.

    Applies detector PSF and energy resolution to arrays of true directions
    and energies.  Energy smearing is fully vectorized (PchipInterpolator
    accepts arrays).  Angular smearing uses a per-event loop because the IRF
    callable wraps scalar interpolation; cone geometry is then vectorized.

    Args:
        true_decs: Array of shape ``(N,)`` with true declinations in radians.
        true_ras: Array of shape ``(N,)`` with true right ascensions in radians.
        true_energies: Array of shape ``(N,)`` with true energies in GeV.
        detector: Detector instance.
        morphology: Event morphology string.
        rng: numpy Generator or None.
        delta_clip: If provided, clip the log-ratio Delta = ln(E_reco/E_true)
            to ``(delta_min, delta_max)`` before exponentiating.  Useful for
            suppressing the extreme left tail of the energy resolution that
            corresponds to catastrophic reconstruction failures.  Has no effect
            when joint smearing is used.  Example: ``(-5, 3)``.

    Returns:
        Tuple ``(reco_decs, reco_ras, reco_energies, ang_errs)`` each of shape
        ``(N,)``, with energies in GeV and angles in radians.
    """
    n    = len(true_energies)
    _rng = rng if rng is not None else np.random.default_rng()
    js   = detector.response.joint_smearing

    if js is not None and morphology in js:
        joint_fn = js[morphology]
        if local_zeniths is None:
            local_zeniths = _fallback_local_zeniths(
                detector, true_decs, true_ras, t, "smear_truth_batch",
            )
        if hasattr(joint_fn, 'batch'):
            reco_energies, psis, ang_errs = joint_fn.batch(
                true_energies, local_zeniths, rng=_rng,
            )
        else:
            results       = [joint_fn(e, z, rng=_rng) for e, z in zip(true_energies, local_zeniths)]
            reco_energies = np.array([r[0].magnitude for r in results])
            psis     = np.array([r[1] for r in results])
            ang_errs = np.array([r[2] for r in results])
    else:
        ang_sampler = detector.response.angular_response.get(morphology)
        if ang_sampler is not None:
            us_ang   = _rng.random(n)
            psis     = ang_sampler(true_energies, us_ang)
            ang_errs = ang_sampler(true_energies, np.full(n, 0.5))
        else:
            psis     = np.full(n, np.nan)
            ang_errs = np.full(n, np.nan)

        e_sampler = detector.response.energy_response.get(morphology)
        if e_sampler is not None:
            deltas = e_sampler(_rng.random(n))
            if delta_clip is not None:
                deltas = np.clip(deltas, delta_clip[0], delta_clip[1])
            reco_energies = np.exp(deltas) * true_energies
        else:
            reco_energies = np.full(n, np.nan)

    if np.any(np.isnan(psis)):
        reco_decs = np.full(n, np.nan)
        reco_ras  = np.full(n, np.nan)
    else:
        reco_decs, reco_ras = _sample_cone_batch(true_decs, true_ras, psis, rng=_rng)
    return reco_decs, reco_ras, reco_energies, ang_errs


def smear_truth(
    true_direction: SkyCoordinate,
    true_energy: float,
    detector: Detector,
    morphology: str,
    rng=None,
    local_zenith: float = None,
    t: float = None,
):
    """Apply detector smearing to a true (direction, energy) scalar event.

    If the detector response includes a joint smearing IRF, the reconstruction
    is drawn from the full conditional distribution
    P(E_reco, PSF, AngErr | E_true, zenith).  Otherwise, energy and angular
    smearing are sampled independently from the 1-D marginal inverse-CDFs
    stored in the HDF5 response file.

    Args:
        true_direction: True event direction as a SkyCoordinate.
        true_energy: True energy in GeV.
        detector: Detector instance.
        morphology: Event morphology string.
        rng: numpy Generator or None.  If None, a fresh Generator is created
            (non-reproducible).  Pass a seeded Generator for reproducibility.
        local_zenith: Local zenith angle in radians.  Required by the joint
            smearing IRF; if omitted it is derived from ``t``, or from the
            declination alone when the detector sits at a geographic pole.
        t: Epoch in MJD, used to derive ``local_zenith`` when that is not
            given.

    Returns:
        Tuple ``(reco_direction, reco_energy, ang_err)`` where
        ``reco_direction`` is a SkyCoordinate, ``reco_energy`` is in GeV,
        and ``ang_err`` is in radians (NaN when no angular response is
        available).

    Raises:
        ValueError: If ``morphology`` is not available for this detector.
    """
    if morphology not in detector.response.available_morphologies:
        raise ValueError(
            f"Morphology '{morphology}' is not available for this detector. "
            f"Available: {detector.response.available_morphologies}."
        )

    _rng = rng if rng is not None else np.random.default_rng()

    js = detector.response.joint_smearing
    if js is not None and morphology in js:
        if local_zenith is None:
            local_zenith = float(_fallback_local_zeniths(
                detector,
                np.array([true_direction.declination]),
                np.array([true_direction.right_ascension]),
                t, "smear_truth",
            )[0])
        reco_energy, psi, ang_err = js[morphology](true_energy, local_zenith, rng=_rng)
    else:
        ang_sampler = detector.response.angular_response.get(morphology)
        if ang_sampler is not None:
            psi     = ang_sampler(true_energy, float(_rng.random()))
            ang_err = ang_sampler(true_energy, 0.5)
        else:
            psi     = np.nan
            ang_err = np.nan

        e_sampler = detector.response.energy_response.get(morphology)
        if e_sampler is not None:
            reco_energy = np.exp(e_sampler(float(_rng.random()))) * true_energy
        else:
            reco_energy = np.nan

    if np.isnan(psi):
        reco_direction = SkyCoordinate(np.nan, np.nan)
    else:
        rd, rr = _sample_cone_batch(
            np.array([true_direction.declination]),
            np.array([true_direction.right_ascension]),
            np.array([psi]),
            rng=_rng,
        )
        reco_direction = SkyCoordinate(float(rd[0]), float(rr[0]))
    return reco_direction, reco_energy, ang_err


def _local_zeniths_from_times(event_times, decs, ras, lat, lon):
    """Compute the instantaneous local zenith angle for each event.

    Uses the analytic formula
        cos(zen) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(HA)
    where HA = LST - RA is derived from the event timestamp via _mjd_to_lst.

    Args:
        event_times: (n,) array of event times in MJD.
        decs: (n,) array of true declinations in radians.
        ras: (n,) array of true right ascensions in radians.
        lat: Detector latitude in radians.
        lon: Detector east longitude in radians.

    Returns:
        ndarray of shape (n,) with zenith angles in radians.
    """
    return _local_coords_from_times(event_times, decs, ras, lat, lon)[0]


def _local_coords_from_times(event_times, decs, ras, lat, lon):
    """Compute local zenith and azimuth angles for each event.

    Azimuth convention: North = 0, East = π/2, increasing clockwise when
    viewed from above (standard astronomical/navigation convention).

    Args:
        event_times: (n,) array of event times in MJD.
        decs: (n,) array of true declinations in radians.
        ras: (n,) array of true right ascensions in radians.
        lat: Detector latitude in radians.
        lon: Detector east longitude in radians.

    Returns:
        Tuple (zeniths, azimuths) each of shape (n,), in radians.
        Zeniths in [0, π]; azimuths in [0, 2π).
    """
    lsts    = _mjd_to_lst(event_times, lon)
    has     = (lsts - ras) % (2.0 * np.pi)
    cos_zen = np.sin(lat) * np.sin(decs) + np.cos(lat) * np.cos(decs) * np.cos(has)
    zeniths = np.arccos(np.clip(cos_zen, -1.0, 1.0))
    sin_az  = -np.cos(decs) * np.sin(has)
    cos_az  = np.sin(decs) * np.cos(lat) - np.cos(decs) * np.cos(has) * np.sin(lat)
    azimuths = np.arctan2(sin_az, cos_az) % (2.0 * np.pi)
    return zeniths, azimuths


def _effa_grid_from_lst(decs, ras, lat, effa_fn, es, lst):
    """Instantaneous effective area on a (dec, RA, E) grid at a given LST.

    For each (dec, RA) cell, computes HA = LST - RA and hence the zenith angle
    analytically, then evaluates effa_fn.  Faster than astropy-based
    zenith_grid because no coordinate transforms are required.

    Args:
        decs: (n_dec,) declinations in radians.
        ras: (n_ra,) right ascensions in radians.
        lat: Detector latitude in radians.
        effa_fn: Callable effa_fn(zenith_rad, energy_GeV) -> cm².
        es: (n_e,) energies in GeV.
        lst: Local sidereal time in radians.

    Returns:
        ndarray of shape (n_dec, n_ra, n_e).
    """
    n_dec, n_ra, n_e = len(decs), len(ras), len(es)
    ha = lst - ras  # (n_ra,)
    cos_zens = (
        np.sin(lat) * np.sin(decs)[:, np.newaxis]
        + np.cos(lat) * np.cos(decs)[:, np.newaxis] * np.cos(ha)[np.newaxis, :]
    )  # (n_dec, n_ra)
    zens = np.arccos(np.clip(cos_zens, -1.0, 1.0))
    zen_flat = np.repeat(zens.ravel(), n_e)
    e_flat   = np.tile(es, n_dec * n_ra)
    return effa_fn(zen_flat, e_flat).reshape(n_dec, n_ra, n_e)


def _mjd_to_lst(mjd, longitude_rad):
    """Approximate local sidereal time in radians from MJD and detector longitude.

    Uses the J2000 GMST polynomial accurate to ~0.1 s over several centuries.

    Args:
        mjd: Scalar or array of Modified Julian Dates.
        longitude_rad: Detector east longitude in radians.

    Returns:
        LST in radians, same shape as *mjd*, in [0, 2π).
    """
    d = np.asarray(mjd, dtype=float) - _J2000_MJD
    gmst_rad = ((18.697374558 + 24.06570982441908 * d) % 24) / 24.0 * 2.0 * np.pi
    return (gmst_rad + longitude_rad) % (2.0 * np.pi)


def _sample_from_slice(d, n, rng):
    """Sample n (sin_dec, RA, log_E) tuples from one pre-built sampling slice.

    Args:
        d: Sampling data dict as returned by _build_sampling_data.
        n: Number of samples to draw.
        rng: numpy Generator.

    Returns:
        Tuple (sin_dec, ra, log_e) each of shape (n,).
    """
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

    sin_dec = rng.uniform(sd_edges[i_dec],  sd_edges[i_dec + 1])
    ra      = rng.uniform(ra_edges[i_ra],   ra_edges[i_ra  + 1])
    log_e   = rng.uniform(le_edges[i_e],    le_edges[i_e   + 1])
    return sin_dec, ra, log_e


def _assign_times_from_slices(t_start, deltat, k_arr, n_slices, rng,
                              longitude=0.0, phase_ref=0.0):
    """Assign event times consistent with the sidereal phase of each slice.

    Slice k covers phase ``[k, k+1) / n_slices`` of the sidereal cycle, where
    the cycle is measured in the same variable used to build the slices:
    local sidereal time for the extended sampler, or hour angle relative to
    the source (``phase_ref = RA_src``) for the point-source sampler.

    The phase of an event at time ``t`` is ``LST(t) - phase_ref``, so the
    target time offset from ``t_start`` follows from inverting

        LST(t) = LST(t_start) + 2 pi (t - t_start) / T_sidereal.

    Ignoring ``LST(t_start)`` — i.e. assuming the observation happens to begin
    at zero sidereal phase — offsets every event time by a constant fraction
    of a sidereal day, which decorrelates the assigned times (and hence the
    per-event zenith and azimuth) from the sky configuration that produced
    them.  The offset is harmless only for a detector at a geographic pole,
    where the response has no hour-angle dependence.

    Args:
        t_start: Observation start in MJD.
        deltat: Observation duration as a pint Quantity with time units.
        k_arr: Integer array of slice indices, shape (n,).
        n_slices: Total number of slices.
        rng: numpy Generator.
        longitude: Detector east longitude in radians.
        phase_ref: Offset in radians between local sidereal time and the
            variable indexing the slices (0 for LST-indexed slices, the source
            right ascension for hour-angle-indexed slices).

    Returns:
        ndarray of shape (n,) with event times in MJD inside
        [t_start, t_start + deltat].
    """
    deltat_days = deltat.to('day').magnitude
    n = len(k_arr)
    if n == 0:
        return np.empty(0, dtype=float)

    if deltat_days < _SIDEREAL_DAY:
        logger.warning(
            "Steady-state sampling over %.3f day is shorter than one sidereal "
            "day, so the window cannot cover every sidereal phase.  Slice "
            "assignment still averages the effective area over a full cycle, "
            "and event times for unobservable phases are clipped to the window "
            "edge.  Use a transient-mode sampler for observations this short.",
            deltat_days,
        )

    # Target phase within the sidereal cycle, jittered inside the slice.
    target = 2.0 * np.pi * (k_arr + rng.random(n)) / n_slices

    lst0 = _mjd_to_lst(t_start, longitude)
    # Fraction of a sidereal day between t_start and the next time the phase
    # reaches `target`.
    frac = ((target + phase_ref - lst0) % (2.0 * np.pi)) / (2.0 * np.pi)

    # Spread over whole sidereal days inside the window.  Draw the day index
    # per event, then reject any time past the end of the window by redrawing
    # it uniformly among the days that do fit.
    n_sd = max(1, int(np.ceil(deltat_days / _SIDEREAL_DAY)))
    day_i = rng.integers(0, n_sd, size=n)
    times = t_start + (day_i + frac) * _SIDEREAL_DAY
    over = times > t_start + deltat_days
    if np.any(over):
        n_fit = np.maximum(
            np.floor((deltat_days / _SIDEREAL_DAY) - frac[over]) + 1, 1
        ).astype(int)
        day_i2 = (rng.random(int(over.sum())) * n_fit).astype(int)
        times[over] = t_start + (day_i2 + frac[over]) * _SIDEREAL_DAY
    return np.clip(times, t_start, t_start + deltat_days)


def _slice_indices_from_times(times, longitude, n_slices, phase_ref=0.0):
    """Map event times to sidereal slice indices (inverse of the above)."""
    lsts = _mjd_to_lst(times, longitude)
    phase = (lsts - phase_ref) % (2.0 * np.pi)
    return (phase / (2.0 * np.pi) * n_slices).astype(int) % n_slices


def _grl_slice_weights(norms, grl, longitude, phase_ref=0.0):
    """Probability of drawing each sidereal slice under a good run list.

    Proportional to (rate in the slice) x (livetime the run list spends in
    that slice).  Falls back to the rate alone if the run list has no
    exposure anywhere.
    """
    norms = np.asarray(norms, dtype=float)
    frac = grl.lst_exposure(longitude, len(norms), phase_ref)
    w = norms * frac
    total = w.sum()
    if total <= 0:
        total_n = norms.sum()
        return norms / total_n if total_n > 0 else np.full(len(norms), 1.0 / len(norms))
    return w / total


def _grl_mean_rate(norms, grl, longitude, phase_ref=0.0):
    """Exposure-weighted mean rate over a good run list [s^-1]."""
    norms = np.asarray(norms, dtype=float)
    frac = grl.lst_exposure(longitude, len(norms), phase_ref)
    return float(np.sum(norms * frac))


def _sample_times_in_slices(grl, k_arr, n_slices, rng, longitude,
                            phase_ref=0.0, pool_factor=40, min_pool=20_000):
    """Draw times from a good run list, conditioned on the sidereal slice.

    ``GoodRunList.sample_times`` draws uniformly over livetime, which yields
    whatever sidereal-phase distribution the run list happens to have.  That
    is the correct *conditional* distribution within a slice, but it carries
    no information about how many events each slice should receive: the rate
    varies across the sidereal cycle whenever the detector is not at a pole.
    Callers therefore choose the slice from the rate weights first, and this
    helper supplies a time drawn from the run list *given* that slice.

    Implemented by drawing a pool of candidate times uniformly over livetime,
    bucketing them by slice, and drawing from the matching bucket.  Slices the
    run list never covers fall back to an unconditioned draw.
    """
    n = len(k_arr)
    if n == 0:
        return np.empty(0, dtype=float)

    pool = grl.sample_times(max(min_pool, pool_factor * n), rng)
    pool_k = _slice_indices_from_times(pool, longitude, n_slices, phase_ref)
    order = np.argsort(pool_k, kind="stable")
    pool_sorted = pool[order]
    starts = np.searchsorted(pool_k[order], np.arange(n_slices), side="left")
    stops = np.searchsorted(pool_k[order], np.arange(n_slices), side="right")

    out = np.empty(n, dtype=float)
    for k in range(n_slices):
        mask = k_arr == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        lo, hi = starts[k], stops[k]
        if hi > lo:
            out[mask] = pool_sorted[lo + (rng.random(n_k) * (hi - lo)).astype(int)]
        else:
            out[mask] = grl.sample_times(n_k, rng)
    return out
