import numpy as np

from ..conventions import SkyCoordinate, EarthCoordinate
from ..detector import Detector

_SIDEREAL_DAY = 0.99726958  # mean sidereal day in solar days
_J2000_MJD    = 51545.0


def _effa_grid_steady_state(decs, earth_coord, effa_fn, es, n_ha_samples):
    """Compute the hour-angle-averaged effective area at each (dec, E) grid point.

        <A_eff(dec, E)> = (1/2pi) * integral_0^{2pi} A_eff(zeta(dec, HA), E) dHA

    cos(zeta) = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(HA)

    Args:
        decs: Array of shape ``(n_dec,)`` with declinations in radians.  Pass
            ``np.array([dec])`` for a single declination (point-source case).
        earth_coord: EarthCoordinate; only the latitude attribute is used.
        effa_fn: Callable with signature ``effa_fn(zenith_rad, energy_GeV) -> float``.
        es: Array of shape ``(n_e,)`` with energies in GeV.
        n_ha_samples: Number of hour-angle samples over ``[0, 2π]``.

    Returns:
        ndarray of shape ``(n_dec, n_e)`` with the hour-angle-averaged effective
        area in cm².
    """
    lat = earth_coord.latitude
    has = np.linspace(0, 2 * np.pi, n_ha_samples, endpoint=False)

    cos_zens = (
        np.sin(lat) * np.sin(decs)[:, np.newaxis]
        + np.cos(lat) * np.cos(decs)[:, np.newaxis] * np.cos(has)[np.newaxis, :]
    )  # (n_dec, n_ha)
    zens = np.arccos(np.clip(cos_zens, -1.0, 1.0))

    zen_flat = np.repeat(zens.ravel(), len(es))
    e_flat   = np.tile(es, len(decs) * n_ha_samples)
    effa_vals = effa_fn(zen_flat, e_flat).reshape(len(decs), n_ha_samples, len(es))
    return effa_vals.mean(axis=1)  # (n_dec, n_e)


def _effa_energy_bounds(effa_fn) -> tuple:
    """Return (e_min_gev, e_max_gev) from an effa callable or list of callables.

    Reads the ``e_min_gev`` / ``e_max_gev`` attributes attached by
    ``effa_helper``.  Falls back to (1e2, 1e7) if the attributes are absent.
    For a list of per-species callables, takes the union of all bounds.
    """
    fns = effa_fn if isinstance(effa_fn, list) else [effa_fn]
    lo = min((getattr(f, 'e_min_gev', 1e2) for f in fns), default=1e2)
    hi = max((getattr(f, 'e_max_gev', 1e7) for f in fns), default=1e7)
    return lo, hi


def build_adaptive_log_energy_grid(
    n_e: int,
    effa_fn,
    src,
    morphology: str,
    n_pilot: int = 500,
    e_min: float = None,
    e_max: float = None,
) -> np.ndarray:
    """Build a quantile-based log-energy grid for use in ExtendedSourceEventSampler.

    Bin edges are placed so that each cell contains roughly equal integrated
    probability mass ``A_eff(E) * flux(E) * E d(ln E)``, evaluated with a
    representative effective area (zenith = pi/2).  This concentrates bins
    near the detection threshold where A_eff rises steeply and the midpoint
    rule error is largest.

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


def smear_truth_batch(
    true_decs: np.ndarray,
    true_ras: np.ndarray,
    true_energies: np.ndarray,
    detector: Detector,
    morphology: str,
    rng=None,
    delta_clip: tuple = None,
    local_zeniths: np.ndarray = None,
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
            lat = detector.location.latitude
            local_zeniths = np.arccos(np.clip(
                np.sin(lat) * np.sin(true_decs), -1.0, 1.0
            ))
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
):
    """Apply detector smearing to a true (direction, energy) scalar event.

    If the detector response includes a joint smearing IRF, the reconstruction
    is drawn from the full conditional distribution
    P(E_reco, PSF, AngErr | E_true, dec).  Otherwise, energy and angular
    smearing are sampled independently from the 1-D marginal inverse-CDFs
    stored in the HDF5 response file.

    Args:
        true_direction: True event direction as a SkyCoordinate.
        true_energy: True energy in GeV.
        detector: Detector instance.
        morphology: Event morphology string.
        rng: numpy Generator or None.  If None, a fresh Generator is created
            (non-reproducible).  Pass a seeded Generator for reproducibility.

    Returns:
        Tuple ``(reco_direction, reco_energy, ang_err)`` where
        ``reco_direction`` is a SkyCoordinate, ``reco_energy`` is in GeV,
        and ``ang_err`` is in radians (0 when joint smearing is unavailable).

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
            lat = detector.location.latitude
            dec = true_direction.declination
            local_zenith = float(np.arccos(np.clip(np.sin(lat) * np.sin(dec), -1.0, 1.0)))
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
    lsts    = _mjd_to_lst(event_times, lon)
    has     = (lsts - ras) % (2.0 * np.pi)
    cos_zen = np.sin(lat) * np.sin(decs) + np.cos(lat) * np.cos(decs) * np.cos(has)
    return np.arccos(np.clip(cos_zen, -1.0, 1.0))


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


def _assign_times_from_slices(t_start, deltat, k_arr, n_slices, rng):
    """Assign event times consistent with sidereal phase implied by slice indices.

    Each event in slice k is placed in a random sidereal day within the
    observation window at the fraction (k + U[0,1)) / n_slices of that day.

    Args:
        t_start: Observation start in MJD.
        deltat: Observation duration as a pint Quantity with time units.
        k_arr: Integer array of slice indices, shape (n,).
        n_slices: Total number of slices.
        rng: numpy Generator.

    Returns:
        ndarray of shape (n,) with event times in MJD, clipped to
        [t_start, t_start + deltat].
    """
    deltat_days = deltat.to('day').magnitude
    n_sd   = max(1, int(np.floor(deltat_days / _SIDEREAL_DAY)))
    phases = (k_arr + rng.random(len(k_arr))) / n_slices
    day_i  = rng.integers(0, n_sd, size=len(k_arr))
    times  = t_start + day_i * _SIDEREAL_DAY + phases * _SIDEREAL_DAY
    return np.clip(times, t_start, t_start + deltat_days)
