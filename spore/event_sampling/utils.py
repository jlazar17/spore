import numpy as np

from ..conventions import SkyCoordinate, EarthCoordinate, sample_cone, ureg
from ..detector import Detector


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

    Parameters
    ----------
    n_e : int
        Number of grid points (cell centres).
    effa_fn : callable or list of callables
        Effective area function(s) ``effa(zenith_rad, energy_GeV) -> cm^2``.
        For per-species responses pass the list; the species are summed.
    src : Source
        Source object with a ``__call__(nu, E)`` interface.
    morphology : str
        ``"track"`` sums nu_mu + anti-nu_mu; anything else sums all six species.
    n_pilot : int
        Number of points in the fine pilot grid.  Default 500.
    e_min, e_max : float or None
        Energy range in GeV.  If None (default), the bounds are read from the
        ``e_min_gev`` / ``e_max_gev`` attributes of ``effa_fn`` (set by
        ``effa_helper`` at load time).  Falls back to 100 GeV – 10 PeV if the
        attributes are absent.

    Returns
    -------
    log_es : ndarray, shape (n_e,)
        Cell-centre positions in ln(E/GeV), non-uniformly spaced.
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
    import inspect
    _call_sig = inspect.signature(src.__call__)
    _needs_dec = "dec" in _call_sig.parameters

    if "track" in morphology:
        if _needs_dec:
            fluxes = (src(neutrinos[2], es_pilot, 0.0)
                      + src(neutrinos[3], es_pilot, 0.0))
        else:
            fluxes = (src(neutrinos[2], es_pilot)
                      + src(neutrinos[3], es_pilot))
    else:
        if _needs_dec:
            fluxes = sum(src(nu, es_pilot, 0.0) for nu in neutrinos)
        else:
            fluxes = sum(src(nu, es_pilot) for nu in neutrinos)
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
    """
    Precompute hierarchical inverse-CDF tables from a 3-D target grid.

    The joint distribution is factored as:

        p(sin_dec, RA, log_E) = p(sin_dec) * p(log_E | sin_dec) * p(RA | sin_dec, log_E)

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
    dict with keys: norm, cdf_dec, cdf_e_given_dec, cdf_ra_given_dec_e,
                    sd_edges, ra_edges, le_edges.
    """
    n_dec, n_ra, n_e = target.shape

    sd_widths = np.diff(sd_edges)
    ra_widths = np.diff(ra_edges)
    le_widths = np.diff(le_edges)

    w = target * es[np.newaxis, np.newaxis, :]

    cell_vols = (
        sd_widths[:, np.newaxis, np.newaxis]
        * ra_widths[np.newaxis, :, np.newaxis]
        * le_widths[np.newaxis, np.newaxis, :]
    )
    norm = float((w * cell_vols).sum())

    # Weight by log-E cell widths so the first/last half-width bins are not
    # over-sampled.  RA widths are uniform and cancel; sd widths are uniform
    # and cancel.  Only le_widths are non-uniform (half-step at the edges).
    le_w = le_widths[np.newaxis, np.newaxis, :]   # (1, 1, n_e)

    w_dec = (w * le_w).sum(axis=(1, 2))
    total_dec = w_dec.sum()
    if total_dec > 0:
        cdf_dec = np.cumsum(w_dec / total_dec)
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


def _build_sampling_data_2d(target, sds, sd_edges, ra_edges):
    """
    Precompute hierarchical inverse-CDF tables from a 2-D target grid.

    Used for monochromatic sources where energy is fixed and only direction
    needs to be sampled.  The joint distribution is factored as:

        p(sin_dec, RA) = p(sin_dec) * p(RA | sin_dec)

    Parameters
    ----------
    target   : ndarray (n_dec, n_ra)   A_eff × spatial_flux on the grid.
    sds      : (n_dec,) sin(dec) cell centres.
    sd_edges : (n_dec+1,) sin(dec) cell edges.
    ra_edges : (n_ra+1,)  RA cell edges in radians.

    Returns
    -------
    dict with keys: norm, cdf_dec, cdf_ra_given_dec, sd_edges, ra_edges.
    """
    n_dec, n_ra = target.shape
    sd_widths = np.diff(sd_edges)
    ra_widths = np.diff(ra_edges)

    cell_vols = sd_widths[:, np.newaxis] * ra_widths[np.newaxis, :]
    norm = float((target * cell_vols).sum())

    w_dec = target.sum(axis=1)
    total_dec = w_dec.sum()
    if total_dec > 0:
        cdf_dec = np.cumsum(w_dec / total_dec)
    else:
        cdf_dec = np.linspace(1.0 / n_dec, 1.0, n_dec)

    row_totals = target.sum(axis=1, keepdims=True)
    safe = np.where(row_totals > 0, row_totals, 1.0)
    cdf_ra_given_dec = np.cumsum(target / safe, axis=1)

    return {
        "norm":             norm,
        "cdf_dec":          cdf_dec,
        "cdf_ra_given_dec": cdf_ra_given_dec,
        "sd_edges":         sd_edges,
        "ra_edges":         ra_edges,
    }


def zenith_grid(
    decs: np.ndarray,
    ras: np.ndarray,
    earth_coord: EarthCoordinate,
    t: float,
) -> np.ndarray:
    """
    Compute the zenith angle at each (dec, RA) grid point for a single
    observation time using a vectorised astropy coordinate transform.

    Parameters
    ----------
    decs : (n_dec,)   Declinations in radians.
    ras  : (n_ra,)    Right ascensions in radians.
    earth_coord : EarthCoordinate
    t    : float      Observation epoch in MJD.

    Returns
    -------
    zeniths : ndarray, shape (n_dec, n_ra)   Zenith angle in radians.
    """
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy import units as u

    dec_grid, ra_grid = np.meshgrid(decs, ras, indexing='ij')
    sc = SkyCoord(ra=ra_grid.ravel() * u.rad, dec=dec_grid.ravel() * u.rad)
    ec = EarthLocation.from_geodetic(
        lon=earth_coord.longitude * u.rad,
        lat=earth_coord.latitude * u.rad,
    )
    altaz = sc.transform_to(AltAz(location=ec, obstime=Time(t, format='mjd')))
    return (np.pi / 2 - np.radians(altaz.alt.deg)).reshape(len(decs), len(ras))

def new_sample(bounds) -> np.ndarray:
    """Draw a uniform random sample from a rectangular parameter space.

    Args:
        bounds: Sequence of (lower, upper) pairs defining the bounds of each
            dimension.

    Returns:
        Array of length len(bounds) with each element drawn uniformly from
        its respective interval.
    """
    return np.array([np.random.uniform(lb, ub) for lb, ub in bounds])

def sample_t_event(t: float, deltat) -> float:
    """Return a uniformly-distributed event time within the observation window.

    Parameters
    ----------
    t : float
        Reference epoch in MJD.
    deltat : pint Quantity or None
        Observation window as a pint time Quantity (e.g. ``10 * ureg.year``).
        If None, returns ``t`` unchanged (fixed-time mode).

    Returns
    -------
    float
        Event time in MJD.
    """
    if deltat is None:
        return t
    deltat_days = deltat.to('day').magnitude
    return t + np.random.uniform(low=-deltat_days / 2, high=deltat_days / 2)


def _sample_cone_batch(
    decs: np.ndarray,
    ras: np.ndarray,
    psis: np.ndarray,
    rng=None,
) -> tuple:
    """Vectorized cone sampling for N events.

    Parameters
    ----------
    decs : (N,) True declinations in radians.
    ras  : (N,) True right ascensions in radians.
    psis : (N,) Half-opening angles (PSF deflections) in radians.

    Returns
    -------
    reco_decs : (N,) ndarray
    reco_ras  : (N,) ndarray
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
) -> tuple:
    """Vectorized smearing for N events.

    Applies detector PSF and energy resolution to arrays of true directions
    and energies.  Energy smearing is fully vectorized (PchipInterpolator
    accepts arrays).  Angular smearing uses a per-event loop because the IRF
    callable wraps scalar interpolation; cone geometry is then vectorized.

    Parameters
    ----------
    true_decs      : (N,) True declinations in radians.
    true_ras       : (N,) True right ascensions in radians.
    true_energies  : (N,) True energies in GeV.
    detector       : Detector
    morphology     : str
    rng            : numpy Generator or None
    delta_clip     : (float, float) or None
        If provided, clip the log-ratio Delta = ln(E_reco/E_true) to
        ``(delta_min, delta_max)`` before exponentiating.  Useful for
        suppressing the extreme left tail of the energy resolution that
        corresponds to catastrophic reconstruction failures.  Has no effect
        when joint smearing is used.  Example: ``delta_clip=(-5, 3)``.

    Returns
    -------
    reco_decs     : (N,) ndarray
    reco_ras      : (N,) ndarray
    reco_energies : (N,) ndarray  [GeV]
    ang_errs      : (N,) ndarray  [radians]
    """
    n    = len(true_energies)
    _rng = rng if rng is not None else np.random.default_rng()
    js   = detector.response.joint_smearing

    if js is not None and morphology in js:
        joint_fn = js[morphology]
        results       = [joint_fn(e, d, rng=_rng) for e, d in zip(true_energies, true_decs)]
        # joint_fn returns reco_energy as a pint Quantity; strip units so the
        # array is plain float64, consistent with the non-joint path.
        reco_energies = np.array([r[0].magnitude for r in results])
        psis     = np.array([r[1] for r in results])
        ang_errs = np.array([r[2] for r in results])
    else:
        ang_sampler = detector.response.angular_response.get(morphology)
        if ang_sampler is not None:
            us_ang   = _rng.random(n)
            psis     = np.array([ang_sampler(e, u) for e, u in zip(true_energies, us_ang)])
            ang_errs = np.array([ang_sampler(e, 0.5) for e in true_energies])
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
    morphology: str
):
    """
    Apply detector smearing to a true (direction, energy) and return the
    reconstructed quantities plus an angular error estimate.

    If the detector response includes a joint smearing IRF (loaded via
    DetectorResponse.from_dataverse), the reconstruction is drawn from the
    full conditional distribution P(E_reco, PSF, AngErr | E_true, dec).
    Otherwise, energy and angular smearing are sampled independently from the
    1-D marginal inverse-CDFs stored in the HDF5 response file, and ang_err
    is set to 0.

    Returns
    -------
    reco_direction : SkyCoordinate
    reco_energy    : float  (internal energy units, GeV)
    ang_err        : float  (radians; 0 when joint smearing is unavailable)
    """
    if morphology not in detector.response.available_morphologies:
        raise ValueError(
            f"Morphology '{morphology}' is not available for this detector. "
            f"Available: {detector.response.available_morphologies}."
        )

    js = detector.response.joint_smearing
    if js is not None and morphology in js:
        reco_energy, psi, ang_err = js[morphology](
            true_energy, true_direction.declination
        )
    else:
        ang_sampler = detector.response.angular_response.get(morphology)
        if ang_sampler is not None:
            psi     = ang_sampler(true_energy, np.random.rand())
            ang_err = ang_sampler(true_energy, 0.5)
        else:
            psi     = np.nan
            ang_err = np.nan

        e_sampler = detector.response.energy_response.get(morphology)
        if e_sampler is not None:
            reco_energy = np.exp(e_sampler(np.random.rand())) * true_energy
        else:
            reco_energy = np.nan

    if np.isnan(psi):
        reco_direction = SkyCoordinate(np.nan, np.nan)
    else:
        reco_direction = sample_cone(true_direction, psi)
    return reco_direction, reco_energy, ang_err
