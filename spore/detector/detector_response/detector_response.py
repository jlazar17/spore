import logging
import numpy as np
import h5py as h5

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator

from .. import Neutrino
from ...conventions import ureg
from ...physics import Morphology

# Seasons that have a dedicated IceCube data-release smearing file.
# All other IC86 seasons reuse IC86_II.
_SEASON_TO_IRF = {
    'IC40':  'IC40',
    'IC59':  'IC59',
    'IC79':  'IC79',
    'IC86_I': 'IC86_I',
}


@dataclass(frozen=True)
class DetectorResponse:
    """Container for a detector's instrument response functions (IRFs).

    Each IRF is stored as a dictionary keyed by morphology string
    ("track" or "cascade"). Values are callables:

    - effective_area[m](zen, e)  → effective area in cm²
    - angular_response[m](e, u)  → deflection angle in radians for quantile u
    - energy_response[m](u)      → ln(E_reco/E_true) for quantile u

    Attributes:
        effective_area: Effective area functions keyed by morphology.
        angular_response: PSF inverse-CDF functions keyed by morphology.
        energy_response: Energy resolution inverse-CDF functions keyed by
            morphology.
        joint_smearing: Optional joint (energy, angle) smearing samplers,
            keyed by morphology name.
    """
    effective_area: Dict[str, Callable]
    angular_response: Dict[str, Callable]
    energy_response: Dict[str, Callable]
    joint_smearing: Optional[Dict[str, Callable]] = field(default=None)

    @property
    def available_morphologies(self):
        """Morphology names for which a full response is loaded."""
        return list(self.effective_area.keys())

    @classmethod
    def from_config(cls, config: Dict, trim_isolated: Optional[bool] = None, smoothing_sigma: Optional[float] = None) -> 'DetectorResponse':
        """Build a DetectorResponse from a config dictionary.

        Loads IRFs from an HDF5 file. The file must use the hierarchical
        format: top-level groups named after registered morphologies (e.g.
        ``track``, ``cascade``), each containing an ``effective_area``
        subgroup and optionally ``angular_response``, ``energy_resolution``,
        and ``smearing`` subgroups.

        Args:
            config: Dictionary with a ``detector_response_file`` key pointing
                to an HDF5 path built by one of the build scripts.
            trim_isolated: If True, zero out energy bins outside the largest
                contiguous non-zero run per zenith column when loading the
                effective area.  If ``None`` (default), the value is read from
                the ``trim_isolated`` attribute of the HDF5 ``meta`` group if
                present, otherwise defaults to ``True``.  Pass an explicit
                bool to override the file metadata.
            smoothing_sigma: Width (in energy bins) of the Gaussian kernel
                used to smooth MC statistical noise in the effective area
                high-energy tail.  If ``None`` (default), the value is read
                from the ``smoothing_sigma`` attribute of the HDF5 ``meta``
                group if present, otherwise the code default
                (``DEFAULT_SMOOTHING_SIGMA``) is used.  Pass an explicit
                float to override the file metadata.  Set to 0 to disable
                smoothing entirely.  **Caveat emptor**: always plot the
                effective area after loading to verify the shape looks
                physically reasonable.

        Returns:
            A configured DetectorResponse instance.

        Raises:
            ValueError: If the file contains no usable response for any
                registered morphology.
        """
        from .utils import DEFAULT_SMOOTHING_SIGMA

        effective_area   = {}
        angular_response = {}
        energy_response  = {}
        joint_smearing   = {}
        registered = sorted(Morphology.registered())

        with h5.File(config["detector_response_file"]) as h5f:
            meta = h5f.get("meta")

            # Resolve trim_isolated: explicit arg > file metadata > code default (True).
            if trim_isolated is None:
                if meta is not None and "trim_isolated" in meta.attrs:
                    trim_isolated = bool(meta.attrs["trim_isolated"])
                    logger.debug("Using trim_isolated=%s from file metadata.", trim_isolated)
                else:
                    trim_isolated = True

            # Resolve smoothing_sigma: explicit arg > file metadata > code default.
            if smoothing_sigma is None:
                if meta is not None and "smoothing_sigma" in meta.attrs:
                    smoothing_sigma = float(meta.attrs["smoothing_sigma"])
                    logger.debug("Using smoothing_sigma=%.2f from file metadata.", smoothing_sigma)
                else:
                    smoothing_sigma = DEFAULT_SMOOTHING_SIGMA

            for morph_name in registered:
                if morph_name not in h5f or not isinstance(h5f[morph_name], h5.Group):
                    continue
                grp = h5f[morph_name]
                if "effective_area" not in grp:
                    continue
                effa = effa_spline_from_group(grp["effective_area"],
                                              trim_isolated=trim_isolated,
                                              smoothing_sigma=smoothing_sigma)
                ang    = ang_spline_from_group(grp["angular_response"])      if "angular_response"  in grp else None
                energy = energy_spline_from_group(grp["energy_resolution"])  if "energy_resolution" in grp else None
                joint  = smearing_sampler_from_group(grp["smearing"])        if "smearing"          in grp else None
                if ang is None and joint is None:
                    logger.warning(
                        "Morphology group '%s' has no angular_response or smearing; "
                        "reconstructed directions will be NaN.",
                        morph_name,
                    )
                if energy is None:
                    logger.warning(
                        "Morphology group '%s' has no energy_resolution; "
                        "reconstructed energies will be NaN.",
                        morph_name,
                    )
                effective_area[morph_name]   = effa
                angular_response[morph_name] = ang
                energy_response[morph_name]  = energy
                if joint is not None:
                    joint_smearing[morph_name] = joint
                logger.debug("Loaded morphology '%s' from HDF5.", morph_name)

            unregistered = [
                k for k, v in h5f.items()
                if isinstance(v, h5.Group) and "effective_area" in v
                and k not in registered
            ]
            if unregistered:
                logger.warning(
                    "HDF5 file contains morphology groups that are not registered "
                    "and will not be loaded: %s. "
                    "Call Morphology.register('<name>') before building the "
                    "DetectorResponse to load them.",
                    sorted(unregistered),
                )

        if not effective_area:
            raise ValueError(
                "Detector response file contains no usable response. "
                "Need at least an effective area for one registered morphology."
            )

        return cls(
            effective_area,
            angular_response,
            energy_response,
            joint_smearing if joint_smearing else None,
        )



# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def effa_spline_from_group(gp: h5.Group, trim_isolated: bool = True, smoothing_sigma: float = 1.5) -> Callable:
    """Load effective area from an HDF5 group.

    Returns a single callable when ``tabulated_values`` has shape (N_E, N_ZEN)
    (single-species format), or a list of 6 callables when it has shape
    (6, N_E, N_ZEN) (per-species format).  The per-species format is required
    for morphologies where different neutrino flavours have significantly
    different detection efficiencies (e.g. HESE cascade at high energies).

    Args:
        gp: HDF5 group containing ``energies``, ``zeniths``, and
            ``tabulated_values`` datasets.  ``lower_bounds`` and
            ``upper_bounds`` datasets are ignored if present (legacy;
            superseded by ``trim_isolated``).
        trim_isolated: If True (default), zero out energy bins outside the
            largest contiguous non-zero run per zenith column before building
            the interpolator.  This removes isolated low-statistics bins at
            the edges of the sensitivity range.  Pass False to use the raw
            tabulated values as-is.

    Returns:
        A single callable or a list of 6 callables, each with signature
        ``f(zen_rad, e_gev) -> cm²``.
    """
    from .utils import effa_helper
    zens          = gp["zeniths"][:]
    es            = gp["energies"][:]       # stored in GeV
    tabulated_raw = gp["tabulated_values"][:]  # stored in cm²

    def _zero_fn(zen, e):
        scalar = np.ndim(zen) == 0
        out = np.zeros(np.atleast_1d(np.asarray(zen)).shape)
        return float(out[0]) if scalar else out

    if tabulated_raw.ndim == 3:  # per-species: shape (6, N_E, N_ZEN)
        fns = []
        for idx in range(6):
            vals = tabulated_raw[idx]
            if np.all(vals == 0):
                fns.append(_zero_fn)
            else:
                fns.append(effa_helper(zens, es, vals, trim_isolated=trim_isolated,
                                       smoothing_sigma=smoothing_sigma))
        return fns
    else:  # 2-D: shape (N_E, N_ZEN)
        vals = tabulated_raw
        if np.all(vals == 0):
            return _zero_fn
        return effa_helper(zens, es, vals, trim_isolated=trim_isolated,
                           smoothing_sigma=smoothing_sigma)

def ang_spline_from_group(gp: h5.Group) -> Callable:
    # PCHIP is monotone-preserving in each dimension, which matters for the
    # u-axis (CDF quantile): linear interpolation is monotone but O(h^2)
    # inaccurate; cubic splines can oscillate.  PCHIP gives smooth, accurate
    # angular smearing without ringing.
    #
    # Both axes are clamped to the stored grid range to prevent PCHIP from
    # extrapolating.  Out-of-range quantiles map to the boundary angle value;
    # out-of-range energies map to the nearest stored energy's PSF.
    i = RegularGridInterpolator(
        (np.log(gp["energies"][:]), gp["us"][:]),  # energies stored in GeV
        gp["inv_cdfs"][:],
        method="pchip",
    )
    e_log_min = float(i.grid[0][0])
    e_log_max = float(i.grid[0][-1])
    u_min = float(i.grid[1][0])
    u_max = float(i.grid[1][-1])

    def fxn(e, u):
        e_arr = np.asarray(e, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        scalar = e_arr.ndim == 0 and u_arr.ndim == 0
        log_e = np.clip(np.log(e_arr.ravel()), e_log_min, e_log_max)
        u_c   = np.clip(u_arr.ravel(), u_min, u_max)
        result = i(np.column_stack([log_e, u_c]))
        return float(result[0]) if scalar else result
    return fxn

def energy_spline_from_group(gp: h5.Group) -> Callable:
    # PchipInterpolator is monotone-preserving, which is a correctness
    # requirement for an inverse-CDF sampler: a non-monotone inv-CDF would
    # map quantiles to wrong energies.  linear interp1d is monotone but
    # inaccurate; natural cubic splines can be non-monotone between nodes.
    #
    # Clamp inputs to the stored quantile range to prevent PCHIP from
    # extrapolating beyond the data.  For tracks the left tail of
    # Delta = ln(E_reco/E_true) is steep; unclamped extrapolation to
    # quantiles below the stored minimum produces very large negative Delta
    # values, sending reco energies to near zero.
    us = gp["us"][:]
    spl = PchipInterpolator(us, gp["inv_cdf"][:])
    u_min, u_max = float(us[0]), float(us[-1])
    return lambda u: spl(np.clip(u, u_min, u_max))

def smearing_sampler_from_group(gp: h5.Group) -> Callable:
    """Build a joint smearing sampler from an HDF5 smearing group.

    Args:
        gp: HDF5 group containing the smearing tables (bin edges and
            fractional counts).

    Returns:
        Callable with signature
        ``sample(true_energy_GeV, zenith_rad, rng=None) -> (reco_energy_GeV, psi_rad, ang_err_rad)``
        where ``zenith_rad`` is the local zenith angle at the detector and the
        sample is drawn from P(E_reco, PSF, AngErr | E_true, zenith) stored as
        fractional counts over a 5-D histogram.
    """
    dec_edges_raw = gp['dec_edges'][:]
    if 'zenith_edges' in gp:
        zenith_edges = gp['zenith_edges'][:]
    else:
        # Backward compat: compute from stored dec edges using the South-Pole
        # identity zenith = arccos(-sin(dec)).
        zenith_edges = np.degrees(np.arccos(-np.sin(np.radians(dec_edges_raw))))
    data = {
        'log10e_true_edges': gp['log10e_true_edges'][:],
        'dec_edges':         dec_edges_raw,
        'zenith_edges':      zenith_edges,
        'log10e_reco_lo':    gp['log10e_reco_lo'][:],
        'log10e_reco_hi':    gp['log10e_reco_hi'][:],
        'psf_lo':            gp['psf_lo'][:],
        'psf_hi':            gp['psf_hi'][:],
        'ang_err_lo':        gp['ang_err_lo'][:],
        'ang_err_hi':        gp['ang_err_hi'][:],
        'fractional_counts': gp['fractional_counts'][:],
    }
    return _build_smearing_sampler(data)


def _write_smearing_group(hf: h5.File, name: str, data: dict) -> None:
    """Write a smearing data dict to an HDF5 group."""
    grp = hf.create_group(name)
    for key, arr in data.items():
        grp.create_dataset(key, data=arr)


def _parse_aeff_csv(path: str):
    """Parse one IceCube data-release effectiveArea.csv.

    Args:
        path: Path to the effectiveArea.csv file.

    Returns:
        Tuple ``(log10e_centers, sindec_centers, aeff_cm2)`` where
        ``log10e_centers`` has shape ``(n_e,)``, ``sindec_centers`` has shape
        ``(n_dec,)``, and ``aeff_cm2`` has shape ``(n_e, n_dec)`` in cm².
    """
    csv = np.genfromtxt(path, comments='#')

    log10e_lo_vals = np.sort(np.unique(csv[:, 0]))
    dec_lo_vals    = np.sort(np.unique(csv[:, 2]))
    dec_hi_map     = {lo: csv[csv[:, 2] == lo, 3][0] for lo in dec_lo_vals}

    log10e_step    = float(np.median(np.diff(log10e_lo_vals)))
    log10e_centers = log10e_lo_vals + log10e_step / 2.0

    dec_centers    = np.array([0.5 * (lo + dec_hi_map[lo]) for lo in dec_lo_vals])
    sindec_centers = np.sin(np.radians(dec_centers))

    n_e   = len(log10e_centers)
    n_dec = len(sindec_centers)
    aeff  = np.zeros((n_e, n_dec))

    for row in csv:
        i_e = int(np.searchsorted(log10e_lo_vals, row[0]))
        i_d = int(np.searchsorted(dec_lo_vals,    row[2]))
        if 0 <= i_e < n_e and 0 <= i_d < n_dec:
            aeff[i_e, i_d] = row[4]   # cm²

    return log10e_centers, sindec_centers, aeff



# ---------------------------------------------------------------------------
# Smearing CSV parsing
# ---------------------------------------------------------------------------

def _smearing_tables_from_file(path: str) -> dict:
    """Parse one IceCube data-release smearing CSV file into sampling tables.

    The smearing file has 14 E_true bins × 3 dec bands × 8800 rows per cell
    (20 E_reco × 20 PSF × 22 AngErr bins).  Two degenerate cells
    (E_true < 2.5 GeV, dec < −10°) are padded to maintain a uniform shape.

    Note:
        ``np.genfromtxt`` on a 56 MB CSV is extremely slow (~10–20 min).
        This function caches the parsed result as a ``.npz`` file alongside
        the CSV on first load; subsequent loads take < 1 s.

    Args:
        path: Path to the IceCube smearing CSV file.

    Returns:
        dict with keys ``log10e_true_edges`` (15,), ``dec_edges`` (4,),
        ``log10e_reco_lo`` (14, 3, 20), ``log10e_reco_hi`` (14, 3, 20),
        ``psf_lo`` (14, 3, 20), ``psf_hi`` (14, 3, 20),
        ``ang_err_lo`` (14, 3, 22), ``ang_err_hi`` (14, 3, 22), and
        ``fractional_counts`` (14, 3, 20, 20, 22).
    """
    import os
    cache_path = os.path.splitext(path)[0] + "_smearing_cache.npz"
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(path):
        c = np.load(cache_path)
        return {k: c[k] for k in c.files}

    try:
        import pandas as _pd
        d = _pd.read_csv(path, comment='#', header=None, sep=r'\s+').to_numpy()
    except ImportError:
        d = np.genfromtxt(path, comments='#')

    etrue_lo_vals = np.sort(np.unique(d[:, 0]))
    dec_lo_vals   = np.sort(np.unique(d[:, 2]))
    n_et  = len(etrue_lo_vals)
    n_dec = len(dec_lo_vals)

    # Global bin edges
    etrue_edges = np.append(etrue_lo_vals, d[:, 1].max())
    dec_edges   = np.append(dec_lo_vals,   d[:, 3].max())

    N_ER = 20; N_PSF = 20; N_AE = 22

    frac   = np.zeros((n_et, n_dec, N_ER, N_PSF, N_AE))
    er_lo  = np.zeros((n_et, n_dec, N_ER))
    er_hi  = np.zeros((n_et, n_dec, N_ER))
    p_lo   = np.zeros((n_et, n_dec, N_PSF))
    p_hi   = np.zeros((n_et, n_dec, N_PSF))
    ae_lo  = np.zeros((n_et, n_dec, N_AE))
    ae_hi  = np.zeros((n_et, n_dec, N_AE))

    for i_et, et in enumerate(etrue_lo_vals):
        for i_dc, dc in enumerate(dec_lo_vals):
            mask = (d[:, 0] == et) & (d[:, 2] == dc)
            sub  = d[mask]
            cf, cel, ceh, cpl, cph, cal, cah = _build_cell_arrays(sub, N_ER, N_PSF, N_AE)
            frac[i_et, i_dc]   = cf
            er_lo[i_et, i_dc]  = cel
            er_hi[i_et, i_dc]  = ceh
            p_lo[i_et, i_dc]   = cpl
            p_hi[i_et, i_dc]   = cph
            ae_lo[i_et, i_dc]  = cal
            ae_hi[i_et, i_dc]  = cah

    # Convert IceCube dec-band edges to local zenith edges via the South-Pole
    # identity: zenith = arccos(-sin(dec)).  At IceCube (lat=-90°) the two
    # coordinate systems are trivially related; storing zenith lets the
    # smearing sampler work correctly for any detector latitude.
    zenith_edges = np.degrees(np.arccos(-np.sin(np.radians(dec_edges))))

    result = {
        'log10e_true_edges':  etrue_edges,
        'dec_edges':          dec_edges,      # kept for reference / backward compat
        'zenith_edges':       zenith_edges,   # primary lookup axis in sampler
        'log10e_reco_lo':     er_lo,
        'log10e_reco_hi':     er_hi,
        'psf_lo':             p_lo,
        'psf_hi':             p_hi,
        'ang_err_lo':         ae_lo,
        'ang_err_hi':         ae_hi,
        'fractional_counts':  frac,
    }
    try:
        np.savez(cache_path, **result)
    except Exception:
        pass  # non-fatal — next run will just re-parse
    return result


def _build_cell_arrays(
    sub: np.ndarray,
    n_er: int = 20,
    n_psf: int = 20,
    n_ae: int = 22,
) -> tuple:
    """Build the fractional counts array and bin edge arrays for a single (E_true, dec) cell.

    Uses (lo, hi) pairs as bin keys to correctly handle the one degenerate
    zero-width AngErr bin present in the IceCube smearing tables.  Cells with
    fewer bins than the target (degenerate low-E / south-sky cells) are
    zero-padded to the target shape.

    Args:
        sub: Array of rows from the smearing CSV for a single (E_true, dec)
            cell.
        n_er: Target number of E_reco bins.  Default 20.
        n_psf: Target number of PSF bins.  Default 20.
        n_ae: Target number of AngErr bins.  Default 22.

    Returns:
        Tuple ``(counts, er_lo, er_hi, psf_lo, psf_hi, ae_lo, ae_hi)``
        where ``counts`` has shape ``(n_er, n_psf, n_ae)`` and the edge
        arrays have shapes ``(n_er,)``, ``(n_psf,)``, and ``(n_ae,)``
        respectively.
    """
    er_bins  = sorted(set(zip(sub[:, 4].round(6), sub[:, 5].round(6))))
    psf_bins = sorted(set(zip(sub[:, 6].round(6), sub[:, 7].round(6))))
    ae_bins  = sorted(set(zip(sub[:, 8].round(6), sub[:, 9].round(6))))

    # Pad degenerate cells with dummy zero-width bins at 0
    while len(er_bins)  < n_er:  er_bins.append((0.0, 0.0))
    while len(psf_bins) < n_psf: psf_bins.append((0.0, 0.0))
    while len(ae_bins)  < n_ae:  ae_bins.append((0.0, 0.0))

    er_idx  = {b: i for i, b in enumerate(er_bins)}
    psf_idx = {b: i for i, b in enumerate(psf_bins)}
    ae_idx  = {b: i for i, b in enumerate(ae_bins)}

    counts = np.zeros((n_er, n_psf, n_ae))
    for row in sub:
        i = er_idx.get((round(row[4], 6), round(row[5], 6)))
        j = psf_idx.get((round(row[6], 6), round(row[7], 6)))
        k = ae_idx.get((round(row[8], 6), round(row[9], 6)))
        if i is not None and j is not None and k is not None:
            counts[i, j, k] = row[10]

    return (
        counts,
        np.array([b[0] for b in er_bins]),
        np.array([b[1] for b in er_bins]),
        np.array([b[0] for b in psf_bins]),
        np.array([b[1] for b in psf_bins]),
        np.array([b[0] for b in ae_bins]),
        np.array([b[1] for b in ae_bins]),
    )


# ---------------------------------------------------------------------------
# Smearing sampler
# ---------------------------------------------------------------------------

def _build_smearing_sampler(data: dict) -> Callable:
    """Build a joint (E_reco, PSF, AngErr) sampler from a smearing data dict.

    Precomputes cumulative distributions for O(log n) sampling at call time.

    Args:
        data: dict as returned by ``_smearing_tables_from_file`` or read from
            HDF5.

    Returns:
        Callable with signature
        ``sample(true_energy_GeV, zenith_rad, rng=None) -> (reco_energy_GeV, psi_rad, ang_err_rad)``
        where ``zenith_rad`` is the local zenith angle at the detector (not
        equatorial declination).  The smearing bands [0°, 80°], [80°, 100°],
        [100°, 180°] correspond to IceCube downgoing, near-horizon, and
        upgoing event classes respectively.
    """
    etrue_edges   = data['log10e_true_edges']   # (n_et+1,)
    zenith_edges  = data['zenith_edges']        # (n_dec+1,)  degrees
    er_lo  = data['log10e_reco_lo']             # (n_et, n_dec, 20)
    er_hi  = data['log10e_reco_hi']
    p_lo   = data['psf_lo']                     # (n_et, n_dec, 20)  degrees
    p_hi   = data['psf_hi']
    ae_lo  = data['ang_err_lo']                 # (n_et, n_dec, 22)  degrees
    ae_hi  = data['ang_err_hi']
    frac   = data['fractional_counts']          # (n_et, n_dec, 20, 20, 22)

    n_et, n_dec, n_er, n_psf, n_ae = frac.shape

    # Precompute cumulative distributions per cell
    flat = frac.reshape(n_et, n_dec, -1)
    totals = flat.sum(axis=-1, keepdims=True)
    # Cells with no counts get a uniform fallback (they will not be reached
    # in practice because the A_eff is zero there)
    flat_norm = np.where(
        totals > 0,
        flat / totals,
        np.ones_like(flat) / flat.shape[-1],
    )
    cum = np.cumsum(flat_norm, axis=-1)   # (n_et, n_dec, n_er*n_psf*n_ae)

    def sample(true_energy_GeV: float, zenith_rad: float, rng=None):
        rand = rng.random  if rng is not None else np.random.rand
        unif = rng.uniform if rng is not None else np.random.uniform

        log10e_t  = np.log10(true_energy_GeV)
        zenith_deg = np.degrees(zenith_rad)

        i_et = int(np.clip(
            np.searchsorted(etrue_edges[1:], log10e_t), 0, n_et - 1
        ))
        i_dc = int(np.clip(
            np.searchsorted(zenith_edges[1:], zenith_deg), 0, n_dec - 1
        ))

        # Inverse-CDF sampling via searchsorted on precomputed cumulative dist
        idx = int(np.searchsorted(cum[i_et, i_dc], rand()))
        idx = min(idx, n_er * n_psf * n_ae - 1)
        i_er, i_ps, i_ae = np.unravel_index(idx, (n_er, n_psf, n_ae))

        log10e_r = unif(er_lo[i_et, i_dc, i_er], er_hi[i_et, i_dc, i_er])
        reco_energy = ureg.Quantity(10.0 ** log10e_r, "GeV")

        psi_deg = unif(p_lo[i_et, i_dc, i_ps], p_hi[i_et, i_dc, i_ps])
        psi_rad = np.radians(psi_deg)

        ae_deg  = unif(ae_lo[i_et, i_dc, i_ae], ae_hi[i_et, i_dc, i_ae])
        ae_rad  = np.radians(ae_deg)

        return reco_energy, psi_rad, ae_rad

    def sample_batch(true_energies_GeV, zenith_rads, rng=None):
        """Vectorized batch sampler. Returns plain numpy arrays (no pint units)."""
        _rng = rng if rng is not None else np.random.default_rng()
        N = len(true_energies_GeV)

        i_et = np.clip(
            np.searchsorted(etrue_edges[1:], np.log10(true_energies_GeV)),
            0, n_et - 1,
        )
        i_dc = np.clip(
            np.searchsorted(zenith_edges[1:], np.degrees(zenith_rads)),
            0, n_dec - 1,
        )

        u = _rng.random(N)
        idx_flat = np.empty(N, dtype=int)

        # Group events by (i_et, i_dc) bin — at most n_et*n_dec unique pairs.
        # Each group shares the same CDF row, so searchsorted is vectorised
        # within the group rather than once per event.
        key = i_et * n_dec + i_dc
        for k_val in np.unique(key):
            mask = key == k_val
            ie = int(k_val // n_dec)
            id_ = int(k_val % n_dec)
            idx_flat[mask] = np.searchsorted(cum[ie, id_], u[mask])

        idx_flat = np.minimum(idx_flat, n_er * n_psf * n_ae - 1)
        i_er, i_ps, i_ae = np.unravel_index(idx_flat, (n_er, n_psf, n_ae))

        log10e_r = _rng.uniform(er_lo[i_et, i_dc, i_er], er_hi[i_et, i_dc, i_er])
        reco_energies = 10.0 ** log10e_r  # GeV, plain float64

        psi_rad = np.radians(_rng.uniform(p_lo[i_et, i_dc, i_ps], p_hi[i_et, i_dc, i_ps]))
        ae_rad  = np.radians(_rng.uniform(ae_lo[i_et, i_dc, i_ae], ae_hi[i_et, i_dc, i_ae]))

        return reco_energies, psi_rad, ae_rad

    sample.batch = sample_batch
    return sample
