import logging
import numpy as np
import h5py as h5

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field
from typing import Tuple, Callable, Dict, List, Optional
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
            present only in files built via from_dataverse.
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
    def from_config(cls, config: Dict, trim_isolated: bool = True) -> 'DetectorResponse':
        """Build a DetectorResponse from a config dictionary.

        Loads IRFs from an HDF5 file. The file must use the hierarchical
        format: top-level groups named after registered morphologies (e.g.
        ``track``, ``cascade``), each containing an ``effective_area``
        subgroup and optionally ``angular_response``, ``energy_resolution``,
        and ``smearing`` subgroups.

        Args:
            config: Dictionary with either a ``detector_response_file`` key
                pointing to an HDF5 path, or a ``detector_response_toml``
                key pointing to a response TOML file.
            trim_isolated: If True (default), zero out energy bins outside the
                largest contiguous non-zero run per zenith column when loading
                the effective area.  This removes isolated low-statistics bins
                at the edges of the sensitivity range.  Pass False to load the
                raw tabulated values as-is.

        Returns:
            A configured DetectorResponse instance.

        Raises:
            ValueError: If the file contains no usable response for any
                registered morphology.
        """
        if "detector_response_toml" in config:
            return cls.from_toml(config["detector_response_toml"])

        effective_area   = {}
        angular_response = {}
        energy_response  = {}
        joint_smearing   = {}
        registered = sorted(Morphology.registered())

        with h5.File(config["detector_response_file"]) as h5f:
            for morph_name in registered:
                if morph_name not in h5f or not isinstance(h5f[morph_name], h5.Group):
                    continue
                grp = h5f[morph_name]
                if "effective_area" not in grp:
                    continue
                effa = (
                    effa_from_dr_group(grp["effective_area_dr"])
                    if "effective_area_dr" in grp
                    else effa_spline_from_group(grp["effective_area"],
                                                trim_isolated=trim_isolated)
                )
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

    @classmethod
    def from_toml(cls, path: str) -> 'DetectorResponse':
        """
        Build a DetectorResponse from a spore TOML response file.

        **Per-morphology format** (preferred)
            Use ``[track]`` and/or ``[cascade]`` top-level tables, each
            containing ``[track.effective_area]``, ``[track.psf]``, etc.
            Only the morphologies that have a table are loaded; the resulting
            DetectorResponse reports them via ``available_morphologies``.

        **Flat format** (backward-compatible, treated as track-only)
            Top-level ``[effective_area]``, ``[psf]``, and ``[smearing]``
            sections are parsed and mapped to the track morphology.

        Within each morphology block, two IRF styles are supported:

        - *Digitized*: ``[<morph>.effective_area]`` with ``file`` and
          ``[<morph>.psf]`` with ``form = "rayleigh_power_law"``.
        - *Data-release*: ``[<morph>.effective_area]`` and
          ``[<morph>.smearing]`` with ``type = "dataverse_csv"``.

        Parameters
        ----------
        path : str
            Path to the TOML file.  Relative paths inside the file are
            resolved relative to the directory containing the TOML.

        Returns
        -------
        DetectorResponse
        """
        import tomllib, os

        path = os.path.abspath(path)
        toml_dir = os.path.dirname(path)

        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)

        # Flat format (backward compat): top-level keys are response sections
        # directly (effective_area, psf, smearing).  Per-morphology format:
        # top-level keys are morphology names each containing those sections.
        _RESPONSE_KEYS = {"effective_area", "psf", "smearing"}
        is_flat = bool(_RESPONSE_KEYS & set(cfg.keys()))

        if is_flat:
            morph_configs = {"track": cfg}
        else:
            morph_configs = {k: v for k, v in cfg.items() if isinstance(v, dict)}

        effective_area   = {}
        angular_response = {}
        energy_response  = {}
        joint_smearing   = {}

        for morph, mcfg in morph_configs.items():
            effa, ang, energy, joint = _parse_morphology_config(mcfg, toml_dir)
            if effa is not None:
                effective_area[morph] = effa
            if ang is not None:
                angular_response[morph] = ang
            if energy is not None:
                energy_response[morph] = energy
            if joint is not None:
                joint_smearing[morph] = joint

        if not effective_area:
            raise ValueError("TOML defines no effective area for any morphology.")

        return cls(
            effective_area,
            angular_response,
            energy_response,
            joint_smearing if joint_smearing else None,
        )

    @classmethod
    def from_dataverse(
        cls,
        irf_dir: str,
        seasons: List[str],
        base_h5: Optional[str] = None,
        h5_path: Optional[str] = None,
    ) -> 'DetectorResponse':
        """
        Build a DetectorResponse using the joint smearing IRF from the IceCube
        data-release CSV files.

        The smearing files encode the joint conditional distribution
        P(E_reco, PSF, AngErr | E_true, dec) as fractional counts over a
        5-dimensional histogram.  Multiple seasons are averaged uniformly.
        Seasons without a dedicated IRF file (IC86_III through IC86_VII) fall
        back to the IC86_II file.

        Parameters
        ----------
        irf_dir : str
            Directory containing the IRF CSV files from the data release.
        seasons : list of str
            Season names to include, e.g. ['IC86_I', 'IC86_II'].
        base_h5 : str or None
            Path to an existing spore HDF5 response file from which the
            effective area, angular response, and energy resolution groups are
            copied.  If None the returned DetectorResponse has empty
            effective_area / angular_response / energy_response dicts.
        h5_path : str or None
            If provided, write the complete response (joint smearing plus any
            groups copied from base_h5) to this HDF5 path.  The file can then
            be reloaded with from_config().

        Returns
        -------
        DetectorResponse
        """
        import os

        # Map each season to its IRF file stem
        irf_stems = set()
        for season in seasons:
            stem = _SEASON_TO_IRF.get(season, 'IC86_II')
            irf_stems.add(stem)

        # Load and average fractional counts across unique IRF files
        frac_sum = None
        n_files = 0
        etrue_edges = dec_edges = None
        er_lo = er_hi = psf_lo = psf_hi = ae_lo = ae_hi = None

        for stem in sorted(irf_stems):
            path = os.path.join(irf_dir, f"{stem}_smearing.csv")
            tables = _smearing_tables_from_file(path)
            if frac_sum is None:
                frac_sum  = tables['fractional_counts'].copy()
                etrue_edges = tables['log10e_true_edges']
                dec_edges   = tables['dec_edges']
                er_lo  = tables['log10e_reco_lo']
                er_hi  = tables['log10e_reco_hi']
                psf_lo = tables['psf_lo']
                psf_hi = tables['psf_hi']
                ae_lo  = tables['ang_err_lo']
                ae_hi  = tables['ang_err_hi']
            else:
                frac_sum += tables['fractional_counts']
            n_files += 1

        # Normalize per (E_true, dec) cell so fracs sum to 1
        frac_avg = frac_sum / n_files
        cell_totals = frac_avg.sum(axis=(-3, -2, -1), keepdims=True)
        frac_avg = np.where(cell_totals > 0, frac_avg / cell_totals, frac_avg)

        smearing_data = {
            'log10e_true_edges':  etrue_edges,
            'dec_edges':          dec_edges,
            'log10e_reco_lo':     er_lo,
            'log10e_reco_hi':     er_hi,
            'psf_lo':             psf_lo,
            'psf_hi':             psf_hi,
            'ang_err_lo':         ae_lo,
            'ang_err_hi':         ae_hi,
            'fractional_counts':  frac_avg,
        }

        # --- Effective area from data release ---
        aeff_sum = None
        n_aeff = 0
        log10e_centers = sindec_centers = None

        for stem in sorted(irf_stems):
            path = os.path.join(irf_dir, f"{stem}_effectiveArea.csv")
            log10e_c, sindec_c, aeff_cm2 = _parse_aeff_csv(path)
            if aeff_sum is None:
                aeff_sum = aeff_cm2.copy()
                log10e_centers = log10e_c
                sindec_centers = sindec_c
            else:
                aeff_sum += aeff_cm2
            n_aeff += 1

        aeff_cm2 = aeff_sum / n_aeff

        aeff_data = {
            'log10e_centers': log10e_centers,
            'sindec_centers': sindec_centers,
            'aeff_cm2':       aeff_cm2,
        }

        # Optionally write to HDF5
        if h5_path is not None:
            with h5.File(h5_path, 'w') as hf:
                # Copy groups from existing base file if provided
                if base_h5 is not None:
                    with h5.File(base_h5, 'r') as src:
                        for key in src.keys():
                            src.copy(key, hf)
                _write_smearing_group(hf, 'track_smearing', smearing_data)
                _write_aeff_dr_group(hf, 'track_effective_area_dr', aeff_data)

        joint_smearing = {
            'track': _build_smearing_sampler(smearing_data),
        }
        track_effa_dr = _build_aeff_callable(log10e_centers, sindec_centers, aeff_cm2)

        # Load base response if provided; replace track A_eff with data-release version
        if base_h5 is not None:
            base = cls.from_config({'detector_response_file': base_h5})
            effective_area = dict(base.effective_area)
            effective_area['track'] = track_effa_dr
            return cls(
                effective_area,
                base.angular_response,
                base.energy_response,
                joint_smearing,
            )

        return cls({'track': track_effa_dr}, {}, {}, joint_smearing)


# ---------------------------------------------------------------------------
# TOML per-morphology parser
# ---------------------------------------------------------------------------

def _parse_morphology_config(mcfg: dict, toml_dir: str):
    """
    Parse one morphology block from a TOML config dict.

    Parameters
    ----------
    mcfg     : dict   Contents of the ``[track]`` or ``[cascade]`` table.
    toml_dir : str    Directory of the TOML file (for resolving relative paths).

    Returns
    -------
    effa   : Callable or None
    ang    : Callable or None   (marginal angular sampler)
    energy : Callable or None   (marginal energy sampler)
    joint  : Callable or None   (joint smearing sampler)
    """
    import os
    from ...conventions import resolve_path

    ea_cfg  = mcfg.get("effective_area", {})
    psf_cfg = mcfg.get("psf", {})
    sm_cfg  = mcfg.get("smearing", {})

    # ------------------------------------------------------------------ #
    # Effective area
    # ------------------------------------------------------------------ #
    effa = None
    if ea_cfg.get("type") == "dataverse_csv":
        irf_dir   = resolve_path(ea_cfg["irf_dir"], toml_dir)
        seasons   = ea_cfg["seasons"]
        irf_stems = set(_SEASON_TO_IRF.get(s, "IC86_II") for s in seasons)
        aeff_sum  = None
        n = 0
        for stem in sorted(irf_stems):
            log10e_c, sindec_c, aeff_cm2 = _parse_aeff_csv(
                os.path.join(irf_dir, f"{stem}_effectiveArea.csv")
            )
            aeff_sum = aeff_cm2.copy() if aeff_sum is None else aeff_sum + aeff_cm2
            n += 1
        aeff_cm2_avg = aeff_sum / n
        effa = _build_aeff_callable(log10e_c, sindec_c, aeff_cm2_avg)
    elif "file" in ea_cfg:
        csv_path = resolve_path(ea_cfg["file"], toml_dir)
        log10e_c, sindec_c, aeff_cm2 = _parse_digitized_aeff_csv(csv_path)
        effa = _build_aeff_callable(log10e_c, sindec_c, aeff_cm2)
    elif ea_cfg:
        raise ValueError(
            "TOML effective_area block must have 'file' or type='dataverse_csv'."
        )

    # ------------------------------------------------------------------ #
    # Joint smearing (data-release format)
    # ------------------------------------------------------------------ #
    joint = None
    if sm_cfg.get("type") == "dataverse_csv":
        irf_dir   = resolve_path(sm_cfg["irf_dir"], toml_dir)
        seasons   = sm_cfg["seasons"]
        irf_stems = set(_SEASON_TO_IRF.get(s, "IC86_II") for s in seasons)
        frac_sum  = None
        n = 0
        for stem in sorted(irf_stems):
            tables = _smearing_tables_from_file(
                os.path.join(irf_dir, f"{stem}_smearing.csv")
            )
            if frac_sum is None:
                frac_sum    = tables["fractional_counts"].copy()
                etrue_edges = tables["log10e_true_edges"]
                dec_edges   = tables["dec_edges"]
                er_lo, er_hi   = tables["log10e_reco_lo"], tables["log10e_reco_hi"]
                psf_lo, psf_hi = tables["psf_lo"],         tables["psf_hi"]
                ae_lo, ae_hi   = tables["ang_err_lo"],     tables["ang_err_hi"]
            else:
                frac_sum += tables["fractional_counts"]
            n += 1
        frac_avg    = frac_sum / n
        cell_totals = frac_avg.sum(axis=(-3, -2, -1), keepdims=True)
        frac_avg    = np.where(cell_totals > 0, frac_avg / cell_totals, frac_avg)
        smearing_data = {
            "log10e_true_edges": etrue_edges, "dec_edges": dec_edges,
            "log10e_reco_lo": er_lo,          "log10e_reco_hi": er_hi,
            "psf_lo": psf_lo,                 "psf_hi": psf_hi,
            "ang_err_lo": ae_lo,              "ang_err_hi": ae_hi,
            "fractional_counts": frac_avg,
        }
        joint = _build_smearing_sampler(smearing_data)

    # ------------------------------------------------------------------ #
    # Marginal PSF (digitized format — skipped when joint smearing present)
    # ------------------------------------------------------------------ #
    ang = None
    if joint is None and psf_cfg:
        form = psf_cfg.get("form")
        if form != "rayleigh_power_law":
            raise ValueError(
                f"Unknown PSF form '{form}'. Supported: 'rayleigh_power_law'."
            )
        ang = _build_rayleigh_power_law_ang_sampler(
            amplitude_deg=float(psf_cfg["amplitude_deg"]),
            pivot_gev=float(psf_cfg["pivot_gev"]),
            index=float(psf_cfg["index"]),
        )
    elif joint is None and effa is not None:
        raise ValueError(
            "Morphology config has effective_area but no [psf] or [smearing] "
            "to define an angular response."
        )

    # ------------------------------------------------------------------ #
    # Energy resolution (optional — not yet implemented)
    # ------------------------------------------------------------------ #
    energy = None

    return effa, ang, energy, joint


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

_CM_IN_EV_INV   = 8065.54815355
_CM2_IN_EV2_INV = _CM_IN_EV_INV ** 2


def effa_spline_from_group(gp: h5.Group, trim_isolated: bool = True):
    """Load effective area from an HDF5 group.

    Returns a single callable when ``tabulated_values`` has shape (N_E, N_ZEN)
    (single-species format), or a list of 6 callables when it has shape
    (6, N_E, N_ZEN) (per-species format).  The per-species format is required
    for morphologies where different neutrino flavours have significantly
    different detection efficiencies (e.g. HESE cascade at high energies).

    Parameters
    ----------
    gp : h5py.Group
        HDF5 group containing ``energies``, ``zeniths``, ``tabulated_values``,
        ``lower_bounds``, and ``upper_bounds`` datasets.
    trim_isolated : bool
        If True (default), zero out energy bins outside the largest contiguous
        non-zero run per zenith column before building the interpolator.  This
        removes isolated low-statistics bins at the edges of the sensitivity
        range.  Pass False to use the raw tabulated values as-is.
    """
    from .utils import effa_helper
    zens = gp["zeniths"][:]
    es = gp["energies"][:]  # stored in GeV
    # stored in eV^{-2}; convert to cm^2 so the sampler norm integrates to events/s
    tabulated_raw = gp["tabulated_values"][:]
    lower_bounds = gp["lower_bounds"][:]
    upper_bounds = gp["upper_bounds"][:]

    def _zero_fn(zen, e):
        scalar = np.ndim(zen) == 0
        out = np.zeros(np.atleast_1d(np.asarray(zen)).shape)
        return float(out[0]) if scalar else out

    if tabulated_raw.ndim == 3:  # per-species: shape (6, N_E, N_ZEN)
        fns = []
        for idx in range(6):
            vals = tabulated_raw[idx] / _CM2_IN_EV2_INV
            if np.all(vals == 0):
                fns.append(_zero_fn)
            else:
                fns.append(effa_helper(zens, es, vals, lower_bounds, upper_bounds,
                                       trim_isolated=trim_isolated))
        return fns
    else:  # 2-D: shape (N_E, N_ZEN)
        vals = tabulated_raw / _CM2_IN_EV2_INV
        if np.all(vals == 0):
            return _zero_fn
        return effa_helper(zens, es, vals, lower_bounds, upper_bounds,
                           trim_isolated=trim_isolated)

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
        log_e = float(np.clip(np.log(e), e_log_min, e_log_max))
        u_c   = float(np.clip(u, u_min, u_max))
        return float(i((log_e, u_c)))
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
    """
    Build a joint smearing sampler from an HDF5 smearing group written by
    from_dataverse().

    Returns a callable with signature::

        sample(true_energy_eV: float, dec_rad: float)
            -> (reco_energy_eV: float, psi_rad: float, ang_err_rad: float)

    The sample is drawn from the conditional distribution
    P(E_reco, PSF, AngErr | E_true, dec) stored as fractional counts over a
    5-D histogram with per-cell bin edges.
    """
    data = {
        'log10e_true_edges': gp['log10e_true_edges'][:],
        'dec_edges':         gp['dec_edges'][:],
        'log10e_reco_lo':    gp['log10e_reco_lo'][:],
        'log10e_reco_hi':    gp['log10e_reco_hi'][:],
        'psf_lo':            gp['psf_lo'][:],
        'psf_hi':            gp['psf_hi'][:],
        'ang_err_lo':        gp['ang_err_lo'][:],
        'ang_err_hi':        gp['ang_err_hi'][:],
        'fractional_counts': gp['fractional_counts'][:],
    }
    return _build_smearing_sampler(data)


def _write_aeff_dr_group(hf: h5.File, name: str, data: dict) -> None:
    """Write a data-release A_eff data dict to an HDF5 group."""
    grp = hf.create_group(name)
    for key, arr in data.items():
        grp.create_dataset(key, data=arr)


def _write_smearing_group(hf: h5.File, name: str, data: dict) -> None:
    """Write a smearing data dict to an HDF5 group."""
    grp = hf.create_group(name)
    for key, arr in data.items():
        grp.create_dataset(key, data=arr)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Data-release effective area
# ---------------------------------------------------------------------------

def _parse_digitized_aeff_csv(path: str):
    """
    Parse a spore digitized effective-area CSV (4-column format).

    Expected columns (whitespace-separated, ``#``-comment lines ignored):

        log10_e_gev   sindec_min   sindec_max   aeff_cm2

    ``log10_e_gev`` is the bin centre; ``sindec_min/max`` bound the
    declination band.  Multiple rows with the same ``log10_e_gev`` but
    different ``sindec`` bands build a 2-D grid; a single dec band reduces
    to a 1-D interpolator broadcast over all declinations.

    Returns
    -------
    log10e_centers : ndarray (n_e,)
    sindec_centers : ndarray (n_dec,)
    aeff_cm2       : ndarray (n_e, n_dec)
    """
    data = np.genfromtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]

    log10e_vals  = np.sort(np.unique(data[:, 0]))
    sindec_lo    = np.sort(np.unique(data[:, 1]))
    sindec_hi_map = {lo: data[data[:, 1] == lo, 2][0] for lo in sindec_lo}
    sindec_centers = np.array([0.5 * (lo + sindec_hi_map[lo]) for lo in sindec_lo])

    n_e   = len(log10e_vals)
    n_dec = len(sindec_centers)
    aeff  = np.zeros((n_e, n_dec))

    for row in data:
        i_e = int(np.searchsorted(log10e_vals, row[0]))
        i_d = int(np.searchsorted(sindec_lo,    row[1]))
        if 0 <= i_e < n_e and 0 <= i_d < n_dec:
            aeff[i_e, i_d] = row[3]

    return log10e_vals, sindec_centers, aeff


def _parse_aeff_csv(path: str):
    """
    Parse one IceCube data-release effectiveArea.csv.

    Returns
    -------
    log10e_centers : ndarray (n_e,)   log10(E_true / GeV) at bin centres
    sindec_centers : ndarray (n_dec,) sin(dec) at band centres
    aeff_cm2       : ndarray (n_e, n_dec) effective area in cm²
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


def _build_aeff_callable(
    log10e_centers: np.ndarray,
    sindec_centers: np.ndarray,
    aeff_cm2: np.ndarray,
) -> Callable:
    """
    Build a track effective-area callable from a (log10E, sin_dec) grid.

    The energy axis is interpolated in log-log space (log A_eff vs log10 E),
    which matches the near-power-law energy dependence of A_eff and gives much
    smoother interpolation than linear in A_eff between grid points.  Zeros in
    the grid are floored at a small sentinel value; the threshold is restored
    by zeroing returns below that floor.

    Parameters
    ----------
    log10e_centers : (n_e,)    log10(E / GeV)
    sindec_centers : (n_dec,)  sin(declination)
    aeff_cm2       : (n_e, n_dec)  effective area in cm²

    Returns
    -------
    Callable with signature effa(zenith_rad, energy_GeV) -> float [cm²]
    """
    nonzero = aeff_cm2[aeff_cm2 > 0]
    floor = nonzero.min() * 1e-6 if nonzero.size > 0 else 1.0
    log_aeff = np.log(np.where(aeff_cm2 > 0, aeff_cm2, floor))

    threshold = floor * 10

    n_dec = len(sindec_centers)
    if n_dec == 1:
        # Single declination band: 1-D interpolation over energy only.
        from scipy.interpolate import PchipInterpolator as _Pchip
        _e_interp = _Pchip(log10e_centers, log_aeff[:, 0], extrapolate=False)

        def effa(zen_rad, e_GeV):
            scalar = np.ndim(e_GeV) == 0
            e_GeV = np.atleast_1d(np.asarray(e_GeV, dtype=float))
            log10e = np.log10(e_GeV)
            v = _e_interp(log10e)
            vals = np.where(np.isnan(v), 0.0, np.exp(v))
            vals = np.where(vals > threshold, vals, 0.0)
            return float(vals[0]) if scalar else vals

        return effa

    # PCHIP requires >= 4 points per dimension; fall back to linear for
    # the dec axis if there are fewer than 4 declination bands.
    method = 'pchip' if n_dec >= 4 else 'linear'
    interp = RegularGridInterpolator(
        (log10e_centers, sindec_centers),
        log_aeff,
        method=method,
        bounds_error=False,
        fill_value=np.log(floor),
    )

    sd_min = float(sindec_centers[0])
    sd_max = float(sindec_centers[-1])

    def effa(zen_rad, e_GeV):
        scalar = np.ndim(zen_rad) == 0 and np.ndim(e_GeV) == 0
        zen_rad = np.atleast_1d(np.asarray(zen_rad, dtype=float))
        e_GeV   = np.atleast_1d(np.asarray(e_GeV,   dtype=float))
        log10e = np.log10(e_GeV)
        # South-Pole convention: sin(dec) = -cos(zenith)
        sindec = np.clip(-np.cos(zen_rad), sd_min, sd_max)
        # Broadcast so zen and e can have different lengths (e.g. scalar zen, array e)
        shape = np.broadcast_shapes(log10e.shape, sindec.shape)
        log10e = np.broadcast_to(log10e, shape)
        sindec = np.broadcast_to(sindec, shape)
        pts  = np.column_stack([log10e, sindec])
        vals = np.exp(interp(pts))
        vals = np.where(vals > threshold, vals, 0.0)
        return float(vals[0]) if scalar else vals

    return effa


def effa_from_dr_group(gp: h5.Group) -> Callable:
    """Build A_eff callable from an HDF5 data-release effective-area group."""
    return _build_aeff_callable(
        gp['log10e_centers'][:],
        gp['sindec_centers'][:],
        gp['aeff_cm2'][:],
    )


def _build_rayleigh_power_law_ang_sampler(
    amplitude_deg: float,
    pivot_gev: float,
    index: float,
) -> Callable:
    """
    Build an angular-response sampler for a Rayleigh PSF whose scale follows
    a power law in energy.

    The median PSF at energy E is:

        median(E) = amplitude_deg * (E / pivot_gev)^{-index}   [degrees]

    The Rayleigh scale is:

        sigma_R(E) = median(E) / sqrt(ln 2)

    Sampling the opening angle:

        psi = sigma_R * sqrt(-2 * ln(1 - u)),   u ~ Uniform(0, 1)

    Parameters
    ----------
    amplitude_deg : float   Median PSF at pivot_gev, in degrees.
    pivot_gev     : float   Pivot energy in GeV.
    index         : float   Power-law index (positive → PSF improves with E).

    Returns
    -------
    Callable with signature ``ang_sampler(energy_GeV, u) -> psi_rad``
    where ``u`` is a Uniform(0, 1) variate.
    """
    _sqrt_ln2 = np.sqrt(np.log(2.0))

    def ang_sampler(energy_GeV: float, u: float) -> float:
        median_deg = amplitude_deg * (energy_GeV / pivot_gev) ** (-index)
        sigma_R_rad = np.radians(median_deg) / _sqrt_ln2
        # Rayleigh inverse CDF: psi = sigma * sqrt(-2 ln(1-u))
        u_clamped = float(np.clip(u, 0.0, 1.0 - 1e-15))
        return sigma_R_rad * np.sqrt(-2.0 * np.log(1.0 - u_clamped))

    return ang_sampler


# ---------------------------------------------------------------------------
# Smearing CSV parsing
# ---------------------------------------------------------------------------

def _smearing_tables_from_file(path: str) -> dict:
    """
    Parse one IceCube data-release smearing CSV file and return a dict with
    the 5-D fractional_counts array and per-cell bin edge arrays.

    The smearing file has 14 E_true bins × 3 dec bands × 8800 rows per cell
    (20 E_reco × 20 PSF × 22 AngErr bins).  Two degenerate cells
    (E_true < 2.5 GeV, dec < −10°) are padded to maintain a uniform shape.

    Returns
    -------
    dict with keys:
        log10e_true_edges  (15,)
        dec_edges          (4,)    degrees
        log10e_reco_lo     (14, 3, 20)
        log10e_reco_hi     (14, 3, 20)
        psf_lo             (14, 3, 20)
        psf_hi             (14, 3, 20)
        ang_err_lo         (14, 3, 22)
        ang_err_hi         (14, 3, 22)
        fractional_counts  (14, 3, 20, 20, 22)
    """
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

    return {
        'log10e_true_edges': etrue_edges,
        'dec_edges':         dec_edges,
        'log10e_reco_lo':    er_lo,
        'log10e_reco_hi':    er_hi,
        'psf_lo':            p_lo,
        'psf_hi':            p_hi,
        'ang_err_lo':        ae_lo,
        'ang_err_hi':        ae_hi,
        'fractional_counts': frac,
    }


def _build_cell_arrays(
    sub: np.ndarray,
    n_er: int = 20,
    n_psf: int = 20,
    n_ae: int = 22,
) -> tuple:
    """
    Build the fractional counts array and bin edge arrays for a single
    (E_true, dec) cell.

    Uses (lo, hi) pairs as bin keys to correctly handle the one degenerate
    zero-width AngErr bin present in the IceCube smearing tables.  Cells
    with fewer bins than the target (degenerate low-E / south-sky cells) are
    zero-padded to the target shape.

    Returns
    -------
    counts  : ndarray (n_er, n_psf, n_ae)
    er_lo   : ndarray (n_er,)
    er_hi   : ndarray (n_er,)
    psf_lo  : ndarray (n_psf,)
    psf_hi  : ndarray (n_psf,)
    ae_lo   : ndarray (n_ae,)
    ae_hi   : ndarray (n_ae,)
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
    """
    Build a joint (E_reco, PSF, AngErr) sampler from a smearing data dict.

    Precomputes cumulative distributions for O(log n) sampling at call time.

    Parameters
    ----------
    data : dict
        As returned by _smearing_tables_from_file or read from HDF5.

    Returns
    -------
    Callable with signature::

        sample(true_energy_GeV: float, dec_rad: float, rng=None)
            -> (reco_energy_GeV: pint.Quantity, psi_rad: float, ang_err_rad: float)
    """
    etrue_edges = data['log10e_true_edges']   # (n_et+1,)
    dec_edges   = data['dec_edges']            # (n_dec+1,)  degrees
    er_lo  = data['log10e_reco_lo']            # (n_et, n_dec, 20)
    er_hi  = data['log10e_reco_hi']
    p_lo   = data['psf_lo']                    # (n_et, n_dec, 20)  degrees
    p_hi   = data['psf_hi']
    ae_lo  = data['ang_err_lo']                # (n_et, n_dec, 22)  degrees
    ae_hi  = data['ang_err_hi']
    frac   = data['fractional_counts']         # (n_et, n_dec, 20, 20, 22)

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

    def sample(true_energy_GeV: float, dec_rad: float, rng=None):
        rand = rng.random  if rng is not None else np.random.rand
        unif = rng.uniform if rng is not None else np.random.uniform

        log10e_t = np.log10(true_energy_GeV)
        dec_deg  = np.degrees(dec_rad)

        i_et = int(np.clip(
            np.searchsorted(etrue_edges[1:], log10e_t), 0, n_et - 1
        ))
        i_dc = int(np.clip(
            np.searchsorted(dec_edges[1:], dec_deg), 0, n_dec - 1
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

    return sample
