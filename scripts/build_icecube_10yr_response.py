"""
Build a spore-format HDF5 detector response file from the IceCube 10-year
point source data release (arXiv:2101.09836).

Data release reference
----------------------
IceCube Collaboration, "IceCube Data for Neutrino Point-Source Searches:
Years 2008-2018", https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/

Download and unpack the data release, then point the environment variable
IC10YR_DATA_DIR at the top-level directory (the one that contains the
"irfs/" and "uptime/" subdirectories).

This is a track-only (muon neutrino) selection.  A placeholder cascade
channel is included for spore compatibility, scaled from the track
effective area with a configurable factor.

Effective area
--------------
Each of the five unique IRF configurations (IC40, IC59, IC79, IC86_I,
IC86_II) has its own effective area CSV.  Seasons IC86_II through IC86_VII
all reuse the IC86_II response (as documented in the data release README).
The output effective area is the livetime-weighted average across all
seasons, computed from the per-season uptime files.

Angular response (PSF)
----------------------
Derived from the IC86_II smearing matrix, which is the dominant
contribution to the total livetime.  The smearing table provides
fractional counts in 5D bins of (E_nu, dec, E_reco, PSF, AngErr).
For each E_nu bin the PSF marginal is obtained by summing over E_reco
and AngErr.  The resulting empirical CDF is interpolated to give the
inverse CDF table spore expects.  Three declination bands are available
(-90/–10, -10/+10, +10/+90 deg); the northern band is used because this
selection is dominated by upgoing tracks.

Energy resolution
-----------------
Derived from the IC86_II smearing matrix in the same way: for each E_nu
bin the log10(E_reco/E_nu) marginal is obtained by summing over PSF and
AngErr.  The result is converted to the ln(E_reco/E_true) convention used
by spore.

Run
---
    # with data release at the default path:
    python scripts/build_icecube_10yr_response.py

    # or with an explicit path:
    IC10YR_DATA_DIR=/path/to/dataverse_files python scripts/build_icecube_10yr_response.py
"""

import os
import sys
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Data release path
# ---------------------------------------------------------------------------

# Download from:
# https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/
_DEFAULT_DATA_DIR = os.path.expanduser("~/Downloads/dataverse_files 2")
DATA_DIR = os.environ.get("IC10YR_DATA_DIR", _DEFAULT_DATA_DIR)

OUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "resources",
    "icecube_10yr_response.h5",
)

# ---------------------------------------------------------------------------
# Unit conversions  (mirror of spore/conventions/units.py)
# ---------------------------------------------------------------------------
_CM_IN_EV_INV   = 8065.54815355       # eV^-1 per cm
_CM2_IN_EV2_INV = _CM_IN_EV_INV ** 2  # eV^-2 per cm²
_EV_PER_GEV     = 1.0e9

# ---------------------------------------------------------------------------
# Season → IRF mapping  (README: IC86-2012 through IC86-2017 reuse IC86_II)
# ---------------------------------------------------------------------------
_SEASON_TO_IRF = {
    "IC40":     "IC40",
    "IC59":     "IC59",
    "IC79":     "IC79",
    "IC86_I":   "IC86_I",
    "IC86_II":  "IC86_II",
    "IC86_III": "IC86_II",
    "IC86_IV":  "IC86_II",
    "IC86_V":   "IC86_II",
    "IC86_VI":  "IC86_II",
    "IC86_VII": "IC86_II",
}

# ---------------------------------------------------------------------------
# Output grid parameters
# ---------------------------------------------------------------------------
EMIN_GEV  = 1e2    # 100 GeV
EMAX_GEV  = 1e8    # 100 PeV
N_E       = 50
N_ZEN     = 20
N_U_ANG   = 200    # quantile points in angular response table
N_U_E     = 200    # quantile points in energy resolution table

# The smearing matrix only covers the northern sky (dec > -10) well for
# tracks.  Use this declination band for PSF and energy resolution.
_PSF_DEC_MIN = 10.0   # deg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_livetimes() -> dict[str, float]:
    """Return per-season livetime in days from the uptime CSV files."""
    uptime_dir = os.path.join(DATA_DIR, "uptime")
    if not os.path.isdir(uptime_dir):
        sys.exit(
            f"ERROR: uptime directory not found at {uptime_dir}\n"
            f"Set IC10YR_DATA_DIR to the unpacked data release directory."
        )
    livetimes: dict[str, float] = {}
    for fname in sorted(os.listdir(uptime_dir)):
        if not fname.endswith(".csv"):
            continue
        season = fname.replace("_exp.csv", "")
        data = np.genfromtxt(os.path.join(uptime_dir, fname), comments="#")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        livetimes[season] = float((data[:, 1] - data[:, 0]).sum())
    return livetimes


def _irf_livetimes(livetimes: dict[str, float]) -> dict[str, float]:
    """Aggregate per-season livetimes into per-IRF totals."""
    totals: dict[str, float] = {}
    for season, lt in livetimes.items():
        irf = _SEASON_TO_IRF[season]
        totals[irf] = totals.get(irf, 0.0) + lt
    return totals


def _load_aeff_csv(irf: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an effective area CSV for one IRF configuration.

    Returns
    -------
    log10e_mid : shape (N_Ebins,)   — bin-centre log10(E_nu / GeV)
    dec_mid    : shape (N_Dbins,)   — bin-centre declination [deg]
    aeff_cm2   : shape (N_Ebins, N_Dbins)
    """
    path = os.path.join(DATA_DIR, "irfs", f"{irf}_effectiveArea.csv")
    data = np.genfromtxt(path, comments="#")
    # columns: log10e_min, log10e_max, dec_min, dec_max, aeff_cm2
    log10e_min = data[:, 0]
    log10e_max = data[:, 1]
    dec_min    = data[:, 2]
    dec_max    = data[:, 3]
    aeff       = data[:, 4]

    e_bins  = np.unique(np.column_stack([log10e_min, log10e_max]), axis=0)
    d_bins  = np.unique(np.column_stack([dec_min,    dec_max]),    axis=0)

    log10e_mid = 0.5 * (e_bins[:, 0] + e_bins[:, 1])
    dec_mid    = 0.5 * (d_bins[:, 0] + d_bins[:, 1])

    aeff_grid = np.zeros((len(log10e_mid), len(dec_mid)))
    for i, (elo, ehi) in enumerate(e_bins):
        for j, (dlo, dhi) in enumerate(d_bins):
            mask = (
                (log10e_min == elo) & (log10e_max == ehi) &
                (dec_min    == dlo) & (dec_max    == dhi)
            )
            if mask.sum() == 1:
                aeff_grid[i, j] = aeff[mask][0]

    return log10e_mid, dec_mid, aeff_grid


def _build_aeff_grid(
    irf_livetimes_days: dict[str, float],
    energies_gev: np.ndarray,
    zeniths_rad: np.ndarray,
) -> np.ndarray:
    """
    Build the livetime-weighted average effective area on the output grid.

    Converts the per-IRF (log10E, dec) tables into a single (N_E, N_ZEN)
    grid of A_eff in eV^-2.  The IceCube geometry is used:
    zenith = π/2 − dec (in radians).

    Parameters
    ----------
    irf_livetimes_days : dict mapping IRF name to its total livetime [days]
    energies_gev       : output energy grid [GeV], shape (N_E,)
    zeniths_rad        : output zenith grid [rad], shape (N_ZEN,)

    Returns
    -------
    aeff_eV2 : shape (N_E, N_ZEN)
    """
    total_lt = sum(irf_livetimes_days.values())
    log10_e_out = np.log10(energies_gev)
    # For IceCube at South Pole: dec_deg = 90 - zenith_deg
    dec_out_deg = 90.0 - np.degrees(zeniths_rad)

    aeff_sum = np.zeros((len(energies_gev), len(zeniths_rad)))

    for irf, lt in irf_livetimes_days.items():
        print(f"  Loading {irf} effective area (livetime={lt:.1f} days) ...")
        log10e_mid, dec_mid, aeff_grid = _load_aeff_csv(irf)

        # Interpolate in (log10E, dec) space; clamp to 0 outside grid
        interp = RegularGridInterpolator(
            (log10e_mid, dec_mid),
            aeff_grid,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        pts = np.array(
            [[le, d] for le in log10_e_out for d in dec_out_deg]
        )
        aeff_interp = interp(pts).reshape(len(energies_gev), len(zeniths_rad))
        aeff_interp = np.clip(aeff_interp, 0.0, None)
        aeff_sum += aeff_interp * lt

    aeff_avg_cm2 = aeff_sum / total_lt
    return aeff_avg_cm2 * _CM2_IN_EV2_INV


def _load_smearing(irf: str = "IC86_II") -> np.ndarray:
    """Load the smearing matrix CSV for one IRF (returns raw numpy array)."""
    path = os.path.join(DATA_DIR, "irfs", f"{irf}_smearing.csv")
    print(f"  Loading {irf} smearing matrix (this may take a moment) ...")
    return np.genfromtxt(path, comments="#")


def _build_angular_response(
    smearing: np.ndarray,
    energies_gev: np.ndarray,
    n_u: int,
    dec_min_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build angular response (PSF) inverse-CDF from the smearing matrix.

    Uses the declination band with dec_min >= dec_min_deg (northern sky),
    marginalising over E_reco and AngErr.  The empirical CDF over PSF bin
    midpoints is interpolated to n_u uniform quantile points.

    Parameters
    ----------
    smearing     : raw smearing array, shape (N, 11)
    energies_gev : output energy grid [GeV], shape (N_E,)
    n_u          : number of quantile points
    dec_min_deg  : use only rows with Dec_nu_min >= this value [deg]

    Returns
    -------
    us       : shape (n_u,)
    inv_cdfs : shape (N_E, n_u), angles in radians
    """
    # columns: 0=log10enu_min, 1=log10enu_max, 2=dec_min, 3=dec_max,
    #          4=log10e_min,   5=log10e_max,   6=psf_min, 7=psf_max,
    #          8=angerr_min,   9=angerr_max,   10=frac
    north = smearing[smearing[:, 2] >= dec_min_deg]

    us = np.linspace(0.0, 1.0 - 1e-9, n_u)
    n_e = len(energies_gev)
    inv_cdfs = np.zeros((n_e, n_u))

    log10_e_out = np.log10(energies_gev)
    enu_bins = np.unique(np.column_stack([north[:, 0], north[:, 1]]), axis=0)
    enu_mids = 0.5 * (enu_bins[:, 0] + enu_bins[:, 1])

    # Per E_nu bin: marginalise over E_reco and AngErr → PSF CDF
    enu_cdfs: list[tuple[np.ndarray, np.ndarray]] = []
    for elo, ehi in enu_bins:
        mask = (north[:, 0] == elo) & (north[:, 1] == ehi)
        sub  = north[mask]
        psf_edges = np.unique(np.column_stack([sub[:, 6], sub[:, 7]]), axis=0)
        psf_marg  = np.array([
            sub[(sub[:, 6] == lo) & (sub[:, 7] == hi), 10].sum()
            for lo, hi in psf_edges
        ])
        psf_mid   = np.radians(0.5 * (psf_edges[:, 0] + psf_edges[:, 1]))
        # Build CDF (psf_marg sums to ~1)
        cdf = np.concatenate([[0.0], np.cumsum(psf_marg)])
        cdf_mid = 0.5 * (np.radians(psf_edges[:, 0]) + np.radians(psf_edges[:, 1]))
        cdf_at_mid = 0.5 * (cdf[:-1] + cdf[1:])
        enu_cdfs.append((cdf_at_mid, psf_mid))

    # Interpolate each E_nu row onto the output energy grid
    for ie, log10_e in enumerate(log10_e_out):
        # Find nearest E_nu bin by log10 distance
        idx = int(np.argmin(np.abs(enu_mids - log10_e)))
        cdf_vals, psf_vals = enu_cdfs[idx]
        # Invert: for each u, find PSF
        inv_cdfs[ie] = np.interp(us, cdf_vals, psf_vals)

    return us, inv_cdfs


def _build_energy_resolution(
    smearing: np.ndarray,
    energies_gev: np.ndarray,
    n_u: int,
    dec_min_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build energy resolution inverse-CDF from the smearing matrix.

    Uses the northern sky declination band, marginalising over PSF and
    AngErr.  The result is expressed as ln(E_reco / E_nu), which is the
    convention used by spore.

    Parameters
    ----------
    smearing     : raw smearing array, shape (N, 11)
    energies_gev : output energy grid [GeV], shape (N_E,)
    n_u          : number of quantile points
    dec_min_deg  : use only rows with Dec_nu_min >= this value [deg]

    Returns
    -------
    us      : shape (n_u,)
    inv_cdf : shape (n_u,)  — ln(E_reco / E_nu), energy-independent average
    """
    north = smearing[smearing[:, 2] >= dec_min_deg]

    enu_bins = np.unique(np.column_stack([north[:, 0], north[:, 1]]), axis=0)
    enu_mids = 0.5 * (enu_bins[:, 0] + enu_bins[:, 1])

    # Build weighted average CDF over E_nu bins (weight by bin width in log10E)
    # log10(E_reco/GeV) bins; convert to ln(E_reco/E_nu) = ln10 * (log10_ereco - log10_enu)
    all_ln_ratios: list[float] = []
    all_weights:   list[float] = []

    log10_e_out  = np.log10(energies_gev)
    e_weight_map = np.ones(len(enu_mids))   # equal weight across E_nu bins

    for i_enu, (elo, ehi) in enumerate(enu_bins):
        enu_mid_gev = 10 ** (0.5 * (elo + ehi))
        mask = (north[:, 0] == elo) & (north[:, 1] == ehi)
        sub  = mask.nonzero()[0]
        if len(sub) == 0:
            continue
        ereco_edges = np.unique(
            np.column_stack([north[sub, 4], north[sub, 5]]), axis=0
        )
        for elo_r, ehi_r in ereco_edges:
            m2 = (
                (north[:, 0] == elo) & (north[:, 1] == ehi) &
                (north[:, 4] == elo_r) & (north[:, 5] == ehi_r)
            )
            frac = north[m2, 10].sum()
            if frac <= 0:
                continue
            ereco_mid_gev = 10 ** (0.5 * (elo_r + ehi_r))
            ln_ratio = np.log(ereco_mid_gev / enu_mid_gev)
            all_ln_ratios.append(ln_ratio)
            all_weights.append(frac * e_weight_map[i_enu])

    all_ln_ratios = np.array(all_ln_ratios)
    all_weights   = np.array(all_weights)
    all_weights  /= all_weights.sum()

    # Sort and build weighted CDF
    order     = np.argsort(all_ln_ratios)
    ln_sorted = all_ln_ratios[order]
    w_sorted  = all_weights[order]
    cdf       = np.cumsum(w_sorted)
    cdf       = np.concatenate([[0.0], cdf])
    ln_sorted = np.concatenate([[ln_sorted[0]], ln_sorted])

    us      = np.linspace(cdf[0], cdf[-1], n_u)
    inv_cdf = np.interp(us, cdf, ln_sorted)

    return us, inv_cdf


def _write_hdf5(
    output_path: str,
    energies_eV: np.ndarray,
    zeniths: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    track_aeff: np.ndarray,
    cascade_aeff: np.ndarray,
    us_ang: np.ndarray,
    inv_cdfs_ang: np.ndarray,
    us_e: np.ndarray,
    inv_cdf_e: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["source"] = (
            "IceCube 10-year point source data release (arXiv:2101.09836); "
            "https://icecube.wisc.edu/data-releases/2021/01/"
            "all-sky-point-source-icecube-data-years-2008-2018/"
        )

        for name, aeff in [
            ("track_effective_area",   track_aeff),
            ("cascade_effective_area", cascade_aeff),
        ]:
            g = f.create_group(name)
            g.create_dataset("energies",         data=energies_eV)
            g.create_dataset("zeniths",          data=zeniths)
            g.create_dataset("tabulated_values", data=aeff)
            g.create_dataset("lower_bounds",     data=lower_bounds)
            g.create_dataset("upper_bounds",     data=upper_bounds)

        for name in ["track_angular_response", "cascade_angular_response"]:
            g = f.create_group(name)
            g.create_dataset("energies",  data=energies_eV)
            g.create_dataset("us",        data=us_ang)
            g.create_dataset("inv_cdfs",  data=inv_cdfs_ang)

        for name in ["track_energy_resolution", "cascade_energy_resolution"]:
            g = f.create_group(name)
            g.create_dataset("us",      data=us_e)
            g.create_dataset("inv_cdf", data=inv_cdf_e)

    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isdir(DATA_DIR):
        sys.exit(
            f"ERROR: data directory not found at {DATA_DIR}\n"
            "Download the 10-year point source data release from:\n"
            "  https://icecube.wisc.edu/data-releases/2021/01/"
            "all-sky-point-source-icecube-data-years-2008-2018/\n"
            "and set IC10YR_DATA_DIR to the unpacked directory."
        )

    # ------------------------------------------------------------------
    # Livetimes
    # ------------------------------------------------------------------
    print("Computing livetimes ...")
    livetimes    = _load_livetimes()
    irf_lts      = _irf_livetimes(livetimes)
    total_lt     = sum(irf_lts.values())
    print(f"  Total livetime: {total_lt:.1f} days = {total_lt/365.25:.2f} years")
    for irf, lt in sorted(irf_lts.items()):
        print(f"  {irf}: {lt:.1f} days")

    # ------------------------------------------------------------------
    # Output grids
    # ------------------------------------------------------------------
    energies_gev = np.logspace(np.log10(EMIN_GEV), np.log10(EMAX_GEV), N_E)
    energies_eV  = energies_gev * _EV_PER_GEV
    zeniths      = np.linspace(0.0, np.pi, N_ZEN)

    log10_emin   = np.log10(energies_eV[0])
    log10_emax   = np.log10(energies_eV[-1])
    lower_bounds = np.array([[log10_emin]])
    upper_bounds = np.array([[log10_emax]])

    # ------------------------------------------------------------------
    # Effective area
    # ------------------------------------------------------------------
    print("\nBuilding livetime-weighted effective area ...")
    track_aeff   = _build_aeff_grid(irf_lts, energies_gev, zeniths)
    cascade_aeff = track_aeff * 0.0   # this is a track-only selection
    print(f"  Track A_eff max: {track_aeff.max():.3e} eV^-2")

    # ------------------------------------------------------------------
    # Smearing (PSF + energy resolution)
    # ------------------------------------------------------------------
    print("\nLoading smearing matrix ...")
    smearing = _load_smearing("IC86_II")
    print(f"  Rows: {len(smearing):,}")

    print(f"\nBuilding angular response (dec_min >= {_PSF_DEC_MIN} deg) ...")
    us_ang, inv_cdfs_ang = _build_angular_response(
        smearing, energies_gev, N_U_ANG, dec_min_deg=_PSF_DEC_MIN
    )
    print(
        f"  PSF at 1 TeV (median): "
        f"{np.degrees(inv_cdfs_ang[N_E//2, N_U_ANG//2]):.2f} deg"
    )

    print(f"\nBuilding energy resolution (dec_min >= {_PSF_DEC_MIN} deg) ...")
    us_e, inv_cdf_e = _build_energy_resolution(
        smearing, energies_gev, N_U_E, dec_min_deg=_PSF_DEC_MIN
    )
    print(f"  ln(E_reco/E_nu) range: [{inv_cdf_e[0]:.2f}, {inv_cdf_e[-1]:.2f}]")

    # ------------------------------------------------------------------
    # Write HDF5
    # ------------------------------------------------------------------
    print(f"\nWriting {OUT_FILE} ...")
    _write_hdf5(
        output_path  = OUT_FILE,
        energies_eV  = energies_eV,
        zeniths      = zeniths,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        track_aeff   = track_aeff,
        cascade_aeff = cascade_aeff,
        us_ang       = us_ang,
        inv_cdfs_ang = inv_cdfs_ang,
        us_e         = us_e,
        inv_cdf_e    = inv_cdf_e,
    )

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    print("\nSanity check:")
    print(
        f"  Energy grid:  {energies_eV[0]:.2e} – {energies_eV[-1]:.2e} eV  "
        f"({N_E} points)"
    )
    print(
        f"  Zenith grid:  {np.degrees(zeniths[0]):.1f} – "
        f"{np.degrees(zeniths[-1]):.1f} deg  ({N_ZEN} points)"
    )
    print(f"  Track A_eff max: {track_aeff.max():.3e} eV^-2")


if __name__ == "__main__":
    main()
