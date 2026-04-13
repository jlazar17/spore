"""
Build a spore-format HDF5 detector response file from digitized figures.

This script converts CSV files digitized from published papers into the HDF5
format expected by spore's Detector class.  It is intended for assembling
an IceCube GFU IRF from figures in arXiv:1610.01814.

Input CSV conventions
---------------------
Effective area (one file per declination band):
    columns: energy_GeV, aeff_m2
    Pass as: --effa /path/to/file.csv:dec_deg  (e.g. --effa aeff_90.csv:-90)

Angular resolution (PSF):
    columns: energy_GeV, median_angle_deg
    The full PSF is modelled as a Rayleigh distribution.
    Pass as: --psf /path/to/file.csv

Energy resolution:
    columns: log10_eratio, cdf   (where eratio = E_reco / E_true)
    Pass as: --energy-res /path/to/file.csv

Output
------
An HDF5 file with the six groups spore expects:
    track_effective_area / cascade_effective_area
    track_angular_response / cascade_angular_response
    track_energy_resolution / cascade_energy_resolution

Geometry assumption
-------------------
This script assumes the detector is at the South Pole (IceCube geometry),
where zenith = pi/2 + declination_rad.  For other detector locations add a
--latitude flag and generalise the conversion.

Run example
-----------
    python scripts/build_gfu_detector_response.py \
        --effa data/aeff_north.csv:90  \
        --effa data/aeff_horizon.csv:0 \
        --effa data/aeff_south.csv:-30 \
        --psf  data/psf_median.csv     \
        --energy-res data/eres.csv     \
        --output resources/gfu_detector_response.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Natural-unit conversion constants (mirror of spore/conventions/units.py)
# ---------------------------------------------------------------------------
_CM_PER_M = 100.0
_EV_PER_GEV = 1.0e9
_METER_IN_EV_INV = 806554.815355          # eV^-1 per metre
_CM_IN_EV_INV = 1.0e-2 * _METER_IN_EV_INV


def _m2_to_eV2(area_m2: np.ndarray) -> np.ndarray:
    """Convert effective area from m^2 to natural units (eV^-2)."""
    area_cm2 = area_m2 * _CM_PER_M**2
    return area_cm2 * _CM_IN_EV_INV**2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert digitized IRF CSVs to spore HDF5 detector response.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--effa",
        metavar="FILE:DEC_DEG",
        action="append",
        default=[],
        help=(
            "CSV file for one effective-area declination band, colon-separated "
            "from the declination in degrees.  Pass once per band."
        ),
    )
    p.add_argument(
        "--psf",
        metavar="FILE",
        help="CSV file: energy_GeV, median_angle_deg (Rayleigh model will be fit).",
    )
    p.add_argument(
        "--energy-res",
        metavar="FILE",
        dest="energy_res",
        help="CSV file: log10_eratio, cdf (where eratio = E_reco / E_true).",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        required=True,
        help="Output HDF5 path.",
    )

    # Grid / model parameters
    p.add_argument(
        "--emin-gev", type=float, default=1e2,
        help="Minimum neutrino energy for the output grid [GeV]. Default: 1e2.",
    )
    p.add_argument(
        "--emax-gev", type=float, default=1e7,
        help="Maximum neutrino energy for the output grid [GeV]. Default: 1e7.",
    )
    p.add_argument(
        "--n-energy", type=int, default=50,
        help="Number of energy grid points (log-spaced). Default: 50.",
    )
    p.add_argument(
        "--n-zenith", type=int, default=20,
        help="Number of zenith grid points. Default: 20.",
    )
    p.add_argument(
        "--n-u-ang", type=int, default=200,
        help="Number of quantile points in angular response table. Default: 200.",
    )
    p.add_argument(
        "--n-u-e", type=int, default=200,
        help="Number of quantile points in energy resolution table. Default: 200.",
    )
    p.add_argument(
        "--psf-floor-deg", type=float, default=0.2,
        help="Systematic PSF floor [degrees]. Default: 0.2.",
    )
    p.add_argument(
        "--cascade-aeff-scale", type=float, default=0.2,
        help=(
            "Scale factor applied to track effective area to approximate the "
            "cascade channel. Default: 0.2."
        ),
    )
    p.add_argument(
        "--cascade-psf-deg", type=float, default=10.0,
        help="Constant cascade PSF (median angle) in degrees. Default: 10.0.",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Effective area helpers
# ---------------------------------------------------------------------------

def _load_effa_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (energy_gev, aeff_eV2) from a CSV with columns energy_GeV, aeff_m2."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    energy_gev = data[:, 0]
    aeff_eV2 = _m2_to_eV2(data[:, 1])
    idx = np.argsort(energy_gev)
    return energy_gev[idx], aeff_eV2[idx]


def _build_effa_grid(
    effa_specs: list[tuple[float, np.ndarray, np.ndarray]],
    energies_gev: np.ndarray,
    zeniths: np.ndarray,
) -> np.ndarray:
    """
    Interpolate per-declination effective-area curves onto (n_e, n_zen) grid.

    Parameters
    ----------
    effa_specs : list of (dec_rad, energy_gev_array, aeff_eV2_array)
    energies_gev : 1-D array, shape (n_e,), output energy grid [GeV]
    zeniths : 1-D array, shape (n_zen,), output zenith grid [radians]

    Returns
    -------
    tabulated_values : ndarray, shape (n_e, n_zen)
    """
    n_e = len(energies_gev)
    n_zen = len(zeniths)
    # Convert zenith to declination for South Pole geometry
    decs = zeniths - np.pi / 2.0    # dec = zen - pi/2

    # Build per-declination interpolated aeff curves (in log space where > 0)
    dec_nodes = np.array([dec for dec, _, _ in effa_specs])
    aeff_at_dec = np.zeros((len(effa_specs), n_e))

    for i, (dec, e_node, a_node) in enumerate(effa_specs):
        # Log-linear interpolation; clamp negatives to 0
        log_a = np.where(a_node > 0, np.log(a_node), np.nan)
        interp = PchipInterpolator(np.log(e_node), log_a, extrapolate=False)
        log_a_grid = interp(np.log(energies_gev))
        aeff_at_dec[i] = np.where(np.isfinite(log_a_grid), np.exp(log_a_grid), 0.0)

    # Interpolate over declination for each energy bin
    tabulated = np.zeros((n_e, n_zen))
    for ie in range(n_e):
        vals = aeff_at_dec[:, ie]
        if np.all(vals == 0):
            continue
        # Linear interpolation in sin(dec) space
        sin_nodes = np.sin(dec_nodes)
        sin_decs = np.sin(decs)
        interp_dec = PchipInterpolator(sin_nodes, vals, extrapolate=False)
        result = interp_dec(sin_decs)
        tabulated[ie] = np.where(np.isfinite(result) & (result > 0), result, 0.0)

    return tabulated


def _build_poly_bounds(energies_gev: np.ndarray, zeniths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build constant polynomial cut bounds covering the full grid.

    The poly_bounds format in spore is:
        lower_bounds : shape (n_cuts, n_coeff)  — polynomial in cos(zenith)
        upper_bounds : shape (n_cuts, n_coeff)  — polynomial in cos(zenith)
    Each row is a polynomial evaluated at cos(zenith) that must satisfy:
        log10(E/GeV) >= lower_bound(cos(zen))   and   log10(E/GeV) <= upper_bound(cos(zen))

    For the GFU IRF we use a single constant cut (zeroth-order polynomial):
        lower: log10(emin_gev)
        upper: log10(emax_gev)
    """
    log10_emin = np.log10(energies_gev[0])
    log10_emax = np.log10(energies_gev[-1])
    # Shape: (1, 1) — one cut, degree-0 polynomial (constant)
    lower_bounds = np.array([[log10_emin]])
    upper_bounds = np.array([[log10_emax]])
    return lower_bounds, upper_bounds


# ---------------------------------------------------------------------------
# Angular response (PSF) helpers
# ---------------------------------------------------------------------------

def _rayleigh_inv_cdf(u: np.ndarray, sigma_rad: float) -> np.ndarray:
    """Rayleigh inverse CDF: theta(u) = sigma * sqrt(-2 * ln(1 - u))."""
    return sigma_rad * np.sqrt(-2.0 * np.log(np.clip(1.0 - u, 1e-12, 1.0)))


def _build_angular_response(
    psf_path: str | None,
    energies_gev: np.ndarray,
    n_u: int,
    psf_floor_deg: float,
    constant_median_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the angular response inv-CDF table.

    Returns
    -------
    us : ndarray, shape (n_u,)
    inv_cdfs : ndarray, shape (n_e, n_u)   — angles in radians
    """
    us = np.linspace(0.0, 1.0 - 1e-9, n_u)
    n_e = len(energies_gev)

    if constant_median_deg is not None:
        # Energy-independent PSF (used for cascades)
        median_rad = np.radians(constant_median_deg)
        sigma = median_rad / np.sqrt(2.0 * np.log(2.0))
        row = _rayleigh_inv_cdf(us, sigma)
        inv_cdfs = np.tile(row, (n_e, 1))
        return us, inv_cdfs

    if psf_path is None:
        raise ValueError("Either psf_path or constant_median_deg must be provided.")

    data = np.loadtxt(psf_path, delimiter=",", comments="#")
    e_node_gev = data[:, 0]
    median_deg = data[:, 1]

    idx = np.argsort(e_node_gev)
    e_node_gev = e_node_gev[idx]
    median_deg = median_deg[idx]

    # Interpolate median angle onto output energy grid
    interp = PchipInterpolator(np.log(e_node_gev), median_deg, extrapolate=True)
    median_grid_deg = interp(np.log(energies_gev))

    # Apply floor
    median_grid_deg = np.maximum(median_grid_deg, psf_floor_deg)

    # Build inv-CDF row for each energy
    inv_cdfs = np.zeros((n_e, n_u))
    for ie, med_deg in enumerate(median_grid_deg):
        sigma = np.radians(med_deg) / np.sqrt(2.0 * np.log(2.0))
        inv_cdfs[ie] = _rayleigh_inv_cdf(us, sigma)

    return us, inv_cdfs


# ---------------------------------------------------------------------------
# Energy resolution helpers
# ---------------------------------------------------------------------------

def _build_energy_resolution(
    eres_path: str | None,
    n_u: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the energy resolution inv-CDF table.

    The CSV is expected to have columns log10_eratio and cdf, where
    eratio = E_reco / E_true.  spore stores ln(eratio) = log10_eratio * ln(10).

    Returns
    -------
    us : ndarray, shape (n_u,)
    inv_cdf : ndarray, shape (n_u,)   — ln(E_reco / E_true)
    """
    if eres_path is None:
        # Gaussian placeholder: sigma=0.3 in ln(E)
        us = np.linspace(0.0, 1.0 - 1e-9, n_u)
        from scipy.special import erfinv
        sigma = 0.3
        inv_cdf = sigma * np.sqrt(2.0) * erfinv(2.0 * us - 1.0)
        return us, inv_cdf

    data = np.loadtxt(eres_path, delimiter=",", comments="#")
    log10_eratio = data[:, 0]
    cdf = data[:, 1]

    idx = np.argsort(cdf)
    cdf = cdf[idx]
    log10_eratio = log10_eratio[idx]

    # Convert log10 -> ln
    ln_eratio = log10_eratio * np.log(10.0)

    # Remove duplicate CDF values to make interpolation valid
    _, unique_idx = np.unique(cdf, return_index=True)
    cdf = cdf[unique_idx]
    ln_eratio = ln_eratio[unique_idx]

    us = np.linspace(cdf[0], cdf[-1], n_u)
    interp = PchipInterpolator(cdf, ln_eratio, extrapolate=False)
    inv_cdf = interp(us)

    return us, inv_cdf


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def _write_hdf5(
    output_path: str,
    energies_gev: np.ndarray,
    zeniths: np.ndarray,
    track_aeff: np.ndarray,
    cascade_aeff: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    track_us_ang: np.ndarray,
    track_inv_cdfs_ang: np.ndarray,
    cascade_us_ang: np.ndarray,
    cascade_inv_cdfs_ang: np.ndarray,
    us_e: np.ndarray,
    inv_cdf_e: np.ndarray,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for name, tab in [
            ("track_effective_area", track_aeff),
            ("cascade_effective_area", cascade_aeff),
        ]:
            g = f.create_group(name)
            g.create_dataset("energies", data=energies_gev)
            g.create_dataset("zeniths", data=zeniths)
            g.create_dataset("tabulated_values", data=tab)
            g.create_dataset("lower_bounds", data=lower_bounds)
            g.create_dataset("upper_bounds", data=upper_bounds)

        for name, us, inv_cdfs in [
            ("track_angular_response", track_us_ang, track_inv_cdfs_ang),
            ("cascade_angular_response", cascade_us_ang, cascade_inv_cdfs_ang),
        ]:
            g = f.create_group(name)
            g.create_dataset("energies", data=energies_gev)
            g.create_dataset("us", data=us)
            g.create_dataset("inv_cdfs", data=inv_cdfs)

        for name in ["track_energy_resolution", "cascade_energy_resolution"]:
            g = f.create_group(name)
            g.create_dataset("us", data=us_e)
            g.create_dataset("inv_cdf", data=inv_cdf_e)

    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    if not args.effa:
        print("Error: at least one --effa FILE:DEC_DEG argument is required.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Parse effective area inputs
    # ------------------------------------------------------------------
    effa_specs = []
    for spec in args.effa:
        if ":" not in spec:
            print(
                f"Error: --effa argument '{spec}' must be in the form FILE:DEC_DEG",
                file=sys.stderr,
            )
            sys.exit(1)
        # Split on the last colon to allow colons in Windows paths
        colon_idx = spec.rfind(":")
        path = spec[:colon_idx]
        dec_deg = float(spec[colon_idx + 1:])
        dec_rad = np.radians(dec_deg)
        e_eV, a_eV2 = _load_effa_csv(path)
        effa_specs.append((dec_rad, e_eV, a_eV2))
        print(f"  Loaded effective area: {path} (dec={dec_deg:.1f} deg, {len(e_eV)} points)")

    # Sort by declination
    effa_specs.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Output grids
    # ------------------------------------------------------------------
    energies_gev = np.logspace(np.log10(args.emin_gev), np.log10(args.emax_gev), args.n_energy)

    # Zeniths: 0 → π (needed by spore; cos(zeniths) is descending, which
    # scipy's RegularGridInterpolator handles via the strictly-decreasing path)
    zeniths = np.linspace(0.0, np.pi, args.n_zenith)

    # ------------------------------------------------------------------
    # Effective area
    # ------------------------------------------------------------------
    print("Building effective area grid...")
    track_aeff = _build_effa_grid(effa_specs, energies_gev, zeniths)
    cascade_aeff = track_aeff * args.cascade_aeff_scale
    lower_bounds, upper_bounds = _build_poly_bounds(energies_gev, zeniths)

    # ------------------------------------------------------------------
    # Angular response
    # ------------------------------------------------------------------
    print("Building angular response tables...")
    track_us_ang, track_inv_cdfs_ang = _build_angular_response(
        psf_path=args.psf,
        energies_gev=energies_gev,
        n_u=args.n_u_ang,
        psf_floor_deg=args.psf_floor_deg,
    )
    cascade_us_ang, cascade_inv_cdfs_ang = _build_angular_response(
        psf_path=None,
        energies_gev=energies_gev,
        n_u=args.n_u_ang,
        psf_floor_deg=args.psf_floor_deg,
        constant_median_deg=args.cascade_psf_deg,
    )

    # ------------------------------------------------------------------
    # Energy resolution
    # ------------------------------------------------------------------
    print("Building energy resolution table...")
    us_e, inv_cdf_e = _build_energy_resolution(args.energy_res, args.n_u_e)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    _write_hdf5(
        output_path=args.output,
        energies_gev=energies_gev,
        zeniths=zeniths,
        track_aeff=track_aeff,
        cascade_aeff=cascade_aeff,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        track_us_ang=track_us_ang,
        track_inv_cdfs_ang=track_inv_cdfs_ang,
        cascade_us_ang=cascade_us_ang,
        cascade_inv_cdfs_ang=cascade_inv_cdfs_ang,
        us_e=us_e,
        inv_cdf_e=inv_cdf_e,
    )

    # ------------------------------------------------------------------
    # Quick sanity check
    # ------------------------------------------------------------------
    print("\nSanity check:")
    print(f"  Energy grid:  {energies_gev[0]:.2e} – {energies_gev[-1]:.2e} GeV  ({len(energies_gev)} points)")
    print(f"  Zenith grid:  {np.degrees(zeniths[0]):.1f} – {np.degrees(zeniths[-1]):.1f} deg  ({len(zeniths)} points)")
    print(f"  Track A_eff max: {track_aeff.max():.3e} eV^-2")
    print(f"  Track PSF at lowest E: {np.degrees(track_inv_cdfs_ang[0, int(0.5 * args.n_u_ang)]):.2f} deg (median)")
    print(f"  Energy res range: ln(E_reco/E_true) in [{inv_cdf_e[0]:.2f}, {inv_cdf_e[-1]:.2f}]")


if __name__ == "__main__":
    main()
