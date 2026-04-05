"""
Build a SPORE-format HDF5 detector response file from the IceCube HESE 7.5-year
Monte Carlo data release (arXiv:2011.03545).

Derives effective area, angular resolution (PSF), and energy resolution for
tracks (morphology=1) and cascades (morphology=0) directly from the published
MC using ``weightOverFluxOverLivetime`` weighting.

Caveat on the PSF
-----------------
The HESE data release provides zenith angles but not azimuth angles for MC
events. The PSF is estimated from the zenith residual:

    psi_proxy = sqrt(2) * |recoZenith - primaryZenith|

The factor sqrt(2) converts the 1-D zenith projection to an estimate of the
2-D angular separation under the assumption that the reconstruction errors are
azimuthally symmetric and of equal size in zenith and azimuth. For cascades
this approximation is reasonable; for tracks the azimuthal resolution is
significantly better than the zenith resolution, so this will slightly
overestimate the track PSF.

Caveat on energy resolution
---------------------------
"Reconstructed deposited energy" is the HESE observable, not the full neutrino
energy. For CC nu_mu (tracks), the deposited energy is only the hadronic vertex
plus the visible muon segment, so the ratio E_reco/E_true < 1 systematically
for tracks. For cascades the deposited energy is close to the full neutrino
energy for CC nu_e and is E*y for NC events.

Run
---
    # with data release at the default path:
    python scripts/build_hese_detector_response.py

    # or with an explicit path:
    HESE_DATA_DIR=/path/to/HESE-7-year-data-release/resources/data \
        python scripts/build_hese_detector_response.py

Output
------
    resources/hese_7yr_detector_response.h5

Data release reference
----------------------
IceCube Collaboration, Phys.Rev.D 104 (2021) 022002, arXiv:2011.03545
https://icecube.wisc.edu/data-releases/2021/12/hese-7-5-year-data/
"""

import os
import sys
import json
import numpy as np
import h5py
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# Download from:
# https://icecube.wisc.edu/data-releases/2021/12/hese-7-5-year-data/
_DEFAULT_DATA_DIR = os.path.expanduser(
    "~/research/CATHODE/HeseCathode/data/"
    "HESE-7-year-data-release-main/HESE-7-year-data-release/resources/data"
)
DATA_DIR = os.environ.get("HESE_DATA_DIR", _DEFAULT_DATA_DIR)
OUT_FILE = os.path.join(_HERE, "..", "resources", "hese_7yr_detector_response.h5")

# ---------------------------------------------------------------------------
# Unit conversion constants  (mirror of spore/conventions/units.py)
# ---------------------------------------------------------------------------
_EV_PER_GEV     = 1.0e9
_CM_IN_EV_INV   = 8065.54815355        # eV^-1 per cm  (ħc = 197.3 MeV·fm)
_CM2_IN_EV2_INV = _CM_IN_EV_INV ** 2  # eV^-2 per cm²


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
EMIN_GEV = 1e3      # 1 TeV  — HESE threshold is ~60 TeV but MC exists below
EMAX_GEV = 1e7      # 10 PeV
N_E      = 25
N_ZEN    = 18       # full-sky: IceCube/HESE is an isotropic detector
N_U_ANG  = 200      # quantile points in angular response table
N_U_E    = 200      # quantile points in energy resolution table

# Minimum number of MC events required in a bin for the PSF fit to be trusted.
# Bins with fewer events fall back to the nearest neighbour that has enough.
PSF_MIN_EVENTS = 10


# ---------------------------------------------------------------------------
# Load HESE MC
# ---------------------------------------------------------------------------
def _load_mc():
    if not os.path.isdir(DATA_DIR):
        sys.exit(
            f"ERROR: data directory not found at {DATA_DIR}\n"
            "Download the HESE 7.5-year data release from:\n"
            "  https://icecube.wisc.edu/data-releases/2021/12/hese-7-5-year-data/\n"
            "and set HESE_DATA_DIR to the 'resources/data' subdirectory."
        )
    print("Loading HESE_mc_truth.json ...")
    with open(os.path.join(DATA_DIR, "HESE_mc_truth.json")) as fh:
        truth = json.load(fh)

    print("Loading HESE_mc_observable.json ...")
    with open(os.path.join(DATA_DIR, "HESE_mc_observable.json")) as fh:
        obs = json.load(fh)

    primary_energy   = np.array(truth["primaryEnergy"])               # GeV
    primary_zenith   = np.array(truth["primaryZenith"])               # rad
    weights          = np.array(truth["weightOverFluxOverLivetime"])  # GeV sr cm²
    interaction_type = np.array(truth["interactionType"])             # 0=muon,1=CC,2=NC,3=GR

    reco_energy      = np.array(obs["recoDepositedEnergy"])           # GeV
    reco_zenith      = np.array(obs["recoZenith"])                    # rad
    reco_morphology  = np.array(obs["recoMorphology"])                # 0=cascade,1=track,2=doublebang

    # Discard atmospheric muons (interactionType == 0)
    is_nu = interaction_type != 0
    print(f"  Neutrino events: {is_nu.sum():,} / {len(is_nu):,} total")
    return (
        primary_energy[is_nu],
        primary_zenith[is_nu],
        weights[is_nu],
        reco_energy[is_nu],
        reco_zenith[is_nu],
        reco_morphology[is_nu],
    )


# ---------------------------------------------------------------------------
# Effective area
# ---------------------------------------------------------------------------
def build_aeff(
    primary_energy_gev: np.ndarray,
    primary_zenith_rad: np.ndarray,
    weights_gev_sr_cm2: np.ndarray,
    energies_gev: np.ndarray,
    zeniths_rad: np.ndarray,
) -> np.ndarray:
    """
    Compute A_eff(E, zenith) in eV^-2 (SPORE natural units).

    A_eff(E, cos_zen) = Σ_i w_i / (ΔE_GeV × 2π × |Δcos_zen|)

    The sum is over MC events in the (E, cos_zen) bin, using true (primary)
    energy and true zenith.  The result is then converted from cm² to eV^-2.

    Parameters
    ----------
    primary_energy_gev, primary_zenith_rad, weights_gev_sr_cm2:
        Per-event arrays for a single morphology.
    energies_gev : shape (N_E,), centre values of the output energy grid.
    zeniths_rad  : shape (N_ZEN,), output zenith grid.

    Returns
    -------
    aeff : shape (N_E, N_ZEN) in eV^-2
    """
    cos_zen_true = np.cos(primary_zenith_rad)
    log10_e_true = np.log10(primary_energy_gev)

    # Bin edges: midpoints between adjacent centres, extended at the boundaries.
    def _edges(centres):
        mids = 0.5 * (centres[:-1] + centres[1:])
        lo   = 2 * centres[0]  - centres[1]
        hi   = 2 * centres[-1] - centres[-2]
        return np.concatenate([[lo], mids, [hi]])

    log10_e_edges  = _edges(np.log10(energies_gev))
    cos_zen_vals   = np.cos(zeniths_rad)          # decreasing: 1 → -1
    # Sort cos_zen edges in increasing order for np.histogram2d
    cos_zen_edges  = _edges(cos_zen_vals[::-1])   # now increasing

    # 2D histogram of summed weights using true kinematics
    H, _, _ = np.histogram2d(
        log10_e_true,
        cos_zen_true,
        bins=[log10_e_edges, cos_zen_edges],
        weights=weights_gev_sr_cm2,
    )
    # H has shape (N_E, N_ZEN) with cos_zen in increasing order — reverse to
    # match zeniths_rad ordering (zenith 0 → cos_zen=1 is first)
    H = H[:, ::-1]

    # Bin widths
    delta_log10_e  = np.diff(log10_e_edges)                     # (N_E,)
    delta_e_gev    = energies_gev * np.log(10) * delta_log10_e  # E × d(lnE)

    delta_cos_zen_sorted = np.diff(cos_zen_edges)                # (N_ZEN,), positive
    delta_cos_zen         = delta_cos_zen_sorted[::-1]           # reverse to match order

    # A_eff in cm²
    aeff_cm2 = H / (
        delta_e_gev[:, np.newaxis] * 2 * np.pi * delta_cos_zen[np.newaxis, :]
    )
    aeff_cm2 = np.where(np.isfinite(aeff_cm2) & (aeff_cm2 > 0), aeff_cm2, 0.0)

    return aeff_cm2 * _CM2_IN_EV2_INV


# ---------------------------------------------------------------------------
# Angular response (PSF)
# ---------------------------------------------------------------------------
def build_angular_response(
    primary_energy_gev: np.ndarray,
    primary_zenith_rad: np.ndarray,
    reco_zenith_rad: np.ndarray,
    energies_gev: np.ndarray,
    n_u: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build energy-dependent angular response inverse-CDF table.

    For each energy bin, the PSF is estimated from the zenith residual:
        psi_proxy = sqrt(2) * |recoZenith - primaryZenith|

    A Rayleigh distribution is fit to the proxy angles by matching the
    empirical median, giving sigma_Rayleigh.  Bins with fewer than
    PSF_MIN_EVENTS events fall back to the nearest bin with enough events.

    Returns
    -------
    us       : shape (n_u,)
    inv_cdfs : shape (N_E, n_u), angles in radians
    """
    us = np.linspace(0.0, 1.0 - 1e-9, n_u)
    N_E = len(energies_gev)
    inv_cdfs = np.zeros((N_E, n_u))

    log10_e = np.log10(primary_energy_gev)
    psi_proxy = np.sqrt(2.0) * np.abs(reco_zenith_rad - primary_zenith_rad)

    log10_e_edges = np.concatenate([
        [2 * np.log10(energies_gev[0])  - np.log10(energies_gev[1])],
        0.5 * (np.log10(energies_gev[:-1]) + np.log10(energies_gev[1:])),
        [2 * np.log10(energies_gev[-1]) - np.log10(energies_gev[-2])],
    ])

    # Per-energy-bin Rayleigh sigma from empirical median psi
    # Rayleigh median = sigma * sqrt(ln 4), so sigma = median / sqrt(ln 4)
    _RAYLEIGH_MEDIAN_FACTOR = np.sqrt(np.log(4.0))

    sigmas = np.full(N_E, np.nan)
    for i_e in range(N_E):
        lo, hi = log10_e_edges[i_e], log10_e_edges[i_e + 1]
        in_bin = (log10_e >= lo) & (log10_e < hi)
        if in_bin.sum() >= PSF_MIN_EVENTS:
            median_psi = np.median(psi_proxy[in_bin])
            sigmas[i_e] = median_psi / _RAYLEIGH_MEDIAN_FACTOR

    # Fill missing bins by interpolation / nearest neighbour
    good = np.where(np.isfinite(sigmas))[0]
    if len(good) == 0:
        raise RuntimeError("No energy bins have enough MC events to estimate the PSF.")
    all_idx = np.arange(N_E)
    sigmas = np.interp(all_idx, good, sigmas[good])

    # Build Rayleigh inverse CDF for each energy bin
    for i_e, sigma in enumerate(sigmas):
        inv_cdfs[i_e] = sigma * np.sqrt(-2.0 * np.log(np.clip(1.0 - us, 1e-12, 1.0)))

    return us, inv_cdfs


# ---------------------------------------------------------------------------
# Energy resolution
# ---------------------------------------------------------------------------
def build_energy_resolution(
    primary_energy_gev: np.ndarray,
    reco_energy_gev: np.ndarray,
    n_u: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build energy-resolution inverse-CDF table (energy-independent).

    SPORE stores ln(E_reco / E_true).  For HESE, E_reco is the reconstructed
    deposited energy and E_true is the primary neutrino energy.  For tracks
    the ratio is systematically < 1 because the invisible muon carries away
    energy; for cascades it is close to 1 for CC nu_e and fractional for NC.

    Returns
    -------
    us      : shape (n_u,)
    inv_cdf : shape (n_u,)
    """
    # Exclude any pathological values
    valid = (reco_energy_gev > 0) & (primary_energy_gev > 0)
    ln_ratio = np.log(reco_energy_gev[valid] / primary_energy_gev[valid])

    # Sort and build empirical CDF
    ln_ratio_sorted = np.sort(ln_ratio)
    n = len(ln_ratio_sorted)
    cdf = (np.arange(n) + 0.5) / n  # mid-point rule

    # Interpolate onto uniform quantile grid, clipping to data range
    us = np.linspace(cdf[0], cdf[-1], n_u)
    inv_cdf = PchipInterpolator(cdf, ln_ratio_sorted)(us)

    # Extend to [0, 1] by constant extrapolation at the tails
    us_full     = np.linspace(0.0, 1.0, n_u)
    inv_cdf_full = np.interp(us_full, us, inv_cdf)

    return us_full, inv_cdf_full


# ---------------------------------------------------------------------------
# HDF5 writer  (same format as build_gfu_detector_response.py)
# ---------------------------------------------------------------------------
def write_hdf5(
    output_path: str,
    energies_eV: np.ndarray,
    zeniths: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    track_aeff: np.ndarray,
    cascade_aeff: np.ndarray,
    track_us_ang: np.ndarray,
    track_inv_cdfs_ang: np.ndarray,
    cascade_us_ang: np.ndarray,
    cascade_inv_cdfs_ang: np.ndarray,
    track_us_e: np.ndarray,
    track_inv_cdf_e: np.ndarray,
    cascade_us_e: np.ndarray,
    cascade_inv_cdf_e: np.ndarray,
):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, "w") as f:
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

        for name, us, inv_cdfs in [
            ("track_angular_response",   track_us_ang,   track_inv_cdfs_ang),
            ("cascade_angular_response", cascade_us_ang, cascade_inv_cdfs_ang),
        ]:
            g = f.create_group(name)
            g.create_dataset("energies",  data=energies_eV)
            g.create_dataset("us",        data=us)
            g.create_dataset("inv_cdfs",  data=inv_cdfs)

        for name, us, inv_cdf in [
            ("track_energy_resolution",   track_us_e,   track_inv_cdf_e),
            ("cascade_energy_resolution", cascade_us_e, cascade_inv_cdf_e),
        ]:
            g = f.create_group(name)
            g.create_dataset("us",      data=us)
            g.create_dataset("inv_cdf", data=inv_cdf)

        f.attrs["source"]    = "IceCube HESE 7.5-year data release (arXiv:2011.03545)"
        f.attrs["emin_gev"]  = EMIN_GEV
        f.attrs["emax_gev"]  = EMAX_GEV
        f.attrs["psf_note"]  = (
            "PSF estimated from psi_proxy = sqrt(2)*|dzenith|; "
            "azimuth info not available in data release."
        )

    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    (
        primary_energy, primary_zenith, weights,
        reco_energy, reco_zenith, reco_morphology,
    ) = _load_mc()

    is_track   = reco_morphology == 1
    is_cascade = reco_morphology == 0
    print(f"  Track events:   {is_track.sum():,}")
    print(f"  Cascade events: {is_cascade.sum():,}")

    # Output grids
    energies_gev = np.logspace(np.log10(EMIN_GEV), np.log10(EMAX_GEV), N_E)
    energies_eV  = energies_gev * _EV_PER_GEV
    zeniths      = np.linspace(0.0, np.pi, N_ZEN)

    log10_emin = np.log10(energies_eV[0])
    log10_emax = np.log10(energies_eV[-1])
    lower_bounds = np.array([[log10_emin]])
    upper_bounds = np.array([[log10_emax]])

    # ------------------------------------------------------------------
    # Effective area
    # ------------------------------------------------------------------
    print("Building track effective area ...")
    track_aeff = build_aeff(
        primary_energy[is_track], primary_zenith[is_track],
        weights[is_track], energies_gev, zeniths,
    )
    print(f"  Track A_eff max: {track_aeff.max():.3e} eV^-2")

    print("Building cascade effective area ...")
    cascade_aeff = build_aeff(
        primary_energy[is_cascade], primary_zenith[is_cascade],
        weights[is_cascade], energies_gev, zeniths,
    )
    print(f"  Cascade A_eff max: {cascade_aeff.max():.3e} eV^-2")

    # ------------------------------------------------------------------
    # Angular response
    # ------------------------------------------------------------------
    print("Building track PSF ...")
    track_us_ang, track_inv_cdfs_ang = build_angular_response(
        primary_energy[is_track], primary_zenith[is_track],
        reco_zenith[is_track], energies_gev, N_U_ANG,
    )
    track_median_psi_deg = np.degrees(track_inv_cdfs_ang[:, N_U_ANG // 2])
    print(f"  Track PSF median: {track_median_psi_deg.min():.1f}–{track_median_psi_deg.max():.1f} deg")

    print("Building cascade PSF ...")
    cascade_us_ang, cascade_inv_cdfs_ang = build_angular_response(
        primary_energy[is_cascade], primary_zenith[is_cascade],
        reco_zenith[is_cascade], energies_gev, N_U_ANG,
    )
    cascade_median_psi_deg = np.degrees(cascade_inv_cdfs_ang[:, N_U_ANG // 2])
    print(f"  Cascade PSF median: {cascade_median_psi_deg.min():.1f}–{cascade_median_psi_deg.max():.1f} deg")

    # ------------------------------------------------------------------
    # Energy resolution
    # ------------------------------------------------------------------
    print("Building track energy resolution ...")
    track_us_e, track_inv_cdf_e = build_energy_resolution(
        primary_energy[is_track], reco_energy[is_track], N_U_E,
    )
    print(f"  Track ln(E_reco/E_true): [{track_inv_cdf_e[0]:.2f}, {track_inv_cdf_e[-1]:.2f}]")

    print("Building cascade energy resolution ...")
    cascade_us_e, cascade_inv_cdf_e = build_energy_resolution(
        primary_energy[is_cascade], reco_energy[is_cascade], N_U_E,
    )
    print(f"  Cascade ln(E_reco/E_true): [{cascade_inv_cdf_e[0]:.2f}, {cascade_inv_cdf_e[-1]:.2f}]")

    # ------------------------------------------------------------------
    # Write HDF5
    # ------------------------------------------------------------------
    write_hdf5(
        output_path=OUT_FILE,
        energies_eV=energies_eV,
        zeniths=zeniths,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        track_aeff=track_aeff,
        cascade_aeff=cascade_aeff,
        track_us_ang=track_us_ang,
        track_inv_cdfs_ang=track_inv_cdfs_ang,
        cascade_us_ang=cascade_us_ang,
        cascade_inv_cdfs_ang=cascade_inv_cdfs_ang,
        track_us_e=track_us_e,
        track_inv_cdf_e=track_inv_cdf_e,
        cascade_us_e=cascade_us_e,
        cascade_inv_cdf_e=cascade_inv_cdf_e,
    )

    # ------------------------------------------------------------------
    # Quick sanity check: events expected under a test flux
    # ------------------------------------------------------------------
    # E^-2 flux, Phi_0 = 1e-18 GeV^-1 cm^-2 s^-1 sr^-1 per flavour
    T_sec   = 7.5 * 365.25 * 24 * 3600
    phi0    = 1e-18
    e_pivot = 1e5  # GeV

    def _expected(aeff_eV2, morph):
        aeff_cm2  = aeff_eV2 / _CM2_IN_EV2_INV
        total = 0.0
        for i_e, e_gev in enumerate(energies_gev[:-1]):
            de_gev = energies_gev[i_e + 1] - e_gev
            # average A_eff over zenith (isotropic)
            mean_aeff = aeff_cm2[i_e].mean()
            phi = phi0 * (e_gev / e_pivot) ** (-2.0)
            total += mean_aeff * phi * 4 * np.pi * de_gev * T_sec
        return total

    n_track   = _expected(track_aeff,   "track")
    n_cascade = _expected(cascade_aeff, "cascade")
    print(f"\nSanity check (E^-2 flux, 7.5 yr):")
    print(f"  Expected track events:   {n_track:.1f}")
    print(f"  Expected cascade events: {n_cascade:.1f}")
    print("  (HESE observed ~18 tracks and ~84 cascades/doublebang; scale depends on flux normalisation)")


if __name__ == "__main__":
    main()
