"""
Build a SPORE-format HDF5 detector response file from the IceCube HESE 7.5-year
Monte Carlo data release (arXiv:2011.03545).

Output groups
-------------
Four morphology groups are written, each with effective_area, angular_response,
and energy_resolution subgroups:

    astro_track    — track channel, geometric A_eff (no veto)
    atmo_track     — track channel, veto-corrected A_eff
    astro_cascade  — cascade channel, geometric A_eff (no veto)
    atmo_cascade   — cascade channel, veto-corrected A_eff

The astrophysical groups use the plain geometric effective area and should be
convolved with an astrophysical flux (e.g. the HESE best-fit power law).

The atmospheric groups fold in the per-event conventionalSelfVetoCorrection from
the data release.  They should be convolved with the *unvetoed* conventional
atmospheric flux (e.g. MCEq H4a+SIBYLL23d or Honda+Gaisser).  The approximation
is that the veto correction is evaluated at the reference atmospheric
normalisation (Phi_conv ≈ 1); for Phi_conv = 1.01 (HESE best fit) this is
essentially exact.

PSF and energy resolution are detector properties and are therefore identical
between the astrophysical and atmospheric groups of the same morphology.

Caveat on the PSF
-----------------
The HESE data release provides zenith angles but not azimuth angles for MC
events. The PSF is estimated from the zenith residual:

    psi_proxy = sqrt(2) * |recoZenith - primaryZenith|

The factor sqrt(2) converts the 1-D zenith projection to an estimate of the
2-D angular separation under the assumption that the reconstruction errors are
azimuthally symmetric and of equal size in zenith and azimuth.

Caveat on energy resolution
---------------------------
"Reconstructed deposited energy" is the HESE observable, not the full neutrino
energy. For CC nu_mu (tracks), the deposited energy is only the hadronic vertex
plus the visible muon segment, so the ratio E_reco/E_true < 1 systematically.

Run
---
    python scripts/build_hese_detector_response.py

    HESE_DATA_DIR=/path/to/HESE-7-year-data-release/resources/data \\
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

# primaryType → SPORE species index  (same as in build_hese_atmospheric_flux.py)
_PTYPE_TO_IDX = {12: 0, -12: 1, 14: 2, -14: 3, 16: 4, -16: 5}

# Species that physically dominate each morphology.  Rare species in a
# morphology (e.g. NuE NC events reconstructed as track) have sparse MC
# statistics and produce noisy A_eff estimates that inflate the rate integral.
# Only the dominant species are included in the per-species A_eff to avoid
# spurious contributions.
#   Track   : NuMu + NuMuBar CC (>99% of the HESE track rate)
#   Cascade : NuE + NuEbar CC + NC(NuMu+NuMuBar) + NuTau CC
#             NuTau contributes via CC decay topology; exclude since Honda has
#             no tau flux and the astro contribution is small (~1-2%).
_MORPH_PTYPES = {
    "track":   {14: 2, -14: 3},                   # NuMu, NuMuBar
    "cascade": {12: 0, -12: 1, 14: 2, -14: 3},    # NuE, NuEbar, NuMu, NuMuBar
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

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
_CM_IN_EV_INV   = 8065.54815355
_CM2_IN_EV2_INV = _CM_IN_EV_INV ** 2

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
EMIN_GEV = 1e3      # 1 TeV
EMAX_GEV = 1e7      # 10 PeV
N_E      = 25
N_ZEN    = 18
N_U_ANG  = 200
N_U_E    = 200

PSF_MIN_EVENTS = 10


# ---------------------------------------------------------------------------
# Load HESE MC (all three JSON files merged)
# ---------------------------------------------------------------------------
def _load_mc():
    if not os.path.isdir(DATA_DIR):
        sys.exit(
            f"ERROR: data directory not found at {DATA_DIR}\n"
            "Download the HESE 7.5-year data release from:\n"
            "  https://icecube.wisc.edu/data-releases/2021/12/hese-7-5-year-data/\n"
            "and set HESE_DATA_DIR to the 'resources/data' subdirectory."
        )

    merged = {}
    for fname in ["HESE_mc_truth.json", "HESE_mc_observable.json", "HESE_mc_flux.json"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"Loading {fname} ...")
        merged.update(json.load(open(path)))

    primary_energy    = np.array(merged["primaryEnergy"])
    primary_zenith    = np.array(merged["primaryZenith"])
    primary_type      = np.array(merged["primaryType"])
    weights           = np.array(merged["weightOverFluxOverLivetime"])
    interaction_type  = np.array(merged["interactionType"])
    reco_energy       = np.array(merged["recoDepositedEnergy"])
    reco_zenith       = np.array(merged["recoZenith"])
    reco_morphology   = np.array(merged["recoMorphology"])
    conv_veto         = np.array(merged["conventionalSelfVetoCorrection"])

    # Discard atmospheric muons (interactionType == 0)
    is_nu = interaction_type != 0
    print(f"  Neutrino events: {is_nu.sum():,} / {len(is_nu):,} total")

    return (
        primary_energy[is_nu],
        primary_zenith[is_nu],
        primary_type[is_nu],
        weights[is_nu],
        reco_energy[is_nu],
        reco_zenith[is_nu],
        reco_morphology[is_nu],
        conv_veto[is_nu],
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
    veto_weights: np.ndarray = None,
) -> np.ndarray:
    """
    Compute A_eff(E, zenith) in eV^-2.

    Parameters
    ----------
    veto_weights : array or None
        Per-event multiplicative factor applied before binning.  Pass
        ``conventionalSelfVetoCorrection`` to build the veto-corrected
        atmospheric effective area.  Pass ``None`` (default) for the plain
        geometric effective area used by the astrophysical groups.
    """
    w = weights_gev_sr_cm2 if veto_weights is None else weights_gev_sr_cm2 * veto_weights

    cos_zen_true = np.cos(primary_zenith_rad)
    log10_e_true = np.log10(primary_energy_gev)

    def _edges(centres):
        mids = 0.5 * (centres[:-1] + centres[1:])
        lo   = 2 * centres[0]  - centres[1]
        hi   = 2 * centres[-1] - centres[-2]
        return np.concatenate([[lo], mids, [hi]])

    log10_e_edges = _edges(np.log10(energies_gev))
    cos_zen_vals  = np.cos(zeniths_rad)
    cos_zen_edges = _edges(cos_zen_vals[::-1])

    H, _, _ = np.histogram2d(
        log10_e_true,
        cos_zen_true,
        bins=[log10_e_edges, cos_zen_edges],
        weights=w,
    )
    H = H[:, ::-1]

    delta_log10_e = np.diff(log10_e_edges)
    delta_e_gev   = energies_gev * np.log(10) * delta_log10_e

    delta_cos_zen_sorted = np.diff(cos_zen_edges)
    delta_cos_zen        = delta_cos_zen_sorted[::-1]

    aeff_cm2 = H / (delta_e_gev[:, np.newaxis] * 2 * np.pi * delta_cos_zen[np.newaxis, :])
    aeff_cm2 = np.where(np.isfinite(aeff_cm2) & (aeff_cm2 > 0), aeff_cm2, 0.0)

    return aeff_cm2 * _CM2_IN_EV2_INV


def build_aeff_per_species(
    primary_energy_gev: np.ndarray,
    primary_zenith_rad: np.ndarray,
    primary_type: np.ndarray,
    weights_gev_sr_cm2: np.ndarray,
    energies_gev: np.ndarray,
    zeniths_rad: np.ndarray,
    morph_label: str,
    veto_weights: np.ndarray = None,
) -> np.ndarray:
    """
    Build per-species A_eff(species, E, zenith) in eV^-2.

    Only includes species in ``_MORPH_PTYPES[morph_label]`` to avoid noisy
    estimates from MC events of minor species (e.g. NuE events that happen to
    be reconstructed as tracks have few events and inflated statistical weights).

    Returns
    -------
    ndarray, shape (6, N_E, N_ZEN)
        Per-species effective areas ordered by SPORE species index
        (NuE=0, NuEbar=1, NuMu=2, NuMuBar=3, NuTau=4, NuTauBar=5).
        Species not in the dominant set are zero.
    """
    n_e   = len(energies_gev)
    n_zen = len(zeniths_rad)
    aeff_all = np.zeros((6, n_e, n_zen))

    allowed = _MORPH_PTYPES[morph_label]
    for ptype, idx in allowed.items():
        sp_mask = primary_type == ptype
        if sp_mask.sum() < PSF_MIN_EVENTS:
            continue
        sp_veto = veto_weights[sp_mask] if veto_weights is not None else None
        aeff_all[idx] = build_aeff(
            primary_energy_gev[sp_mask],
            primary_zenith_rad[sp_mask],
            weights_gev_sr_cm2[sp_mask],
            energies_gev,
            zeniths_rad,
            veto_weights=sp_veto,
        )
    return aeff_all


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
    """Build energy-dependent PSF inverse-CDF from the zenith residual proxy."""
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

    _RAYLEIGH_MEDIAN_FACTOR = np.sqrt(np.log(4.0))

    sigmas = np.full(N_E, np.nan)
    for i_e in range(N_E):
        lo, hi = log10_e_edges[i_e], log10_e_edges[i_e + 1]
        in_bin = (log10_e >= lo) & (log10_e < hi)
        if in_bin.sum() >= PSF_MIN_EVENTS:
            median_psi = np.median(psi_proxy[in_bin])
            sigmas[i_e] = median_psi / _RAYLEIGH_MEDIAN_FACTOR

    good = np.where(np.isfinite(sigmas))[0]
    if len(good) == 0:
        raise RuntimeError("No energy bins have enough MC events to estimate the PSF.")
    sigmas = np.interp(np.arange(N_E), good, sigmas[good])

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
    """Build energy-resolution inverse-CDF (ln E_reco / E_true)."""
    valid    = (reco_energy_gev > 0) & (primary_energy_gev > 0)
    ln_ratio = np.log(reco_energy_gev[valid] / primary_energy_gev[valid])

    ln_ratio_sorted = np.sort(ln_ratio)
    n   = len(ln_ratio_sorted)
    cdf = (np.arange(n) + 0.5) / n

    us      = np.linspace(cdf[0], cdf[-1], n_u)
    inv_cdf = PchipInterpolator(cdf, ln_ratio_sorted)(us)

    us_full      = np.linspace(0.0, 1.0, n_u)
    inv_cdf_full = np.interp(us_full, us, inv_cdf)

    return us_full, inv_cdf_full


# ---------------------------------------------------------------------------
# HDF5 writer — hierarchical format with four morphology groups
# ---------------------------------------------------------------------------
def write_hdf5(
    output_path: str,
    energies_gev: np.ndarray,
    zeniths: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    morphology_data: dict,
) -> None:
    """
    Write the detector response file.

    Parameters
    ----------
    morphology_data : dict
        Keys are group names (e.g. "astro_track").  Values are dicts with:
            aeff        : ndarray — either (N_E, N_ZEN) aggregate (astro groups)
                          or (6, N_E, N_ZEN) per-species (atmo groups), in eV^-2
            us_ang      : ndarray (n_u_ang,)
            inv_cdfs_ang: ndarray (N_E, n_u_ang)
            us_e        : ndarray (n_u_e,)
            inv_cdf_e   : ndarray (n_u_e,)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["source"]   = "IceCube HESE 7.5-year data release (arXiv:2011.03545)"
        f.attrs["emin_gev"] = EMIN_GEV
        f.attrs["emax_gev"] = EMAX_GEV
        f.attrs["psf_note"] = (
            "PSF estimated from psi_proxy = sqrt(2)*|dzenith|; "
            "azimuth info not available in data release."
        )
        f.attrs["atmo_veto_note"] = (
            "atmo_* groups use A_eff weighted by conventionalSelfVetoCorrection. "
            "Use with the unvetoed conventional atmospheric flux. "
            "Approximation is valid near the reference normalisation Phi_conv=1."
        )

        for morph, d in morphology_data.items():
            mg = f.create_group(morph)

            g = mg.create_group("effective_area")
            g.create_dataset("energies",         data=energies_gev)
            g.create_dataset("zeniths",          data=zeniths)
            g.create_dataset("tabulated_values", data=d["aeff"])
            g.create_dataset("lower_bounds",     data=lower_bounds)
            g.create_dataset("upper_bounds",     data=upper_bounds)

            g = mg.create_group("angular_response")
            g.create_dataset("energies",  data=energies_gev)
            g.create_dataset("us",        data=d["us_ang"])
            g.create_dataset("inv_cdfs",  data=d["inv_cdfs_ang"])

            g = mg.create_group("energy_resolution")
            g.create_dataset("us",      data=d["us_e"])
            g.create_dataset("inv_cdf", data=d["inv_cdf_e"])

    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    (
        primary_energy, primary_zenith, primary_type, weights,
        reco_energy, reco_zenith, reco_morphology, conv_veto,
    ) = _load_mc()

    is_track      = reco_morphology == 1
    is_cascade    = reco_morphology == 0
    is_doublebang = reco_morphology == 2
    print(f"  Track events:      {is_track.sum():,}")
    print(f"  Cascade events:    {is_cascade.sum():,}")
    print(f"  Doublebang events: {is_doublebang.sum():,}")

    energies_gev = np.logspace(np.log10(EMIN_GEV), np.log10(EMAX_GEV), N_E)
    zeniths      = np.linspace(0.0, np.pi, N_ZEN)

    log10_emin_gev = np.log10(energies_gev[0])
    log10_emax_gev = np.log10(energies_gev[-1])
    lower_bounds = np.array([[log10_emin_gev]])
    upper_bounds = np.array([[log10_emax_gev]])

    # ------------------------------------------------------------------
    # Per-species effective areas — shape (6, N_E, N_ZEN) in eV^-2.
    # Each species has its own A_eff because the cascade/track detection
    # efficiency varies significantly by flavour at HESE energies (e.g.
    # NuE CC deposits 100% of energy → high cascade efficiency; NuMu NC
    # deposits only ~20% → needs much higher primary energy to pass the
    # 60 TeV deposited threshold).  Storing per-species A_effs lets the
    # SPORE sampler correctly weight Σ A_eff_sp × phi_sp without the
    # equal-species assumption that breaks for non-democratic fluxes.
    # ------------------------------------------------------------------
    # Astrophysical A_eff: aggregate over all species, then evenly split into
    # 6 equal per-species slices (shape 6 x N_E x N_ZEN).
    # The astrophysical flux is democratic (equal per species), so the correct
    # rate formula N = T ∫ Σ_sp A_eff_sp × phi_per_sp dE dΩ reduces to
    # T ∫ A_eff_total × phi_per_sp dE dΩ when we store A_eff_total/6 for every
    # species.  This avoids the noisy individual-species estimates caused by
    # sparse minority MC events (e.g. NuEbar in the track sample).
    print("\nBuilding astrophysical effective areas (aggregate / 6 broadcast) ...")
    aeff_track_agg = build_aeff(
        primary_energy[is_track], primary_zenith[is_track],
        weights[is_track], energies_gev, zeniths,
    )
    astro_track_aeff = np.stack([aeff_track_agg / 6.0] * 6, axis=0)  # (6, N_E, N_ZEN)
    print(f"  astro_track:   max (per-species) = {astro_track_aeff.max():.3e} eV^-2")

    aeff_cascade_agg = build_aeff(
        primary_energy[is_cascade], primary_zenith[is_cascade],
        weights[is_cascade], energies_gev, zeniths,
    )
    astro_cascade_aeff = np.stack([aeff_cascade_agg / 6.0] * 6, axis=0)  # (6, N_E, N_ZEN)
    print(f"  astro_cascade: max (per-species) = {astro_cascade_aeff.max():.3e} eV^-2")

    aeff_doublebang_agg = build_aeff(
        primary_energy[is_doublebang], primary_zenith[is_doublebang],
        weights[is_doublebang], energies_gev, zeniths,
    )
    astro_doublebang_aeff = np.stack([aeff_doublebang_agg / 6.0] * 6, axis=0)  # (6, N_E, N_ZEN)
    print(f"  astro_doublebang: max (per-species) = {astro_doublebang_aeff.max():.3e} eV^-2")

    # Atmospheric A_eff: per-species (3D: 6 x N_E x N_ZEN), veto-corrected.
    # The atmospheric flux is non-democratic (NuE cascade A_eff >> NuMu cascade
    # A_eff at HESE energies), so the correct rate requires Σ_sp A_eff_sp × phi_sp.
    print("\nBuilding per-species atmospheric effective areas (3D) ...")
    atmo_track_aeff = build_aeff_per_species(
        primary_energy[is_track], primary_zenith[is_track],
        primary_type[is_track], weights[is_track],
        energies_gev, zeniths, morph_label="track",
        veto_weights=conv_veto[is_track],
    )
    print(f"  atmo_track:   max = {atmo_track_aeff.max():.3e} eV^-2, nonzero = {(atmo_track_aeff > 0).sum()}")
    atmo_cascade_aeff = build_aeff_per_species(
        primary_energy[is_cascade], primary_zenith[is_cascade],
        primary_type[is_cascade], weights[is_cascade],
        energies_gev, zeniths, morph_label="cascade",
        veto_weights=conv_veto[is_cascade],
    )
    print(f"  atmo_cascade: max = {atmo_cascade_aeff.max():.3e} eV^-2, nonzero = {(atmo_cascade_aeff > 0).sum()}")

    # ------------------------------------------------------------------
    # PSF and energy resolution — detector property, same for astro/atmo
    # ------------------------------------------------------------------
    print("\nBuilding track PSF and energy resolution ...")
    track_us_ang, track_inv_cdfs_ang = build_angular_response(
        primary_energy[is_track], primary_zenith[is_track],
        reco_zenith[is_track], energies_gev, N_U_ANG,
    )
    track_us_e, track_inv_cdf_e = build_energy_resolution(
        primary_energy[is_track], reco_energy[is_track], N_U_E,
    )
    print(
        f"  Track PSF median: "
        f"{np.degrees(track_inv_cdfs_ang[:, N_U_ANG//2]).min():.1f}–"
        f"{np.degrees(track_inv_cdfs_ang[:, N_U_ANG//2]).max():.1f} deg"
    )
    print(f"  Track ln(E_reco/E_true): [{track_inv_cdf_e[0]:.2f}, {track_inv_cdf_e[-1]:.2f}]")

    print("\nBuilding cascade PSF and energy resolution ...")
    cascade_us_ang, cascade_inv_cdfs_ang = build_angular_response(
        primary_energy[is_cascade], primary_zenith[is_cascade],
        reco_zenith[is_cascade], energies_gev, N_U_ANG,
    )
    cascade_us_e, cascade_inv_cdf_e = build_energy_resolution(
        primary_energy[is_cascade], reco_energy[is_cascade], N_U_E,
    )
    print(
        f"  Cascade PSF median: "
        f"{np.degrees(cascade_inv_cdfs_ang[:, N_U_ANG//2]).min():.1f}–"
        f"{np.degrees(cascade_inv_cdfs_ang[:, N_U_ANG//2]).max():.1f} deg"
    )
    print(f"  Cascade ln(E_reco/E_true): [{cascade_inv_cdf_e[0]:.2f}, {cascade_inv_cdf_e[-1]:.2f}]")

    print("\nBuilding doublebang PSF and energy resolution ...")
    doublebang_us_ang, doublebang_inv_cdfs_ang = build_angular_response(
        primary_energy[is_doublebang], primary_zenith[is_doublebang],
        reco_zenith[is_doublebang], energies_gev, N_U_ANG,
    )
    doublebang_us_e, doublebang_inv_cdf_e = build_energy_resolution(
        primary_energy[is_doublebang], reco_energy[is_doublebang], N_U_E,
    )
    print(
        f"  Doublebang PSF median: "
        f"{np.degrees(doublebang_inv_cdfs_ang[:, N_U_ANG//2]).min():.1f}–"
        f"{np.degrees(doublebang_inv_cdfs_ang[:, N_U_ANG//2]).max():.1f} deg"
    )
    print(f"  Doublebang ln(E_reco/E_true): [{doublebang_inv_cdf_e[0]:.2f}, {doublebang_inv_cdf_e[-1]:.2f}]")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    morphology_data = {
        "astro_track": dict(
            aeff=astro_track_aeff,
            us_ang=track_us_ang, inv_cdfs_ang=track_inv_cdfs_ang,
            us_e=track_us_e,     inv_cdf_e=track_inv_cdf_e,
        ),
        "atmo_track": dict(
            aeff=atmo_track_aeff,
            us_ang=track_us_ang, inv_cdfs_ang=track_inv_cdfs_ang,
            us_e=track_us_e,     inv_cdf_e=track_inv_cdf_e,
        ),
        "astro_cascade": dict(
            aeff=astro_cascade_aeff,
            us_ang=cascade_us_ang, inv_cdfs_ang=cascade_inv_cdfs_ang,
            us_e=cascade_us_e,     inv_cdf_e=cascade_inv_cdf_e,
        ),
        "atmo_cascade": dict(
            aeff=atmo_cascade_aeff,
            us_ang=cascade_us_ang, inv_cdfs_ang=cascade_inv_cdfs_ang,
            us_e=cascade_us_e,     inv_cdf_e=cascade_inv_cdf_e,
        ),
        "astro_doublebang": dict(
            aeff=astro_doublebang_aeff,
            us_ang=doublebang_us_ang, inv_cdfs_ang=doublebang_inv_cdfs_ang,
            us_e=doublebang_us_e,     inv_cdf_e=doublebang_inv_cdf_e,
        ),
    }

    print(f"\nWriting {OUT_FILE} ...")
    write_hdf5(
        output_path  = OUT_FILE,
        energies_gev = energies_gev,
        zeniths      = zeniths,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds,
        morphology_data = morphology_data,
    )

    # ------------------------------------------------------------------
    # Sanity check: expected events under HESE best-fit astrophysical flux
    # Paper reports ~102 total, ~18 tracks, ~84 cascades
    # ------------------------------------------------------------------
    T_sec    = 7.5 * 365.25 * 24 * 3600
    phi0     = 6.37e-18 / 6.0  # per-species normalization (total / 6 flavors)
    e_pivot  = 1e5             # GeV
    gamma    = 2.87

    sin_w = np.sin(zeniths); sin_w /= sin_w.sum()

    def _expected_astro(aeff_6sp_eV2):
        """Compute expected astro events: N = T × Σ_sp ∫ A_eff_sp × phi_per_sp dE dΩ."""
        # Each species: sky-average over zenith, sum over energy
        phi_per_sp = phi0 * (energies_gev / e_pivot) ** (-gamma)
        de = np.gradient(energies_gev)
        total = 0.0
        for sp_idx in range(6):
            aeff_cm2_avg = (
                aeff_6sp_eV2[sp_idx] * sin_w[np.newaxis, :]
            ).sum(axis=1) / _CM2_IN_EV2_INV   # (N_E,) sky-averaged
            total += float((aeff_cm2_avg * phi_per_sp * 4 * np.pi * de * T_sec).sum())
        return total

    n_astro_track      = _expected_astro(astro_track_aeff)
    n_astro_cascade    = _expected_astro(astro_cascade_aeff)
    n_astro_doublebang = _expected_astro(astro_doublebang_aeff)
    print(f"\nSanity check — astrophysical (Phi_per_species={phi0:.2e}, gamma={gamma}, 7.5 yr):")
    print(f"  astro_track:      {n_astro_track:.1f}  (direct MC sum ~11.8)")
    print(f"  astro_cascade:    {n_astro_cascade:.1f}  (direct MC sum ~59.8)")
    print(f"  astro_doublebang: {n_astro_doublebang:.1f}  (direct MC sum ~2.8)")
    print(f"  astro total:      {n_astro_track + n_astro_cascade + n_astro_doublebang:.1f}  (direct MC sum ~74.4)")
    print("  (atmospheric contribution not shown here — requires Honda flux)")


if __name__ == "__main__":
    main()
