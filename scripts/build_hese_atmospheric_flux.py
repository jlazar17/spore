"""
Build the unvetoed Honda+Gaisser conventional atmospheric neutrino flux
directly from the HESE 7.5-year MC data release.

Each MC event stores the Honda+Gaisser flux evaluated at its true primary
energy and zenith angle in the ``pionFlux`` and ``kaonFlux`` fields.  These
values are binned onto a 2D (sin_dec, log_E) grid per neutrino species to
reconstruct the flux as a smooth function of direction and energy.

The self-veto correction is deliberately NOT applied here: it is an
instrumental effect already folded into the ``atmo_*`` A_eff groups of
``hese_7yr_detector_response.h5``.  Use this flux with those A_eff groups.

Output
------
    resources/hese_7yr_atmo_flux.h5

    Group  ``conventional``
        sindecs  (N_DEC,)           — sin(declination) grid
        energies (N_E,)             — energy grid in GeV
        fluxes   (6, N_DEC, N_E)   — differential flux per species,
                                      GeV^-1 cm^-2 s^-1 sr^-1

Species ordering (SPORE convention)
------------------------------------
    0  NuE       (primaryType = 12)
    1  NuEbar    (primaryType = -12)
    2  NuMu      (primaryType = 14)
    3  NuMuBar   (primaryType = -14)
    4  NuTau     — zero (Honda+Gaisser has no tau component)
    5  NuTauBar  — zero

Data release reference
----------------------
IceCube Collaboration, Phys.Rev.D 104 (2021) 022002, arXiv:2011.03545
"""

import os
import json
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, "..")

_DEFAULT_DATA_DIR = os.path.join(
    _HERE, "..", "resources", "data_releases",
    "hese_7yr_data_release", "resources", "data",
)
DATA_DIR = os.environ.get("HESE_DATA_DIR", _DEFAULT_DATA_DIR)
OUT_FILE = os.path.join(_ROOT, "resources", "hese_7yr_atmo_flux.h5")

# ---------------------------------------------------------------------------
# Grid parameters  (match existing hese_flux.h5 for easy comparison)
# ---------------------------------------------------------------------------
N_DEC  = 40
N_E    = 80
E_MIN_GEV = 1e2   # 100 GeV
E_MAX_GEV = 1e6   # 1 PeV

# primaryType → SPORE species index
_PTYPE_TO_IDX = {12: 0, -12: 1, 14: 2, -14: 3, 16: 4, -16: 5}
_SPECIES_NAMES = ["NuE", "NuEbar", "NuMu", "NuMuBar", "NuTau", "NuTauBar"]


# ---------------------------------------------------------------------------
# Load MC
# ---------------------------------------------------------------------------
def _load_mc():
    merged = {}
    for fname in ["HESE_mc_truth.json", "HESE_mc_observable.json", "HESE_mc_flux.json"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"  Loading {fname} ...")
        merged.update(json.load(open(path)))

    E       = np.array(merged["primaryEnergy"])
    zen     = np.array(merged["primaryZenith"])
    ptype   = np.array(merged["primaryType"])
    pion    = np.array(merged["pionFlux"])
    kaon    = np.array(merged["kaonFlux"])
    weights = np.array(merged["weightOverFluxOverLivetime"])
    itype   = np.array(merged["interactionType"])

    is_nu = itype != 0
    return E[is_nu], zen[is_nu], ptype[is_nu], (pion + kaon)[is_nu], weights[is_nu]


# ---------------------------------------------------------------------------
# Build flux grid for one species
# ---------------------------------------------------------------------------
def _build_species_flux(E_mc, zen_mc, flux_mc, weights_mc, sindecs, log_es):
    """
    Reconstruct phi(sin_dec, log_E) for one species from MC event samples.

    For each grid cell the weighted-mean of the per-event Honda+Gaisser flux
    values is computed.  Empty cells are filled by 2D linear interpolation
    from neighbouring populated cells.

    Parameters
    ----------
    E_mc, zen_mc : MC event true energies (GeV) and zeniths (rad)
    flux_mc      : per-event (pionFlux + kaonFlux) in GeV^-1 cm^-2 s^-1 sr^-1
    weights_mc   : per-event weightOverFluxOverLivetime
    sindecs      : (N_DEC,) target sin(dec) grid  — NOTE: dec = pi/2 - zen
    log_es       : (N_E,)   target log(E/GeV) grid

    Returns
    -------
    grid : ndarray, shape (N_DEC, N_E)
    """
    sin_dec_mc = np.sin(np.pi / 2 - zen_mc)   # zenith → declination → sin(dec)
    log_e_mc   = np.log(E_mc)

    n_dec = len(sindecs)
    n_e   = len(log_es)

    # Grid edges
    dsd = (sindecs[-1] - sindecs[0]) / (n_dec - 1)
    sd_edges = np.clip(
        np.concatenate([[sindecs[0] - dsd/2],
                        (sindecs[:-1] + sindecs[1:]) / 2,
                        [sindecs[-1] + dsd/2]]),
        -1.0, 1.0,
    )
    dle = (log_es[-1] - log_es[0]) / (n_e - 1)
    le_edges = np.concatenate([[log_es[0] - dle/2],
                                (log_es[:-1] + log_es[1:]) / 2,
                                [log_es[-1] + dle/2]])

    # Weighted sum and weight sum per bin
    w_flux_sum, _ = np.histogramdd(
        np.column_stack([sin_dec_mc, log_e_mc]),
        bins=[sd_edges, le_edges],
        weights=weights_mc * flux_mc,
    )
    w_sum, _ = np.histogramdd(
        np.column_stack([sin_dec_mc, log_e_mc]),
        bins=[sd_edges, le_edges],
        weights=weights_mc,
    )

    populated = w_sum > 0
    grid = np.where(populated, w_flux_sum / np.where(populated, w_sum, 1.0), 0.0)

    # Fill empty cells by interpolating from populated neighbours
    n_empty = (~populated).sum()
    if n_empty > 0 and populated.sum() > 3:
        # Build interpolator from populated cells and evaluate at empty ones
        idx_pop = np.argwhere(populated)
        sd_pop  = sindecs[idx_pop[:, 0]]
        le_pop  = log_es[idx_pop[:, 1]]
        val_pop = grid[populated]

        sd_all = sindecs[np.argwhere(~populated)[:, 0]]
        le_all = log_es[np.argwhere(~populated)[:, 1]]

        try:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
            pts = np.column_stack([sd_pop, le_pop])
            lin = LinearNDInterpolator(pts, val_pop, fill_value=np.nan)
            filled = lin(np.column_stack([sd_all, le_all]))

            nan_mask = np.isnan(filled)
            if nan_mask.any():
                nn = NearestNDInterpolator(pts, val_pop)
                filled[nan_mask] = nn(np.column_stack([sd_all[nan_mask], le_all[nan_mask]]))

            idx_empty = np.argwhere(~populated)
            for (i, j), v in zip(idx_empty, filled):
                grid[i, j] = max(v, 0.0)
        except Exception:
            pass   # leave zeros if interpolation fails

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading MC ...")
    E_mc, zen_mc, ptype_mc, flux_mc, weights_mc = _load_mc()
    print(f"  {len(E_mc):,} neutrino events loaded")

    sindecs = np.linspace(-1.0, 1.0, N_DEC)
    log_es  = np.linspace(np.log(E_MIN_GEV), np.log(E_MAX_GEV), N_E)
    energies_gev = np.exp(log_es)

    fluxes = np.zeros((6, N_DEC, N_E))

    for ptype, idx in sorted(_PTYPE_TO_IDX.items()):
        name = _SPECIES_NAMES[idx]
        mask = ptype_mc == ptype
        if mask.sum() == 0 or flux_mc[mask].max() == 0:
            print(f"  {name:8s} (type {ptype:+d}): zero flux — skipping")
            continue
        print(f"  {name:8s} (type {ptype:+d}): {mask.sum():,} events ...")
        fluxes[idx] = _build_species_flux(
            E_mc[mask], zen_mc[mask], flux_mc[mask], weights_mc[mask],
            sindecs, log_es,
        )
        nonzero = fluxes[idx][fluxes[idx] > 0]
        print(f"    flux range: {nonzero.min():.2e} – {nonzero.max():.2e} GeV^-1 cm^-2 s^-1 sr^-1")

    # Quick sanity check: integrate to get expected atmospheric event count
    # (without veto, just to verify the flux magnitude is correct)
    _CM2_IN_EV2_INV = 8065.54815355 ** 2
    T = 7.5 * 365.25 * 24 * 3600
    print("\nSanity check (no veto, Honda+Gaisser integrated over all sky):")
    for idx, name in enumerate(_SPECIES_NAMES):
        phi = fluxes[idx]  # (N_DEC, N_E)
        # rough integral: Σ phi * dE * dOmega = phi * E * dlogE * dsd * 2pi
        dlogE = np.gradient(log_es)
        dsd   = np.gradient(sindecs)
        integral = (phi * energies_gev[np.newaxis, :] * dlogE[np.newaxis, :]
                    * dsd[:, np.newaxis] * 2 * np.pi).sum()
        print(f"  {name:8s}: integrated flux = {integral:.2e} cm^-2 s^-1")

    print(f"\nWriting {OUT_FILE} ...")
    os.makedirs(os.path.dirname(os.path.abspath(OUT_FILE)), exist_ok=True)
    with h5py.File(OUT_FILE, "w") as f:
        grp = f.create_group("conventional")
        grp.attrs["source"]      = "IceCube HESE 7.5yr MC pionFlux + kaonFlux (Honda+Gaisser)"
        grp.attrs["reference"]   = "arXiv:2011.03545"
        grp.attrs["self_veto"]   = "NOT applied — use with atmo_* A_eff groups from hese_7yr_detector_response.h5"
        grp.attrs["units_flux"]  = "GeV^-1 cm^-2 s^-1 sr^-1"
        grp.attrs["units_E"]     = "GeV"
        grp.create_dataset("sindecs",  data=sindecs)
        grp.create_dataset("energies", data=energies_gev)
        grp.create_dataset("fluxes",   data=fluxes)
    print("Done.")


if __name__ == "__main__":
    main()
