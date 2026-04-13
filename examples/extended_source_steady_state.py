"""
Extended source sampling — steady-state mode.

Simulates one year of track events from an isotropic E^-2 diffuse flux
observed by a Mediterranean detector.  In steady-state mode the effective
area at each sky position is averaged analytically over all hour angles,
which is the correct treatment for observations much longer than one
sidereal day.  The resulting A_eff is RA-independent and the sampled
declination distribution directly reflects the detector's time-averaged
sky coverage.

Run from the project root:
    python examples/extended_source_steady_state.py
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from spore.conventions import ureg
from spore.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import ExtendedSourceEventSampler

RESOURCES = Path(__file__).parent.parent / "resources"
RESPONSE  = str(RESOURCES / "icecube_10yr_response.h5")
FLUX_FILE = str(Path(__file__).parent / "_steady_state_flux.h5")

# ---------------------------------------------------------------------------
# Isotropic E^-2 flux, equal for NuMu and NuMuBar.
# ---------------------------------------------------------------------------
N_DEC = 40
N_E   = 50
sindecs      = np.linspace(-1.0, 1.0, N_DEC)
energies_gev = np.logspace(2.0, 6.0, N_E)

PHI_0 = 1e-12   # GeV^-1 cm^-2 s^-1 sr^-1 at 1 TeV

fluxes = np.zeros((6, N_DEC, N_E))
for j, E in enumerate(energies_gev):
    phi = PHI_0 * (E / 1e3) ** (-2.0)
    fluxes[2, :, j] = 0.5 * phi   # NuMu,    isotropic
    fluxes[3, :, j] = 0.5 * phi   # NuMuBar, isotropic

with h5py.File(FLUX_FILE, "w") as f:
    grp = f.create_group("flux")
    grp.create_dataset("sindecs",  data=sindecs)
    grp.create_dataset("energies", data=energies_gev)
    grp.create_dataset("fluxes",   data=fluxes)

# ---------------------------------------------------------------------------
# Detector: Mediterranean
# ---------------------------------------------------------------------------
T_OBS = ureg.Quantity(365.25, "day")   # one year

det = Detector.from_config({
    "properties": {
        "latitude":  36.3,
        "longitude": 16.1,
        "medium":    "Water",
    },
    "response": {"detector_response_file": RESPONSE},
})

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
src = ExtendedSource.from_config({"flux": {"location": f"{FLUX_FILE}:flux"}})

# ---------------------------------------------------------------------------
# Steady-state sampler: pass n_time_samples to enable hour-angle averaging
# of the effective area over a full diurnal cycle.
# ---------------------------------------------------------------------------
print("Building sampler (steady-state mode — A_eff averaged over diurnal cycle)...")
sampler = ExtendedSourceEventSampler(
    det, src,
    n_time_samples=100,
)

print("Sampling one year of track events...")
events = sampler.sample_events("track", deltat=T_OBS)
print(f"  Got {len(events)} events")

# ---------------------------------------------------------------------------
# Plot: declination distribution and energy spectrum.
# For an isotropic flux and a time-averaged A_eff, the declination
# distribution reflects the detector's solid-angle-weighted sensitivity.
# ---------------------------------------------------------------------------
reco_decs = np.degrees([e.reco_direction.declination     for e in events])
reco_ras  = np.degrees([e.reco_direction.right_ascension for e in events])
log10e    = [np.log10(e.reco_energy.to("GeV").magnitude) for e in events]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Extended source — steady-state mode (1 year, isotropic E^-2, Mediterranean detector)")

ax = axes[0]
ax.hist(reco_decs, bins=30, color="steelblue", edgecolor="white", lw=0.5)
ax.axvline(36.3, color="tomato", lw=1.5, ls="--", label="Detector latitude")
ax.set_xlabel("Reconstructed declination [deg]")
ax.set_ylabel("Events / bin")
ax.set_title("Declination distribution\n(reflects time-averaged A_eff)")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(log10e, bins=25, color="steelblue", edgecolor="white", lw=0.5)
ax.set_xlabel(r"$\log_{10}(E_\mathrm{reco}\,/\,\mathrm{GeV})$")
ax.set_ylabel("Events / bin")
ax.set_title("Reconstructed energy spectrum")

plt.tight_layout()
out = Path(__file__).parent / "extended_source_steady_state.png"
plt.savefig(out, dpi=150)
print(f"Plot saved to {out}")
plt.show()
