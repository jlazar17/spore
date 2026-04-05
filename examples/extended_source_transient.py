"""
Extended source sampling — transient mode.

Simulates a short neutrino flare (6 hours) from a spatially extended source
centred on the Galactic Centre, observed by a Mediterranean detector at a
fixed epoch.  In transient mode the sky-to-detector geometry is evaluated
once at the reference epoch, which is the correct treatment when the
observation window is short compared to one sidereal day.

Run from the project root:
    python examples/extended_source_transient.py
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from spore.conventions import units
from spore.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import ExtendedSourceEventSampler

RESOURCES = Path(__file__).parent.parent / "resources"
RESPONSE  = str(RESOURCES / "example_detector_response.h5")
FLUX_FILE = str(Path(__file__).parent / "_transient_flux.h5")

# ---------------------------------------------------------------------------
# Build a Gaussian-in-dec extended flux centred on the Galactic Centre
# (dec ~ -29 deg, RA ~ 266 deg).  The flux is isotropic in RA and has a
# 5-degree Gaussian width in declination.  Spectrum: E^-2.
# ---------------------------------------------------------------------------
N_DEC = 40
N_E   = 50
sindecs      = np.linspace(-1.0, 1.0, N_DEC)
energies_gev = np.logspace(2.0, 6.0, N_E)

GC_DEC_RAD = np.radians(-29.0)
WIDTH_RAD   = np.radians(5.0)
PHI_0       = 1e-12   # GeV^-1 cm^-2 s^-1 sr^-1 at 1 TeV

fluxes = np.zeros((6, N_DEC, N_E))
for i, sd in enumerate(sindecs):
    dec = np.arcsin(sd)
    angular_weight = np.exp(-0.5 * ((dec - GC_DEC_RAD) / WIDTH_RAD) ** 2)
    for j, E in enumerate(energies_gev):
        phi = PHI_0 * (E / 1e3) ** (-2.0) * angular_weight
        fluxes[2, i, j] = 0.5 * phi   # NuMu
        fluxes[3, i, j] = 0.5 * phi   # NuMuBar

with h5py.File(FLUX_FILE, "w") as f:
    grp = f.create_group("flux")
    grp.create_dataset("sindecs",  data=sindecs)
    grp.create_dataset("energies", data=energies_gev)
    grp.create_dataset("fluxes",   data=fluxes)

# ---------------------------------------------------------------------------
# Detector: Mediterranean (KM3NeT ARCA-like)
# ---------------------------------------------------------------------------
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
# Transient sampler: deltat=None means geometry fixed at t0.
# t0 is set internally to MJD 51544 (J2000).  A flare at this epoch means
# the Galactic Centre is at a specific azimuth and the A_eff reflects the
# actual detector pointing at that instant.
# ---------------------------------------------------------------------------
print("Building sampler (transient mode — geometry fixed at t0)...")
sampler = ExtendedSourceEventSampler(det, src, burnin=5_000, deltat=None)

# 6-hour observation window
T_FLARE = 6 * 3600 * units.sec
N_EVENTS = 500

print(f"Sampling {N_EVENTS} track events over a 6-hour flare...")
events = sampler.sample_events("track", nevent=N_EVENTS, oversample=5)
print(f"  Got {len(events)} events")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
true_decs  = np.degrees([e.true_direction.declination  for e in events])
reco_decs  = np.degrees([e.reco_direction.declination  for e in events])
true_ras   = np.degrees([e.true_direction.right_ascension for e in events])
reco_ras   = np.degrees([e.reco_direction.right_ascension for e in events])
log10e     = [np.log10(e.reco_energy / units.GeV) for e in events]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Extended source — transient mode (6-hour flare, Mediterranean detector)")

ax = axes[0]
ax.scatter(reco_ras, reco_decs, s=6, alpha=0.5, color="steelblue", label="Reco")
ax.scatter(true_ras, true_decs, s=6, alpha=0.3, color="tomato",    label="True")
ax.axhline(-29, color="k", lw=1, ls="--", label="GC dec")
ax.set_xlabel("Right ascension [deg]")
ax.set_ylabel("Declination [deg]")
ax.set_title("Sky positions")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(log10e, bins=20, color="steelblue", edgecolor="white", lw=0.5)
ax.set_xlabel(r"$\log_{10}(E_\mathrm{reco}\,/\,\mathrm{GeV})$")
ax.set_ylabel("Events / bin")
ax.set_title("Reconstructed energy")

plt.tight_layout()
out = Path(__file__).parent / "extended_source_transient.png"
plt.savefig(out, dpi=150)
print(f"Plot saved to {out}")
plt.show()
