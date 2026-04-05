"""
Multi-detector realtime point source search example.

This script demonstrates how to use MultiDetectorPointSourceSampler to
simulate a joint neutrino search across two detectors — a South Pole
detector (IceCube-like) and a Northern Hemisphere detector (Mediterranean-like).

The source is a southern-sky point source at declination -30 degrees.  At
that declination the source is upgoing for the South Pole detector (zenith ~60
degrees, a clean track signal region) and below the horizon for the Northern
detector (accessible only via Earth-crossing muon neutrinos). This geometry
makes the South Pole detector more sensitive to this particular source, but the
combined event set would feed a joint likelihood analysis.

Run from the project root:
    python examples/multi_detector_source_search.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from spore.conventions import SkyCoordinate
from spore.conventions.units import units
from spore.physics import neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.source.point_source import PointSource
from spore.detector.detector import Detector
from spore.event_sampling import MultiDetectorPointSourceSampler

RESOURCES = Path(__file__).parent.parent / "resources"
RESPONSE_FILE = str(RESOURCES / "example_detector_response.h5")

# ---------------------------------------------------------------------------
# Source: power-law flux at declination -30 degrees, RA 83.8 degrees
# (roughly the direction of the Crab Nebula)
# ---------------------------------------------------------------------------
emin = 1e2 * units.GeV
emax = 1e6 * units.GeV
pivot = 1e5 * units.GeV

pl = PowerLaw(gamma=2.0, emin=emin, emax=emax, pivot=pivot)

# Normalization: 1e-12 GeV^-1 cm^-2 s^-1 at the pivot, equal for all flavors
norm = 1e-12 / units.GeV / units.cm**2 / units.sec
flux = Flux(
    normalizations={nu: norm for nu in neutrinos},
    distributions={nu: pl for nu in neutrinos},
)
source = PointSource(
    flux=flux,
    location=SkyCoordinate(np.radians(-30.0), np.radians(83.8)),
)

# ---------------------------------------------------------------------------
# Detectors
#
# Both detectors share the same response file (the example IceCube response)
# so that the example is self-contained. In a real analysis each detector
# would have its own measured effective area, PSF, and energy resolution.
# ---------------------------------------------------------------------------
south_pole_config = {
    "properties": {
        "latitude": -90.0,   # South Pole
        "longitude": 0.0,
        "medium": "Ice",
    },
    "response": {"detector_response_file": RESPONSE_FILE},
}

northern_config = {
    "properties": {
        "latitude": 36.3,    # Mediterranean (KM3NeT ARCA-like coordinates)
        "longitude": 16.1,
        "medium": "Water",
    },
    "response": {"detector_response_file": RESPONSE_FILE},
}

south_pole_det = Detector.from_config(south_pole_config)
northern_det = Detector.from_config(northern_config)

# ---------------------------------------------------------------------------
# Multi-detector sampler
# ---------------------------------------------------------------------------
sampler = MultiDetectorPointSourceSampler(
    detectors=[
        (south_pole_det, "SouthPole"),
        (northern_det, "Northern"),
    ],
    source=source,
)

# ---------------------------------------------------------------------------
# Sample one year of track events, reference epoch MJD 60355
# ---------------------------------------------------------------------------
T_OBS = 365.25 * units.day   # one year in natural units (eV^-1)
T_MJD = 60355.0

print("Sampling one year of track events across both detectors...")
events = sampler.sample_events("track", t=T_MJD, deltat=T_OBS)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
by_detector = defaultdict(list)
for event in events:
    by_detector[event.detector_id].append(event)

print(f"\nTotal events: {len(events)}")
for det_name in sampler.detector_names:
    det_events = by_detector[det_name]
    if not det_events:
        print(f"  {det_name}: 0 events")
        continue
    true_energies_tev = [e.true_energy / units.TeV for e in det_events]
    reco_energies_tev = [e.reco_energy / units.TeV for e in det_events]
    print(
        f"  {det_name}: {len(det_events)} events | "
        f"median true E = {np.median(true_energies_tev):.2f} TeV | "
        f"median reco E = {np.median(reco_energies_tev):.2f} TeV"
    )

# ---------------------------------------------------------------------------
# Angular separation between true and reconstructed direction
# ---------------------------------------------------------------------------
def angular_separation(e):
    v_true = e.true_direction.to_cartesian()
    v_reco = e.reco_direction.to_cartesian()
    return np.degrees(np.arccos(np.clip(np.dot(v_true, v_reco), -1.0, 1.0)))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Multi-detector point source simulation (1 year, tracks)")

colors = {"SouthPole": "steelblue", "Northern": "tomato"}

# --- Sky map of reconstructed directions ---
ax = axes[0]
for det_name in sampler.detector_names:
    det_events = by_detector[det_name]
    if not det_events:
        continue
    ras = np.degrees([e.reco_direction.right_ascension for e in det_events])
    decs = np.degrees([e.reco_direction.declination for e in det_events])
    ax.scatter(ras, decs, s=8, alpha=0.5, color=colors[det_name], label=det_name)

ax.scatter(
    [np.degrees(source.location.right_ascension)],
    [np.degrees(source.location.declination)],
    s=150, marker="*", color="gold", zorder=5, label="Source"
)
ax.set_xlabel("Right ascension [deg]")
ax.set_ylabel("Declination [deg]")
ax.set_title("Reconstructed directions")
ax.legend(fontsize=8)

# --- Energy spectra ---
ax = axes[1]
e_bins = np.logspace(-1, 3, 25)   # 0.1 TeV – 1000 TeV
for det_name in sampler.detector_names:
    det_events = by_detector[det_name]
    if not det_events:
        continue
    energies = [e.reco_energy / units.TeV for e in det_events]
    ax.hist(energies, bins=e_bins, histtype="step", color=colors[det_name],
            label=det_name, linewidth=1.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Reconstructed energy [TeV]")
ax.set_ylabel("Events / bin")
ax.set_title("Reconstructed energy spectra")
ax.legend(fontsize=8)

# --- Angular resolution (68th percentile) ---
ax = axes[2]
for det_name in sampler.detector_names:
    det_events = by_detector[det_name]
    if not det_events:
        continue
    seps = [angular_separation(e) for e in det_events]
    reco_energies = [e.reco_energy / units.TeV for e in det_events]
    idx_sort = np.argsort(reco_energies)
    ax.scatter(
        np.array(reco_energies)[idx_sort],
        np.array(seps)[idx_sort],
        s=4, alpha=0.4, color=colors[det_name], label=det_name
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Reconstructed energy [TeV]")
ax.set_ylabel("Angular separation [deg]")
ax.set_title("True vs. reco direction offset")
ax.legend(fontsize=8)

plt.tight_layout()
plot_path = Path(__file__).parent / "multi_detector_example.png"
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
plt.show()
