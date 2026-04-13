"""
Multi-detector point source search example.

Demonstrates how to simulate a joint neutrino search across two detectors —
a South Polar detector and a Mediterranean detector — by passing a list of
detectors to PointSourceEventSampler.  Each detector independently Poisson-
samples its expected event count; the returned event list is tagged by an
integer detector index (0 = South Polar, 1 = Mediterranean).

The source is a southern-sky point source at declination -30 degrees.  At
that declination the source is upgoing for the South Polar detector (zenith
~60 degrees, a clean track signal region) and below the horizon for the
Mediterranean detector (accessible only via Earth-crossing muon neutrinos).
This geometry makes the South Polar detector more sensitive to this particular
source, but the combined event set feeds a joint likelihood analysis.

Run from the project root:
    python examples/multi_detector_source_search.py
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from spore.conventions import ureg, SkyCoordinate
from spore.source import PointSource
from spore.detector import Detector
from spore.event_sampling import PointSourceEventSampler

RESOURCES = Path(__file__).parent.parent / "resources"
RESPONSE_FILE = str(RESOURCES / "icecube_10yr_response.h5")

# ---------------------------------------------------------------------------
# Source: soft power-law at declination -30 degrees
# Both detectors share the same response file so the example is self-
# contained; in a real analysis each would use its own measured IRFs.
# ---------------------------------------------------------------------------
SOURCE_CONFIG = {
    "flux": {
        "norm":  1e-14,   # GeV^-1 cm^-2 s^-1 per species
        "gamma": 2.5,
        "pivot": 1e3,     # GeV
        "emin":  1e2,
        "emax":  1e7,
    },
    "location": {
        "right_ascension": 83.8,
        "declination":    -30.0,
    },
}
source = PointSource.from_config(SOURCE_CONFIG)

# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
south_pole_det = Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
    "response":   {"detector_response_file": RESPONSE_FILE},
})

mediterranean_det = Detector.from_config({
    "properties": {"latitude": 36.3, "longitude": 16.1, "medium": "Water"},
    "response":   {"detector_response_file": RESPONSE_FILE},
})

# ---------------------------------------------------------------------------
# Multi-detector sampler: pass a list of detectors.
# detector_id on each event is the integer index into this list (0 or 1).
# n_time_samples enables diurnal-average effective area, appropriate for a
# one-year observation.
# ---------------------------------------------------------------------------
sampler = PointSourceEventSampler(
    [south_pole_det, mediterranean_det],
    source,
    n_time_samples=50,
)

# ---------------------------------------------------------------------------
# Sample one year of track events
# ---------------------------------------------------------------------------
T_OBS = ureg.Quantity(365.25, "day")

print("Sampling one year of track events across both detectors...")
events = sampler.sample_events("track", deltat=T_OBS)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
DET_LABELS = {0: "South Polar", 1: "Mediterranean"}
by_detector = defaultdict(list)
for event in events:
    by_detector[event.detector_id].append(event)

print(f"\nTotal events: {len(events)}")
for det_id, label in DET_LABELS.items():
    det_events = by_detector[det_id]
    if not det_events:
        print(f"  {label}: 0 events")
        continue
    true_e_tev = [e.true_energy.to("TeV").magnitude for e in det_events]
    reco_e_tev = [e.reco_energy.to("TeV").magnitude for e in det_events]
    print(
        f"  {label}: {len(det_events)} events | "
        f"median true E = {np.median(true_e_tev):.2f} TeV | "
        f"median reco E = {np.median(reco_e_tev):.2f} TeV"
    )

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Multi-detector point source simulation (1 year, tracks)")

colors = {0: "steelblue", 1: "tomato"}

# --- Sky map of reconstructed directions ---
ax = axes[0]
for det_id, label in DET_LABELS.items():
    det_events = by_detector[det_id]
    if not det_events:
        continue
    ras  = np.degrees([e.reco_direction.right_ascension for e in det_events])
    decs = np.degrees([e.reco_direction.declination     for e in det_events])
    ax.scatter(ras, decs, s=8, alpha=0.5, color=colors[det_id], label=label)

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
for det_id, label in DET_LABELS.items():
    det_events = by_detector[det_id]
    if not det_events:
        continue
    energies = [e.reco_energy.to("TeV").magnitude for e in det_events]
    ax.hist(energies, bins=e_bins, histtype="step", color=colors[det_id],
            label=label, linewidth=1.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Reconstructed energy [TeV]")
ax.set_ylabel("Events / bin")
ax.set_title("Reconstructed energy spectra")
ax.legend(fontsize=8)

# --- Angular separation true vs reco ---
ax = axes[2]
for det_id, label in DET_LABELS.items():
    det_events = by_detector[det_id]
    if not det_events:
        continue
    seps = [
        np.degrees(e.true_direction.separation(e.reco_direction))
        for e in det_events
    ]
    reco_energies = [e.reco_energy.to("TeV").magnitude for e in det_events]
    idx_sort = np.argsort(reco_energies)
    ax.scatter(
        np.array(reco_energies)[idx_sort],
        np.array(seps)[idx_sort],
        s=4, alpha=0.4, color=colors[det_id], label=label,
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
