"""
Eddington bias: demonstration and correction.

For steeply falling neutrino spectra (typical astrophysical E^{-2.87}),
energy smearing redistributes events systematically to lower reconstructed
energies — an Eddington bias.  For HESE tracks this effect is especially
pronounced because the deposited energy (hadronic vertex + visible muon
segment) is only ~50% of the true neutrino energy on average, giving a
median log-energy shift of Delta = ln(E_reco/E_true) ≈ -0.65.

Key concept — threshold loss vs. spectral redistribution
---------------------------------------------------------
The HESE detector only observes events above ~60 TeV reco energy.  Smearing
moves events in two ways:

  1. Spectral redistribution: events with E_true in 60–300 TeV scatter down
     into lower reco-energy bins, inflating those bins (correction > 1).
  2. Threshold loss: events with E_true just above the analysis threshold
     scatter below it and disappear from the observable sample (correction < 1
     at the lowest observable reco-energy bins when the energy axis starts AT
     the threshold).

To see BOTH effects simultaneously the energy axis must START well below the
60 TeV HESE threshold (this script uses 10 TeV).  If you restrict the axis to
60 TeV+ you see only threshold loss (all corrections < 1), which looks like a
uniform suppression and is less instructive.

Two uses of the correction array
---------------------------------
  Forward-fold (recommended): multiply a naive A_eff × flux prediction by
      the correction to get the expected SPORE reco-energy histogram:
          expected_smeared[k] = naive_prediction[k] × correction[k]

  Debias (diagnostic): divide a sampled reco-energy histogram by the
      correction to recover the unsmeared distribution:
          debiased[k] = sampled_smeared[k] / correction[k]

This script:
  1. Samples 7.5 years of HESE astrophysical track events.
  2. Computes correction factors over 10 TeV – 10 PeV (wide range).
  3. Shows three curves:
       - Raw reco-energy spectrum (includes Eddington bias).
       - Debiased spectrum (raw / correction factors).
       - Unsmeared baseline (delta_clip=(0,0), i.e. E_reco = E_true).
  4. Demonstrates that the debiased and unsmeared curves agree, confirming
     that the excess at low reco energies is entirely due to energy smearing.

Run from the project root:
    python examples/eddington_bias.py

Requires:
    resources/hese_7yr_detector_response.h5  — HESE 7.5-yr IRF
    resources/hese_7yr_astro_flux.h5         — best-fit astrophysical flux
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spore.conventions import EarthCoordinate, ureg
from spore.detector import Detector
from spore.detector.detector_response import DetectorResponse
from spore.detector.detector import Medium
from spore.physics import Morphology
from spore.source import ExtendedSource
from spore.source.flux import Flux
from spore.event_sampling import SourceSampler

for _morph in ["astro_cascade", "astro_track", "atmo_cascade", "atmo_track", "astro_doublebang"]:
    Morphology.register(_morph)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).parent.parent
OUTPUT_DIR      = Path(__file__).parent / "output"
IRF_PATH        = str(ROOT / "resources" / "hese_7yr_detector_response.h5")
ASTRO_FLUX_PATH = str(ROOT / "resources" / "hese_7yr_astro_flux.h5")

LIVETIME = ureg.Quantity(7.5 * 365.25, "day")
MORPHOLOGY = "astro_track"
N_PSEUDO   = 300   # pseudo-experiments for Eddington correction estimate
N_SAMPLES  = 200   # pseudo-experiments for the comparison plot

# ---------------------------------------------------------------------------
# Build detector and sampler
# ---------------------------------------------------------------------------
_ICECUBE = EarthCoordinate(np.radians(-90.0), 0.0)

full_response = DetectorResponse.from_config({"detector_response_file": IRF_PATH})

def _filter(response, prefix):
    return DetectorResponse(
        effective_area={k: v for k, v in response.effective_area.items() if k.startswith(prefix)},
        angular_response={k: v for k, v in response.angular_response.items() if k.startswith(prefix)},
        energy_response={k: v for k, v in response.energy_response.items() if k.startswith(prefix)},
    )

astro_det  = Detector(_ICECUBE, Medium.Ice, _filter(full_response, "astro"))
astro_flux = Flux.from_config({"location": f"{ASTRO_FLUX_PATH}:astrophysical"})

print("Building sampler ...")
sampler = SourceSampler(
    astro_det, ExtendedSource(astro_flux),
    n_time_samples=100, n_e=60, n_ra=30, n_dec=30,
)
print(f"  Expected {sampler.expected_events(MORPHOLOGY, LIVETIME):.1f} events in {LIVETIME}")

# ---------------------------------------------------------------------------
# Energy bins: 10 TeV – 10 PeV (start BELOW the 60 TeV HESE threshold so
# that the spectral-redistribution excess at low reco energies is visible).
# The HESE analysis threshold (~60 TeV) is marked on the plots.
# ---------------------------------------------------------------------------
e_bins = np.logspace(np.log10(1e4), 7, 16)   # GeV: 10 TeV – 10 PeV, 15 bins
centers = np.sqrt(e_bins[:-1] * e_bins[1:])
HESE_THRESHOLD_GEV = 6e4   # ~60 TeV reco-energy analysis threshold

# ---------------------------------------------------------------------------
# Step 1: compute Eddington correction factors
# correction[k] = smeared_reco[k] / unsmeared_true[k]
#   > 1: smearing pumps events into bin k from neighbouring bins (excess)
#   < 1: smearing moves events out of bin k (deficit)
# ---------------------------------------------------------------------------
print(f"\nComputing Eddington correction ({N_PSEUDO} pseudo-experiments) ...")
correction = sampler.eddington_correction(MORPHOLOGY, e_bins, n_pseudo=N_PSEUDO)

print("\nlog10(E_reco / GeV)   correction factor")
for c, f in zip(centers, correction):
    threshold_note = "  [HESE threshold]" if abs(np.log10(c) - np.log10(HESE_THRESHOLD_GEV)) < 0.15 else ""
    marker = "  <- excess (spectral redistribution)" if f > 1.05 else (
             "  <- deficit (threshold loss)" if f < 0.95 else "")
    print(f"  {np.log10(c):.2f}                {f:.3f}{marker}{threshold_note}")

# ---------------------------------------------------------------------------
# Step 2: accumulate smeared and unsmeared spectra over N_SAMPLES runs
# ---------------------------------------------------------------------------
print(f"\nAccumulating {N_SAMPLES} pseudo-experiments ...")

smeared_counts   = np.zeros(len(e_bins) - 1)
unsmeared_counts = np.zeros(len(e_bins) - 1)

for seed in range(N_SAMPLES):
    # Full smearing
    evs = sampler.sample_events(MORPHOLOGY, deltat=LIVETIME, seed=seed)
    reco_es = np.array([ev.reco_energy.to("GeV").magnitude for ev in evs])
    smeared_counts += np.histogram(reco_es, bins=e_bins)[0]

    # No smearing: E_reco = E_true
    evs0 = sampler.sample_events(MORPHOLOGY, deltat=LIVETIME, seed=seed,
                                  delta_clip=(0, 0))
    reco_es0 = np.array([ev.reco_energy.to("GeV").magnitude for ev in evs0])
    unsmeared_counts += np.histogram(reco_es0, bins=e_bins)[0]

# Normalize to events per run
smeared_per_run   = smeared_counts   / N_SAMPLES
unsmeared_per_run = unsmeared_counts / N_SAMPLES

# Debiased: divide each bin by its correction factor
with np.errstate(invalid="ignore"):
    debiased_per_run = np.where(
        np.isfinite(correction) & (correction > 0),
        smeared_per_run / correction,
        np.nan,
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Eddington bias — HESE astrophysical tracks, {LIVETIME}\n"
    r"$\Phi \propto E^{-2.87}$, averaged over " + f"{N_SAMPLES} pseudo-experiments",
    fontsize=11,
)

bin_widths = np.diff(e_bins)

def _add_threshold(ax):
    """Draw a vertical line at the HESE 60 TeV analysis threshold."""
    lo, hi = ax.get_ylim()
    ax.axvline(HESE_THRESHOLD_GEV, color="k", lw=1, ls=":", alpha=0.5)
    ax.text(HESE_THRESHOLD_GEV * 1.06, lo + 0.03 * (hi - lo),
            "HESE\nthreshold", fontsize=7, va="bottom", color="k", alpha=0.7)

# --- Left panel: spectra ---
ax = axes[0]
kw = dict(where="mid", lw=2)
ax.step(centers, smeared_per_run,   label="Smeared (E_reco, full IRF)",      color="C0", **kw)
ax.step(centers, unsmeared_per_run, label="Unsmeared (E_reco = E_true)",      color="C2", **kw, ls="--")
ax.step(centers, debiased_per_run,  label="Debiased (smeared / correction)",  color="C1", **kw, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"Reconstructed energy [GeV]")
ax.set_ylabel("Mean events per 7.5-yr run")
ax.set_title("Energy spectra  (10 TeV – 10 PeV)")
ax.legend(fontsize=9)
ax.set_xlim(e_bins[0], e_bins[-1])
_add_threshold(ax)
# Annotate the sub-threshold excess if it is visible
_sub = smeared_per_run[centers < HESE_THRESHOLD_GEV]
if _sub.max() > 0:
    _peak_idx = np.argmax(_sub)
    _peak_x = centers[centers < HESE_THRESHOLD_GEV][_peak_idx]
    _peak_y = _sub[_peak_idx]
    ax.annotate(
        "sub-threshold\nredistribution\n(excess)",
        xy=(_peak_x, _peak_y),
        xytext=(_peak_x * 0.7, _peak_y * 1.8 + ax.get_ylim()[1] * 0.05),
        fontsize=7, color="C0",
        arrowprops=dict(arrowstyle="->", color="C0", lw=0.8),
    )

# --- Right panel: correction factors ---
ax = axes[1]
valid = np.isfinite(correction)
ax.step(centers[valid], correction[valid], where="mid", color="C3", lw=2,
        label="correction = smeared / unsmeared")
ax.axhline(1.0, color="gray", lw=1, ls="--")
ax.fill_between(centers[valid], 1.0, correction[valid],
                where=correction[valid] > 1, alpha=0.2, color="C0",
                label="Excess: smearing pumps events in  (>1)")
ax.fill_between(centers[valid], correction[valid], 1.0,
                where=correction[valid] < 1, alpha=0.2, color="C2",
                label="Deficit: smearing pushes events out  (<1)")
ax.set_xscale("log")
ax.set_xlabel(r"Reconstructed energy [GeV]")
ax.set_ylabel("Correction factor  (smeared / unsmeared)")
ax.set_title("Eddington correction factors")
ax.legend(fontsize=8)
ax.set_xlim(e_bins[0], e_bins[-1])
_add_threshold(ax)

plt.tight_layout()
out = OUTPUT_DIR / "eddington_bias.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()
