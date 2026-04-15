"""Effective area loading parameter figures for the SPORE paper appendix.

This script produces two figures that document the two tunable HDF5 metadata
parameters that control how effective areas are loaded and pre-processed:

  effa_trim_isolated.png
      Side-by-side comparison of the PS-10yr track effective area with
      ``trim_isolated=True`` (default) and ``trim_isolated=False``.  Shows
      the isolated low-statistics bins that are removed.

  effa_smoothing_sigma.png
      Overlay of PS-10yr track effective area (upgoing dec band, fixed zenith)
      for several values of ``smoothing_sigma`` from 0 (no smoothing) to 2.0.
      Illustrates the trade-off between suppressing MC noise and preserving the
      high-energy shape.

Both figures use the PS-10yr IRF (``resources/configs/ps10yr_detector_response.h5``)
which ships with ``smoothing_sigma=1.3`` and ``trim_isolated=True`` baked into
its ``meta`` group.

The HDF5 metadata format is::

    h5f["meta"].attrs["smoothing_sigma"]  # float  — Gaussian kernel σ in energy bins
    h5f["meta"].attrs["trim_isolated"]    # bool   — whether to zero isolated edge bins

Priority for both parameters: explicit caller argument > HDF5 meta > code default.

Usage
-----
Run from the repository root::

    python scripts/paper_plots/effa_loading_appendix.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

from spore.detector.detector_response.utils import effa_helper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO       = Path(__file__).resolve().parents[2]
PS_RESPONSE = _REPO / "resources" / "configs" / "ps10yr_detector_response.h5"
OUT_DIR     = _REPO / "scripts" / "paper_plots"

# ---------------------------------------------------------------------------
# Load raw tables
# ---------------------------------------------------------------------------
with h5py.File(PS_RESPONSE) as hf:
    grp  = hf["track/effective_area"]
    zens_raw = grp["zeniths"][:]
    es_raw   = grp["energies"][:]
    vals_raw = grp["tabulated_values"][:]

# Upgoing dec band: pick the zenith column closest to 140° (zenith ~ 2.44 rad)
ZEN_TARGET = np.radians(140.0)
iz_plot = int(np.argmin(np.abs(zens_raw - ZEN_TARGET)))
zen_label = f"zenith = {np.degrees(zens_raw[iz_plot]):.0f}°"

e_plot = np.logspace(np.log10(es_raw[0]), np.log10(es_raw[-1]), 300)

# ---------------------------------------------------------------------------
# Helper: build A_eff(E) for a single zenith column with given options
# ---------------------------------------------------------------------------
def _effa_col(trim_isolated: bool, smoothing_sigma: float) -> np.ndarray:
    fn = effa_helper(zens_raw, es_raw, vals_raw,
                     trim_isolated=trim_isolated,
                     smoothing_sigma=smoothing_sigma)
    return np.array([fn(zens_raw[iz_plot], e) for e in e_plot])


# ---------------------------------------------------------------------------
# Figure 1: trim_isolated comparison
# ---------------------------------------------------------------------------
aeff_trimmed   = _effa_col(trim_isolated=True,  smoothing_sigma=0.0)
aeff_untrimmed = _effa_col(trim_isolated=False, smoothing_sigma=0.0)

# Raw tabulated column (for reference dots)
raw_col = vals_raw[:, iz_plot]
raw_nz  = raw_col > 0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, vals, title in [
    (axes[0], aeff_untrimmed, "trim_isolated = False"),
    (axes[1], aeff_trimmed,   "trim_isolated = True  (default)"),
]:
    ax.plot(e_plot / 1e3, vals / 1e4, lw=1.6, color="#4477AA")
    ax.scatter(es_raw[raw_nz] / 1e3, raw_col[raw_nz] / 1e4,
               s=12, zorder=5, color="#EE6677", label="tabulated (non-zero)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True energy [TeV]", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, lw=0.4, alpha=0.4, which="both")
    ax.set_xlim(e_plot[0] / 1e3, e_plot[-1] / 1e3)

axes[0].set_ylabel(r"Effective area [m$^2$]", fontsize=11)
axes[0].legend(fontsize=9, loc="upper left")

fig.suptitle(
    f"PS-10yr track effective area — {zen_label}\n"
    "Effect of trim_isolated  (smoothing_sigma = 0)",
    fontsize=11,
)
plt.tight_layout()
out = OUT_DIR / "effa_trim_isolated.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: smoothing_sigma comparison
# ---------------------------------------------------------------------------
sigmas = [0.0, 0.5, 1.0, 1.3, 2.0]
colors = ["#BBBBBB", "#88CCEE", "#44AA99", "#117733", "#882255"]
labels = [
    "σ = 0  (no smoothing)",
    "σ = 0.5",
    "σ = 1.0",
    "σ = 1.3  (PS-10yr default)",
    "σ = 2.0",
]

fig, ax = plt.subplots(figsize=(7, 5))

# Plot from most-smoothed to least so the raw data is on top of the smooth curves
for sigma, color, label in zip(sigmas[::-1], colors[::-1], labels[::-1]):
    aeff = _effa_col(trim_isolated=True, smoothing_sigma=sigma)
    lw = 2.2 if sigma == 1.3 else 1.4
    ax.plot(e_plot / 1e3, aeff / 1e4, lw=lw, color=color, label=label)

ax.scatter(es_raw[raw_nz] / 1e3, raw_col[raw_nz] / 1e4,
           s=14, zorder=6, color="k", label="tabulated (non-zero)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True energy [TeV]", fontsize=11)
ax.set_ylabel(r"Effective area [m$^2$]", fontsize=11)
ax.set_title(
    f"PS-10yr track effective area — {zen_label}\n"
    "Effect of smoothing_sigma  (trim_isolated = True)",
    fontsize=11,
)
ax.set_xlim(e_plot[0] / 1e3, e_plot[-1] / 1e3)
ax.grid(True, lw=0.4, alpha=0.4, which="both")
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
out = OUT_DIR / "effa_smoothing_sigma.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()
