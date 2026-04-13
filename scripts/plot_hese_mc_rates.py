"""
Plot expected HESE 7.5yr event rates broken down by flux component using
the SPORE framework.

Loads the HESE IRF via DetectorResponse, wraps each flux (astrophysical /
atmospheric) in an ExtendedSource, and uses ExtendedSourceEventSampler to
compute expected counts and sample events.  The resulting energy spectra are
plotted as a stacked histogram with step-function lines, overlaid with the
actual observed HESE data.

Four components are shown:
    astro_cascade   astrophysical, cascade channel
    astro_track     astrophysical, track channel
    atmo_cascade    conventional atmospheric, cascade channel
    atmo_track      conventional atmospheric, track channel

Usage
-----
    python scripts/plot_hese_mc_rates.py [--out figures/hese_mc_rates.png]
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spore.conventions import EarthCoordinate, ureg
from spore.detector import Detector
from spore.detector.detector_response import DetectorResponse
from spore.detector.detector import Medium
from spore.source import ExtendedSource
from spore.source.flux import Flux
from spore.event_sampling import ExtendedSourceEventSampler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, "..")
IRF_PATH  = os.path.join(_ROOT, "resources", "hese_7yr_detector_response.h5")
ASTRO_FLUX_PATH = os.path.join(_ROOT, "scratch", "hese_flux.h5")
ATMO_FLUX_PATH  = os.path.join(_ROOT, "resources", "hese_7yr_atmo_flux.h5")
FIG_DIR   = os.path.join(_ROOT, "figures")
_STYLE    = os.path.join(_ROOT, "resources", "paper.mplstyle")

_DEFAULT_DATA_DIR = os.path.expanduser(
    "~/research/CATHODE/HeseCathode/data/"
    "HESE-7-year-data-release-main/HESE-7-year-data-release/resources/data"
)
DATA_DIR = os.environ.get("HESE_DATA_DIR", _DEFAULT_DATA_DIR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIVETIME = ureg.Quantity(7.5 * 365.25, "day")

# Energy bins spanning the HESE 7.5yr sample range (~20 TeV – 2 PeV)
E_BINS_TEV = np.logspace(np.log10(15), np.log10(2e3), 11)   # 10 bins

# Events to sample per morphology for building the energy spectrum shape
N_SAMPLE = 5_000

_ORDER  = ["astro_cascade", "astro_track", "atmo_cascade", "atmo_track"]
_LABELS = {
    "astro_cascade": r"Astro cascade",
    "astro_track":   r"Astro track",
    "atmo_cascade":  r"Atmo cascade",
    "atmo_track":    r"Atmo track",
}

# IceCube location
_ICECUBE = EarthCoordinate(np.radians(-90.0), 0.0)


# ---------------------------------------------------------------------------
# Build SPORE objects
# ---------------------------------------------------------------------------
def _build_samplers():
    """
    Return two ExtendedSourceEventSamplers:
        astro_sampler  — morphologies astro_track, astro_cascade
        atmo_sampler   — morphologies atmo_track,  atmo_cascade
    """
    # --- Load full IRF and split by component ---
    full_response = DetectorResponse.from_config(
        {"detector_response_file": IRF_PATH}
    )

    def _filter(response, prefix):
        return DetectorResponse(
            effective_area={k: v for k, v in response.effective_area.items()
                            if k.startswith(prefix)},
            angular_response={k: v for k, v in response.angular_response.items()
                              if k.startswith(prefix)},
            energy_response={k: v for k, v in response.energy_response.items()
                             if k.startswith(prefix)},
        )

    astro_det = Detector(_ICECUBE, Medium.Ice, _filter(full_response, "astro"))
    atmo_det  = Detector(_ICECUBE, Medium.Ice, _filter(full_response, "atmo"))

    # --- Load fluxes ---
    astro_flux = Flux.from_config({"location": f"{ASTRO_FLUX_PATH}:astrophysical"})
    atmo_flux  = Flux.from_config({"location": f"{ATMO_FLUX_PATH}:conventional"})

    astro_src = ExtendedSource(astro_flux)
    atmo_src  = ExtendedSource(atmo_flux)

    # Steady-state mode: A_eff averaged over diurnal cycle (correct for 7.5 yr)
    print("Building astrophysical sampler ...")
    astro_sampler = ExtendedSourceEventSampler(
        astro_det, astro_src, steady_state=True,
    )
    print("Building atmospheric sampler ...")
    atmo_sampler = ExtendedSourceEventSampler(
        atmo_det, atmo_src, steady_state=True,
    )

    return astro_sampler, atmo_sampler


# ---------------------------------------------------------------------------
# Compute expected counts and energy spectra
# ---------------------------------------------------------------------------
def _build_histograms(astro_sampler, atmo_sampler):
    """
    For each morphology, compute:
        expected  — total expected events over the livetime
        hist      — expected events per reco-energy bin

    Returns
    -------
    expected : dict  morphology → float
    hists    : dict  morphology → ndarray (n_bins,)
    """
    bins_gev = E_BINS_TEV * 1e3

    samplers = {
        "astro": astro_sampler,
        "atmo":  atmo_sampler,
    }
    morph_to_sampler = {
        "astro_cascade": astro_sampler,
        "astro_track":   astro_sampler,
        "atmo_cascade":  atmo_sampler,
        "atmo_track":    atmo_sampler,
    }

    expected = {}
    hists    = {}

    for morph in _ORDER:
        sampler = morph_to_sampler[morph]
        n_exp = sampler.expected_events(morph, LIVETIME)
        expected[morph] = n_exp
        print(f"  {morph:<18}: expected {n_exp:.1f} events")

        if n_exp <= 0:
            hists[morph] = np.zeros(len(E_BINS_TEV) - 1)
            continue

        events = sampler.sample_events(morph, nevent=N_SAMPLE)
        reco_energies = np.array([e.reco_energy.magnitude for e in events])  # GeV

        raw, _ = np.histogram(reco_energies, bins=bins_gev)
        # Scale from N_SAMPLE to expected count
        hists[morph] = raw * (n_exp / N_SAMPLE)

    return expected, hists


# ---------------------------------------------------------------------------
# Load observed data
# ---------------------------------------------------------------------------
def _load_data():
    path = os.path.join(DATA_DIR, "HESE_data.json")
    if not os.path.isfile(path):
        print(f"WARNING: data file not found at {path}, skipping data overlay")
        return None, None
    d = json.load(open(path))
    return np.array(d["recoDepositedEnergy"]), np.array(d["recoMorphology"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot(expected, hists, data_E_gev, data_morph, out_path):
    if os.path.isfile(_STYLE):
        plt.style.use(_STYLE)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors     = [c["color"] for c in prop_cycle]
    color_map  = {k: colors[i] for i, k in enumerate(_ORDER)}

    bin_centres = np.sqrt(E_BINS_TEV[:-1] * E_BINS_TEV[1:])
    bin_widths  = np.diff(np.log10(E_BINS_TEV))

    fig, ax = plt.subplots()

    # --- stacked filled bars ---
    bottom = np.zeros(len(bin_centres))
    for key in _ORDER:
        h = hists[key]
        ax.bar(
            np.log10(bin_centres),
            h,
            width=bin_widths * 0.92,
            bottom=bottom,
            color=color_map[key],
            alpha=0.7,
            label=_LABELS[key],
        )
        bottom = bottom + h

    # --- stacked step lines ---
    bottom_line = np.zeros(len(bin_centres))
    for key in _ORDER:
        top    = bottom_line + hists[key]
        x_step = np.repeat(np.log10(E_BINS_TEV), 2)[1:-1]
        y_step = np.repeat(top, 2)
        ax.plot(x_step, y_step, color=color_map[key], lw=1.5, zorder=5)
        bottom_line = top

    # --- observed data points ---
    if data_E_gev is not None:
        bins_gev = E_BINS_TEV * 1e3
        for mask, marker, label in [
            (data_morph == 0, "o", "Data (cascade)"),
            (data_morph == 1, "s", "Data (track)"),
        ]:
            counts, _ = np.histogram(data_E_gev[mask], bins=bins_gev)
            nonzero = counts > 0
            ax.errorbar(
                np.log10(bin_centres[nonzero]),
                counts[nonzero],
                yerr=np.sqrt(counts[nonzero]),
                fmt=marker,
                color="black",
                ms=5,
                lw=1.2,
                label=label,
                zorder=10,
            )

    ax.set_xlabel(r"$\log_{10}(E_\mathrm{reco}\,/\,\mathrm{TeV})$")
    ax.set_ylabel("Expected events")
    ax.set_xlim(np.log10(E_BINS_TEV[0]), np.log10(E_BINS_TEV[-1]))
    ax.set_xticks(np.arange(
        int(np.ceil(np.log10(E_BINS_TEV[0]))),
        int(np.floor(np.log10(E_BINS_TEV[-1]))) + 1,
    ))
    ax.set_xticklabels([
        f"${v:.0f}$" for v in np.arange(
            int(np.ceil(np.log10(E_BINS_TEV[0]))),
            int(np.floor(np.log10(E_BINS_TEV[-1]))) + 1,
        )
    ])
    ax.legend(loc="upper right", fontsize=11, ncol=2)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        default=os.path.join(FIG_DIR, "hese_mc_rates.png"),
        help="Output file path (default: figures/hese_mc_rates.png)",
    )
    args = parser.parse_args()

    astro_sampler, atmo_sampler = _build_samplers()

    print("\nComputing expected events and sampling spectra ...")
    expected, hists = _build_histograms(astro_sampler, atmo_sampler)

    total = sum(expected.values())
    print(f"\n  Total expected: {total:.1f}")

    data_E, data_morph = _load_data()

    print("\nPlotting ...")
    _plot(expected, hists, data_E, data_morph, args.out)


if __name__ == "__main__":
    main()
