"""Quantify the spectral mismatch in the released smearing kernel (Sec. 6.2).

The IceCube point-source releases tabulate P(E_reco, PSF, AngErr | E_nu, dec)
in true-energy bins half a decade wide.  Within each bin the kernel is a single
average, weighted by the *simulation* spectrum used to produce the release.  A
user who folds a different spectrum through those tables is therefore applying
each kernel at the wrong mean true energy, and the size of that error depends on
how much the folded spectrum differs from the simulation one.

This script measures the effect directly.  For each true-energy bin it computes

    <log10 E_nu>  weighted by  A_eff(E) * Phi(E) * E dlnE

for a simulation-like E^-2 reference and for several folded spectra, and reports
the offset between them.  The numbers quoted in Sec. 6.2 come from here.

The point of the comparison is that the offset is a property of the *pair*
(response, flux), not of the response alone: a hard spectrum folded through the
same tables sees a much smaller shift than the conventional atmospheric flux.

Usage
-----
    python scripts/paper_plots/kernel_spectral_mismatch.py
"""

import os

import h5py
import numpy as np

from spore.detector.detector import Detector
from spore.physics import neutrinos
from spore.source import ExtendedSource

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(HERE, "..", "..")
RESPONSE = os.path.join(REPO, "resources", "configs", "ps14yr_detector_response.h5")
ATM_H5 = os.path.join(REPO, "resources", "atmo_flux_models.h5")
ATMO_FLUX_MODEL = "mceq_h4a_sibyll23d"

# Representative northern-sky declination.  The conclusion is not sensitive to
# this choice; the atmospheric spectrum's steepness is what drives the offset.
DEC_DEG = 30.0

# Upper edge of the tabulated atmospheric flux.
E_MAX = 1e6

# Spectral index the release's simulation weighting is taken to approximate.
GAMMA_SIM = 2.0

# Comparison spectra.  E^-2.87 is the astrophysical index used elsewhere in the
# paper; E^-2.5 is an intermediate case.
GAMMA_COMPARE = (2.5, 2.87)

# Bins below this log10(E/GeV) define the "low energy" average that Sec. 6.2
# quotes, i.e. the region where the discrepancy appears.
LOW_E_CUT = 3.5


def _mean_log10e(lo, hi, weight, n=400):
    """<log10 E> within [lo, hi] in log10(E/GeV) under ``weight(E)``."""
    hi = min(hi, np.log10(E_MAX))
    if hi <= lo:
        return np.nan
    x = np.linspace(lo, hi, n)
    e = 10.0 ** x
    w = weight(e) * e * np.log(10.0)      # convert to d/dlog10E
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.nan
    return float(np.trapezoid(w * x, x) / np.trapezoid(w, x))


def main():
    det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0,
                       "depth": 1945, "medium": "Ice"},
        "response": {"detector_response_file": RESPONSE},
    })
    effa = det.response.effective_area["track"]
    atmo = ExtendedSource.from_config(
        {"flux": {"location": f"{ATM_H5}:{ATMO_FLUX_MODEL}"}})

    with h5py.File(RESPONSE) as hf:
        edges = hf["track/smearing/log10e_true_edges"][:]

    dec_rad = np.radians(DEC_DEG)
    zen = np.arccos(-np.sin(dec_rad))

    def aeff(e):
        return np.asarray(effa(np.full(len(e), zen), e), float)

    def atmo_flux(e):
        e = np.clip(e, 100.0, E_MAX)
        return np.asarray(
            atmo(neutrinos[2], e, dec_rad, 0.0) + atmo(neutrinos[3], e, dec_rad, 0.0),
            float,
        )

    labels = ["atmospheric"] + [f"E^-{g}" for g in GAMMA_COMPARE]
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo >= np.log10(E_MAX):
            break
        ref = _mean_log10e(lo, hi, lambda e: aeff(e) * e ** -GAMMA_SIM)
        vals = [_mean_log10e(lo, hi, lambda e: aeff(e) * atmo_flux(e))]
        vals += [_mean_log10e(lo, hi, lambda e, g=g: aeff(e) * e ** -g)
                 for g in GAMMA_COMPARE]
        if np.isnan(ref) or any(np.isnan(v) for v in vals):
            continue
        rows.append((lo, hi, [v - ref for v in vals]))

    hdr = "  ".join(f"{l:>11s}" for l in labels)
    print(f"Offset in <log10(E_nu/GeV)> relative to an E^-{GAMMA_SIM:g} "
          f"simulation weighting, at dec = {DEC_DEG:g} deg\n")
    print(f"{'true-energy bin':>18s}  {hdr}")
    for lo, hi, offs in rows:
        print(f"  [{lo:.2f}, {hi:.2f}]    " + "  ".join(f"{o:+11.3f}" for o in offs))

    low = [offs for lo, _, offs in rows if lo < LOW_E_CUT]
    means = [float(np.mean([o[k] for o in low])) for k in range(len(labels))]
    print(f"\nMean below 10^{LOW_E_CUT:g} GeV:")
    for label, m in zip(labels, means):
        print(f"  {label:>12s}: {m:+.3f} dex")
    print(f"\nAtmospheric events sit {100 * (1 - 10 ** means[0]):.0f}% lower in "
          f"energy than the kernel supplied for them.")
    for label, m in zip(labels[1:], means[1:]):
        print(f"Atmospheric offset is {means[0] / m:.1f}x the {label} case.")


if __name__ == "__main__":
    main()
