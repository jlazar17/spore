"""
Build a spore HDF5 detector response file from the IceCube 14-year track
data release (IceTracks-DR2, doi:10.7910/DVN/MMIIZA).

DR2 publishes the same eleven-column smearing schema and the same
effective-area column layout as the 10-year release (IceTracks-DR1), so this
script is a close sibling of ``build_ps10yr_detector_response.py``.  The two
substantive differences are:

  * DR2 unifies all IC86 seasons into a single ``IC86`` IRF, rather than
    publishing one IRF per season.  There is nothing to average over.
  * The smearing matrix is binned in 41 declination bands rather than 3, and
    uses 20 rather than 22 angular-error bins per cell.  ``spore`` reads both
    counts from the file, so no shape is assumed here.

The effective-area binning is identical between the two releases (50 bands
with the same edges, 0.2 dex in energy); DR2 truncates at $10^{8.7}$ GeV
rather than $10^{10}$ GeV.

Output layout::

    meta/
        smoothing_sigma  (float attr)
        trim_isolated    (bool attr)
    track/
        effective_area/
            energies         (n_e,)    GeV
            zeniths          (n_dec,)  radians
            tabulated_values (n_e, n_dec)  cm²
        smearing/
            log10e_true_edges  (15,)
            dec_edges          (42,)   degrees — kept for reference
            zenith_edges       (42,)   degrees — primary lookup axis
            log10e_reco_lo/hi  (14, 41, 20)
            psf_lo/hi          (14, 41, 20)
            ang_err_lo/hi      (14, 41, 20)
            fractional_counts  (14, 41, 20, 20, 20)

Usage
-----
    python scripts/build_ps14yr_detector_response.py \\
        --irf-dir resources/data_releases/ps14yr_data_release/irfs \\
        --output resources/configs/ps14yr_detector_response.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np

# Make the spore package importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spore.detector.detector_response.detector_response import (
    _parse_aeff_csv,
    _smearing_tables_from_file,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IRF_DIR = os.path.join(_HERE, "..", "resources", "data_releases",
                                "ps14yr_data_release", "irfs")
_DEFAULT_OUTPUT = os.path.join(_HERE, "..", "resources", "configs",
                               "ps14yr_detector_response.h5")

# DR2 unifies the IC86 configuration; this is the only stem the round-trip uses.
_STEM = "IC86"

# Matches the value baked into the DR1 response file, so the two are compared
# on equal pre-processing footing rather than differing by a smoothing choice.
SMOOTHING_SIGMA = 1.3
TRIM_ISOLATED = True


def build(irf_dir: str, output_path: str, stem: str = _STEM) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        meta = hf.require_group("meta")
        meta.attrs["smoothing_sigma"] = SMOOTHING_SIGMA
        meta.attrs["trim_isolated"] = TRIM_ISOLATED

        grp = hf.require_group("track")

        # ── Effective area ───────────────────────────────────────────────
        aeff_path = os.path.join(irf_dir, f"{stem}_effectiveArea.tab")
        print(f"  Reading A_eff: {os.path.basename(aeff_path)}")
        log10e_c, sindec_c, aeff_cm2 = _parse_aeff_csv(aeff_path)

        # Same axis relabeling as DR1: at the South Pole sin(dec) = -cos(zen),
        # so this is a pure relabeling with no averaging applied.
        energies_gev = 10.0 ** log10e_c
        zeniths_rad = np.arccos(-sindec_c)

        ea_grp = grp.require_group("effective_area")
        ea_grp.create_dataset("energies", data=energies_gev)
        ea_grp.create_dataset("zeniths", data=zeniths_rad)
        ea_grp.create_dataset("tabulated_values", data=aeff_cm2)
        print(f"  Wrote track/effective_area  "
              f"(shape {aeff_cm2.shape}, max={aeff_cm2.max():.3e} cm²)")

        # ── Joint smearing ───────────────────────────────────────────────
        sm_path = os.path.join(irf_dir, f"{stem}_smearing.csv")
        print(f"  Reading smearing: {os.path.basename(sm_path)} "
              f"({os.path.getsize(sm_path) / 1e6:.0f} MB; first parse is slow)")
        tables = _smearing_tables_from_file(sm_path)

        frac = tables["fractional_counts"]
        # Renormalise each (E_true, dec) cell to unit total, matching the DR1
        # builder.  DR2 cells already sum to ~1 by construction, but cells with
        # no simulated events sum to 0 and must be left alone.
        cell_totals = frac.sum(axis=(-3, -2, -1), keepdims=True)
        frac = np.where(cell_totals > 0, frac / cell_totals, frac)

        sm_grp = grp.require_group("smearing")
        for key in ("log10e_true_edges", "dec_edges", "zenith_edges",
                    "log10e_reco_lo", "log10e_reco_hi",
                    "psf_lo", "psf_hi", "ang_err_lo", "ang_err_hi"):
            sm_grp.create_dataset(key, data=tables[key])
        sm_grp.create_dataset("fractional_counts", data=frac)
        print(f"  Wrote track/smearing  (frac shape {frac.shape})")

    print(f"\nDone. Written to: {output_path}")
    _print_sizes(output_path)


def _print_sizes(path: str) -> None:
    """Print dataset shapes and sizes inside the HDF5."""
    with h5py.File(path, "r") as hf:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:55s}  {str(obj.shape):30s}  {obj.nbytes / 1e6:.2f} MB")
        print("\nHDF5 contents:")
        hf.visititems(_visit)
    print(f"\nTotal file size: {os.stat(path).st_size / 1e6:.1f} MB")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--irf-dir", default=_DEFAULT_IRF_DIR,
                   help=f"Path to the DR2 irfs/ directory (default: {_DEFAULT_IRF_DIR})")
    p.add_argument("--output", default=_DEFAULT_OUTPUT,
                   help=f"Output HDF5 path (default: {_DEFAULT_OUTPUT})")
    p.add_argument("--stem", default=_STEM,
                   help=f"IRF stem to build (default: {_STEM})")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    print(f"IRF directory : {args.irf_dir}")
    print(f"Writing HDF5  : {args.output}")
    build(args.irf_dir, args.output, args.stem)
