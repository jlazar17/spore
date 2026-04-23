"""
Build a spore HDF5 detector response file from the IceCube 10-year
point-source data-release IRF CSV files.

Reads the effective area and joint smearing from the data-release CSVs,
averages across all IC86 seasons, and writes a single HDF5 file in the
spore hierarchical format::

    track/
        effective_area/
            energies         (n_e,)    GeV
            zeniths          (n_dec,)  radians
            tabulated_values (n_e, n_dec)  cm²
            lower_bounds     (1, 1)
            upper_bounds     (1, 1)
        smearing/
            log10e_true_edges  (15,)
            dec_edges          (4,)    degrees — kept for reference
            zenith_edges       (4,)    degrees — primary lookup axis
            log10e_reco_lo     (14, 3, 20)
            log10e_reco_hi     (14, 3, 20)
            psf_lo             (14, 3, 20)
            psf_hi             (14, 3, 20)
            ang_err_lo         (14, 3, 22)
            ang_err_hi         (14, 3, 22)
            fractional_counts  (14, 3, 20, 20, 22)

The resulting file can be loaded with::

    DetectorResponse.from_config({"detector_response_file": "<path>.h5"})

Usage
-----
    python scripts/build_ps10yr_detector_response.py \\
        --irf-dir /path/to/dataverse_files/irfs \\
        --output resources/ps10yr_detector_response.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np

# Make the spore package importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spore.detector.detector_response.detector_response import (
    _SEASON_TO_IRF,
    _parse_aeff_csv,
    _smearing_tables_from_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_SEASONS = [
    "IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI", "IC86_VII",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IRF_DIR = os.path.join(_HERE, "..", "resources", "data_releases",
                                "ps10yr_data_release", "irfs")
_DEFAULT_OUTPUT  = os.path.join(_HERE, "..", "resources", "ps10yr_detector_response.h5")


def _irf_stems(seasons):
    return sorted({_SEASON_TO_IRF.get(s, "IC86_II") for s in seasons})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(irf_dir: str, output_path: str) -> None:
    stems = _irf_stems(_ALL_SEASONS)

    with h5py.File(output_path, "w") as hf:
        grp = hf.require_group("track")

        # ── Effective area ───────────────────────────────────────────────
        aeff_sum = None
        n = 0
        for stem in stems:
            path = os.path.join(irf_dir, f"{stem}_effectiveArea.csv")
            print(f"  Reading A_eff: {os.path.basename(path)}")
            log10e_c, sindec_c, aeff_cm2 = _parse_aeff_csv(path)
            aeff_sum = aeff_cm2.copy() if aeff_sum is None else aeff_sum + aeff_cm2
            n += 1
        aeff_avg = aeff_sum / n

        # Convert from data-release (log10E, sin_dec) axes to the canonical
        # spline format (E_GeV, zenith_rad).  At the South Pole sin(dec) =
        # -cos(zen), so this is a pure axis relabeling — no averaging applied.
        energies_gev = 10.0 ** log10e_c
        zeniths_rad  = np.arccos(-sindec_c)
        lower_bounds = np.array([[log10e_c[0]]])
        upper_bounds = np.array([[log10e_c[-1]]])

        ea_grp = grp.require_group("effective_area")
        ea_grp.create_dataset("energies",         data=energies_gev)
        ea_grp.create_dataset("zeniths",          data=zeniths_rad)
        ea_grp.create_dataset("tabulated_values", data=aeff_avg)
        ea_grp.create_dataset("lower_bounds",     data=lower_bounds)
        ea_grp.create_dataset("upper_bounds",     data=upper_bounds)
        print(f"  Wrote track/effective_area  "
              f"(shape {aeff_avg.shape}, max={aeff_avg.max():.3e} cm²)")

        # ── Joint smearing ───────────────────────────────────────────────
        frac_sum = None
        n = 0
        for stem in stems:
            path = os.path.join(irf_dir, f"{stem}_smearing.csv")
            print(f"  Reading smearing: {os.path.basename(path)}")
            tables = _smearing_tables_from_file(path)
            if frac_sum is None:
                frac_sum     = tables["fractional_counts"].copy()
                etrue_edges  = tables["log10e_true_edges"]
                dec_edges    = tables["dec_edges"]
                zenith_edges = tables["zenith_edges"]
                er_lo, er_hi = tables["log10e_reco_lo"], tables["log10e_reco_hi"]
                psf_lo, psf_hi = tables["psf_lo"],       tables["psf_hi"]
                ae_lo, ae_hi = tables["ang_err_lo"],      tables["ang_err_hi"]
            else:
                frac_sum += tables["fractional_counts"]
            n += 1

        frac_avg    = frac_sum / n
        cell_totals = frac_avg.sum(axis=(-3, -2, -1), keepdims=True)
        frac_avg    = np.where(cell_totals > 0, frac_avg / cell_totals, frac_avg)

        sm_grp = grp.require_group("smearing")
        sm_grp.create_dataset("log10e_true_edges",  data=etrue_edges)
        sm_grp.create_dataset("dec_edges",          data=dec_edges)
        sm_grp.create_dataset("zenith_edges",       data=zenith_edges)
        sm_grp.create_dataset("log10e_reco_lo",     data=er_lo)
        sm_grp.create_dataset("log10e_reco_hi",     data=er_hi)
        sm_grp.create_dataset("psf_lo",             data=psf_lo)
        sm_grp.create_dataset("psf_hi",             data=psf_hi)
        sm_grp.create_dataset("ang_err_lo",         data=ae_lo)
        sm_grp.create_dataset("ang_err_hi",         data=ae_hi)
        sm_grp.create_dataset("fractional_counts",  data=frac_avg)
        print(f"  Wrote track/smearing  "
              f"(frac shape {frac_avg.shape}, averaged over {n} IRF file(s))")

    print(f"\nDone. Written to: {output_path}")
    _print_sizes(output_path)


def _print_sizes(path: str) -> None:
    """Print dataset shapes and sizes inside the HDF5."""
    with h5py.File(path, "r") as hf:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                mb = obj.nbytes / 1e6
                print(f"  {name:55s}  {str(obj.shape):30s}  {mb:.2f} MB")
        print("\nHDF5 contents:")
        hf.visititems(_visit)
    stat = os.stat(path)
    print(f"\nTotal file size: {stat.st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--irf-dir", default=_DEFAULT_IRF_DIR,
        help=f"Path to the data-release irfs/ directory (default: {_DEFAULT_IRF_DIR})",
    )
    p.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help=f"Output HDF5 path (default: {_DEFAULT_OUTPUT})",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    print(f"IRF directory : {args.irf_dir}")
    print(f"Writing HDF5  : {args.output}")
    build(args.irf_dir, args.output)
