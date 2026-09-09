# Scripts

## Build scripts

One-time scripts that generate the HDF5 resource files from raw data releases.
Run these from the repo root. Each script defaults to reading from
`resources/data_releases/` and writing to `resources/`; see the `--help` for
overrides.

### `build_ps10yr_detector_response.py`

Builds `resources/ps10yr_detector_response.h5` from the IceCube 10-year
point-source data release IRF CSV files.

**Requires:** `resources/data_releases/ps10yr_data_release/irfs/`

**Download:** https://icecube.wisc.edu/data-releases/2021/01/all-sky-point-source-icecube-data-years-2008-2018/

Unpack the archive and symlink (or copy) the top-level directory to
`resources/data_releases/ps10yr_data_release`.

```
python scripts/build_ps10yr_detector_response.py
```

---

### `build_ps14yr_detector_response.py`

Builds `resources/configs/ps14yr_detector_response.h5` from the IceCube 14-year
track data release (IceTracks-DR2).

**Requires:** `resources/data_releases/ps14yr_data_release/irfs/`

**Download:** https://doi.org/10.7910/DVN/MMIIZA

Fetch the `irfs/`, `events/` and `uptime/` directories into
`resources/data_releases/ps14yr_data_release/`. Note that `IC86_smearing.csv`
is 817 MB; the first parse takes several minutes and is then cached alongside
the CSV as a `.npz`.

DR2 uses the same column schema as the 10-year release but bins the smearing
matrix in 41 declination bands rather than 3, and 20 rather than 22
angular-error bins per cell. `spore` reads both counts from the file, so no
code change is needed to switch between releases.

```
python scripts/build_ps14yr_detector_response.py
```

---

### `build_hese_detector_response.py`

Builds `resources/hese_7yr_detector_response.h5` from the IceCube HESE
7.5-year Monte Carlo data release JSON files.

**Requires:** `resources/data_releases/hese_7yr_data_release/resources/data/`

**Download:** https://icecube.wisc.edu/data-releases/2020/09/icecube-7-5-year-hese-data-release/

Unpack the archive and symlink (or copy) the top-level
`HESE-7-year-data-release` directory to
`resources/data_releases/hese_7yr_data_release`.

```
python scripts/build_hese_detector_response.py
```

Or point to a custom location:

```
HESE_DATA_DIR=/path/to/HESE-7-year-data-release/resources/data \
    python scripts/build_hese_detector_response.py
```

---

### `build_hese_atmospheric_flux.py`

Builds `resources/hese_7yr_atmo_flux.h5` — the Honda+Gaisser conventional
atmospheric flux reconstructed from the same HESE MC release.

**Requires:** same data release as `build_hese_detector_response.py`.

```
python scripts/build_hese_atmospheric_flux.py
```

---

## Paper plot scripts

Scripts that reproduce figures in the paper. Each caches intermediate results
as HDF5 files on first run; subsequent runs skip the sampling step.

All scripts expect the built HDF5 resource files in `resources/` and the data
releases in `resources/data_releases/` as described above.

| Script | Figure |
|--------|--------|
| `paper_plots/point_source_demo.py` | Point-source sampling demo |
| `paper_plots/pstracks_roundtrip.py` | PS-10yr round-trip (superseded; kept for comparison) |
| `paper_plots/pstracks_roundtrip_dr2.py` | PS-14yr (DR2) round-trip -- the paper figure |
| `paper_plots/multidetector_skymap.py` | Multi-detector skymap |
| `paper_plots/effa_loading_appendix.py` | Effective area loading appendix |
| `paper_plots/irf_validation_appendix.py` | IRF round-trip appendix |
| `paper_plots/effective_area.py` | Effective area figure |
| `paper_plots/energy_grid_blend.py` | Adaptive energy-grid blend |
| `paper_plots/kernel_spectral_mismatch.py` | Sec. 6.2 kernel spectral-mismatch numbers |
| `paper_plots/ngc1068_reference_comparison.py` | Sec. 6.3 comparison against the published NGC 1068 template |
