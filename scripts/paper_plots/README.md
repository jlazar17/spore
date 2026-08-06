# Paper Plotting Scripts

Scripts in this directory reproduce the figures in the SPORE paper.
Flux models and detector responses are read from `../../resources/`.
External data releases are read from `../../resources/data_releases/`; see the
README there for download links.  Place or symlink them as below.

## Required Data Releases

Before running these scripts, make each release available at the path shown:

### HESE 7.5-year data release

```
ln -s /path/to/HESE-7-year-data-release ../../resources/data_releases/hese_7yr_data_release
```

Expected contents:
- `hese_7yr_data_release/resources/data/HESE_data.json`
- `hese_7yr_data_release/resources/data/HESE_mc_truth.json`
- `hese_7yr_data_release/resources/data/HESE_mc_observable.json`
- `hese_7yr_data_release/resources/data/HESE_mc_flux.json`
- `hese_7yr_data_release/data_loader.py`

Available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TPCILVS

### IceCube 10-year point-source tracks data release

```
ln -s /path/to/dataverse_files ../../resources/data_releases/ps10yr_data_release
```

Expected contents:
- `ps10yr_data_release/uptime/IC86_I_exp.csv` (and IC86_II through IC86_VII)
- `ps10yr_data_release/events/IC86_I_exp.csv` (and IC86_II through IC86_VII)

Available at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DPQNMF

## Resource Files (in `../../resources/`)

These files are read directly from the repository's `resources/` directory:

| File | Description |
|------|-------------|
| `hese_7yr_detector_response.h5` | HESE 7.5-year detector IRF (hierarchical HDF5) |
| `hese_7yr_atmo_flux.h5` | Atmospheric flux model for HESE |
| `hese_7yr_astro_flux.h5` | Astrophysical flux model for HESE |
| `atmo_flux_models.h5` | Atmospheric flux models (Honda, MCEq variants) for PS tracks |
| `ps10yr_combined_flux.h5` | Combined astrophysical + atmospheric flux for PS tracks |
| `paper.mplstyle` | Matplotlib style sheet for paper figures |
| `configs/ps10yr_detector_response.h5` | PS-10yr detector IRF (hierarchical HDF5) |
| `plotting_data.h5` | Pre-computed data cached between figure scripts |

## Workflow

Each script computes data and saves it to `resources/plotting_data.h5`.
The actual figure rendering is done by `paper/plots.py`, which reads from
that file:

```
# 1. Run the data scripts (slow — samples events and builds templates)
python scripts/paper_plots/hese_comparison.py
python scripts/paper_plots/pstracks_roundtrip.py
python scripts/paper_plots/multidetector_skymap.py
python scripts/paper_plots/effective_area.py
python scripts/paper_plots/point_source_demo.py

# 2. Render all figures (fast — reads cached data)
cd paper && python plots.py --datafile ../resources/plotting_data.h5
```

## Scripts

| Script | HDF5 group written | Description |
|--------|--------------------|-------------|
| `hese_comparison.py` | `figure_4` | HESE 7.5-year comparison: energy spectrum and LLH test statistic |
| `pstracks_roundtrip.py` | `figure_5` | PS tracks 10-year round-trip: declination and energy-proxy distributions vs IC86 data |
| `multidetector_skymap.py` | `multidetector_skymap` | (RA, sin δ) samples for IceCube and KM3NeT, instantaneous and diurnal-average |
| `effective_area.py` | `effective_area` | Track effective area vs energy at five declinations for both detectors |
| `point_source_demo.py` | `point_source_demo` | Point source ψ² templates and sampled data |
