# SPORE

**S**ampling **P**ipeline for **O**bservatory **R**esponse **E**stimation

SPORE simulates neutrino detector observations from first principles. Given a source flux model and tabulated instrument response functions (IRFs), it produces Monte Carlo event samples with realistic energy and angular smearing — without requiring access to proprietary detector simulation chains.

It supports point sources, spatially extended sources, multiple detectors, and both steady-state (diurnal-average) and transient (fixed-epoch) observation modes. A fully worked introduction is in `examples/spore_quickstart.ipynb`.

---

## Installation

```bash
git clone https://github.com/jlazar17/spore.git
cd spore
pip install .
```

Core dependencies: `numpy`, `scipy`, `astropy`, `h5py`, `pint`.

---

## Quickstart

```python
import spore

# Load a detector from an HDF5 response file
det = spore.Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
    "response":   {"detector_response_file": "resources/configs/ps10yr_detector_response.h5"},
})

# Define a power-law point source
src = spore.PointSource.from_config({
    "flux":     {"norm": 1e-18, "gamma": 2.0, "pivot": 1e5, "emin": 1e2, "emax": 1e7},
    "location": {"declination": -30.0, "right_ascension": 83.8},
})

# Steady-state sampler: effective area averaged over the diurnal cycle
sampler = spore.PointSourceEventSampler(det, src, n_time_samples=100)

one_year = spore.ureg.Quantity(365.25, "day")
print(f"Expected events: {sampler.expected_events('track', one_year):.1f}")

events = sampler.sample_events("track", deltat=one_year, seed=42)
for ev in events[:3]:
    print(f"  E_true={ev.true_energy.to('TeV'):.2f}  E_reco={ev.reco_energy.to('TeV'):.2f}")
```

---

## Core Concepts

### Detector

A `Detector` holds the geographic location and the IRFs of one telescope. IRFs are loaded from an HDF5 file built by the scripts in `scripts/`.

```python
det = spore.Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
    "response":   {"detector_response_file": "path/to/response.h5"},
})
```

The HDF5 file is organised by morphology (e.g. `track`, `cascade`). Each morphology group contains:

| Dataset group | Contents |
|---|---|
| `effective_area` | A_eff(zenith, energy) table in cm² |
| `angular_response` | PSF inverse-CDF over (log energy, quantile) |
| `energy_resolution` | Δ = ln(E_reco/E_true) inverse-CDF over quantile |
| `smearing` *(optional)* | Joint (E_reco, PSF, AngErr) sampler |

The built-in morphologies `"track"` and `"cascade"` are registered by default. Register additional morphologies before loading a response:

```python
spore.Morphology.register("astro_track")
spore.Morphology.register("astro_cascade")
det = spore.Detector.from_config({..., "response": {"detector_response_file": "hese_7yr_detector_response.h5"}})
```

### Sources

**Point source** at a fixed sky location:

```python
src = spore.PointSource.from_config({
    "flux":     {"norm": 1e-18, "gamma": 2.0, "pivot": 1e5, "emin": 1e2, "emax": 1e7},
    "location": {"declination": -30.0, "right_ascension": 83.8},
})
```

Flux units: `norm` is in GeV⁻¹ cm⁻² s⁻¹ per neutrino species; `pivot` and energy limits are in GeV.

**Extended source** from a tabulated flux HDF5 file:

```python
src = spore.ExtendedSource.from_config({"flux": {"location": "path/to/flux.h5:group"}})
```

The HDF5 group must contain:
- `sindecs` — sin(dec) grid, shape `(N_dec,)`
- `energies` — energies in GeV, shape `(N_e,)`
- `fluxes` — shape `(6, N_dec, N_e)` for RA-independent sources, or `(6, N_dec, N_ra, N_e)` for RA-dependent sources

Species ordering: [ν_e, ν̄_e, ν_μ, ν̄_μ, ν_τ, ν̄_τ]. Flux units: GeV⁻¹ cm⁻² s⁻¹ sr⁻¹.

### Samplers

Both `PointSourceEventSampler` and `ExtendedSourceEventSampler` share the same API:

| Method | Description |
|---|---|
| `expected_events(morphology, deltat)` | Mean event count for a given livetime |
| `sample_events(morphology, deltat=..., nevent=..., seed=...)` | Draw a list of `Event` objects |
| `expected_rate(morphology)` | Instantaneous rate as a pint Quantity (1/s) |
| `eddington_correction(morphology, e_bins)` | Reco/true correction factors over energy bins |

**Observation modes:**

| Mode | Constructor argument | When to use |
|---|---|---|
| Snapshot (default) | `steady_state=False` | Short flares; South-Pole detectors |
| Steady-state | `n_time_samples=N` | Long observations; off-pole detectors |

**Sampling modes:**

```python
# Poisson: draw N from Poisson(expected), spread times over [t, t+deltat]
events = sampler.sample_events("track", deltat=one_year)

# Fixed count
events = sampler.sample_events("track", nevent=100)

# Fixed count, times still spread over the window
events = sampler.sample_events("track", nevent=100, deltat=one_year)
```

### Multi-detector

Pass a list of `Detector` objects to either sampler. Each event is tagged with its `detector_id` (integer index into the list).

```python
sampler = spore.PointSourceEventSampler([det_a, det_b], src, n_time_samples=100)
events = sampler.sample_events("track", deltat=one_year)
# ev.detector_id is 0 (det_a) or 1 (det_b)
```

---

## Event fields

| Field | Type | Description |
|---|---|---|
| `true_direction` | `SkyCoordinate` | True neutrino direction (dec, RA in radians) |
| `reco_direction` | `SkyCoordinate` | Reconstructed direction after PSF smearing |
| `true_energy` | `pint.Quantity` | True neutrino energy (GeV) |
| `reco_energy` | `pint.Quantity` | Reconstructed deposited energy (GeV) |
| `time` | `float` | Event time in MJD |
| `morphology` | `str` | Event morphology string |
| `ang_err` | `pint.Quantity` | Per-event angular uncertainty (radians) |
| `detector_id` | `int` or `str` | Detector index in multi-detector mode |

```python
ev.true_energy.to("TeV")           # unit conversion via pint
ev.reco_direction.declination      # radians
ev.to_dict()                       # flat dict for serialisation
```

---

## Saving and loading events

```python
from spore.event_sampling.io import write_events, read_events

write_events(events, "output.h5", group="run_01")
loaded = read_events("output.h5", group="run_01")
```

---

## Examples

| File | Description |
|---|---|
| `examples/spore_quickstart.ipynb` | Full walkthrough: IRF inspection, point source, extended source, sanity checks |
| `examples/extended_source_steady_state.py` | Isotropic E⁻² flux, Mediterranean detector, steady-state mode |
| `examples/extended_source_transient.py` | Gaussian extended source, transient mode |
| `examples/multi_detector_source_search.py` | Joint South Pole + Mediterranean point-source search |
| `examples/ra_dependent_extended_source.py` | RA-dependent extended source (galactic halo) |

Run any script from the project root:

```bash
python examples/extended_source_steady_state.py
```

---

## Building detector responses

Scripts in `scripts/` convert data-release files into SPORE HDF5 response files:

```bash
python scripts/build_ps10yr_detector_response.py
python scripts/build_hese_detector_response.py
python scripts/build_gfu_detector_response.py
```

---

## Project structure

```
spore/
  conventions/       SkyCoordinate, EarthCoordinate, units
  detector/          Detector, DetectorResponse, IRF loaders
  event_sampling/    PointSourceEventSampler, ExtendedSourceEventSampler, Event, I/O
  physics/           Neutrino, Morphology
  source/            PointSource, ExtendedSource, Flux, spectral distributions
resources/
  configs/           Detector response HDF5 files
scripts/             IRF build scripts
examples/            Example scripts and quickstart notebook
tests/
  unit/              Unit and regression tests
  integration/       End-to-end sampler tests
  validation/        Value-range and physics sanity checks
```

---

## Citation

If you use SPORE in your work, please cite the accompanying paper (in preparation):

> Lazar, de Wasseige, Wilmet — *SPORE: A Sampling Pipeline for Observatory Response Estimation in Multi-Detector Neutrino Point Source Searches* (2025)

## Authors

Jeffrey Lazar, Gwen de Wasseige, Perrine Wilmet  
Centre for Cosmology, Particle Physics and Phenomenology (CP3), UCLouvain

## License

LGPL-3.0-or-later. See `LICENSE` for details.
