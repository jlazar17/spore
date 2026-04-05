# SPORE

**Sampling Pipeline for Observatory Response Estimation**

SPORE is a Python package for simulating neutrino point source signals and backgrounds in one or more neutrino detectors. It is designed for external groups — gravitational wave collaborations, gamma-ray astronomers, and others — who need realistic neutrino detector sensitivity estimates without direct access to proprietary detector simulation chains.

Given tabulated instrument response functions (effective area, angular response, and energy resolution) for any neutrino telescope, SPORE samples realistic event lists including reconstructed energy, direction, and morphology. It supports point sources, extended sources, galactic halo (dark matter) sources, and joint multi-detector analyses.

## Features

- **Selection-agnostic**: works with any published or privately computed IRFs in the standard HDF5 format
- **Three IRF components**: effective area, point spread function (inverse-CDF), energy resolution (inverse-CDF)
- **Four source types**: point source, extended source, galactic halo (NFW/Einasto), multi-detector
- **Config-driven**: detectors and sources can be fully specified via TOML files — no Python required
- **Natural units throughout**: energies in eV, angles in radians, times in seconds via `pint`
- **Fast**: inverse-CDF + PCHIP interpolation avoids rejection sampling; samplers cache expensive integrals across repeated pseudo-experiments

## Installation

SPORE uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
git clone https://github.com/jlazar17/spore.git
cd spore
poetry install
```

To enter the virtual environment:

```bash
poetry shell
```

To use SPORE from a Jupyter notebook, install the Poetry kernel:

```bash
pip install --user poetry-kernel
jupyter notebook   # then select the "Poetry" kernel
```

## Quick Start

### Point source signal (config-driven)

```python
from spore.detector import Detector
from spore.source.point_source import PointSource
from spore.conventions import SkyCoordinate
from spore.conventions.units import units
from spore.source.flux.flux import Flux
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.physics import neutrinos
from spore.event_sampling import PointSourceEventSampler

# Load a detector from an HDF5 response file
detector = Detector.from_toml("resources/configs/icecube_ps10yr.toml")

# Define a power-law point source (NGC 1068-like)
source = PointSource(
    coord=SkyCoordinate(ra=40.67 * units.deg, dec=-0.01 * units.deg),
    flux=Flux(
        distribution=PowerLaw(gamma=3.2),
        norm=1.5e-11 / units.GeV / units.cm**2 / units.s,
        e_min=1e3 * units.GeV,
        e_max=1e6 * units.GeV,
    ),
    neutrino_type=neutrinos.NuMu,
)

# Sample events over a 10-year observation window
sampler = PointSourceEventSampler(source=source, detector=detector)
events = sampler.sample(delta_t=10 * units.year)

print(f"Sampled {len(events)} signal events")
print(f"  E_reco range: {events.reco_energy.min():.2e} – {events.reco_energy.max():.2e}")
```

### Multi-detector joint search

```python
# See examples/multi_detector_source_search.py for a full worked example
# combining an IceCube-like South Pole detector with a Mediterranean detector.
```

### Dark matter from the Galactic halo

```python
# See examples/nfw_halo_sampling.py for NFW profile sampling
# and examples/line_spectrum_halo_sampling.py for line spectra.
```

## Examples

The `examples/` directory contains complete, runnable scripts:

| Script | Description |
|--------|-------------|
| `multi_detector_source_search.py` | Joint IceCube + KM3NeT point source search |
| `nfw_halo_sampling.py` | DM annihilation from the Galactic halo (NFW profile) |
| `line_spectrum_halo_sampling.py` | DM decay line spectrum |
| `extended_source_steady_state.py` | Steady-state extended source |
| `extended_source_transient.py` | Transient extended source |

Run any example from the project root:

```bash
python examples/multi_detector_source_search.py
```

## Detector Response Format

Detector IRFs are stored as HDF5 files with six groups:

```
/track_effective_area/
    energies          [eV]         shape (N_E,)
    zeniths           [rad]        shape (N_zen,)
    tabulated_values  [eV^-2]      shape (N_E, N_zen)
    lower_bounds                   polynomial cut coefficients
    upper_bounds                   polynomial cut coefficients

/track_angular_response/
    energies          [eV]         shape (N_E,)
    quantiles                      shape (N_u,)
    inv_cdfs          [rad]        shape (N_E, N_u)

/track_energy_resolution/
    quantiles                      shape (N_u,)
    inv_cdfs          [dimensionless, ln(E_reco/E_true)]   shape (N_u,)
```

The same structure is repeated for `cascade_*`. See `notebooks/detector_response_format.ipynb` for a full walkthrough and `scripts/build_hese_detector_response.py` for an example of constructing a response file from published MC data.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks covering:

- `spore_overview.ipynb` — high-level introduction and workflow
- `event_sampling_demo.ipynb` — step-by-step sampling walkthrough
- `detector_response_format.ipynb` — HDF5 IRF format reference
- `CDF_sampling.ipynb` — inverse-CDF sampling technique

## Validation

SPORE has been validated against two IceCube data releases:

1. **IceCube 10-year point-source release** (arXiv:2101.09836): atmospheric background event rate and declination distribution reproduced at the ~10% level consistent with atmospheric flux normalisation uncertainty.
2. **HESE 7.5-year data release** (arXiv:2011.03545): predicted event counts (tracks and cascades) compared against 102 observed HESE events using best-fit flux parameters from Table VI.1; goodness-of-fit p-value of 0.62 over 10,000 Poisson pseudo-experiments.

Validation scripts are in `scratch/hese_validation.py` and `scripts/`.

## Citation

If you use SPORE in your work, please cite the accompanying paper (in preparation):

> Lazar, de Wasseige, Wilmet — *SPORE: A Sampling Pipeline for Observatory Response Estimation in Multi-Detector Neutrino Point Source Searches* (2025)

## Authors

- Jeffrey Lazar
- Gwen de Wasseige
- Perrine Wilmet

Centre for Cosmology, Particle Physics and Phenomenology (CP3),
Universite catholique de Louvain, Louvain-la-Neuve, Belgium

## License

SPORE is released under the [GNU Lesser General Public License v3.0 or later](LICENSE).
