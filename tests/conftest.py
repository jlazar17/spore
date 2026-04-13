import numpy as np
import pytest
import h5py

from spore.conventions import SkyCoordinate, EarthCoordinate
from spore.conventions import ureg
from spore.physics import Neutrino, neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.source.point_source import PointSource
from spore.detector.detector import Detector
from spore.event_sampling import PointSourceEventSampler


@pytest.fixture
def powerlaw_dist():
    return PowerLaw(gamma=2.0, emin=1e2, emax=1e6, pivot=1e5)  # 100 GeV – 1 PeV


@pytest.fixture
def point_source(powerlaw_dist):
    norm = 1e-18  # GeV⁻¹ cm⁻² s⁻¹
    normalizations = {nu: norm for nu in neutrinos}
    distributions = {nu: powerlaw_dist for nu in neutrinos}
    flux = Flux(normalizations, distributions)
    location = SkyCoordinate(np.radians(-30.0), np.radians(50.0))
    return PointSource(flux, location)


@pytest.fixture
def synthetic_detector_response_file(tmp_path):
    """
    Minimal synthetic detector response HDF5 file.

    Effective area grid starts at 1 TeV (1e12 eV) so that energies below
    that in the sampler's energy grid return 0, giving the CDF builder a
    clean zero-effa prefix to skip over.
    """
    filepath = tmp_path / "detector_response.h5"

    n_zen = 10
    n_e = 15
    n_u_ang = 50
    n_u_e = 100

    # Zeniths stored descending so cos(zeniths) is ascending, as required
    # by RegularGridInterpolator.
    zeniths = np.linspace(np.pi, 0.0, n_zen)
    energies = np.logspace(3, 6, n_e)             # 1 TeV – 1 PeV in GeV
    tabulated_values = np.ones((n_e, n_zen)) * 1e4  # constant 1e4 cm^2

    # Polynomial bounds that accept the full energy range:
    #   lower: linear f(cz)=0  → log10(e) > 0 always True for e ≥ 1 TeV
    #   upper: constant f=20   → log10(e) < 20 always True
    lower_bounds = np.array([[0.0, 0.0]])
    upper_bounds = np.array([[20.0]])

    us_ang = np.linspace(0.0, 1.0, n_u_ang)
    # Each energy row maps u in [0,1] to a deflection angle in [0, 0.5] rad
    inv_cdfs = np.tile(np.linspace(0.0, 0.5, n_u_ang), (n_e, 1))

    us_e = np.linspace(0.0, 1.0, n_u_e)
    inv_cdf_e = np.linspace(-1.0, 1.0, n_u_e)   # log-scale energy smearing

    with h5py.File(filepath, "w") as f:
        for morph in ["track", "cascade"]:
            mg = f.create_group(morph)

            g = mg.create_group("effective_area")
            g.create_dataset("zeniths", data=zeniths)
            g.create_dataset("energies", data=energies)
            g.create_dataset("tabulated_values", data=tabulated_values)
            g.create_dataset("lower_bounds", data=lower_bounds)
            g.create_dataset("upper_bounds", data=upper_bounds)

            g = mg.create_group("angular_response")
            g.create_dataset("energies", data=energies)
            g.create_dataset("us", data=us_ang)
            g.create_dataset("inv_cdfs", data=inv_cdfs)

            g = mg.create_group("energy_resolution")
            g.create_dataset("us", data=us_e)
            g.create_dataset("inv_cdf", data=inv_cdf_e)

    return str(filepath)


@pytest.fixture
def detector(synthetic_detector_response_file):
    config = {
        "properties": {
            "latitude": -90.0,
            "longitude": 0.0,
            "medium": "Ice",
        },
        "response": {
            "detector_response_file": synthetic_detector_response_file,
        },
    }
    return Detector.from_config(config)


@pytest.fixture
def point_source_sampler(detector, point_source):
    return PointSourceEventSampler(detector, point_source)
