"""Tests for _effa_grid_steady_state and per-species effective area loading."""
import numpy as np
import pytest
import h5py

from spore.event_sampling.utils import _effa_grid_steady_state
from spore.conventions import EarthCoordinate
from spore.detector.detector import Detector
from spore.detector.detector_response.detector_response import effa_spline_from_group


@pytest.fixture
def south_pole():
    return EarthCoordinate(latitude=-np.pi / 2, longitude=0.0)


@pytest.fixture
def equator():
    return EarthCoordinate(latitude=0.0, longitude=0.0)


@pytest.fixture
def flat_effa():
    """Effective area callable that returns a constant 1e4 cm² everywhere."""
    def f(zen, e):
        scalar = np.ndim(zen) == 0
        out = np.ones(np.atleast_1d(np.asarray(zen)).shape) * 1e4
        return float(out[0]) if scalar else out
    return f


class TestEffaGridSteadyState:
    def test_output_shape(self, south_pole, flat_effa):
        decs = np.linspace(-np.pi / 2, np.pi / 2, 5)
        es = np.logspace(3, 6, 8)
        result = _effa_grid_steady_state(decs, south_pole, flat_effa, es, n_ha_samples=20)
        assert result.shape == (5, 8)

    def test_flat_effa_returns_constant(self, south_pole, flat_effa):
        # A flat A_eff should be unchanged by hour-angle averaging.
        decs = np.linspace(-np.pi / 2, np.pi / 2, 4)
        es = np.logspace(3, 6, 6)
        result = _effa_grid_steady_state(decs, south_pole, flat_effa, es, n_ha_samples=50)
        np.testing.assert_allclose(result, 1e4, rtol=1e-10)

    def test_single_declination_squeezed(self, south_pole, flat_effa):
        # Point-source usage: np.array([dec]) → result[0] is a 1-D array.
        dec = np.radians(-30.0)
        es = np.logspace(3, 6, 5)
        result = _effa_grid_steady_state(np.array([dec]), south_pole, flat_effa, es, n_ha_samples=20)
        assert result.shape == (1, 5)
        np.testing.assert_allclose(result[0], 1e4, rtol=1e-10)

    def test_south_pole_dec_independence(self, south_pole, flat_effa):
        # At the South Pole, cos(zen) = -sin(dec), so the averaged A_eff depends
        # only on dec magnitude.  For a flat A_eff all results must be equal.
        decs = np.linspace(-np.pi / 3, np.pi / 3, 7)
        es = np.array([1e4, 1e5])
        result = _effa_grid_steady_state(decs, south_pole, flat_effa, es, n_ha_samples=100)
        np.testing.assert_allclose(result, 1e4, rtol=1e-10)

    def test_n_ha_samples_convergence(self, equator, flat_effa):
        # More HA samples → same answer for flat A_eff.
        decs = np.array([0.0, np.pi / 6])
        es = np.array([1e4])
        r10  = _effa_grid_steady_state(decs, equator, flat_effa, es, n_ha_samples=10)
        r200 = _effa_grid_steady_state(decs, equator, flat_effa, es, n_ha_samples=200)
        np.testing.assert_allclose(r10, r200, rtol=1e-10)

    def test_zero_effa_stays_zero(self, south_pole):
        def zero_fn(zen, e):
            scalar = np.ndim(zen) == 0
            out = np.zeros(np.atleast_1d(np.asarray(zen)).shape)
            return float(out[0]) if scalar else out

        decs = np.array([0.0])
        es = np.array([1e4])
        result = _effa_grid_steady_state(decs, south_pole, zero_fn, es, n_ha_samples=20)
        assert result[0, 0] == 0.0


class TestPerSpeciesEffaLoading:
    """Tests for the 3-D (6, N_E, N_ZEN) per-species tabulated_values path."""

    @pytest.fixture
    def per_species_response_file(self, tmp_path):
        """HDF5 with shape (6, N_E, N_ZEN) tabulated_values, non-uniform across species."""
        filepath = tmp_path / "per_species.h5"
        n_zen, n_e = 5, 8
        zeniths = np.linspace(np.pi, 0.0, n_zen)
        energies = np.logspace(3, 6, n_e)
        # Species 0 has 2× the effective area of the others.
        tabulated = np.ones((6, n_e, n_zen)) * 1e4
        tabulated[0] *= 2.0

        with h5py.File(filepath, "w") as f:
            for morph in ["track", "cascade"]:
                mg = f.create_group(morph)
                g = mg.create_group("effective_area")
                g.create_dataset("zeniths", data=zeniths)
                g.create_dataset("energies", data=energies)
                g.create_dataset("tabulated_values", data=tabulated)
                g = mg.create_group("angular_response")
                g.create_dataset("energies", data=energies)
                g.create_dataset("us", data=np.linspace(0, 1, 20))
                g.create_dataset("inv_cdfs", data=np.tile(np.linspace(0, 0.5, 20), (n_e, 1)))
                g = mg.create_group("energy_resolution")
                g.create_dataset("us", data=np.linspace(0, 1, 30))
                g.create_dataset("inv_cdf", data=np.linspace(-1, 1, 30))
        return str(filepath)

    def test_per_species_loads_as_list(self, per_species_response_file):
        config = {
            "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
            "response": {"detector_response_file": per_species_response_file},
        }
        det = Detector.from_config(config)
        effa = det.response.effective_area["track"]
        assert isinstance(effa, list)
        assert len(effa) == 6

    def test_per_species_species_zero_has_higher_effa(self, per_species_response_file):
        config = {
            "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
            "response": {"detector_response_file": per_species_response_file},
        }
        det = Detector.from_config(config)
        effa = det.response.effective_area["track"]
        zen, e = np.pi / 2, 1e4
        val0 = effa[0](zen, e)
        val1 = effa[1](zen, e)
        assert val0 == pytest.approx(val1 * 2.0, rel=0.05)

    def test_effa_spline_from_group_2d_returns_callable(self, tmp_path):
        """2-D (N_E, N_ZEN) path returns a single callable."""
        filepath = tmp_path / "twod.h5"
        n_zen, n_e = 5, 8
        zeniths = np.linspace(np.pi, 0.0, n_zen)
        energies = np.logspace(3, 6, n_e)
        vals = np.ones((n_e, n_zen)) * 1e4
        with h5py.File(filepath, "w") as f:
            g = f.create_group("effective_area")
            g.create_dataset("zeniths", data=zeniths)
            g.create_dataset("energies", data=energies)
            g.create_dataset("tabulated_values", data=vals)
        with h5py.File(filepath, "r") as f:
            fn = effa_spline_from_group(f["effective_area"])
        assert callable(fn)
        assert fn(np.pi / 2, 1e4) > 0
