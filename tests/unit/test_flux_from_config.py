"""Tests for Flux.from_config() — power-law and HDF5 tabulated formats."""

import numpy as np
import pytest
import h5py

from spore.source.flux.flux import Flux
from spore.physics import neutrinos


# ---------------------------------------------------------------------------
# Fixtures for HDF5-backed flux files
# ---------------------------------------------------------------------------

@pytest.fixture
def flux_1d_file(tmp_path):
    """Minimal 1-D (energy-only) HDF5 flux file."""
    n_e = 20
    es = np.logspace(2, 6, n_e)       # 100 GeV – 1 PeV
    # Uniform non-zero flux for each of the 6 species
    fluxes = np.ones((6, n_e)) * 1e-18
    path = str(tmp_path / "flux_1d.h5")
    with h5py.File(path, "w") as f:
        gp = f.create_group("test")
        gp.create_dataset("energies", data=es)
        gp.create_dataset("fluxes",   data=fluxes)
    return path


@pytest.fixture
def flux_2d_file(tmp_path):
    """Minimal 2-D (energy × declination) HDF5 flux file."""
    n_sd, n_e = 10, 20
    sindecs = np.linspace(-1.0, 1.0, n_sd)
    es      = np.logspace(2, 6, n_e)
    # Simple separable flux, non-zero for all species
    fluxes = np.ones((6, n_sd, n_e)) * 1e-18
    path = str(tmp_path / "flux_2d.h5")
    with h5py.File(path, "w") as f:
        gp = f.create_group("test")
        gp.create_dataset("sindecs",  data=sindecs)
        gp.create_dataset("energies", data=es)
        gp.create_dataset("fluxes",   data=fluxes)
    return path


@pytest.fixture
def flux_3d_file(tmp_path):
    """Minimal 3-D (energy × declination × RA) HDF5 flux file."""
    n_sd, n_ra, n_e = 6, 8, 12
    sindecs = np.linspace(-1.0, 1.0, n_sd)
    ras     = np.linspace(0.0, 2 * np.pi, n_ra, endpoint=False)
    es      = np.logspace(2, 6, n_e)
    fluxes  = np.ones((6, n_sd, n_ra, n_e)) * 1e-18
    path = str(tmp_path / "flux_3d.h5")
    with h5py.File(path, "w") as f:
        gp = f.create_group("test")
        gp.create_dataset("sindecs",  data=sindecs)
        gp.create_dataset("ras",      data=ras)
        gp.create_dataset("energies", data=es)
        gp.create_dataset("fluxes",   data=fluxes)
    return path


# ---------------------------------------------------------------------------
# Power-law config
# ---------------------------------------------------------------------------

class TestFluxFromConfigPowerLaw:
    _base = {
        "gamma": 2.0,
        "emin":  1e2,
        "emax":  1e6,
        "pivot": 1e4,
        "norm_per_species": 1e-18,
    }

    def test_returns_flux_instance(self):
        flux = Flux.from_config(self._base)
        assert isinstance(flux, Flux)

    def test_callable_for_each_species(self):
        flux = Flux.from_config(self._base)
        for nu in neutrinos:
            val = flux(nu, 1e4, 0.0, 0.0)
            assert isinstance(val, float)

    def test_flux_positive(self):
        flux = Flux.from_config(self._base)
        for nu in neutrinos:
            assert flux(nu, 1e4, 0.0, 0.0) > 0.0

    def test_e_min_gev_matches_config(self):
        flux = Flux.from_config(self._base)
        assert flux.e_min_gev == pytest.approx(1e2)

    def test_e_max_gev_matches_config(self):
        flux = Flux.from_config(self._base)
        assert flux.e_max_gev == pytest.approx(1e6)

    def test_norm_key_alias_works(self):
        cfg = dict(self._base)
        del cfg["norm_per_species"]
        cfg["norm"] = 1e-18
        flux = Flux.from_config(cfg)
        assert isinstance(flux, Flux)

    def test_normalization_kwarg_applied(self):
        flux1 = Flux.from_config(self._base, normalization=1.0)
        flux2 = Flux.from_config(self._base, normalization=2.0)
        nu = neutrinos[0]
        assert flux2(nu, 1e4, 0.0, 0.0) == pytest.approx(
            2.0 * flux1(nu, 1e4, 0.0, 0.0), rel=1e-6
        )

    def test_normalization_setter_scales_flux(self):
        flux = Flux.from_config(self._base)
        nu = neutrinos[0]
        val_before = flux(nu, 1e4, 0.0, 0.0)
        flux.normalization = 3.0
        assert flux(nu, 1e4, 0.0, 0.0) == pytest.approx(3.0 * val_before, rel=1e-6)

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            Flux.from_config({"bad_key": 1.0})

    def test_uses_ra_is_false_for_powerlaw(self):
        flux = Flux.from_config(self._base)
        assert flux.uses_ra is False

    def test_gamma_one_handled(self):
        cfg = dict(self._base)
        cfg["gamma"] = 1.0
        flux = Flux.from_config(cfg)
        assert isinstance(flux, Flux)
        assert flux(neutrinos[0], 1e4, 0.0, 0.0) > 0.0


# ---------------------------------------------------------------------------
# 1-D tabulated HDF5
# ---------------------------------------------------------------------------

class TestFluxFromConfig1D:
    def test_returns_flux_instance(self, flux_1d_file):
        flux = Flux.from_config({"location": f"{flux_1d_file}:test"})
        assert isinstance(flux, Flux)

    def test_callable_returns_positive(self, flux_1d_file):
        flux = Flux.from_config({"location": f"{flux_1d_file}:test"})
        for nu in neutrinos:
            assert flux(nu, 1e4, 0.0, 0.0) > 0.0

    def test_e_bounds_stored(self, flux_1d_file):
        flux = Flux.from_config({"location": f"{flux_1d_file}:test"})
        assert flux.e_min_gev == pytest.approx(1e2, rel=1e-3)
        assert flux.e_max_gev == pytest.approx(1e6, rel=1e-3)

    def test_uses_ra_is_false(self, flux_1d_file):
        flux = Flux.from_config({"location": f"{flux_1d_file}:test"})
        assert flux.uses_ra is False

    def test_normalization_kwarg_scales_result(self, flux_1d_file):
        flux1 = Flux.from_config({"location": f"{flux_1d_file}:test"}, normalization=1.0)
        flux2 = Flux.from_config({"location": f"{flux_1d_file}:test"}, normalization=5.0)
        nu = neutrinos[0]
        assert flux2(nu, 1e4, 0.0, 0.0) == pytest.approx(
            5.0 * flux1(nu, 1e4, 0.0, 0.0), rel=1e-6
        )


# ---------------------------------------------------------------------------
# 2-D tabulated HDF5
# ---------------------------------------------------------------------------

class TestFluxFromConfig2D:
    def test_returns_flux_instance(self, flux_2d_file):
        flux = Flux.from_config({"location": f"{flux_2d_file}:test"})
        assert isinstance(flux, Flux)

    def test_callable_returns_positive(self, flux_2d_file):
        flux = Flux.from_config({"location": f"{flux_2d_file}:test"})
        for nu in neutrinos:
            assert flux(nu, 1e4, 0.0, 0.0) > 0.0

    def test_uses_ra_is_false(self, flux_2d_file):
        flux = Flux.from_config({"location": f"{flux_2d_file}:test"})
        assert flux.uses_ra is False

    def test_e_bounds_stored(self, flux_2d_file):
        flux = Flux.from_config({"location": f"{flux_2d_file}:test"})
        assert flux.e_min_gev == pytest.approx(1e2, rel=1e-3)
        assert flux.e_max_gev == pytest.approx(1e6, rel=1e-3)


# ---------------------------------------------------------------------------
# 3-D tabulated HDF5
# ---------------------------------------------------------------------------

class TestFluxFromConfig3D:
    def test_returns_flux_instance(self, flux_3d_file):
        flux = Flux.from_config({"location": f"{flux_3d_file}:test"})
        assert isinstance(flux, Flux)

    def test_uses_ra_is_true(self, flux_3d_file):
        flux = Flux.from_config({"location": f"{flux_3d_file}:test"})
        assert flux.uses_ra is True

    def test_callable_returns_positive(self, flux_3d_file):
        flux = Flux.from_config({"location": f"{flux_3d_file}:test"})
        for nu in neutrinos:
            assert flux(nu, 1e4, 0.0, np.pi) > 0.0

    def test_e_bounds_stored(self, flux_3d_file):
        flux = Flux.from_config({"location": f"{flux_3d_file}:test"})
        assert flux.e_min_gev == pytest.approx(1e2, rel=1e-3)
        assert flux.e_max_gev == pytest.approx(1e6, rel=1e-3)
