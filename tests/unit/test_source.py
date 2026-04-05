import numpy as np
import pytest
from scipy.integrate import quad

from spore.conventions import SkyCoordinate
from spore.conventions.units import units
from spore.physics import Neutrino, neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.source.point_source import PointSource


class TestPowerLaw:
    def test_density_in_range_is_positive(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl.density(1e13) > 0

    def test_density_below_emin_raises(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        with pytest.raises(ValueError):
            pl.density(1e10)

    def test_density_above_emax_raises(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        with pytest.raises(ValueError):
            pl.density(1e16)

    def test_density_at_emin_boundary_is_valid(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl.density(1e11) > 0

    def test_density_at_emax_boundary_is_valid(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl.density(1e15) > 0

    def test_integrates_to_pivot_gamma(self):
        # PowerLaw.density is NOT a unit-normalized PDF. It integrates to
        # pivot^gamma over [emin, emax]; the Flux normalization absorbs this.
        emin, emax, pivot, gamma = 1e11, 1e15, 1e13, 2.0
        pl = PowerLaw(gamma, emin, emax, pivot)
        val, _ = quad(pl.density, emin, emax)
        assert val == pytest.approx(pivot**gamma, rel=1e-4)

    def test_gamma_one_does_not_raise(self):
        pl = PowerLaw(1.0, 1e11, 1e15, 1e13)
        assert pl.density(1e12) > 0

    def test_gamma_one_integrates_to_pivot(self):
        emin, emax, pivot = 1e11, 1e15, 1e13
        pl = PowerLaw(1.0, emin, emax, pivot)
        val, _ = quad(pl.density, emin, emax)
        assert val == pytest.approx(pivot, rel=1e-4)

    def test_emin_emax_stored_correctly(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl.emin == pytest.approx(1e11)
        assert pl.emax == pytest.approx(1e15)

    def test_gamma_stored_correctly(self):
        pl = PowerLaw(2.7, 1e11, 1e15, 1e13)
        assert pl.gamma == pytest.approx(2.7)


class TestFlux:
    def test_call_returns_float(self, powerlaw_dist):
        norm = 1e-18
        normalizations = {nu: norm for nu in neutrinos}
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        result = flux(Neutrino.NuMu, powerlaw_dist.emin * 2)
        assert isinstance(result, float)

    def test_call_returns_positive(self, powerlaw_dist):
        norm = 1e-18
        normalizations = {nu: norm for nu in neutrinos}
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        result = flux(Neutrino.NuMu, powerlaw_dist.emin * 2)
        assert result > 0

    def test_different_normalizations_scale_result(self, powerlaw_dist):
        e = powerlaw_dist.emin * 2
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux1 = Flux({nu: 1e-18 for nu in neutrinos}, distributions)
        flux2 = Flux({nu: 2e-18 for nu in neutrinos}, distributions)
        assert flux2(Neutrino.NuE, e) == pytest.approx(2 * flux1(Neutrino.NuE, e))


class TestPointSource:
    def test_call_returns_positive(self, point_source):
        e = 1e12
        result = point_source(Neutrino.NuMu, e)
        assert result > 0

    def test_location_is_sky_coordinate(self, point_source):
        assert isinstance(point_source.location, SkyCoordinate)

    def test_location_stored_correctly(self):
        from spore.source.flux.distributions.power_laws import PowerLaw
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        normalizations = {nu: 1e-18 for nu in neutrinos}
        distributions = {nu: pl for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        loc = SkyCoordinate(0.5, 1.0)
        src = PointSource(flux, loc)
        assert src.location.declination == pytest.approx(0.5)
        assert src.location.right_ascension == pytest.approx(1.0)
