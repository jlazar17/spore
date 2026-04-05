import numpy as np
import pytest

from spore.physics import Neutrino, neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.conventions.units import units


class TestFluxDictShapes:
    def test_normalizations_has_six_keys(self, powerlaw_dist):
        normalizations = {nu: 1e-18 for nu in neutrinos}
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        assert len(flux._normalizations) == 6

    def test_distributions_has_six_keys(self, powerlaw_dist):
        normalizations = {nu: 1e-18 for nu in neutrinos}
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        assert len(flux._distributions) == 6

    def test_normalization_keys_are_neutrino_enum(self, powerlaw_dist):
        normalizations = {nu: 1e-18 for nu in neutrinos}
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        for key in flux._normalizations:
            assert isinstance(key, Neutrino)

    def test_distribution_keys_are_neutrino_enum(self, powerlaw_dist):
        normalizations = {nu: 1e-18 for nu in neutrinos}
        distributions = {nu: powerlaw_dist for nu in neutrinos}
        flux = Flux(normalizations, distributions)
        for key in flux._distributions:
            assert isinstance(key, Neutrino)


class TestDetectorResponseStructure:
    def test_effective_area_has_exactly_track_and_cascade(self, detector):
        assert "track" in detector.response.effective_area
        assert "cascade" in detector.response.effective_area

    def test_angular_response_has_track_and_cascade(self, detector):
        assert "track" in detector.response.angular_response
        assert "cascade" in detector.response.angular_response

    def test_energy_response_has_track_and_cascade(self, detector):
        assert "track" in detector.response.energy_response
        assert "cascade" in detector.response.energy_response

    def test_effective_area_callable_accepts_zen_and_energy(self, detector):
        effa = detector.response.effective_area["track"]
        result = effa(np.pi / 2, 1e13)
        assert result is not None

    def test_angular_response_callable_accepts_energy_and_u(self, detector):
        ang = detector.response.angular_response["track"]
        result = ang(1e13, 0.5)
        assert result is not None

    def test_energy_response_callable_accepts_u(self, detector):
        e_resp = detector.response.energy_response["cascade"]
        result = e_resp(0.5)
        assert result is not None

    def test_effective_area_returns_scalar(self, detector):
        effa = detector.response.effective_area["cascade"]
        result = effa(np.pi / 3, 1e13)
        assert np.ndim(result) == 0

    def test_angular_response_returns_scalar(self, detector):
        ang = detector.response.angular_response["cascade"]
        result = ang(1e13, 0.5)
        assert np.ndim(result) == 0

    def test_energy_response_returns_scalar(self, detector):
        e_resp = detector.response.energy_response["track"]
        result = e_resp(0.5)
        assert np.ndim(result) == 0


class TestPowerLawProperties:
    def test_emin_less_than_emax(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl.emin < pl.emax

    def test_pivot_stored(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl._pivot == pytest.approx(1e13)

    def test_norm_is_positive(self):
        pl = PowerLaw(2.0, 1e11, 1e15, 1e13)
        assert pl._norm > 0
