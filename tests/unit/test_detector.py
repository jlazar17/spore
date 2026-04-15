import numpy as np
import pytest

from spore.detector.detector import Detector, Medium
from spore.detector.detector_response.detector_response import DetectorResponse

class TestMedium:
    def test_ice_exists(self):
        assert Medium.Ice is not None

    def test_water_exists(self):
        assert Medium.Water is not None

    def test_two_members(self):
        assert len(Medium) == 2


class TestDetector:
    def test_from_config_returns_detector(self, detector):
        assert isinstance(detector, Detector)

    def test_medium_is_medium_enum(self, detector):
        assert isinstance(detector.medium, Medium)

    def test_medium_is_ice(self, detector):
        assert detector.medium == Medium.Ice

    def test_response_is_detector_response(self, detector):
        assert isinstance(detector.response, DetectorResponse)


class TestDetectorResponse:
    def test_effective_area_has_track_key(self, detector):
        assert "track" in detector.response.effective_area

    def test_effective_area_has_cascade_key(self, detector):
        assert "cascade" in detector.response.effective_area

    def test_effective_area_values_are_callable(self, detector):
        assert callable(detector.response.effective_area["track"])
        assert callable(detector.response.effective_area["cascade"])

    def test_angular_response_has_track_and_cascade(self, detector):
        assert callable(detector.response.angular_response["track"])
        assert callable(detector.response.angular_response["cascade"])

    def test_energy_response_has_track_and_cascade(self, detector):
        assert callable(detector.response.energy_response["track"])
        assert callable(detector.response.energy_response["cascade"])

    def test_effective_area_returns_nonnegative(self, detector):
        effa = detector.response.effective_area["track"]
        assert effa(np.pi / 2, 1e13) >= 0

    def test_effective_area_returns_zero_outside_energy_range(self, detector):
        effa = detector.response.effective_area["track"]
        # 1e10 eV is well below the synthetic grid minimum of 1e12
        assert effa(np.pi / 2, 1e10) == 0.0

    def test_angular_response_returns_scalar(self, detector):
        ang = detector.response.angular_response["track"]
        result = ang(1e13, 0.5)
        assert np.ndim(result) == 0

    def test_energy_response_returns_scalar(self, detector):
        e_resp = detector.response.energy_response["track"]
        result = e_resp(0.5)
        assert np.ndim(result) == 0


