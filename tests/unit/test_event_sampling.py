import numpy as np
import pytest

from spore.conventions import SkyCoordinate
from spore.event_sampling.event import Event
from spore.event_sampling.utils import smear_truth


class TestEvent:
    def test_construction(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, "cascade")
        assert event.morphology == "cascade"

    def test_fields_stored_correctly(self):
        true_dir = SkyCoordinate(0.1, 0.2)
        reco_dir = SkyCoordinate(0.15, 0.25)
        event = Event(true_dir, reco_dir, 1e12, 9e11, 60355.0, "track")
        assert event.true_energy.magnitude == pytest.approx(1e12)
        assert event.reco_energy.magnitude == pytest.approx(9e11)
        assert event.morphology == "track"

    def test_cascade_morphology_stored(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, "cascade")
        assert event.morphology == "cascade"

    def test_track_morphology_stored(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, "track")
        assert event.morphology == "track"


class TestSmearTruth:
    def test_returns_sky_coordinate_and_float(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        reco_dir, reco_e, _ = smear_truth(sc, 1e13, detector, "track")
        assert isinstance(reco_dir, SkyCoordinate)
        assert isinstance(reco_e, float)

    def test_cascade_morphology_works(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        reco_dir, reco_e, _ = smear_truth(sc, 1e13, detector, "cascade")
        assert isinstance(reco_dir, SkyCoordinate)
        assert reco_e > 0

    def test_reco_energy_is_positive(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        for _ in range(10):
            _, reco_e, _ = smear_truth(sc, 1e13, detector, "track")
            assert reco_e > 0

    def test_reco_direction_is_valid(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        for _ in range(10):
            reco_dir, _, _ = smear_truth(sc, 1e13, detector, "cascade")
            assert -np.pi / 2 <= reco_dir.declination <= np.pi / 2
            assert 0.0 <= reco_dir.right_ascension <= 2 * np.pi

    def test_invalid_morphology_raises(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        with pytest.raises(ValueError):
            smear_truth(sc, 1e13, detector, "muon")

    def test_invalid_morphology_message(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        with pytest.raises(ValueError, match="muon"):
            smear_truth(sc, 1e13, detector, "muon")
