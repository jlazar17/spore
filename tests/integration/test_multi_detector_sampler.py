import pytest

import numpy as np
import h5py

from spore.conventions import SkyCoordinate
from spore.event_sampling import PointSourceEventSampler
from spore.event_sampling.event import Event
from spore.detector.detector import Detector


@pytest.fixture
def second_detector(tmp_path):
    """A second synthetic detector at a different geographic location."""
    filepath = tmp_path / "detector2.h5"
    n_zen, n_e, n_u_ang, n_u_e = 10, 15, 50, 100
    zeniths = np.linspace(np.pi, 0.0, n_zen)
    energies = np.logspace(3, 6, n_e)
    tabulated_values = np.ones((n_e, n_zen)) * 1e4
    lower_bounds = np.array([[0.0, 0.0]])
    upper_bounds = np.array([[20.0]])
    us_ang = np.linspace(0.0, 1.0, n_u_ang)
    inv_cdfs = np.tile(np.linspace(0.0, 0.5, n_u_ang), (n_e, 1))
    us_e = np.linspace(0.0, 1.0, n_u_e)
    inv_cdf_e = np.linspace(-1.0, 1.0, n_u_e)

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

    config = {
        "properties": {
            "latitude": 36.3,
            "longitude": 16.1,
            "medium": "Water",
        },
        "response": {"detector_response_file": str(filepath)},
    }
    return Detector.from_config(config)


@pytest.fixture
def multi_sampler(detector, second_detector, point_source):
    return PointSourceEventSampler([detector, second_detector], point_source)


@pytest.mark.slow
class TestMultiDetectorSampler:
    def test_len(self, multi_sampler):
        assert len(multi_sampler._samplers) == 2

    def test_nevent_returns_events_from_each_detector(self, multi_sampler):
        events = multi_sampler.sample_events("track", nevent=5)
        ids = {e.detector_id for e in events}
        assert ids == {0, 1}

    def test_nevent_returns_correct_count_per_detector(self, multi_sampler):
        events = multi_sampler.sample_events("track", nevent=7)
        counts = {0: 0, 1: 0}
        for e in events:
            counts[e.detector_id] += 1
        assert counts[0] == 7
        assert counts[1] == 7

    def test_deltat_returns_list(self, multi_sampler):
        from spore.conventions import ureg
        events = multi_sampler.sample_events("cascade", t=60355.0, deltat=30 * ureg.day)
        assert isinstance(events, list)

    def test_all_events_have_integer_detector_id(self, multi_sampler):
        events = multi_sampler.sample_events("track", nevent=5)
        for event in events:
            assert event.detector_id in (0, 1)

    def test_events_are_event_instances(self, multi_sampler):
        events = multi_sampler.sample_events("cascade", nevent=5)
        for event in events:
            assert isinstance(event, Event)

    def test_true_energies_are_positive(self, multi_sampler):
        events = multi_sampler.sample_events("track", nevent=10)
        for event in events:
            assert event.true_energy.magnitude > 0

    def test_reco_energies_are_positive(self, multi_sampler):
        events = multi_sampler.sample_events("cascade", nevent=10)
        for event in events:
            assert event.reco_energy.magnitude > 0

    def test_directions_are_sky_coordinates(self, multi_sampler):
        events = multi_sampler.sample_events("track", nevent=5)
        for event in events:
            assert isinstance(event.true_direction, SkyCoordinate)
            assert isinstance(event.reco_direction, SkyCoordinate)

    def test_both_nevent_and_deltat_raises(self, multi_sampler):
        with pytest.raises(ValueError):
            multi_sampler.sample_events("track", nevent=5, deltat=1e20)

    def test_neither_nevent_nor_deltat_raises(self, multi_sampler):
        with pytest.raises(ValueError):
            multi_sampler.sample_events("track")

    def test_invalid_morphology_raises(self, multi_sampler):
        with pytest.raises(ValueError):
            multi_sampler.sample_events("muon", nevent=5)

    def test_detector_id_defaults_to_empty_string(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, "track")
        assert event.detector_id == ""
