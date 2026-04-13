import numpy as np
import pytest
import h5py

from spore.conventions import SkyCoordinate
from spore.conventions import ureg
from spore.event_sampling.event import Event
from spore.event_sampling import PointSourceEventSampler
from spore.detector.detector import Detector


def _write_minimal_response(path, energies_gev):
    """Write a minimal but complete HDF5 detector response at the given energy grid."""
    n_e = len(energies_gev)
    n_zen, n_u_ang, n_u_e = 5, 30, 60
    zeniths = np.linspace(np.pi, 0.0, n_zen)
    with h5py.File(path, "w") as f:
        for morph in ["track", "cascade"]:
            mg = f.create_group(morph)
            g = mg.create_group("effective_area")
            g.create_dataset("zeniths", data=zeniths)
            g.create_dataset("energies", data=energies_gev)
            g.create_dataset("tabulated_values", data=np.ones((n_e, n_zen)) * 1e4)
            g.create_dataset("lower_bounds", data=np.array([[0.0, 0.0]]))
            g.create_dataset("upper_bounds", data=np.array([[20.0]]))
            g = mg.create_group("angular_response")
            g.create_dataset("energies", data=energies_gev)
            g.create_dataset("us", data=np.linspace(0.0, 1.0, n_u_ang))
            g.create_dataset("inv_cdfs", data=np.tile(np.linspace(0.0, 0.5, n_u_ang), (n_e, 1)))
            g = mg.create_group("energy_resolution")
            g.create_dataset("us", data=np.linspace(0.0, 1.0, n_u_e))
            g.create_dataset("inv_cdf", data=np.linspace(-1.0, 1.0, n_u_e))


@pytest.fixture
def low_threshold_detector(tmp_path):
    """Effective area grid covers 1 GeV – 1 PeV, so the sampler's full energy
    range (100 GeV – 1 PeV) has non-zero effective area at every grid point.
    This exercises the fix for the IndexError when effas contains no zeros."""
    _write_minimal_response(tmp_path / "low.h5", np.logspace(0, 6, 12))  # 1 GeV – 1 PeV
    config = {
        "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
        "response": {"detector_response_file": str(tmp_path / "low.h5")},
    }
    return Detector.from_config(config)


@pytest.fixture
def high_threshold_detector(tmp_path):
    """Effective area grid covers only 10 EeV – 100 EeV, far above the
    sampler's energy range (100 GeV – 1 PeV).  Every call to effa_fxn(e)
    for e in self._es returns 0, giving an all-zero effas array and
    exercising the empty-es guard."""
    _write_minimal_response(tmp_path / "high.h5", np.logspace(10, 11, 8))  # 10 EeV – 100 EeV in GeV
    config = {
        "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
        "response": {"detector_response_file": str(tmp_path / "high.h5")},
    }
    return Detector.from_config(config)


@pytest.mark.slow
class TestPointSourceEventSampler:
    def test_nevent_returns_exact_count(self, point_source_sampler):
        events = point_source_sampler.sample_events("track", nevent=10)
        assert len(events) == 10

    def test_nevent_zero_returns_empty_list(self, point_source_sampler):
        events = point_source_sampler.sample_events("cascade", nevent=0)
        assert events == []

    def test_deltat_returns_list(self, point_source_sampler):
        events = point_source_sampler.sample_events("cascade", t=60355.0, deltat=86400.0 * ureg.s)
        assert isinstance(events, list)

    def test_events_are_event_instances(self, point_source_sampler):
        events = point_source_sampler.sample_events("track", nevent=5)
        for event in events:
            assert isinstance(event, Event)

    def test_true_energies_are_positive(self, point_source_sampler):
        events = point_source_sampler.sample_events("cascade", nevent=20)
        for event in events:
            assert event.true_energy.magnitude > 0

    def test_reco_energies_are_positive(self, point_source_sampler):
        events = point_source_sampler.sample_events("cascade", nevent=20)
        for event in events:
            assert event.reco_energy.magnitude > 0

    def test_directions_are_sky_coordinates(self, point_source_sampler):
        events = point_source_sampler.sample_events("track", nevent=5)
        for event in events:
            assert isinstance(event.true_direction, SkyCoordinate)
            assert isinstance(event.reco_direction, SkyCoordinate)

    def test_track_morphology_stored(self, point_source_sampler):
        events = point_source_sampler.sample_events("track", nevent=10)
        for event in events:
            assert event.morphology == "track"

    def test_cascade_morphology_stored(self, point_source_sampler):
        events = point_source_sampler.sample_events("cascade", nevent=10)
        for event in events:
            assert event.morphology == "cascade"

    def test_true_direction_matches_source_location(self, point_source_sampler, point_source):
        events = point_source_sampler.sample_events("track", nevent=5)
        for event in events:
            assert event.true_direction.declination == pytest.approx(
                point_source.location.declination
            )
            assert event.true_direction.right_ascension == pytest.approx(
                point_source.location.right_ascension
            )

    def test_both_nevent_and_deltat_raises(self, point_source_sampler):
        with pytest.raises(ValueError):
            point_source_sampler.sample_events("track", nevent=5, deltat=100.0)

    def test_neither_nevent_nor_deltat_raises(self, point_source_sampler):
        with pytest.raises(ValueError):
            point_source_sampler.sample_events("track")

    def test_invalid_morphology_raises(self, point_source_sampler):
        with pytest.raises(ValueError):
            point_source_sampler.sample_events("muon", nevent=5)

    def test_result_is_cached_after_first_call(self, point_source_sampler):
        t = 60355.0
        point_source_sampler.sample_events("track", t=t, nevent=1)
        assert (t, "track") in point_source_sampler._cache

    def test_cache_is_reused_on_second_call(self, point_source_sampler):
        t = 60355.0
        point_source_sampler.sample_events("track", t=t, nevent=1)
        spl_first, _ = point_source_sampler._cache[(t, "track")]
        point_source_sampler.sample_events("track", t=t, nevent=1)
        spl_second, _ = point_source_sampler._cache[(t, "track")]
        assert spl_first is spl_second

    def test_different_times_have_separate_cache_entries(self, point_source_sampler):
        t1, t2 = 60355.0, 60356.0
        point_source_sampler.sample_events("track", t=t1, nevent=1)
        point_source_sampler.sample_events("track", t=t2, nevent=1)
        assert (t1, "track") in point_source_sampler._cache
        assert (t2, "track") in point_source_sampler._cache


@pytest.mark.slow
class TestPointSourceEventSamplerEdgeCases:
    """Tests targeting specific bugs in the CDF builder."""

    def test_effa_nonzero_everywhere_does_not_crash(
        self, low_threshold_detector, point_source
    ):
        # Fix 1: when the effective area grid starts below the sampler's energy
        # minimum, effas has no zeros and the old code raised IndexError.
        sampler = PointSourceEventSampler(low_threshold_detector, point_source)
        events = sampler.sample_events("track", nevent=5)
        assert len(events) == 5

    def test_effa_nonzero_everywhere_cache_entry_is_valid(
        self, low_threshold_detector, point_source
    ):
        # The cached spline must not be None when the effective area is
        # non-zero across the entire energy range.
        sampler = PointSourceEventSampler(low_threshold_detector, point_source)
        t = 60355.0
        sampler.sample_events("track", t=t, nevent=1)
        spl, norm = sampler._cache[(t, "track")]
        assert spl is not None
        assert norm > 0

    def test_unobservable_source_returns_empty_list(
        self, high_threshold_detector, point_source
    ):
        # Fix 1: when the detector threshold is above the source energy range,
        # effas is all zeros and the sampler must return [] without crashing.
        sampler = PointSourceEventSampler(high_threshold_detector, point_source)
        events = sampler.sample_events("track", nevent=10)
        assert events == []

    def test_unobservable_source_caches_zero_norm(
        self, high_threshold_detector, point_source
    ):
        # The cached norm must be zero and spline must be None when the source
        # is unobservable, so Poisson sampling gives 0 events.
        sampler = PointSourceEventSampler(high_threshold_detector, point_source)
        t = 60355.0
        sampler.sample_events("track", t=t, nevent=10)
        spl, norm = sampler._cache[(t, "track")]
        assert spl is None
        assert norm == 0.0

    def test_unobservable_source_deltat_returns_empty_list(
        self, high_threshold_detector, point_source
    ):
        # With deltat, Poisson(0) == 0, so the result must also be [].
        sampler = PointSourceEventSampler(high_threshold_detector, point_source)
        events = sampler.sample_events("track", t=60355.0, deltat=365 * ureg.day)
        assert events == []
