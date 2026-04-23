"""Tests for spore.event_sampling.io — HDF5 round-trip serialization."""

import math
import numpy as np
import pytest
import h5py

from spore.conventions import SkyCoordinate
from spore.event_sampling.event import Event
from spore.event_sampling.io import write_events, read_events, list_groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    true_dec=0.1, true_ra=0.2, reco_dec=0.15, reco_ra=0.25,
    true_e=1e13, reco_e=9e12, time=60355.0, morphology="track",
    detector_id="", ang_err=0.05, zenith=1.2, azimuth=2.3,
):
    return Event(
        true_direction=SkyCoordinate(true_dec, true_ra),
        reco_direction=SkyCoordinate(reco_dec, reco_ra),
        true_energy=true_e,
        reco_energy=reco_e,
        time=time,
        morphology=morphology,
        detector_id=detector_id,
        ang_err=ang_err,
        zenith=zenith,
        azimuth=azimuth,
    )


# ---------------------------------------------------------------------------
# Event.to_dict
# ---------------------------------------------------------------------------

class TestEventToDict:
    def test_all_expected_keys_present(self):
        e = _make_event()
        d = e.to_dict()
        for key in (
            "true_dec", "true_ra", "reco_dec", "reco_ra",
            "true_energy", "reco_energy", "time", "morphology",
            "detector_id", "ang_err", "zenith", "azimuth",
        ):
            assert key in d, f"Missing key: {key}"

    def test_float_fields_are_python_scalars(self):
        e = _make_event()
        d = e.to_dict()
        for key in ("true_dec", "true_ra", "reco_dec", "reco_ra",
                    "true_energy", "reco_energy", "time", "ang_err",
                    "zenith", "azimuth"):
            assert isinstance(d[key], float), f"{key} should be float, got {type(d[key])}"

    def test_values_match_constructor(self):
        e = _make_event(true_dec=0.3, true_ra=1.1, true_e=5e12, zenith=0.7, azimuth=4.1)
        d = e.to_dict()
        assert d["true_dec"]    == pytest.approx(0.3)
        assert d["true_ra"]     == pytest.approx(1.1)
        assert d["true_energy"] == pytest.approx(5e12)
        assert d["zenith"]      == pytest.approx(0.7)
        assert d["azimuth"]     == pytest.approx(4.1)

    def test_zenith_nan_when_not_set(self):
        e = _make_event(zenith=None, azimuth=None)
        d = e.to_dict()
        assert math.isnan(d["zenith"])
        assert math.isnan(d["azimuth"])

    def test_morphology_is_string(self):
        e = _make_event(morphology="cascade")
        assert e.to_dict()["morphology"] == "cascade"

    def test_detector_id_string_preserved(self):
        e = _make_event(detector_id="IceCube")
        assert e.to_dict()["detector_id"] == "IceCube"


# ---------------------------------------------------------------------------
# write_events / read_events round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_single_event_round_trips(self, tmp_path):
        path = str(tmp_path / "events.h5")
        ev = _make_event()
        write_events([ev], path)
        recovered = read_events(path)
        assert len(recovered) == 1
        r = recovered[0]
        assert r.true_direction.declination    == pytest.approx(ev.true_direction.declination)
        assert r.true_direction.right_ascension == pytest.approx(ev.true_direction.right_ascension)
        assert r.reco_direction.declination    == pytest.approx(ev.reco_direction.declination)
        assert r.reco_direction.right_ascension == pytest.approx(ev.reco_direction.right_ascension)
        assert r._true_e   == pytest.approx(ev._true_e)
        assert r._reco_e   == pytest.approx(ev._reco_e)
        assert r.time      == pytest.approx(ev.time)
        assert r.morphology == ev.morphology
        assert r._ang_err  == pytest.approx(ev._ang_err)
        assert r._zenith_rad  == pytest.approx(ev._zenith_rad)
        assert r._azimuth_rad == pytest.approx(ev._azimuth_rad)

    def test_multiple_events_round_trip(self, tmp_path):
        path = str(tmp_path / "events.h5")
        events = [_make_event(true_dec=i * 0.1, time=60355.0 + i) for i in range(5)]
        write_events(events, path)
        recovered = read_events(path)
        assert len(recovered) == 5
        for orig, rec in zip(events, recovered):
            assert rec._true_e == pytest.approx(orig._true_e)
            assert rec.time    == pytest.approx(orig.time)

    def test_empty_events_round_trip(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([], path)
        recovered = read_events(path)
        assert recovered == []

    def test_morphology_preserved(self, tmp_path):
        path = str(tmp_path / "events.h5")
        events = [_make_event(morphology="track"), _make_event(morphology="cascade")]
        write_events(events, path)
        recovered = read_events(path)
        assert recovered[0].morphology == "track"
        assert recovered[1].morphology == "cascade"

    def test_string_detector_id_round_trips(self, tmp_path):
        path = str(tmp_path / "events.h5")
        ev = _make_event(detector_id="KM3NeT")
        write_events([ev], path)
        rec = read_events(path)[0]
        assert rec.detector_id == "KM3NeT"

    def test_integer_detector_id_round_trips(self, tmp_path):
        path = str(tmp_path / "events.h5")
        ev = _make_event(detector_id=42)
        write_events([ev], path)
        rec = read_events(path)[0]
        assert rec.detector_id == 42

    def test_nan_zenith_azimuth_round_trips(self, tmp_path):
        path = str(tmp_path / "events.h5")
        ev = _make_event(zenith=None, azimuth=None)
        write_events([ev], path)
        rec = read_events(path)[0]
        assert math.isnan(rec._zenith_rad)
        assert math.isnan(rec._azimuth_rad)

    def test_write_appends_to_existing_file(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([_make_event()], path, group="signal")
        write_events([_make_event()], path, group="background")
        assert len(read_events(path, group="signal"))     == 1
        assert len(read_events(path, group="background")) == 1

    def test_write_overwrites_same_group(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([_make_event(), _make_event()], path, group="events")
        write_events([_make_event()],               path, group="events")
        assert len(read_events(path, group="events")) == 1

    def test_recovered_events_are_Event_instances(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([_make_event()], path)
        recovered = read_events(path)
        assert all(isinstance(e, Event) for e in recovered)


# ---------------------------------------------------------------------------
# Backward compatibility: files without zenith/azimuth
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def _write_legacy_file(self, tmp_path):
        """Write an HDF5 file in the pre-zenith/azimuth format."""
        path = str(tmp_path / "legacy.h5")
        n = 3
        with h5py.File(path, "w") as f:
            gp = f.create_group("events")
            gp.attrs["n_events"] = n
            gp.create_dataset("true_dec",    data=np.zeros(n))
            gp.create_dataset("true_ra",     data=np.zeros(n))
            gp.create_dataset("reco_dec",    data=np.zeros(n))
            gp.create_dataset("reco_ra",     data=np.zeros(n))
            gp.create_dataset("true_energy", data=np.full(n, 1e13))
            gp.create_dataset("reco_energy", data=np.full(n, 9e12))
            gp.create_dataset("time",        data=np.full(n, 60355.0))
            gp.create_dataset("ang_err",     data=np.full(n, 0.1))
            dt = h5py.string_dtype()
            gp.create_dataset("morphology",  data=np.array(["track"] * n, dtype=object), dtype=dt)
            gp.create_dataset("detector_id", data=np.array([""] * n,      dtype=object), dtype=dt)
        return path

    def test_legacy_file_reads_without_error(self, tmp_path):
        path = self._write_legacy_file(tmp_path)
        events = read_events(path)
        assert len(events) == 3

    def test_legacy_file_zenith_is_nan(self, tmp_path):
        path = self._write_legacy_file(tmp_path)
        for ev in read_events(path):
            assert math.isnan(ev._zenith_rad)

    def test_legacy_file_azimuth_is_nan(self, tmp_path):
        path = self._write_legacy_file(tmp_path)
        for ev in read_events(path):
            assert math.isnan(ev._azimuth_rad)


# ---------------------------------------------------------------------------
# list_groups
# ---------------------------------------------------------------------------

class TestListGroups:
    def test_single_group_listed(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([_make_event()], path, group="signal")
        assert list_groups(path) == ["signal"]

    def test_multiple_groups_listed(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([_make_event()], path, group="signal")
        write_events([_make_event()], path, group="background")
        groups = list_groups(path)
        assert set(groups) == {"signal", "background"}

    def test_empty_events_group_listed(self, tmp_path):
        path = str(tmp_path / "events.h5")
        write_events([], path, group="empty")
        assert "empty" in list_groups(path)
