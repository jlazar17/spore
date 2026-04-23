"""Integration tests for good run list (GRL) sampling.

Covers:
- GRL passed to sample_events / expected_events on both sampler types
- Event timestamps constrained to good run windows
- Expected event count matches deltat=grl.total_livetime
- Mutual exclusivity of grl and deltat
- Implicit reference epoch (t defaults to grl.t_start)
- Passing a raw path string instead of a GoodRunList object
"""

import numpy as np
import pytest

from spore.conventions import ureg
from spore.physics import neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.source.extended_source import ExtendedSource
from spore.event_sampling import PointSourceEventSampler, ExtendedSourceEventSampler
from spore.event_sampling.good_run_list import GoodRunList


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extended_source(powerlaw_dist):
    norm = 1e-18
    flux = Flux(
        {nu: norm for nu in neutrinos},
        {nu: powerlaw_dist for nu in neutrinos},
    )
    return ExtendedSource(flux)


@pytest.fixture
def extended_source_sampler(detector, extended_source):
    return ExtendedSourceEventSampler(detector, extended_source, n_dec=5, n_ra=5, n_e=5)


@pytest.fixture
def grl():
    """A small GRL with two clearly separated runs."""
    return GoodRunList(np.array([
        [60355.0, 60365.0],   # 10 days
        [60380.0, 60395.0],   # 15 days
    ]))


@pytest.fixture
def grl_file(tmp_path):
    """GRL written as an IceCube-style uptime CSV."""
    runs = np.array([[60355.0, 60365.0], [60380.0, 60395.0]])
    path = tmp_path / "IC86_I_exp.csv"
    np.savetxt(path, runs, header="MJD_start[days]  MJD_stop[days]", comments="# ")
    return str(path)


# ---------------------------------------------------------------------------
# PointSourceEventSampler + GRL
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPointSourceSamplerGRL:
    def test_grl_returns_list(self, point_source_sampler, grl):
        events = point_source_sampler.sample_events("track", grl=grl, seed=0)
        assert isinstance(events, list)

    def test_event_times_within_good_runs(self, point_source_sampler, grl):
        events = point_source_sampler.sample_events("track", grl=grl, nevent=50, seed=1)
        for ev in events:
            in_run = any(r[0] <= ev.time <= r[1] for r in grl.runs)
            assert in_run, f"event time {ev.time} outside all good runs"

    def test_no_events_in_gap(self, point_source_sampler, grl):
        events = point_source_sampler.sample_events("track", grl=grl, nevent=200, seed=2)
        gap_lo, gap_hi = 60365.0, 60380.0
        times = [ev.time for ev in events]
        assert not any(gap_lo < t < gap_hi for t in times)

    def test_expected_events_matches_total_livetime(self, point_source_sampler, grl):
        via_grl    = point_source_sampler.expected_events("track", grl=grl)
        via_deltat = point_source_sampler.expected_events(
            "track", deltat=grl.total_livetime
        )
        assert via_grl == pytest.approx(via_deltat)

    def test_grl_and_deltat_together_raises(self, point_source_sampler, grl):
        with pytest.raises(ValueError, match="at most one"):
            point_source_sampler.sample_events(
                "track", grl=grl, deltat=ureg.Quantity(10.0, "day")
            )

    def test_expected_events_grl_and_deltat_together_raises(self, point_source_sampler, grl):
        with pytest.raises(ValueError, match="at most one"):
            point_source_sampler.expected_events(
                "track", grl=grl, deltat=ureg.Quantity(10.0, "day")
            )

    def test_t_defaults_to_grl_t_start(self, point_source_sampler, grl):
        events = point_source_sampler.sample_events("track", grl=grl, nevent=50, seed=3)
        # All times should be >= grl.t_start
        assert all(ev.time >= grl.t_start for ev in events)

    def test_path_string_accepted_in_place_of_grl_object(
        self, point_source_sampler, grl_file, grl
    ):
        # Passing a raw path string should work identically to a GoodRunList.
        events = point_source_sampler.sample_events("track", grl=grl_file, nevent=10, seed=4)
        assert isinstance(events, list)
        for ev in events:
            in_run = any(r[0] <= ev.time <= r[1] for r in grl.runs)
            assert in_run

    def test_expected_events_path_string(self, point_source_sampler, grl_file, grl):
        via_path = point_source_sampler.expected_events("track", grl=grl_file)
        via_obj  = point_source_sampler.expected_events("track", grl=grl)
        assert via_path == pytest.approx(via_obj)


# ---------------------------------------------------------------------------
# ExtendedSourceEventSampler + GRL
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestExtendedSourceSamplerGRL:
    def test_grl_returns_list(self, extended_source_sampler, grl):
        events = extended_source_sampler.sample_events("track", grl=grl, seed=0)
        assert isinstance(events, list)

    def test_event_times_within_good_runs(self, extended_source_sampler, grl):
        events = extended_source_sampler.sample_events("track", grl=grl, nevent=50, seed=5)
        for ev in events:
            in_run = any(r[0] <= ev.time <= r[1] for r in grl.runs)
            assert in_run, f"event time {ev.time} outside all good runs"

    def test_expected_events_matches_total_livetime(self, extended_source_sampler, grl):
        via_grl    = extended_source_sampler.expected_events("track", grl=grl)
        via_deltat = extended_source_sampler.expected_events(
            "track", deltat=grl.total_livetime
        )
        assert via_grl == pytest.approx(via_deltat)

    def test_grl_and_deltat_together_raises(self, extended_source_sampler, grl):
        with pytest.raises(ValueError, match="at most one"):
            extended_source_sampler.sample_events(
                "track", grl=grl, deltat=ureg.Quantity(10.0, "day")
            )
