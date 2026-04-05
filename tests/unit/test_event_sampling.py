import numpy as np
import pytest

from spore.conventions import SkyCoordinate
from spore.event_sampling.event import Event
from spore.event_sampling.metropolis_hastings import metropolis_hastings
from spore.event_sampling.utils import smear_truth


class TestEvent:
    def test_construction(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, 1)
        assert event.morphology == 1

    def test_fields_stored_correctly(self):
        true_dir = SkyCoordinate(0.1, 0.2)
        reco_dir = SkyCoordinate(0.15, 0.25)
        event = Event(true_dir, reco_dir, 1e12, 9e11, 60355.0, 2)
        assert event.true_energy.magnitude == pytest.approx(1e12)
        assert event.reco_energy.magnitude == pytest.approx(9e11)
        assert event.morphology == 2

    def test_cascade_morphology_id_is_one(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, 1)
        assert event.morphology == 1

    def test_track_morphology_id_is_two(self):
        sc = SkyCoordinate(0.0, 0.0)
        event = Event(sc, sc, 1e12, 1e12, 60355.0, 2)
        assert event.morphology == 2


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


class TestMetropolisHastings:
    def test_output_shape(self):
        target = lambda x: 1.0
        x0 = np.array([0.5, 1.0, np.log(1e13)])
        bounds = [(0.0, 1.0), (0.0, 2 * np.pi), (np.log(1e11), np.log(1e15))]
        samples = metropolis_hastings(target, x0, bounds, size=50)
        assert samples.shape == (50, 3)

    def test_samples_within_bounds(self):
        bounds = [(0.0, 1.0), (0.0, 2 * np.pi), (np.log(1e11), np.log(1e15))]
        target = lambda x: 1.0
        x0 = np.array([0.5, 1.0, np.log(1e13)])
        samples = metropolis_hastings(target, x0, bounds, size=100)
        for i, (lb, ub) in enumerate(bounds):
            assert np.all(samples[:, i] >= lb)
            assert np.all(samples[:, i] <= ub)

    def test_uniform_target_mixes_well(self):
        # With a uniform target all proposals are accepted, so the chain
        # performs a random walk.  Starting from (0.5, 0.5) it should explore
        # a broad region of the space within 200 steps.
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        target = lambda x: 1.0
        x0 = np.array([0.5, 0.5])
        samples = metropolis_hastings(target, x0, bounds, size=200)
        # Chain should visit both halves of the unit square in at least one dim.
        assert np.any(samples[:, 0] < 0.4)
        assert np.any(samples[:, 0] > 0.6)

    # ------------------------------------------------------------------
    # Tests for bug fixes
    # ------------------------------------------------------------------

    def test_size_zero_returns_empty_array(self):
        # Fix 2 (upstream): size=0 must return shape (0, n_dims), not crash.
        x0 = np.array([0.5, 0.5])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        samples = metropolis_hastings(lambda x: 1.0, x0, bounds, size=0)
        assert samples.shape == (0, 2)

    def test_zero_density_initial_state_does_not_get_stuck(self):
        # Fix 5: when target(x0) == 0 the chain must escape rather than freeze.
        # Target is 1 for x > 0.5, 0 otherwise.  Start in the dead zone.
        target = lambda x: 1.0 if x[0] > 0.5 else 0.0
        x0 = np.array([0.1])
        bounds = [(0.0, 1.0)]
        samples = metropolis_hastings(target, x0, bounds, size=500)
        # After 500 steps the chain should have crossed into the live region.
        assert np.any(samples[:, 0] > 0.5)

    def test_zero_density_at_both_states_does_not_produce_nan(self):
        # Fix 5: 0/0 must not silently produce NaN samples.
        # Target is zero everywhere except in a tiny sliver.
        target = lambda x: 1.0 if 0.49 < x[0] < 0.51 else 0.0
        x0 = np.array([0.0])  # starts in zero-density region
        bounds = [(0.0, 1.0)]
        samples = metropolis_hastings(target, x0, bounds, size=200)
        assert not np.any(np.isnan(samples))

    def test_local_proposal_step_size_limits_jumps(self):
        # Fix 4: with a small explicit step size, consecutive samples must
        # differ by at most step_size in each dimension.
        step = 0.05
        x0 = np.array([0.5])
        bounds = [(0.0, 1.0)]
        samples = metropolis_hastings(
            lambda x: 1.0, x0, bounds, size=100, step_size=np.array([step])
        )
        jumps = np.abs(np.diff(samples[:, 0]))
        assert np.all(jumps <= step + 1e-12)

    def test_local_proposal_concentrates_near_peaked_target(self):
        # Fix 4: old independent-uniform sampler would rarely accept proposals
        # near a narrow peak.  With local proposals, samples should concentrate
        # in the high-density region within a reasonable number of steps.
        # Target: 1 if x in [0.45, 0.55], else 0 (10 % of range).
        target = lambda x: 1.0 if 0.45 <= x[0] <= 0.55 else 0.0
        x0 = np.array([0.5])
        bounds = [(0.0, 1.0)]
        # Default step (~10 % of range = 0.1) will cross the live region easily.
        samples = metropolis_hastings(target, x0, bounds, size=300)
        fraction_in_peak = np.mean((samples[:, 0] >= 0.45) & (samples[:, 0] <= 0.55))
        # Chain should spend the majority of time in the live region.
        assert fraction_in_peak > 0.5

    def test_reflection_avoids_boundary_accumulation(self):
        # Clipping caused artificial pile-up at the boundary; reflection should
        # not.  With a uniform target the chain should distribute roughly
        # uniformly — boundary bins must not be overrepresented.
        x0 = np.array([0.5])
        bounds = [(0.0, 1.0)]
        samples = metropolis_hastings(lambda x: 1.0, x0, bounds, size=2000)
        # Fraction of samples at the exact lower boundary must be negligible.
        at_lower = np.sum(samples[:, 0] == 0.0)
        assert at_lower < 10  # should be essentially zero for reflected chain

    def test_periodic_dimension_wraps_not_accumulates(self):
        # A periodic dimension (e.g. RA) must wrap around rather than pile up
        # at 0 or 2π.  With a uniform target the chain should visit both ends
        # of the range without accumulating at either boundary.
        x0 = np.array([0.1])
        bounds = [(0.0, 1.0)]
        samples = metropolis_hastings(
            lambda x: 1.0, x0, bounds, size=2000, periodic=[True],
            step_size=np.array([0.3])
        )
        at_lower = np.sum(samples[:, 0] == 0.0)
        at_upper = np.sum(samples[:, 0] == 1.0)
        assert at_lower < 5
        assert at_upper < 5

    def test_periodic_dimension_stays_in_bounds(self):
        x0 = np.array([0.5])
        bounds = [(0.0, 1.0)]
        samples = metropolis_hastings(
            lambda x: 1.0, x0, bounds, size=500, periodic=[True],
            step_size=np.array([0.4])
        )
        assert np.all(samples[:, 0] >= 0.0)
        assert np.all(samples[:, 0] < 1.0)
