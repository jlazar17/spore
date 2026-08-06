"""Regression tests: fixed-seed outputs that must not silently change.

Each test pins a specific numerical output produced by the current implementation.
If a change causes any of these values to shift, the test will fail loudly,
prompting a deliberate decision about whether the change is intended.

All tests rely only on the shared ``detector`` / ``point_source`` /
``point_source_sampler`` fixtures from conftest.py so the synthetic IRF is
identical across every run.
"""
import numpy as np
import pytest

from spore.conventions import SkyCoordinate, ureg
from spore.event_sampling.utils import smear_truth


class TestSmearTruthRegression:
    """smear_truth with a fixed Generator must produce bit-identical output."""

    def test_track_reco_declination(self, detector):
        rng = np.random.default_rng(42)
        sc = SkyCoordinate(0.0, 0.0)
        reco_dir, _, _ = smear_truth(sc, 1e4, detector, "track", rng=rng)
        assert reco_dir.declination == pytest.approx(0.2402932864, rel=1e-8)

    def test_track_reco_ra(self, detector):
        rng = np.random.default_rng(42)
        sc = SkyCoordinate(0.0, 0.0)
        reco_dir, _, _ = smear_truth(sc, 1e4, detector, "track", rng=rng)
        assert reco_dir.right_ascension == pytest.approx(0.3063257015, rel=1e-8)

    def test_track_reco_energy(self, detector):
        rng = np.random.default_rng(42)
        sc = SkyCoordinate(0.0, 0.0)
        _, reco_e, _ = smear_truth(sc, 1e4, detector, "track", rng=rng)
        assert reco_e == pytest.approx(8849.331969, rel=1e-6)

    def test_track_ang_err(self, detector):
        # ang_err = ang_sampler(e, 0.5); the synthetic inv_CDF maps 0.5 → 0.25.
        rng = np.random.default_rng(42)
        sc = SkyCoordinate(0.0, 0.0)
        _, _, ang_err = smear_truth(sc, 1e4, detector, "track", rng=rng)
        assert ang_err == pytest.approx(0.25, rel=1e-8)

    def test_same_seed_gives_identical_results(self, detector):
        sc = SkyCoordinate(0.1, 1.5)
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        dir_a, e_a, ae_a = smear_truth(sc, 5e3, detector, "cascade", rng=rng_a)
        dir_b, e_b, ae_b = smear_truth(sc, 5e3, detector, "cascade", rng=rng_b)
        assert dir_a.declination == dir_b.declination
        assert dir_a.right_ascension == dir_b.right_ascension
        assert e_a == e_b
        assert ae_a == ae_b

    def test_different_seeds_give_different_directions(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        dir_a, _, _ = smear_truth(sc, 1e4, detector, "track", rng=np.random.default_rng(1))
        dir_b, _, _ = smear_truth(sc, 1e4, detector, "track", rng=np.random.default_rng(2))
        # Different seeds should give different azimuthal angles (not identical RA).
        assert dir_a.right_ascension != dir_b.right_ascension


@pytest.mark.slow
class TestPointSourceSamplerRegression:
    """PointSourceEventSampler with a fixed seed must produce stable outputs."""

    def test_expected_events_1yr(self, point_source_sampler):
        n = point_source_sampler.expected_events(
            "track", ureg.Quantity(365.25 * 24 * 3600, "s")
        )
        assert n == pytest.approx(634.9918254142343, rel=1e-5)

    def test_nevent_3_seed42_true_energies(self, point_source_sampler):
        events = point_source_sampler.sample_events("track", nevent=3, seed=42)
        expected = [4412.290293, 1781.198620, 7044.512448]
        for ev, exp in zip(events, expected):
            assert ev.true_energy.magnitude == pytest.approx(exp, rel=1e-5)

    def test_nevent_3_seed42_reco_energies(self, point_source_sampler):
        events = point_source_sampler.sample_events("track", nevent=3, seed=42)
        expected = [7438.530378, 3156.346318, 3348.378514]
        for ev, exp in zip(events, expected):
            assert ev.reco_energy.magnitude == pytest.approx(exp, rel=1e-5)

    def test_same_seed_gives_identical_events(self, point_source_sampler):
        a = point_source_sampler.sample_events("cascade", nevent=5, seed=7)
        b = point_source_sampler.sample_events("cascade", nevent=5, seed=7)
        for ea, eb in zip(a, b):
            assert ea.true_energy.magnitude == eb.true_energy.magnitude
            assert ea.reco_energy.magnitude == eb.reco_energy.magnitude
            assert ea.true_direction.declination == eb.true_direction.declination

    def test_different_seeds_give_different_energies(self, point_source_sampler):
        a = point_source_sampler.sample_events("track", nevent=5, seed=1)
        b = point_source_sampler.sample_events("track", nevent=5, seed=2)
        energies_a = [ev.true_energy.magnitude for ev in a]
        energies_b = [ev.true_energy.magnitude for ev in b]
        assert energies_a != energies_b
