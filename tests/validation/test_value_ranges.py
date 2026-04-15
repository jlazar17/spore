import numpy as np
import pytest

from spore.conventions import SkyCoordinate
from spore.conventions import ureg
from spore.event_sampling.utils import smear_truth
from spore.source.flux.distributions.power_laws import PowerLaw


class TestPowerLawValueRanges:
    def test_density_nonnegative_across_range(self):
        pl = PowerLaw(2.0, 1e2, 1e6, 1e3)
        for e in np.logspace(2, 6, 100):
            assert pl.density(e) >= 0

    def test_log_slope_matches_negative_gamma(self):
        # The spectral slope d(log density)/d(log e) should equal -gamma,
        # regardless of normalization or pivot choice.
        for gamma in [1.5, 2.0, 2.7, 3.0]:
            pl = PowerLaw(gamma, 1e2, 1e6, 1e3)
            e1, e2 = 1e3, 1e5
            log_slope = np.log(pl.density(e2) / pl.density(e1)) / np.log(e2 / e1)
            assert log_slope == pytest.approx(-gamma, rel=1e-6)

    def test_density_decreases_with_energy_for_gamma_gt_zero(self):
        pl = PowerLaw(2.0, 1e2, 1e6, 1e3)
        energies = np.logspace(2, 6, 10)
        densities = [pl.density(e) for e in energies]
        assert all(densities[i] > densities[i + 1] for i in range(len(densities) - 1))


class TestSmearedEventValueRanges:
    def test_reco_direction_declination_in_range(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        for i in range(20):
            reco_dir, _, _ = smear_truth(sc, 1e4, detector, "track", rng=np.random.default_rng(i))  # 10 TeV in GeV
            assert -np.pi / 2 <= reco_dir.declination <= np.pi / 2

    def test_reco_direction_ra_in_range(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        for i in range(20):
            reco_dir, _, _ = smear_truth(sc, 1e4, detector, "cascade", rng=np.random.default_rng(i))
            assert 0.0 <= reco_dir.right_ascension <= 2 * np.pi

    def test_reco_energy_positive(self, detector):
        sc = SkyCoordinate(0.0, 0.0)
        for i in range(20):
            _, reco_e, _ = smear_truth(sc, 1e4, detector, "track", rng=np.random.default_rng(i))
            assert reco_e > 0


class TestEffectiveAreaValueRanges:
    def test_effective_area_nonnegative_for_track(self, detector):
        effa = detector.response.effective_area["track"]
        test_cases = [
            (np.pi / 4, 1e3),   # 1 TeV
            (np.pi / 2, 1e4),   # 10 TeV
            (3 * np.pi / 4, 1e5),  # 100 TeV
        ]
        for zen, e in test_cases:
            assert effa(zen, e) >= 0

    def test_effective_area_nonnegative_for_cascade(self, detector):
        effa = detector.response.effective_area["cascade"]
        test_cases = [
            (np.pi / 4, 1e3),
            (np.pi / 2, 1e4),
            (3 * np.pi / 4, 1e5),
        ]
        for zen, e in test_cases:
            assert effa(zen, e) >= 0

    def test_effective_area_zero_below_grid_minimum(self, detector):
        effa = detector.response.effective_area["track"]
        # Synthetic grid starts at 1 TeV (1e3 GeV); anything below should return 0
        assert effa(np.pi / 2, 10) == 0.0    # 10 GeV < 1 TeV
        assert effa(np.pi / 2, 100) == 0.0   # 100 GeV < 1 TeV


class TestAngularResponseValueRanges:
    def test_angular_response_nonnegative(self, detector):
        ang = detector.response.angular_response["track"]
        for u in np.linspace(0.05, 0.95, 10):
            assert ang(1e4, u) >= 0

    def test_angular_response_increases_with_u(self, detector):
        # The synthetic inverse CDF maps [0,1] → [0, 0.5], so it is monotone.
        ang = detector.response.angular_response["cascade"]
        us = np.linspace(0.1, 0.9, 9)
        values = [ang(1e4, u) for u in us]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


class TestSampledEventValueRanges:
    @pytest.mark.slow
    def test_morphology_is_valid(self, point_source_sampler):
        for morphology in ["track", "cascade"]:
            events = point_source_sampler.sample_events(morphology, nevent=10)
            for event in events:
                assert event.morphology == morphology

    @pytest.mark.slow
    def test_sampled_energies_are_positive(self, point_source_sampler):
        events = point_source_sampler.sample_events("cascade", nevent=30)
        for event in events:
            assert event.true_energy.magnitude > 0
            assert event.reco_energy.magnitude > 0
