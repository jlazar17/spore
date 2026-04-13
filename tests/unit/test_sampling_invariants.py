"""
Tests for sampling invariants: CDF validity, interpolator boundary behaviour,
and distribution correctness.

These tests are designed to catch the class of bugs found during the boundary
audit:
  - CDF last value < 1 causing argmax to silently return 0 (lowest bin)
  - PCHIP/spline extrapolation beyond the stored quantile range producing
    extreme reco energies or angles
  - Half-width edge bins being over-sampled in the hierarchical sampler
"""

import numpy as np
import pytest
from scipy.stats import ks_1samp, uniform

from spore.event_sampling.utils import _build_sampling_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_target(n_dec=10, n_ra=8, n_e=12):
    """Return a uniform-weight target grid and the matching edge arrays."""
    sds    = np.linspace(-1.0, 1.0, n_dec)
    ras    = np.linspace(0.0, 2.0 * np.pi, n_ra, endpoint=False)
    log_es = np.linspace(np.log(1e2), np.log(1e6), n_e)
    es     = np.exp(log_es)

    dsd = (sds[-1] - sds[0]) / (n_dec - 1)
    sd_edges = np.clip(
        np.concatenate([[sds[0] - dsd / 2],
                        (sds[:-1] + sds[1:]) / 2,
                        [sds[-1] + dsd / 2]]),
        -1.0, 1.0,
    )
    ra_edges = np.linspace(0.0, 2.0 * np.pi, n_ra + 1)
    le_edges = np.concatenate([[log_es[0]],
                               (log_es[:-1] + log_es[1:]) / 2,
                               [log_es[-1]]])

    target = np.ones((n_dec, n_ra, n_e))
    return target, es, log_es, sds, sd_edges, ra_edges, le_edges


# ---------------------------------------------------------------------------
# CDF validity
# ---------------------------------------------------------------------------

class TestCDFValidity:
    """CDFs must be monotone and end at exactly 1.0."""

    def test_cdf_dec_ends_at_one(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert d["cdf_dec"][-1] == pytest.approx(1.0)

    def test_cdf_e_given_dec_ends_at_one(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert np.all(d["cdf_e_given_dec"][:, -1] == pytest.approx(1.0))

    def test_cdf_ra_given_dec_e_ends_at_one(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert np.all(d["cdf_ra_given_dec_e"][:, :, -1] == pytest.approx(1.0))

    def test_cdf_dec_is_monotone(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert np.all(np.diff(d["cdf_dec"]) >= 0)

    def test_cdf_e_given_dec_is_monotone(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert np.all(np.diff(d["cdf_e_given_dec"], axis=1) >= 0)

    def test_cdf_ra_given_dec_e_is_monotone(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert np.all(np.diff(d["cdf_ra_given_dec_e"], axis=2) >= 0)

    def test_cdf_dec_ends_at_one_with_sparse_target(self):
        """CDFs must still end at 1.0 when most of the grid has zero weight."""
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target()
        target[:] = 0.0
        target[5, 3, 8] = 1.0  # single non-zero cell
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        assert d["cdf_dec"][-1] == pytest.approx(1.0)
        assert np.all(d["cdf_e_given_dec"][:, -1] == pytest.approx(1.0))
        assert np.all(d["cdf_ra_given_dec_e"][:, :, -1] == pytest.approx(1.0))


# ---------------------------------------------------------------------------
# Boundary quantile behaviour for energy and angular splines
# ---------------------------------------------------------------------------

class TestSplineBoundaryBehaviour:
    """Splines must not extrapolate to extreme values at quantile boundaries."""

    def test_energy_spline_at_u_zero(self, detector):
        """Quantile u=0 must return a finite, physically plausible Delta."""
        e_sampler = detector.response.energy_response["track"]
        delta = e_sampler(0.0)
        assert np.isfinite(delta), "energy spline returned non-finite at u=0"
        assert delta > -20, f"energy spline extrapolated to extreme negative at u=0: {delta}"

    def test_energy_spline_at_u_one(self, detector):
        """Quantile u=1 must return a finite Delta."""
        e_sampler = detector.response.energy_response["track"]
        delta = e_sampler(1.0)
        assert np.isfinite(delta), "energy spline returned non-finite at u=1"

    def test_energy_spline_slightly_below_zero(self, detector):
        """u slightly below 0 must clamp, not extrapolate wildly."""
        e_sampler = detector.response.energy_response["track"]
        delta = e_sampler(-1e-10)
        assert np.isfinite(delta)
        assert delta > -20

    def test_energy_spline_slightly_above_one(self, detector):
        """u slightly above 1 must clamp, not extrapolate wildly."""
        e_sampler = detector.response.energy_response["track"]
        delta = e_sampler(1.0 + 1e-10)
        assert np.isfinite(delta)

    def test_ang_spline_at_u_zero(self, detector):
        """Angular spline at u=0 must return a non-negative finite angle."""
        ang_sampler = detector.response.angular_response["track"]
        psi = ang_sampler(1e13, 0.0)
        assert np.isfinite(psi)
        assert psi >= 0.0

    def test_ang_spline_at_u_one(self, detector):
        """Angular spline at u=1 must return a finite angle."""
        ang_sampler = detector.response.angular_response["track"]
        psi = ang_sampler(1e13, 1.0)
        assert np.isfinite(psi)
        assert psi >= 0.0

    def test_ang_spline_slightly_above_u_max(self, detector):
        """u slightly above the stored maximum must clamp, not blow up."""
        ang_sampler = detector.response.angular_response["track"]
        psi = ang_sampler(1e13, 1.0 + 1e-10)
        assert np.isfinite(psi)
        assert psi >= 0.0

    def test_reco_energy_never_extreme(self, detector):
        """Smeared reco energies over 10k draws must stay within 6 orders of true."""
        from spore.event_sampling.utils import smear_truth_batch
        rng = np.random.default_rng(0)
        n = 10_000
        true_e = np.full(n, 1e13)  # 10 TeV in GeV
        true_dec = np.zeros(n)
        true_ra  = np.zeros(n)
        _, _, reco_e, _ = smear_truth_batch(true_dec, true_ra, true_e, detector, "track", rng=rng)
        assert np.all(reco_e > 0), "reco energies must be positive"
        ratio = reco_e / true_e
        assert np.all(ratio > 1e-6), f"extreme downward smearing: min ratio = {ratio.min():.2e}"
        assert np.all(ratio < 1e6),  f"extreme upward smearing: max ratio = {ratio.max():.2e}"


# ---------------------------------------------------------------------------
# Distribution correctness: uniform source → uniform samples
# ---------------------------------------------------------------------------

class TestDistributionCorrectness:
    """For a uniform-weight source, sampled coordinates should be uniform."""

    @pytest.fixture
    def uniform_sampling_data(self):
        target, es, log_es, sds, sd_edges, ra_edges, le_edges = _make_uniform_target(
            n_dec=20, n_ra=16, n_e=20
        )
        return _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)

    def test_sin_dec_is_uniform(self, uniform_sampling_data):
        """Marginal sin(dec) distribution must be uniform for a uniform source."""
        d = uniform_sampling_data
        rng = np.random.default_rng(42)
        n = 5_000
        n_dec = len(d["cdf_dec"])
        i_dec = np.clip(np.searchsorted(d["cdf_dec"], rng.random(n)), 0, n_dec - 1)
        sd_edges = d["sd_edges"]
        sin_dec = rng.uniform(sd_edges[i_dec], sd_edges[i_dec + 1])

        result = ks_1samp(sin_dec, uniform(loc=-1.0, scale=2.0).cdf)
        assert result.pvalue > 0.01, (
            f"sin(dec) distribution not uniform: KS p={result.pvalue:.4f}"
        )

    def test_log_energy_is_uniform(self):
        """For target ∝ 1/E the weight w = target*E = 1, so log(E) must be uniform."""
        n_dec, n_ra, n_e = 20, 16, 20
        sds    = np.linspace(-1.0, 1.0, n_dec)
        log_es = np.linspace(np.log(1e2), np.log(1e6), n_e)
        es     = np.exp(log_es)
        ras    = np.linspace(0.0, 2.0 * np.pi, n_ra, endpoint=False)
        dsd = (sds[-1] - sds[0]) / (n_dec - 1)
        sd_edges = np.clip(
            np.concatenate([[sds[0] - dsd / 2],
                            (sds[:-1] + sds[1:]) / 2,
                            [sds[-1] + dsd / 2]]),
            -1.0, 1.0,
        )
        ra_edges = np.linspace(0.0, 2.0 * np.pi, n_ra + 1)
        le_edges = np.concatenate([[log_es[0]],
                                   (log_es[:-1] + log_es[1:]) / 2,
                                   [log_es[-1]]])

        # target ∝ 1/E → w = target * E = 1 everywhere → uniform in log-E
        target = np.ones((n_dec, n_ra, n_e)) / es[np.newaxis, np.newaxis, :]
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)

        rng = np.random.default_rng(42)
        n = 5_000
        i_dec = np.clip(np.searchsorted(d["cdf_dec"], rng.random(n)), 0, n_dec - 1)
        cdfs_e = d["cdf_e_given_dec"][i_dec]
        i_e = np.clip(
            np.argmax(cdfs_e >= rng.random(n)[:, None], axis=1), 0, n_e - 1
        )
        log_e = rng.uniform(le_edges[i_e], le_edges[i_e + 1])

        result = ks_1samp(
            log_e,
            uniform(loc=le_edges[0], scale=le_edges[-1] - le_edges[0]).cdf
        )
        assert result.pvalue > 0.01, (
            f"log(E) distribution not uniform for 1/E source: KS p={result.pvalue:.4f}"
        )

    def test_boundary_bins_not_over_sampled(self):
        """Boundary bin density (count/width) must match interior bins for a 1/E source."""
        n_dec, n_ra, n_e = 20, 16, 20
        sds    = np.linspace(-1.0, 1.0, n_dec)
        log_es = np.linspace(np.log(1e2), np.log(1e6), n_e)
        es     = np.exp(log_es)
        ras    = np.linspace(0.0, 2.0 * np.pi, n_ra, endpoint=False)
        dsd = (sds[-1] - sds[0]) / (n_dec - 1)
        sd_edges = np.clip(
            np.concatenate([[sds[0] - dsd / 2],
                            (sds[:-1] + sds[1:]) / 2,
                            [sds[-1] + dsd / 2]]),
            -1.0, 1.0,
        )
        ra_edges = np.linspace(0.0, 2.0 * np.pi, n_ra + 1)
        le_edges = np.concatenate([[log_es[0]],
                                   (log_es[:-1] + log_es[1:]) / 2,
                                   [log_es[-1]]])

        # 1/E target → uniform density in log-E
        target = np.ones((n_dec, n_ra, n_e)) / es[np.newaxis, np.newaxis, :]
        d = _build_sampling_data(target, es, log_es, sds, sd_edges, ra_edges, le_edges)
        le_widths = np.diff(le_edges)

        rng = np.random.default_rng(42)
        n = 50_000
        i_dec = np.clip(np.searchsorted(d["cdf_dec"], rng.random(n)), 0, n_dec - 1)
        cdfs_e = d["cdf_e_given_dec"][i_dec]
        i_e = np.clip(
            np.argmax(cdfs_e >= rng.random(n)[:, None], axis=1), 0, n_e - 1
        )
        counts = np.bincount(i_e, minlength=n_e).astype(float)

        # Density = count / bin_width; for uniform distribution all densities equal.
        # Boundary bins have half-width so they should have ~half the raw counts.
        # Allow 20% tolerance around the expected ratio.
        density = counts / le_widths
        mean_density = density[1:-1].mean()  # mean of interior bins
        assert density[0]  == pytest.approx(mean_density, rel=0.2), \
            f"first bin density {density[0]:.1f} deviates from interior mean {mean_density:.1f}"
        assert density[-1] == pytest.approx(mean_density, rel=0.2), \
            f"last bin density {density[-1]:.1f} deviates from interior mean {mean_density:.1f}"
