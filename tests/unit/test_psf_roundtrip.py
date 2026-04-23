"""Round-trip validation of the angular smearing pipeline.

Two independent properties are tested:

1. PSF recovery: the empirical CDF of sampled angular separations ψ = ang_sep(true, reco)
   must match the stored PSF inverse-CDF to within sampling noise (KS test).

2. Azimuthal isotropy: the azimuthal angle φ around the true direction must be
   uniform in [0, 2π) — any systematic break here would indicate a bug in the
   cone-sampling geometry.

The synthetic detector fixture from conftest.py has
    PSF: ang_sampler(E, u) = 0.5 * u  →  ψ ~ Uniform(0, 0.5) radians
so the expected CDF is F(ψ) = 2ψ for ψ ∈ [0, 0.5].
"""

import numpy as np
import pytest
from scipy.stats import ks_1samp


def _ang_sep(dec1, ra1, dec2, ra2):
    """Great-circle angular separation in radians."""
    cos_psi = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return np.arccos(np.clip(cos_psi, -1.0, 1.0))


def _azimuthal_angle(true_dec, true_ra, reco_decs, reco_ras):
    """Azimuthal angle φ of each reco direction around the true direction.

    Constructs a local orthonormal frame (x, y, z) at the true direction and
    projects each reco unit vector onto x and y to get φ = atan2(y-component,
    x-component).
    """
    # True direction unit vector
    vz = np.array([
        np.cos(true_dec) * np.cos(true_ra),
        np.cos(true_dec) * np.sin(true_ra),
        np.sin(true_dec),
    ])

    # Build orthonormal x-axis: use [0,0,1] unless vz is nearly polar
    ref = np.array([0.0, 0.0, 1.0]) if abs(vz[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    vx = np.cross(ref, vz)
    vx /= np.linalg.norm(vx)
    vy = np.cross(vz, vx)

    reco_vecs = np.stack([
        np.cos(reco_decs) * np.cos(reco_ras),
        np.cos(reco_decs) * np.sin(reco_ras),
        np.sin(reco_decs),
    ], axis=1)  # (n, 3)

    comp_x = reco_vecs @ vx
    comp_y = reco_vecs @ vy
    return np.arctan2(comp_y, comp_x) % (2.0 * np.pi)


class TestPSFRoundTrip:
    """Sampled angular separations must recover the stored PSF CDF."""

    def test_psf_cdf_recovered_track(self, detector):
        """KS test: empirical ψ-CDF vs. theoretical for track morphology."""
        from spore.event_sampling.utils import smear_truth_batch

        rng = np.random.default_rng(0)
        n = 5_000
        true_e   = np.full(n, 1e13)      # 10 TeV — well within stored range
        true_dec = np.full(n, 0.0)
        true_ra  = np.full(n, 1.0)

        _, reco_ras, _, _ = smear_truth_batch(
            true_dec, true_ra, true_e, detector, "track", rng=rng,
        )
        reco_decs, reco_ras, _, _ = smear_truth_batch(
            true_dec, true_ra, true_e, detector, "track", rng=rng,
        )

        psi = _ang_sep(true_dec, true_ra, reco_decs, reco_ras)

        # Synthetic PSF: ang_sampler(E, u) = 0.5 * u  →  ψ ~ Uniform(0, 0.5)
        # Theoretical CDF: F(ψ) = ψ / 0.5 = 2ψ for ψ ∈ [0, 0.5]
        def theoretical_cdf(psi_vals):
            return np.clip(psi_vals / 0.5, 0.0, 1.0)

        result = ks_1samp(psi, theoretical_cdf)
        assert result.pvalue > 0.01, (
            f"PSF CDF not recovered for track: KS p={result.pvalue:.4f}, "
            f"median ψ={np.median(psi):.4f} rad (expected 0.25)"
        )

    def test_psf_cdf_recovered_cascade(self, detector):
        """KS test: empirical ψ-CDF vs. theoretical for cascade morphology."""
        from spore.event_sampling.utils import smear_truth_batch

        rng = np.random.default_rng(1)
        n = 5_000
        true_e   = np.full(n, 1e13)
        true_dec = np.full(n, 0.5)
        true_ra  = np.full(n, 2.0)

        reco_decs, reco_ras, _, _ = smear_truth_batch(
            true_dec, true_ra, true_e, detector, "cascade", rng=rng,
        )

        psi = _ang_sep(true_dec, true_ra, reco_decs, reco_ras)

        def theoretical_cdf(psi_vals):
            return np.clip(psi_vals / 0.5, 0.0, 1.0)

        result = ks_1samp(psi, theoretical_cdf)
        assert result.pvalue > 0.01, (
            f"PSF CDF not recovered for cascade: KS p={result.pvalue:.4f}"
        )

    def test_psf_median_matches_ang_err(self, detector):
        """Median sampled ψ must match ang_sampler(E, 0.5) — the reported ang_err."""
        from spore.event_sampling.utils import smear_truth_batch

        rng = np.random.default_rng(2)
        n = 10_000
        true_e   = np.full(n, 1e13)
        true_dec = np.full(n, 0.0)
        true_ra  = np.full(n, 0.0)

        reco_decs, reco_ras, _, ang_errs = smear_truth_batch(
            true_dec, true_ra, true_e, detector, "track", rng=rng,
        )
        psi = _ang_sep(true_dec, true_ra, reco_decs, reco_ras)

        expected_median = float(np.unique(ang_errs)[0])  # ang_sampler(1e13, 0.5)
        assert np.median(psi) == pytest.approx(expected_median, rel=0.05), (
            f"median ψ={np.median(psi):.4f} does not match ang_err={expected_median:.4f}"
        )


class TestAzimuthalIsotropy:
    """Reconstructed directions must be azimuthally uniform around the true direction."""

    @pytest.mark.parametrize("true_dec,true_ra", [
        (0.0,  0.0),   # equator, prime meridian
        (0.8,  1.5),   # mid-latitude
        (1.4,  3.0),   # near north pole
        (-1.4, 0.5),   # near south pole
    ])
    def test_phi_is_uniform(self, detector, true_dec, true_ra):
        from spore.event_sampling.utils import smear_truth_batch
        from scipy.stats import ks_1samp, uniform

        rng = np.random.default_rng(42)
        n = 5_000
        true_e   = np.full(n, 1e13)
        true_decs = np.full(n, true_dec)
        true_ras  = np.full(n, true_ra)

        reco_decs, reco_ras, _, _ = smear_truth_batch(
            true_decs, true_ras, true_e, detector, "track", rng=rng,
        )

        phi = _azimuthal_angle(true_dec, true_ra, reco_decs, reco_ras)
        result = ks_1samp(phi, uniform(loc=0.0, scale=2.0 * np.pi).cdf)
        assert result.pvalue > 0.01, (
            f"Azimuthal distribution not uniform at dec={true_dec:.2f}, "
            f"ra={true_ra:.2f}: KS p={result.pvalue:.4f}"
        )

    def test_phi_uniform_with_local_zenith(self, detector):
        """Azimuthal isotropy must hold when a local zenith is provided."""
        from spore.event_sampling.utils import smear_truth_batch
        from scipy.stats import ks_1samp, uniform

        rng = np.random.default_rng(7)
        n = 5_000
        true_dec = 0.3
        true_ra  = 1.0
        true_e   = np.full(n, 1e13)
        true_decs = np.full(n, true_dec)
        true_ras  = np.full(n, true_ra)
        zeniths   = np.full(n, 1.2)  # fixed arbitrary zenith

        reco_decs, reco_ras, _, _ = smear_truth_batch(
            true_decs, true_ras, true_e, detector, "track", rng=rng,
            local_zeniths=zeniths,
        )

        phi = _azimuthal_angle(true_dec, true_ra, reco_decs, reco_ras)
        result = ks_1samp(phi, uniform(loc=0.0, scale=2.0 * np.pi).cdf)
        assert result.pvalue > 0.01, (
            f"Azimuthal distribution not uniform with explicit zenith: "
            f"KS p={result.pvalue:.4f}"
        )
