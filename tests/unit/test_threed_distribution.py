"""Tests for TabulatedEnergyDecRAFlux (3-D distribution with RA dependence)."""
import numpy as np
import pytest

from scipy.interpolate import RegularGridInterpolator

from spore.source.flux.distributions.threed_distribution import TabulatedEnergyDecRAFlux


@pytest.fixture
def flat_3d_dist():
    """Uniform density = 1 everywhere in (sin_dec, RA, log_E) space."""
    sindecs = np.linspace(-1.0, 1.0, 5)
    ras     = np.linspace(0.0, 2 * np.pi, 7)
    log_es  = np.linspace(np.log(1e3), np.log(1e6), 6)
    log_density = np.zeros((5, 7, 6))  # log(1) = 0 → density = 1
    spl = RegularGridInterpolator(
        (sindecs, ras, log_es), log_density, method="linear", bounds_error=False, fill_value=0.0
    )
    return TabulatedEnergyDecRAFlux(
        emin=1e3, emax=1e6,
        decmin=np.arcsin(-1.0), decmax=np.arcsin(1.0),
        ramin=0.0, ramax=2 * np.pi,
        spl=spl,
    )


class TestTabulatedEnergyDecRAFlux:
    def test_scalar_density_returns_float(self, flat_3d_dist):
        result = flat_3d_dist.density(1e4, dec=0.0, ra=np.pi)
        assert isinstance(result, float)

    def test_flat_density_is_one(self, flat_3d_dist):
        result = flat_3d_dist.density(1e4, dec=0.0, ra=np.pi)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_array_density_shape(self, flat_3d_dist):
        es = np.array([1e3, 1e4, 1e5])
        result = flat_3d_dist.density(es, dec=np.zeros(3), ra=np.full(3, np.pi))
        assert result.shape == (3,)

    def test_energy_out_of_range_raises(self, flat_3d_dist):
        with pytest.raises(ValueError, match="Energy"):
            flat_3d_dist.density(1e2, dec=0.0, ra=np.pi)

    def test_dec_out_of_range_raises(self, flat_3d_dist):
        with pytest.raises(ValueError, match="Declination"):
            flat_3d_dist.density(1e4, dec=2.0, ra=np.pi)  # dec > pi/2

    def test_ra_out_of_range_raises(self, flat_3d_dist):
        with pytest.raises(ValueError, match="RA"):
            flat_3d_dist.density(1e4, dec=0.0, ra=7.0)  # RA > 2π

    def test_dec_required(self, flat_3d_dist):
        with pytest.raises((ValueError, TypeError)):
            flat_3d_dist.density(1e4, ra=np.pi)

    def test_ra_required(self, flat_3d_dist):
        with pytest.raises((ValueError, TypeError)):
            flat_3d_dist.density(1e4, dec=0.0)


class TestBatchDensity:
    def test_output_shape(self, flat_3d_dist):
        es   = np.logspace(3, 6, 4)
        decs = np.linspace(-np.pi / 3, np.pi / 3, 5)
        ras  = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
        result = flat_3d_dist.batch_density(es, decs, ras)
        assert result.shape == (5, 6, 4)

    def test_flat_batch_density_is_one(self, flat_3d_dist):
        es   = np.array([1e4, 1e5])
        decs = np.array([-0.3, 0.0, 0.3])
        ras  = np.array([0.5, 1.5, 3.0, 5.0])
        result = flat_3d_dist.batch_density(es, decs, ras)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_batch_matches_scalar(self, flat_3d_dist):
        """batch_density and scalar density must agree at matching points."""
        es   = np.array([1e4, 1e5])
        decs = np.array([-0.3, 0.0])
        ras  = np.array([1.0, 2.0, 3.0])
        batch = flat_3d_dist.batch_density(es, decs, ras)
        for i, dec in enumerate(decs):
            for j, ra in enumerate(ras):
                for k, e in enumerate(es):
                    scalar = flat_3d_dist.density(e, dec=dec, ra=ra)
                    assert batch[i, j, k] == pytest.approx(scalar, rel=1e-6)
