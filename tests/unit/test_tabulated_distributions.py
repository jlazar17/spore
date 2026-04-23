"""Tests for TabulatedEnergyFlux and TabulatedEnergyDecFlux."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from spore.source.flux.distributions.oned_distribution import TabulatedEnergyFlux
from spore.source.flux.distributions.twod_distribution import TabulatedEnergyDecFlux


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_1d_dist(emin=1e2, emax=1e6, gamma=2.0):
    """Power-law spline in log-log space."""
    es = np.logspace(np.log10(emin), np.log10(emax), 30)
    log_es = np.log(es)
    # E^{-gamma} normalised to 1 at pivot=1e4 GeV
    vals = (es / 1e4) ** (-gamma)
    norm = np.trapezoid(vals, log_es)
    log_vals = np.log(vals / norm)
    spl = CubicSpline(log_es, log_vals)
    return TabulatedEnergyFlux(emin, emax, spl)


def _make_2d_dist(emin=1e2, emax=1e6, decmin=-np.pi/2, decmax=np.pi/2):
    n_sd, n_e = 10, 20
    sindecs = np.linspace(np.sin(decmin), np.sin(decmax), n_sd)
    es      = np.logspace(np.log10(emin), np.log10(emax), n_e)
    log_es  = np.log(es)
    # Simple separable: E^{-2} * (1 + sin_dec)
    flx = np.outer(1.0 + sindecs, es ** (-2.0))  # (n_sd, n_e)
    norm = np.trapezoid(
        np.trapezoid(flx * es[np.newaxis, :], log_es, axis=1),
        sindecs,
    )
    log_flx_norm = np.log(np.maximum(flx / norm, flx[flx > 0].min() * 1e-6))
    spl = RegularGridInterpolator((sindecs, log_es), log_flx_norm, method="pchip")
    return TabulatedEnergyDecFlux(emin, emax, decmin, decmax, spl)


# ---------------------------------------------------------------------------
# TabulatedEnergyFlux
# ---------------------------------------------------------------------------

class TestTabulatedEnergyFlux:
    def test_density_midpoint_positive(self):
        dist = _make_1d_dist()
        assert dist.density(1e4) > 0.0

    def test_density_scalar_returns_float(self):
        dist = _make_1d_dist()
        result = dist.density(1e4)
        assert isinstance(result, float)

    def test_density_array_returns_array(self):
        dist = _make_1d_dist()
        es = np.logspace(2, 6, 10)
        result = dist.density(es)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    def test_density_array_all_positive(self):
        dist = _make_1d_dist()
        es = np.logspace(2.1, 5.9, 20)
        assert np.all(dist.density(es) > 0.0)

    def test_density_below_emin_raises(self):
        dist = _make_1d_dist(emin=1e2)
        with pytest.raises(ValueError):
            dist.density(1.0)

    def test_density_above_emax_raises(self):
        dist = _make_1d_dist(emax=1e6)
        with pytest.raises(ValueError):
            dist.density(1e7)

    def test_density_array_oob_raises(self):
        dist = _make_1d_dist()
        with pytest.raises(ValueError):
            dist.density(np.array([1e3, 1e9]))

    def test_emin_emax_stored(self):
        dist = _make_1d_dist(emin=500.0, emax=1e5)
        assert dist.emin == pytest.approx(500.0)
        assert dist.emax == pytest.approx(1e5)

    def test_density_at_emin_boundary(self):
        dist = _make_1d_dist(emin=1e2, emax=1e6)
        assert dist.density(1e2) > 0.0

    def test_density_at_emax_boundary(self):
        dist = _make_1d_dist(emin=1e2, emax=1e6)
        assert dist.density(1e6) > 0.0

    def test_dec_and_ra_args_ignored(self):
        dist = _make_1d_dist()
        val_no_dec = dist.density(1e4)
        val_with_dec = dist.density(1e4, dec=0.3, ra=1.0)
        assert val_no_dec == pytest.approx(val_with_dec)

    def test_density_decreasing_for_power_law(self):
        dist = _make_1d_dist(gamma=2.0)
        e_lo, e_hi = 1e3, 1e5
        assert dist.density(e_lo) > dist.density(e_hi)


# ---------------------------------------------------------------------------
# TabulatedEnergyDecFlux
# ---------------------------------------------------------------------------

class TestTabulatedEnergyDecFlux:
    def test_density_midpoint_positive(self):
        dist = _make_2d_dist()
        assert dist.density(1e4, dec=0.0) > 0.0

    def test_density_scalar_returns_float(self):
        dist = _make_2d_dist()
        result = dist.density(1e4, dec=0.0)
        assert isinstance(result, float)

    def test_density_array_returns_array(self):
        dist = _make_2d_dist()
        es   = np.logspace(2.5, 5.5, 8)
        decs = np.linspace(-0.5, 0.5, 8)
        result = dist.density(es, dec=decs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)

    def test_density_array_all_positive(self):
        dist = _make_2d_dist()
        es   = np.logspace(2.5, 5.5, 10)
        decs = np.linspace(-0.5, 0.5, 10)
        assert np.all(dist.density(es, dec=decs) > 0.0)

    def test_density_below_emin_raises(self):
        dist = _make_2d_dist(emin=1e2)
        with pytest.raises(ValueError):
            dist.density(1.0, dec=0.0)

    def test_density_above_emax_raises(self):
        dist = _make_2d_dist(emax=1e6)
        with pytest.raises(ValueError):
            dist.density(1e8, dec=0.0)

    def test_density_below_decmin_raises(self):
        dist = _make_2d_dist(decmin=-np.pi / 4, decmax=np.pi / 4)
        with pytest.raises(ValueError):
            dist.density(1e4, dec=-np.pi / 2)

    def test_density_above_decmax_raises(self):
        dist = _make_2d_dist(decmin=-np.pi / 4, decmax=np.pi / 4)
        with pytest.raises(ValueError):
            dist.density(1e4, dec=np.pi / 2)

    def test_decmin_decmax_properties(self):
        decmin, decmax = -np.pi / 6, np.pi / 6
        dist = _make_2d_dist(decmin=decmin, decmax=decmax)
        assert dist.decmin == pytest.approx(decmin)
        assert dist.decmax == pytest.approx(decmax)

    def test_emin_emax_stored(self):
        dist = _make_2d_dist(emin=200.0, emax=5e5)
        assert dist.emin == pytest.approx(200.0)
        assert dist.emax == pytest.approx(5e5)

    def test_density_at_emin_boundary(self):
        dist = _make_2d_dist(emin=1e2, emax=1e6)
        assert dist.density(1e2, dec=0.0) > 0.0

    def test_density_at_decmin_boundary(self):
        dist = _make_2d_dist(decmin=-np.pi / 4, decmax=np.pi / 4)
        assert dist.density(1e4, dec=-np.pi / 4) > 0.0

    def test_density_varies_with_declination(self):
        # The fixture has flux proportional to (1 + sin_dec), so higher dec → more flux.
        dist = _make_2d_dist()
        val_south = dist.density(1e4, dec=-np.pi / 4)
        val_north = dist.density(1e4, dec=np.pi / 4)
        assert val_north > val_south
