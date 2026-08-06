"""Ground-truth tests: verify expected event counts against analytic formulae.

The synthetic detector fixture has a *constant* effective area (1e4 cm²) across
all zeniths and energies in [1 TeV, 1 PeV], and a PowerLaw flux with gamma=2,
emin=100 GeV, emax=1 PeV, pivot=1e5 GeV, and per-flavor normalisation 1e-18.

For these choices the expected-event integral has a closed form:

  N = A_eff × (# flavors) × norm × norm_factor × pivot² × (1/E_lo - 1/E_hi) × T

where

  norm_factor = 1 / (1/emin_flux - 1/emax_flux) = 1/(1/100 - 1/1e6) ≈ 100.01
  [E_lo, E_hi] = intersection of IRF and flux energy ranges = [1e3, 1e6] GeV

This gives, for one year:
  point source (track, 2 flavors):   ≈ 630.6 events
  extended source (track, full sky):  ≈ 7924 events   (× solid angle = 4π sr)

Both samplers are tested against these analytic values at 5% tolerance to
account for discretisation error (50-point grids).
"""

import numpy as np
import pytest

from spore.conventions import SkyCoordinate, ureg


# ---------------------------------------------------------------------------
# Shared analytic reference
# ---------------------------------------------------------------------------

_A_EFF_CONST  = 1e4                    # cm²  (synthetic fixture)
_FLUX_NORM    = 1e-18                  # GeV⁻¹ cm⁻² s⁻¹ per flavor
_PL_EMIN      = 1e2                    # GeV  (PowerLaw emin)
_PL_EMAX      = 1e6                    # GeV  (PowerLaw emax)
_PIVOT        = 1e5                    # GeV
_IRF_EMIN     = 1e3                    # GeV  (IRF / effa grid minimum)
_IRF_EMAX     = 1e6                    # GeV  (IRF / effa grid maximum)
_T_YR_S       = 365.25 * 86400        # s

# norm_factor normalises the density to integrate to 1 over [_PL_EMIN, _PL_EMAX]
# in energy-linear space (matches PowerLaw with gamma ≠ 1).
_NORM_FACTOR  = 1.0 / (1.0/_PL_EMIN - 1.0/_PL_EMAX)  # ≈ 100.01

# ∫_{IRF_EMIN}^{IRF_EMAX} density(E) dE  =  norm_factor × pivot² × (1/E_lo - 1/E_hi)
_INT_DENSITY  = _NORM_FACTOR * _PIVOT**2 * (1.0/_IRF_EMIN - 1.0/_IRF_EMAX)

# Ground-truth event rates  (events s⁻¹)
_RATE_PS_TRACK  = _A_EFF_CONST * 2 * _FLUX_NORM * _INT_DENSITY   # numu + numu_bar
_RATE_EXT_TRACK = _RATE_PS_TRACK * 4 * np.pi  # × full-sky solid angle d(sin_dec)×dRA

_EVENTS_PS_YR   = _RATE_PS_TRACK  * _T_YR_S   # ≈ 630.6
_EVENTS_EXT_YR  = _RATE_EXT_TRACK * _T_YR_S   # ≈ 7924


# ---------------------------------------------------------------------------
# Point source
# ---------------------------------------------------------------------------

class TestPointSourceGroundTruth:
    """PointSourceEventSampler expected events must match the analytic value.

    The 50-point log-E grid introduces at most ~1 % quadrature error; we
    allow 5 % to give headroom for IRF interpolation.
    """

    def test_expected_events_1yr_vs_analytic(self, point_source_sampler):
        n = point_source_sampler.expected_events(
            "track", ureg.Quantity(_T_YR_S, "s")
        )
        assert n == pytest.approx(_EVENTS_PS_YR, rel=0.05), (
            f"Point source expected events ({n:.2f}) deviates >5% "
            f"from analytic reference ({_EVENTS_PS_YR:.2f})"
        )

    def test_cascade_expected_events_vs_analytic(self, point_source_sampler):
        # Cascade uses all 6 flavors; analytic value scales linearly.
        n = point_source_sampler.expected_events(
            "cascade", ureg.Quantity(_T_YR_S, "s")
        )
        analytic = _RATE_PS_TRACK * 3 * _T_YR_S  # 6 flavors = 3× numu pair
        assert n == pytest.approx(analytic, rel=0.05)

    def test_expected_events_scales_linearly_with_time(self, point_source_sampler):
        t1 = ureg.Quantity(_T_YR_S, "s")
        t2 = ureg.Quantity(2 * _T_YR_S, "s")
        n1 = point_source_sampler.expected_events("track", t1)
        n2 = point_source_sampler.expected_events("track", t2)
        assert n2 == pytest.approx(2 * n1, rel=1e-10)


# ---------------------------------------------------------------------------
# Extended source
# ---------------------------------------------------------------------------

@pytest.fixture
def extended_source_sampler(detector):
    from spore.physics import neutrinos
    from spore.source.flux.distributions.power_laws import PowerLaw
    from spore.source.flux.flux import Flux
    from spore.source.extended_source import ExtendedSource
    from spore.event_sampling import ExtendedSourceEventSampler

    pl = PowerLaw(gamma=2.0, emin=_PL_EMIN, emax=_PL_EMAX, pivot=_PIVOT)
    flux = Flux(
        {nu: _FLUX_NORM for nu in neutrinos},
        {nu: pl for nu in neutrinos},
    )
    src = ExtendedSource(flux)
    return ExtendedSourceEventSampler(detector, src, n_dec=50, n_ra=50, n_e=50)


@pytest.mark.slow
class TestExtendedSourceGroundTruth:
    """ExtendedSourceEventSampler expected events must match the analytic value.

    The isotropic, full-sky source with constant A_eff gives:
        N = A_eff × flux_integrated × 4π × T
    The 50×50×50 midpoint-rule grid introduces at most ~1 % error; we
    allow 5 % to give headroom for boundary effects and grid discretisation.
    """

    def test_expected_events_1yr_vs_analytic(self, extended_source_sampler):
        n = extended_source_sampler.expected_events(
            "track", ureg.Quantity(_T_YR_S, "s")
        )
        assert n == pytest.approx(_EVENTS_EXT_YR, rel=0.05), (
            f"Extended source expected events ({n:.2f}) deviates >5% "
            f"from analytic reference ({_EVENTS_EXT_YR:.2f})"
        )

    def test_expected_events_scales_linearly_with_time(self, extended_source_sampler):
        t1 = ureg.Quantity(_T_YR_S, "s")
        t2 = ureg.Quantity(2 * _T_YR_S, "s")
        n1 = extended_source_sampler.expected_events("track", t1)
        n2 = extended_source_sampler.expected_events("track", t2)
        assert n2 == pytest.approx(2 * n1, rel=1e-10)

    def test_extended_source_vs_point_source_ratio(
        self, extended_source_sampler, point_source_sampler
    ):
        """The full-sky isotropic integral should equal the point source integral
        scaled by 4π (the full-sky solid angle in d(sin_dec) dRA coordinates).
        This validates the solid-angle normalisation of both samplers.
        """
        t = ureg.Quantity(_T_YR_S, "s")
        n_ext = extended_source_sampler.expected_events("track", t)
        n_ps  = point_source_sampler.expected_events("track", t)
        assert n_ext == pytest.approx(4 * np.pi * n_ps, rel=0.05)
