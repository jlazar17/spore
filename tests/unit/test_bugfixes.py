"""Regression tests for bugs found in the code audit.

Each test fails against the pre-fix implementation.
"""
import numpy as np
import pytest

from spore.conventions import SkyCoordinate, EarthCoordinate, ureg
from spore.detector.detector import Detector, Medium
from spore.detector.detector_response.utils import (
    effa_helper, _largest_contiguous_nonzero,
)
from spore.event_sampling.good_run_list import GoodRunList
from spore.event_sampling.utils import (
    _assign_times_from_slices, _slice_indices_from_times, _mjd_to_lst,
    _grl_slice_weights, _grl_mean_rate, _sample_times_in_slices, smear_truth,
)

_MED_LON = np.radians(16.1)


@pytest.fixture
def grl(tmp_path):
    """200 contiguous days of near-full uptime."""
    runs = np.array([[60000.0 + i, 60000.0 + i + 0.98] for i in range(200)])
    path = tmp_path / "uptime.csv"
    np.savetxt(path, runs, header="MJD_start[days] MJD_stop[days]", comments="# ")
    return GoodRunList.from_path(str(path))


class TestSiderealPhaseOfAssignedTimes:
    """Assigned event times must land in the sidereal slice they came from."""

    @pytest.mark.parametrize("t_start", [60355.83, 60000.0, 59123.456])
    @pytest.mark.parametrize("phase_ref", [0.0, 1.3])
    def test_round_trip(self, t_start, phase_ref):
        rng = np.random.default_rng(0)
        n_slices = 24
        k = rng.integers(0, n_slices, size=4000)
        times = _assign_times_from_slices(
            t_start, ureg.Quantity(180.0, "day"), k, n_slices, rng,
            longitude=_MED_LON, phase_ref=phase_ref,
        )
        k_back = _slice_indices_from_times(times, _MED_LON, n_slices, phase_ref)
        assert np.array_equal(k_back, k)

    def test_short_window_warns_and_stays_in_range(self, caplog):
        """A sub-sidereal-day window cannot cover every phase; warn, don't crash."""
        rng = np.random.default_rng(3)
        t_start, days = 60355.83, 0.25
        k = rng.integers(0, 24, size=200)
        with caplog.at_level("WARNING"):
            times = _assign_times_from_slices(
                t_start, ureg.Quantity(days, "day"), k, 24, rng,
                longitude=_MED_LON,
            )
        assert "sidereal" in caplog.text
        assert times.min() >= t_start
        assert times.max() <= t_start + days

    def test_times_stay_inside_window(self):
        rng = np.random.default_rng(1)
        t_start, days = 60355.83, 7.0
        k = rng.integers(0, 24, size=2000)
        times = _assign_times_from_slices(
            t_start, ureg.Quantity(days, "day"), k, 24, rng, longitude=_MED_LON,
        )
        assert times.min() >= t_start
        assert times.max() <= t_start + days


class TestSiderealTimeAgainstAstropy:
    """_mjd_to_lst must agree with astropy, not sit half a sidereal day away."""

    @pytest.mark.parametrize("mjd", [60355.83, 59000.0, 61000.25, 55000.5])
    @pytest.mark.parametrize("lon_deg", [0.0, 16.1, -63.4])
    def test_matches_astropy_mean_sidereal_time(self, mjd, lon_deg):
        from astropy.time import Time
        from astropy import units as u

        lon = np.radians(lon_deg)
        expected = float(
            Time(mjd, format="mjd").sidereal_time("mean", longitude=lon * u.rad).rad
        )
        got = _mjd_to_lst(mjd, lon)
        diff = (expected - got + np.pi) % (2 * np.pi) - np.pi
        assert abs(np.degrees(diff)) < 0.01

    def test_analytic_and_astropy_zenith_grids_agree(self):
        """The two coordinate paths in the package must not diverge."""
        from spore.event_sampling.utils import zenith_grid

        decs = np.arcsin(np.linspace(-0.99, 0.99, 15))
        ras = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        ec = EarthCoordinate(np.radians(36.3), np.radians(16.1))
        t = 60355.83
        astro = zenith_grid(decs, ras, ec, t)
        lst = _mjd_to_lst(t, ec.longitude)
        ha = lst - ras
        cz = (np.sin(ec.latitude) * np.sin(decs)[:, None]
              + np.cos(ec.latitude) * np.cos(decs)[:, None] * np.cos(ha)[None, :])
        analytic = np.arccos(np.clip(cz, -1, 1))
        # Residual is precession/nutation/aberration, absent from mean GMST.
        assert np.max(np.abs(astro - analytic)) < np.radians(1.0)


class TestGrlSliceWeighting:
    """Under a GRL the slice choice must follow rate x exposure, not exposure alone."""

    def test_weights_follow_rate(self, grl):
        norms = np.zeros(24)
        norms[5] = 3.0
        norms[6] = 1.0
        w = _grl_slice_weights(norms, grl, _MED_LON)
        assert w[5] == pytest.approx(0.75, rel=1e-2)
        assert w[6] == pytest.approx(0.25, rel=1e-2)
        assert w.sum() == pytest.approx(1.0)

    def test_mean_rate_is_exposure_weighted(self, grl):
        # Near-uniform sidereal coverage over 200 days: the exposure-weighted
        # mean must match the plain mean.
        norms = np.linspace(1.0, 5.0, 24)
        assert _grl_mean_rate(norms, grl, _MED_LON) == pytest.approx(
            norms.mean(), rel=2e-2
        )

    def test_lst_exposure_normalised(self, grl):
        frac = grl.lst_exposure(_MED_LON, 24)
        assert frac.sum() == pytest.approx(1.0)
        assert np.all(frac > 0)

    def test_times_drawn_in_requested_slice(self, grl):
        rng = np.random.default_rng(2)
        k = rng.integers(0, 24, size=1500)
        times = _sample_times_in_slices(grl, k, 24, rng, _MED_LON)
        k_back = _slice_indices_from_times(times, _MED_LON, 24)
        # Times come from a finite pool of good-run samples, so allow a small
        # fallback fraction for slices the run list barely covers.
        assert np.mean(k_back == k) > 0.99


class TestEffectiveAreaSmoothingIsIndependentOfTrimming:
    def test_smoothing_applies_without_trimming(self):
        zens = np.radians(np.array([10.0, 90.0, 170.0]))
        es = np.logspace(2, 6, 20)
        rng = np.random.default_rng(0)
        vals = np.abs(np.outer(es ** 1.5, np.ones(3))
                      * (1 + 0.3 * rng.standard_normal((20, 3))))
        unsmoothed = effa_helper(zens, es, vals.copy(),
                                 trim_isolated=False, smoothing_sigma=0.0)
        smoothed = effa_helper(zens, es, vals.copy(),
                               trim_isolated=False, smoothing_sigma=3.0)
        q = np.full(8, np.radians(90.0))
        eq = np.logspace(2.5, 5.5, 8)
        assert not np.allclose(unsmoothed(q, eq), smoothed(q, eq))


class TestLargestContiguousNonzero:
    """The point-source energy trim keeps the main body, not just the tail."""

    def test_trailing_zero_does_not_discard_everything(self):
        arr = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 7.0, 0.0])
        assert _largest_contiguous_nonzero(arr) == (2, 5)

    def test_all_zero(self):
        assert _largest_contiguous_nonzero(np.zeros(5)) is None


class TestPointSourcePerSpeciesEffectiveArea:
    """A response storing one effective area per species must be usable."""

    def test_sampler_handles_species_list(self, per_species_detector, point_source):
        from spore.event_sampling import SourceSampler

        sampler = SourceSampler(per_species_detector, point_source)
        n = sampler.expected_events("track", deltat=ureg.Quantity(365.0, "day"))
        assert np.isfinite(n) and n > 0
        events = sampler.sample_events("track", deltat=ureg.Quantity(365.0, "day"),
                                       seed=0)
        assert len(events) > 0

    def test_matches_single_species_sum(self, per_species_detector,
                                        summed_detector, point_source):
        """A muon-only per-species response must match the numu heuristic.

        The single-species ``track`` path weights the flux by nu_mu + nu_mu_bar;
        the per-species path weights each species by its own effective area.
        With area only in the two muon entries the two must agree.
        """
        from spore.event_sampling import SourceSampler

        a = SourceSampler(per_species_detector, point_source).expected_events(
            "track", deltat=ureg.Quantity(365.0, "day"))
        b = SourceSampler(summed_detector, point_source).expected_events(
            "track", deltat=ureg.Quantity(365.0, "day"))
        assert a == pytest.approx(b, rel=1e-6)


class TestExpectedRate:
    """Public API with no coverage before this audit."""

    def test_rate_times_livetime_matches_expected_events(self, point_source_sampler):
        rate = point_source_sampler.expected_rate("track")
        one_year = ureg.Quantity(365.25, "day")
        n = point_source_sampler.expected_events("track", one_year)
        assert float((rate * one_year).to("dimensionless")) == pytest.approx(n, rel=1e-9)

    def test_multi_detector_returns_one_rate_per_detector(self, detector, point_source):
        from spore.event_sampling import SourceSampler

        sampler = SourceSampler([detector, detector], point_source)
        rate = sampler.expected_rate("track")
        assert len(rate.magnitude) == 2


class TestSmearTruthZenithFallback:
    """The joint-smearing band must not be chosen with a polar-only formula."""

    def test_non_polar_without_epoch_raises(self, joint_smearing_detector_factory):
        det = joint_smearing_detector_factory(latitude_deg=36.3)
        sc = SkyCoordinate(np.radians(20.0), np.radians(45.0))
        with pytest.raises(ValueError, match="local zenith"):
            smear_truth(sc, 1e4, det, "track", rng=np.random.default_rng(0))

    def test_non_polar_with_epoch_works(self, joint_smearing_detector_factory):
        det = joint_smearing_detector_factory(latitude_deg=36.3)
        sc = SkyCoordinate(np.radians(20.0), np.radians(45.0))
        _, reco_e, _ = smear_truth(sc, 1e4, det, "track",
                                   rng=np.random.default_rng(0), t=60355.83)
        assert np.isfinite(reco_e.magnitude)

    def test_polar_without_epoch_still_works(self, joint_smearing_detector_factory):
        det = joint_smearing_detector_factory(latitude_deg=-90.0)
        sc = SkyCoordinate(np.radians(20.0), np.radians(45.0))
        _, reco_e, _ = smear_truth(sc, 1e4, det, "track",
                                   rng=np.random.default_rng(0))
        assert np.isfinite(reco_e.magnitude)


class TestSmearingAngularUnits:
    """The smearing group stores its angular axes in degrees, not radians.

    A file written in radians indexes a valid band regardless -- every
    declination collapses into the middle one -- so it has to be rejected at
    load time rather than silently mis-banded.
    """

    def test_radian_dec_edges_rejected(self, tmp_path):
        import h5py
        from spore.detector.detector_response.detector_response import (
            smearing_sampler_from_group,
        )
        from tests.conftest import _write_response

        path = tmp_path / "radians.h5"
        _write_response(
            path,
            tabulated_values=np.ones((3, 4)),
            zeniths=np.linspace(0.1, 3.0, 4),
            energies=np.logspace(2, 5, 3),
            joint_smearing=True,
        )
        with h5py.File(path, "r+") as f:
            g = f["track/smearing"]
            deg = g["dec_edges"][:]
            del g["dec_edges"]
            g.create_dataset("dec_edges", data=np.radians(deg))
            with pytest.raises(ValueError, match="looks like radians"):
                smearing_sampler_from_group(g)

    def test_degree_dec_edges_accepted(self, joint_smearing_detector_factory):
        det = joint_smearing_detector_factory(latitude_deg=-90.0)
        assert det.response.joint_smearing is not None


class TestPoleExtrapolationDoesNotUseDeadColumns:
    """Extrapolating to cos(zen) = ±1 must not read the log-zero sentinel.

    Dead cells are stored as a large negative sentinel in log space.  When the
    outermost zenith column is live but its neighbour is dead -- which happens
    whenever the sensitive zenith range ends between two tabulated columns, as
    it does in the IceCube 14-year track release around \\SI{20}{\\TeV} -- a
    slope taken across that pair is of order (log A - sentinel) / dcz.
    Extrapolating it to the pole inflated the effective area by ten orders of
    magnitude, which in turn swamped the sampler with spurious events from the
    dead half of the sky.
    """

    @staticmethod
    def _tabulated():
        # Three zenith columns; at the lowest energies only the column nearest
        # cos(zen) = +1 is live, so its neighbour is the sentinel.
        zens = np.radians(np.array([8.0, 20.0, 26.0]))
        es = np.logspace(2, 6, 12)
        vals = np.zeros((len(es), 3))
        vals[:, 0] = 100.0 * (es / es[0]) ** 0.5      # live everywhere
        vals[6:, 1] = 50.0 * (es[6:] / es[0]) ** 0.5  # dead below index 6
        vals[6:, 2] = 25.0 * (es[6:] / es[0]) ** 0.5
        return zens, es, vals

    def test_pole_value_is_bounded_by_the_tabulated_maximum(self):
        zens, es, vals = self._tabulated()
        f = effa_helper(zens, es, vals.copy(),
                        trim_isolated=False, smoothing_sigma=0.0)
        # cos(zen) = +1 is zenith 0, just outside the tabulated grid.
        q = np.zeros(len(es))
        got = f(q, es)
        assert np.all(np.isfinite(got))
        # Never more effective area than was tabulated at that energy.
        assert np.all(got <= vals.max(axis=1) * (1.0 + 1e-9)), (
            f"pole extrapolation exceeded the tabulated maximum: "
            f"max ratio {np.max(got / np.maximum(vals.max(axis=1), 1e-30)):.3g}"
        )

    def test_upward_extrapolation_is_capped_at_the_tabulated_maximum(self):
        """Extrapolation may not invent more area than was ever tabulated.

        With both edge columns live the log slope is well defined and is still
        used, but a profile rising toward the pole would otherwise extrapolate
        past every tabulated value (here to 220.7 from an edge of 200.0).  The
        cap holds it at the row maximum.  On the real IceCube responses the row
        maximum is set by the opposite hemisphere and sits orders of magnitude
        above the edge column, so the cap does not bind and the 10-year
        response loads bit-identically.
        """
        zens = np.radians(np.array([8.0, 20.0, 26.0]))
        es = np.logspace(2, 6, 12)
        vals = np.outer(np.ones(len(es)), np.array([200.0, 120.0, 60.0]))
        f = effa_helper(zens, es, vals.copy(),
                        trim_isolated=False, smoothing_sigma=0.0)
        got = f(np.zeros(len(es)), es)
        # Rising toward cos(zen)=+1, so the pole must exceed the edge column
        # but stay capped at the row maximum (here the edge column itself).
        assert np.all(got >= 200.0 * (1.0 - 1e-9))
        assert np.all(got <= 200.0 * (1.0 + 1e-9))
