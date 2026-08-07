"""Tests for local coordinate utilities and Event zenith/azimuth attributes.

Covers:
- _mjd_to_lst: J2000 epoch, output range, longitude offset
- _local_zeniths_from_times: transit geometry, polar detector
- _local_coords_from_times: zenith agreement with _local_zeniths_from_times,
  azimuth range and transit values, hour-angle symmetry
- Event.zenith / Event.azimuth: pint Quantity wrapping, NaN defaults
"""

import math
import numpy as np
import pytest

from spore.conventions import SkyCoordinate
from spore.event_sampling.event import Event
from spore.event_sampling.utils import (
    _mjd_to_lst,
    _local_zeniths_from_times,
    _local_coords_from_times,
)

# J2000 epoch: JD 2451545.0 = 2000-01-01 12:00 = MJD 51544.5.
# Imported from the module rather than restated, so this file cannot drift
# from the constant the GMST polynomial is actually expanded about.
from spore.event_sampling.utils import _J2000_MJD


# ---------------------------------------------------------------------------
# _mjd_to_lst
# ---------------------------------------------------------------------------

class TestMjdToLst:
    def test_output_in_range(self):
        mjds = np.linspace(51545.0, 51545.0 + 365, 100)
        lsts = _mjd_to_lst(mjds, 0.0)
        assert np.all(lsts >= 0.0)
        assert np.all(lsts < 2.0 * np.pi)

    def test_j2000_epoch_lon0(self):
        gmst_hours = 18.697374558
        expected = (gmst_hours / 24.0) * 2.0 * np.pi
        result = _mjd_to_lst(np.array([_J2000_MJD]), 0.0)
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_longitude_offset_applied(self):
        lon = np.pi / 4
        lst_lon0 = _mjd_to_lst(np.array([_J2000_MJD]), 0.0)[0]
        lst_lonx = _mjd_to_lst(np.array([_J2000_MJD]), lon)[0]
        assert lst_lonx == pytest.approx((lst_lon0 + lon) % (2 * np.pi), rel=1e-6)

    def test_scalar_input_works(self):
        result = _mjd_to_lst(_J2000_MJD, 0.0)
        assert result.shape == ()

    def test_array_output_matches_scalar_loop(self):
        mjds = np.linspace(_J2000_MJD, _J2000_MJD + 30, 20)
        lon = 0.5
        arr_result = _mjd_to_lst(mjds, lon)
        scalar_results = np.array([float(_mjd_to_lst(np.array([m]), lon)[0]) for m in mjds])
        np.testing.assert_allclose(arr_result, scalar_results, rtol=1e-10)


# ---------------------------------------------------------------------------
# _local_zeniths_from_times
# ---------------------------------------------------------------------------

class TestLocalZenithsFromTimes:
    def test_output_shape(self):
        n = 10
        times = np.full(n, _J2000_MJD)
        decs  = np.zeros(n)
        ras   = np.zeros(n)
        zens  = _local_zeniths_from_times(times, decs, ras, lat=0.0, lon=0.0)
        assert zens.shape == (n,)

    def test_zenith_in_valid_range(self):
        rng = np.random.default_rng(0)
        n = 200
        times = rng.uniform(_J2000_MJD, _J2000_MJD + 365, n)
        decs  = rng.uniform(-np.pi / 2, np.pi / 2, n)
        ras   = rng.uniform(0, 2 * np.pi, n)
        lat   = np.radians(-35.0)
        lon   = np.radians(20.0)
        zens  = _local_zeniths_from_times(times, decs, ras, lat, lon)
        assert np.all(zens >= 0.0)
        assert np.all(zens <= np.pi)

    def test_transit_geometry(self):
        # At transit HA=0: cos(zen) = sin(lat)*sin(dec) + cos(lat)*cos(dec)
        # Choose lat and dec so HA=0 occurs at J2000 MJD for lon=0.
        lat = np.radians(45.0)
        lon = 0.0
        lst0 = float(_mjd_to_lst(np.array([_J2000_MJD]), lon)[0])
        # Place source RA = LST so HA=0 at J2000.
        dec = np.radians(30.0)
        ra  = lst0
        n   = 1
        times = np.array([_J2000_MJD])
        zens = _local_zeniths_from_times(times, np.array([dec]), np.array([ra]), lat, lon)
        expected = np.arccos(
            np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec)
        )
        assert zens[0] == pytest.approx(expected, rel=1e-5)

    def test_south_pole_zenith_independent_of_ra(self):
        # At lat=-π/2: cos(zen) = -sin(dec), independent of HA.
        lat = -np.pi / 2
        lon = 0.0
        dec = np.radians(-30.0)
        n   = 10
        ras   = np.linspace(0, 2 * np.pi, n, endpoint=False)
        times = np.full(n, _J2000_MJD)
        decs  = np.full(n, dec)
        zens  = _local_zeniths_from_times(times, decs, ras, lat, lon)
        expected = float(np.arccos(np.clip(-np.sin(dec), -1, 1)))
        np.testing.assert_allclose(zens, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# _local_coords_from_times
# ---------------------------------------------------------------------------

class TestLocalCoordsFromTimes:
    def test_output_shapes(self):
        n = 15
        times = np.full(n, _J2000_MJD)
        decs  = np.zeros(n)
        ras   = np.zeros(n)
        zens, azs = _local_coords_from_times(times, decs, ras, lat=0.0, lon=0.0)
        assert zens.shape == (n,)
        assert azs.shape  == (n,)

    def test_zenith_matches_local_zeniths_from_times(self):
        rng   = np.random.default_rng(1)
        n     = 50
        times = rng.uniform(_J2000_MJD, _J2000_MJD + 365, n)
        decs  = rng.uniform(-np.pi / 2, np.pi / 2, n)
        ras   = rng.uniform(0, 2 * np.pi, n)
        lat   = np.radians(-26.0)
        lon   = np.radians(31.0)
        zens_new, _ = _local_coords_from_times(times, decs, ras, lat, lon)
        zens_ref    = _local_zeniths_from_times(times, decs, ras, lat, lon)
        np.testing.assert_allclose(zens_new, zens_ref, atol=1e-12)

    def test_azimuth_in_valid_range(self):
        rng   = np.random.default_rng(2)
        n     = 200
        times = rng.uniform(_J2000_MJD, _J2000_MJD + 365, n)
        decs  = rng.uniform(-np.pi / 2, np.pi / 2, n)
        ras   = rng.uniform(0, 2 * np.pi, n)
        lat   = np.radians(0.0)
        lon   = np.radians(0.0)
        _, azs = _local_coords_from_times(times, decs, ras, lat, lon)
        assert np.all(azs >= 0.0)
        assert np.all(azs < 2.0 * np.pi)

    def test_zenith_in_valid_range(self):
        rng   = np.random.default_rng(3)
        n     = 200
        times = rng.uniform(_J2000_MJD, _J2000_MJD + 365, n)
        decs  = rng.uniform(-np.pi / 2, np.pi / 2, n)
        ras   = rng.uniform(0, 2 * np.pi, n)
        lat   = np.radians(-35.0)
        lon   = np.radians(0.0)
        zens, _ = _local_coords_from_times(times, decs, ras, lat, lon)
        assert np.all(zens >= 0.0)
        assert np.all(zens <= np.pi)

    def test_transit_azimuth_source_above_lat(self):
        # At transit (HA=0) for dec > lat: source transits to the North → azimuth = 0.
        lat = np.radians(0.0)   # equator
        lon = 0.0
        dec = np.radians(30.0)  # dec > lat
        lst0 = float(_mjd_to_lst(np.array([_J2000_MJD]), lon)[0])
        ra   = lst0  # HA = LST - RA = 0 at J2000
        _, azs = _local_coords_from_times(
            np.array([_J2000_MJD]), np.array([dec]), np.array([ra]), lat, lon
        )
        # sin_az = -cos(dec)*sin(0) = 0; cos_az = sin(dec-lat) > 0 → North = 0
        assert azs[0] == pytest.approx(0.0, abs=1e-6)

    def test_transit_azimuth_source_below_lat(self):
        # At transit (HA=0) for dec < lat: source transits to the South → azimuth = π.
        lat = np.radians(60.0)
        lon = 0.0
        dec = np.radians(10.0)  # dec < lat
        lst0 = float(_mjd_to_lst(np.array([_J2000_MJD]), lon)[0])
        ra   = lst0
        _, azs = _local_coords_from_times(
            np.array([_J2000_MJD]), np.array([dec]), np.array([ra]), lat, lon
        )
        # sin_az = 0; cos_az = sin(dec-lat) < 0 → South = π
        assert azs[0] == pytest.approx(np.pi, abs=1e-6)

    def test_hour_angle_symmetry(self):
        # HA and -HA should give mirror-image azimuths: az(-HA) = 2π - az(HA).
        # We achieve HA_target and -HA_target by setting RA = LST - HA_target
        # and RA = LST + HA_target respectively.
        lat = np.radians(30.0)
        lon = 0.0
        lst0 = float(_mjd_to_lst(np.array([_J2000_MJD]), lon)[0])
        dec  = np.radians(20.0)
        ha   = np.radians(45.0)

        ra_plus  = (lst0 - ha)  % (2 * np.pi)
        ra_minus = (lst0 + ha)  % (2 * np.pi)

        _, az_plus  = _local_coords_from_times(
            np.array([_J2000_MJD]), np.array([dec]), np.array([ra_plus]),  lat, lon
        )
        _, az_minus = _local_coords_from_times(
            np.array([_J2000_MJD]), np.array([dec]), np.array([ra_minus]), lat, lon
        )
        # Azimuths should sum to 2π (mod 2π).
        total = (az_plus[0] + az_minus[0]) % (2 * np.pi)
        assert total == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Event zenith / azimuth properties
# ---------------------------------------------------------------------------

class TestEventLocalCoordProperties:
    def _make_event(self, zenith=None, azimuth=None):
        sc = SkyCoordinate(0.0, 0.0)
        return Event(sc, sc, 1e13, 1e13, 60355.0, "track",
                     zenith=zenith, azimuth=azimuth)

    def test_zenith_nan_by_default(self):
        e = self._make_event()
        assert math.isnan(e._zenith_rad)

    def test_azimuth_nan_by_default(self):
        e = self._make_event()
        assert math.isnan(e._azimuth_rad)

    def test_zenith_property_returns_pint_quantity(self):
        import pint
        e = self._make_event(zenith=1.2, azimuth=2.3)
        assert isinstance(e.zenith, pint.Quantity)

    def test_azimuth_property_returns_pint_quantity(self):
        import pint
        e = self._make_event(zenith=1.2, azimuth=2.3)
        assert isinstance(e.azimuth, pint.Quantity)

    def test_zenith_property_units_are_radians(self):
        e = self._make_event(zenith=1.2, azimuth=0.0)
        assert str(e.zenith.units) == "radian"

    def test_azimuth_property_units_are_radians(self):
        e = self._make_event(zenith=0.0, azimuth=2.3)
        assert str(e.azimuth.units) == "radian"

    def test_zenith_magnitude_matches_constructor(self):
        e = self._make_event(zenith=1.234, azimuth=0.0)
        assert e.zenith.magnitude == pytest.approx(1.234)

    def test_azimuth_magnitude_matches_constructor(self):
        e = self._make_event(zenith=0.0, azimuth=5.678)
        assert e.azimuth.magnitude == pytest.approx(5.678)

    def test_zenith_stored_in_slot(self):
        e = self._make_event(zenith=0.9, azimuth=0.0)
        assert e._zenith_rad == pytest.approx(0.9)

    def test_azimuth_stored_in_slot(self):
        e = self._make_event(zenith=0.0, azimuth=3.1)
        assert e._azimuth_rad == pytest.approx(3.1)

    def test_pint_quantity_zenith_accepted(self):
        from spore.conventions import ureg
        e = self._make_event(zenith=ureg.Quantity(0.5, "rad"))
        assert e._zenith_rad == pytest.approx(0.5)

    def test_pint_quantity_azimuth_accepted(self):
        from spore.conventions import ureg
        e = self._make_event(azimuth=ureg.Quantity(1.5, "rad"))
        assert e._azimuth_rad == pytest.approx(1.5)
