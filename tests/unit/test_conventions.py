import warnings

import numpy as np
import pytest

from spore.conventions import SkyCoordinate, LocalCoordinate, EarthCoordinate
from spore.conventions.utils import sky_to_local


class TestLocalCoordinate:
    def test_valid_construction(self):
        lc = LocalCoordinate(zenith=np.pi / 4, azimuth=np.pi)
        assert lc.zenith == pytest.approx(np.pi / 4)
        assert lc.azimuth == pytest.approx(np.pi)

    def test_zenith_too_large_raises(self):
        with pytest.raises(ValueError):
            LocalCoordinate(zenith=np.pi + 0.1, azimuth=0.0)

    def test_zenith_negative_raises(self):
        with pytest.raises(ValueError):
            LocalCoordinate(zenith=-0.1, azimuth=0.0)

    def test_zenith_boundary_values_are_valid(self):
        LocalCoordinate(zenith=0.0, azimuth=0.0)
        LocalCoordinate(zenith=np.pi, azimuth=0.0)

    def test_azimuth_out_of_range_is_corrected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = LocalCoordinate(zenith=1.0, azimuth=3 * np.pi)
        assert 0.0 <= lc.azimuth <= 2 * np.pi


class TestSkyCoordinate:
    def test_valid_construction(self):
        sc = SkyCoordinate(declination=0.3, right_ascension=np.pi)
        assert sc.declination == pytest.approx(0.3)
        assert sc.right_ascension == pytest.approx(np.pi)

    def test_declination_too_large_raises(self):
        with pytest.raises(ValueError):
            SkyCoordinate(declination=np.pi, right_ascension=0.0)

    def test_declination_too_small_raises(self):
        with pytest.raises(ValueError):
            SkyCoordinate(declination=-np.pi, right_ascension=0.0)

    def test_declination_boundary_values_are_valid(self):
        SkyCoordinate(declination=np.pi / 2, right_ascension=0.0)
        SkyCoordinate(declination=-np.pi / 2, right_ascension=0.0)

    def test_ra_out_of_range_is_corrected(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = SkyCoordinate(declination=0.0, right_ascension=3 * np.pi)
        assert 0.0 <= sc.right_ascension <= 2 * np.pi

    def test_to_cartesian_shape(self):
        sc = SkyCoordinate(0.0, 0.0)
        v = sc.to_cartesian()
        assert v.shape == (3,)

    def test_to_cartesian_is_unit_vector(self):
        sc = SkyCoordinate(0.3, 1.2)
        assert np.linalg.norm(sc.to_cartesian()) == pytest.approx(1.0)

    def test_to_cartesian_north_pole(self):
        north = SkyCoordinate(np.pi / 2, 0.0)
        assert north.to_cartesian() == pytest.approx([0.0, 0.0, 1.0], abs=1e-10)

    def test_to_cartesian_south_pole(self):
        south = SkyCoordinate(-np.pi / 2, 0.0)
        assert south.to_cartesian() == pytest.approx([0.0, 0.0, -1.0], abs=1e-10)


class TestEarthCoordinate:
    def test_valid_construction(self):
        ec = EarthCoordinate(latitude=-np.pi / 4, longitude=np.pi / 2)
        assert ec.latitude == pytest.approx(-np.pi / 4)

    def test_latitude_too_large_raises(self):
        with pytest.raises(ValueError):
            EarthCoordinate(latitude=np.pi, longitude=0.0)

    def test_latitude_too_small_raises(self):
        with pytest.raises(ValueError):
            EarthCoordinate(latitude=-np.pi, longitude=0.0)

    def test_longitude_too_large_raises(self):
        with pytest.raises(ValueError):
            EarthCoordinate(latitude=0.0, longitude=np.pi + 0.1)

    def test_longitude_too_small_raises(self):
        with pytest.raises(ValueError):
            EarthCoordinate(latitude=0.0, longitude=-np.pi - 0.1)

    def test_boundary_values_are_valid(self):
        EarthCoordinate(latitude=np.pi / 2, longitude=np.pi)
        EarthCoordinate(latitude=-np.pi / 2, longitude=-np.pi)


class TestSkyToLocal:
    def test_returns_local_coordinate(self):
        sc = SkyCoordinate(0.0, 0.0)
        ec = EarthCoordinate(-np.pi / 2, 0.0)
        lc = sky_to_local(sc, ec, t=60355.0)
        assert isinstance(lc, LocalCoordinate)

    def test_zenith_in_valid_range(self):
        sc = SkyCoordinate(-np.pi / 4, 1.0)
        ec = EarthCoordinate(-np.pi / 2, 0.0)
        lc = sky_to_local(sc, ec, t=60355.0)
        assert 0.0 <= lc.zenith <= np.pi

    def test_azimuth_in_valid_range(self):
        sc = SkyCoordinate(-np.pi / 4, 1.0)
        ec = EarthCoordinate(-np.pi / 2, 0.0)
        lc = sky_to_local(sc, ec, t=60355.0)
        assert 0.0 <= lc.azimuth <= 2 * np.pi


