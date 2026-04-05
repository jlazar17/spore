import warnings

import numpy as np
import pytest

from spore.conventions import SkyCoordinate, LocalCoordinate, EarthCoordinate
from spore.conventions.utils import orthonormal_basis, sample_cone, sample_ring, sky_to_local


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


class TestOrthonormalBasis:
    def test_output_vectors_are_unit_length(self):
        v = np.array([0.0, 0.0, 1.0])
        u, w = orthonormal_basis(v.copy())
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert np.linalg.norm(w) == pytest.approx(1.0)

    def test_output_vectors_are_orthogonal(self):
        v = np.array([1.0, 0.0, 0.0])
        u, w = orthonormal_basis(v.copy())
        assert np.dot(u, w) == pytest.approx(0.0, abs=1e-10)

    def test_non_axis_aligned_input(self):
        v = np.array([1.0, 1.0, 1.0])
        u, w = orthonormal_basis(v.copy())
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert np.linalg.norm(w) == pytest.approx(1.0)
        assert np.dot(u, w) == pytest.approx(0.0, abs=1e-10)

    def test_wrong_dimension_raises(self):
        with pytest.raises(ValueError):
            orthonormal_basis(np.array([1.0, 0.0]))


class TestSampleRing:
    def test_output_is_unit_vector(self):
        v = np.array([0.0, 0.0, 1.0])
        result = sample_ring(v.copy(), 0.3)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-10)

    def test_angle_from_input_equals_psi(self):
        v = np.array([0.0, 0.0, 1.0])
        psi = 0.4
        result = sample_ring(v.copy(), psi)
        cos_angle = np.clip(np.dot(np.array([0.0, 0.0, 1.0]), result), -1.0, 1.0)
        assert np.arccos(cos_angle) == pytest.approx(psi, abs=1e-10)

    def test_zero_angle_returns_original_direction(self):
        v = np.array([1.0, 0.0, 0.0])
        result = sample_ring(v.copy(), 0.0)
        assert result == pytest.approx([1.0, 0.0, 0.0], abs=1e-10)


class TestSampleCone:
    def test_returns_sky_coordinate(self):
        sc = SkyCoordinate(0.0, 0.0)
        result = sample_cone(sc, 0.1)
        assert isinstance(result, SkyCoordinate)

    def test_zero_angle_returns_same_direction(self):
        sc = SkyCoordinate(0.3, 1.0)
        result = sample_cone(sc, 0.0)
        assert result.declination == pytest.approx(sc.declination, abs=1e-10)
        assert result.right_ascension == pytest.approx(sc.right_ascension, abs=1e-10)

    def test_angular_distance_equals_psi(self):
        sc = SkyCoordinate(0.0, 0.0)
        psi = 0.3
        result = sample_cone(sc, psi)
        cos_angle = np.clip(np.dot(sc.to_cartesian(), result.to_cartesian()), -1.0, 1.0)
        assert np.arccos(cos_angle) == pytest.approx(psi, abs=1e-10)

    def test_result_is_valid_sky_coordinate(self):
        sc = SkyCoordinate(0.5, 2.0)
        for _ in range(20):
            result = sample_cone(sc, 0.2)
            assert -np.pi / 2 <= result.declination <= np.pi / 2
            assert 0.0 <= result.right_ascension <= 2 * np.pi
