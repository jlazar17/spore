import logging
import numpy as np

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LocalCoordinate:
    """Detector-local coordinate expressed as zenith and azimuth angles (radians).

    Attributes:
        zenith: Zenith angle in radians, in [0, π].
        azimuth: Azimuth angle in radians, wrapped to [0, 2π).
    """
    zenith: float
    azimuth: float

    def __post_init__(self):
        if self.zenith < 0 or np.pi < self.zenith:
            raise ValueError("zenith out of range")
        if self.azimuth < 0 or 2*np.pi < self.azimuth:
            logger.warning("Azimuth %.6f out of [0, 2π); wrapping.", self.azimuth)
            self.azimuth = self.azimuth % (2*np.pi)

@dataclass
class SkyCoordinate:
    """Equatorial sky coordinate (radians).

    Attributes:
        declination: Declination in radians, in [-π/2, π/2].
        right_ascension: Right ascension in radians, wrapped to [0, 2π).
    """
    declination: float
    right_ascension: float

    def __post_init__(self):
        if self.declination < -np.pi / 2 or np.pi / 2 < self.declination:
            raise ValueError("declination out of range")
        if self.right_ascension < 0 or 2*np.pi < self.right_ascension:
            logger.warning("Right ascension %.6f out of [0, 2π); wrapping.", self.right_ascension)
            self.right_ascension = self.right_ascension % (2*np.pi)

    def to_cartesian(self) -> np.ndarray:
        """Convert to a unit Cartesian vector on the celestial sphere.

        Returns:
            Unit vector [x, y, z] with x = cos(dec)cos(ra),
            y = cos(dec)sin(ra), z = sin(dec).
        """
        x = np.cos(self.declination) * np.cos(self.right_ascension)
        y = np.cos(self.declination) * np.sin(self.right_ascension)
        z = np.sin(self.declination)
        return np.array([x, y, z])

    def separation(self, other: 'SkyCoordinate') -> float:
        """Great-circle angular separation from another sky coordinate.

        Uses the cross-product formula, which is numerically stable for
        both very small and very large separations.

        Args:
            other: The other sky coordinate.

        Returns:
            Separation in radians, in [0, π].
        """
        a = self.to_cartesian()
        b = other.to_cartesian()
        cross = np.linalg.norm(np.cross(a, b))
        dot   = np.dot(a, b)
        return float(np.arctan2(cross, dot))
        
@dataclass
class EarthCoordinate:
    """Geodetic surface coordinate of a detector (radians).

    Attributes:
        latitude: Geodetic latitude in radians, in [-π/2, π/2].
        longitude: Geodetic longitude in radians, in [-π, π].
    """
    latitude: float
    longitude: float

    def __post_init__(self):
        if self.latitude < -np.pi / 2 or np.pi / 2 < self.latitude:
            raise ValueError("latitude out of range")
        if self.longitude < -np.pi or np.pi < self.longitude:
            raise ValueError("longitude out of range")
