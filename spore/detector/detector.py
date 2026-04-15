import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from ..conventions import EarthCoordinate
from .detector_response import DetectorResponse

class Medium(Enum):
    Ice = 1
    Water = 2

@dataclass(frozen=True)
class Detector:
    """A neutrino detector defined by its location, medium, and IRFs.

    Attributes:
        location: Geodetic position of the detector.
        medium: Detection medium (Ice or Water).
        response: Instrument response functions (effective area, PSF, energy
            resolution).
    """
    location: EarthCoordinate
    medium: Medium
    response: DetectorResponse

    @classmethod
    def from_config(cls, config: Dict, trim_isolated: Optional[bool] = None, smoothing_sigma: Optional[float] = None) -> 'Detector':
        """Build a Detector from a config dictionary.

        Args:
            config: Dictionary with a ``properties`` sub-dict (keys:
                latitude, longitude in degrees; medium as "Ice" or "Water")
                and a ``response`` sub-dict accepted by
                DetectorResponse.from_config.
            trim_isolated: If True, zero out energy bins outside the largest
                contiguous non-zero run per zenith column when loading the
                effective area.  If ``None`` (default), the value is read from
                the HDF5 file's ``meta`` group if present, otherwise defaults
                to ``True``.  Pass an explicit bool to override the file
                metadata.
            smoothing_sigma: Width (in energy bins) of the Gaussian kernel
                used to smooth MC statistical noise in the effective area
                high-energy tail.  If ``None`` (default), the value is read
                from the HDF5 file's ``meta`` group if present, otherwise the
                code default is used.  Pass an explicit float to override.
                Set to 0 to disable smoothing entirely.

        Returns:
            A configured Detector instance.
        """
        location = EarthCoordinate(
            np.radians(config["properties"]["latitude"]),
            np.radians(config["properties"]["longitude"]),
        )
        medium = getattr(Medium, config["properties"]["medium"])
        detector_response = DetectorResponse.from_config(
            config["response"], trim_isolated=trim_isolated, smoothing_sigma=smoothing_sigma
        )
        return cls(location, medium, detector_response)

