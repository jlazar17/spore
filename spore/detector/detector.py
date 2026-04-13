import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Dict

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
    def from_config(cls, config: Dict) -> 'Detector':
        """Build a Detector from a config dictionary.

        Args:
            config: Dictionary with a ``properties`` sub-dict (keys:
                latitude, longitude in degrees; medium as "Ice" or "Water")
                and a ``response`` sub-dict accepted by
                DetectorResponse.from_config.

        Returns:
            A configured Detector instance.
        """
        location = EarthCoordinate(
            np.radians(config["properties"]["latitude"]),
            np.radians(config["properties"]["longitude"]),
        )
        medium = getattr(Medium, config["properties"]["medium"])
        detector_response = DetectorResponse.from_config(config["response"])
        return cls(location, medium, detector_response)

    @classmethod
    def from_toml(cls, path: str) -> 'Detector':
        """
        Build a Detector from a single TOML file.

        The TOML must contain a ``[properties]`` table and a ``[response]``
        table.  The ``[response]`` table must have a ``toml`` key pointing
        to a spore detector-response TOML file.  That path is resolved
        relative to the detector TOML's directory first, then relative to
        the current working directory, then as an absolute path.

        Parameters
        ----------
        path : str
            Path to the detector TOML file.

        Example TOML
        ------------
        .. code-block:: toml

            [properties]
            latitude  = -90.0
            longitude =   0.0
                medium    = "Ice"

            [response]
            toml = "ps10yr_response.toml"

        Example usage
        -------------
        det = Detector.from_toml("resources/configs/icecube_detector.toml")
        """
        import tomllib, os
        from ..conventions import resolve_path
        from .detector_response import DetectorResponse

        path = os.path.abspath(path)
        toml_dir = os.path.dirname(path)
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)

        location = EarthCoordinate(
            np.radians(cfg["properties"]["latitude"]),
            np.radians(cfg["properties"]["longitude"]),
        )
        medium = getattr(Medium, cfg["properties"]["medium"])
        response_path = resolve_path(cfg["response"]["toml"], toml_dir)
        response = DetectorResponse.from_toml(response_path)
        return cls(location, medium, response)

    @classmethod
    def from_dataverse(
        cls,
        properties: Dict,
        irf_dir: str,
        seasons,
        base_h5: str = None,
        h5_path: str = None,
    ):
        """
        Build a Detector whose smearing IRF comes from the IceCube data-release
        CSV files rather than a pre-built HDF5.

        Parameters
        ----------
        properties : dict
            Same ``properties`` sub-dict as used in from_config, e.g.
            ``{"latitude": -90.0, "longitude": 0.0, "medium": "Ice"}``.
        irf_dir : str
            Directory containing the data-release IRF CSV files.
        seasons : list of str
            Seasons to include, e.g. ``['IC86_I', 'IC86_II']``.
        base_h5 : str or None
            Existing spore HDF5 response file to copy effective area and
            1-D smearing groups from.
        h5_path : str or None
            If given, write the combined response to this HDF5 path so it can
            be reloaded later with from_config().

        Example
        -------
        det = Detector.from_dataverse(
            properties={"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
            irf_dir="~/Downloads/dataverse_files 2/irfs",
            seasons=["IC86_I", "IC86_II"],
            base_h5="resources/example_detector_response.h5",
            h5_path="resources/icecube_10yr_response.h5",
        )
        """
        location = EarthCoordinate(
            np.radians(properties["latitude"]),
            np.radians(properties["longitude"]),
        )
        medium = getattr(Medium, properties["medium"])
        response = DetectorResponse.from_dataverse(
            irf_dir=irf_dir,
            seasons=seasons,
            base_h5=base_h5,
            h5_path=h5_path,
        )
        return cls(location, medium, response)
