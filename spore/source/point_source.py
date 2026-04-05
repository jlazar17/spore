import numpy as np

from typing import Dict

from ..conventions import SkyCoordinate
from . import Neutrino
from .source import Source
from .flux import Flux

class PointSource(Source):
    """A neutrino point source at a fixed sky location.

    Args:
        flux: The neutrino flux model for this source.
        location: The equatorial sky coordinate of the source.
    """

    def __init__(self, flux: Flux, location: SkyCoordinate):
        self._location = location
        super().__init__(flux)

    @property
    def location(self) -> SkyCoordinate:
        """Sky coordinate of the source."""
        return self._location

    def __call__(self, nu: Neutrino, e: float):
        return self.flux(nu, e)

    @classmethod
    def from_toml(cls, path: str) -> 'PointSource':
        """
        Build a PointSource from a TOML file.

        The TOML must contain a ``[location]`` table with
        ``right_ascension`` and ``declination`` (degrees), and a
        ``[flux]`` table accepted by ``Flux.from_config``.

        Parameters
        ----------
        path : str
            Path to the source TOML file.

        Example
        -------
        src = PointSource.from_toml("resources/configs/my_source.toml")
        """
        import tomllib, os
        path = os.path.abspath(path)
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        return cls.from_config(cfg)

    @classmethod
    def from_config(cls, config: Dict) -> 'PointSource':
        """Build a PointSource from a config dictionary.

        Args:
            config: Dictionary with a ``location`` sub-dict (keys:
                ``declination``, ``right_ascension`` in degrees) and a
                ``flux`` sub-dict accepted by Flux.from_config.

        Returns:
            A configured PointSource instance.
        """
        location = SkyCoordinate(
            np.radians(config["location"]["declination"]),
            np.radians(config["location"]["right_ascension"])
        )
        flux = Flux.from_config(config["flux"])
        return cls(flux, location)
