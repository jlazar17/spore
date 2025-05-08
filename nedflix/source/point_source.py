import numpy as np

from typing import Dict

from ..conventions import SkyCoordinate
from . import Neutrino
from .source import Source
from .flux import Flux

class PointSource(Source):

    def __init__(self, flux: Flux, location: SkyCoordinate):
        self._location = location
        super().__init__(flux)

    @property
    def location(self):
        return self._location

    def __call__(self, nu: Neutrino, e: float):
        return self.flux(nu, e)

    @classmethod
    def from_config(cls, config: Dict) -> Source:
        location = SkyCoordinate(
            np.radians(config["location"]["declination"]),
            np.radians(config["location"]["right_ascension"])
        )
        flux = Flux.from_config(config["flux"])
        return cls(flux, location)
