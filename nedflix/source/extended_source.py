import numpy as np

from typing import Dict

from . import Neutrino
from .source import Source
from .flux import Flux

class ExtendedSource(Source):

    def __init__(self, flux: Flux):
        super().__init__(flux)

    def __call__(self, nu: Neutrino, e: float, dec: float) -> float:
        return self.flux(nu, e, dec)

    @classmethod
    def from_config(cls, config: Dict):
        if "location" in config.keys():
            from warnings import warn
            warn("config has location information for surrounding source")

        flux = Flux.from_config(config["flux"])
        return cls(flux)
