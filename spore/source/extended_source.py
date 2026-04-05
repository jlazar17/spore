import numpy as np

from typing import Dict

from . import Neutrino
from .source import Source
from .flux import Flux

class ExtendedSource(Source):
    """A spatially extended neutrino source with a 2D flux model.

    The flux may depend on both energy and declination, as provided by a
    UserProvidedDist2D or similar 2D Distribution.

    Args:
        flux: The neutrino flux model. Must support evaluation with a
            declination argument.
    """

    def __init__(self, flux: Flux):
        super().__init__(flux)

    def __call__(self, nu: Neutrino, e: float, dec: float) -> float:
        return self.flux(nu, e, dec)

    @classmethod
    def from_config(cls, config: Dict) -> 'ExtendedSource':
        """Build an ExtendedSource from a config dictionary.

        Args:
            config: Dictionary with a ``flux`` sub-dict accepted by
                Flux.from_config. A ``location`` key is ignored with a
                warning since extended sources span the sky.

        Returns:
            A configured ExtendedSource instance.
        """
        if "location" in config.keys():
            from warnings import warn
            warn("config has location information for surrounding source")

        flux = Flux.from_config(config["flux"])
        return cls(flux)
