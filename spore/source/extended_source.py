
from typing import Dict

from . import Neutrino
from .source import Source
from .flux import Flux
from ..config import load_config

class ExtendedSource(Source):
    """A spatially extended neutrino source with a 2D or 3D flux model.

    The flux may depend on energy and declination (2D), or on energy,
    declination, and right ascension (3D).  When the underlying flux uses a
    TabulatedEnergyDecRAFlux distribution, uses_ra is automatically True and
    the sampler will build a full (dec, RA, energy) grid.

    Args:
        flux: The neutrino flux model.
    """

    def __init__(self, flux: Flux):
        super().__init__(flux)

    @property
    def uses_ra(self) -> bool:
        """True if the flux requires a right-ascension argument."""
        return self.flux.uses_ra

    def __call__(self, nu: Neutrino, e: float, dec: float, ra: float = None) -> float:
        return self.flux(nu, e, dec, ra)

    @classmethod
    def from_config(cls, config: Dict, normalization: float = 1.0) -> 'ExtendedSource':
        """Build an ExtendedSource from a config dictionary.

        Args:
            config: Dictionary with a ``flux`` sub-dict accepted by
                Flux.from_config. A ``location`` key is ignored with a
                warning since extended sources span the sky.
            normalization: Global multiplicative scaling applied to the flux.
                Default 1.0 (no scaling).

        Returns:
            A configured ExtendedSource instance.
        """
        config = load_config(config)
        if "location" in config.keys():
            import logging
            logging.getLogger(__name__).warning(
                "Config contains a 'location' key, which is ignored for extended sources."
            )

        flux = Flux.from_config(config["flux"], normalization=normalization)
        return cls(flux)
