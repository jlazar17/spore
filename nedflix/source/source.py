import numpy as np

from dataclasses import dataclass
from typing import Dict

from ..conventions import SkyCoordinate
from .flux import Flux, flux_from_config

@dataclass(frozen=True)
class Source:
    location: SkyCoordinate
    flux: Flux

def source_from_config(config: Dict) -> Source:
    location = SkyCoordinate(
        np.radians(config["location"]["declination"]),
        np.radians(config["location"]["right_ascension"])
    )
    flux = flux_from_config(config["flux"])
    return Source(location, flux)
