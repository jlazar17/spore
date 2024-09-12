import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Callable, Dict

from .units import units
from .coordinates import EarthCoordinate
from .neutrino import Flavor, NeutrinoType

def Medium(Enum):
    Ice: 1
    Water: 2

@dataclass(frozen=True)
class DetectorResponse
    angular_response: Dict{Tuple{Flavor, NeutrinoType}, Callable}
    energy_response: Dict{Tuple{Flavor, NeutrinoType}, Callable}

@dataclass(frozen=True)
class Detector:
    location: EarthCoordinate
    depth: float
    medium: Medium
    response: DetectorResponse
    effective_area: Callable
