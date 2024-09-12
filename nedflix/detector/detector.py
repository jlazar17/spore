import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ..coordinates import EarthCoordinate
from ..neutrino import Flavor, NeutrinoType, neutrinos
from . import units
from .detector_response import DetectorResponse, detector_response_from_config

def Medium(Enum):
    Ice: 1
    Water: 2

@dataclass(frozen=True)
class Detector:
    location: EarthCoordinate
    depth: float
    medium: Medium
    response: DetectorResponse

def detector_from_config(config: Dict) -> Detector:
    location = EarthCoordinate(
        config["properties"]["latitude"],
        config["properties"]["longitude"],
    )
    depth = config["properties"]["depth"] * units.m
    medium = getattr(Medium, config["properties"]["medium"])
    detector_response = detector_response_from_config(config["response"])
    return Detector(location, depth, medium, detector_response)
