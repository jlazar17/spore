import numpy as np

from dataclasses import dataclass
from typing import Tuple
from enum import Enum

from .coordinates import SkyCoordinate

class InteractionType(Enum):
    ChargedCurrent: 1
    NeutralCurrent: 2

@dataclass(frozen=False)
class Event:
    interaction: InteractionType    
    initial_neutrino: Neutrino
    true_direction: SkyCoordinate
    reco_direction: SkyCoordinate
    true_energy: float
    reco_energy: float
