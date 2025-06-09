from dataclasses import dataclass

from ..conventions import SkyCoordinate

@dataclass
class Event:
    true_direction: SkyCoordinate
    reco_direction: SkyCoordinate
    true_energy: float
    reco_energy: float
    signalness: float
    time: float
    morphology: int
