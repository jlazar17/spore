from dataclasses import dataclass

from .coordinates import SkyCoordinate

@dataclass(frozen=False)
class Event:
    interaction: InteractionType    
    initial_neutrino: Neutrino
    true_direction: SkyCoordinate
    reco_direction: SkyCoordinate
    true_energy: float
    reco_energy: float
