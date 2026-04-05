from enum import Enum
from dataclasses import dataclass

class Flavor(Enum):
    """Neutrino lepton flavor."""
    Electron = 1
    Muon = 2
    Tau = 3

class NeutrinoType(Enum):
    """Particle vs. antiparticle distinction."""
    Nu = 1
    NuBar = 2

@dataclass(frozen=True)
class NeutrinoDef:
    """Internal definition of a single neutrino species.

    Attributes:
        pdg_id: PDG Monte Carlo particle ID (e.g. 12 for νe, -12 for ν̄e).
        nutype: Particle or antiparticle.
        flavor: Lepton flavor.
    """
    pdg_id: int
    nutype: NeutrinoType
    flavor: Flavor

    def __int__(self) -> int:
        return self.pdg_id


class Neutrino(Enum):
    """Enumeration of the six standard neutrino species.

    Each member holds a NeutrinoDef with PDG id, type, and flavor.
    Use ``neutrinos`` for the canonical ordered list of all six species.
    """
    NuE = NeutrinoDef(12, NeutrinoType.Nu, Flavor.Electron)
    NuMu = NeutrinoDef(14, NeutrinoType.Nu, Flavor.Muon)
    NuTau = NeutrinoDef(16, NeutrinoType.Nu, Flavor.Tau)
    NuEBar = NeutrinoDef(-12, NeutrinoType.NuBar, Flavor.Electron)
    NuMuBar = NeutrinoDef(-14, NeutrinoType.NuBar, Flavor.Muon)
    NuTauBar = NeutrinoDef(-16, NeutrinoType.NuBar, Flavor.Tau)


    def __int__(self):
        return int(self.value)
neutrinos = [Neutrino.NuE, Neutrino.NuEBar, Neutrino.NuMu, Neutrino.NuMuBar, Neutrino.NuTau, Neutrino.NuTauBar]