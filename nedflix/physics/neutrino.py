from enum import Enum

class Flavor(Enum):
    Electron = 1
    Muon = 2
    Tau = 3

class NeutrinoType(Enum):
    Nu = 1
    NuBar = 2

@dataclass(frozen=True)
class NeutrinoDef:
    pdg_id: int
    nutype: NeutrinoType
    flavor: Flavor

    def __int__(self) -> int:
        return self.pdg_id


class Neutrino(Enum):
    NuE = NeutrinoDef(12, NeutrinoType.Nu, Flavor.Electron)
    NuMu = NeutrinoDef(14, NeutrinoType.Nu, Flavor.Muon)
    NuTau = NeutrinoDef(16, NeutrinoType.Nu, Flavor.Tau)
    NuEBar = NeutrinoDef(-12, NeutrinoType.NuBar, Flavor.Electron)
    NuMuBar = NeutrinoDef(-14, NeutrinoType.NuBar, Flavor.Muon)
    NuTauBar = NeutrinoDef(-16, NeutrinoType.NuBar, Flavor.Tau)
