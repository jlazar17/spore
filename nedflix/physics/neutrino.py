from enum import Enum

class Flavor(Enum):
    Electron: 1
    Muon: 2
    Tau: 3

class NeutrinoType(Enum):
    Nu: 1
    NuBar: 2

@dataclass(frozen=True)
class Neutrino:
    pdg_id: int
    nutype: NeutrinoType
    flavor: Flavor

neutrinos = {
    12: Neutrino(12, NeutrinoType.Nu, Flavor.Electron),
    14: Neutrino(14, NeutrinoType.Nu, Flavor.Muon),
    16: Neutrino(16, NeutrinoType.Nu, Flavor.Tau),
    -12: Neutrino(-12, NeutrinoType.NuBar, Flavor.Electron),
    -12: Neutrino(-14, NeutrinoType.NuBar, Flavor.Muon),
    -12: Neutrino(-16, NeutrinoType.NuBar, Flavor.Tau)
}
