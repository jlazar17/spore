from abc import ABC, abstractmethod
from typing import Dict

from .. import units

class EnergyDistribution(ABC):
    def __init__(self, emin: float, emax: float):
        self._emin = emin
        self._emax = emax
    
    @property
    def emin(self) -> float:
        return self._emin

    @property
    def emax(self) -> float:
        return self._emax

    @abstractmethod
    def pdf(self, e) -> float:
        pass

    @abstractmethod
    def cdf(self, e) -> float:
        pass

    @abstractmethod
    def sample_energy(self) -> float:
        pass

def energy_distribution_from_config(config: Dict) -> EnergyDistribution:
    # Make CDF from provided tabulated file
    if "file" in config.keys() and len(config["file"]) > 0:
        raise NotImplementedError("Making energy from file not implemented yet")
    else:
        from .power_laws import PowerLaw
        gamma = config["gamma"]
        emin = config["emin"] * units.GeV
        emax = config["emax"] * units.GeV
        distribution = PowerLaw(gamma, emin, emax)
    return distribution
