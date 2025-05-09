import numpy as np

from typing import Dict, Optional

from .distribution import Distribution
from .. import units

class PowerLaw(Distribution):

    def __init__(self, gamma: float, emin: float, emax: float, pivot: float):
        emin, emax, pivot = float(emin), float(emax), float(pivot)
        self._gamma, self._pivot = gamma, pivot
        if gamma==1:
            norm = 1 / np.log(emax / emin)
        else:
            mg = 1 - gamma
            norm = mg / (np.power(emax, mg) - np.power(emin, mg))
        self._norm = norm
        super().__init__(emin, emax)

    @property
    def gamma(self) -> float:
        return self._gamma

    def density(self, e: float, dec: Optional[float]=None) -> float:
        if not (self.emin <= e <= self.emax):
            raise ValueError(f"Energy {e} not in range [{self._emin}, {self._emax}]")
        return self._norm * (e / self._pivot)**-self.gamma

    #def sample_energy(self) -> float:
    #    u = np.random.rand()
    #    if self.gamma == 1:
    #        b = self.emax ** u
    #        a = self.emin ** (u - 1)
    #        return b / a
    #    mg = 1 - self.gamma
    #    val = (u * self.emax ** mg + (1 - u) * self.emin**mg) ** (1 / mg)
    #    return val

    @classmethod
    def from_config(cls, config: Dict):
        gamma = config["gamma"]
        emin = config["emin"] * units.GeV
        emax = config["emax"] * units.GeV
        pivot = config["pivot"] * units.GeV
        distribution = cls(gamma, emin, emax, pivot)
        return distribution
