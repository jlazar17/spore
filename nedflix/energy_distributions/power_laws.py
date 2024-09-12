import numpy as np

from .energy_distribution import EnergyDistribution

class PowerLaw(EnergyDistribution):

    def __init__(self, gamma: float, emin: float, emax: float):
        self._gamma = gamma
        super(PowerLaw, self).__init__(emin, emax)

    @property
    def gamma(self) -> float:
        return self._gamma

    def pdf(self, e: float) -> float:
        raise NotImplementedError("PDF not implemented")

    def cdf(self, e: float) -> float:
        raise NotImplementedError("CDF not implemented")

    def sample_energy(self) -> float:
        u = np.random.rand()
        if self.gamma == 1:
            b = self.emax ** u
            a = self.emin ** (u - 1)
            return b / a
        mg = 1 - self.gamma
        val = (u * self.emax ** mg + (1 - u) * self.emin**mg) ** (1 / mg)
        return val

