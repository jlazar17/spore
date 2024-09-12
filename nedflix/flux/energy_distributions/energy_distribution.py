from abc import ABC, abstractmethod

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
