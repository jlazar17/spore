from abc import ABC, abstractmethod

from .. import units

class Distribution(ABC):
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
    def density(self, e: float, dec: float) -> float:
        pass
