import numpy as np
import h5py as h5

from typing import Optional
from scipy.interpolate import CubicSpline

from .distribution import Distribution

class UserProvidedDist1D(Distribution):
    """
    Class for handling 1D distribution provided by the user
    """
    def __init__(self, emin: float, emax: float, spl: CubicSpline):
        self._spl = spl
        super().__init__(emin, emax)

    def density(self, e: float, dec: Optional[float]=None) -> float:
        if not (self.emin <= e <= self.emax):
            raise ValueError(f"Energy {e} not in range [{self._emin}, {self._emax}]")
        return np.exp(self._spl(np.log(e)))
   
    @classmethod
    def from_file(cls, filename: str):
        raise ValueError("Not implemented") 
