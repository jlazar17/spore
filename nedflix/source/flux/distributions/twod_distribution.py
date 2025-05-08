import numpy as np
import h5py as h5

from scipy.interpolate import RegularGridInterpolator

from .distribution import Distribution

class UserProvidedDist2D(Distribution):
    """
    Class for handling 2D distribution provided by the user via tabulated data
    """
    def __init__(self, emin: float, emax: float, decmin: float, decmax: float, spl: RegularGridInterpolator):
        self._decmin = decmin
        self._decmax = decmax
        self._spl = spl
        super().__init__(emin, emax)
    
    @property
    def decmin(self):
        return self._decmin
    
    @property
    def decmax(self):
        return self._decmax

    def density(self, e, dec) -> float:
        if not (self.emin <= e <= self.emax):
            raise ValueError(f"Energy {e} not in range [{self.emin}, {self.emax}]")
        if not (self.decmin <= dec <= self.decmax):
            raise ValueError(f"Declination {dec} not in range [{self.decmin}, {self.decmax}]")
        return np.exp(self._spl((np.sin(dec), np.log(e))))

    @classmethod
    def from_file(cls, filename: str):
        raise ValueError("Not implemented")
