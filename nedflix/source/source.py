from abc import ABC

from . import Neutrino
from .flux import Flux

class Source(ABC):
    
    def __init__(self, flux: Flux):
        self._flux = flux

    @property
    def flux(self):
        return self._flux

    def __call__(self, nu: Neutrino, e: float, dec: float=None):
        return self.flux(nu, e, dec)
