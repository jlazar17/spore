import numpy as np

from typing import Dict, Optional

from .distribution import Distribution
from .. import units

class PowerLaw(Distribution):
    """Normalised power-law spectral distribution E^{-gamma}.

    The distribution is normalised so that the integral over [emin, emax]
    is unity, making it suitable as the distribution argument in Flux.

    Args:
        gamma: Spectral index. Must be positive.
        emin: Minimum energy in eV.
        emax: Maximum energy in eV.
        pivot: Pivot energy in eV.
    """

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
        """Spectral index of the power law."""
        return self._gamma

    def density(self, e: float, dec: float = None) -> float:
        """Evaluate the normalised power-law density at energy e.

        Args:
            e: Energy in eV. May be a scalar or numpy array.
            dec: Ignored; accepted for interface compatibility.

        Returns:
            Normalised spectral density in eV^{-1}.

        Raises:
            ValueError: If e is outside [emin, emax].
        """
        e = np.asarray(e, dtype=float)
        scalar = e.ndim == 0
        e = np.atleast_1d(e)
        oob = (e < self.emin) | (e > self.emax)
        if oob.any():
            raise ValueError(f"Energy {e[oob][0]} not in range [{self.emin}, {self.emax}]")
        result = self._norm * (e / self._pivot) ** -self.gamma
        return float(result[0]) if scalar else result

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
    def from_config(cls, config: Dict) -> 'PowerLaw':
        """Build a PowerLaw from a config dictionary.

        Args:
            config: Dictionary with keys gamma, emin, emax, pivot (all in GeV).

        Returns:
            A configured PowerLaw instance.
        """
        gamma = config["gamma"]
        emin = config["emin"] * units.GeV
        emax = config["emax"] * units.GeV
        pivot = config["pivot"] * units.GeV
        distribution = cls(gamma, emin, emax, pivot)
        return distribution
