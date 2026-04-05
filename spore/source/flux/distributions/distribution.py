from abc import ABC, abstractmethod

from .. import units

class Distribution(ABC):
    """Abstract base class for neutrino energy (and optionally declination) distributions.

    All energies are in eV. Subclasses must implement density().

    Args:
        emin: Minimum energy in eV.
        emax: Maximum energy in eV.
    """
    def __init__(self, emin: float, emax: float):
        self._emin = emin
        self._emax = emax

    @property
    def emin(self) -> float:
        """Minimum energy of the distribution in eV."""
        return self._emin

    @property
    def emax(self) -> float:
        """Maximum energy of the distribution in eV."""
        return self._emax

    @abstractmethod
    def density(self, e: float, dec: float = None) -> float:
        """Evaluate the normalised spectral shape at energy e.

        Args:
            e: Energy in eV.
            dec: Declination in radians. Only used by 2D distributions.

        Returns:
            Spectral density (eV^{-1}), normalised so that the integral over
            [emin, emax] is unity.
        """
        pass
