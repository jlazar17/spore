from abc import ABC, abstractmethod


class Distribution(ABC):
    """Abstract base class for neutrino energy (and optionally declination) distributions.

    All energies are in GeV. Subclasses must implement density().

    Args:
        emin: Minimum energy in GeV.
        emax: Maximum energy in GeV.
    """
    def __init__(self, emin: float, emax: float):
        self._emin = emin
        self._emax = emax

    @property
    def emin(self) -> float:
        """Minimum energy of the distribution in GeV."""
        return self._emin

    @property
    def emax(self) -> float:
        """Maximum energy of the distribution in GeV."""
        return self._emax

    @abstractmethod
    def density(self, e: float, dec: float = None, ra: float = None) -> float:
        """Evaluate the spectral shape at energy e.

        Args:
            e: Energy in GeV.
            dec: Declination in radians. Only used by 2D+ distributions.
            ra: Right ascension in radians. Only used by 3D distributions.

        Returns:
            Spectral density (GeV^{-1}), normalised so that the integral over
            [emin, emax] is unity for 1D distributions; includes spatial
            dependence for 2D/3D distributions.
        """
        pass
