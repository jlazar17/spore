import numpy as np

from typing import Optional
from scipy.interpolate import CubicSpline

from .distribution import Distribution

class TabulatedEnergyFlux(Distribution):
    """Energy-only spectral distribution backed by a user-provided spline.

    Intended for tabulated fluxes loaded from HDF5 via Flux.from_config.
    The spline is evaluated in log-log space: density(e) = exp(spl(log(e))).

    Args:
        emin: Minimum energy in GeV.
        emax: Maximum energy in GeV.
        spl: CubicSpline interpolating log(density) vs log(energy).
    """
    def __init__(self, emin: float, emax: float, spl: CubicSpline):
        self._spl = spl
        super().__init__(emin, emax)

    def density(self, e: float, dec: float = None, ra: float = None) -> float:
        """Evaluate the spectral density at energy e.

        Args:
            e: Energy in GeV. May be a scalar or numpy array.
            dec: Ignored; accepted for interface compatibility.
            ra: Ignored; accepted for interface compatibility.

        Returns:
            Spectral density in GeV^{-1}.

        Raises:
            ValueError: If e is outside [emin, emax].
        """
        e = np.asarray(e, dtype=float)
        scalar = e.ndim == 0
        e = np.atleast_1d(e)
        oob = (e < self.emin) | (e > self.emax)
        if oob.any():
            raise ValueError(f"Energy {e[oob][0]} not in range [{self.emin}, {self.emax}]")
        result = np.exp(self._spl(np.log(e)))
        return float(result[0]) if scalar else result
