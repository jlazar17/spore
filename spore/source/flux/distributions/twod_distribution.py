import numpy as np
import h5py as h5

from scipy.interpolate import RegularGridInterpolator

from .distribution import Distribution

class TabulatedEnergyDecFlux(Distribution):
    """Energy and declination spectral distribution backed by a user-provided interpolator.

    Intended for tabulated fluxes that vary with both energy and sky position,
    loaded from HDF5 via Flux.from_config. The interpolator operates in
    (sin(dec), log(energy)) space with log-density output.

    Args:
        emin: Minimum energy in GeV.
        emax: Maximum energy in GeV.
        decmin: Minimum declination in radians.
        decmax: Maximum declination in radians.
        spl: RegularGridInterpolator over (sin(dec), log(energy)) returning
            log(density).
    """
    def __init__(self, emin: float, emax: float, decmin: float, decmax: float, spl: RegularGridInterpolator):
        self._decmin = decmin
        self._decmax = decmax
        self._spl = spl
        super().__init__(emin, emax)
    
    @property
    def decmin(self) -> float:
        """Minimum declination of the distribution in radians."""
        return self._decmin

    @property
    def decmax(self) -> float:
        """Maximum declination of the distribution in radians."""
        return self._decmax

    def density(self, e: float, dec: float = None, ra: float = None) -> float:
        """Evaluate the spectral density at energy e and declination dec.

        Args:
            e: Energy in eV. May be a scalar or numpy array.
            dec: Declination in radians. Must match the shape of e.

        Returns:
            Spectral density in eV^{-1} sr^{-1}.

        Raises:
            ValueError: If e is outside [emin, emax] or dec outside
                [decmin, decmax].
        """
        e   = np.asarray(e,   dtype=float)
        scalar = e.ndim == 0 and np.ndim(dec) == 0
        e   = np.atleast_1d(e)
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        oob_e   = (e   < self.emin)   | (e   > self.emax)
        oob_dec = (dec < self.decmin) | (dec > self.decmax)
        if oob_e.any():
            raise ValueError(f"Energy {e[oob_e][0]} not in range [{self.emin}, {self.emax}]")
        if oob_dec.any():
            raise ValueError(f"Declination {dec[oob_dec][0]} not in range [{self.decmin}, {self.decmax}]")
        pts = np.column_stack([np.sin(dec) * np.ones(len(e)), np.log(e)])
        result = np.exp(self._spl(pts))
        return float(result[0]) if scalar else result

    @classmethod
    def from_file(cls, filename: str) -> 'TabulatedEnergyDecFlux':
        """Not implemented. Use Flux.from_config with an HDF5 location string."""
        raise ValueError("Not implemented")
