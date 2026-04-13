import numpy as np

from scipy.interpolate import RegularGridInterpolator

from .distribution import Distribution


class TabulatedEnergyDecRAFlux(Distribution):
    """Energy, declination, and right-ascension spectral distribution.

    Intended for tabulated fluxes that vary with energy and full sky position,
    loaded from HDF5 via Flux.from_config. The interpolator operates in
    (sin(dec), RA, log(energy)) space with log-density output.

    Args:
        emin: Minimum energy in GeV.
        emax: Maximum energy in GeV.
        decmin: Minimum declination in radians.
        decmax: Maximum declination in radians.
        ramin: Minimum right ascension in radians.
        ramax: Maximum right ascension in radians.
        spl: RegularGridInterpolator over (sin(dec), RA, log(energy)) returning
            log(density).
    """
    def __init__(
        self,
        emin: float,
        emax: float,
        decmin: float,
        decmax: float,
        ramin: float,
        ramax: float,
        spl: RegularGridInterpolator,
    ):
        self._decmin = decmin
        self._decmax = decmax
        self._ramin = ramin
        self._ramax = ramax
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

    @property
    def ramin(self) -> float:
        """Minimum right ascension of the distribution in radians."""
        return self._ramin

    @property
    def ramax(self) -> float:
        """Maximum right ascension of the distribution in radians."""
        return self._ramax

    def density(self, e: float, dec: float = None, ra: float = None) -> float:
        """Evaluate the spectral density at energy e, declination dec, and RA ra.

        Args:
            e: Energy in GeV. May be a scalar or numpy array.
            dec: Declination in radians. Must match the shape of e.
            ra: Right ascension in radians. Must match the shape of e.

        Returns:
            Spectral density in GeV^{-1} sr^{-1}.

        Raises:
            ValueError: If dec or ra are None, or if any argument is outside
                its valid range.
        """
        if dec is None:
            raise ValueError("dec is required for TabulatedEnergyDecRAFlux")
        if ra is None:
            raise ValueError("ra is required for TabulatedEnergyDecRAFlux")

        e   = np.asarray(e,   dtype=float)
        scalar = e.ndim == 0 and np.ndim(dec) == 0 and np.ndim(ra) == 0
        e   = np.atleast_1d(e)
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        ra  = np.atleast_1d(np.asarray(ra,  dtype=float))

        oob_e   = (e   < self.emin)   | (e   > self.emax)
        oob_dec = (dec < self.decmin) | (dec > self.decmax)
        oob_ra  = (ra  < self.ramin)  | (ra  > self.ramax)
        if oob_e.any():
            raise ValueError(f"Energy {e[oob_e][0]} not in range [{self.emin}, {self.emax}]")
        if oob_dec.any():
            raise ValueError(f"Declination {dec[oob_dec][0]} not in range [{self.decmin}, {self.decmax}]")
        if oob_ra.any():
            raise ValueError(f"RA {ra[oob_ra][0]} not in range [{self.ramin}, {self.ramax}]")

        pts = np.column_stack([
            np.sin(dec) * np.ones(len(e)),
            ra * np.ones(len(e)),
            np.log(e),
        ])
        result = np.exp(self._spl(pts))
        return float(result[0]) if scalar else result

    @classmethod
    def from_file(cls, filename: str) -> 'TabulatedEnergyDecRAFlux':
        """Not implemented. Use Flux.from_config with an HDF5 location string."""
        raise ValueError("Not implemented")
