import numpy as np
import h5py as h5

from typing import Optional, Dict

from . import Neutrino, neutrinos
from .distributions import Distribution 

class Flux:
    """Per-species neutrino flux model.

    Stores a normalization and a spectral Distribution for each of the six
    standard neutrino species. The flux at energy e for species nu is:
        Phi(nu, e) = normalization[nu] * distribution[nu].density(e)

    Args:
        normalizations: Mapping from Neutrino species to flux normalization
            in natural units (eV^{-1} cm^{-2} s^{-1} sr^{-1}).
        distributions: Mapping from Neutrino species to a Distribution
            that describes the spectral shape.
    """

    def __init__(
            self,
            normalizations: Dict[Neutrino, float],
            distributions: Dict[Neutrino, Distribution],
            normalization: float = 1.0,
        ):

        self._normalizations = normalizations
        self._distributions = distributions
        self._normalization = normalization

    @property
    def uses_ra(self) -> bool:
        """True if the flux distribution requires a right-ascension argument."""
        from .distributions import TabulatedEnergyDecRAFlux
        return any(
            isinstance(d, TabulatedEnergyDecRAFlux)
            for d in self._distributions.values()
        )

    @property
    def e_min_gev(self) -> float:
        """Lower energy bound of the flux distribution in GeV."""
        return min(d.emin for d in self._distributions.values())

    @property
    def e_max_gev(self) -> float:
        """Upper energy bound of the flux distribution in GeV."""
        return max(d.emax for d in self._distributions.values())

    @property
    def normalization(self) -> float:
        """Global multiplicative scaling applied on top of per-species normalizations."""
        return self._normalization

    @normalization.setter
    def normalization(self, value: float) -> None:
        self._normalization = float(value)

    def __call__(self, nu: Neutrino, e: float, dec: Optional[float] = None, ra: Optional[float] = None) -> float:
        """Evaluate the flux for neutrino species nu at energy e.

        Args:
            nu: Neutrino species.
            e: Energy in GeV.
            dec: Declination in radians. Required for 2D+ flux models.
            ra: Right ascension in radians. Required for 3D flux models.

        Returns:
            Differential flux in GeV^{-1} cm^{-2} s^{-1} sr^{-1}.
        """
        distribution = self._distributions[nu]
        norm = self._normalization * self._normalizations[nu]
        if dec is None:
            return norm * distribution.density(e)
        if ra is None:
            return norm * distribution.density(e, dec)
        return norm * distribution.density(e, dec, ra)

    @classmethod
    def from_config(cls, config: Dict, normalization: float = 1.0) -> 'Flux':
        """Build a Flux from a config dictionary.

        Supports two formats:

        Power-law (keys: gamma, emin, emax, pivot, norm_per_species or norm)::

            {"gamma": 2.0, "emin": 1e2, "emax": 1e6, "pivot": 1e5,
             "norm_per_species": 1e-18}

        Tabulated HDF5 (key: location as "filename.h5:groupname")::

            {"location": "fluxes.h5:combined"}

        Args:
            config: Configuration dictionary.

        Returns:
            A configured Flux instance.
        """
        has_powerlaw = all(x in config for x in ["gamma", "emin", "emax"]) and (
            "norm_per_species" in config or "norm" in config
        )
        if has_powerlaw:
            from .distributions import PowerLaw
            pl = PowerLaw.from_config(config)
            if "norm_per_species" in config:
                raw_norm = config["norm_per_species"]
            else:
                raw_norm = config["norm"]
            # raw_norm is in GeV⁻¹ cm⁻² s⁻¹ — no conversion needed
            norm = raw_norm
            if config["gamma"]==1:
                norm *= np.log(config["emax"] / config["emin"])
            else:
                emin = config["emin"]   # GeV
                emax = config["emax"]   # GeV
                p = (1 - config["gamma"])
                norm *= (emax**p - emin**p) / p
            normalizations = {nu: norm for nu in neutrinos}
            distributions = {nu: pl for nu in neutrinos}
            return cls(normalizations, distributions, normalization=normalization)

        if "location" not in config.keys():
            raise ValueError("Unable to parse requested flux")

        filename, groupname = config["location"].split(":")
        with h5.File(filename) as h5f:
            gp = h5f[groupname]
            ndim = gp["fluxes"].ndim
            if gp["fluxes"].shape[0]!=6:
                raise ValueError("Fist dimension must have size 6.")
        if ndim not in [2, 3, 4]:
            raise ValueError(f"dimensionality {ndim} invalid")
        if ndim == 2:
            from .utils import parse_1d_file
            from .distributions import TabulatedEnergyFlux
            norms, spls, emin, emax = parse_1d_file(config["location"])
            normalizations = {nu: n for nu, n in zip(neutrinos, norms)}
            distributions = {
                nu: TabulatedEnergyFlux(emin, emax, spl) for nu, spl in zip(neutrinos, spls)
            }
        elif ndim == 3:
            from .utils import parse_2d_file
            from .distributions import TabulatedEnergyDecFlux
            norms, spls, decmin, decmax, emin, emax = parse_2d_file(config["location"])
            normalizations = {nu: n for nu, n in zip(neutrinos, norms)}
            distributions = {
                nu: TabulatedEnergyDecFlux(emin, emax, decmin, decmax, spl) for nu, spl in zip(neutrinos, spls)
            }
        else:  # ndim == 4
            from .utils import parse_3d_file
            from .distributions import TabulatedEnergyDecRAFlux
            norms, spls, decmin, decmax, ramin, ramax, emin, emax = parse_3d_file(config["location"])
            normalizations = {nu: n for nu, n in zip(neutrinos, norms)}
            distributions = {
                nu: TabulatedEnergyDecRAFlux(emin, emax, decmin, decmax, ramin, ramax, spl)
                for nu, spl in zip(neutrinos, spls)
            }
        return cls(normalizations, distributions, normalization=normalization)
