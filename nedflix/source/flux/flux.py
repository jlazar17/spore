import numpy as np
import h5py as h5

from typing import Optional, Dict

from . import Neutrino, neutrinos, units
from .distributions import Distribution 

class Flux:
    
    def __init__(
            self,
            normalizations: Dict[Neutrino, float],
            distributions: Dict[Neutrino, Distribution]
        ):
        """
        normalization: flux normalization calculated at point where reference density calulated
        distribution: `Distribution` describing the energy (and potentially dec) functional form
        """

        self._normalizations = normalizations
        self._distributions = distributions

    def __call__(self, nu: Neutrino, e: float, dec: Optional[float]=None):
        distribution, normalization = self._distributions[nu], self._normalizations[nu]
        if dec is None:
            return normalization * distribution.density(e)
        return normalization * distribution.density(e, dec)

    @classmethod
    def from_config(cls, config: Dict):
        if all([x in config.keys() for x in "gamma emin emax norm".split()]):
            from .distributions import PowerLaw
            pl = PowerLaw.from_config(config)
            norm = config["norm"] / units.GeV / units.cm**2 / units.sec
            if config["gamma"]==1:
                norm *= np.log(config["emax"] / config["emin"])
            else:
                emin = config["emin"] * units.GeV
                emax = config["emax"] * units.GeV
                p = (1 - config["gamma"])
                norm *= (emax**p - emin**p) / p
            normalizations = {nu: norm for nu in neutrinos}
            distributions = {nu: pl for nu in neutrinos}
            return cls(normalizations, distributions)

        if "location" not in config.keys():
            raise ValueError("Unable to parse requested flux")

        filename, groupname = config["location"].split(":")
        with h5.File(filename) as h5f:
            gp = h5f[groupname]
            ndim = gp["fluxes"].ndim
            if gp["fluxes"].shape[0]!=6:
                raise ValueError("Fist dimension must have size 6.")
        if ndim not in [2, 3]:
            raise ValueError(f"dimensionality {ndim} invalid")
        if ndim==2:
            from .utils import parse_1d_file
            from .distributions import UserProvidedDist1D
            norms, spls, emin, emax = parse_1d_file(config["location"])
            normalizations = {nu: n for nu, n in zip(neutrinos, norms)}
            distributions = {
                nu: UserProvidedDist1D(emin, emax, spl) for nu, spl in zip(neutrinos, spls)
            }
        else:
            from .utils import parse_2d_file
            from .distributions import UserProvidedDist2D
            norms, spls, decmin, decmax, emin, emax = parse_2d_file(config["location"])
            normalizations = {nu: n for nu, n in zip(neutrinos, norms)}
            distributions = {
                nu: UserProvidedDist2D(emin, emax, decmin, decmax, spl) for nu, spl in zip(neutrinos, spls)
            }
        return cls(normalizations, distributions)
