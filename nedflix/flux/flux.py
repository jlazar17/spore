import numpy as np

from dataclasses import dataclass
from typing import Dict

from .. import units
from .energy_distributions import EnergyDistribution

@dataclass
class Flux:
    pivot: float
    normalization: float
    energy_distribution: EnergyDistribution

    def __post_init__(self):
        emin = self.energy_distribution.emin
        emax = self.energy_distribution.emax
        if self.pivot < emin or emax < self.pivot:
            raise ValueError("Pivot not with bounds of energy distribution")

    def __call__(self, e: float):
        emin = self.energy_distribution.emin
        emax = self.energy_distribution.emin
        if e < emin or e > emax:
            raise ValueError(f"Energy {e} not in range [{emin}, {emax}]")
        return self.normalization * self.energy_distribution.pdf(e) / self.energy_distribution.pdf(pivot)

    def sample_energy(self):
        return self.energy_distribution.sample_energy()

def flux_from_config(config: Dict) -> Flux:
    from .energy_distributions import energy_distribution_from_config
    pivot = config["pivot"] * units.GeV
    normalization = config["normalization"] / (units.GeV * units.cm**2 * units.sec)
    energy_distribution = energy_distribution_from_config(config["energy"])
    return Flux(pivot, normalization, energy_distribution)
