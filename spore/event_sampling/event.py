from dataclasses import dataclass, field
from typing import Union

import pint

from ..conventions import SkyCoordinate, ureg


@dataclass
class Event:
    """A single sampled neutrino event with true and reconstructed quantities.

    Energies are stored as pint Quantities in eV; bare floats are converted
    automatically. Angular error is stored in radians.

    Attributes:
        true_direction: True neutrino arrival direction.
        reco_direction: Reconstructed arrival direction after PSF smearing.
        true_energy: True neutrino energy (eV).
        reco_energy: Reconstructed deposited energy (eV).
        time: Event time in modified Julian days.
        morphology: Event morphology code (1 = cascade, 2 = track).
        detector_id: Identifier of the detector that recorded the event.
            Empty string for single-detector samplers.
        ang_err: Per-event angular uncertainty (radians).
    """
    true_direction: SkyCoordinate
    reco_direction: SkyCoordinate
    true_energy: Union[float, pint.Quantity]
    reco_energy: Union[float, pint.Quantity]
    time: float
    morphology: int
    detector_id: str = ""
    ang_err: Union[float, pint.Quantity] = 0.0

    def __post_init__(self):
        if not isinstance(self.true_energy, pint.Quantity):
            self.true_energy = ureg.Quantity(self.true_energy, "eV")
        if not isinstance(self.reco_energy, pint.Quantity):
            self.reco_energy = ureg.Quantity(self.reco_energy, "eV")
        if not isinstance(self.ang_err, pint.Quantity):
            self.ang_err = ureg.Quantity(self.ang_err, "rad")

    def to_dict(self) -> dict:
        """Serialize the event to a flat dictionary of Python scalars.

        Returns:
            Dictionary with keys: true_dec, true_ra, reco_dec, reco_ra,
            true_energy, reco_energy, time, morphology, detector_id, ang_err.
            Energy and angular error values are stripped of units.
        """
        return {
            "true_dec":    self.true_direction.declination,
            "true_ra":     self.true_direction.right_ascension,
            "reco_dec":    self.reco_direction.declination,
            "reco_ra":     self.reco_direction.right_ascension,
            "true_energy": self.true_energy.magnitude,
            "reco_energy": self.reco_energy.magnitude,
            "time":        self.time,
            "morphology":  self.morphology,
            "detector_id": self.detector_id,
            "ang_err":     self.ang_err.magnitude,
        }
