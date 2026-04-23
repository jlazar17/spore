from typing import Union

import pint

from ..conventions import SkyCoordinate, ureg


class Event:
    """A single sampled neutrino event with true and reconstructed quantities.

    Energies are stored internally as plain floats (GeV) and exposed as pint
    Quantities via properties.  This avoids the overhead of pint allocation in
    the constructor, which matters when sampling millions of events.

    Attributes:
        true_direction: True neutrino arrival direction.
        reco_direction: Reconstructed arrival direction after PSF smearing.
        true_energy: True neutrino energy (pint Quantity, GeV).
        reco_energy: Reconstructed deposited energy (pint Quantity, GeV).
        time: Event time in modified Julian days.
        morphology: Event morphology name (e.g. "track", "cascade").
        detector_id: Identifier of the detector that recorded the event.
            Empty string for single-detector samplers.
        ang_err: Per-event angular uncertainty (pint Quantity, radians).
    """

    __slots__ = (
        'true_direction', 'reco_direction',
        '_true_e', '_reco_e', '_ang_err',
        '_zenith_rad', '_azimuth_rad',
        'time', 'morphology', 'detector_id',
    )

    def __init__(
        self,
        true_direction: SkyCoordinate,
        reco_direction: SkyCoordinate,
        true_energy: Union[float, pint.Quantity],
        reco_energy: Union[float, pint.Quantity],
        time: float,
        morphology: str,
        detector_id: Union[str, int] = "",
        ang_err: Union[float, pint.Quantity] = 0.0,
        zenith: Union[float, pint.Quantity, None] = None,
        azimuth: Union[float, pint.Quantity, None] = None,
    ):
        self.true_direction = true_direction
        self.reco_direction = reco_direction
        self._true_e = true_energy.magnitude if isinstance(true_energy, pint.Quantity) else float(true_energy)
        self._reco_e = reco_energy.magnitude if isinstance(reco_energy, pint.Quantity) else float(reco_energy)
        self._ang_err = ang_err.magnitude if isinstance(ang_err, pint.Quantity) else float(ang_err)
        self._zenith_rad  = float(zenith.magnitude  if isinstance(zenith,  pint.Quantity) else zenith)  if zenith  is not None else float('nan')
        self._azimuth_rad = float(azimuth.magnitude if isinstance(azimuth, pint.Quantity) else azimuth) if azimuth is not None else float('nan')
        self.time = float(time)
        self.morphology = morphology
        self.detector_id = detector_id

    @property
    def true_energy(self) -> pint.Quantity:
        return ureg.Quantity(self._true_e, "GeV")

    @property
    def reco_energy(self) -> pint.Quantity:
        return ureg.Quantity(self._reco_e, "GeV")

    @property
    def ang_err(self) -> pint.Quantity:
        return ureg.Quantity(self._ang_err, "rad")

    @property
    def zenith(self) -> pint.Quantity:
        return ureg.Quantity(self._zenith_rad, "rad")

    @property
    def azimuth(self) -> pint.Quantity:
        return ureg.Quantity(self._azimuth_rad, "rad")

    def to_dict(self) -> dict:
        """Serialize the event to a flat dictionary of Python scalars.

        Returns:
            Dictionary with keys: true_dec, true_ra, reco_dec, reco_ra,
            true_energy, reco_energy, time, morphology, detector_id, ang_err,
            zenith, azimuth.  Energies in GeV; angles in radians.
            zenith and azimuth are NaN when not computed.
        """
        return {
            "true_dec":    self.true_direction.declination,
            "true_ra":     self.true_direction.right_ascension,
            "reco_dec":    self.reco_direction.declination,
            "reco_ra":     self.reco_direction.right_ascension,
            "true_energy": self._true_e,
            "reco_energy": self._reco_e,
            "time":        self.time,
            "morphology":  self.morphology,
            "detector_id": self.detector_id,
            "ang_err":     self._ang_err,
            "zenith":      self._zenith_rad,
            "azimuth":     self._azimuth_rad,
        }
