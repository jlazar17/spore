import numpy as np

from typing import Optional

from ..conventions import units, SkyCoordinate, sky_to_local
from ..physics import neutrinos
from ..source import PointSource
from ..detector import Detector

from .event import Event
from .event_sampler import EventSampler
from .utils import smear_truth

class PointSourceEventSampler(EventSampler):
    """Event sampler for a neutrino point source.

    Builds an inverse-CDF energy sampler from the source flux convolved with
    the detector effective area at the source zenith angle. Results are cached
    per (time, morphology) so repeated pseudo-experiments are fast.

    Args:
        det: The detector to simulate.
        src: The point source to sample from.
    """

    def __init__(self, det: Detector, src: PointSource):
        super().__init__(det, src)
        es = np.logspace(2, 6, 50) * units.GeV
        self._es = es

        self._cache = {}

    def _ensure_cached(self, morphology: str, t: float):
        """Populate the energy CDF cache for (morphology, t) if not already done."""
        if (t, morphology) not in self._cache:
            lc = sky_to_local(self._src.location, self._det.location, t)
            effa_fxn = lambda e: self._det.response.effective_area[morphology](lc.zenith, e)
            if morphology == "track":
                f = lambda e: effa_fxn(e) * (self._src(neutrinos[2], e) + self._src(neutrinos[3], e))
            else:
                f = lambda e: effa_fxn(e) * sum([self._src(neutrino, e) for neutrino in neutrinos])

            effas = np.array([effa_fxn(e) for e in self._es])
            zeros = np.where(effas == 0)[0]
            idx = zeros[-1] if len(zeros) > 0 else -1
            es = self._es[idx + 1:]

            if len(es) == 0:
                self._cache[(t, morphology)] = None, 0.0
            else:
                from scipy.integrate import quad
                from scipy.interpolate import PchipInterpolator
                hack = 1.0
                g = lambda le: np.exp(le) * f(np.exp(le)) * hack
                cdfs = np.concatenate([[0], [quad(g, np.log(es[0]), np.log(e))[0] for e in es[1:]]])

                cdfs_, es_, last = [], [], None
                for v, e in zip(cdfs, es):
                    if last is not None:
                        if v <= last:
                            continue
                    cdfs_.append(v)
                    es_.append(e)
                    last = v

                cdfs = np.array(cdfs_)
                es = np.array(es_)

                norm = cdfs[-1] / hack
                cdfs = cdfs / cdfs[-1]
                spl = PchipInterpolator(cdfs, np.log(es))
                self._cache[(t, morphology)] = spl, norm

    def expected_events(
        self,
        morphology: str,
        deltat: float,
        t: Optional[float] = None,
    ) -> float:
        """Return the expected number of events for a given livetime.

        Args:
            morphology: Event morphology, either "track" or "cascade".
            deltat: Observation window in natural units (eV^{-1}).
            t: Reference epoch in MJD. Defaults to the sampler's internal epoch.

        Returns:
            Expected number of events (mean of the Poisson distribution).
        """
        if t is None:
            t = self._t0
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )
        self._ensure_cached(morphology, t)
        _, norm = self._cache[(t, morphology)]
        return norm * deltat

    def sample_events(
        self,
        morphology: str,
        deltat: Optional[float] = None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        oversample: int = 1,
    ):
        """Sample events from the point source.

        Args:
            morphology: Event morphology to sample, either "track" or "cascade".
            deltat: Observation window in natural units (eV^{-1}). The number
                of events is drawn from Poisson(expected_events). Mutually
                exclusive with nevent.
            nevent: Fixed number of events to draw. Mutually exclusive with
                deltat.
            t: Reference epoch in MJD. Defaults to the sampler's internal epoch.
            oversample: Unused for point sources; accepted for API consistency.

        Returns:
            List of Event objects with true and reconstructed energies,
            directions, and angular uncertainties.
        """

        if not ((deltat is None) ^ (nevent is None)):
            raise ValueError("Cannot determine number of events to sample.")

        if morphology not in ("track", "cascade"):
            raise ValueError(f"Invalid morphology '{morphology}': must be 'track' or 'cascade'.")
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )

        if t is None:
            t = self._t0

        self._ensure_cached(morphology, t)
        spl, norm = self._cache[(t, morphology)]

        if nevent is None:
            nevent = np.random.poisson(norm * deltat)

        if nevent == 0 or spl is None:
            return []

        us = np.random.rand(nevent)
        true_energies = np.exp(spl(us))
        true_direction = self._src.location
        events = []
        for true_energy in true_energies:
            reco_direction, reco_energy, ang_err = smear_truth(
                true_direction,
                true_energy,
                self._det,
                morphology
            )
            if deltat is None:
                t_event = t
            else:
                t_event = t + np.random.uniform(low=-deltat/2, high=deltat/2) / (24 * 3600 * units.sec)
            morphology_id = 1
            if morphology=="track":
                morphology_id = 2
            event = Event(
                true_direction,
                reco_direction,
                true_energy,
                reco_energy,
                t_event,
                morphology_id,
                ang_err=ang_err,
            )
            events.append(event)
        return events
