import numpy as np

from abc import ABC, abstractmethod
from typing import List, Optional

from ..source import Source
from ..detector import Detector
from .event import Event

class EventSampler(ABC):
    """Abstract base class for all SPORE event samplers.

    Subclasses implement sample_events and expected_events for a specific
    source geometry (point source, extended source, galactic halo).

    Args:
        det: The detector to simulate.
        src: The neutrino source to sample from.
    """

    def __init__(self, det: Detector, src: Source):
        self._det = det
        self._src = src
        self._track_xt = None
        self._cascade_xt = None
        self._t0 = 60_355.0

    @abstractmethod
    def sample_events(
        self,
        morphology: str,
        deltat: Optional[float] = None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        oversample: int = 1,
    ) -> List[Event]:
        """Sample a list of events from the source and detector.

        Exactly one of deltat or nevent must be provided. When deltat is given
        the number of events is drawn from a Poisson distribution with mean
        equal to expected_events(morphology, deltat, t).

        Args:
            morphology: Event morphology to sample, either "track" or "cascade".
            deltat: Observation window in natural units (eV^{-1}). Mutually
                exclusive with nevent.
            nevent: Fixed number of events to sample. Mutually exclusive with
                deltat.
            t: Reference epoch in MJD. Defaults to the sampler's internal
                reference epoch (2024-01-01).
            oversample: Draw this many MCMC samples per returned event and keep
                every oversample-th one. Only used by MCMC-based samplers.

        Returns:
            List of sampled Event objects.
        """
        pass

    @abstractmethod
    def expected_events(
        self,
        morphology: str,
        deltat: float,
        t: Optional[float] = None,
    ) -> float:
        """Return the expected (mean) number of events for a given livetime.

        Parameters
        ----------
        morphology : {"track", "cascade"}
        deltat : float
            Observation window in natural units (eV^{-1}).
        t : float or None
            Reference epoch in MJD.  Defaults to the sampler's reference epoch.

        Returns
        -------
        float
            Expected number of events (before Poisson sampling).
        """
        pass
