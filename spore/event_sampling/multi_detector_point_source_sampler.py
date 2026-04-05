from typing import Dict, List, Optional, Tuple

from ..detector import Detector
from ..source import PointSource
from .event import Event
from .point_source_event_sampler import PointSourceEventSampler


class MultiDetectorPointSourceSampler:
    """
    Samples point source events across multiple detectors simultaneously.

    Each detector independently contributes events according to its own
    effective area, angular resolution, and energy resolution. All detectors
    observe the same source over the same time window, and events are tagged
    with the originating detector via the `detector_id` field on `Event`.

    Parameters
    ----------
    detectors : list of (Detector, name) pairs
        Each entry provides the detector configuration and a unique string
        identifier. The name is written into `Event.detector_id`.
    source : PointSource
        The source is shared across all detectors.

    Example
    -------
    sampler = MultiDetectorPointSourceSampler(
        detectors=[(icecube, "IceCube"), (km3net, "KM3NeT")],
        source=source,
    )
    events = sampler.sample_events("track", t=60355.0, deltat=365 * units.day)
    """

    def __init__(self, detectors: List[Tuple[Detector, str]], source: PointSource):
        names = [name for _, name in detectors]
        if len(names) != len(set(names)):
            raise ValueError("Detector names must be unique.")
        self._samplers: Dict[str, PointSourceEventSampler] = {
            name: PointSourceEventSampler(det, source)
            for det, name in detectors
        }

    @property
    def detector_names(self) -> List[str]:
        return list(self._samplers.keys())

    def sample_events(
        self,
        morphology: str,
        t: Optional[float] = None,
        deltat: Optional[float] = None,
        nevent: Optional[int] = None,
    ) -> List[Event]:
        """
        Sample events from all detectors over a shared observation window.

        Parameters
        ----------
        morphology : "track" or "cascade"
        t : observation time in MJD. Defaults to the sampler's reference epoch
            if not given.
        deltat : observation window in natural units (eV^-1), e.g.
            ``365 * units.day``. Each detector independently Poisson-samples
            its expected event count from its own effective area integral.
        nevent : if given instead of deltat, draw exactly this many events
            from each detector (useful for testing and pseudo-experiments).

        Returns
        -------
        List[Event]
            Combined events from all detectors, each with `detector_id` set
            to the corresponding detector name. Events are ordered by detector,
            not by time.
        """
        if not ((deltat is None) ^ (nevent is None)):
            raise ValueError("Specify exactly one of deltat or nevent.")

        all_events: List[Event] = []
        for name, sampler in self._samplers.items():
            events = sampler.sample_events(
                morphology, t=t, deltat=deltat, nevent=nevent
            )
            for event in events:
                event.detector_id = name
            all_events.extend(events)
        return all_events
