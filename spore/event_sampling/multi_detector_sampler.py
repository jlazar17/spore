from typing import List, Optional

from ..detector import Detector
from .event import Event
from .event_sampler import EventSampler


def _sampler_cls_for(source) -> type:
    """Return the appropriate EventSampler subclass for the given source."""
    # Lazy imports to avoid circular dependencies.
    from .point_source_event_sampler import PointSourceEventSampler
    from .extended_source_event_sampler import ExtendedSourceEventSampler
    from ..source.point_source import PointSource
    from ..source.extended_source import ExtendedSource

    if isinstance(source, PointSource):
        return PointSourceEventSampler
    if isinstance(source, ExtendedSource):
        return ExtendedSourceEventSampler
    raise TypeError(
        f"No sampler registered for source type '{type(source).__name__}'. "
        "Expected PointSource or ExtendedSource (including GalacticHalo subclasses)."
    )


class MultiDetectorSampler:
    """
    Sample events from any source type across multiple detectors.

    The appropriate per-detector sampler class is selected automatically
    based on the source type:

    * ``PointSource``    → ``PointSourceEventSampler``
    * ``ExtendedSource`` → ``ExtendedSourceEventSampler``

    Detectors are identified by their index in the input list.
    Each sampled event carries an integer ``detector_id`` equal to that index.

    Parameters
    ----------
    detectors : list of Detector
        Detectors to include in the joint simulation.
    source :
        Source object shared by all detectors.
    **kwargs
        Extra keyword arguments forwarded to each per-detector sampler
        constructor (e.g. ``deltat``, ``n_dec``, ``n_e`` for the
        extended-source and halo samplers).

    Example
    -------
    sampler = MultiDetectorSampler(
        [icecube, km3net],
        source,
        deltat=10 * ureg.year,
    )
    events = sampler.sample_events("track", deltat=10 * ureg.year)
    """

    def __init__(self, detectors: List[Detector], source, **kwargs):
        cls = _sampler_cls_for(source)
        self._samplers: List[EventSampler] = [
            cls(det, source, **kwargs) for det in detectors
        ]

    def __len__(self) -> int:
        return len(self._samplers)

    def _resolve_deltat(self, deltat):
        """Return a per-detector list of deltat values.

        A single Quantity is broadcast to all detectors.
        A list must have the same length as the number of detectors.
        """
        if deltat is None:
            return [None] * len(self._samplers)
        if isinstance(deltat, list):
            if len(deltat) != len(self._samplers):
                raise ValueError(
                    f"deltat list length ({len(deltat)}) must match the number "
                    f"of detectors ({len(self._samplers)})."
                )
            return deltat
        return [deltat] * len(self._samplers)

    def expected_events(
        self,
        morphology: str,
        deltat,
        t: Optional[float] = None,
    ) -> List[float]:
        """Return the expected event count per detector.

        Parameters
        ----------
        morphology : str
        deltat : pint Quantity or list of pint Quantity
            A single Quantity is applied to every detector.
            A list assigns a different livetime to each detector (in order).
        t : float or None
            Reference epoch in MJD.

        Returns
        -------
        list of float
            One entry per detector, in the same order as the input list.
        """
        deltats = self._resolve_deltat(deltat)
        return [
            s.expected_events(morphology, dt, t=t)
            for s, dt in zip(self._samplers, deltats)
        ]

    def sample_events(
        self,
        morphology: str,
        deltat=None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
    ) -> List[Event]:
        """
        Sample events from all detectors.

        Parameters
        ----------
        morphology : str
            e.g. ``"track"`` or ``"cascade"``.
        deltat : pint Quantity, list of pint Quantity, or None
            Observation livetime.  A single Quantity is applied to every
            detector; a list assigns a different livetime to each detector
            (must have the same length as the detector list).
            Mutually exclusive with ``nevent``.
        nevent : int or None
            Draw exactly this many events from each detector.
            Mutually exclusive with ``deltat``.
        t : float or None
            Reference epoch in MJD.

        Returns
        -------
        list of Event
            Combined events from all detectors.
            Each event's ``detector_id`` is the integer index of its detector.
            Events are ordered by detector, not by time.
        """
        if not ((deltat is None) ^ (nevent is None)):
            raise ValueError("Specify exactly one of deltat or nevent.")

        deltats = self._resolve_deltat(deltat)
        all_events: List[Event] = []
        for idx, (sampler, dt) in enumerate(zip(self._samplers, deltats)):
            events = sampler.sample_events(
                morphology, t=t, deltat=dt, nevent=nevent
            )
            for event in events:
                event.detector_id = idx
            all_events.extend(events)
        return all_events
