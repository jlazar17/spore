import numpy as np

from abc import ABC, abstractmethod
from typing import List, Optional

from ..source import Source
from ..detector import Detector
from .event import Event

class EventSampler(ABC):

    def __init__(self, det: Detector, src: Source):
        self._det = det
        self._src = src
        self._track_xt = None
        self._cascade_xt = None
        self._t0 = 60_355.0

    @abstractmethod
    def sample_events(
        self,
        morphology,
        t: Optional[float]=None,
        nevent: Optional[int]=None,
        deltat: Optional[float]=None
    ) -> List[Event]:
        pass
