import numpy as np

from typing import Optional
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from ..conventions import units, SkyCoordinate, sky_to_local
from ..physics import neutrinos
from ..source import PointSource
from ..detector import Detector

from .event_sampler import EventSampler

class PointSourceEventSampler(EventSampler):

    def __init__(self, det: Detector, src: PointSource):
        super().__init__(det, src)
        es = np.logspace(2, 6, 100) * units.GeV
        self._es = es

        self._cache = {}

    def sample_events(
        self,
        morphology: str,
        t: Optional[float]=None,
        deltat: Optional[float] = None,
        nevent: Optional[int]=None,
        oversample: int=1
    ):

        if not ((deltat is None) ^ (nevent is None)):
            raise ValueError("Cannot determine number of events to sample.")

        if morphology not in "track cascade".split():
            raise ValueError(f"Invalid morphology {morphology}")

        if t is None:
            t = self._t0

        if (t, morphology) not in self._cache.keys():
            lc = sky_to_local(self._src.location, self._det.location, t)
            effa_fxn = lambda e: self._det.response.effective_area[morphology](lc.zenith, e)
            if morphology=="track":
                f = lambda e: effa_fxn(e) * (self._src(neutrinos[2], e) + self._src(neutrinos[3], e))
            else:
                f = lambda e: effa_fxn(e) * sum([self._src(neutrino, e) for neutrino in neutrinos])

            #hack = 1 / f(1e5 * units.GeV)
            g = lambda le: np.exp(le) * f(np.exp(le))
            #g = lambda le: np.exp(le) * f(np.exp(le)) * hack
            cdfs = np.array([quad(g, np.log(self._es[0]), np.log(e))[0] for e in self._es])

            norm = cdfs[-1]
            
            cdfs = cdfs / cdfs[-1]
            idx = np.argwhere(cdfs==0)[-1][0]
            cdfs = cdfs[idx:]
            spl = CubicSpline(cdfs, np.log(self._es[idx:]))
            self._cache[(t, morphology)] = spl, norm

        spl, norm = self._cache[(t, morphology)]

        if nevent is None:
            nevent = np.random.poisson(norm * deltat)

        us = np.random.rand(nevent)
        return np.exp(spl(us))
