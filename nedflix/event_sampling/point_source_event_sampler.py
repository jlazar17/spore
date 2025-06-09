import numpy as np

from typing import Optional
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..conventions import units, SkyCoordinate, sky_to_local
from ..physics import neutrinos
from ..source import PointSource
from ..detector import Detector

from .event import Event
from .event_sampler import EventSampler
from .utils import smear_truth

class PointSourceEventSampler(EventSampler):

    def __init__(self, det: Detector, src: PointSource):
        super().__init__(det, src)
        es = np.logspace(2, 6, 50) * units.GeV
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

            effas = np.array([effa_fxn(e) for e in self._es])
            idx = np.where(effas==0)[0][-1]
            es = self._es[idx+1:]

            #hack = 1 / f(es[-1])
            hack = 1.0
            g = lambda le: np.exp(le) * f(np.exp(le)) * hack
            cdfs = np.concat([[0], [quad(g, np.log(self._es[0]), np.log(e))[0] for e in es[1:]]])

            # This is some horrible stuff because of numerical issues
            idxs = np.where(np.diff(cdfs) > 0)[0]
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
            #spl = CubicSpline(cdfs, np.log(es))

            self._cache[(t, morphology)] = spl, norm

        spl, norm = self._cache[(t, morphology)]

        if nevent is None:
            nevent = np.random.poisson(norm * deltat)

        us = np.random.rand(nevent)
        true_energies = np.exp(spl(us))
        true_direction = self._src.location
        events = []
        for true_energy in true_energies:
            reco_direction, reco_energy = smear_truth(
                true_direction,
                true_energy,
                self._det,
                morphology
            )
            signalness = np.random.rand()
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
                signalness,
                t_event,
                morphology_id
            )
            events.append(event)
        return events
