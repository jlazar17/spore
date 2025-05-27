import numpy as np

from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import quad
from typing import Optional

from ..conventions import units, SkyCoordinate, sky_to_local
from ..physics import neutrinos
from ..source import ExtendedSource
from ..detector import Detector

from .event_sampler import EventSampler
from .event import Event
from .metropolis_hastings import metropolis_hastings
from .utils import new_sample, smear_truth

class ExtendedSourceEventSampler(EventSampler):

    def __init__(self, det: Detector, src: ExtendedSource, burnin: int=10_000):

        self._burnin = burnin
        super().__init__(det, src)

        ras = np.linspace(0, 2*np.pi, 21)
        decs = np.asin(np.linspace(-1, 1, 20))
        es = np.logspace(2, 6, 22) * units.GeV

        self._bounds = [
            (np.sin(decs).min(), np.sin(decs).max()),
            (ras.min(), ras.max()),
            (np.log(es).min(), np.log(es).max()),
        ]
    
        self._ras = ras
        self._decs = decs
        self._es = es
        self._burnin = burnin
        
        fluxes = np.zeros((6,) + decs.shape + ras.shape + es.shape)
        effas = np.zeros((2,) + decs.shape + ras.shape + es.shape)
        
        for jdx, dec in enumerate(tqdm(decs)):
            for idx, nu in enumerate(neutrinos):
                fluxes[idx, jdx, :, :] = [src(nu, e, dec) for e in es]
            for kdx, ra in enumerate(ras):
                sc = SkyCoordinate(dec, ra)
                lc = sky_to_local(sc, det.location, self._t0)
                effas[0, jdx, kdx, :] = [det.response.effective_area["track"](lc.zenith, e) for e in es]
                effas[1, jdx, kdx, :] = [det.response.effective_area["cascade"](lc.zenith, e) for e in es]
                
        track_vals = effas[0, :, :, :] * (fluxes[2, :, :, :] + fluxes[3, :, :, :])
        cascade_vals = effas[1, :, :, :] * np.sum(fluxes, axis=0)
        track_target = RegularGridInterpolator((np.sin(decs), ras, np.log(es)), track_vals)
        cascade_target = RegularGridInterpolator((np.sin(decs), ras, np.log(es)), cascade_vals)

        hack = 1e24

        sds = np.sin(decs)
        out = np.zeros(sds.shape + ras.shape)
        norms = []
        for target in [track_target, cascade_target]:
            for idx, sd in enumerate(tqdm(sds)):
                for jdx, ra in enumerate(ras):
                    f = lambda le: np.exp(le) * target((sd, ra, le)) * hack
                    val, err = quad(f, np.log(es.min()), np.log(es.max()))
                    out[idx, jdx] = val / hack
            intermediate0 = np.zeros(sds.shape)
            for idx in range(len(sds)):
                spl = CubicSpline(ras, out[idx, :])
                val, err = quad(spl, 0, 2*np.pi)
                intermediate0[idx] = val

            spl = CubicSpline(sds, intermediate0 * hack)
            val, err = quad(spl, -1, 1) 
            norm = val / hack
            norms.append(norm)
            
        self._d = {
            "track": (None, track_target, norms[0]),
            "cascade": (None, cascade_target, norms[1]),
        }

    def sample_events(
        self,
        morphology: str,
        oversample: int=1,
        t: Optional[float]=None,
        nevent: Optional[int]=None,
        deltat: Optional[float]=None,
        track=False
    ):
        
        if not ((nevent is None) ^ (deltat is None)):
            raise ValueError("Cannot compute number of events to sample")

        if morphology not in "track cascade".split():
            raise ValueError("Invalid morphology")
            
        xt, target, norm = self._d[morphology]
        if nevent is None:
            nevent = np.random.poisson(norm * deltat)

        if t is None:
            t = self._t0
            
        if xt is None:
            xt = np.array([
                np.random.uniform(),
                np.random.uniform(self._ras.min(), self._ras.max()),
                np.random.uniform(np.log(self._es.min()), np.log(self._es.max()))
            ])
            xt = metropolis_hastings(target, xt, self._bounds, size=self._burnin)[-1, :]
            self._d[morphology] = xt, target, norm
            
        res = metropolis_hastings(target, xt, self._bounds, size=nevent * oversample, track=track)
        if len(res)==0:
            return np.array([])
        xt = res[-1, :]
        self._d[morphology] = xt, target, norm
        offset = 2 * np.pi * ((t - self._t0) % 1)
        res[:, 1] += offset
        res[:, 1] = np.mod(res[:, 1], 2 * np.pi)

        res = res[::oversample]
        events = []
        for x in res:
            true_direction = SkyCoordinate(np.arcsin(x[0]), x[1])
            true_energy = np.exp(x[2])
            reco_direction, reco_energy = smear_truth(
                true_direction,
                true_energy,
                self._det,
                morphology
            )
            # TODO replace this
            signalness = np.random.rand()
            if deltat is None:
                t_event = t
            else:
                t_event = t + np.random.uniform(low=-deltat/2, high=deltat/2) / (24 * 3600 * units.sec)
            event = Event(
                true_direction,
                reco_direction,
                true_energy,
                reco_energy,
                signalness,
                t_event
            )
            events.append(event)
        
        return events
