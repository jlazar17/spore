import numpy as np

from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

from ..conventions import units, SkyCoordinate, sky_to_local
from ..physics import neutrinos
from ..source import ExtendedSource
from ..detector import Detector

from .event_sampler import EventSampler
from .metropolis_hastings import metropolis_hastings
from .utils import new_sample

class ExtendedSourceEventSampler(EventSampler):

    def __init__(self, det: Detector, src: ExtendedSource, burnin: int=10_000):

        super().__init__(det, src, burnin=burnin)

        ras = np.linspace(0, 2*np.pi, 101)
        decs = np.asin(np.linspace(-1, 1, 100))
        es = np.logspace(2, 6, 100) * units.GeV

        self._bounds = [
            (np.sin(decs).min(), np.sin(decs).max()),
            (ras.min(), ras.max()),
            (np.log(es).min(), np.log(es).max()),
        ]
    
        self._ras = ras
        self._decs = decs
        self._es = es
        if burnin is None:
            self._burnin = 10_000
        
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
            
        self._d = {
            "track": (None, track_target),
            "cascade": (None, cascade_target),
        }

    def sample_events(self, morphology, t=None, nevent=1, oversample=1):
        
        if morphology not in "track cascade".split():
            raise ValueError("Invalid morphology")
            
        if t is None:
            t = self._t0
            
        xt, target = self._d[morphology]
        if xt is None:
            xt = np.array([
                np.random.uniform(),
                np.random.uniform(self._ras.min(), self._ras.max()),
                np.random.uniform(np.log(self._es.min()), np.log(self._es.max()))
            ])
            xt = metropolis_hastings(target, xt, self._bounds, size=self._burnin)[-1, :]
            self._d[morphology] = xt, target
            
        res = metropolis_hastings(target, xt, self._bounds, size=nevent * oversample)
        xt = res[-1, :]
        self._d[morphology] = xt, target
        offset = 2 * np.pi * ((t - self._t0) % 1)
        res[:, 1] += offset
        res[:, 1] = np.mod(res[:, 1], 2 * np.pi)
        return res[::oversample]
