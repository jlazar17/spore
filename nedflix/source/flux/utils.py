import h5py as h5
import numpy as np

from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.integrate import quad

def parse_2d_file(location: str):
    filename, groupname = location.split(":")
    with h5.File(filename) as h5f:
        gp = h5f[groupname]
        sindecs = gp["sindecs"][:]
        es = gp["energies"][:]
        fluxes = gp["fluxes"][:]
    
    # Verify equal spacing where expected. Not sure if this is necessary
    # But I didn't test it with other stuff
    dfsd = np.diff(sindecs)
    dfle = np.diff(np.log(es))
    if not np.all(np.isclose(dfsd, dfsd[0])):
        raise ValueError
    if not np.all(np.isclose(dfle, dfle[0])):
        raise ValueError

    norms = np.zeros(6)
    spls = []
    for idx in range(len(norms)):
        oned_ints = np.zeros(len(sindecs))
        for jdx in range(len(oned_ints)):
            flx = fluxes[idx, jdx, :]
            if np.all(flx==0):
                continue
            spl = CubicSpline(np.log(es), np.log(flx))
            f = lambda e: np.exp(spl(np.log(e)))
            hack = 1 / f(es[0])
            g = lambda le: hack * np.exp(le) * f(np.exp(le))
            val, err = quad(g, np.log(es[0]), np.log(es[-1]))
            if err / val > 1e-5:
                raise ValueError
            oned_ints[jdx] = val / hack

        if np.all(oned_ints==0):
            spls.append(lambda x: 0.0)
            continue

        spl = CubicSpline(sindecs, oned_ints)
        hack = 1 / spl(sindecs[0])
        h = lambda x: hack * spl(x)
        val, err = quad(h, -1, 1)
        if err / val > 1e-5:
            raise ValueError
        norm = val / hack
        norms[idx] = norm

        spls.append(RegularGridInterpolator((sindecs, np.log(es)), np.log(fluxes[idx, :, :] / norm)))

    return norms, spls, np.arcsin(sindecs.min()), np.arcsin(sindecs.max()), es.min(), es.max()

def parse_1d_file(location: str):
    filename, groupname = location.split(":")
    with h5.File(filename) as h5f:
        gp = h5f[groupname]
        es = gp["energies"][:]
        fluxes = gp["fluxes"][:, :]

    dfle = np.diff(np.log(es))
    if not np.all(np.isclose(dfle, dfle[0])):
        raise ValueError

    norms, spls = np.zeros(6), []
    for idx in range(len(norms)):
        flx = fluxes[idx, :]
        if np.all(flx==0):
            continue
        spl = CubicSpline(np.log(es), np.log(flx))
        f = lambda e: np.exp(spl(np.log(e)))
        hack = 1 / f(es[0])
        g = lambda le: hack * np.exp(le) * f(np.exp(le))
        val, err = quad(g, np.log(es[0]), np.log(es[-1]))
        if err / val > 1e-5:
            raise ValueError
        spls.append(spl)
        norms[idx] = val / hack
    return norms, spls, es.min(), es.max()
