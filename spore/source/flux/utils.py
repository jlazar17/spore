import h5py as h5
import numpy as np

from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.integrate import quad

from . import units

def parse_2d_file(location: str):
    """Load a 2D (energy × declination) flux table from an HDF5 file.

    The HDF5 group must contain datasets ``sindecs``, ``energies`` (GeV),
    and ``fluxes`` of shape (6, n_dec, n_e) in GeV^{-1} cm^{-2} s^{-1}.

    Args:
        location: String of the form ``"filename.h5:groupname"``.

    Returns:
        Tuple (norms, spls, decmin, decmax, emin, emax) where norms is an
        array of per-species flux normalizations, spls is a list of
        RegularGridInterpolator objects in (sin(dec), log(E)) space,
        and the remaining four values give the grid extent in radians and eV.

    Raises:
        ValueError: If the energy or declination grid is not uniformly spaced
            in log(E) or sin(dec) respectively.
    """
    filename, groupname = location.split(":")
    with h5.File(filename) as h5f:
        gp = h5f[groupname]
        sindecs = gp["sindecs"][:]
        es = gp["energies"][:] * units.GeV
        fluxes = gp["fluxes"][:] / units.GeV / units.cm**2 / units.sec
    
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
    log_es = np.log(es)
    for idx in range(len(norms)):
        flx2d = fluxes[idx, :, :]   # shape (n_dec, n_e)

        if np.all(flx2d == 0):
            spls.append(lambda x: 0.0)
            continue

        # Normalisation: integrate E * flux over log(E) per declination slice,
        # then integrate over sin(dec).  Using trapezoid directly on the flux
        # values avoids taking log(0) for spectra with zeros at grid nodes,
        # which would silently propagate NaN through a log-space spline.
        oned_ints = np.trapezoid(es * flx2d, log_es, axis=1)  # shape (n_dec,)

        spl_dec = CubicSpline(sindecs, oned_ints)
        val, err = quad(spl_dec, -1, 1)
        if val == 0:
            spls.append(lambda x: 0.0)
            continue
        norm = val
        norms[idx] = norm

        # Log-space PCHIP grid for density evaluation.  Replace exact zeros
        # with a small floor so that log() is finite; PCHIP then naturally
        # interpolates to very small values near zero regions rather than
        # producing NaN.  The floor is chosen well below any physical flux so
        # it never attracts the MCMC sampler.
        floor = flx2d[flx2d > 0].min() * 1e-6 if np.any(flx2d > 0) else 1.0
        log_flx_norm = np.log(np.maximum(flx2d / norm, floor))

        # PCHIP: monotone-preserving, no ringing near sharp features or peaks.
        spls.append(RegularGridInterpolator((sindecs, log_es), log_flx_norm, method="pchip"))

    return norms, spls, np.arcsin(sindecs.min()), np.arcsin(sindecs.max()), es.min(), es.max()

def parse_1d_file(location: str):
    """Load a 1D (energy-only) flux table from an HDF5 file.

    The HDF5 group must contain datasets ``energies`` (GeV) and ``fluxes``
    of shape (6, n_e) in GeV^{-1} cm^{-2} s^{-1}.

    Args:
        location: String of the form ``"filename.h5:groupname"``.

    Returns:
        Tuple (norms, spls, emin, emax) where norms is an array of per-species
        flux normalizations, spls is a list of CubicSpline objects in log(E)
        space, and emin/emax give the grid extent in eV.

    Raises:
        ValueError: If the energy grid is not uniformly spaced in log(E).
    """
    filename, groupname = location.split(":")
    with h5.File(filename) as h5f:
        gp = h5f[groupname]
        es = gp["energies"][:] * units.GeV
        fluxes = gp["fluxes"][:, :] / units.GeV / units.cm**2 / units.sec

    dfle = np.diff(np.log(es))
    if not np.all(np.isclose(dfle, dfle[0])):
        raise ValueError

    log_es = np.log(es)
    norms, spls = np.zeros(6), []
    for idx in range(len(norms)):
        flx = fluxes[idx, :]
        if np.all(flx == 0):
            continue

        # Zero-safe normalisation via trapezoid (avoids log(0) = -inf).
        norm = np.trapezoid(es * flx, log_es)
        if norm == 0:
            continue

        norms[idx] = norm

        # Log-space CubicSpline for 1D density evaluation.  Floor zeros so
        # that log() is finite; the floor is far enough below the physical
        # flux that it never affects sampling.
        floor = flx[flx > 0].min() * 1e-6 if np.any(flx > 0) else 1.0
        log_flx_norm = np.log(np.maximum(flx / norm, floor))
        spls.append(CubicSpline(log_es, log_flx_norm))
    return norms, spls, es.min(), es.max()
