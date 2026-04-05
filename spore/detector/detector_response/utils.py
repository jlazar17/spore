import sys
import numpy as np

from typing import Callable
from scipy.interpolate import RegularGridInterpolator

def effa_helper(
    zens: np.ndarray,
    es:np.ndarray,
    tabulated_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> Callable:
    """
    Helper function for constructing the effective area spline from
    detector response file saved values.

    params
    ______
    decs: 1D array of declinations with shape (N, )
    es: 1D array of energies with shape (M, )
    tabulated_values: 2D array of effective areas with shape (M, N)
    lower_bounds: Array of coefficients that define the lower polynomial cuts
    upper_bounds: Array of coefficients that define the upper polynomial cuts

    returns
    _______
    fxn: Function that takes declination and energy and returns effective area
    """
    cos_zens = np.cos(zens)
    tabulated_values_scrubbed = np.where(
        tabulated_values > 0,
        tabulated_values,
        tabulated_values[tabulated_values > 0].min()
        #sys.float_info.min
    )
    i = RegularGridInterpolator(
        (np.log(es), cos_zens),
        np.log(tabulated_values_scrubbed)
    )

    le_min, le_max = i.grid[0][0], i.grid[0][-1]
    cz_min, cz_max = i.grid[1][0], i.grid[1][-1]

    def f(zen, e):
        scalar = np.ndim(zen) == 0
        zen = np.atleast_1d(np.asarray(zen, dtype=float))
        e   = np.atleast_1d(np.asarray(e,   dtype=float))
        cz  = np.cos(zen)
        le  = np.log(e)
        mask = (
            (le >= le_min) & (le <= le_max) &
            (cz >= cz_min) & (cz <= cz_max) &
            poly_bounds(cz, e, lower_bounds, upper_bounds)
        )
        result = np.zeros(zen.shape, dtype=float)
        if mask.any():
            pts = np.column_stack([le[mask], cz[mask]])
            result[mask] = np.exp(i(pts))
        return float(result[0]) if scalar else result

    return f

def poly_bounds(
    coszen: np.ndarray,
    e: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> np.ndarray:
    """
    Vectorized bounds check for the effective area.

    params
    ______
    coszen: cos of the zenith angle — scalar or ndarray
    e: energy — scalar or ndarray (same shape as coszen)
    lower_bounds: Array of coefficients that define the lower polynomial cuts
    upper_bounds: Array of coefficients that define the upper polynomial cuts

    returns
    _______
    passed: bool ndarray (or bool scalar) — True where the point lies in the
        region we are considering the effective area
    """
    scalar = np.ndim(coszen) == 0
    coszen = np.atleast_1d(np.asarray(coszen, dtype=float))
    e      = np.atleast_1d(np.asarray(e,      dtype=float))
    y = np.log10(e)

    upper_bool = np.zeros(coszen.shape, dtype=bool)
    for bound in upper_bounds:
        upper_bool |= y < np.poly1d(bound)(coszen)

    lower_bool = np.zeros(coszen.shape, dtype=bool)
    for bound in lower_bounds:
        lower_bool |= y > np.poly1d(bound)(coszen)

    result = lower_bool & upper_bool
    return bool(result[0]) if scalar else result
