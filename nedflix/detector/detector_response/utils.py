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

    def f(zen, e, interp, lbs, ubs):
        cz = np.cos(zen)
        le = np.log(e)
        # return 0 if outside of interpolator bounds
        if le < interp.grid[0][0] or le > interp.grid[0][-1]:
            return 0.0
        if cz < interp.grid[1][0] or cz > interp.grid[1][-1]:
            return 0.0
        # return 0 if bounds say so
        if not poly_bounds(cz, e, lower_bounds, upper_bounds):
            return 0.0
        return np.exp(interp((le, cz)))
    
    fxn = lambda zen, e: f(zen, e, i, lower_bounds, upper_bounds)
    return fxn

def poly_bounds(
    coszen: float,
    e: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> bool:
    """
    Helper function for computing the bounds we put on the effective area

    params
    ______
    coszen: cos of the zenith angle
    e: energy
    lower_bounds: Array of coefficients that define the lower polynomial cuts
    upper_bounds: Array of coefficients that define the upper polynomial cuts

    returns
    _______
    passed: Whether the point lies in the region we are considering the
        effective area
    """
    y = np.log10(e)
    
    upper_bool = False
    for bound in upper_bounds:
        f = np.poly1d(bound)
        if y < f(coszen):
            upper_bool = True
            break
    
    lower_bool = False
    for bound in lower_bounds:
        f = np.poly1d(bound)
        if y > f(coszen):
            lower_bool = True
    
    return lower_bool and upper_bool
