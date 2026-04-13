import sys
import numpy as np

from typing import Callable
from scipy.interpolate import RegularGridInterpolator


def _largest_contiguous_nonzero(arr: np.ndarray):
    """Return (lo, hi) inclusive indices of the largest contiguous non-zero run.

    Returns None if the array is entirely zero.
    """
    is_nz   = arr > 0
    changes = np.diff(is_nz.astype(int), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]   # exclusive
    if len(starts) == 0:
        return None
    best = np.argmax(ends - starts)
    return starts[best], ends[best] - 1    # inclusive


def _trim_isolated_bins(tabulated_values: np.ndarray) -> np.ndarray:
    """Zero out energy bins outside the largest contiguous non-zero run per zenith column.

    For each zenith column the algorithm finds the longest contiguous run of
    non-zero energy bins and zeros everything outside it.  This removes
    isolated low-statistics bins at the edges of the sensitivity range without
    touching the main body of the effective area.

    Parameters
    ----------
    tabulated_values : ndarray, shape (N_E, N_ZEN)

    Returns
    -------
    ndarray, shape (N_E, N_ZEN)  — copy with isolated bins zeroed.
    """
    out = tabulated_values.copy()
    for iz in range(tabulated_values.shape[1]):
        result = _largest_contiguous_nonzero(tabulated_values[:, iz])
        if result is None:
            out[:, iz] = 0.0
            continue
        lo, hi = result
        out[:lo, iz]    = 0.0
        out[hi + 1:, iz] = 0.0
    return out


def effa_helper(
    zens: np.ndarray,
    es: np.ndarray,
    tabulated_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    trim_isolated: bool = True,
) -> Callable:
    """
    Helper function for constructing the effective area spline from
    detector response file saved values.

    params
    ______
    zens: 1D array of zenith angles with shape (N,)
    es: 1D array of energies in GeV with shape (M,)
    tabulated_values: 2D array of effective areas with shape (M, N)
    lower_bounds: Array of coefficients that define the lower polynomial cuts
    upper_bounds: Array of coefficients that define the upper polynomial cuts
    trim_isolated: If True (default), zero out energy bins outside the largest
        contiguous non-zero run per zenith column before interpolating.  This
        removes isolated low-statistics bins at the edges of the sensitivity
        range.  Set to False to use the raw tabulated values as-is.

    returns
    _______
    fxn: Function that takes zenith and energy and returns effective area
    """
    if trim_isolated:
        tabulated_values = _trim_isolated_bins(tabulated_values)

    cos_zens = np.cos(zens)
    # Use NaN for zero/trimmed cells so the interpolator propagates NaN into
    # any grid cell that touches a dead bin, rather than filling with a floor
    # value that would appear as spurious low-level effective area.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.where(tabulated_values > 0, np.log(tabulated_values), np.nan)
    i = RegularGridInterpolator(
        (np.log(es), cos_zens),
        log_vals,
        bounds_error=False,
        fill_value=np.nan,
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
            raw = i(pts)
            # NaN indicates a trimmed/dead cell — leave those as 0
            result[mask] = np.where(np.isnan(raw), 0.0, np.exp(raw))
        return float(result[0]) if scalar else result

    f.e_min_gev = float(np.exp(le_min))
    f.e_max_gev = float(np.exp(le_max))
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
