import numpy as np

from typing import Callable, Optional
from scipy.interpolate import PchipInterpolator

# Code-level default for effective area Gaussian smoothing (energy bins).
# Can be overridden per-file via the HDF5 meta group or per-call via the
# smoothing_sigma argument.
DEFAULT_SMOOTHING_SIGMA: float = 1.5


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

    Args:
        tabulated_values: ndarray of shape ``(N_E, N_ZEN)``.

    Returns:
        ndarray of shape ``(N_E, N_ZEN)`` — a copy with isolated bins zeroed.
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


def _smooth_columns(tabulated_values: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-smooth each zenith column in log-energy / log-A_eff space.

    MC statistical fluctuations in the high-energy tail of tabulated IRFs
    produce bin-to-bin noise that propagates into the interpolated effective
    area.  Applying a Gaussian filter with sigma ~ 1 energy bin smooths
    through these fluctuations while preserving the overall spectral shape.
    The filter is applied only within the non-zero range of each column so
    that edge effects do not bleed into zeroed bins.

    Args:
        tabulated_values: ndarray of shape ``(N_E, N_ZEN)``.
        sigma: Standard deviation of the Gaussian kernel in units of energy
            bins.  Default 1.0 (one bin width).

    Returns:
        ndarray of shape ``(N_E, N_ZEN)`` — a copy with each column smoothed.
    """
    from scipy.ndimage import gaussian_filter1d

    out = tabulated_values.copy()
    for iz in range(tabulated_values.shape[1]):
        col = out[:, iz]
        nz  = np.where(col > 0)[0]
        if len(nz) < 3:
            continue
        lo, hi = nz[0], nz[-1]
        log_slice = np.log(col[lo : hi + 1])
        out[lo : hi + 1, iz] = np.exp(gaussian_filter1d(log_slice, sigma=sigma))
    return out


def effa_helper(
    zens: np.ndarray,
    es: np.ndarray,
    tabulated_values: np.ndarray,
    trim_isolated: bool = True,
    smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
) -> Callable:
    """Build an effective area callable from tabulated (energy, zenith) values.

    Uses PCHIP interpolation along the log-energy axis (smooth, monotone-
    preserving within each interval) and linear interpolation along the
    cos(zenith) axis (no ringing across zenith nodes).  The cos(zenith) grid
    is linearly extrapolated to the hard boundaries cos = ±1 (zenith = 0° and
    180°) so that queries at the exact poles return physically reasonable
    values rather than the zero fill.

    Args:
        zens: Array of shape ``(N_ZEN,)`` with zenith angles in radians.
        es: Array of shape ``(N_E,)`` with energies in GeV.
        tabulated_values: Array of shape ``(N_E, N_ZEN)`` with effective area
            in cm².
        trim_isolated: If True (default), zero out energy bins outside the
            largest contiguous non-zero run per zenith column before
            interpolating.  This removes isolated low-statistics bins at the
            edges of the sensitivity range.  Set to False to use the raw
            tabulated values as-is.
        smoothing_sigma: Standard deviation (in energy bins) of the Gaussian
            kernel applied to each zenith column after trimming.  Smooths
            over MC statistical noise in the high-energy tail.  Default 1.5.
            Set to 0 to disable smoothing.  **Caveat emptor**: the right value
            is IRF-dependent — always plot the effective area after loading
            and verify the shape looks physically reasonable before using
            SPORE for analysis.

    Returns:
        Callable ``f(zen, e) -> cm²`` with ``f.e_min_gev`` and
        ``f.e_max_gev`` attributes set to the IRF energy grid bounds.
    """
    if trim_isolated:
        tabulated_values = _trim_isolated_bins(tabulated_values)
        if smoothing_sigma > 0:
            tabulated_values = _smooth_columns(tabulated_values, sigma=smoothing_sigma)

    # Dead/trimmed cells use a large negative sentinel in log-space so that
    # PCHIP (which requires all-finite inputs) can be applied.  The sentinel
    # exp(-100) ≈ 3.7e-44 is far below any physical effective area and is
    # zeroed by the threshold check in the returned callable.
    _LOG_ZERO_SENTINEL = -100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.where(tabulated_values > 0, np.log(tabulated_values), _LOG_ZERO_SENTINEL)

    # Sort to ascending cos(zen) so that np.searchsorted works correctly.
    cos_zens = np.cos(zens)
    if cos_zens[0] > cos_zens[-1]:
        cos_zens = cos_zens[::-1]
        log_vals = log_vals[:, ::-1]

    # Linearly extrapolate in cos(zen) to the hard boundaries zen=180° (cos=-1)
    # and zen=0° (cos=+1) so that queries at the poles return valid values.
    if cos_zens[0] > -1.0:
        dcz = cos_zens[1] - cos_zens[0]
        slope = (log_vals[:, 1] - log_vals[:, 0]) / dcz
        log_vals = np.column_stack([log_vals[:, 0] + slope * (-1.0 - cos_zens[0]), log_vals])
        cos_zens = np.concatenate([[-1.0], cos_zens])

    if cos_zens[-1] < 1.0:
        dcz = cos_zens[-1] - cos_zens[-2]
        slope = (log_vals[:, -1] - log_vals[:, -2]) / dcz
        log_vals = np.column_stack([log_vals, log_vals[:, -1] + slope * (1.0 - cos_zens[-1])])
        cos_zens = np.concatenate([cos_zens, [1.0]])

    # Build one PCHIP spline per cos(zen) column.  Evaluating all splines at a
    # query energy and then linearly interpolating between the two bracketing
    # cos(zen) columns gives PCHIP smoothness in energy without the ringing
    # that a 2-D PCHIP produces in the zenith direction at high energies where
    # the effective area drops steeply.
    log_es = np.log(es)
    splines = [PchipInterpolator(log_es, log_vals[:, iz]) for iz in range(len(cos_zens))]

    le_min = log_es[0]
    le_max = log_es[-1]
    cz_min = float(cos_zens[0])
    cz_max = float(cos_zens[-1])

    _nonzero = tabulated_values[tabulated_values > 0]
    _threshold = float(_nonzero.min()) * 0.01 if _nonzero.size > 0 else 1e-30

    def f(zen, e):
        scalar = np.ndim(zen) == 0
        zen = np.atleast_1d(np.asarray(zen, dtype=float))
        e   = np.atleast_1d(np.asarray(e,   dtype=float))
        cz  = np.cos(zen)
        le  = np.log(e)

        mask = (le >= le_min) & (le <= le_max)
        result = np.zeros(zen.shape, dtype=float)

        if not mask.any():
            return float(result[0]) if scalar else result

        le_m  = le[mask]
        cz_m  = np.clip(cz[mask], cz_min, cz_max)

        # Find the two bracketing cos(zen) columns for each query point.
        idx  = np.searchsorted(cos_zens, cz_m)
        idx  = np.clip(idx, 1, len(cos_zens) - 1)
        iz0  = idx - 1
        iz1  = idx

        # Evaluate the PCHIP energy splines for both bracketing columns.
        # splines[iz](le_m) is vectorised over le_m for a fixed column.
        # We need shape (n_mask,) for each column, then gather per-point.
        # Building a (n_cz, n_mask) matrix and indexing is the fastest route.
        unique_cols = np.unique(np.concatenate([iz0, iz1]))
        col_vals = {ic: splines[ic](le_m) for ic in unique_cols}

        v0 = np.array([col_vals[ic][k] for k, ic in enumerate(iz0)])
        v1 = np.array([col_vals[ic][k] for k, ic in enumerate(iz1)])

        # Linear interpolation between the two bracketing cos(zen) values.
        cz0 = cos_zens[iz0]
        cz1 = cos_zens[iz1]
        dcz = cz1 - cz0
        t   = np.where(dcz > 0, (cz_m - cz0) / dcz, 0.0)
        log_v = v0 + t * (v1 - v0)

        raw = np.exp(log_v)
        result[mask] = np.where(raw > _threshold, raw, 0.0)
        return float(result[0]) if scalar else result

    f.e_min_gev = float(np.exp(le_min))
    f.e_max_gev = float(np.exp(le_max))
    return f
