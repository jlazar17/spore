import numpy as np

from tqdm import tqdm


def metropolis_hastings(
    target_density,
    x0,
    bounds,
    size=500000,
    step_size=None,
    track=False,
    periodic=None,
):
    """
    Metropolis-Hastings sampler with a local random-walk proposal.

    Parameters
    ----------
    target_density : callable
        Unnormalised target density. Must accept a 1-D array and return a
        non-negative scalar.
    x0 : ndarray
        Initial state, shape (n_dims,).
    bounds : list of (lb, ub) pairs
        Hard bounds for each dimension.
    size : int
        Number of samples to draw (after the caller's burn-in).
    step_size : ndarray or None
        Per-dimension half-width of the symmetric uniform proposal
        U(xt - step, xt + step).  If None, defaults to 10 % of each
        dimension's range.
    track : bool
        Show a tqdm progress bar.
    periodic : sequence of bool or None
        If True for dimension i, proposals that leave the bounds are wrapped
        around (modular arithmetic) rather than reflected.  Use this for
        angles such as right ascension where the two boundary values are
        physically identical.  If None, all dimensions are treated as
        non-periodic.
    """
    bounds_arr = np.array(bounds)
    lbs = bounds_arr[:, 0]
    ubs = bounds_arr[:, 1]

    if step_size is None:
        step_size = (ubs - lbs) * 0.1

    step_size = np.asarray(step_size)

    n_dims = len(bounds)
    if periodic is None:
        periodic = [False] * n_dims
    periodic = list(periodic)

    xt = x0.copy()
    samples = []
    itr = range(size)
    if track:
        itr = tqdm(itr)

    for _ in itr:
        xt_candidate = xt + np.random.uniform(-step_size, step_size)
        for i in range(len(xt_candidate)):
            lb, ub = lbs[i], ubs[i]
            if periodic[i]:
                # Wrap around: physically lb and ub are the same point.
                xt_candidate[i] = lb + (xt_candidate[i] - lb) % (ub - lb)
            else:
                # Reflect off boundaries.  Clipping would map all out-of-bounds
                # proposals to the boundary value, creating artificial
                # accumulation there.  Reflection folds the proposal back inside
                # the domain and keeps the proposal distribution symmetric.
                if xt_candidate[i] < lb:
                    xt_candidate[i] = 2 * lb - xt_candidate[i]
                if xt_candidate[i] > ub:
                    xt_candidate[i] = 2 * ub - xt_candidate[i]
                # Fallback clip in case a very large step double-reflects out of range
                xt_candidate[i] = np.clip(xt_candidate[i], lb, ub)

        try:
            denom = target_density(xt)
            if denom == 0:
                accept_prob = 1.0
            else:
                accept_prob = target_density(xt_candidate) / denom
        except ValueError as e:
            print(bounds)
            print(xt, xt_candidate)
            raise e

        if np.random.uniform(0, 1) < accept_prob:
            xt = xt_candidate
        samples.append(xt.copy())

    samples = np.array(samples)
    samples = np.reshape(samples, [samples.shape[0], xt.shape[0]])
    return samples
