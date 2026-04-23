"""Good run list (GRL) support for SPORE event samplers.

A GoodRunList holds a set of (start, stop) MJD intervals representing periods
of valid detector livetime.  It can be constructed from a single uptime CSV
file, a directory of CSV files, or a list of either.

The uptime CSV format used by IceCube public data releases is::

    #  MJD_start[days]  MJD_stop[days]
       55694.99047453   55695.32498842
       ...

(two whitespace-separated columns, comment lines prefixed with ``#``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

from ..conventions import ureg


class GoodRunList:
    """A collection of good-run intervals with utilities for livetime and time sampling.

    Args:
        runs: Array of shape (N, 2) where each row is [mjd_start, mjd_stop].
    """

    def __init__(self, runs: np.ndarray) -> None:
        runs = np.asarray(runs, dtype=float)
        if runs.ndim != 2 or runs.shape[1] != 2:
            raise ValueError("runs must have shape (N, 2)")
        durations = runs[:, 1] - runs[:, 0]
        if np.any(durations < 0):
            raise ValueError("All runs must have stop >= start")
        self._runs = runs
        self._durations = durations
        self._total_days = float(durations.sum())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _load_one(cls, path: Union[str, Path]) -> np.ndarray:
        """Load runs from a single file or every *_exp*.csv in a directory."""
        path = Path(path)
        if path.is_file():
            return np.atleast_2d(np.genfromtxt(path, comments="#"))
        if path.is_dir():
            files = sorted(path.glob("*_exp*.csv"))
            if not files:
                raise FileNotFoundError(
                    f"No *_exp*.csv files found in directory: {path}"
                )
            return np.vstack([np.atleast_2d(np.genfromtxt(f, comments="#")) for f in files])
        raise FileNotFoundError(f"Path does not exist: {path}")

    @classmethod
    def from_path(
        cls, source: Union[str, Path, List[Union[str, Path]]]
    ) -> "GoodRunList":
        """Construct a GoodRunList from a path, directory, or list of either.

        Args:
            source: One of:

                * A path to a single uptime CSV file.
                * A path to a directory — all ``*_exp*.csv`` files are loaded.
                * A list of file and/or directory paths — each element is
                  resolved by the same file-or-directory logic and the results
                  are concatenated.
        """
        if isinstance(source, list):
            parts = [cls._load_one(p) for p in source]
            runs = np.vstack(parts)
        else:
            runs = cls._load_one(source)
        return cls(runs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def runs(self) -> np.ndarray:
        """(N, 2) array of [mjd_start, mjd_stop] pairs."""
        return self._runs

    @property
    def total_livetime(self):
        """Total livetime as a pint Quantity in days."""
        return ureg.Quantity(self._total_days, "day")

    @property
    def t_start(self) -> float:
        """MJD of the earliest run start — used as the sampler reference epoch."""
        return float(self._runs[:, 0].min())

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_times(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` MJD arrival times uniformly distributed within good runs.

        Each run is selected with probability proportional to its duration,
        then a uniform draw is made within that run.

        Args:
            n: Number of times to draw.
            rng: NumPy random Generator.

        Returns:
            Array of shape ``(n,)`` with MJD floats.
        """
        if n == 0:
            return np.empty(0, dtype=float)
        weights = self._durations / self._total_days
        run_idx = rng.choice(len(self._runs), size=n, p=weights)
        starts  = self._runs[run_idx, 0]
        widths  = self._durations[run_idx]
        return starts + rng.uniform(0.0, 1.0, n) * widths
