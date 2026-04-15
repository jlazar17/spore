import logging
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

from ..conventions import ureg
from ..source import Source
from ..detector import Detector
from .event import Event

_STEADY_STATE_THRESHOLD = ureg.Quantity(15.0, "min")


class EventSampler(ABC):
    """Abstract base class for all SPORE event samplers.

    Subclasses implement sample_events and expected_events for a specific
    source geometry (point source, extended source, galactic halo).

    Args:
        det: The detector to simulate.
        src: The neutrino source to sample from.
    """

    def __init__(self, det: Detector, src: Source):
        self._det = det
        self._src = src
        self._track_xt = None
        self._cascade_xt = None
        self._t0 = 60_355.83
        self._steady_state = False  # subclasses override this

    def _resolve_per_detector(self, value, name: str) -> list:
        """Broadcast a scalar or validate a per-detector list.

        None is broadcast as [None, None, ...].  A list must have the same
        length as the number of detectors.  Any other value is broadcast.
        """
        n = len(self._samplers)
        if value is None:
            return [None] * n
        if isinstance(value, list):
            if len(value) != n:
                raise ValueError(
                    f"{name} list length ({len(value)}) must match the number "
                    f"of detectors ({n})."
                )
            return value
        return [value] * n

    def _check_steady_state(self, deltat) -> None:
        """Warn if a long deltat is used without steady-state averaging."""
        if deltat is None or self._steady_state:
            return
        if deltat.to("min") > _STEADY_STATE_THRESHOLD:
            logger.warning(
                "deltat=%s exceeds %s but steady_state=False. "
                "The effective area is evaluated at a single epoch, which "
                "underestimates the true time-averaged rate for non-polar "
                "detectors. Pass steady_state=True to the sampler constructor "
                "to enable hour-angle averaging.",
                deltat, _STEADY_STATE_THRESHOLD,
            )

    def sample_events(
        self,
        morphology: str,
        deltat=None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        seed=None,
        delta_clip: tuple = None,
    ) -> List[Event]:
        """Sample a list of events from the source and detector.

        Exactly one of deltat or nevent must be provided. When deltat is given
        the number of events is drawn from a Poisson distribution with mean
        equal to expected_events(morphology, deltat, t).

        Args:
            morphology: Event morphology to sample, either "track" or "cascade".
            deltat: Observation window as a pint Quantity with time units.
                Controls the Poisson mean when ``nevent`` is None, and sets the
                time spread ``[t, t + deltat]`` whenever it is provided.
            nevent: Fixed event count.  When provided alongside ``deltat``,
                the count is fixed but times are still drawn over
                ``[t, t + deltat]``.  At least one of ``deltat`` or ``nevent``
                must be given.
            t: Reference epoch in MJD. Event times start at ``t``.  Defaults
                to the sampler's internal reference epoch (2024-01-01).
            seed: Random seed. Pass different values across pseudo-experiments.
            delta_clip: Clips the log-energy smearing Delta = ln(E_reco/E_true)
                to (lo, hi). Useful for suppressing tail artefacts in validation.

        Returns:
            List of sampled Event objects.
        """
        if self._multi:
            deltats     = self._resolve_per_detector(deltat,     "deltat")
            ts          = self._resolve_per_detector(t,          "t")
            delta_clips = self._resolve_per_detector(delta_clip, "delta_clip")
            import numpy as np
            rng = np.random.default_rng(seed)
            all_events = []
            for idx, (sampler, dt, t_, dc) in enumerate(
                zip(self._samplers, deltats, ts, delta_clips)
            ):
                events = sampler.sample_events(
                    morphology, deltat=dt, nevent=nevent, t=t_,
                    seed=int(rng.integers(2**32)), delta_clip=dc,
                )
                for ev in events:
                    ev.detector_id = idx
                all_events.extend(events)
            return all_events
        return self._sample_events_single(
            morphology, deltat=deltat, nevent=nevent, t=t, seed=seed, delta_clip=delta_clip,
        )

    @abstractmethod
    def _sample_events_single(
        self,
        morphology: str,
        deltat=None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        seed=None,
        delta_clip: tuple = None,
    ) -> List[Event]:
        """Single-detector implementation of sample_events.  Override in subclasses."""
        pass

    def expected_events(
        self,
        morphology: str,
        deltat,
        t: Optional[float] = None,
    ):
        """Return the expected (mean) number of events for a given livetime.

        Args:
            morphology: Event morphology, e.g. "track" or "cascade".
            deltat: Observation window as a pint Quantity with time units.
            t: Reference epoch in MJD.  Defaults to the sampler's reference epoch.

        Returns:
            Expected number of events (before Poisson sampling).  Returns a list
            when multi-detector mode is active.
        """
        if self._multi:
            deltats = self._resolve_per_detector(deltat, "deltat")
            ts      = self._resolve_per_detector(t,      "t")
            return [
                s._expected_events_single(morphology, dt, t=t_)
                for s, dt, t_ in zip(self._samplers, deltats, ts)
            ]
        return self._expected_events_single(morphology, deltat, t=t)

    @abstractmethod
    def _expected_events_single(
        self,
        morphology: str,
        deltat,
        t: Optional[float] = None,
    ) -> float:
        """Single-detector implementation of expected_events.  Override in subclasses."""
        pass

    def eddington_correction(
        self,
        morphology: str,
        e_bins,
        n_pseudo: int = 500,
        seed: int = 42,
    ) -> np.ndarray:
        """Estimate Eddington bias correction factors for reco-energy bins.

        Runs ``n_pseudo`` pseudo-experiments at fixed event count, histograms
        both the true energies and the reconstructed energies, and returns
        their ratio per bin.

        Because true energies are drawn from A_eff × flux, the true-energy
        histogram *is* the unsmeared prediction.  The reco-energy histogram
        is the smeared prediction.  The ratio

            correction[k] = smeared[k] / unsmeared[k]

        converts an unsmeared (naive A_eff × flux) prediction into the correct
        smeared expectation, or equivalently can be used to debias a sampled
        reco-energy histogram back to the unsmeared distribution.

        For steeply falling spectra combined with a systematic downward energy
        shift (e.g. HESE tracks, median Δ ≈ −0.65), correction > 1 at low
        reco energies and correction < 1 at high reco energies.

        Args:
            morphology: Event morphology (e.g. "astro_track").
            e_bins: Reconstructed energy bin edges in GeV.  Log-spaced bins are
                recommended.
            n_pseudo: Number of pseudo-experiments.  Default 500 gives ~1–2%
                statistical uncertainty on the correction factors.
            seed: Base random seed; each pseudo-experiment uses ``seed + i``.

        Returns:
            ndarray of shape ``(len(e_bins) - 1,)`` with ratio smeared /
            unsmeared per bin.  Bins with zero unsmeared counts return
            ``np.nan``.
        """
        e_bins = np.asarray(e_bins, dtype=float)
        smeared   = np.zeros(len(e_bins) - 1)
        unsmeared = np.zeros(len(e_bins) - 1)

        # Use a fixed nevent so both histograms have the same denominator.
        n_ev = max(1, round(self.expected_events(morphology, ureg.Quantity(365.25, "day"))))

        for i in range(n_pseudo):
            events = self.sample_events(morphology, nevent=n_ev, seed=seed + i)
            if not events:
                continue
            true_es = np.array([ev.true_energy.to("GeV").magnitude for ev in events])
            reco_es = np.array([ev.reco_energy.to("GeV").magnitude for ev in events])
            smeared   += np.histogram(reco_es,  bins=e_bins)[0]
            unsmeared += np.histogram(true_es,  bins=e_bins)[0]

        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(unsmeared > 0, smeared / unsmeared, np.nan)

    def expected_rate(
        self,
        morphology: str,
        t: Optional[float] = None,
    ):
        """Return the expected event rate as a pint Quantity in s^{-1}.

        Args:
            morphology: Event morphology, e.g. "track" or "cascade".
            t: Reference epoch in MJD.

        Returns:
            Event rate as a pint Quantity with units of 1/s.
        """
        one_second = ureg.Quantity(1.0, "s")
        return ureg.Quantity(
            self.expected_events(morphology, one_second, t=t), "1/s"
        )
