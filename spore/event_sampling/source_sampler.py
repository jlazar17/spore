"""Factory function for constructing event samplers from a source object.

``SourceSampler`` dispatches to ``PointSourceEventSampler`` or
``ExtendedSourceEventSampler`` based on the type of *src*, so callers only
need to know one name.  The returned object is a full ``EventSampler`` with
all the usual methods (``sample_events``, ``expected_events``, etc.).
"""

from ..source import PointSource, ExtendedSource
from .point_source_event_sampler import PointSourceEventSampler
from .extended_source_event_sampler import ExtendedSourceEventSampler


def SourceSampler(
    det,
    src,
    steady_state: bool = False,
    n_time_samples: int = None,
    n_e: int = 50,
    n_dec: int = 40,
    n_ra: int = 40,
    adaptive_energy_grid: bool = True,
    e_min: float = None,
    e_max: float = None,
):
    """Return the appropriate event sampler for *src*.

    Dispatches to ``PointSourceEventSampler`` for ``PointSource`` objects and
    ``ExtendedSourceEventSampler`` for ``ExtendedSource`` objects.  Parameters
    that do not apply to the selected sampler (``n_dec``, ``n_ra``, and
    ``adaptive_energy_grid`` for point sources) are silently ignored.

    Args:
        det: A ``Detector`` or list of ``Detector`` objects.
        src: The neutrino source.  Must be a ``PointSource`` or
            ``ExtendedSource``.
        steady_state: Average the effective area over a full diurnal cycle.
            Appropriate for observations spanning many sidereal days.
        n_time_samples: Hour-angle samples for the diurnal average.  Implies
            ``steady_state=True``.  Default 100 when steady-state mode is
            active.
        n_e: Number of energy grid points.  Default 50.
        n_dec: Number of sin(dec) grid points (extended source only).
            Default 40.
        n_ra: Number of RA grid points (extended source only).  Default 40.
        adaptive_energy_grid: Concentrate energy grid points near the
            detection threshold (extended source only).  Default True.
        e_min: Minimum energy in GeV.  If None, read from the detector
            response.
        e_max: Maximum energy in GeV.  If None, read from the detector
            response.

    Returns:
        A ``PointSourceEventSampler`` or ``ExtendedSourceEventSampler``
        instance.

    Raises:
        TypeError: If *src* is not a ``PointSource`` or ``ExtendedSource``.
    """
    if isinstance(src, PointSource):
        return PointSourceEventSampler(
            det, src,
            steady_state=steady_state,
            n_time_samples=n_time_samples,
            n_e=n_e,
            e_min=e_min,
            e_max=e_max,
        )
    if isinstance(src, ExtendedSource):
        return ExtendedSourceEventSampler(
            det, src,
            steady_state=steady_state,
            n_time_samples=n_time_samples,
            n_e=n_e,
            n_dec=n_dec,
            n_ra=n_ra,
            adaptive_energy_grid=adaptive_energy_grid,
            e_min=e_min,
            e_max=e_max,
        )
    raise TypeError(
        f"src must be a PointSource or ExtendedSource, got {type(src).__name__}"
    )
