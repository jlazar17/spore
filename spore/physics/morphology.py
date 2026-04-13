from typing import FrozenSet, Optional


class Morphology:
    """String constants and registry for event morphology labels.

    Standard morphologies are pre-registered with their neutrino flavor sets.
    To add a new one, call ``Morphology.register`` before constructing any
    ``DetectorResponse``:

        Morphology.register("double_cascade")

    An explicit flavor set can be provided:

        Morphology.register("tau_track", flavors="numu")
        Morphology.register("radio_shower", flavors="all")

    If *flavors* is omitted the flavor set is inferred from the name: morphology
    names containing ``"track"`` map to ``"numu"`` (ν_μ + ν̄_μ only); all others
    map to ``"all"`` (all six species).

    The registered name is returned by ``register``, so it can also be used as
    an inline assignment:

        DOUBLE_CASCADE = Morphology.register("double_cascade")

    ``DetectorResponse`` loaders iterate over the registry when scanning HDF5
    files and warn about any groups that match the morphology naming convention
    but are not registered.
    """

    # Maps registered name -> flavor set: "numu" or "all"
    _registry: dict = {
        "track":   "numu",
        "cascade": "all",
    }

    TRACK   = "track"
    CASCADE = "cascade"

    @classmethod
    def register(cls, name: str, flavors: Optional[str] = None) -> str:
        """Add *name* to the morphology registry and return it.

        Args:
            name: Morphology label to register (e.g. ``"double_cascade"``).
            flavors: Neutrino flavor set to use when building the sampling
                weight grid.  Either ``"numu"`` (ν_μ + ν̄_μ only, appropriate
                for track-like morphologies) or ``"all"`` (all six species,
                appropriate for cascade-like morphologies).  If ``None``
                (default), the flavor set is inferred from the name: names
                containing ``"track"`` get ``"numu"``; all others get
                ``"all"``.

        Returns:
            The registered name, unchanged.

        Raises:
            ValueError: If *flavors* is not ``None``, ``"numu"``, or
                ``"all"``.
        """
        if flavors is not None and flavors not in ("numu", "all"):
            raise ValueError(
                f"flavors must be 'numu' or 'all', got {flavors!r}"
            )
        if flavors is None:
            flavors = "numu" if "track" in name else "all"
        cls._registry[name] = flavors
        return name

    @classmethod
    def registered(cls) -> frozenset:
        """Return an immutable snapshot of the current registry (names only)."""
        return frozenset(cls._registry)

    @classmethod
    def flavor_set(cls, name: str) -> str:
        """Return the flavor set for a registered morphology.

        Falls back to the name-based heuristic for unregistered names so that
        samplers can call this without first checking registration status.

        Args:
            name: Morphology label.

        Returns:
            ``"numu"`` or ``"all"``.
        """
        if name in cls._registry:
            return cls._registry[name]
        return "numu" if "track" in name else "all"
