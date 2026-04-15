import os
import numpy as np

from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy import units as u

from ..conventions import SkyCoordinate, LocalCoordinate, EarthCoordinate


def resolve_path(path: str, toml_dir: str) -> str:
    """
    Resolve a path referenced inside a TOML file.

    Resolution order:
    1. Relative to the directory containing the TOML file.
    2. Relative to the current working directory.
    3. As an absolute path (or already-absolute path passed through directly).

    Parameters
    ----------
    path : str
        The raw path string from the TOML value.
    toml_dir : str
        Absolute path to the directory containing the TOML file.

    Returns
    -------
    str
        The resolved absolute path.

    Raises
    ------
    FileNotFoundError
        If none of the three candidates exist on disk.
    """
    candidates = [
        os.path.join(toml_dir, path),
        os.path.join(os.getcwd(), path),
        path,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Could not resolve path {path!r}.\nTried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )

def sky_to_local(
    sc: SkyCoordinate,
    ec: EarthCoordinate,
    t: float,
) -> LocalCoordinate:
    """
    Convert a sky-fixed coordinate to a local detector coordinate given
    a particular time. Uses `astropy` as an interface to accomplish this

    params
    ______
    sc: the sky-fixed coordinate
    ec: the earth coordinate of the detector
    t: time at which the event was observed in modified julian days

    returns
    _______
    lc: the local, detector coordinate
    """
    t = Time(t, format="mjd")
    ec = EarthLocation.from_geodetic(
        lon=ec.longitude * u.rad,
        lat=ec.latitude * u.rad
    )
    sc = SkyCoord(
        ra=sc.right_ascension * u.rad,
        dec=sc.declination * u.rad
    )

    altaz = sc.transform_to(AltAz(location=ec, obstime=t))
    az  = float(altaz.az.rad)
    # arccos(sin(alt)) = π/2 - alt but stays in [0, π] even when floating-point
    # noise pushes alt fractionally above 90°, avoiding a ValueError in LocalCoordinate.
    zen = float(np.arccos(np.sin(altaz.alt.rad)))
    return LocalCoordinate(zen, az)

