import os
import numpy as np

from typing import Tuple
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
    az = np.radians(altaz.az.deg)
    zen = np.pi / 2 - np.radians(altaz.alt.deg)
    return LocalCoordinate(zen, az)

def sample_cone(
    sc: SkyCoordinate,
    psi: float,
) -> SkyCoordinate:
    """Sample a direction uniformly on a cone of half-opening angle psi.

    Args:
        sc: Central direction of the cone.
        psi: Half-opening angle of the cone in radians.

    Returns:
        A sky coordinate drawn uniformly from the ring at angular distance
        psi from sc.
    """
    xyz = sc.to_cartesian()
    xyz_prime = sample_ring(xyz, psi)
    dec = np.arcsin(xyz_prime[2])
    ra = np.arctan2(xyz_prime[1], xyz_prime[0])
    # Move ra to the expected branch
    if ra < 0:
        ra += 2 * np.pi
    return SkyCoordinate(dec, ra)
    

def orthonormal_basis(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute two unit vectors orthogonal to v and to each other.

    Args:
        v: A 3-element unit vector.

    Returns:
        Tuple (u, w) of two unit vectors forming a right-handed basis
        with v.

    Raises:
        ValueError: If v does not have exactly 3 elements.
    """
    if len(v)!=3:
        raise ValueError("This only works for three vectors")
    v /= np.linalg.norm(v)
    # This is the tricky bit.
    # You can run into dumb numerical issues if you don't do this check
    if abs(v[0]) < 0.9:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([0.0, 1.0, 0.0])
    u = np.cross(v, v1)
    w = np.cross(v, u)
    u /= np.linalg.norm(u)
    w /= np.linalg.norm(w)
    return u, w

def sample_ring(v: np.ndarray, psi: float) -> np.ndarray:
    """Sample a unit vector uniformly from the ring at angle psi from v.

    Args:
        v: Central unit vector (will be normalised).
        psi: Opening angle in radians.

    Returns:
        A unit vector at angular distance psi from v with uniform random
        azimuth.
    """
    v /= np.linalg.norm(v)
    u, w = orthonormal_basis(v)
    phi = np.random.uniform(0, 2 * np.pi)
    point = np.sin(psi) * (np.cos(phi) * u + np.sin(phi) * w) + np.cos(psi)*v
    return point
