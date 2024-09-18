import numpy as np

from typing import Tuple
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy import units as u

from ..conventions import SkyCoordinate, LocalCoordinate, EarthCoordinate

def sky_to_local(
    sc: SkyCoordinate,
    ec: EarthCoordinate,
    t: float,
    depth: float=0.0
) -> LocalCoordinate:
    """
    Convert a sky-fixed coordinate to a local detector coordinate given
    a particular time. Uses `astropy` as an interface to accomplish this

    params
    ______
    sc: the sky-fixed coordinate
    ec: the earth coordinate of the detector
    t: time at which the event was observed
    [depth]: depth of the detector

    returns
    _______
    lc: the local, detector coordinate
    """
    if depth!=0:
        raise NotImplementedError("Depth handling not implemented yet")
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
    psi: float
) -> SkyCoordinate:
    xyz = sc.to_cartesian()
    xyz_prime = sample_ring(xyz, psi)
    dec = np.arcsin(xyz_prime[2])
    ra = np.arctan2(xyz_prime[1], xyz_prime[0])
    # Move ra to the expected branch
    if ra < 0:
        ra += 2 * np.pi
    return SkyCoordinate(dec, ra)
    

def orthonormal_basis(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    v /= np.linalg.norm(v)
    u, w = orthonormal_basis(v)
    phi = np.random.uniform(0, 2 * np.pi)
    point = np.sin(psi) * (np.cos(phi) * u + np.sin(phi) * w) + np.cos(psi)*v
    return point
