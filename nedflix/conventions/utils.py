import numpy as np

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
