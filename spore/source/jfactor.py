"""
J-factor computation for dark matter density profiles.

The J-factor is the line-of-sight integral of the squared DM density:

    J(ψ) = ∫₀^{D_max} ρ²(r(l, ψ)) dl

where r(l, ψ) = sqrt(R_sun² + l² − 2 R_sun l cos ψ) and ψ is the
angle from the Galactic Centre.

Densities are in GeV cm⁻³, distances in kpc. The returned J-factor
is in GeV² cm⁻⁵.
"""

import numpy as np
from scipy.integrate import quad

KPC_TO_CM = 3.0857e21   # cm per kpc
_R_SUN_DEFAULT = 8.5    # kpc, IAU recommended


class NFWProfile:
    r"""Navarro-Frenk-White dark matter density profile.

    ρ(r) = ρ_s / [(r / r_s)(1 + r / r_s)²]

    Parameters
    ----------
    rho_s : float
        Characteristic density [GeV cm⁻³].
    r_s : float
        Scale radius [kpc].
    r_sun : float
        Sun-to-GC distance [kpc]. Default 8.5.
    """

    def __init__(self, rho_s: float, r_s: float, r_sun: float = _R_SUN_DEFAULT):
        self.rho_s = rho_s
        self.r_s = r_s
        self.r_sun = r_sun

    def density(self, r: float) -> float:
        """Density at Galactocentric radius r [kpc], in GeV cm⁻³."""
        x = r / self.r_s
        return self.rho_s / (x * (1.0 + x) ** 2)


class EinastoProfile:
    r"""Einasto dark matter density profile.

    ρ(r) = ρ_s exp(−(2/α)[(r / r_s)^α − 1])

    Parameters
    ----------
    rho_s : float
        Characteristic density [GeV cm⁻³].
    r_s : float
        Scale radius [kpc].
    alpha : float
        Shape parameter. Default 0.17 (Milky Way best-fit, Retana-Montenegro+2012).
    r_sun : float
        Sun-to-GC distance [kpc]. Default 8.5.
    """

    def __init__(
        self,
        rho_s: float,
        r_s: float,
        alpha: float = 0.17,
        r_sun: float = _R_SUN_DEFAULT,
    ):
        self.rho_s = rho_s
        self.r_s = r_s
        self.alpha = alpha
        self.r_sun = r_sun

    def density(self, r: float) -> float:
        """Density at Galactocentric radius r [kpc], in GeV cm⁻³."""
        return self.rho_s * np.exp(
            -(2.0 / self.alpha) * ((r / self.r_s) ** self.alpha - 1.0)
        )


def j_factor(
    profile,
    psi: float,
    d_max_kpc: float = 200.0,
) -> float:
    """
    Line-of-sight integral of ρ²(r) toward angle ψ from the Galactic Centre.

    Parameters
    ----------
    profile : NFWProfile or EinastoProfile
        DM density profile.
    psi : float
        Angle from the Galactic Centre [radians].
    d_max_kpc : float
        Integration truncation distance [kpc]. Default 200.

    Returns
    -------
    float
        J-factor in GeV² cm⁻⁵.
    """
    cos_psi = np.cos(psi)
    r_sun = profile.r_sun

    def integrand(l):
        r2 = r_sun**2 + l**2 - 2.0 * r_sun * l * cos_psi
        r = np.sqrt(max(r2, 1e-4))
        return profile.density(r) ** 2

    val, _ = quad(integrand, 0.0, d_max_kpc, limit=200)
    return val * KPC_TO_CM


def psi_from_radec_grid(ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
    """
    Vectorised version of psi_from_radec for arrays of (RA, dec) pairs.

    Parameters
    ----------
    ra_rad  : ndarray  Right ascensions [radians], any shape.
    dec_rad : ndarray  Declinations [radians], same shape as ra_rad.

    Returns
    -------
    ndarray  Angles from the Galactic Centre [radians], same shape as input.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    shape = np.asarray(ra_rad).shape
    c  = SkyCoord(
        ra=np.degrees(ra_rad).ravel() * u.deg,
        dec=np.degrees(dec_rad).ravel() * u.deg,
        frame="icrs",
    )
    gc = SkyCoord(ra=266.4051 * u.deg, dec=-28.9362 * u.deg, frame="icrs")
    return c.separation(gc).rad.reshape(shape)


def psi_from_radec(ra_rad: float, dec_rad: float) -> float:
    """
    Angle from the Galactic Centre for a given equatorial position.

    Uses the IAU standard Galactic pole and zero-longitude direction
    (J2000.0): pole at RA=192.86°, Dec=27.13°; GC at l=0, b=0
    corresponds to RA=266.40°, Dec=−28.94°.

    Parameters
    ----------
    ra_rad : float
        Right ascension [radians].
    dec_rad : float
        Declination [radians].

    Returns
    -------
    float
        Angle from the Galactic Centre [radians].
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra=np.degrees(ra_rad) * u.deg, dec=np.degrees(dec_rad) * u.deg, frame="icrs")
    gc = SkyCoord(ra=266.4051 * u.deg, dec=-28.9362 * u.deg, frame="icrs")
    return c.separation(gc).rad
