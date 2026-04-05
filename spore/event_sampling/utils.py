import numpy as np

from ..conventions import SkyCoordinate, EarthCoordinate, sample_cone
from ..detector import Detector


def zenith_grid(
    decs: np.ndarray,
    ras: np.ndarray,
    earth_coord: EarthCoordinate,
    t: float,
) -> np.ndarray:
    """
    Compute the zenith angle at each (dec, RA) grid point for a single
    observation time using a vectorised astropy coordinate transform.

    Parameters
    ----------
    decs : (n_dec,)   Declinations in radians.
    ras  : (n_ra,)    Right ascensions in radians.
    earth_coord : EarthCoordinate
    t    : float      Observation epoch in MJD.

    Returns
    -------
    zeniths : ndarray, shape (n_dec, n_ra)   Zenith angle in radians.
    """
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy import units as u

    dec_grid, ra_grid = np.meshgrid(decs, ras, indexing='ij')
    sc = SkyCoord(ra=ra_grid.ravel() * u.rad, dec=dec_grid.ravel() * u.rad)
    ec = EarthLocation.from_geodetic(
        lon=earth_coord.longitude * u.rad,
        lat=earth_coord.latitude * u.rad,
    )
    altaz = sc.transform_to(AltAz(location=ec, obstime=Time(t, format='mjd')))
    return (np.pi / 2 - np.radians(altaz.alt.deg)).reshape(len(decs), len(ras))

def new_sample(bounds) -> np.ndarray:
    """Draw a uniform random sample from a rectangular parameter space.

    Args:
        bounds: Sequence of (lower, upper) pairs defining the bounds of each
            dimension.

    Returns:
        Array of length len(bounds) with each element drawn uniformly from
        its respective interval.
    """
    return np.array([np.random.uniform(lb, ub) for lb, ub in bounds])

def smear_truth(
    true_direction: SkyCoordinate,
    true_energy: float,
    detector: Detector,
    morphology: str
):
    """
    Apply detector smearing to a true (direction, energy) and return the
    reconstructed quantities plus an angular error estimate.

    If the detector response includes a joint smearing IRF (loaded via
    DetectorResponse.from_dataverse), the reconstruction is drawn from the
    full conditional distribution P(E_reco, PSF, AngErr | E_true, dec).
    Otherwise, energy and angular smearing are sampled independently from the
    1-D marginal inverse-CDFs stored in the HDF5 response file, and ang_err
    is set to 0.

    Returns
    -------
    reco_direction : SkyCoordinate
    reco_energy    : float  (internal energy units, eV)
    ang_err        : float  (radians; 0 when joint smearing is unavailable)
    """
    if morphology not in ("track", "cascade"):
        raise ValueError(f"Invalid morphology '{morphology}': must be 'track' or 'cascade'.")
    if morphology not in detector.response.available_morphologies:
        raise ValueError(
            f"Morphology '{morphology}' is not available for this detector. "
            f"Available: {detector.response.available_morphologies}."
        )

    js = detector.response.joint_smearing
    if js is not None and morphology in js:
        reco_energy, psi, ang_err = js[morphology](
            true_energy, true_direction.declination
        )
    else:
        ang_sampler = detector.response.angular_response[morphology]
        psi         = ang_sampler(true_energy, np.random.rand())
        # Angular error estimate: sigma_R from the Rayleigh sampler.
        # For the power-law Rayleigh model, sigma_R = psi / sqrt(-2 ln(1-u)),
        # but we approximate it as the median PSF evaluated at true_energy by
        # re-calling the sampler at u=0.5 (the median quantile).
        ang_err = ang_sampler(true_energy, 0.5)
        if morphology in detector.response.energy_response:
            e_sampler   = detector.response.energy_response[morphology]
            reco_energy = np.exp(e_sampler(np.random.rand())) * true_energy
        else:
            reco_energy = true_energy

    reco_direction = sample_cone(true_direction, psi)
    return reco_direction, reco_energy, ang_err
