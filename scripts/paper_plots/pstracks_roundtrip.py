import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from matplotlib.colors import to_rgb
from matplotlib import colormaps
from scipy.special import gammaln
from scipy.optimize import minimize

from spore.conventions import SkyCoordinate, ureg
from spore.detector.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import SourceSampler

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
DATA    = os.path.join(REPO, "resources", "data_releases", "ps10yr_data_release")
ATM_H5  = os.path.join(REPO, "resources", "atmo_flux_models.h5")
AST_H5  = os.path.join(REPO, "resources", "ps10yr_combined_flux.h5")
STYLE   = os.path.join(REPO, "resources", "paper.mplstyle")
OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")

IC86_EVENTS = {
    'IC86_I':   'IC86_I_exp.csv',   'IC86_II':  'IC86_II_exp.csv',
    'IC86_III': 'IC86_III_exp.csv', 'IC86_IV':  'IC86_IV_exp.csv',
    'IC86_V':   'IC86_V_exp.csv',   'IC86_VI':  'IC86_VI_exp.csv',
    'IC86_VII': 'IC86_VII_exp-1.csv',
}
IC86_SEASONS = ['IC86_I', 'IC86_II', 'IC86_III', 'IC86_IV', 'IC86_V', 'IC86_VI', 'IC86_VII']

# Base seed for the pseudo-experiments; each iteration gets SEED + offset + pe.
SEED           = 0
ATMO_FIT_OFF   = 0
ATMO_SEED_OFF  = 100_000
ASTRO_SEED_OFF = 200_000


def _reco_sindecs(events):
    """sin(reco declination) for a list of events, as a float array."""
    return np.sin(np.fromiter(
        (e.reco_direction.declination for e in events), float, len(events)
    ))


def _reco_energies(events):
    """Reconstructed energies [GeV] for a list of events, as a float array.

    Reads the cached scalar rather than the ``reco_energy`` property: building
    one pint Quantity per event costs ~30 s for a full IC86 sample.
    """
    return np.fromiter((e._reco_e for e in events), float, len(events))


def poisson_llh(n, mu):
    """
    Poisson log-likelihood  ln L = sum [n ln mu - mu - ln(n!)].

    Parameters
    ----------
    n  : ndarray  observed counts (need not be integer for sampled pseudo-data)
    mu : ndarray  expected counts (must be > 0 where n > 0)

    Returns
    -------
    float
    """
    mu_safe = np.where(mu > 0, mu, 1e-300)
    return float(np.sum(n * np.log(mu_safe) - mu - gammaln(n + 1)))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    plt.style.use(STYLE)

    det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0, "depth": 1945, "medium": "Ice"},
        "response": {"detector_response_file": os.path.join(REPO, "resources", "configs", "ps10yr_detector_response.h5")},
    })

    atmo_flux_model = "mceq_gsf_sibyll23d"

    atmo_src = ExtendedSource.from_config({
        "flux": {"location": f"{ATM_H5}:{atmo_flux_model}"}
    })

    astro_src = ExtendedSource.from_config({
        "flux": {"location": f"{AST_H5}:astrophysical"}
    })

    all_ev = np.vstack([
        np.genfromtxt(os.path.join(DATA, "events", f), comments='#')
        for f in IC86_EVENTS.values()
    ])

    sindec = np.sin(np.radians(all_ev[:, 4]))

    total_days = 0
    for season in IC86_SEASONS:
        fname = f'{season}_exp.csv'
        ut = np.genfromtxt(os.path.join(DATA, 'uptime', fname), comments='#')
        total_days += (ut[:, 1] - ut[:, 0]).sum()

    T_OBS = ureg.Quantity(total_days, "day")
    logging.info("IC86 livetime: %.1f days", total_days)

    sd_bins  = np.linspace(0, 1, 21)
    sd_cents = (sd_bins[1:] + sd_bins[:-1]) / 2

    # Energy proxy binning for the second panel; the data release stores
    # log10(E/GeV) in column 1.
    e_bins  = np.logspace(2, 6, 33)
    e_cents = np.sqrt(e_bins[1:] * e_bins[:-1])

    north = sindec > 0
    h_data,   _ = np.histogram(sindec[north], bins=sd_bins)
    h_data_e, _ = np.histogram(10 ** all_ev[north, 1], bins=e_bins)

    N = 40
    logging.info("Building initial atmo sampler …")
    atmo_sampler = SourceSampler(det, atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)

    logging.info("Fitting atmo normalisation (%d pseudo-experiments) …", N)
    h_atmo = np.zeros(sd_cents.shape)
    for pe in range(N):
        if pe % 10 == 0:
            logging.info("  pseudo-exp %d/%d", pe, N)
        atmo_events = atmo_sampler.sample_events(
            "track", deltat=T_OBS, seed=SEED + ATMO_FIT_OFF + pe
        )
        atmo_reco_sindecs = _reco_sindecs(atmo_events)
        _h_atmo, _ = np.histogram(atmo_reco_sindecs[atmo_reco_sindecs > 0], bins=sd_bins)
        h_atmo += _h_atmo
    h_atmo /= N

    h = abs(poisson_llh(h_data, h_atmo))
    f = lambda x: -poisson_llh(h_data, x * h_atmo) / h
    res = minimize(f, 1.0, method='L-BFGS-B')
    norm = float(res.x[0])
    logging.info("  norm = %.4f", norm)

    atmo_src = ExtendedSource.from_config(
        {"flux": {"location": f"{ATM_H5}:{atmo_flux_model}"}},
        normalization=norm,
    )

    logging.info("Rebuilding atmo sampler with fitted norm …")
    atmo_sampler = SourceSampler(det, atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
    logging.info("Building astro sampler …")
    astro_sampler = SourceSampler(det, astro_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)

    logging.info("Sampling atmo (%d pseudo-experiments) …", N)
    h_atmo_sum   = np.zeros(sd_cents.shape)
    h_atmo_e_sum = np.zeros(e_cents.shape)
    for pe in range(N):
        if pe % 10 == 0:
            logging.info("  pseudo-exp %d/%d", pe, N)
        atmo_events       = atmo_sampler.sample_events(
            "track", deltat=T_OBS, seed=SEED + ATMO_SEED_OFF + pe
        )
        atmo_reco_sindecs = _reco_sindecs(atmo_events)
        up = atmo_reco_sindecs > 0
        _h_atmo, _ = np.histogram(atmo_reco_sindecs[up], bins=sd_bins)
        h_atmo_sum += _h_atmo
        # Same northern-sky selection as the declination panel.
        _h_atmo_e, _ = np.histogram(_reco_energies(atmo_events)[up], bins=e_bins)
        h_atmo_e_sum += _h_atmo_e
    h_atmo   = h_atmo_sum / N
    h_atmo_e = h_atmo_e_sum / N

    logging.info("Sampling astro (%d pseudo-experiments) …", N)
    h_astro_sum   = np.zeros(sd_cents.shape)
    h_astro_e_sum = np.zeros(e_cents.shape)
    for pe in range(N):
        if pe % 10 == 0:
            logging.info("  pseudo-exp %d/%d", pe, N)
        astro_events       = astro_sampler.sample_events(
            "track", deltat=T_OBS, seed=SEED + ASTRO_SEED_OFF + pe
        )
        astro_reco_sindecs = _reco_sindecs(astro_events)
        up = astro_reco_sindecs > 0
        _h_astro, _ = np.histogram(astro_reco_sindecs[up], bins=sd_bins)
        h_astro_sum += _h_astro
        _h_astro_e, _ = np.histogram(_reco_energies(astro_events)[up], bins=e_bins)
        h_astro_e_sum += _h_astro_e
    h_astro   = h_astro_sum / N
    h_astro_e = h_astro_e_sum / N

    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w") as h5f:
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "figure_5" in h5f.keys():
            del h5f["figure_5"]
        gp = h5f.create_group("figure_5")
        gp["sd_cents"]  = sd_cents
        gp["h_atmo"]    = h_atmo
        gp["h_astro"]   = h_astro
        gp["h_data"]    = h_data
        gp["e_cents"]   = e_cents
        gp["h_atmo_e"]  = h_atmo_e
        gp["h_astro_e"] = h_astro_e
        gp["h_data_e"]  = h_data_e
