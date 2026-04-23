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

plt.style.use(STYLE)

IC86_EVENTS = {
    'IC86_I':   'IC86_I_exp.csv',   'IC86_II':  'IC86_II_exp.csv',
    'IC86_III': 'IC86_III_exp.csv', 'IC86_IV':  'IC86_IV_exp.csv',
    'IC86_V':   'IC86_V_exp.csv',   'IC86_VI':  'IC86_VI_exp.csv',
    'IC86_VII': 'IC86_VII_exp-1.csv',
}
IC86_SEASONS = ['IC86_I', 'IC86_II', 'IC86_III', 'IC86_IV', 'IC86_V', 'IC86_VI', 'IC86_VII']


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
    det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0, "depth": 1945, "medium": "Ice"},
        "response": {"detector_response_file": os.path.join(REPO, "resources", "configs", "ps10yr_detector_response.h5")},
    })

    atmo_flux_models = "honda2006 mceq_gsf_sibyll23d mceq_h3a_sibyll23d mceq_h4a_sibyll23d".split()

    atmo_srcs = [ExtendedSource.from_config({
        "flux": {"location": f"{ATM_H5}:{flux_model}"}
    }) for flux_model in atmo_flux_models]

    astro_src = ExtendedSource.from_config({
        "flux": {"location": f"{AST_H5}:astrophysical"}
    })

    all_ev = np.vstack([
        np.genfromtxt(os.path.join(DATA, "events", f), comments='#')
        for f in IC86_EVENTS.values()
    ])

    log10e = all_ev[:, 1]           # log10(E_reco / GeV)
    sindec = np.sin(np.radians(all_ev[:, 4]))

    total_days = 0
    for season in IC86_SEASONS:
        fname = f'{season}_exp.csv'
        ut = np.genfromtxt(os.path.join(DATA, 'uptime', fname), comments='#')
        total_days += (ut[:, 1] - ut[:, 0]).sum()

    T_OBS = ureg.Quantity(total_days, "day")
    print(f"IC86 livetime: {total_days:.1f} days")

    e_bins  = np.logspace(1, 8, 36)
    e_cents = (e_bins[1:] + e_bins[:-1]) / 2

    h_data, _ = np.histogram(np.power(10, log10e)[sindec > 0], bins=e_bins)

    N = 40
    norms = []
    atmo_samplers = [
        SourceSampler(det, atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
        for atmo_src in atmo_srcs
    ]
    for atmo_sampler in atmo_samplers:
        h_atmo = np.zeros(e_cents.shape)
        for _ in range(N):
            atmo_events = atmo_sampler.sample_events("track", deltat=T_OBS)
            atmo_reco_e_gev = np.array([e.reco_energy.to('GeV').magnitude for e in atmo_events])
            atmo_reco_decs  = np.array([e.reco_direction.declination for e in atmo_events])
            _h_atmo, _ = np.histogram(atmo_reco_e_gev[atmo_reco_decs > 0], bins=e_bins)
            h_atmo += _h_atmo
        h_atmo /= N

        h = abs(poisson_llh(h_data, h_atmo))
        f = lambda x: -poisson_llh(h_data, x * h_atmo) / h
        res = minimize(f, 1.0, method='L-BFGS-B')
        norms.append(float(res.x[0]))

    atmo_srcs = [ExtendedSource.from_config(
        {"flux": {"location": f"{ATM_H5}:{flux_model}"}},
        normalization=norm
    ) for (flux_model, norm) in zip(atmo_flux_models, norms)]

    atmo_samplers = [
        SourceSampler(det, atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
        for atmo_src in atmo_srcs
    ]
    astro_sampler = SourceSampler(det, astro_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)

    h_atmos = np.zeros(e_cents.shape + (4,))

    for (idx, sampler) in enumerate(atmo_samplers):
        h_atmo_sum = np.zeros(e_cents.shape)
        for _ in range(N):
            atmo_events     = sampler.sample_events("track", deltat=T_OBS)
            atmo_reco_e_gev = np.array([e.reco_energy.to('GeV').magnitude for e in atmo_events])
            atmo_reco_decs  = np.array([e.reco_direction.declination for e in atmo_events])
            _h_atmo, _ = np.histogram(atmo_reco_e_gev[atmo_reco_decs > 0], bins=e_bins)
            h_atmo_sum += _h_atmo
        h_atmos[:, idx] = h_atmo_sum / N

    h_astro_sum = np.zeros(e_cents.shape)
    for _ in range(N):
        astro_events     = astro_sampler.sample_events("track", deltat=T_OBS)
        astro_reco_decs  = np.array([e.reco_direction.declination     for e in astro_events])
        astro_reco_e_gev = np.array([e.reco_energy.to('GeV').magnitude for e in astro_events])
        _h_astro, _ = np.histogram(astro_reco_e_gev[astro_reco_decs > 0], bins=e_bins)
        h_astro_sum += _h_astro
    h_astro = h_astro_sum / N

    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w") as h5f:
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "figure_5" in h5f.keys():
            del h5f["figure_5"]
        gp = h5f.create_group("figure_5")
        gp["e_cents"] = e_cents
        gp["h_atmos"] = h_atmos
        gp["h_astro"] = h_astro
        gp["h_data"]  = h_data
