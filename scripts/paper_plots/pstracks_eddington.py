"""
Compute and cache Eddington-bias data for the IceCube 10-year tracks sample.

Mirrors pstracks_roundtrip.py but additionally records:
  - Unsmeared spectra (E_reco = E_true, delta_clip=(0,0)) for each component.
  - Per-model Eddington correction factors (smeared_reco / unsmeared_true).

The figure this feeds shows the naive A_eff × flux prediction alongside the
smearing-corrected prediction and the observed data, making the energy
redistribution effect visible.

Plotting is handled by paper/plots.py under the group ``figure_5_eddington``.
"""
import os
import numpy as np
import h5py as h5

from scipy.special import gammaln
from scipy.optimize import minimize

from spore.conventions import ureg
from spore.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import ExtendedSourceEventSampler

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
DATA    = os.path.join(HERE, "ps10yr_data_release")
ATM_H5  = os.path.join(REPO, "resources", "atmo_flux_models.h5")
AST_H5  = os.path.join(REPO, "resources", "ps10yr_combined_flux.h5")
OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")

IC86_EVENTS = {
    'IC86_I':   'IC86_I_exp.csv',   'IC86_II':  'IC86_II_exp.csv',
    'IC86_III': 'IC86_III_exp.csv', 'IC86_IV':  'IC86_IV_exp.csv',
    'IC86_V':   'IC86_V_exp.csv',   'IC86_VI':  'IC86_VI_exp.csv',
    'IC86_VII': 'IC86_VII_exp-1.csv',
}
IC86_SEASONS = ['IC86_I', 'IC86_II', 'IC86_III', 'IC86_IV', 'IC86_V', 'IC86_VI', 'IC86_VII']

ATM_FLUX_MODELS = "honda2006 mceq_gsf_sibyll23d mceq_h3a_sibyll23d mceq_h4a_sibyll23d".split()

N_PSEUDO       = 100   # pseudo-experiments for Eddington correction estimate
N_SAMPLES      = 40    # pseudo-experiments for averaged spectra
MORPHOLOGY     = "track"


def poisson_llh(n, mu):
    mu_safe = np.where(mu > 0, mu, 1e-300)
    return float(np.sum(n * np.log(mu_safe) - mu - gammaln(n + 1)))


def _reco_energies_north(events):
    """Reconstructed energies [GeV] for northern-sky (sin δ > 0) events."""
    decs  = np.array([e.reco_direction.declination     for e in events])
    e_gev = np.array([e.reco_energy.to("GeV").magnitude for e in events])
    return e_gev[decs > 0]


def _reco_energies_north_unsmeared(events):
    """True energies [GeV] for northern-sky events (unsmeared proxy)."""
    decs  = np.array([e.reco_direction.declination     for e in events])
    e_gev = np.array([e.true_energy.to("GeV").magnitude for e in events])
    return e_gev[decs > 0]


if __name__ == "__main__":
    det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0, "depth": 1945, "medium": "Ice"},
        "response": {"detector_response_toml": os.path.join(REPO, "resources", "configs", "ps10yr_response.toml")},
    })

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    all_ev = np.vstack([
        np.genfromtxt(os.path.join(DATA, "events", f), comments='#')
        for f in IC86_EVENTS.values()
    ])
    log10e = all_ev[:, 1]
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

    # ------------------------------------------------------------------
    # Step 1: fit atmospheric normalizations (same as pstracks_roundtrip)
    # ------------------------------------------------------------------
    atmo_srcs = [
        ExtendedSource.from_config({"flux": {"location": f"{ATM_H5}:{m}"}})
        for m in ATM_FLUX_MODELS
    ]
    astro_src = ExtendedSource.from_config({"flux": {"location": f"{AST_H5}:astrophysical"}})

    atmo_samplers = [
        ExtendedSourceEventSampler(det, src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
        for src in atmo_srcs
    ]
    astro_sampler = ExtendedSourceEventSampler(
        det, astro_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100
    )

    print("Fitting atmospheric normalizations...")
    norms = []
    for sampler in atmo_samplers:
        h_atmo = np.zeros(e_cents.shape)
        for _ in range(N_SAMPLES):
            evs = sampler.sample_events(MORPHOLOGY, deltat=T_OBS)
            h_atmo += np.histogram(_reco_energies_north(evs), bins=e_bins)[0]
        h_atmo /= N_SAMPLES
        ref = abs(poisson_llh(h_data, h_atmo))
        res = minimize(lambda x: -poisson_llh(h_data, x * h_atmo) / ref, 1.0, method='L-BFGS-B')
        norms.append(float(res.x[0]))
        print(f"  {ATM_FLUX_MODELS[len(norms)-1]}: norm = {norms[-1]:.3f}")

    # Rebuild samplers with fitted normalizations
    atmo_srcs_fit = [
        ExtendedSource.from_config({"flux": {"location": f"{ATM_H5}:{m}"}}, normalization=n)
        for m, n in zip(ATM_FLUX_MODELS, norms)
    ]
    atmo_samplers_fit = [
        ExtendedSourceEventSampler(det, src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
        for src in atmo_srcs_fit
    ]

    # ------------------------------------------------------------------
    # Step 2: compute Eddington correction factors per atmospheric model
    # ------------------------------------------------------------------
    print(f"\nComputing Eddington corrections ({N_PSEUDO} pseudo-experiments each)...")
    corrections = np.zeros((len(ATM_FLUX_MODELS), len(e_cents)))
    for idx, sampler in enumerate(atmo_samplers_fit):
        print(f"  {ATM_FLUX_MODELS[idx]}...")
        corrections[idx] = sampler.eddington_correction(MORPHOLOGY, e_bins, n_pseudo=N_PSEUDO)

    print("  astrophysical...")
    correction_astro = astro_sampler.eddington_correction(MORPHOLOGY, e_bins, n_pseudo=N_PSEUDO)

    # ------------------------------------------------------------------
    # Step 3: accumulate smeared and unsmeared spectra
    # ------------------------------------------------------------------
    print(f"\nAccumulating spectra ({N_SAMPLES} samples each)...")
    h_atmos_smeared   = np.zeros((len(e_cents), len(ATM_FLUX_MODELS)))
    h_atmos_unsmeared = np.zeros((len(e_cents), len(ATM_FLUX_MODELS)))
    h_astro_smeared   = np.zeros(len(e_cents))
    h_astro_unsmeared = np.zeros(len(e_cents))

    for idx, sampler in enumerate(atmo_samplers_fit):
        for _ in range(N_SAMPLES):
            evs = sampler.sample_events(MORPHOLOGY, deltat=T_OBS)
            h_atmos_smeared[:, idx]   += np.histogram(_reco_energies_north(evs),           bins=e_bins)[0]
            h_atmos_unsmeared[:, idx] += np.histogram(_reco_energies_north_unsmeared(evs), bins=e_bins)[0]
    h_atmos_smeared   /= N_SAMPLES
    h_atmos_unsmeared /= N_SAMPLES

    for _ in range(N_SAMPLES):
        evs = astro_sampler.sample_events(MORPHOLOGY, deltat=T_OBS)
        h_astro_smeared   += np.histogram(_reco_energies_north(evs),           bins=e_bins)[0]
        h_astro_unsmeared += np.histogram(_reco_energies_north_unsmeared(evs), bins=e_bins)[0]
    h_astro_smeared   /= N_SAMPLES
    h_astro_unsmeared /= N_SAMPLES

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w"):
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "figure_5_eddington" in h5f:
            del h5f["figure_5_eddington"]
        gp = h5f.create_group("figure_5_eddington")
        gp["e_cents"]           = e_cents
        gp["e_bins"]            = e_bins
        gp["h_data"]            = h_data
        gp["h_atmos_smeared"]   = h_atmos_smeared    # (n_bins, n_models)
        gp["h_atmos_unsmeared"] = h_atmos_unsmeared  # (n_bins, n_models)
        gp["h_astro_smeared"]   = h_astro_smeared
        gp["h_astro_unsmeared"] = h_astro_unsmeared
        gp["corrections_atmo"]  = corrections         # (n_models, n_bins)
        gp["correction_astro"]  = correction_astro
        gp.attrs["atm_flux_models"] = ATM_FLUX_MODELS

    print(f"\nWrote figure_5_eddington to {OUTFILE}")
