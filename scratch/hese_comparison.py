import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from matplotlib.colors import to_rgb
from matplotlib import colormaps
from tqdm import tqdm
from scipy.special import gammaln

from spore.conventions import EarthCoordinate, ureg
from spore.detector import Detector
from spore.detector.detector_response import DetectorResponse
from spore.detector.detector import Medium
from spore.source import ExtendedSource
from spore.source.flux import Flux
from spore.physics import Morphology
from spore.event_sampling import ExtendedSourceEventSampler

Morphology.register('astro_cascade')
Morphology.register('astro_doublebang')
Morphology.register('astro_track')
Morphology.register('atmo_cascade')
Morphology.register('atmo_track')

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
STYLE   = os.path.join(REPO, "resources", "paper.mplstyle")

plt.style.use(STYLE)

COLUMN_WIDTH = 6.09263285024
colors = colormaps["tab20b"](np.linspace(0, 1, 20))

IRF_PATH        = os.path.join(REPO, "resources", "hese_7yr_detector_response.h5")
ASTRO_FLUX_PATH = os.path.join(REPO, "resources", "hese_7yr_astro_flux.h5")
ATMO_FLUX_PATH  = os.path.join(REPO, "resources", "hese_7yr_atmo_flux.h5")
LIVETIME        = ureg.Quantity(2635, "day")
LIVETIME_SEC    = LIVETIME.to("s").magnitude
_ICECUBE        = EarthCoordinate(np.radians(-90.0), 0.0)

_DATA_RELEASE = os.path.join(HERE, "hese_7yr_data_release")
_DATA_DIR     = os.path.join(_DATA_RELEASE, "resources", "data")


def load_hese_mc(data_dir=_DATA_DIR, min_e=0):
    """
    Load HESE 7.5-year MC and return a dict of arrays, one entry per neutrino event.

    Muon background events (interactionType == 0) are excluded.

    Returns
    -------
    dict with keys:
      primary_energy   : GeV
      primary_zenith   : rad
      primary_type     : PDG code  (+-12 NuE, +-14 NuMu, +-16 NuTau)
      reco_energy      : GeV  (deposited)
      reco_zenith      : rad
      reco_morphology  : 0=cascade, 1=track, 2=doublebang
      w                : weightOverFluxOverLivetime  [GeV sr cm^2]
      pion_flux        : Honda pion flux at truth (E, zen)
      kaon_flux        : Honda kaon flux at truth (E, zen)
      prompt_flux      : prompt flux at truth (E, zen)
      veto_conv        : conventionalSelfVetoCorrection
      veto_prompt      : promptSelfVetoCorrection
    """
    merged = {}
    for fname in ["HESE_mc_truth.json", "HESE_mc_observable.json", "HESE_mc_flux.json"]:
        merged.update(json.load(open(os.path.join(data_dir, fname))))

    is_nu = np.array(merged["interactionType"]) != 0
    has_e = np.array(merged["recoDepositedEnergy"]) >= min_e

    mask = np.logical_and(is_nu, has_e)
    return dict(
        primary_energy  = np.array(merged["primaryEnergy"])[mask],
        primary_zenith  = np.array(merged["primaryZenith"])[mask],
        primary_type    = np.array(merged["primaryType"])[mask],
        reco_energy     = np.array(merged["recoDepositedEnergy"])[mask],
        reco_zenith     = np.array(merged["recoZenith"])[mask],
        reco_morphology = np.array(merged["recoMorphology"])[mask],
        w               = np.array(merged["weightOverFluxOverLivetime"])[mask],
        pion_flux       = np.array(merged["pionFlux"])[mask],
        kaon_flux       = np.array(merged["kaonFlux"])[mask],
        prompt_flux     = np.array(merged["promptFlux"])[mask],
        veto_conv       = np.array(merged["conventionalSelfVetoCorrection"])[mask],
        veto_prompt     = np.array(merged["promptSelfVetoCorrection"])[mask],
    )


def astro_weights(mc, norm=6.37e-18, gamma=2.87, pivot_gev=1e5,
                  livetime_sec=LIVETIME_SEC):
    """
    Expected events per MC event under a single power-law astrophysical flux.

    N = livetime x sum_i  w_i x (norm/6) x (E_i / pivot)^{-gamma}

    Parameters
    ----------
    norm  : float
      Six-neutrino total normalization in GeV^{-1} cm^{-2} s^{-1} sr^{-1}.
      Default is the HESE 7.5yr best-fit 6.37e-18.
    gamma : float
      Spectral index.  Default 2.87.
    """
    phi = (norm / 6.0) * (mc["primary_energy"] / pivot_gev) ** (-gamma)
    return livetime_sec * mc["w"] * phi


def atmo_conv_weights(mc, conv_norm=1.0, kpi_ratio=1.00013744,
                      cr_delta_gamma=-0.05309302, nunubar_ratio=0.99815326,
                      cr_pivot_gev=2020.0, livetime_sec=LIVETIME_SEC):
    """
    Expected events per MC event under the conventional atmospheric flux,
    matching the data release parametrisation.

    N = livetime x sum_i  w_i x veto_i
          x conv_norm x (E_i / cr_pivot)^{-cr_delta_gamma}
          x (pion_i + kpi_ratio x kaon_i)
          x nunubar_weight_i

    Parameters
    ----------
    conv_norm       : overall conventional normalisation (1.0 = reference)
    kpi_ratio       : kaon-to-pion ratio scale factor   (best-fit ~1.00014)
    cr_delta_gamma  : CR spectral tilt index             (best-fit ~-0.053)
    nunubar_ratio   : nu/nubar ratio parameter           (best-fit ~0.998)
                    neutrinos get this weight, antineutrinos get (2 - nunubar_ratio)
    cr_pivot_gev    : pivot energy for CR tilt [GeV]     (2020 GeV)
    """
    tilt      = (mc["primary_energy"] / cr_pivot_gev) ** (-cr_delta_gamma)
    honda     = mc["pion_flux"] + kpi_ratio * mc["kaon_flux"]
    nu_weight = np.where(mc["primary_type"] > 0, nunubar_ratio, 2.0 - nunubar_ratio)
    return livetime_sec * mc["w"] * mc["veto_conv"] * conv_norm * tilt * honda * nu_weight


def atmo_prompt_weights(mc, prompt_norm=0.0, livetime_sec=LIVETIME_SEC):
    """
    Expected events per MC event under the prompt atmospheric flux.

    N = livetime x sum_i  w_i x veto_prompt_i x prompt_norm x prompt_flux_i

    Parameters
    ----------
    prompt_norm : float
        Prompt flux normalization (0.0 = best-fit, consistent with zero).
    """
    return livetime_sec * mc["w"] * mc["veto_prompt"] * prompt_norm * mc["prompt_flux"]


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


def min_width_interval(x, fraction):
    """Return (lo, hi) of the shortest interval containing `fraction` of x."""
    x = np.sort(x)
    n = len(x)
    k = int(np.ceil(fraction * n))
    widths = x[k - 1:] - x[:n - k + 1]
    i = np.argmin(widths)
    return x[i], x[i + k - 1]


if __name__ == "__main__":
    full_response = DetectorResponse.from_config({"detector_response_file": IRF_PATH})

    def _filter(response, prefix):
        return DetectorResponse(
            effective_area={k: v for k, v in response.effective_area.items() if k.startswith(prefix)},
            angular_response={k: v for k, v in response.angular_response.items() if k.startswith(prefix)},
            energy_response={k: v for k, v in response.energy_response.items() if k.startswith(prefix)},
        )

    astro_det = Detector(_ICECUBE, Medium.Ice, _filter(full_response, "astro"))
    atmo_det  = Detector(_ICECUBE, Medium.Ice, _filter(full_response, "atmo"))

    astro_flux = Flux.from_config({"location": f"{ASTRO_FLUX_PATH}:astrophysical"})
    atmo_flux  = Flux.from_config({"location": f"{ATMO_FLUX_PATH}:conventional"})

    astro_sampler = ExtendedSourceEventSampler(
        astro_det, ExtendedSource(astro_flux), n_time_samples=100, n_e=100, n_ra=100, n_dec=100
    )
    atmo_sampler = ExtendedSourceEventSampler(
        atmo_det, ExtendedSource(atmo_flux), n_time_samples=100, n_e=100, n_ra=100, n_dec=100
    )

    mc = load_hese_mc()

    sys.path.insert(0, _DATA_RELEASE)
    import data_loader
    data = data_loader.load_data(
        os.path.join(_DATA_RELEASE, "resources", "data", "HESE_data.json"),
        emin=0.0,
    )

    es    = np.logspace(4, 7, 28)
    cents = (es[1:] + es[:-1]) / 2
    seed  = 170
    events = (
        astro_sampler.sample_events("astro_track",      deltat=LIVETIME, seed=seed, delta_clip=(-1.5, 3)) +
        astro_sampler.sample_events("astro_cascade",    deltat=LIVETIME, seed=seed, delta_clip=(-1.5, 3)) +
        astro_sampler.sample_events("astro_doublebang", deltat=LIVETIME, seed=seed, delta_clip=(-1.5, 3)) +
        atmo_sampler.sample_events("atmo_track",        deltat=LIVETIME, seed=seed, delta_clip=(-1.5, 3)) +
        atmo_sampler.sample_events("atmo_cascade",      deltat=LIVETIME, seed=seed, delta_clip=(-1.5, 3))
    )
    reco_es = np.array([ev.reco_energy.to("GeV").magnitude for ev in events])

    h_astro,  _ = np.histogram(mc["reco_energy"], weights=astro_weights(mc),      bins=es)
    h_atmo,   _ = np.histogram(mc["reco_energy"], weights=atmo_conv_weights(mc),  bins=es)
    h_sample, _ = np.histogram(reco_es, bins=es)
    h_data,   _ = np.histogram(data["recoDepositedEnergy"], bins=es)

    mc_mask = mc["reco_energy"] > 6e4
    h_astro_cut, _ = np.histogram(mc["reco_energy"][mc_mask], weights=astro_weights(mc)[mc_mask],     bins=es)
    h_atmo_cut,  _ = np.histogram(mc["reco_energy"][mc_mask], weights=atmo_conv_weights(mc)[mc_mask], bins=es)

    llh0 = poisson_llh(h_astro_cut + h_atmo_cut, h_astro_cut + h_atmo_cut)

    delta_llhs = []
    for _ in tqdm(range(10_00)):
        events_ = (
            astro_sampler.sample_events("astro_track",      deltat=LIVETIME) +
            astro_sampler.sample_events("astro_cascade",    deltat=LIVETIME) +
            astro_sampler.sample_events("astro_doublebang", deltat=LIVETIME) +
            atmo_sampler.sample_events("atmo_track",        deltat=LIVETIME) +
            atmo_sampler.sample_events("atmo_cascade",      deltat=LIVETIME)
        )
        reco_es_ = np.array([ev.reco_energy.to("GeV").magnitude for ev in events_])
        mask_ = reco_es_ > 6e4
        h_sample_cut_, _ = np.histogram(reco_es_[mask_], bins=es)
        delta_llhs.append(poisson_llh(h_sample_cut_, h_astro_cut + h_atmo_cut) - llh0)

    delta_llhs = np.array(delta_llhs)

    sample_mask = reco_es > 6e4
    h_sample_cut, _ = np.histogram(reco_es[sample_mask], bins=es)
    delta_llh_sample = poisson_llh(h_sample_cut, h_astro_cut + h_atmo_cut) - llh0

    data_mask = data["recoDepositedEnergy"] > 6e4
    h_data_cut, _ = np.histogram(data["recoDepositedEnergy"][data_mask], bins=es)
    delta_llh_data = poisson_llh(h_data_cut, h_astro_cut + h_atmo_cut) - llh0

    # --- Figure: energy spectrum (left) + LLH distribution (right) ---
    fig, axs = plt.subplots(1, 2, figsize=(1.8 * COLUMN_WIDTH, 1.8 * 9 / 16 / 2 * COLUMN_WIDTH))

    ax = axs[0]
    ax.fill_between(
        cents, h_atmo, h_atmo + h_astro,
        step="mid", label="Astrophysical",
        facecolor=to_rgb(colors[-1]) + (0.5,),
        edgecolor=to_rgb(colors[-1]) + (1.0,),
    )
    ax.fill_between(
        cents, 0, h_atmo,
        step="mid", label="Atmospheric",
        facecolor=to_rgb("lightskyblue") + (0.5,),
        edgecolor=to_rgb("lightskyblue") + (1.0,),
    )
    ax.errorbar(
        cents / 1.03, h_sample,
        marker="o", linestyle="none", yerr=np.sqrt(h_sample),
        label="Sampled", color=colors[14],
    )
    ax.errorbar(
        cents * 1.03, h_data,
        marker="o", linestyle="none", yerr=np.sqrt(h_data),
        label="HESE (2021)", color=colors[1],
    )
    ax.fill_between(
        [1, 6e4], [1e-6, 1e-6], [1e6, 1e6],
        facecolor=to_rgb("lightgrey") + (0.5,),
        edgecolor="lightgrey",
    )
    ax.axvline(6e4, color="lightgrey")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 1e7)
    ax.set_ylim(0.1, 40)
    ax.set_xlabel(r"$E_{\mathrm{dep.}}~\left[\mathrm{GeV}\right]$")
    ax.set_ylabel(r"$N_{\mathrm{event}}$")
    ax.legend()

    ax = axs[1]
    h_hist, bins_hist = np.histogram(-2 * delta_llhs, bins=20)
    q = (bins_hist[1:] + bins_hist[:-1]) / 2

    lo1, hi1 = min_width_interval(-2 * delta_llhs, 0.68)
    lo2, hi2 = min_width_interval(-2 * delta_llhs, 0.95)
    lo3, hi3 = min_width_interval(-2 * delta_llhs, 0.997)

    xs1 = np.linspace(lo1, hi1, 1000)
    ys1 = np.array([int(np.argmax(bins_hist - x > 0) - 1) for x in xs1])
    xs2 = np.linspace(lo2, hi2, 1000)
    ys2 = np.array([int(np.argmax(bins_hist - x > 0) - 1) for x in xs2])
    xs3 = np.linspace(lo3, hi3, 1000)
    ys3 = np.array([int(np.argmax(bins_hist - x > 0) - 1) for x in xs3])

    ax.step(q, h_hist, where="mid", color=colors[15])
    ax.fill_between(xs1, 0, h_hist[ys1], step="mid", facecolor=to_rgb(colors[15]) + (0.2,), edgecolor="none")
    ax.fill_between(xs2, 0, h_hist[ys2], step="mid", facecolor=to_rgb(colors[15]) + (0.2,), edgecolor="none")
    ax.fill_between(xs3, 0, h_hist[ys3], step="mid", facecolor=to_rgb(colors[15]) + (0.2,), edgecolor="none")
    ax.axvline(-2 * delta_llh_data,   color=colors[1],  label="HESE (2021)", lw=2)
    ax.axvline(-2 * delta_llh_sample, color=colors[14], label="Sampled",     lw=2)
    ax.set_ylim(0, None)
    ax.set_xlim(max(q[0], lo3), min(q[-1], hi3))
    ax.legend()
    ax.set_ylabel(r"$N_{\mathrm{sample}}$")
    ax.set_xlabel(r"$-2\left[\mathrm{LLH}(\mu \mid n)-\mathrm{LLH}(\mu \mid \mu)\right]$")

    plt.savefig("hese_figure.pdf")

    OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")
    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w") as h5f:
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "figure_4" in h5f.keys():
            del h5f["figure_4"]
        gp = h5f.create_group("figure_4")
        gp["e_cents"]            = cents
        gp["h_atmo"]             = h_atmo
        gp["h_sample"]           = h_sample
        gp["h_astro"]            = h_astro
        gp["h_data"]             = h_data
        gp["ntwodeltallhs"]      = -2 * delta_llhs
        gp["ntwodeltallhdata"]   = -2 * delta_llh_data
        gp["ntwodeltallhsample"] = -2 * delta_llh_sample
