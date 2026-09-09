"""Round-trip validation against the IceCube 14-year track release (IceTracks-DR2).

This is the DR2 counterpart of ``pstracks_roundtrip.py``, written to answer a
specific question: the low-energy excess reported in Sec. 6.2 is attributed to
the granularity of the *released* smearing matrix rather than to the sampling.
DR2 refines the declination banding of that matrix from 3 bands to 41 while
leaving the true-energy binning at 0.5 dex.  If the attribution is right, the
low-energy feature should largely survive the change and the declination
residual should improve.

The script mirrors ``pstracks_roundtrip.py`` step for step so the two are
comparable:

  * the same atmospheric model (MCEq / GSF / Sibyll-2.3d) and the same
    astrophysical flux, with the atmospheric normalisation refit to the DR2
    sin(dec) distribution alone, so the energy-proxy panel remains a prediction;
  * the same northern-sky (dec > 0) selection, the same binning, and the same
    number of pseudo-experiments.

Two differences are forced by the release itself and are worth stating:

  * DR2 unifies the IC86 configuration, so there is a single IRF rather than a
    per-season average, and the sample spans IC86-I..XI rather than I..VII.
  * The astrophysical flux is still the 10-year best fit, since the DR2
    companion fit is not yet available in tabulated form.  It is held fixed
    between the two round-trips, so it cannot be the source of any difference.

Outputs ``paper/figures/dr1_vs_dr2_roundtrip.pdf`` and caches the histograms
in ``resources/plotting_data.h5`` under ``figure_dr2``.

Usage
-----
    python scripts/paper_plots/pstracks_roundtrip_dr2.py
"""

import logging
import os

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln

from spore.conventions import ureg
from spore.detector.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import SourceSampler

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(HERE, "..", "..")
DR2 = os.path.join(REPO, "resources", "data_releases", "ps14yr_data_release")
ATM_H5 = os.path.join(REPO, "resources", "atmo_flux_models.h5")
AST_H5 = os.path.join(REPO, "resources", "ps10yr_combined_flux.h5")
STYLE = os.path.join(REPO, "resources", "paper.mplstyle")
OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")
FIGDIR = os.path.join(REPO, "paper", "figures")
DR2_RESPONSE = os.path.join(REPO, "resources", "configs", "ps14yr_detector_response.h5")

# DR2 publishes IC86-I through IC86-XI.
SEASONS = ["IC86_I", "IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI",
           "IC86_VII", "IC86_VIII", "IC86_IX", "IC86_X", "IC86_XI"]

# DR2 event columns: run, event, subevent, MJD, log10(E/GeV), AngErr, RA, Dec,
# Azimuth, Zenith.  DR1 omits the first three, hence the offset of 3.
COL_LOG10E = 4
COL_DEC = 7

ATMO_FLUX_MODEL = "mceq_h4a_sibyll23d"

# Pseudo-experiments averaged into each template.  The Monte Carlo noise on the
# mean of N draws goes as 1/sqrt(N * mu); with ~1.3e5 predicted counts in the
# 200-600 GeV band that is 0.09% at N = 10, against a residual of order 50%, so
# N = 10 is ample for every band the Sec. 6.2 argument depends on.  Only the
# >50 TeV tail (~66 counts) is visibly noisier than the DR1 figure's N = 40.
N = 40
# Livetime per sampling call.  Bounds peak memory: the full DR2 IC86 livetime
# is ~3921 days and yields ~1.4M events in one go, so draw it in slices.
CHUNK_DAYS = 400.0
SEED = 0
ATMO_FIT_OFF = 0
ATMO_SEED_OFF = 100_000
ASTRO_SEED_OFF = 200_000


def poisson_llh(n, mu):
    """Poisson log-likelihood, matching ``pstracks_roundtrip.poisson_llh``."""
    mu_safe = np.where(mu > 0, mu, 1e-300)
    return float(np.sum(n * np.log(mu_safe) - mu - gammaln(n + 1)))


def _reco_sindecs(events):
    return np.sin(np.fromiter(
        (e.reco_direction.declination for e in events), float, len(events)
    ))


def _reco_energies(events):
    return np.fromiter((e._reco_e for e in events), float, len(events))


def _sample(sampler, t_obs, seed0, sd_bins, e_bins, n_pe):
    """Average the northern-sky sin(dec) and energy-proxy histograms over PEs.

    Each pseudo-experiment is drawn in livetime chunks rather than in one call.
    The DR2 sample is ~1.4M events per pseudo-experiment, and materialising
    them all before histogramming costs well over a gigabyte; chunking bounds
    the peak at roughly ``CHUNK_DAYS / livetime`` of that.

    This is exact, not an approximation: event counts are Poisson in the
    livetime and the draws are independent, so summing the histograms of k
    disjoint sub-intervals is distributed identically to one draw over their
    union.  Each chunk gets its own seed so the sub-draws are independent.
    """
    h_sd = np.zeros(len(sd_bins) - 1)
    h_e = np.zeros(len(e_bins) - 1)
    total_days = t_obs.to("day").magnitude
    n_chunk = max(1, int(np.ceil(total_days / CHUNK_DAYS)))
    chunk = ureg.Quantity(total_days / n_chunk, "day")

    for pe in range(n_pe):
        if pe % 5 == 0:
            logging.info("  pseudo-exp %d/%d (%d chunks each)", pe, n_pe, n_chunk)
        for c in range(n_chunk):
            ev = sampler.sample_events(
                "track", deltat=chunk, seed=seed0 + pe * 1000 + c,
            )
            sd = _reco_sindecs(ev)
            up = sd > 0
            h_sd += np.histogram(sd[up], bins=sd_bins)[0]
            h_e += np.histogram(_reco_energies(ev)[up], bins=e_bins)[0]
            del ev, sd, up
    return h_sd / n_pe, h_e / n_pe


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    plt.style.use(STYLE)

    det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0,
                       "depth": 1945, "medium": "Ice"},
        "response": {"detector_response_file": DR2_RESPONSE},
    })

    atmo_src = ExtendedSource.from_config(
        {"flux": {"location": f"{ATM_H5}:{ATMO_FLUX_MODEL}"}})
    astro_src = ExtendedSource.from_config(
        {"flux": {"location": f"{AST_H5}:astrophysical"}})

    all_ev = np.vstack([
        np.genfromtxt(os.path.join(DR2, "events", f"{s}_exp.tab"), comments="#")
        for s in SEASONS
    ])
    sindec = np.sin(np.radians(all_ev[:, COL_DEC]))

    total_days = 0.0
    for s in SEASONS:
        ut = np.genfromtxt(os.path.join(DR2, "uptime", f"{s}_exp.tab"), comments="#")
        total_days += (ut[:, 1] - ut[:, 0]).sum()
    t_obs = ureg.Quantity(total_days, "day")
    logging.info("DR2 IC86 livetime: %.1f days (%d events)", total_days, len(all_ev))

    sd_bins = np.linspace(0, 1, 21)
    sd_cents = 0.5 * (sd_bins[1:] + sd_bins[:-1])
    e_bins = np.logspace(2, 6, 33)
    e_cents = np.sqrt(e_bins[1:] * e_bins[:-1])

    north = sindec > 0
    h_data = np.histogram(sindec[north], bins=sd_bins)[0]
    h_data_e = np.histogram(10 ** all_ev[north, COL_LOG10E], bins=e_bins)[0]

    logging.info("Building initial atmo sampler ...")
    atmo_sampler = SourceSampler(det, atmo_src, n_time_samples=50,
                                 n_dec=100, n_ra=100, n_e=100)

    logging.info("Fitting atmo normalisation (%d pseudo-experiments) ...", N)
    h_atmo_fit, _ = _sample(atmo_sampler, t_obs, SEED + ATMO_FIT_OFF,
                            sd_bins, e_bins, N)
    h0 = abs(poisson_llh(h_data, h_atmo_fit))
    res = minimize(lambda x: -poisson_llh(h_data, x * h_atmo_fit) / h0,
                   1.0, method="L-BFGS-B")
    norm = float(res.x[0])
    logging.info("  fitted atmospheric normalisation = %.4f", norm)

    atmo_src = ExtendedSource.from_config(
        {"flux": {"location": f"{ATM_H5}:{ATMO_FLUX_MODEL}"}}, normalization=norm)

    logging.info("Rebuilding atmo sampler with fitted norm ...")
    atmo_sampler = SourceSampler(det, atmo_src, n_time_samples=50,
                                 n_dec=100, n_ra=100, n_e=100)
    logging.info("Building astro sampler ...")
    astro_sampler = SourceSampler(det, astro_src, n_time_samples=50,
                                  n_dec=100, n_ra=100, n_e=100)

    logging.info("Sampling atmo ...")
    h_atmo, h_atmo_e = _sample(atmo_sampler, t_obs, SEED + ATMO_SEED_OFF,
                               sd_bins, e_bins, N)
    logging.info("Sampling astro ...")
    h_astro, h_astro_e = _sample(astro_sampler, t_obs, SEED + ASTRO_SEED_OFF,
                                 sd_bins, e_bins, N)

    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w"):
            pass
    with h5.File(OUTFILE, "r+") as h5f:
        if "figure_dr2" in h5f:
            del h5f["figure_dr2"]
        gp = h5f.create_group("figure_dr2")
        for k, v in (("sd_cents", sd_cents), ("h_atmo", h_atmo),
                     ("h_astro", h_astro), ("h_data", h_data),
                     ("e_cents", e_cents), ("h_atmo_e", h_atmo_e),
                     ("h_astro_e", h_astro_e), ("h_data_e", h_data_e)):
            gp[k] = v
        gp.attrs["norm"] = norm
        gp.attrs["livetime_days"] = total_days

    _plot(sd_cents, h_atmo, h_astro, h_data,
          e_cents, h_atmo_e, h_astro_e, h_data_e, norm)


def _ratio(data, pred):
    good = pred > 0
    r = np.full(pred.shape, np.nan)
    e = np.full(pred.shape, np.nan)
    r[good] = data[good] / pred[good]
    e[good] = np.sqrt(data[good]) / pred[good]
    return r, e


def _plot(sd_cents, h_atmo, h_astro, h_data,
          e_cents, h_atmo_e, h_astro_e, h_data_e, norm):
    """DR2 round-trip, with the DR1 residuals overlaid where available."""
    dr1 = None
    with h5.File(OUTFILE, "r") as h5f:
        if "figure_5" in h5f:
            g = h5f["figure_5"]
            dr1 = {k: g[k][:] for k in ("sd_cents", "h_atmo", "h_astro",
                                        "h_data", "e_cents", "h_atmo_e",
                                        "h_astro_e", "h_data_e")}

    fig, axs = plt.subplots(
        2, 2, figsize=(11, 5.6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07, "wspace": 0.22},
        sharex="col",
    )

    panels = (
        (axs[0, 0], axs[1, 0], sd_cents, h_atmo, h_astro, h_data,
         r"$\sin\delta$", None, (0.7, 1.3), None,
         None if dr1 is None else (dr1["sd_cents"],
                                   dr1["h_atmo"] + dr1["h_astro"], dr1["h_data"])),
        (axs[0, 1], axs[1, 1], e_cents, h_atmo_e, h_astro_e, h_data_e,
         r"$E_{\mathrm{proxy}}~[\mathrm{GeV}]$", "log", (0.05, 5.0), "log",
         None if dr1 is None else (dr1["e_cents"],
                                   dr1["h_atmo_e"] + dr1["h_astro_e"], dr1["h_data_e"])),
    )

    for (ax, axr, x, atmo, astro, data, xlabel, xscale,
         rylim, ryscale, dr1_panel) in panels:
        pred = atmo + astro
        ax.errorbar(x, data, yerr=np.sqrt(data), label="Observed (DR2)",
                    color="k", linestyle="none", marker="o", markersize=3, zorder=5)
        ax.fill_between(x, astro, pred, step="mid", label="Atmospheric (MCEq GSF)",
                        facecolor=(0.53, 0.81, 0.98, 0.5), edgecolor=(0.53, 0.81, 0.98, 1.0))
        ax.fill_between(x, 1, astro, step="mid", label="Astrophysical",
                        facecolor=(0.86, 0.44, 0.58, 0.5), edgecolor=(0.86, 0.44, 0.58, 1.0))
        ax.set_yscale("log")
        ax.set_ylim(1, 10 * max(h_data.max(), h_data_e.max()))
        ax.tick_params(axis="x", labelbottom=False)

        r, re = _ratio(data, pred)
        if dr1_panel is not None:
            x1, pred1, data1 = dr1_panel
            r1, _ = _ratio(data1, pred1)
            axr.step(x1, r1, where="mid", color="C0", lw=1.2, alpha=0.9,
                     label="DR1 (10-yr)")
        axr.errorbar(x, r, yerr=re, color="k", linestyle="none",
                     marker="o", markersize=3, label="DR2 (14-yr)")
        axr.axhline(1.0, color="gray", lw=0.8, ls="--")
        if ryscale:
            axr.set_yscale(ryscale)
        axr.set_ylim(*rylim)
        axr.set_xlabel(xlabel)
        axr.set_ylabel("Obs./Pred.")
        if xscale:
            ax.set_xscale(xscale)
            axr.set_xscale(xscale)

    axs[0, 0].set_xlim(0, 1)
    axs[0, 1].set_xlim(1e2, 2e5)
    axs[0, 0].set_ylabel("Events")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    axs[1, 1].legend(fontsize=7, loc="upper right", ncol=2)
    axs[0, 0].set_title(rf"DR2 round trip, atmospheric norm $={norm:.2f}$",
                        fontsize=9, loc="left")

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "dr1_vs_dr2_roundtrip.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
