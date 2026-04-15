"""IRF round-trip validation figures for the SPORE paper appendix.

For several monochromatic true energies, events are injected directly into the
smearing samplers and the resulting marginal distributions are compared against
the stored IRF tables.  Because the injected flux is monochromatic there is no
spectral convolution: any discrepancy is a pure sampler artefact.

Four figures are produced and saved individually:

  ps10yr_energy_smearing.png   PS-10yr track  — CDF of log10(E_reco / E_true)
  ps10yr_psf.png               PS-10yr track  — CDF of psi (degrees)
  hese_energy_smearing.png     HESE-7yr astro_cascade — CDF of log10(E_reco / E_true)
  hese_psf.png                 HESE-7yr astro_cascade — CDF of psi (degrees)

Each figure contains four curves (one per true energy).  Solid lines show the
analytic CDF derived directly from the IRF table; dashed lines show the
empirical CDF from N_SAMPLES monochromatic draws through the SPORE sampler.
Curves sharing a true energy share a colour.

Usage
-----
Run from the repository root::

    python scripts/paper_plots/irf_validation_appendix.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

import spore
from spore import Detector, Morphology
from spore.event_sampling.utils import smear_truth
from spore.conventions import SkyCoordinate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO         = Path(__file__).resolve().parents[2]
PS_RESPONSE   = _REPO / "resources" / "configs" / "ps10yr_detector_response.h5"
HESE_RESPONSE = _REPO / "resources" / "hese_7yr_detector_response.h5"
OUT_DIR       = _REPO / "scripts" / "paper_plots"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# True energies (GeV) — placed at PS-10yr bin centres for unambiguous look-up.
E_TRUE_GEV = [
    10 ** 4.25,   #  ~18  TeV
    10 ** 5.25,   #  ~180 TeV
    10 ** 5.75,   #  ~560 TeV
    10 ** 6.25,   #  ~1.8 PeV
]
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]   # Paul Tol colorblind-safe

N_SAMPLES = 100_000
RNG_SEED  = 42

# PS-10yr: upgoing dec band (zenith 100–180 deg)
PS_DEC_BAND_IDX   = 2
PS_ZENITH_MID_RAD = np.radians(140.0)

# HESE: PSF and energy resolution are zenith-independent
HESE_SC           = SkyCoordinate(0.0, 0.0)
HESE_ZENITH_RAD   = np.radians(90.0)
HESE_MORPH        = "astro_cascade"

LN10 = np.log(10.0)

# ---------------------------------------------------------------------------
# Load detectors
# ---------------------------------------------------------------------------
ps_det    = Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
    "response":   {"detector_response_file": str(PS_RESPONSE)},
})
ps_sampler = ps_det.response.joint_smearing["track"]

for name in ["astro_cascade", "astro_track", "atmo_cascade",
             "atmo_track", "astro_doublebang"]:
    Morphology.register(name)

hese_det = Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
    "response":   {"detector_response_file": str(HESE_RESPONSE)},
})

# ---------------------------------------------------------------------------
# Load raw IRF tables
# ---------------------------------------------------------------------------
with h5py.File(PS_RESPONSE) as hf:
    sm = hf["track/smearing"]
    ps_log10et_edges  = sm["log10e_true_edges"][:]
    ps_log10er_lo_all = sm["log10e_reco_lo"][:]
    ps_log10er_hi_all = sm["log10e_reco_hi"][:]
    ps_psf_lo_all     = sm["psf_lo"][:]
    ps_psf_hi_all     = sm["psf_hi"][:]
    ps_frac_all       = sm["fractional_counts"][:]

with h5py.File(HESE_RESPONSE) as hf:
    grp              = hf[HESE_MORPH]
    hese_er_us       = grp["energy_resolution/us"][:]
    hese_er_icdf     = grp["energy_resolution/inv_cdf"][:]
    hese_ar_es       = grp["angular_response/energies"][:]
    hese_ar_us       = grp["angular_response/us"][:]
    hese_ar_icdfs    = grp["angular_response/inv_cdfs"][:]

# ---------------------------------------------------------------------------
# Analytic CDF helpers
# ---------------------------------------------------------------------------

def ps_analytic_energy_cdf(log10et_mid):
    i_et = int(np.clip(
        np.searchsorted(ps_log10et_edges[1:], log10et_mid),
        0, len(ps_log10et_edges) - 2,
    ))
    frac = ps_frac_all[i_et, PS_DEC_BAND_IDX].copy()
    frac /= frac.sum()
    frac_e  = frac.sum(axis=(1, 2))
    er_mid  = 0.5 * (ps_log10er_lo_all[i_et, PS_DEC_BAND_IDX]
                     + ps_log10er_hi_all[i_et, PS_DEC_BAND_IDX])
    x       = er_mid - log10et_mid
    order   = np.argsort(x)
    x, frac_e = x[order], frac_e[order]
    return (np.concatenate([[x[0] - 1e-9], x]),
            np.clip(np.concatenate([[0.0], np.cumsum(frac_e)]), 0, 1))


def ps_analytic_psf_cdf(log10et_mid):
    i_et = int(np.clip(
        np.searchsorted(ps_log10et_edges[1:], log10et_mid),
        0, len(ps_log10et_edges) - 2,
    ))
    frac = ps_frac_all[i_et, PS_DEC_BAND_IDX].copy()
    frac /= frac.sum()
    frac_p = frac.sum(axis=(0, 2))
    p_mid  = 0.5 * (ps_psf_lo_all[i_et, PS_DEC_BAND_IDX]
                    + ps_psf_hi_all[i_et, PS_DEC_BAND_IDX])
    order  = np.argsort(p_mid)
    p_mid, frac_p = p_mid[order], frac_p[order]
    return (np.concatenate([[max(p_mid[0] - 1e-9, 1e-6)], p_mid]),
            np.clip(np.concatenate([[0.0], np.cumsum(frac_p)]), 0, 1))


def hese_analytic_energy_cdf():
    """Energy-independent; same for all E_true."""
    return hese_er_icdf / LN10, hese_er_us


def hese_analytic_psf_cdf(e_true_gev):
    i_e = int(np.argmin(np.abs(np.log(hese_ar_es) - np.log(e_true_gev))))
    return np.degrees(hese_ar_icdfs[i_e]), hese_ar_us


# ---------------------------------------------------------------------------
# Draw all samples up front
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
cdf_u = np.arange(1, N_SAMPLES + 1) / N_SAMPLES

ps_energy_samp  = {}   # log10et_mid -> sorted array of log10(Ereco/Etrue)
ps_psf_samp     = {}   # log10et_mid -> sorted array of psi (deg)
hese_energy_samp = {}
hese_psf_samp    = {}

for e_true_gev in E_TRUE_GEV:
    log10et = np.log10(e_true_gev)
    print(f"Sampling E_true = {e_true_gev/1e3:.0f} TeV ...")

    # PS-10yr
    ps_er = np.empty(N_SAMPLES)
    ps_psi = np.empty(N_SAMPLES)
    for k in range(N_SAMPLES):
        e_reco, psi_rad, _ = ps_sampler(e_true_gev, PS_ZENITH_MID_RAD, rng=rng)
        ps_er[k]  = np.log10(e_reco.magnitude)
        ps_psi[k] = np.degrees(psi_rad)
    ps_energy_samp[log10et] = np.sort(ps_er - log10et)
    ps_psf_samp[log10et]    = np.sort(ps_psi)

    # HESE
    hese_er  = np.empty(N_SAMPLES)
    hese_psi = np.empty(N_SAMPLES)
    for k in range(N_SAMPLES):
        reco_dir, reco_e, _ = smear_truth(HESE_SC, e_true_gev, hese_det,
                                           HESE_MORPH, rng=rng)
        e_gev = reco_e.magnitude if hasattr(reco_e, "magnitude") else float(reco_e)
        hese_er[k]  = np.log10(e_gev)
        hese_psi[k] = np.degrees(HESE_SC.separation(reco_dir))
    hese_energy_samp[log10et] = np.sort(hese_er - log10et)
    hese_psf_samp[log10et]    = np.sort(hese_psi)

print("Sampling done.\n")

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
IRF_LS   = "-"
SPORE_LS = "--"
LW       = 1.8

ENERGY_LABELS = [
    f"$\\approx{e/1e3:.0f}$ TeV" for e in E_TRUE_GEV
]


def _add_legend(ax):
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="k", ls=IRF_LS,   lw=LW, label="IRF table"),
        Line2D([0], [0], color="k", ls=SPORE_LS, lw=LW, label="SPORE sampler"),
    ]
    color_handles = [
        Line2D([0], [0], color=c, ls="-", lw=LW, label=lbl)
        for c, lbl in zip(COLORS, ENERGY_LABELS)
    ]
    leg1 = ax.legend(handles=style_handles, fontsize=9, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, fontsize=9, loc="lower right")


# ---------------------------------------------------------------------------
# Figure 1: PS-10yr energy smearing
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for e_true_gev, color in zip(E_TRUE_GEV, COLORS):
    log10et = np.log10(e_true_gev)
    x_irf, y_irf = ps_analytic_energy_cdf(log10et)
    ax.step(x_irf, y_irf, where="post", lw=LW, color=color, ls=IRF_LS)
    ax.plot(ps_energy_samp[log10et], cdf_u,   lw=LW, color=color, ls=SPORE_LS)

ax.axvline(0.0, color="gray", lw=0.8, ls=":")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"$\log_{10}(E_{\rm reco}\,/\,E_{\rm true})$", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
ax.set_title("PS-10yr track — energy smearing\nupgoing dec band (dec > 10°)", fontsize=11)
ax.grid(True, lw=0.4, alpha=0.5)
_add_legend(ax)
plt.tight_layout()
out = OUT_DIR / "ps10yr_energy_smearing.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: PS-10yr PSF
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for e_true_gev, color in zip(E_TRUE_GEV, COLORS):
    log10et = np.log10(e_true_gev)
    x_irf, y_irf = ps_analytic_psf_cdf(log10et)
    ax.step(x_irf, y_irf, where="post", lw=LW, color=color, ls=IRF_LS)
    ax.plot(ps_psf_samp[log10et], cdf_u,    lw=LW, color=color, ls=SPORE_LS)

ax.set_xscale("log")
ax.set_xlim(0.1, 180)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"$\psi$ [deg]", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
ax.set_title("PS-10yr track — PSF\nupgoing dec band (dec > 10°)", fontsize=11)
ax.grid(True, lw=0.4, alpha=0.5, which="both")
_add_legend(ax)
plt.tight_layout()
out = OUT_DIR / "ps10yr_psf.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: HESE energy smearing
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
x_irf_e, y_irf_e = hese_analytic_energy_cdf()
for e_true_gev, color in zip(E_TRUE_GEV, COLORS):
    log10et = np.log10(e_true_gev)
    # analytic CDF is energy-independent — draw once per energy for legend pairing
    ax.plot(x_irf_e, y_irf_e,                    lw=LW, color=color, ls=IRF_LS)
    ax.plot(hese_energy_samp[log10et], cdf_u,    lw=LW, color=color, ls=SPORE_LS)

ax.axvline(0.0, color="gray", lw=0.8, ls=":")
ax.set_xlim(-5.0, 1.5)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"$\log_{10}(E_{\rm reco}\,/\,E_{\rm true})$", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
ax.set_title(
    "HESE-7yr astro\\_cascade — energy smearing\n"
    "(energy-independent IRF; all analytic curves overlap)",
    fontsize=11,
)
ax.grid(True, lw=0.4, alpha=0.5)
_add_legend(ax)
plt.tight_layout()
out = OUT_DIR / "hese_energy_smearing.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4: HESE PSF
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for e_true_gev, color in zip(E_TRUE_GEV, COLORS):
    log10et = np.log10(e_true_gev)
    x_irf_p, y_irf_p = hese_analytic_psf_cdf(e_true_gev)
    ax.plot(x_irf_p, y_irf_p,                  lw=LW, color=color, ls=IRF_LS)
    ax.plot(hese_psf_samp[log10et], cdf_u,     lw=LW, color=color, ls=SPORE_LS)

ax.set_xscale("log")
ax.set_xlim(0.1, 60)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel(r"$\psi$ [deg]", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
ax.set_title("HESE-7yr astro\\_cascade — PSF", fontsize=11)
ax.grid(True, lw=0.4, alpha=0.5, which="both")
_add_legend(ax)
plt.tight_layout()
out = OUT_DIR / "hese_psf.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()
