"""IRF round-trip validation figures for the SPORE paper appendix.

For several monochromatic true energies, events are injected directly into the
smearing samplers and the resulting marginal distributions are compared against
the stored IRF tables.  Because the injected flux is monochromatic there is no
spectral convolution: any discrepancy is a pure sampler artefact.

Four figures are produced and saved individually:

  ps14yr_energy_smearing.pdf   PS-14yr track  — CDF of log10(E_reco / E_true)
  ps14yr_psf.pdf               PS-14yr track  — CDF of psi (degrees)
  hese_energy_smearing.pdf     HESE-7yr astro_cascade — CDF of log10(E_reco / E_true)
  hese_psf.pdf                 HESE-7yr astro_cascade — CDF of psi (degrees)

Each figure contains four curves (one per true energy).  Solid lines show the
analytic CDF derived directly from the IRF table; dashed lines show the
empirical CDF from N_SAMPLES monochromatic draws through the SPORE sampler.
Curves sharing a true energy share a colour.

Usage
-----
Run from the repository root::

    python scripts/paper_plots/irf_validation_appendix.py

Or import and call ``main()`` from a notebook::

    from scripts.paper_plots.irf_validation_appendix import main
    main()
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

import spore
from spore import Detector, Morphology
from spore.event_sampling.utils import smear_truth
from spore.conventions import SkyCoordinate

_REPO         = Path(__file__).resolve().parents[2]
PS_RESPONSE   = _REPO / "resources" / "configs" / "ps14yr_detector_response.h5"
HESE_RESPONSE = _REPO / "resources" / "hese_7yr_detector_response.h5"
OUT_DIR       = _REPO / "paper" / "figures"

# True energies (GeV) — placed at PS-14yr bin centres for unambiguous look-up.
E_TRUE_GEV = [
    10 ** 4.25,   #  ~18  TeV
    10 ** 5.25,   #  ~180 TeV
    10 ** 5.75,   #  ~560 TeV
    10 ** 6.25,   #  ~1.8 PeV
]
COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]   # Paul Tol colorblind-safe

N_SAMPLES = 100_000
RNG_SEED  = 42

# Northern (upgoing) sky.  The declination band must be looked up rather than
# hard-coded: the 10-year release bins the smearing matrix in three bands and
# the 14-year release in 41, so a fixed index does not mean the same thing in
# both.  The sampling zenith is derived from the same declination, so the
# analytic and sampled curves refer to the same band by construction.
PS_DEC_DEG = 30.0
PS_ZENITH_MID_RAD = np.arccos(-np.sin(np.radians(PS_DEC_DEG)))


def ps_dec_band_index(ps_tables, dec_deg=PS_DEC_DEG):
    """Index of the smearing declination band containing ``dec_deg``."""
    edges = np.asarray(ps_tables["dec_edges"], float)
    return int(np.clip(np.searchsorted(edges, dec_deg) - 1, 0, len(edges) - 2))

# HESE: PSF and energy resolution are zenith-independent
HESE_SC         = SkyCoordinate(0.0, 0.0)
HESE_ZENITH_RAD = np.radians(90.0)
HESE_MORPH      = "astro_cascade"

LN10 = np.log(10.0)


def load_ps_tables(ps_response_path=PS_RESPONSE):
    """Load PS-14yr smearing tables from HDF5."""
    with h5py.File(ps_response_path) as hf:
        sm = hf["track/smearing"]
        return dict(
            log10et_edges  = sm["log10e_true_edges"][:],
            dec_edges      = sm["dec_edges"][:],
            log10er_lo_all = sm["log10e_reco_lo"][:],
            log10er_hi_all = sm["log10e_reco_hi"][:],
            psf_lo_all     = sm["psf_lo"][:],
            psf_hi_all     = sm["psf_hi"][:],
            frac_all       = sm["fractional_counts"][:],
        )


def load_hese_tables(hese_response_path=HESE_RESPONSE, morph=HESE_MORPH):
    """Load HESE IRF tables for the given morphology from HDF5."""
    with h5py.File(hese_response_path) as hf:
        grp = hf[morph]
        return dict(
            er_us    = grp["energy_resolution/us"][:],
            er_icdf  = grp["energy_resolution/inv_cdf"][:],
            ar_es    = grp["angular_response/energies"][:],
            ar_us    = grp["angular_response/us"][:],
            ar_icdfs = grp["angular_response/inv_cdfs"][:],
        )


def ps_analytic_energy_cdf(log10et_mid, ps_tables, dec_band_idx=None):
    """Analytic energy smearing CDF for a given log10(E_true) from PS-14yr tables."""
    if dec_band_idx is None:
        dec_band_idx = ps_dec_band_index(ps_tables)
    edges = ps_tables["log10et_edges"]
    frac  = ps_tables["frac_all"]
    lo    = ps_tables["log10er_lo_all"]
    hi    = ps_tables["log10er_hi_all"]

    i_et = int(np.clip(np.searchsorted(edges[1:], log10et_mid), 0, len(edges) - 2))
    f = frac[i_et, dec_band_idx].copy()
    f /= f.sum()
    frac_e = f.sum(axis=(1, 2))
    er_mid = 0.5 * (lo[i_et, dec_band_idx] + hi[i_et, dec_band_idx])
    x      = er_mid - log10et_mid
    order  = np.argsort(x)
    x, frac_e = x[order], frac_e[order]
    return (np.concatenate([[x[0] - 1e-9], x]),
            np.clip(np.concatenate([[0.0], np.cumsum(frac_e)]), 0, 1))


def ps_analytic_psf_cdf(log10et_mid, ps_tables, dec_band_idx=None):
    """Analytic PSF CDF for a given log10(E_true) from PS-14yr tables."""
    if dec_band_idx is None:
        dec_band_idx = ps_dec_band_index(ps_tables)
    edges = ps_tables["log10et_edges"]
    frac  = ps_tables["frac_all"]
    p_lo  = ps_tables["psf_lo_all"]
    p_hi  = ps_tables["psf_hi_all"]

    i_et = int(np.clip(np.searchsorted(edges[1:], log10et_mid), 0, len(edges) - 2))
    f = frac[i_et, dec_band_idx].copy()
    f /= f.sum()
    frac_p = f.sum(axis=(0, 2))
    p_mid  = 0.5 * (p_lo[i_et, dec_band_idx] + p_hi[i_et, dec_band_idx])
    order  = np.argsort(p_mid)
    p_mid, frac_p = p_mid[order], frac_p[order]
    return (np.concatenate([[max(p_mid[0] - 1e-9, 1e-6)], p_mid]),
            np.clip(np.concatenate([[0.0], np.cumsum(frac_p)]), 0, 1))


def hese_analytic_energy_cdf(hese_tables):
    """Analytic energy smearing CDF from HESE tables (energy-independent)."""
    return hese_tables["er_icdf"] / LN10, hese_tables["er_us"]


def hese_analytic_psf_cdf(e_true_gev, hese_tables):
    """Analytic PSF CDF for a given E_true from HESE tables."""
    i_e = int(np.argmin(np.abs(np.log(hese_tables["ar_es"]) - np.log(e_true_gev))))
    return np.degrees(hese_tables["ar_icdfs"][i_e]), hese_tables["ar_us"]


def draw_samples(ps_det, hese_det, e_true_list=E_TRUE_GEV, n_samples=N_SAMPLES, seed=RNG_SEED):
    """
    Draw monochromatic samples through PS-14yr and HESE smearing.

    Returns
    -------
    ps_energy_samp  : dict  log10(E_true) -> sorted array of log10(E_reco/E_true)
    ps_psf_samp     : dict  log10(E_true) -> sorted array of psi (deg)
    hese_energy_samp: dict  log10(E_true) -> sorted array of log10(E_reco/E_true)
    hese_psf_samp   : dict  log10(E_true) -> sorted array of psi (deg)
    """
    ps_sampler = ps_det.response.joint_smearing["track"]
    rng = np.random.default_rng(seed)

    ps_energy_samp   = {}
    ps_psf_samp      = {}
    hese_energy_samp = {}
    hese_psf_samp    = {}

    for e_true_gev in e_true_list:
        log10et = np.log10(e_true_gev)
        print(f"Sampling E_true = {e_true_gev/1e3:.0f} TeV ...")

        ps_er  = np.empty(n_samples)
        ps_psi = np.empty(n_samples)
        for k in range(n_samples):
            e_reco, psi_rad, _ = ps_sampler(e_true_gev, PS_ZENITH_MID_RAD, rng=rng)
            ps_er[k]  = np.log10(e_reco.magnitude)
            ps_psi[k] = np.degrees(psi_rad)
        ps_energy_samp[log10et] = np.sort(ps_er - log10et)
        ps_psf_samp[log10et]    = np.sort(ps_psi)

        hese_er  = np.empty(n_samples)
        hese_psi = np.empty(n_samples)
        for k in range(n_samples):
            reco_dir, reco_e, _ = smear_truth(HESE_SC, e_true_gev, hese_det,
                                               HESE_MORPH, rng=rng)
            e_gev = reco_e.magnitude if hasattr(reco_e, "magnitude") else float(reco_e)
            hese_er[k]  = np.log10(e_gev)
            hese_psi[k] = np.degrees(HESE_SC.separation(reco_dir))
        hese_energy_samp[log10et] = np.sort(hese_er - log10et)
        hese_psf_samp[log10et]    = np.sort(hese_psi)

    print("Sampling done.\n")
    return ps_energy_samp, ps_psf_samp, hese_energy_samp, hese_psf_samp


def _add_legend(ax, irf_ls="-", spore_ls="--", lw=1.8, e_true_list=E_TRUE_GEV):
    from matplotlib.lines import Line2D
    energy_labels = [f"$\\approx{e/1e3:.0f}$ TeV" for e in e_true_list]
    style_handles = [
        Line2D([0], [0], color="k", ls=irf_ls,   lw=lw, label="IRF table"),
        Line2D([0], [0], color="k", ls=spore_ls, lw=lw, label="SPORE sampler"),
    ]
    color_handles = [
        Line2D([0], [0], color=c, ls="-", lw=lw, label=lbl)
        for c, lbl in zip(COLORS, energy_labels)
    ]
    leg1 = ax.legend(handles=style_handles, fontsize=9, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, fontsize=9, loc="lower right")


def make_figures(ps_tables, hese_tables,
                 ps_energy_samp, ps_psf_samp, hese_energy_samp, hese_psf_samp,
                 e_true_list=E_TRUE_GEV, out_dir=OUT_DIR):
    """Generate and save the four validation figures."""
    out_dir  = Path(out_dir)
    irf_ls   = "-"
    spore_ls = "--"
    lw       = 1.8
    cdf_u    = np.arange(1, N_SAMPLES + 1) / N_SAMPLES

    # Figure 1: PS-14yr energy smearing
    fig, ax = plt.subplots(figsize=(6, 5))
    for e_true_gev, color in zip(e_true_list, COLORS):
        log10et = np.log10(e_true_gev)
        x_irf, y_irf = ps_analytic_energy_cdf(log10et, ps_tables)
        ax.step(x_irf, y_irf, where="post", lw=lw, color=color, ls=irf_ls)
        ax.plot(ps_energy_samp[log10et], cdf_u, lw=lw, color=color, ls=spore_ls)
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\log_{10}(E_{\rm reco}\,/\,E_{\rm true})$", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("PS-14yr track — energy smearing\nupgoing dec band (dec > 10°)", fontsize=11)
    ax.grid(True, lw=0.4, alpha=0.5)
    _add_legend(ax, irf_ls=irf_ls, spore_ls=spore_ls, lw=lw, e_true_list=e_true_list)
    plt.tight_layout()
    out = out_dir / "ps14yr_energy_smearing.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # Figure 2: PS-14yr PSF
    fig, ax = plt.subplots(figsize=(6, 5))
    for e_true_gev, color in zip(e_true_list, COLORS):
        log10et = np.log10(e_true_gev)
        x_irf, y_irf = ps_analytic_psf_cdf(log10et, ps_tables)
        ax.step(x_irf, y_irf, where="post", lw=lw, color=color, ls=irf_ls)
        ax.plot(ps_psf_samp[log10et], cdf_u, lw=lw, color=color, ls=spore_ls)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 180)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\psi$ [deg]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("PS-14yr track — PSF\nupgoing dec band (dec > 10°)", fontsize=11)
    ax.grid(True, lw=0.4, alpha=0.5, which="both")
    _add_legend(ax, irf_ls=irf_ls, spore_ls=spore_ls, lw=lw, e_true_list=e_true_list)
    plt.tight_layout()
    out = out_dir / "ps14yr_psf.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # Figure 3: HESE energy smearing
    fig, ax = plt.subplots(figsize=(6, 5))
    x_irf_e, y_irf_e = hese_analytic_energy_cdf(hese_tables)
    for e_true_gev, color in zip(e_true_list, COLORS):
        log10et = np.log10(e_true_gev)
        ax.plot(x_irf_e, y_irf_e,                  lw=lw, color=color, ls=irf_ls)
        ax.plot(hese_energy_samp[log10et], cdf_u,  lw=lw, color=color, ls=spore_ls)
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
    _add_legend(ax, irf_ls=irf_ls, spore_ls=spore_ls, lw=lw, e_true_list=e_true_list)
    plt.tight_layout()
    out = out_dir / "hese_energy_smearing.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # Figure 4: HESE PSF
    fig, ax = plt.subplots(figsize=(6, 5))
    for e_true_gev, color in zip(e_true_list, COLORS):
        log10et = np.log10(e_true_gev)
        x_irf_p, y_irf_p = hese_analytic_psf_cdf(e_true_gev, hese_tables)
        ax.plot(x_irf_p, y_irf_p,                lw=lw, color=color, ls=irf_ls)
        ax.plot(hese_psf_samp[log10et], cdf_u,  lw=lw, color=color, ls=spore_ls)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 60)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\psi$ [deg]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("HESE-7yr astro\\_cascade — PSF", fontsize=11)
    ax.grid(True, lw=0.4, alpha=0.5, which="both")
    _add_legend(ax, irf_ls=irf_ls, spore_ls=spore_ls, lw=lw, e_true_list=e_true_list)
    plt.tight_layout()
    out = out_dir / "hese_psf.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main(ps_response_path=PS_RESPONSE, hese_response_path=HESE_RESPONSE,
         out_dir=OUT_DIR, e_true_list=E_TRUE_GEV, n_samples=N_SAMPLES, seed=RNG_SEED):
    """Run the full IRF validation: sample, then generate all four figures."""
    for name in ["astro_cascade", "astro_track", "atmo_cascade",
                 "atmo_track", "astro_doublebang"]:
        Morphology.register(name)

    ps_det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
        "response":   {"detector_response_file": str(ps_response_path)},
    })
    hese_det = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
        "response":   {"detector_response_file": str(hese_response_path)},
    })

    ps_tables   = load_ps_tables(ps_response_path)
    hese_tables = load_hese_tables(hese_response_path)

    ps_energy_samp, ps_psf_samp, hese_energy_samp, hese_psf_samp = draw_samples(
        ps_det, hese_det, e_true_list=e_true_list, n_samples=n_samples, seed=seed
    )

    make_figures(
        ps_tables, hese_tables,
        ps_energy_samp, ps_psf_samp, hese_energy_samp, hese_psf_samp,
        e_true_list=e_true_list, out_dir=out_dir,
    )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
