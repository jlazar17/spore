import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib import colormaps
from colorspacious import cspace_converter

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
STYLE   = os.path.join(REPO, "resources", "paper.mplstyle")

# Rendered figures land here unless --outdir says otherwise.
OUTDIR  = os.path.join(REPO, "figures")

plt.style.use(STYLE)
COLORS = colormaps["tab20b"](np.linspace(0, 1, 20))
# Single-column width in inches; double column = 2 * COLUMN_WIDTH
COLUMN_WIDTH = 6.09263285024

blues = colormaps["Blues"]
reds  = colormaps["Reds"]


def _gray_from_lightness(lightness: float) -> tuple:
    """Return an sRGB gray with the given CAM02-UCS lightness."""
    rgb = cspace_converter("CAM02-UCS", "sRGB1")([lightness, 0, 0])
    return tuple(float(x) for x in rgb)

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--datafile", type=str,
        default=os.path.join(REPO, "resources", "plotting_data.h5"),
        help="HDF5 file of pre-computed figure data (default: the copy "
             "shipped in resources/).",
    )
    parser.add_argument(
        "--outdir", type=str, default=OUTDIR,
        help="Directory to write the rendered PDFs into.",
    )
    return parser.parse_args()

def min_width_interval(x, fraction):
    """Return (lo, hi) of the shortest interval containing `fraction` of x."""
    x = np.sort(x)
    n = len(x)
    k = int(np.ceil(fraction * n))   # number of points the window must contain
    widths = x[k - 1:] - x[:n - k + 1]
    i = np.argmin(widths)
    return x[i], x[i + k - 1]

def hese_comparison(datafile):
    # The left panel carries four overlapping series; give it the extra width.
    fig, axs = plt.subplots(
        1, 2,
        figsize=(1.8 * COLUMN_WIDTH, 0.62 * COLUMN_WIDTH),
        gridspec_kw={"width_ratios": [1.6, 1.0], "wspace": 0.28},
    )

    ax = axs[0]
    es = np.logspace(4, 7, 28)
    cents = (es[1:] + es[:-1]) / 2

    with h5.File(datafile) as h5f:
        gp = h5f["figure_4"]
        cents = gp["e_cents"][:]
        h_atmo = gp["h_atmo"][:]
        h_sample = gp["h_sample"][:]
        h_astro = gp["h_astro"][:]
        h_data = gp["h_data"][:]
        ntwodeltallhs = gp["ntwodeltallhs"][:]
        ntwodeltallhdata = gp["ntwodeltallhdata"][()]
        ntwodeltallhsample = gp["ntwodeltallhsample"][()]

    ax.fill_between(
        cents,
        h_atmo,
        h_atmo + h_astro,
        step="mid",
        label="Astrophysical",
        facecolor=to_rgb(COLORS[-1]) + (0.5,),
        edgecolor=to_rgb(COLORS[-1])  + (1.0,)
    )
    
    ax.fill_between(
        cents,
        0,
        h_atmo,
        step="mid",
        label="Atmospheric",
        facecolor=to_rgb("lightskyblue") + (0.5,),
        edgecolor=to_rgb("lightskyblue")  + (1.0,)
    
    )
    
    # Offset the two point series horizontally so their error bars stay separable.
    ax.errorbar(
        cents / 1.045,
        h_sample,
        marker="o",
        markersize=4,
        linestyle="none",
        elinewidth=1.2,
        capsize=2,
        yerr=np.sqrt(h_sample),
        label="Sampled",
        color=COLORS[14],
        zorder=5,
    )

    ax.errorbar(
        cents * 1.045,
        h_data,
        marker="s",
        markersize=4,
        linestyle="none",
        elinewidth=1.2,
        capsize=2,
        yerr=np.sqrt(h_data),
        label="HESE (2021)",
        color=COLORS[1],
        zorder=5,
    )

    ax.fill_between(
        [1, 6e4],
        [1e-6, 1e-6],
        [1e6, 1e6],
        facecolor=to_rgb("lightgrey") + (0.5,),
        edgecolor="lightgrey"
    )
    ax.axvline(6e4, color="lightgrey")
    ax.text(
        2.4e4, 0.13, "Not used\nin LLH",
        fontsize=9, color="grey", ha="center", va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Data run out just above 1e6 GeV; the extra decade only added white space.
    ax.set_xlim(1e4, 3e6)
    # Headroom above the peak leaves the legend clear of the points.
    ax.set_ylim(0.1, 300)

    ax.set_xlabel(r"$E_{\mathrm{dep.}}~\left[\mathrm{GeV}\right]$")
    ax.set_ylabel(r"$N_{\mathrm{event}}$")

    ax.legend(loc="upper right", ncol=2, fontsize=9.5, handletextpad=0.5,
              columnspacing=1.0, borderaxespad=0.5)

    ax = axs[1]

    h, bins = np.histogram(ntwodeltallhs, bins=20)
    q = (bins[1:] + bins[:-1]) / 2

    lo1, hi1 = min_width_interval(ntwodeltallhs, 0.68)
    lo2, hi2 = min_width_interval(ntwodeltallhs, 0.95)
    lo3, hi3 = min_width_interval(ntwodeltallhs, 0.997)

    # Draw widest band first so the nested intervals darken toward the core.
    band_color = to_rgb(COLORS[15])
    for (lo, hi), alpha, lbl in (
        ((lo3, hi3), 0.20, r"$99.7\%$"),
        ((lo2, hi2), 0.30, r"$95\%$"),
        ((lo1, hi1), 0.45, r"$68\%$"),
    ):
        xs = np.linspace(lo, hi, 1000)
        ys = np.array([int(np.argmax(bins - x > 0) - 1) for x in xs])
        ax.fill_between(xs, 0, h[ys], step="mid",
                        facecolor=band_color + (alpha,), edgecolor="none",
                        label=lbl)

    ax.step(q, h, where="mid", color=band_color, lw=1.2)

    ax.axvline(ntwodeltallhdata, color=COLORS[1], label="HESE (2021)", lw=2)
    ax.axvline(ntwodeltallhsample, color=COLORS[14], label="Sampled", lw=2,
               linestyle="--")

    ax.set_ylim(0, 1.35 * h.max())
    ax.set_xlim(max(q[0], lo3), min(q[-1], hi3))

    ax.legend(loc="upper right", fontsize=9.5, handletextpad=0.5,
              labelspacing=0.35, borderaxespad=0.5)

    ax.set_ylabel(r"$N_{\mathrm{sample}}$")
    ax.set_xlabel(r"$-2\left[\mathrm{LLH}(\mu \mid n)-\mathrm{LLH}(\mu \mid \mu)\right]$")

    fig.savefig(os.path.join(OUTDIR, "hese_figure.pdf"))
    plt.close(fig)

def ps_track_roundtrip(datafile):
    with h5.File(datafile) as h5f:
        gp        = h5f["figure_5"]
        sd_cents  = gp["sd_cents"][:]
        h_atmo    = gp["h_atmo"][:]
        h_astro   = gp["h_astro"][:]
        h_data    = gp["h_data"][:]
        e_cents   = gp["e_cents"][:]
        h_atmo_e  = gp["h_atmo_e"][:]
        h_astro_e = gp["h_astro_e"][:]
        h_data_e  = gp["h_data_e"][:]

    fig, axs = plt.subplots(
        2, 2,
        figsize=(1.8 * COLUMN_WIDTH, 0.85 * COLUMN_WIDTH),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06, "wspace": 0.22},
        sharex="col",
    )

    # The two ratio panels need different scales: the declination residuals sit
    # within a few percent of unity, the energy-proxy ones span a decade.
    panels = (
        (axs[0, 0], axs[1, 0], sd_cents, h_atmo,   h_astro,   h_data,
         r"$\sin\delta$", None, (0.7, 1.3), None),
        (axs[0, 1], axs[1, 1], e_cents,  h_atmo_e, h_astro_e, h_data_e,
         r"$E_{\mathrm{proxy}}~\left[\mathrm{GeV}\right]$", "log", (0.05, 5.0), "log"),
    )

    for (ax_main, ax_ratio, x, atmo, astro, data, xlabel, xscale,
         ratio_ylim, ratio_yscale) in panels:
        pred = atmo + astro
        # Guard against empty template bins in the sparsely populated tails.
        good = pred > 0
        ratio     = np.full(pred.shape, np.nan)
        ratio_err = np.full(pred.shape, np.nan)
        ratio[good]     = data[good] / pred[good]
        ratio_err[good] = np.sqrt(data[good]) / pred[good]

        ax_main.errorbar(
            x, data, yerr=np.sqrt(data),
            label="Observed", color="k", linestyle="none",
            marker="o", markersize=3, zorder=5,
        )
        ax_main.fill_between(
            x, astro, pred,
            step="mid",
            label="Atmospheric (MCEq GSF)",
            facecolor=to_rgb("lightskyblue") + (0.5,),
            edgecolor=to_rgb("lightskyblue") + (1.0,),
        )
        ax_main.fill_between(
            x, 1, astro,
            step="mid",
            label="Astrophysical",
            facecolor=to_rgb(COLORS[-1]) + (0.5,),
            edgecolor=to_rgb(COLORS[-1]) + (1.0,),
        )

        ax_main.set_yscale("log")
        ax_main.set_ylim(1, 10 * max(h_data.max(), h_data_e.max()))
        ax_main.tick_params(axis="x", labelbottom=False)

        ax_ratio.errorbar(
            x, ratio, yerr=ratio_err,
            color="k", linestyle="none", marker="o", markersize=3,
        )
        ax_ratio.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        if ratio_yscale is not None:
            ax_ratio.set_yscale(ratio_yscale)
        ax_ratio.set_ylim(*ratio_ylim)
        ax_ratio.set_xlabel(xlabel)
        ax_ratio.set_ylabel(r"Obs./Pred.")

        if xscale is not None:
            ax_main.set_xscale(xscale)
            ax_ratio.set_xscale(xscale)

    axs[0, 0].set_xlim(0, 1)
    # Above ~2e5 GeV the northern-sky sample is empty in both data and template.
    axs[0, 1].set_xlim(1e2, 2e5)

    # Same counts axis on both mains, so only the left one carries the labels.
    axs[0, 1].tick_params(axis="y", labelleft=False)

    # Decade ticks alone are too sparse on the short log residual panel.
    axs[1, 1].set_yticks([0.1, 0.3, 1.0, 3.0])
    axs[1, 1].set_yticklabels(["0.1", "0.3", "1", "3"])
    axs[1, 1].minorticks_off()

    axs[0, 0].legend(loc="upper left", fontsize=9.5)
    axs[0, 0].set_ylabel(r"$N_{\mathrm{event}}$")

    fig.savefig(os.path.join(OUTDIR, "10yr_ps_data_vs_sampled.pdf"))
    plt.close(fig)

def ps_track_eddington(datafile):
    with h5.File(datafile) as h5f:
        gp = h5f["figure_5_eddington"]
        e_cents           = gp["e_cents"][:]
        h_data            = gp["h_data"][:]
        h_atmos_smeared   = gp["h_atmos_smeared"][:]    # (n_bins, n_models)
        h_atmos_unsmeared = gp["h_atmos_unsmeared"][:]
        h_astro_smeared   = gp["h_astro_smeared"][:]
        h_astro_unsmeared = gp["h_astro_unsmeared"][:]

    fig, axs = plt.subplots(1, 2, figsize=(1.8 * COLUMN_WIDTH, 1.8 * 9 / 16 / 2 * COLUMN_WIDTH))

    for ax, (h_atmo, h_astro, title) in zip(axs, [
        (h_atmos_unsmeared, h_astro_unsmeared, "Unsmeared ($E_\\mathrm{reco} = E_\\mathrm{true}$)"),
        (h_atmos_smeared,   h_astro_smeared,   "Smeared (full IRF)"),
    ]):
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.errorbar(
            e_cents, h_data, yerr=np.sqrt(h_data),
            label="Observed", color="k", linestyle="none", marker="o",
        )

        ax.fill_between(
            e_cents,
            h_astro,
            h_atmo.min(axis=1) + h_astro,
            step="mid",
            label="Atmospheric",
            facecolor=to_rgb("lightskyblue") + (0.5,),
            edgecolor=to_rgb("lightskyblue")  + (1.0,),
        )
        ax.fill_between(
            e_cents,
            h_atmo.min(axis=1) + h_astro,
            h_atmo.max(axis=1) + h_astro,
            step="mid",
            facecolor=to_rgb("lightskyblue") + (0.5,),
            edgecolor="none",
        )
        ax.fill_between(
            e_cents,
            h_atmo.min(axis=1) + h_astro,
            h_atmo.max(axis=1) + h_astro,
            step="mid",
            facecolor="none",
            edgecolor=to_rgb("lightskyblue") + (1.0,),
            hatch="\\",
        )
        ax.fill_between(
            e_cents, 1, h_astro,
            step="mid",
            label="Astrophysical",
            facecolor=to_rgb(COLORS[-1]) + (0.5,),
            edgecolor=to_rgb(COLORS[-1]) + (1.0,),
        )

        ax.legend()
        ax.set_ylim(1, None)
        ax.set_xlim(100, 1e6)
        ax.set_xlabel(r"$E_{\mathrm{proxy}}~\left[\mathrm{GeV}\right]$")
        ax.set_ylabel(r"$N_{\mathrm{event}}$")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "10yr_ps_eddington.pdf"))


def multidetector_skymap(datafile):
    with h5.File(datafile) as h5f:
        gp = h5f["multidetector_skymap"]
        ic_inst_ra      = gp["ic_instant_ra"][:]
        ic_inst_sindec  = gp["ic_instant_sindec"][:]
        km3_inst_ra     = gp["km3_instant_ra"][:]
        km3_inst_sindec = gp["km3_instant_sindec"][:]
        ic_st_ra        = gp["ic_steady_ra"][:]
        ic_st_sindec    = gp["ic_steady_sindec"][:]
        km3_st_ra       = gp["km3_steady_ra"][:]
        km3_st_sindec   = gp["km3_steady_sindec"][:]

    # Taller panels than the old 16:9 strip: sin(dec) is the axis carrying the
    # message, and squashing it hid the Mediterranean band structure.
    fig, axs = plt.subplots(
        1, 2,
        figsize=(1.8 * COLUMN_WIDTH, 0.78 * COLUMN_WIDTH),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    marker_kw = dict(s=14, alpha=0.55, linewidths=0.4, edgecolors="white")

    panels = (
        (axs[0], ic_inst_ra, ic_inst_sindec, km3_inst_ra, km3_inst_sindec, "Instantaneous"),
        (axs[1], ic_st_ra,   ic_st_sindec,   km3_st_ra,   km3_st_sindec,   "Diurnal average"),
    )
    for ax, ic_ra, ic_sd, km3_ra, km3_sd, title in panels:
        ax.scatter(ic_ra,  ic_sd,  color=blues(0.85), label="South Polar",   **marker_kw)
        ax.scatter(km3_ra, km3_sd, color=reds(0.85),  label="Mediterranean", **marker_kw)
        ax.set_xlabel(r"$\alpha~\left[\mathrm{rad}\right]$")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1, 1)
        ax.set_title(title, fontsize=11)

    axs[0].set_ylabel(r"$\sin(\delta)$")
    # One legend above the pair, clear of the points entirely.
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06), markerscale=1.6)

    fig.savefig(os.path.join(OUTDIR, "multidetector_samples.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def effective_area(datafile):
    with h5.File(datafile) as h5f:
        gp       = h5f["effective_area"]
        effas    = gp["effas"][:]       # shape (2, n_dec, n_e)
        es       = gp["es"][:]
        decs_rad = gp["decs_rad"][:]

    xmin = 0.3

    # One panel per detector: overlaying ten curves from two colour ramps in a
    # single axes forced a legend whose grey declination swatches matched none
    # of the plotted colours.  Split by detector, and let each panel's own ramp
    # carry the declination.
    fig, axs = plt.subplots(
        1, 2,
        figsize=(1.8 * COLUMN_WIDTH, 0.66 * COLUMN_WIDTH),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    for ax, i_det, cmap, title in (
        (axs[0], 0, blues, "South Polar"),
        (axs[1], 1, reds,  "Mediterranean"),
    ):
        handles = []
        for idx in range(len(decs_rad)):
            x = ((len(decs_rad) - 1) - idx) * (1 - xmin) / (len(decs_rad) - 1) + xmin
            ax.step(es, effas[i_det, idx, :], where="mid", color=cmap(x), lw=2)
            handles.append(Line2D(
                [], [], color=cmap(x),
                label=r"$\delta=%d^{\circ}$" % np.degrees(decs_rad[idx]),
            ))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(es[0], es[-1])
        ax.set_ylim(10, None)
        ax.set_xlabel(r"$E~\left[\mathrm{GeV}\right]$")
        ax.set_title(title, fontsize=11)
        # Lower right is the only region the curves leave free.
        ax.legend(handles=handles, loc="lower right", frameon=False,
                  fontsize=9.5, labelspacing=0.3, handlelength=1.4)

    axs[0].set_ylabel(r"$A_{\mathrm{eff}}~\left[\mathrm{cm}^{2}\right]$")

    fig.savefig(os.path.join(OUTDIR, "effective_area.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def point_source_demo(datafile):
    with h5.File(datafile) as h5f:
        gp           = h5f["point_source_demo"]
        psi2_cents   = gp["psi2_cents"][:]
        ngc_template = gp["ngc_template"][:]
        bg_template  = gp["bg_template"][:]
        h_data       = gp["h_data"][:]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 9 / 16 * COLUMN_WIDTH))

    ax.fill_between(
        psi2_cents,
        bg_template,
        bg_template + ngc_template,
        step="mid",
        edgecolor=COLORS[-1],
        facecolor=to_rgb(COLORS[-1]) + (0.3,),
        label="Signal",
    )
    ax.fill_between(
        psi2_cents,
        np.zeros(bg_template.shape),
        bg_template,
        step="mid",
        edgecolor="lightskyblue",
        facecolor=to_rgb("lightskyblue") + (0.3,),
        label="Background",
    )
    ax.errorbar(
        psi2_cents, h_data, yerr=np.sqrt(h_data),
        marker="o", markersize=4, elinewidth=1.2, capsize=2,
        linestyle="none", color="black", label="Sampled",
    )

    ax.set_xlim(0, 5)
    # Headroom so the legend clears the on-source bins.
    ax.set_ylim(0, 1.45 * np.max(h_data[psi2_cents < 5]))
    ax.set_xlabel(r"$\Delta\psi^{2}~\left[\deg^{2}\right]$")
    ax.set_ylabel(r"$N_{\mathrm{event}}$")
    ax.legend(frameon=True, loc=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "point_source_demo.pdf"))


def main(args=None):
    global OUTDIR
    if args is None:
        args = parse_args()
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    datafile = args.datafile
    hese_comparison(datafile)
    ps_track_roundtrip(datafile)
    multidetector_skymap(datafile)
    effective_area(datafile)
    point_source_demo(datafile)

if __name__=="__main__":
    main()
    
