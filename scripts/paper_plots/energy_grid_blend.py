"""Adaptive energy-grid blending figure for the SPORE paper.

Produces ``energy_grid_blend.pdf``, which shows how the extended-source sampler
places its energy grid and what the two extremes of the blend cost.

The construction illustrated is the one in ``build_adaptive_log_energy_grid``.
The sampler needs ``n_E`` cell centres spanning the energy range, and takes them
at equal quantiles of

    F = (1 - f) F_w + f F_u,

where ``F_w`` is the cumulative distribution of the target and ``F_u`` is
uniform in ln E.  ``f = 0`` is pure equal-mass placement, ``f = 1`` a uniform
log grid, and the package default is ``f = 0.5``.

The target here is a generic log parabola rather than any particular instrument
response and flux: the behaviour being illustrated is a property of the grid
construction, not of a specific detector, and a synthetic target keeps the
figure self-contained and free of data dependencies.  It is chosen to fall
steeply enough at high energy to expose the ``f = 0`` failure, which is the
regime that motivates the blend.  The grid itself is built by calling the
package routine, so the figure exercises the shipped code path.

Layout
------
The lower panel is the construction: the three cumulative distributions,
and one rug of ticks per case beneath.  The ticks are the cell centres, placed
at equally spaced quantiles of the corresponding curve, so where a curve is
steep they bunch together and where it is flat they spread apart.

The upper panel is the consequence: the target against the distribution each
grid actually samples from.  That distribution is not approximated.  The
sampler weights cell ``i`` by the target at its centre and draws ln E uniformly
inside the cell, so the sampled density is exactly piecewise constant, equal to
the target at each cell centre across the width of that cell.  It is drawn as
such, with no Monte Carlo and hence no sampling noise.

Usage
-----
    python scripts/paper_plots/energy_grid_blend.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from spore.event_sampling.utils import build_adaptive_log_energy_grid

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(HERE, "..", "..")
STYLE = os.path.join(REPO, "resources", "paper.mplstyle")
OUTDIR = os.path.join(REPO, "paper", "figures")

# The package default, and what the upper panel shows.
N_E = 40

# Synthetic target, chosen to expose both failure modes at once: a narrow line
# on a broad continuum, of the kind a monoenergetic decay channel produces, and
# a steep fall above it.  The line is narrower than a uniform cell, so a grid
# spaced uniformly in ln E cannot resolve it; the steep fall exhausts the
# probability mass, so equal-mass placement runs out of resolution at the top of
# the range.
E_MIN, E_MAX = 1e2, 1e6
LOG10_E_CONTINUUM_PEAK = 3.0
CURVATURE = 0.55
LOG10_E_LINE = 3.5
LINE_WIDTH = 0.02           # decades; about a fifth of the uniform cell width
LINE_AMPLITUDE = 5.0

# (blend fraction, colour, linestyle)
CASES = (
    (0.0, "C0", "-"),
    (0.5, "k", "-"),
    (1.0, "C1", "--"),
)

LONG = {0.0: r"$f = 0$  (equal mass)",
        0.5: r"$f = 0.5$  (default)",
        1.0: r"$f = 1$  (uniform in $\ln E$)"}
SHORT = {0.0: r"$f = 0$", 0.5: r"$f = 0.5$", 1.0: r"$f = 1$"}


def target_density(e):
    """Target PDF per ln E, up to normalisation: a narrow line on a continuum."""
    x = np.log10(np.asarray(e, float))
    continuum = 10.0 ** (-CURVATURE * (x - LOG10_E_CONTINUUM_PEAK) ** 2)
    line = LINE_AMPLITUDE * np.exp(-0.5 * ((x - LOG10_E_LINE) / LINE_WIDTH) ** 2)
    return continuum + line


class _SyntheticSource:
    """Stands in for a source so the package grid routine can be used directly.

    ``build_adaptive_log_energy_grid`` forms its pilot weight as
    ``A_eff * flux * E``.  Pairing a unit effective area with a flux of
    ``target / E``, split across the two track species, makes that weight equal
    to ``target``.
    """

    def __call__(self, nu, e, dec, ra):
        e = np.asarray(e, float)
        return 0.5 * target_density(e) / e


def _unit_effective_area(zen, e):
    return np.ones_like(np.asarray(e, float))


def cell_edges(log_es):
    """Cell edges from centres, as ExtendedSourceEventSampler builds them."""
    return np.concatenate([[log_es[0]],
                           0.5 * (log_es[:-1] + log_es[1:]),
                           [log_es[-1]]])


def sampled_density(log_es, log_pilot, w_pilot):
    """The density the sampler draws from, in ln E, normalised to unit integral.

    Cell ``i`` carries probability ``w(centre_i) * width_i`` and is filled
    uniformly in ln E, so the density inside it is ``w(centre_i)``.
    """
    edges = cell_edges(log_es)
    w_centre = np.interp(log_es, log_pilot, w_pilot)
    widths = np.diff(edges)
    return edges, w_centre / float(np.sum(w_centre * widths))


def main():
    plt.style.use(STYLE)
    src = _SyntheticSource()

    log_pilot = np.linspace(np.log(E_MIN), np.log(E_MAX), 600)
    es_pilot = np.exp(log_pilot)
    w_pilot = target_density(es_pilot)

    # Cumulative distributions, exactly as build_adaptive_log_energy_grid forms them.
    cum_w = np.cumsum(w_pilot * np.gradient(log_pilot))
    cum_w /= cum_w[-1]
    cum_u = (log_pilot - log_pilot[0]) / (log_pilot[-1] - log_pilot[0])

    quantiles = np.linspace(0.0, 1.0, N_E)
    w_norm = w_pilot / np.trapezoid(w_pilot, log_pilot)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.6, 4.9), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.08},
    )

    ax_top.plot(es_pilot, w_norm, color="0.4", lw=2.4, zorder=1,
                label="target PDF")

    rug_y0, rug_dy = -0.085, 0.072
    for idx, (f, colour, ls) in enumerate(CASES):
        log_es = build_adaptive_log_energy_grid(
            N_E, _unit_effective_area, src, "track",
            e_min=E_MIN, e_max=E_MAX, uniform_fraction=f,
        )
        edges, dens = sampled_density(log_es, log_pilot, w_pilot)
        ax_top.stairs(dens, np.exp(edges), color=colour, lw=1.7,
                      linestyle=ls, zorder=3, label=LONG[f])

        cum = (1.0 - f) * cum_w + f * cum_u
        ax_bot.plot(es_pilot, cum, color=colour, lw=1.9, linestyle=ls, zorder=3)
        y = rug_y0 - idx * rug_dy
        ax_bot.plot(np.exp(log_es), np.full(N_E, y), "|", color=colour,
                    ms=7, mew=1.0, clip_on=False, zorder=4)
        ax_bot.annotate(SHORT[f], xy=(1.008, y),
                        xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=7.5, color=colour,
                        annotation_clip=False)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_xlim(E_MIN, E_MAX)
    ax_top.set_ylim(2e-6 * w_norm.max(), 5.0 * w_norm.max())
    ax_top.set_ylabel(r"density per $\ln E$")
    ax_top.legend(loc="lower left", fontsize=8, handletextpad=0.6,
                  labelspacing=0.35)

    ax_bot.set_xscale("log")
    ax_bot.set_xlim(E_MIN, E_MAX)
    ax_bot.set_ylim(-0.28, 1.03)
    ax_bot.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bot.set_xlabel(r"$E$  [arb.]")
    ax_bot.set_ylabel("cumulative probability")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "energy_grid_blend.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
