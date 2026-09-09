"""Compare the sampled NGC 1068 signal template against the published one.

Section 6.3 compares the \\psi^2 signal template produced by SPORE against
figure 2b of the IceCube NGC 1068 evidence paper (Science 378 (2022) 538,
arXiv:2211.09972).  That comparison used to be by eye.  This script makes it
quantitative by digitising the published figure from the arXiv source, which
ships it as a vector PDF, so the numbers come from the drawing coordinates
rather than from pixel-peeping a raster.

Method
------
1.  Download the arXiv e-print tarball and take ``fig2b.pdf``.
2.  Convert to SVG with ``pdftocairo`` and pull out the two filled step paths
    by their fill colours (blue = signal, orange = background).
3.  Calibrate the axes from the rendered tick marks rather than from the SVG
    stroke list, which contains additional strokes that are easy to mistake
    for ticks.  Three independent checks confirm the calibration: the axis
    bottom maps to 0, the left edge maps to \\psi^2 = 0, and the extracted
    background is flat in \\psi^2, as an isotropic background must be.
4.  Rebin the SPORE template onto the published binning and compare shapes.

Requires ``pdftocairo`` and ``pdftoppm`` (poppler) on PATH, and network access
on first run; the digitised values are cached alongside this script.

Usage
-----
    python scripts/paper_plots/ngc1068_reference_comparison.py
"""

import json
import os
import re
import subprocess
import tarfile
import urllib.request

import h5py as h5
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(HERE, "..", "..")
PLOTDATA = os.path.join(REPO, "resources", "plotting_data.h5")
CACHE = os.path.join(HERE, "ngc1068_fig2b_digitised.json")

ARXIV_ID = "2211.09972"
EPRINT_URL = f"https://arxiv.org/e-print/{ARXIV_ID}"

# Axis tick values on the published figure.
Y_TICK_VALUES = np.array([80.0, 60.0, 40.0, 20.0])

# Fill colours pdftocairo emits for the two filled histograms.
FILL_SIGNAL = r"0%, 39\.607239%, 74\.116516%"
FILL_BACKGROUND = r"89\.01825%, 44\.7052%, 13\.33313%"


def _fetch_figure(workdir):
    """Download the arXiv source and return the path to fig2b.pdf."""
    pdf = os.path.join(workdir, "fig2b.pdf")
    if os.path.exists(pdf):
        return pdf
    tgz = os.path.join(workdir, "eprint.tar.gz")
    if not os.path.exists(tgz):
        req = urllib.request.Request(
            EPRINT_URL, headers={"User-Agent": "spore-paper-figure-check"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tgz, "wb") as f:
            f.write(r.read())
    with tarfile.open(tgz) as t:
        t.extractall(workdir)
    if not os.path.exists(pdf):
        raise FileNotFoundError("fig2b.pdf not present in the arXiv source")
    return pdf


def _tick_positions(pdf, workdir, dpi=600):
    """Measure tick-mark positions, in PDF units, from a high-resolution render.

    The SVG stroke list also contains short horizontal strokes that are not
    ticks; reading the ticks off the raster avoids picking those up.
    """
    from matplotlib import image as mpimg
    stem = os.path.join(workdir, "hi")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-gray", pdf, stem],
                   check=True, capture_output=True)
    png = stem + "-1.png"
    im = mpimg.imread(png)
    im = im[..., 0] if im.ndim == 3 else im
    height, width = im.shape

    # Page is square in this figure; scale is px per PDF unit.
    scale = width / 242.599616

    def _runs(idx):
        groups, cur = [], [idx[0]]
        for v in idx[1:]:
            if v - cur[-1] <= 3:
                cur.append(v)
            else:
                groups.append(cur)
                cur = [v]
        groups.append(cur)
        return [(g[0] + g[-1]) / 2 / scale for g in groups]

    # y ticks: full-darkness columns immediately left of the axes spine.
    xa, xb = int(40.2 * scale), int(43.3 * scale)
    col = (im[:, xa:xb] < 0.5).sum(axis=1)
    y_ticks = _runs(np.where(col >= (xb - xa) * 0.8)[0])

    # x ticks: full-darkness rows immediately below the axes spine.
    ya, yb = int(197.2 * scale), int(200.2 * scale)
    row = (im[ya:yb, :] < 0.5).sum(axis=0)
    x_ticks = _runs(np.where(row >= (yb - ya) * 0.8)[0])
    return np.array(x_ticks), np.array(y_ticks)


def _data_points(pdf, workdir, fx, fy, dpi=600):
    """Read the black data markers off the raster render.

    Matplotlib reuses a single marker XObject, so only some markers survive as
    inline paths in the SVG; locating the discs in the raster recovers all of
    them.  Each marker is a filled circle about 4.5 PDF units across, while the
    error bars are roughly 1.2 units wide, so a window centred on the bin and
    a threshold on the dark run separates the two cleanly.
    """
    from matplotlib import image as mpimg
    im = mpimg.imread(os.path.join(workdir, "hi-1.png"))
    im = im[..., 0] if im.ndim == 3 else im
    scale = im.shape[1] / 242.599616
    dark = im < 0.45

    left, right, nbin = 43.445312, 235.011719, 15
    width = (right - left) / nbin
    centers = left + width * (np.arange(nbin) + 0.5)

    xs, ys = [], []
    for cd in centers:
        x0, x1 = int((cd - 2.6) * scale), int((cd + 2.6) * scale)
        counts = dark[:, x0:x1].sum(axis=1)
        rows = np.where(counts >= 0.75 * (x1 - x0))[0]
        rows = rows[(rows > int(43.5 * scale)) & (rows < int(196.4 * scale))]
        if len(rows) == 0:
            xs.append(float(fx(cd)))
            ys.append(float("nan"))
            continue
        groups, cur = [], [rows[0]]
        for r in rows[1:]:
            if r - cur[-1] <= 4:
                cur.append(r)
            else:
                groups.append(cur)
                cur = [r]
        groups.append(cur)
        g = max(groups, key=len)
        xs.append(float(fx(cd)))
        ys.append(float(fy((g[0] + g[-1]) / 2 / scale)))
    return {"centers": xs, "values": ys}


def digitise(workdir):
    """Return {'signal': .., 'background': ..} in data coordinates."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)

    pdf = _fetch_figure(workdir)
    svg = os.path.join(workdir, "fig2b.svg")
    subprocess.run(["pdftocairo", "-svg", pdf, svg], check=True, capture_output=True)
    s = open(svg).read()

    x_ticks, y_ticks = _tick_positions(pdf, workdir)
    # x ticks are the integers 0..4; y ticks descend 80, 60, 40, 20.
    fx = np.poly1d(np.polyfit(x_ticks, np.arange(len(x_ticks)), 1))
    fy = np.poly1d(np.polyfit(y_ticks, Y_TICK_VALUES[:len(y_ticks)], 1))

    checks = {
        "axis_bottom_maps_to": float(fy(196.679688)),
        "axis_left_maps_to": float(fx(43.445312)),
        "axis_right_maps_to": float(fx(235.011719)),
    }

    out = {"_provenance": {"arxiv": ARXIV_ID, "figure": "fig2b.pdf",
                           "checks": checks}}
    for name, frag in (("signal", FILL_SIGNAL), ("background", FILL_BACKGROUND)):
        m = re.search(r'<path fill-rule="nonzero" fill="rgb\(' + frag
                      + r'[^"]*"[^>]*?d="([^"]+)"', s)
        if m is None:
            raise RuntimeError(f"could not locate the {name} path in {svg}")
        pts = [(float(a), float(b))
               for a, b in re.findall(r"([\d.\-]+)\s+([\d.\-]+)", m.group(1))]
        xs = np.array([fx(p[0]) for p in pts])
        ys = np.array([fy(p[1]) for p in pts])
        # A filled step outline: the bin values are its horizontal top edges.
        vals = {}
        for i in range(len(pts) - 1):
            if abs(ys[i] - ys[i + 1]) < 1e-6 and abs(xs[i] - xs[i + 1]) > 1e-6 \
                    and ys[i] > 0.5:
                vals[round(0.5 * (xs[i] + xs[i + 1]), 4)] = ys[i]
        centers = sorted(vals)
        out[name] = {"centers": centers, "values": [vals[c] for c in centers]}

    out["data"] = _data_points(pdf, workdir, fx, fy)
    # Event counts must be integers; how close the extraction lands to them is
    # an independent check on the calibration.
    dv = np.asarray(out["data"]["values"], float)
    out["_provenance"]["checks"]["data_max_dev_from_integer"] = float(
        np.nanmax(np.abs(dv - np.round(dv))))

    with open(CACHE, "w") as f:
        json.dump(out, f, indent=1)
    return out


def _stats(values, edges):
    p = values / values.sum()
    cum = np.cumsum(p)
    return (float(np.interp(0.5, cum, edges[1:])),
            float(np.interp(0.68, cum, edges[1:])),
            float(p[0.5 * (edges[1:] + edges[:-1]) < 1.0].sum()))


def main():
    workdir = os.path.join(HERE, "_ngc1068_ref")
    os.makedirs(workdir, exist_ok=True)
    ref = digitise(workdir)

    print("calibration checks (should be ~0, ~0, ~4.94):")
    for k, v in ref["_provenance"]["checks"].items():
        print(f"  {k}: {v:+.4f}")

    c = np.array(ref["signal"]["centers"])
    v = np.array(ref["signal"]["values"])
    bg = np.array(ref["background"]["values"])
    w = c[1] - c[0]
    edges = np.append(c - w / 2, c[-1] + w / 2)
    print(f"\ndata markers land within "
          f"{ref['_provenance']['checks'].get('data_max_dev_from_integer', float('nan')):.3f}"
          f" of an integer -- counts must be integers, so this is an "
          f"independent calibration check")
    print(f"\nextracted background is flat to "
          f"{100 * bg.std() / bg.mean():.1f}% rms -- expected for an "
          f"isotropic background in psi^2")

    with h5.File(PLOTDATA) as f:
        g = f["point_source_demo"]
        sp_c = g["psi2_cents"][:]
        sp_v = g["ngc_template"][:]
        sp_b = g["psi2_bins"][:]

    dens = sp_v / np.diff(sp_b)
    sp_re = np.array([
        np.sum(dens * np.clip(np.minimum(sp_b[1:], hi) - np.maximum(sp_b[:-1], lo),
                              0, None))
        for lo, hi in zip(edges[:-1], edges[1:])
    ])

    print(f"\n{'':24s} {'total':>8s} {'median psi':>11s} {'68% psi^2':>10s} "
          f"{'frac<1':>7s}")
    for label, vals in (("IceCube 2022 fig 2b", v), ("SPORE / IceTracks-DR2", sp_re)):
        med, q68, f1 = _stats(vals, edges)
        print(f"{label:24s} {vals.sum():8.1f} {np.sqrt(med):10.3f}d "
              f"{q68:10.3f} {f1:7.3f}")

    pi, ps = v / v.sum(), sp_re / sp_re.sum()
    print(f"\nfirst-bin fraction: reference {pi[0]:.3f}, SPORE {ps[0]:.3f} "
          f"(ratio {ps[0] / pi[0]:.3f})")
    print(f"max |shape ratio - 1| over the 15 bins: "
          f"{np.max(np.abs(ps / pi - 1)):.2f}")


if __name__ == "__main__":
    main()
