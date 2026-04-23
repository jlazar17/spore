"""Run this script to regenerate spore_quickstart.ipynb."""
import json
from pathlib import Path


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip().splitlines(keepends=True),
    }


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip().splitlines(keepends=True),
    }


cells = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(md(
    "# SPORE quickstart\n\n"
    "This notebook walks through the core workflow:\n\n"
    "1. Loading a detector IRF and inspecting the effective area\n"
    "2. Point source simulation (steady-state mode)\n"
    "3. Extended source simulation\n"
    "4. Sanity checks: Poisson statistics, Eddington bias, angular resolution vs energy\n\n"
    "All paths are resolved relative to the repository root, so the notebook "
    "can be run from any working directory."
))

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
cells.append(md("## 1  Setup"))

cells.append(code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import h5py\n"
    "import tempfile\n"
    "from pathlib import Path\n"
    "\n"
    "import spore\n"
    "from spore import (\n"
    "    Detector, PointSource, ExtendedSource,\n"
    "    SourceSampler, SkyCoordinate, ureg,\n"
    ")\n"
    "\n"
    "# Locate the repository root (works regardless of working directory)\n"
    "_p = Path.cwd()\n"
    "while not (_p / 'pyproject.toml').exists() and _p != _p.parent:\n"
    "    _p = _p.parent\n"
    "REPO_ROOT = _p\n"
    "RESPONSE_FILE = str(REPO_ROOT / 'resources' / 'configs' / 'ps10yr_detector_response.h5')\n"
    "print('repo root   :', REPO_ROOT)\n"
    "print('response exists:', Path(RESPONSE_FILE).exists())\n"
))

# ---------------------------------------------------------------------------
# 2. Detector
# ---------------------------------------------------------------------------
cells.append(md(
    "## 2  Loading a detector\n\n"
    "`Detector.from_config` accepts a plain dictionary.  "
    "We use the IceCube 10-year point-source sample response built by "
    "`scripts/build_ps10yr_detector_response.py`."
))

cells.append(code(
    "det = Detector.from_config({\n"
    "    'properties': {'latitude': -90.0, 'longitude': 0.0, 'medium': 'Ice'},\n"
    "    'response':   {'detector_response_file': RESPONSE_FILE},\n"
    "})\n"
    "\n"
    "print('Location  : lat={:.1f} deg, lon={:.1f} deg'.format(\n"
    "    np.degrees(det.location.latitude), np.degrees(det.location.longitude)))\n"
    "print('Morphologies:', det.response.available_morphologies)\n"
    "effa = det.response.effective_area['track']\n"
    "print('A_eff range : {:.0f} GeV -- {:.2e} GeV'.format(\n"
    "    effa.e_min_gev, effa.e_max_gev))\n"
))

cells.append(md("### 2.1  Effective area vs energy"))

cells.append(code(
    "effa = det.response.effective_area['track']\n"
    "energies = np.logspace(np.log10(effa.e_min_gev), np.log10(effa.e_max_gev), 200)\n"
    "\n"
    "zenith_cases = [\n"
    "    (np.pi * 0.6, 'zen=108 deg (upgoing)'),\n"
    "    (np.pi * 0.7, 'zen=126 deg'),\n"
    "    (np.pi * 0.8, 'zen=144 deg'),\n"
    "    (np.pi * 0.9, 'zen=162 deg'),\n"
    "]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "for zen, label in zenith_cases:\n"
    "    vals = effa(np.full(len(energies), zen), energies)\n"
    "    ax.plot(energies / 1e3, vals, label=label)\n"
    "\n"
    "ax.set_xscale('log')\n"
    "ax.set_yscale('log')\n"
    "ax.set_xlabel('Energy [TeV]')\n"
    "ax.set_ylabel('A_eff [cm^2]')\n"
    "ax.set_title('Track effective area -- IceCube PS-10yr')\n"
    "ax.legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

cells.append(md(
    "### 2.2  PSF median angular resolution vs energy\n\n"
    "When the detector response stores a standalone `angular_response` callable, "
    "`angular_response(E, 0.5)` gives the median PSF deflection at energy E.  "
    "The IceCube PS-10yr response uses a joint smearing table instead, so no "
    "standalone angular-response callable is available here; we skip this plot "
    "and demonstrate angular resolution empirically in section 5.2."
))

cells.append(code(
    "ang = det.response.angular_response['track']\n"
    "if ang is not None:\n"
    "    e_grid = np.logspace(np.log10(effa.e_min_gev), np.log10(effa.e_max_gev), 80)\n"
    "    median_psf_deg = np.degrees([ang(e, 0.5) for e in e_grid])\n"
    "\n"
    "    fig, ax = plt.subplots(figsize=(7, 4))\n"
    "    ax.plot(e_grid / 1e3, median_psf_deg, color='steelblue')\n"
    "    ax.set_xscale('log')\n"
    "    ax.set_xlabel('Energy [TeV]')\n"
    "    ax.set_ylabel('Median PSF [deg]')\n"
    "    ax.set_title('Track angular resolution (median) -- IceCube PS-10yr')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "else:\n"
    "    print('angular_response not available for this detector response '\n"
    "          '(joint smearing IRF); see section 5.2 for empirical PSF.')\n"
))

# ---------------------------------------------------------------------------
# 3. Point source
# ---------------------------------------------------------------------------
cells.append(md(
    "## 3  Point source simulation\n\n"
    "We simulate TXS 0506+056 (dec +5.7 deg), the first IceCube blazar candidate, "
    "with a soft E^-2.7 power-law spectrum.  "
    "`n_time_samples=100` enables steady-state mode: the effective area is averaged "
    "over 100 hour-angle samples uniformly spaced over one diurnal cycle.  "
    "This is the correct treatment for observations spanning many sidereal days."
))

cells.append(code(
    "src = PointSource.from_config({\n"
    "    'flux': {\n"
    "        'norm':  1e-15,   # GeV^-1 cm^-2 s^-1 per species\n"
    "        'gamma': 2.7,\n"
    "        'pivot': 1e3,     # 1 TeV\n"
    "        'emin':  1e2,\n"
    "        'emax':  1e7,\n"
    "    },\n"
    "    'location': {'declination': 5.7, 'right_ascension': 77.4},\n"
    "})\n"
    "print('Source: dec={:.2f} deg, RA={:.2f} deg'.format(\n"
    "    np.degrees(src.location.declination),\n"
    "    np.degrees(src.location.right_ascension)))\n"
))

cells.append(code(
    "sampler_ps = SourceSampler(det, src, n_time_samples=100)\n"
    "\n"
    "one_year = ureg.Quantity(365.25, 'day')\n"
    "n_exp = sampler_ps.expected_events('track', one_year)\n"
    "print(f'Expected track events in one year: {n_exp:.2f}')\n"
))

cells.append(md("### 3.1  Sample one year of events (Poisson mode)"))

cells.append(code(
    "events_ps = sampler_ps.sample_events('track', deltat=one_year, seed=1)\n"
    "print(f'Sampled {len(events_ps)} events')\n"
    "\n"
    "true_e_tev  = np.array([ev.true_energy.to('TeV').magnitude  for ev in events_ps])\n"
    "reco_e_tev  = np.array([ev.reco_energy.to('TeV').magnitude  for ev in events_ps])\n"
    "reco_decs   = np.degrees([ev.reco_direction.declination      for ev in events_ps])\n"
    "reco_ras    = np.degrees([ev.reco_direction.right_ascension  for ev in events_ps])\n"
    "ang_seps    = np.degrees([ev.true_direction.separation(ev.reco_direction)\n"
    "                          for ev in events_ps])\n"
))

cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
    "src_dec = np.degrees(src.location.declination)\n"
    "src_ra  = np.degrees(src.location.right_ascension)\n"
    "\n"
    "ax = axes[0]\n"
    "ax.scatter(reco_ras, reco_decs, s=12, alpha=0.6, color='steelblue', label='Reco')\n"
    "ax.scatter([src_ra], [src_dec], s=200, marker='*', color='gold',\n"
    "           zorder=5, label='True position', edgecolors='k', linewidths=0.5)\n"
    "ax.set_xlabel('Right ascension [deg]')\n"
    "ax.set_ylabel('Declination [deg]')\n"
    "ax.set_title('Reconstructed directions')\n"
    "ax.legend(fontsize=9)\n"
    "\n"
    "ax = axes[1]\n"
    "bins = np.logspace(-1, 4, 30)\n"
    "ax.hist(reco_e_tev, bins=bins, color='steelblue', edgecolor='white', lw=0.5)\n"
    "ax.set_xscale('log')\n"
    "ax.set_xlabel('Reconstructed energy [TeV]')\n"
    "ax.set_ylabel('Events / bin')\n"
    "ax.set_title('Reco energy spectrum (E^-2.7)')\n"
    "\n"
    "ax = axes[2]\n"
    "ax.hist(ang_seps, bins=25, color='steelblue', edgecolor='white', lw=0.5)\n"
    "ax.set_xlabel('Angular separation (true to reco) [deg]')\n"
    "ax.set_ylabel('Events / bin')\n"
    "ax.set_title('PSF spread')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

cells.append(md(
    "### 3.2  True vs reconstructed energy\n\n"
    "For steeply falling spectra the energy resolution causes more events to scatter "
    "*upward* in energy than downward (Eddington bias).  "
    "We can see this directly by comparing E_true and E_reco for the same events."
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "sc = ax.scatter(true_e_tev, reco_e_tev, s=10, alpha=0.5,\n"
    "                c=np.log10(true_e_tev), cmap='viridis')\n"
    "lim = [min(true_e_tev.min(), reco_e_tev.min()) * 0.5,\n"
    "       max(true_e_tev.max(), reco_e_tev.max()) * 2.0]\n"
    "ax.plot(lim, lim, 'k--', lw=1, label='E_reco = E_true')\n"
    "ax.set_xscale('log'); ax.set_yscale('log')\n"
    "ax.set_xlim(lim); ax.set_ylim(lim)\n"
    "ax.set_xlabel('True energy [TeV]')\n"
    "ax.set_ylabel('Reconstructed energy [TeV]')\n"
    "ax.set_title('Energy smearing')\n"
    "plt.colorbar(sc, ax=ax, label='log10(E_true / TeV)')\n"
    "ax.legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ---------------------------------------------------------------------------
# 4. Extended source
# ---------------------------------------------------------------------------
cells.append(md(
    "## 4  Extended source simulation\n\n"
    "We model an isotropic diffuse E^-2 astrophysical flux.  "
    "The flux is written to a temporary HDF5 file and loaded via "
    "`ExtendedSource.from_config`.  "
    "The sampler runs in steady-state mode (diurnal-average A_eff)."
))

cells.append(code(
    "N_DEC, N_E  = 30, 40\n"
    "sindecs     = np.linspace(-1.0, 1.0, N_DEC)\n"
    "energies    = np.logspace(2.0, 6.0, N_E)   # 100 GeV -- 1 PeV\n"
    "\n"
    "PHI_0 = 1e-18   # GeV^-1 cm^-2 s^-1 sr^-1 per species\n"
    "phi   = PHI_0 * (energies / 1e5) ** (-2.0)\n"
    "\n"
    "fluxes      = np.zeros((6, N_DEC, N_E))\n"
    "fluxes[2]   = phi[np.newaxis, :]   # nu_mu\n"
    "fluxes[3]   = phi[np.newaxis, :]   # nu_mu_bar\n"
    "\n"
    "_tmpdir   = tempfile.mkdtemp()\n"
    "FLUX_FILE = str(Path(_tmpdir) / 'isotropic_flux.h5')\n"
    "\n"
    "with h5py.File(FLUX_FILE, 'w') as f:\n"
    "    grp = f.create_group('flux')\n"
    "    grp.create_dataset('sindecs',  data=sindecs)\n"
    "    grp.create_dataset('energies', data=energies)\n"
    "    grp.create_dataset('fluxes',   data=fluxes)\n"
    "\n"
    "print('Flux file written to', FLUX_FILE)\n"
))

cells.append(code(
    "src_ext = ExtendedSource.from_config({'flux': {'location': f'{FLUX_FILE}:flux'}})\n"
    "\n"
    "print('Building extended-source sampler (steady-state)...')\n"
    "sampler_ext = SourceSampler(\n"
    "    det, src_ext,\n"
    "    n_dec=30, n_ra=30, n_e=30,\n"
    "    n_time_samples=100,\n"
    ")\n"
    "\n"
    "n_exp_ext = sampler_ext.expected_events('track', one_year)\n"
    "print(f'Expected track events in one year: {n_exp_ext:.1f}')\n"
))

cells.append(code(
    "events_ext  = sampler_ext.sample_events('track', deltat=one_year, seed=2)\n"
    "print(f'Sampled {len(events_ext)} events')\n"
    "\n"
    "ext_decs   = np.degrees([ev.true_direction.declination     for ev in events_ext])\n"
    "ext_log10e = [np.log10(ev.reco_energy.to('GeV').magnitude) for ev in events_ext]\n"
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "fig.suptitle('Extended source (isotropic E^-2) -- IceCube, 1 year, steady-state')\n"
    "\n"
    "ax = axes[0]\n"
    "ax.hist(ext_decs, bins=np.linspace(-90, 90, 37),\n"
    "        color='steelblue', edgecolor='white', lw=0.5)\n"
    "ax.set_xlabel('True declination [deg]')\n"
    "ax.set_ylabel('Events / 5 deg bin')\n"
    "ax.set_title('Declination distribution\\n(reflects sky-averaged A_eff)')\n"
    "\n"
    "ax = axes[1]\n"
    "ax.hist(ext_log10e, bins=25, color='steelblue', edgecolor='white', lw=0.5)\n"
    "ax.set_xlabel('log10(E_reco / GeV)')\n"
    "ax.set_ylabel('Events / bin')\n"
    "ax.set_title('Reconstructed energy spectrum')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ---------------------------------------------------------------------------
# 5. Sanity checks
# ---------------------------------------------------------------------------
cells.append(md(
    "## 5  Sanity checks\n\n"
    "### 5.1  Poisson statistics\n\n"
    "Run 300 pseudo-experiments and check that the event count distribution "
    "matches Poisson(expected)."
))

cells.append(code(
    "N_PSEUDO = 300\n"
    "counts = np.array([\n"
    "    len(sampler_ps.sample_events('track', deltat=one_year, seed=i))\n"
    "    for i in range(N_PSEUDO)\n"
    "])\n"
    "\n"
    "from scipy.stats import poisson as scipy_poisson\n"
    "k = np.arange(max(0, counts.min() - 3), counts.max() + 4)\n"
    "pmf = scipy_poisson.pmf(k, n_exp)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.hist(counts, bins=np.arange(counts.min() - 0.5, counts.max() + 1.5),\n"
    "        density=True, color='steelblue', edgecolor='white', lw=0.5, label='Sampled')\n"
    "ax.plot(k, pmf, 'o-', color='tomato', ms=4,\n"
    "        label=f'Poisson(mu={n_exp:.2f})')\n"
    "ax.axvline(counts.mean(), color='k', ls='--', lw=1,\n"
    "           label=f'Sample mean = {counts.mean():.2f}')\n"
    "ax.set_xlabel('Number of events')\n"
    "ax.set_ylabel('Probability')\n"
    "ax.set_title(f'Poisson sampling check ({N_PSEUDO} pseudo-experiments)')\n"
    "ax.legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print(f'Expected mean : {n_exp:.2f}')\n"
    "print(f'Sample mean   : {counts.mean():.2f}  (std {counts.std():.2f})')\n"
    "print(f'Poisson std   : {n_exp**0.5:.2f}')\n"
))

cells.append(md(
    "### 5.2  Angular resolution vs reconstructed energy\n\n"
    "The PSF should decrease with increasing energy. "
    "We sample a large fixed set and measure the median angular separation per energy bin."
))

cells.append(code(
    "events_large = sampler_ps.sample_events('track', nevent=2000, seed=3)\n"
    "\n"
    "e_reco = np.array([ev.reco_energy.to('TeV').magnitude for ev in events_large])\n"
    "sep    = np.degrees([ev.true_direction.separation(ev.reco_direction)\n"
    "                     for ev in events_large])\n"
    "\n"
    "e_bins  = np.logspace(-1, 4, 12)\n"
    "e_mids  = np.sqrt(e_bins[:-1] * e_bins[1:])\n"
    "med_sep = []\n"
    "for lo, hi in zip(e_bins[:-1], e_bins[1:]):\n"
    "    mask = (e_reco >= lo) & (e_reco < hi)\n"
    "    med_sep.append(np.median(sep[mask]) if mask.sum() > 0 else np.nan)\n"
    "med_sep = np.array(med_sep)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.plot(e_mids, med_sep, 'o-', color='steelblue', ms=5)\n"
    "ax.set_xscale('log')\n"
    "ax.set_xlabel('Reconstructed energy [TeV]')\n"
    "ax.set_ylabel('Median angular separation [deg]')\n"
    "ax.set_title('Angular resolution vs energy (2000 events, fixed count)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

cells.append(md(
    "### 5.3  Eddington bias correction\n\n"
    "`eddington_correction` estimates the ratio smeared/unsmeared per reco-energy bin "
    "over many pseudo-experiments.  "
    "For a soft spectrum + downward energy shift, the correction exceeds 1 at low "
    "energies and falls below 1 at high energies."
))

cells.append(code(
    "e_bins_gev = np.logspace(2, 7, 15)\n"
    "correction = sampler_ps.eddington_correction(\n"
    "    'track', e_bins_gev, n_pseudo=200, seed=10\n"
    ")\n"
    "\n"
    "e_mids_gev = np.sqrt(e_bins_gev[:-1] * e_bins_gev[1:])\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.plot(e_mids_gev / 1e3, correction, 'o-', color='steelblue', ms=5)\n"
    "ax.axhline(1.0, color='k', ls='--', lw=1, label='No correction')\n"
    "ax.set_xscale('log')\n"
    "ax.set_xlabel('Energy [TeV]')\n"
    "ax.set_ylabel('Smeared / unsmeared')\n"
    "ax.set_title('Eddington bias correction (200 pseudo-experiments)')\n"
    "ax.legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "spore_quickstart.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Written:", out)
