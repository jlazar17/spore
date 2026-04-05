"""
Monochromatic (line) neutrino injection from the Galactic halo.

Demonstrates GalacticHaloEventSampler with a LineSpectrum — a delta-function
energy spectrum at E_0 = 50 TeV.  This mimics the neutrino signal from DM
annihilation into a two-body final state that produces monochromatic neutrinos
(e.g., χχ → νν̄).

DM parameters
-------------
  m_chi     = 50 TeV   (E_0 = m_chi for the νν̄ channel)
  <sigma v>  = 1.23e-24 cm^3/s
  Profile    = NFW with canonical Milky Way parameters

Detector: KM3NeT ARCA geometry (uniform A_eff = 2.25e13 cm²).
  Latitude  = +36.3 N  (Mediterranean Sea, off Sicily)
  Longitude = +16.1 E

The energy spectrum panel shows the reco energy histogram — since true energy
is a delta function at E_0, the spread shows the detector energy resolution.

Run from the project root:
    python examples/line_spectrum_halo_sampling.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spore.conventions import SkyCoordinate
from spore.conventions.units import units
from spore.detector.detector import Detector
from spore.event_sampling import GalacticHaloEventSampler
from spore.source import NFWProfile, GalacticHaloSource, LineSpectrum
from spore.source.jfactor import psi_from_radec, j_factor

# ---------------------------------------------------------------------------
# Detector — uniform A_eff at KM3NeT ARCA location (Mediterranean Sea)
#   Latitude:  +36.3 N
#   Longitude: +16.1 E
# ---------------------------------------------------------------------------
RESOURCES = Path(__file__).parent.parent / "resources"
RESPONSE_FILE = str(RESOURCES / "uniform_aeff_detector_response.h5")

detector_config = {
    "properties": {
        "latitude": 36.3,
        "longitude": 16.1,
        "medium": "Water",
    },
    "response": {"detector_response_file": RESPONSE_FILE},
}
detector = Detector.from_config(detector_config)

# ---------------------------------------------------------------------------
# Dark matter source — monochromatic line spectrum at E_0 = m_chi = 50 TeV
#
# For χχ → νν̄, each annihilation produces two neutrinos each with E = m_chi.
# LineSpectrum stores E_0 in eV; m_chi = 50 TeV = 5e13 eV.
# ---------------------------------------------------------------------------
M_CHI_GEV  = 5e4           # 50 TeV in GeV
E_0_EV     = M_CHI_GEV * 1e9   # 5e13 eV
SIGMA_V    = 1.23e-24       # cm^3/s

profile  = NFWProfile(rho_s=0.3, r_s=20.0, r_sun=8.5)
spectrum = LineSpectrum(E_0_EV, n_nu=2.0)
source   = GalacticHaloSource(
    profile=profile,
    spectrum=spectrum,
    m_chi_gev=M_CHI_GEV,
    sigma_v_cm3s=SIGMA_V,
)

# ---------------------------------------------------------------------------
# Sample events — 10-year observation window, track morphology
# ---------------------------------------------------------------------------
T_OBS_YEARS = 10.0
T_OBS       = T_OBS_YEARS * 365.25 * units.day
T_MJD       = 60355.0

print(f"DM mass:         {M_CHI_GEV/1e3:.0f} TeV")
print(f"Line energy:     {M_CHI_GEV/1e3:.0f} TeV  (monochromatic, χχ → νν̄)")
print(f"sigma v:         {SIGMA_V:.1e} cm^3/s")
print(f"Observation:     {T_OBS_YEARS:.0f} years")
print(f"Detector:        KM3NeT ARCA location (lat=+36.3 N, lon=+16.1 E), uniform A_eff")
print(f"\nBuilding flux × effective-area grid (2D over sky, takes ~1 min)...")

sampler = GalacticHaloEventSampler(detector, source, burnin=10_000, n_dec=50, n_ra=51)

print("Sampling track events...")
events = sampler.sample_events("track", t=T_MJD, deltat=T_OBS)
print(f"Sampled {len(events)} track events.\n")

if len(events) == 0:
    print("No events sampled — try increasing SIGMA_V or T_OBS_YEARS.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Extract quantities
# ---------------------------------------------------------------------------
true_decs  = np.degrees([e.true_direction.declination     for e in events])
true_ras   = np.degrees([e.true_direction.right_ascension  for e in events])
reco_decs  = np.degrees([e.reco_direction.declination     for e in events])
reco_ras   = np.degrees([e.reco_direction.right_ascension  for e in events])
reco_e_tev = np.array([e.reco_energy / units.TeV for e in events])

# Angle from Galactic Centre for each true direction
psis = np.degrees([
    psi_from_radec(np.radians(ra), np.radians(dec))
    for ra, dec in zip(true_ras, true_decs)
])

# Angular separation between true and reco direction
def angular_separation(e):
    v_t = e.true_direction.to_cartesian()
    v_r = e.reco_direction.to_cartesian()
    return np.degrees(np.arccos(np.clip(np.dot(v_t, v_r), -1.0, 1.0)))

ang_seps = np.array([angular_separation(e) for e in events])

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
E_0_TEV = M_CHI_GEV / 1e3
print(f"True energy:   {E_0_TEV:.0f} TeV (delta function, all events)")
print(f"Reco energy:   median {np.median(reco_e_tev):.2f} TeV, "
      f"range [{reco_e_tev.min():.2f}, {reco_e_tev.max():.2f}] TeV")
print(f"Angle from GC: median {np.median(psis):.1f} deg "
      f"(expect concentration near GC)")
print(f"Angular res:   68th pct {np.percentile(ang_seps, 68):.2f} deg")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    f"NFW halo — monochromatic line signal (KM3NeT ARCA location, uniform $A_{{\\rm eff}}$)\n"
    f"$m_\\chi$ = {E_0_TEV:.0f} TeV, "
    f"$\\chi\\chi \\to \\nu\\bar{{\\nu}}$, "
    f"$\\langle\\sigma v\\rangle$ = {SIGMA_V:.0e} cm$^3$/s, "
    f"{T_OBS_YEARS:.0f} yr",
    fontsize=12,
)

# --- Sky map of reconstructed directions (equatorial) ---
ax = axes[0, 0]
sc = ax.scatter(reco_ras, reco_decs, s=6, alpha=0.5, c=reco_e_tev,
                cmap="plasma", rasterized=True)
ax.scatter([266.4], [-28.9], s=200, marker="*", color="lime",
           zorder=5, label="Galactic Centre")
plt.colorbar(sc, ax=ax, label=r"$E_\mathrm{reco}$ [TeV]")
ax.set_xlabel("Right ascension [deg]")
ax.set_ylabel("Declination [deg]")
ax.set_title("Reconstructed directions (equatorial)")
ax.legend(fontsize=8)

# --- Distribution of angle from Galactic Centre (PDF + NFW overlay) ---
ax = axes[0, 1]
psi_bins = np.linspace(0, 180, 41)
counts, edges = np.histogram(psis, bins=psi_bins)
bin_centres = 0.5 * (edges[:-1] + edges[1:])
pdf_events = counts / (counts.sum() * (psi_bins[1] - psi_bins[0]))
ax.step(bin_centres, pdf_events, where="mid", linewidth=1.5, color="steelblue",
        label="Sampled events")

psi_theory_deg = np.linspace(0.1, 179.9, 500)
psi_theory_rad = np.radians(psi_theory_deg)
j_vals = np.array([j_factor(profile, p) for p in psi_theory_rad])
pdf_theory = j_vals * np.sin(psi_theory_rad)
norm_theory = np.trapezoid(pdf_theory, psi_theory_deg)
pdf_theory /= norm_theory
ax.plot(psi_theory_deg, pdf_theory, color="tomato", linewidth=1.5,
        label="NFW J(ψ)·sin(ψ)")

ax.set_xlabel("Angle from Galactic Centre [deg]")
ax.set_ylabel("Probability density [deg$^{-1}$]")
ax.set_title("Angular distribution from GC (true directions)")
ax.axvline(np.median(psis), color="gray", linestyle="--",
           label=f"Median = {np.median(psis):.1f}°")
ax.legend(fontsize=8)

# --- Reconstructed energy spectrum (true energy is a delta function) ---
ax = axes[1, 0]
e_lo = max(reco_e_tev.min() * 0.5, E_0_TEV * 0.1)
e_hi = reco_e_tev.max() * 1.5
e_bins = np.logspace(np.log10(e_lo), np.log10(e_hi), 40)
ax.hist(reco_e_tev, bins=e_bins, histtype="step", linewidth=1.5,
        color="steelblue", label="Reco energy")
ax.axvline(E_0_TEV, color="tomato", linestyle="--", linewidth=1.5,
           label=f"True $E_0$ = {E_0_TEV:.0f} TeV")
ax.set_xscale("log")
ax.set_xlabel("Energy [TeV]")
ax.set_ylabel("Events / bin")
ax.set_title("Reconstructed energy (true = delta function at $E_0$)")
ax.legend()

# --- Angular resolution ---
ax = axes[1, 1]
ax.scatter(reco_e_tev, ang_seps, s=4, alpha=0.4, color="steelblue",
           rasterized=True)
ax.set_xscale("log")
ax.axvline(E_0_TEV, color="tomato", linestyle="--", linewidth=1.0,
           label=f"$E_0$ = {E_0_TEV:.0f} TeV")
ax.axhline(np.percentile(ang_seps, 68), color="gray", linestyle="--",
           label=f"68th pct = {np.percentile(ang_seps, 68):.2f}°")
ax.set_xlabel("Reconstructed energy [TeV]")
ax.set_ylabel("True–reco angular separation [deg]")
ax.set_title("Angular resolution")
ax.legend(fontsize=8)

plt.tight_layout()
plot_path = Path(__file__).parent / "line_spectrum_halo_sampling.png"
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
plt.show()
