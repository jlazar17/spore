"""
Dark matter annihilation from the Galactic halo — event sampling example.

Demonstrates GalacticHaloEventSampler with an NFW density profile and a
soft neutrino spectrum (SoftSpectrum, proxy for the bb-bar channel).

DM parameters
-------------
  m_chi     = 10 TeV
  <sigma v>  = 3e-26 cm^3/s  (canonical thermal relic)
  Profile    = NFW with canonical Milky Way parameters

Detector: KM3NeT ARCA geometry (IceCube detector response reused).
  Latitude  = +36.3 N  (Mediterranean Sea, off Sicily)
  Longitude = +16.1 E
  Depth     = 3500 m
  Medium    = Water

At this latitude the Galactic Centre (Dec ~ -29 deg) transits below the
horizon and is accessible as an upgoing source, unlike IceCube.

The expected number of signal events is small for a single year even at
this cross section, so we use a large observation window (10 years) and an
overestimated effective area to collect a statistically useful sample.

Run from the project root:
    python examples/nfw_halo_sampling.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from spore.conventions import SkyCoordinate
from spore.conventions.units import units
from spore.detector.detector import Detector
from spore.event_sampling import GalacticHaloEventSampler
from spore.source import NFWProfile, GalacticHaloSource, SoftSpectrum
from spore.source.jfactor import psi_from_radec, j_factor

# ---------------------------------------------------------------------------
# Detector — IceCube response at KM3NeT ARCA location (Mediterranean Sea)
#   Latitude:  +36.3 N
#   Longitude: +16.1 E
#   Depth:     3500 m
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
# Dark matter source
#
# NFW profile: canonical Milky Way parameters
#   rho_s = 0.3 GeV/cm^3  (local DM density at r_sun)
#   r_s   = 20 kpc
#   r_sun = 8.5 kpc
#
# Spectrum: SoftSpectrum as a proxy for the bb-bar channel.
#   a=1.5, b=1.0 gives a shape broadly consistent with PPPC4DMID tables
#   at m_chi ~ 10 TeV.
# ---------------------------------------------------------------------------
M_CHI_GEV = 1e4          # 10 TeV DM mass
SIGMA_V    = 1.23e-24     # cm^3/s — yields ~1 000 events / 10 yr

profile  = NFWProfile(rho_s=0.3, r_s=20.0, r_sun=8.5)
spectrum = SoftSpectrum(M_CHI_GEV * 1e9, a=1.5, b=1.0, n_nu=3.0)
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
print(f"sigma v:         {SIGMA_V:.1e} cm^3/s")
print(f"Observation:     {T_OBS_YEARS:.0f} years")
print(f"Detector:        KM3NeT ARCA location (lat=+36.3 N, lon=+16.1 E)")
print(f"\nBuilding flux × effective-area grid (50×51 dec×RA, takes ~5 min)...")

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
true_decs  = np.degrees([e.true_direction.declination    for e in events])
true_ras   = np.degrees([e.true_direction.right_ascension for e in events])
reco_decs  = np.degrees([e.reco_direction.declination    for e in events])
reco_ras   = np.degrees([e.reco_direction.right_ascension for e in events])
true_e_tev = np.array([e.true_energy  / units.TeV for e in events])
reco_e_tev = np.array([e.reco_energy  / units.TeV for e in events])

# Angle from Galactic Centre for each true direction
psis = np.degrees([
    psi_from_radec(
        np.radians(ra),
        np.radians(dec),
    )
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
print(f"True energy:   median {np.median(true_e_tev):.2f} TeV, "
      f"range [{true_e_tev.min():.2f}, {true_e_tev.max():.2f}] TeV")
print(f"Reco energy:   median {np.median(reco_e_tev):.2f} TeV")
print(f"Angle from GC: median {np.median(psis):.1f} deg "
      f"(expect concentration near GC)")
print(f"Angular res:   68th pct {np.percentile(ang_seps, 68):.2f} deg")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    f"NFW halo — DM annihilation tracks (KM3NeT ARCA location, uniform $A_{{\\rm eff}}$)\n"
    f"$m_\\chi$ = {M_CHI_GEV/1e3:.0f} TeV, "
    f"$\\langle\\sigma v\\rangle$ = {SIGMA_V:.0e} cm$^3$/s, "
    f"{T_OBS_YEARS:.0f} yr",
    fontsize=13,
)

# --- Sky map of reconstructed directions (equatorial) ---
ax = axes[0, 0]
sc = ax.scatter(reco_ras, reco_decs, s=6, alpha=0.5, c=np.log10(reco_e_tev),
                cmap="plasma", rasterized=True)
ax.scatter([266.4], [-28.9], s=200, marker="*", color="lime",
           zorder=5, label="Galactic Centre")
plt.colorbar(sc, ax=ax, label=r"$\log_{10}(E_\mathrm{reco}\ /\ \mathrm{TeV})$")
ax.set_xlabel("Right ascension [deg]")
ax.set_ylabel("Declination [deg]")
ax.set_title("Reconstructed directions (equatorial)")
ax.legend(fontsize=8)

# --- Distribution of angle from Galactic Centre (PDF + NFW overlay) ---
ax = axes[0, 1]
psi_bins = np.linspace(0, 180, 41)
bin_width_rad = np.radians(psi_bins[1] - psi_bins[0])
counts, edges = np.histogram(psis, bins=psi_bins)
bin_centres = 0.5 * (edges[:-1] + edges[1:])
# Normalise to a PDF over psi (probability per degree)
pdf_events = counts / (counts.sum() * (psi_bins[1] - psi_bins[0]))
ax.step(bin_centres, pdf_events, where="mid", linewidth=1.5, color="steelblue",
        label="Sampled events")

# NFW theoretical curve: dP/dpsi ∝ J(psi) × sin(psi), normalised
psi_theory_deg = np.linspace(0.1, 179.9, 500)
psi_theory_rad = np.radians(psi_theory_deg)
j_vals = np.array([j_factor(profile, p) for p in psi_theory_rad])
# Weight by sin(psi) to account for the solid-angle element
pdf_theory = j_vals * np.sin(psi_theory_rad)
# Normalise so area under the curve = 1 (over psi in degrees)
norm = np.trapz(pdf_theory, psi_theory_deg)
pdf_theory /= norm
ax.plot(psi_theory_deg, pdf_theory, color="tomato", linewidth=1.5,
        label="NFW J(ψ)·sin(ψ)")

ax.set_xlabel("Angle from Galactic Centre [deg]")
ax.set_ylabel("Probability density [deg⁻¹]")
ax.set_title("Angular distribution from GC (true directions)")
ax.axvline(np.median(psis), color="gray", linestyle="--",
           label=f"Median = {np.median(psis):.1f}°")
ax.legend(fontsize=8)

# --- True energy spectrum ---
ax = axes[1, 0]
e_bins = np.logspace(
    np.log10(true_e_tev.min() * 0.5),
    np.log10(M_CHI_GEV / 1e3 * 1.5),
    30,
)
ax.hist(true_e_tev, bins=e_bins, histtype="step", linewidth=1.5,
        color="steelblue", label="True")
ax.hist(reco_e_tev, bins=e_bins, histtype="step", linewidth=1.5,
        color="tomato", linestyle="--", label="Reco")
ax.axvline(M_CHI_GEV / 1e3, color="gray", linestyle=":",
           label=f"$m_\\chi$ = {M_CHI_GEV/1e3:.0f} TeV")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [TeV]")
ax.set_ylabel("Events / bin")
ax.set_title("Energy spectrum")
ax.legend()

# --- Angular resolution ---
ax = axes[1, 1]
ax.scatter(reco_e_tev, ang_seps, s=4, alpha=0.4, color="steelblue",
           rasterized=True)
ax.set_xscale("log")
ax.set_xlabel("Reconstructed energy [TeV]")
ax.set_ylabel("True–reco angular separation [deg]")
ax.set_title("Angular resolution vs energy")
ax.axhline(np.percentile(ang_seps, 68), color="tomato", linestyle="--",
           label=f"68th pct = {np.percentile(ang_seps, 68):.2f}°")
ax.legend()

plt.tight_layout()
plot_path = Path(__file__).parent / "nfw_halo_sampling.png"
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
plt.show()
