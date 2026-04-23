"""
Generate neutrino events from dark matter annihilation or decay in the Galactic Halo.

The spatial template is the J-factor (annihilation) or D-factor (decay) sky map,
computed by line-of-sight integration through an NFW or Einasto density profile
centred on the Galactic Centre.  The energy spectrum is a monoenergetic line
broadened by a narrow Gaussian in log-energy space, placed at

    E_line = m_chi          (annihilation)
    E_line = m_chi / 2      (decay)

reflecting the kinematics of a two-body process.  The line width (sigma_log = 0.05
in log(E)) is much narrower than any realistic detector energy resolution, so the
reconstructed-energy distribution is dominated by the detector IRF, not the source
model.

Assumptions
-----------
* Democratic neutrino flavor mixing after propagation: each of the six standard
  species receives one sixth of the total flux.
* For annihilation, the total number of neutrinos per event is 2 (chi chi -> nu nubar).
* For decay, the total number of neutrinos per event is 2 (chi -> nu nubar).
* Default cross-section / lifetime represent optimistic but physically motivated
  values; they can be overridden on the command line.

Usage
-----
Poisson mode (sample over a livetime):

    python examples/galactic_halo_line_spectrum.py --mass 1e4 --livetime 10

Fixed-N mode (exactly N events regardless of rate):

    python examples/galactic_halo_line_spectrum.py --mass 1e4 --n-events 200 \\
        --mode decay --profile einasto --morphology cascade

Run from the project root:

    python examples/galactic_halo_line_spectrum.py [options]

"""

import argparse
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator, CubicSpline

from spore.conventions import ureg
from spore.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import SourceSampler
from spore.event_sampling.io import write_events

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO          = Path(__file__).parent.parent
RESOURCES     = REPO / "resources"
OUTPUT_DIR    = Path(__file__).parent / "output"
RESPONSE      = str(RESOURCES / "configs" / "ps10yr_detector_response.h5")
FLUX_FILE     = str(OUTPUT_DIR / "_halo_flux.h5")
JFACTOR_CACHE = str(OUTPUT_DIR / "_jfactor_cache.h5")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KPC_TO_CM = 3.0857e21    # 1 kpc in cm
GEV_PER_MSUN = 1.115e57  # unused here but handy for cross-checks

# ---------------------------------------------------------------------------
# Default dark matter halo profile parameters
# ---------------------------------------------------------------------------
R_SUN_KPC  = 8.5          # Sun–GC distance [kpc]
RHO_SUN    = 0.3          # local DM density [GeV cm^-3]
R_S_KPC    = 20.0         # NFW / Einasto scale radius [kpc]
EINASTO_ALPHA = 0.17      # Einasto shape parameter
D_MAX_KPC  = 200.0        # truncation radius for LOS integration [kpc]
N_LOS      = 600          # quadrature points along each line of sight
R_MIN_KPC  = 0.1          # inner cutoff to regularise the NFW cusp [kpc]


# ---------------------------------------------------------------------------
# Density profiles
# ---------------------------------------------------------------------------

def _nfw_rho_s(r_s_kpc, rho_sun, r_sun_kpc):
    """NFW scale density normalised to rho_sun at r_sun."""
    x = r_sun_kpc / r_s_kpc
    return rho_sun * x * (1.0 + x) ** 2


def nfw_density(r_kpc, r_s_kpc=R_S_KPC, rho_sun=RHO_SUN, r_sun_kpc=R_SUN_KPC):
    """NFW density [GeV cm^-3] at galactocentric radius r [kpc]."""
    rho_s = _nfw_rho_s(r_s_kpc, rho_sun, r_sun_kpc)
    x = np.maximum(r_kpc, R_MIN_KPC) / r_s_kpc
    return rho_s / (x * (1.0 + x) ** 2)


def _einasto_rho_s(r_s_kpc, alpha, rho_sun, r_sun_kpc):
    """Einasto scale density normalised to rho_sun at r_sun."""
    x_sun = r_sun_kpc / r_s_kpc
    exponent = -2.0 / alpha * (x_sun ** alpha - 1.0)
    return rho_sun / np.exp(exponent)


def einasto_density(r_kpc, r_s_kpc=R_S_KPC, alpha=EINASTO_ALPHA,
                    rho_sun=RHO_SUN, r_sun_kpc=R_SUN_KPC):
    """Einasto density [GeV cm^-3] at galactocentric radius r [kpc]."""
    rho_s = _einasto_rho_s(r_s_kpc, alpha, rho_sun, r_sun_kpc)
    x = np.maximum(r_kpc, R_MIN_KPC) / r_s_kpc
    exponent = -2.0 / alpha * (x ** alpha - 1.0)
    return rho_s * np.exp(exponent)


# ---------------------------------------------------------------------------
# J / D factor cache
# ---------------------------------------------------------------------------

def _jfactor_key(profile, mode, n_dec, n_ra):
    return f"{profile}_{mode}_{n_dec}_{n_ra}"


def load_jfactor(profile, mode, sindecs, ras):
    """Return cached (jd_map, sindecs, ras) or None if not present."""
    key = _jfactor_key(profile, mode, len(sindecs), len(ras))
    if not os.path.exists(JFACTOR_CACHE):
        return None
    with h5py.File(JFACTOR_CACHE, "r") as f:
        if key not in f:
            return None
        gp = f[key]
        return gp["jd_map"][:], gp["sindecs"][:], gp["ras"][:]


def save_jfactor(profile, mode, sindecs, ras, jd_map):
    key = _jfactor_key(profile, mode, len(sindecs), len(ras))
    with h5py.File(JFACTOR_CACHE, "a") as f:
        if key in f:
            del f[key]
        gp = f.create_group(key)
        gp.create_dataset("jd_map",  data=jd_map)
        gp.create_dataset("sindecs", data=sindecs)
        gp.create_dataset("ras",     data=ras)


# ---------------------------------------------------------------------------
# J / D factor maps
# ---------------------------------------------------------------------------

def _galactic_lb(dec_rad, ra_rad):
    """Convert equatorial (dec, RA) [rad] to galactic (l, b) [rad]."""
    c = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad, frame="icrs")
    g = c.galactic
    return float(g.l.rad), float(g.b.rad)


def _los_r(d_kpc, l_rad, b_rad, r_sun_kpc=R_SUN_KPC):
    """Galactocentric distance [kpc] at LOS depth d from Earth."""
    cos_angle = np.cos(b_rad) * np.cos(l_rad)
    return np.sqrt(
        d_kpc ** 2 + r_sun_kpc ** 2 - 2.0 * d_kpc * r_sun_kpc * cos_angle
    )


def compute_jd_map(sindecs, ras, profile, mode, n_los=N_LOS, d_max=D_MAX_KPC):
    """
    Compute the J-factor (annihilation) or D-factor (decay) sky map.

    Parameters
    ----------
    sindecs : ndarray, shape (n_dec,)
        Uniformly spaced sin(declination) grid.
    ras : ndarray, shape (n_ra,)
        Uniformly spaced right-ascension grid [rad].
    profile : callable
        rho(r_kpc) -> density [GeV cm^-3].
    mode : {"annihilation", "decay"}
    n_los : int
        Number of quadrature points along each line of sight.
    d_max : float
        Maximum LOS depth [kpc].

    Returns
    -------
    jd_map : ndarray, shape (n_dec, n_ra)
        J-factor [GeV^2 cm^-5] or D-factor [GeV cm^-2] per steradian.
    """
    d_kpc = np.linspace(0.0, d_max, n_los)
    d_cm  = d_kpc * KPC_TO_CM

    # Convert all (dec, RA) pixels to galactic (l, b) in one batched SkyCoord
    # call instead of 3200 individual calls, avoiding repeated astropy frame
    # initialisation overhead and its associated memory accumulation.
    decs_2d, ras_2d = np.meshgrid(
        np.arcsin(np.clip(sindecs, -1.0, 1.0)), ras, indexing='ij',
    )
    coords = SkyCoord(
        ra=ras_2d.ravel() * u.rad,
        dec=decs_2d.ravel() * u.rad,
        frame="icrs",
    )
    gal = coords.galactic
    ls = gal.l.rad.reshape(len(sindecs), len(ras))   # (n_dec, n_ra)
    bs = gal.b.rad.reshape(len(sindecs), len(ras))   # (n_dec, n_ra)

    # Broadcast LOS integration over (n_dec, n_ra, n_los) in one shot.
    # Peak allocation: ~15 MB for the default 40 × 80 × 600 grid.
    cos_angle = (np.cos(bs) * np.cos(ls))[:, :, np.newaxis]  # (n_dec, n_ra, 1)
    d_3d = d_kpc[np.newaxis, np.newaxis, :]                   # (1, 1, n_los)
    r = np.sqrt(d_3d**2 + R_SUN_KPC**2 - 2.0 * d_3d * R_SUN_KPC * cos_angle)
    r = np.maximum(r, R_MIN_KPC)
    rho = profile(r)                                          # (n_dec, n_ra, n_los)
    integrand = rho**2 if mode == "annihilation" else rho
    return np.trapezoid(integrand, d_cm, axis=-1)


# ---------------------------------------------------------------------------
# Line-spectrum energy grid
# ---------------------------------------------------------------------------

def line_spectrum(energies_gev, e_line_gev, sigma_log=0.05):
    """
    Gaussian line spectrum in log-energy space.

    Parameters
    ----------
    energies_gev : ndarray
        Energy grid [GeV].
    e_line_gev : float
        Line energy [GeV].
    sigma_log : float
        Width in log(E) (natural log).  Default 0.05 corresponds to roughly
        5% in energy, much narrower than any realistic detector resolution.

    Returns
    -------
    ndarray, shape (n_e,)
        dN/dE [GeV^-1], normalised so that integral over log(E) of
        E * (dN/dE) equals 2 (two neutrinos per annihilation/decay event).
    """
    log_e      = np.log(energies_gev)
    log_e_line = np.log(e_line_gev)
    # Gaussian in log-E space: g(log E) = A exp(-(log E - log E0)^2 / (2 sigma^2))
    # dN/dE = g / E (since g = E * dN/dE in log-E parameterisation)
    g = np.exp(-0.5 * ((log_e - log_e_line) / sigma_log) ** 2)
    # Normalise so that integral_log_E (E * dN/dE) d(log E) = integral g d(log E) = 2
    norm = np.trapezoid(g, log_e)
    if norm == 0:
        return np.zeros_like(energies_gev)
    return 2.0 * g / (norm * energies_gev)   # [GeV^-1]


# ---------------------------------------------------------------------------
# Assemble flux grid
# ---------------------------------------------------------------------------

def build_flux_grid(sindecs, ras, energies, jd_map, spectrum, prefactor):
    """
    Assemble the 4-D flux array (6 species × dec × RA × energy).

    The total flux is:
        Phi(dec, RA, E) = prefactor * jd_map(dec, RA) * spectrum(E)

    Split equally among all 6 standard neutrino species (democratic mixing).

    Returns
    -------
    fluxes : ndarray, shape (6, n_dec, n_ra, n_e)
        Differential flux per species [GeV^-1 cm^-2 s^-1 sr^-1].
    """
    # outer product: (n_dec, n_ra, 1) * (1, 1, n_e)
    spatial = jd_map[:, :, np.newaxis]              # (n_dec, n_ra, 1)
    spec    = spectrum[np.newaxis, np.newaxis, :]    # (1, 1, n_e)
    per_species = (prefactor / 6.0) * spatial * spec  # (n_dec, n_ra, n_e)
    return np.tile(per_species[np.newaxis], (6, 1, 1, 1))


# ---------------------------------------------------------------------------
# Write HDF5 flux file
# ---------------------------------------------------------------------------

def write_flux_h5(path, sindecs, ras, energies, fluxes):
    with h5py.File(path, "w") as f:
        gp = f.create_group("dm_line")
        gp.create_dataset("sindecs",  data=sindecs)
        gp.create_dataset("ras",      data=ras)
        gp.create_dataset("energies", data=energies)
        gp.create_dataset("fluxes",   data=fluxes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mass",      type=float, default=1e4,
                   help="DM particle mass [GeV].  Default: 1e4")
    p.add_argument("--mode",      choices=["annihilation", "decay"],
                   default="annihilation",
                   help="Production mechanism.  Default: annihilation")
    p.add_argument("--profile",   choices=["nfw", "einasto"],
                   default="nfw",
                   help="Dark matter density profile.  Default: nfw")
    p.add_argument("--morphology", choices=["track", "cascade"],
                   default="track",
                   help="Event morphology to sample.  Default: track")

    rate_grp = p.add_mutually_exclusive_group(required=True)
    rate_grp.add_argument("--n-events", type=int,
                          help="Generate exactly this many events (fixed-N mode).")
    rate_grp.add_argument("--livetime", type=float,
                          help="Observation livetime [years] for Poisson sampling.")

    p.add_argument("--sigmav",    type=float, default=3e-26,
                   help="Annihilation cross-section <sigma v> [cm^3 s^-1].  "
                        "Default: 3e-26 (thermal relic)")
    p.add_argument("--lifetime",  type=float, default=1e28,
                   help="Decay lifetime tau [s].  Default: 1e28")
    p.add_argument("--output",    default=str(OUTPUT_DIR / "halo_events.h5"),
                   help="Output HDF5 file for sampled events.  Default: examples/output/halo_events.h5")
    p.add_argument("--n-dec",     type=int, default=40,
                   help="Declination grid points.  Default: 40")
    p.add_argument("--n-ra",      type=int, default=80,
                   help="Right-ascension grid points.  Default: 80")
    p.add_argument("--n-energy",  type=int, default=200,
                   help="Energy grid points (log-uniform).  Default: 200")
    p.add_argument("--sigma-log", type=float, default=0.05,
                   help="Line width in log(E).  Default: 0.05")
    p.add_argument("--n-trials",  type=int, default=1,
                   help="Number of times to call sample_events with the same "
                        "sampler.  Events from all trials are written to the "
                        "output file under groups 'signal_0', 'signal_1', .... "
                        "Default: 1.")
    p.add_argument("--plot",      action="store_true",
                   help="Show a sky map and reconstructed-energy histogram.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    m       = args.mass
    mode    = args.mode
    profile = args.profile

    # --- line energy ---
    e_line = m if mode == "annihilation" else m / 2.0
    print(f"DM mass:    {m:.3g} GeV")
    print(f"Mode:       {mode}")
    print(f"Profile:    {profile}")
    print(f"E_line:     {e_line:.3g} GeV")

    # --- density profile ---
    rho_fn = nfw_density if profile == "nfw" else einasto_density

    # --- prefactor [GeV^-1 cm^-2 s^-1 sr^-1 / (J or D factor)] ---
    if mode == "annihilation":
        # dPhi/dE dOmega = sigmav / (2 * 4pi * m^2) * dN/dE * J
        prefactor = args.sigmav / (2.0 * 4.0 * np.pi * m ** 2)
    else:
        # dPhi/dE dOmega = 1 / (4pi * m * tau) * dN/dE * D
        prefactor = 1.0 / (4.0 * np.pi * m * args.lifetime)

    # --- grids ---
    sindecs  = np.linspace(-1.0, 1.0, args.n_dec)
    ras      = np.linspace(0.0, 2.0 * np.pi, args.n_ra, endpoint=False)

    # Energy range: ±8 sigma_log in ln(E) around the line.
    # Keeping the grid this narrow ensures the Gaussian never underflows to
    # exactly 0 in float64 (exp(-32) ~ 1e-14), which would break the log
    # interpolation in parse_3d_file.
    n_sigma = 8.0
    e_lo = max(1e2, e_line * np.exp(-n_sigma * args.sigma_log))
    e_hi = min(1e7, e_line * np.exp(+n_sigma * args.sigma_log))
    energies = np.logspace(np.log10(e_lo), np.log10(e_hi), args.n_energy)

    # --- J / D factor map (cached) ---
    cached = load_jfactor(args.profile, mode, sindecs, ras)
    if cached is not None:
        jd_map, sindecs, ras = cached
        print(f"\nLoaded {mode} factor map from cache ({args.n_dec} x {args.n_ra} grid).")
    else:
        print(f"\nComputing {mode} factor map ({args.n_dec} x {args.n_ra} grid)...")
        jd_map = compute_jd_map(sindecs, ras, rho_fn, mode)
        save_jfactor(args.profile, mode, sindecs, ras, jd_map)
        print(f"  Saved to {JFACTOR_CACHE}")

    gc_gal = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic")
    gc_eq  = gc_gal.icrs
    gc_dec = float(gc_eq.dec.rad)
    gc_ra  = float(gc_eq.ra.rad)
    i_gc   = np.argmin(np.abs(sindecs - np.sin(gc_dec)))
    j_gc   = np.argmin(np.abs(ras - gc_ra))
    print(f"  GC at (dec, RA) = ({np.degrees(gc_dec):.1f} deg, "
          f"{np.degrees(gc_ra):.1f} deg)")
    print(f"  Peak factor = {jd_map[i_gc, j_gc]:.3e} "
          f"{'GeV^2 cm^-5' if mode == 'annihilation' else 'GeV cm^-2'}")

    # --- energy spectrum ---
    spectrum = line_spectrum(energies, e_line, sigma_log=args.sigma_log)

    # --- flux grid ---
    fluxes = build_flux_grid(sindecs, ras, energies, jd_map, spectrum, prefactor)
    print(f"\nPeak flux: {fluxes.max():.3e} GeV^-1 cm^-2 s^-1 sr^-1")

    # --- write HDF5 ---
    write_flux_h5(FLUX_FILE, sindecs, ras, energies, fluxes)

    # --- detector and sampler ---
    det = Detector.from_config({
        "properties": {
            "latitude": 36.3, "longitude": 16.1,
            "depth": 3500, "medium": "Water",
        },
        "response": {"detector_response_file": RESPONSE},
    })

    src = ExtendedSource.from_config({"flux": {"location": f"{FLUX_FILE}:dm_line"}})

    print("\nBuilding sampler (steady-state mode)...")
    sampler = SourceSampler(
        det, src,
        n_time_samples=50,
        n_dec=args.n_dec,
        n_ra=args.n_ra,
        n_e=args.n_energy,
    )

    # --- sample events ---
    if args.n_events is not None and args.n_trials == 1:
        # Fast path: single trial, single group name for backwards compatibility
        print(f"Sampling {args.n_events} {args.morphology} events (fixed-N mode)...")
        events = sampler.sample_events(args.morphology, nevent=args.n_events)
        print(f"  Got {len(events)} events")
        write_events(events, args.output, group="signal")
        print(f"Events written to {args.output} (group 'signal')")
    else:
        if args.livetime is not None:
            t_obs = ureg.Quantity(args.livetime * 365.25, "day")
            n_exp = sampler.expected_events(args.morphology, t_obs)
            print(f"Poisson mode: expected {n_exp:.1f} events per trial "
                  f"over {args.livetime:.3g} yr")
        rng = np.random.default_rng()
        all_events = []
        for trial in range(args.n_trials):
            seed = int(rng.integers(2**32))
            if args.n_events is not None:
                trial_events = sampler.sample_events(
                    args.morphology, nevent=args.n_events, seed=seed)
            else:
                trial_events = sampler.sample_events(
                    args.morphology, deltat=t_obs, seed=seed)
            all_events.append(trial_events)
            group = f"signal_{trial}"
            write_events(trial_events, args.output, group=group)
            print(f"  Trial {trial+1}/{args.n_trials}: "
                  f"{len(trial_events)} events → {args.output} (group '{group}')")
        events = [e for trial_events in all_events for e in trial_events]
        print(f"Total: {len(events)} events across {args.n_trials} trials")

    # --- optional plot ---
    if args.plot and events:
        reco_decs_deg = np.degrees([e.reco_direction.declination     for e in events])
        reco_ras_deg  = np.degrees([e.reco_direction.right_ascension for e in events])
        log10_e_reco  = [np.log10(e.reco_energy.to("GeV").magnitude) for e in events]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"Galactic halo DM {mode}  |  {profile.upper()} profile  |  "
            f"m_{{chi}} = {m:.3g} GeV  |  {len(events)} events"
        )

        ax = axes[0]
        ax.scatter(reco_ras_deg, reco_decs_deg, s=4, alpha=0.5, color="steelblue")
        ax.scatter(np.degrees(gc_ra), np.degrees(gc_dec),
                   marker="*", s=200, color="red", zorder=5, label="GC")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_title("Reconstructed sky positions")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.hist(log10_e_reco, bins=30, color="steelblue", edgecolor="white", lw=0.5)
        ax.axvline(np.log10(e_line), color="tomato", lw=1.5, ls="--",
                   label=f"$E_{{line}}$ = {e_line:.3g} GeV")
        ax.set_xlabel(r"$\log_{10}(E_\mathrm{reco}\,/\,\mathrm{GeV})$")
        ax.set_ylabel("Events / bin")
        ax.set_title("Reconstructed energy spectrum")
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.hist(reco_decs_deg, bins=30, color="steelblue", edgecolor="white", lw=0.5)
        ax.axvline(np.degrees(gc_dec), color="tomato", lw=1.5, ls="--",
                   label=f"GC dec = {np.degrees(gc_dec):.1f} deg")
        ax.set_xlabel("Reconstructed declination [deg]")
        ax.set_ylabel("Events / bin")
        ax.set_title("Declination distribution")
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_png = Path(args.output).with_suffix(".png")
        plt.savefig(out_png, dpi=150)
        print(f"Plot saved to {out_png}")
        plt.show()


if __name__ == "__main__":
    main()
