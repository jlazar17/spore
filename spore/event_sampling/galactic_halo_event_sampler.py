"""
Event sampler for neutrinos from dark matter annihilation in the Galactic halo.

This is a drop-in replacement for ExtendedSourceEventSampler designed for
GalacticHaloSource.  The only structural difference is that the source flux
is evaluated at each (declination, right ascension) grid point separately,
rather than being broadcast over RA from a declination-only callable.  This
correctly captures the J-factor variation across the sky.

All other sampling logic (MCMC, smearing, Poisson counting, time handling)
is identical to ExtendedSourceEventSampler.
"""

import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.integrate import quad
from typing import Optional

from ..conventions import units, SkyCoordinate, sky_to_local
from ..physics import neutrinos
from ..detector import Detector
from ..source.galactic_halo_source import GalacticHaloSource
from ..source.jfactor import j_factor as _j_factor, psi_from_radec_grid as _psi_from_radec_grid

# J-factor unit conversion: GeV² cm⁻⁵ → natural units (eV basis)
_J_CONV = (1.0e9) ** 2 * (1.0 / units.cm) ** 5

from .event_sampler import EventSampler
from .event import Event
from .metropolis_hastings import metropolis_hastings
from .utils import smear_truth, zenith_grid


def _clipped_interp(grids, vals, method="pchip"):
    """
    Build a RegularGridInterpolator clipped to non-negative values.

    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is used by
    default.  Unlike natural cubic splines, PCHIP is monotone-preserving: the
    interpolant between any two adjacent grid nodes stays within the range of
    those two values.  This means it cannot overshoot upward or go negative
    when the data are non-negative, making it robust against arbitrary user
    inputs with sharp features (spectral cutoffs, A_eff edges, step-function
    spectra) without requiring the clipping.  The clip is retained as a
    belt-and-suspenders guard for floating-point edge cases.

    Accuracy: PCHIP is O(h^2) in the worst case but much more accurate than
    piecewise-linear in smooth regions because it matches local slopes.  It is
    O(h^4) in monotone smooth regions, comparable to natural cubic splines but
    without the ringing risk.
    """
    interp = RegularGridInterpolator(grids, vals, method=method)
    def f(x):
        return max(0.0, interp(x).item())
    return f


class GalacticHaloEventSampler(EventSampler):
    """
    Samples neutrino events from dark matter annihilation in the Galactic halo.

    The expected event rate integrates:

        dR/dE dΩ = A_eff(ζ(Ω, t), E) × dΦ/dE dΩ (E, RA, Dec)

    over energy and solid angle, where dΦ/dE dΩ is provided by
    GalacticHaloSource and includes the J-factor.

    Parameters
    ----------
    det : Detector
        Detector configuration and response.
    src : GalacticHaloSource
        Dark matter halo source.
    burnin : int
        Number of Metropolis-Hastings burn-in steps. Default 10 000.
    n_dec : int
        Number of declination grid points. Default 20.
    n_ra : int
        Number of right-ascension grid points. Default 21.
    n_e : int
        Number of energy grid points (log-spaced). Default 22.
    """

    def __init__(
        self,
        det: Detector,
        src: GalacticHaloSource,
        burnin: int = 10_000,
        n_dec: int = 20,
        n_ra: int = 21,
        n_e: int = 22,
    ):
        self._burnin = burnin
        super().__init__(det, src)

        ras = np.linspace(0, 2 * np.pi, n_ra)
        decs = np.arcsin(np.linspace(-1, 1, n_dec))
        sds = np.sin(decs)

        self._ras = ras
        self._decs = decs
        self._is_monochromatic = src.is_monochromatic

        # ------------------------------------------------------------------
        # Precompute psi and J-factor for all (dec, ra) grid points.
        # psi is vectorised via array SkyCoord; j_factor uses quad and
        # must be evaluated per direction.
        # ------------------------------------------------------------------
        dec_grid, ra_grid = np.meshgrid(decs, ras, indexing='ij')  # (n_dec, n_ra)
        psi_grid = _psi_from_radec_grid(ra_grid, dec_grid)          # (n_dec, n_ra)

        j_grid = np.zeros((len(decs), len(ras)))
        for jdx in range(len(decs)):
            for kdx in range(len(ras)):
                j_grid[jdx, kdx] = (
                    _j_factor(src._profile, psi_grid[jdx, kdx], d_max_kpc=src._d_max_kpc)
                    * _J_CONV
                )

        # Zenith angles for all (dec, ra) pairs — one vectorised astropy call.
        zeniths = zenith_grid(decs, ras, det.location, self._t0)  # (n_dec, n_ra)

        if src.is_monochromatic:
            # ------------------------------------------------------------------
            # Monochromatic (line) spectrum: energy is fixed to E_0.
            # Drop the energy dimension — MCMC runs over (sin_dec, ra) only.
            # ------------------------------------------------------------------
            E_0 = src.E_0_eV
            self._E_0 = E_0
            self._bounds = [
                (sds.min(), sds.max()),
                (ras.min(), ras.max()),
            ]

            n_nu = next(iter(src._spectra.values())).n_nu
            spatial_grid = src._prefactor * n_nu * j_grid   # (n_dec, n_ra)

            track_effa_fn   = det.response.effective_area.get("track")
            cascade_effa_fn = det.response.effective_area.get("cascade")
            zen_flat = zeniths.ravel()

            if track_effa_fn:
                track_vals = spatial_grid * track_effa_fn(
                    zen_flat, np.full_like(zen_flat, E_0)
                ).reshape(len(decs), len(ras))
            else:
                track_vals = np.zeros((len(decs), len(ras)))

            if cascade_effa_fn:
                cascade_vals = spatial_grid * cascade_effa_fn(
                    zen_flat, np.full_like(zen_flat, E_0)
                ).reshape(len(decs), len(ras))
            else:
                cascade_vals = np.zeros((len(decs), len(ras)))

            track_target   = _clipped_interp((sds, ras), track_vals)
            cascade_target = _clipped_interp((sds, ras), cascade_vals)

            hack = 1e24
            norms = []
            for vals_2d in [track_vals, cascade_vals]:
                out = np.zeros(sds.shape)
                for idx in range(len(sds)):
                    spl = CubicSpline(ras, vals_2d[idx, :] * hack)
                    val, _ = quad(spl, 0, 2 * np.pi)
                    out[idx] = val / hack
                spl = CubicSpline(sds, out * hack)
                val, _ = quad(spl, -1, 1)
                norms.append(val / hack)

        else:
            # ------------------------------------------------------------------
            # Continuous spectrum: 3-D MCMC over (sin_dec, ra, log_E).
            # ------------------------------------------------------------------
            es = np.logspace(2, 6, n_e) * units.GeV
            self._es = es

            self._bounds = [
                (sds.min(), sds.max()),
                (ras.min(), ras.max()),
                (np.log(es).min(), np.log(es).max()),
            ]

            # Spectrum values for each species and energy: (6, n_e)
            spectrum_vals = np.array([
                [src._spectra[nu](e) for e in es]
                for nu in neutrinos
            ])

            # Flux grid via broadcasting: prefactor × J(dec,ra) × spectrum(nu,e)
            # Shape: (6, n_dec, n_ra, n_e)
            fluxes = (
                src._prefactor
                * j_grid[np.newaxis, :, :, np.newaxis]
                * spectrum_vals[:, np.newaxis, np.newaxis, :]
            )

            # Aeff grid — one vectorised call per morphology
            zen_flat = np.repeat(zeniths.ravel(), n_e)   # (n_dec * n_ra * n_e,)
            e_flat   = np.tile(es, len(decs) * len(ras)) # (n_dec * n_ra * n_e,)

            track_effa   = det.response.effective_area.get("track")
            cascade_effa = det.response.effective_area.get("cascade")

            effa_track   = (
                track_effa(zen_flat, e_flat).reshape(len(decs), len(ras), n_e)
                if track_effa else np.zeros((len(decs), len(ras), n_e))
            )
            effa_cascade = (
                cascade_effa(zen_flat, e_flat).reshape(len(decs), len(ras), n_e)
                if cascade_effa else np.zeros((len(decs), len(ras), n_e))
            )

            track_vals   = effa_track   * (fluxes[2] + fluxes[3])
            cascade_vals = effa_cascade * fluxes.sum(axis=0)

            hack = 1e24
            log_es = np.log(es)
            out = np.zeros((len(sds), len(ras)))
            norms = []
            for vals_3d in [track_vals, cascade_vals]:
                for idx in range(len(sds)):
                    for jdx in range(len(ras)):
                        out[idx, jdx] = np.trapezoid(np.exp(log_es) * vals_3d[idx, jdx, :], log_es)
                intermediate = np.zeros(sds.shape)
                for idx in range(len(sds)):
                    spl = CubicSpline(ras, out[idx, :])
                    val, _ = quad(spl, 0, 2 * np.pi)
                    intermediate[idx] = val
                spl = CubicSpline(sds, intermediate * hack)
                val, _ = quad(spl, -1, 1)
                norms.append(val / hack)

            track_target   = _clipped_interp((sds, ras, log_es), track_vals)
            cascade_target = _clipped_interp((sds, ras, log_es), cascade_vals)

        self._d = {
            "track": (None, track_target, norms[0]),
            "cascade": (None, cascade_target, norms[1]),
        }

    def expected_events(
        self,
        morphology: str,
        deltat: float,
        t: Optional[float] = None,
    ) -> float:
        if morphology not in self._d:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {list(self._d.keys())}."
            )
        _, _, norm = self._d[morphology]
        return norm * deltat

    def sample_events(
        self,
        morphology: str,
        deltat: Optional[float] = None,
        nevent: Optional[int] = None,
        t: Optional[float] = None,
        oversample: int = 1,
        verbose: bool = False,
    ):
        """
        Sample neutrino events from the Galactic halo.

        Parameters
        ----------
        morphology : "track" or "cascade"
        oversample : int
            Draw oversample × nevent MCMC samples and thin. Default 1.
        t : float, optional
            Reference time [MJD]. Defaults to the sampler's epoch.
        nevent : int, optional
            Draw exactly this many events per detector (for pseudo-experiments).
        deltat : float, optional
            Observation window in natural units. Poisson-samples event count.
        track : bool
            Show MCMC progress bar. Default False.

        Returns
        -------
        list of Event
        """
        if not ((nevent is None) ^ (deltat is None)):
            raise ValueError("Specify exactly one of deltat or nevent.")

        if morphology not in ("track", "cascade"):
            raise ValueError(f"Invalid morphology '{morphology}': must be 'track' or 'cascade'.")
        if morphology not in self._det.response.available_morphologies:
            raise ValueError(
                f"Morphology '{morphology}' is not available for this detector. "
                f"Available: {self._det.response.available_morphologies}."
            )

        xt, target, norm = self._d[morphology]
        if nevent is None:
            nevent = np.random.poisson(norm * deltat)

        if t is None:
            t = self._t0

        if self._is_monochromatic:
            periodic = [False, True]
            if xt is None:
                xt = np.array([
                    np.random.uniform(),
                    np.random.uniform(self._ras.min(), self._ras.max()),
                ])
                xt = metropolis_hastings(target, xt, self._bounds, size=self._burnin, periodic=periodic)[-1, :]
                self._d[morphology] = xt, target, norm
        else:
            periodic = [False, True, False]
            if xt is None:
                xt = np.array([
                    np.random.uniform(),
                    np.random.uniform(self._ras.min(), self._ras.max()),
                    np.random.uniform(np.log(self._es.min()), np.log(self._es.max())),
                ])
                xt = metropolis_hastings(target, xt, self._bounds, size=self._burnin, periodic=periodic)[-1, :]
                self._d[morphology] = xt, target, norm

        res = metropolis_hastings(
            target, xt, self._bounds, size=nevent * oversample, track=verbose,
            periodic=periodic,
        )
        if len(res) == 0:
            return []

        xt = res[-1, :]
        self._d[morphology] = xt, target, norm

        # Account for Earth's rotation between reference epoch and t
        offset = 2 * np.pi * ((t - self._t0) % 1)
        res[:, 1] = np.mod(res[:, 1] + offset, 2 * np.pi)

        res = res[::oversample]
        events = []
        for x in res:
            true_direction = SkyCoordinate(np.arcsin(x[0]), x[1])
            true_energy = self._E_0 if self._is_monochromatic else np.exp(x[2])
            reco_direction, reco_energy, ang_err = smear_truth(
                true_direction, true_energy, self._det, morphology
            )
            if deltat is None:
                t_event = t
            else:
                t_event = (
                    t
                    + np.random.uniform(low=-deltat / 2, high=deltat / 2)
                    / (24 * 3600 * units.sec)
                )
            morphology_id = 2 if morphology == "track" else 1
            events.append(Event(
                true_direction,
                reco_direction,
                true_energy,
                reco_energy,
                t_event,
                morphology_id,
                ang_err=ang_err,
            ))

        return events
