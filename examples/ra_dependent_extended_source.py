"""
Example: RA-dependent extended source via a custom Distribution subclass.

This example shows how to implement a spatially non-uniform extended source
whose flux depends on right ascension as well as declination.  Galactic halo
dark matter annihilation is used as the motivating physics, but the pattern
applies to any source with intrinsic RA structure (e.g. a galactic plane
template or a large-scale-structure cross-correlation map).

The key steps are:
  1. Subclass Distribution and implement density(e, dec, ra).
  2. Build a Flux from that Distribution.
  3. Subclass ExtendedSource, set uses_ra = True, and pass the Flux to super().
  4. Hand the source to ExtendedSourceEventSampler as usual.

Run from the project root:
    python examples/ra_dependent_extended_source.py
"""

from pathlib import Path

import numpy as np
from scipy.integrate import quad

from spore.conventions import ureg
from spore.detector.detector import Detector
from spore.event_sampling import ExtendedSourceEventSampler
from spore.physics import neutrinos
from spore.source.extended_source import ExtendedSource
from spore.source.flux import Flux
from spore.source.flux.distributions import Distribution

# ---------------------------------------------------------------------------
# Inline physics: NFW profile and J-factor
# ---------------------------------------------------------------------------

# Galactic Centre in J2000 equatorial coordinates
_GC_RA_RAD  = np.radians(266.4)
_GC_DEC_RAD = np.radians(-28.9)


def _psi_from_radec(ra, dec):
    """Angular separation from the Galactic Centre [radians]."""
    cos_psi = (
        np.sin(_GC_DEC_RAD) * np.sin(dec)
        + np.cos(_GC_DEC_RAD) * np.cos(dec) * np.cos(ra - _GC_RA_RAD)
    )
    return np.arccos(np.clip(cos_psi, -1.0, 1.0))


def _nfw_density(r, rho_s, r_s):
    """NFW mass density [arbitrary units]."""
    return rho_s / ((r / r_s) * (1.0 + r / r_s) ** 2)


def _j_factor(psi, rho_s=0.3, r_s=20.0, r_sun=8.5, d_max_kpc=200.0):
    """Line-of-sight integral of rho^2 along direction psi from GC [kpc GeV^2 cm^-6]."""
    def integrand(ell):
        r = np.sqrt(ell**2 + r_sun**2 - 2.0 * ell * r_sun * np.cos(psi))
        return _nfw_density(r, rho_s, r_s) ** 2

    result, _ = quad(integrand, 0.0, d_max_kpc, limit=200)
    return result


# ---------------------------------------------------------------------------
# Step 1: custom Distribution subclass
# ---------------------------------------------------------------------------

class NFWAnnihilationDistribution(Distribution):
    """
    Spectral-spatial distribution for DM annihilation in an NFW halo.

    density(e, dec, ra) = dN/dE(e) * J(psi(dec, ra))

    The physics prefactor (sigma_v / 8 pi m_chi^2) is stored separately as
    the Flux normalization, following the standard Flux(normalization, distribution)
    convention.

    Args:
        m_chi_gev: DM mass [GeV].
        emin: Minimum energy [GeV].
        emax: Maximum energy [GeV].
        rho_s: NFW scale density [GeV cm^-3].
        r_s: NFW scale radius [kpc].
        r_sun: Sun-GC distance [kpc].
        d_max_kpc: Line-of-sight truncation [kpc].
    """

    def __init__(self, m_chi_gev, emin, emax, rho_s=0.3, r_s=20.0, r_sun=8.5, d_max_kpc=200.0):
        super().__init__(emin, emax)
        self._m_chi = m_chi_gev
        self._rho_s = rho_s
        self._r_s   = r_s
        self._r_sun = r_sun
        self._d_max = d_max_kpc

    def _spectrum(self, e):
        """Flat box spectrum: dN/dE = 2/m_chi for 0 < E < m_chi, else 0."""
        e = np.asarray(e, dtype=float)
        return np.where((e > 0.0) & (e < self._m_chi), 2.0 / self._m_chi, 0.0)

    def density(self, e, dec=None, ra=None):
        psi     = _psi_from_radec(ra, dec)
        spatial = _j_factor(psi, self._rho_s, self._r_s, self._r_sun, self._d_max)
        return self._spectrum(e) * spatial


# ---------------------------------------------------------------------------
# Step 2–3: ExtendedSource subclass with uses_ra = True
# ---------------------------------------------------------------------------

class NFWAnnihilationSource(ExtendedSource):
    """
    Neutrino source from DM annihilation in an NFW halo.

    Sets uses_ra = True so that ExtendedSourceEventSampler evaluates the flux
    at every (dec, ra) grid cell rather than broadcasting a dec-only template.
    """

    uses_ra: bool = True

    def __init__(self, m_chi_gev, sigma_v_cm3s, rho_s=0.3, r_s=20.0, r_sun=8.5, d_max_kpc=200.0):
        prefactor   = sigma_v_cm3s / (8.0 * np.pi * m_chi_gev**2)
        distribution = NFWAnnihilationDistribution(
            m_chi_gev, emin=0.0, emax=m_chi_gev,
            rho_s=rho_s, r_s=r_s, r_sun=r_sun, d_max_kpc=d_max_kpc,
        )
        flux = Flux(
            {nu: prefactor for nu in neutrinos},
            {nu: distribution for nu in neutrinos},
        )
        super().__init__(flux)


# ---------------------------------------------------------------------------
# Step 4: sample events
# ---------------------------------------------------------------------------

RESOURCES    = Path(__file__).parent.parent / "resources"
RESPONSE_FILE = str(RESOURCES / "icecube_10yr_response.h5")

config = {
    "properties": {"latitude": -90.0, "longitude": 0.0, "medium": "Ice"},
    "response":   {"detector_response_file": RESPONSE_FILE},
}
detector = Detector.from_config(config)

source  = NFWAnnihilationSource(
    m_chi_gev    = 1e4,    # 10 TeV DM
    sigma_v_cm3s = 3e-26,  # canonical thermal relic cross section
)

sampler = ExtendedSourceEventSampler(detector, source, n_dec=20, n_ra=21, n_e=20)

T      = ureg.Quantity(10, "year")
events = sampler.sample_events("track", deltat=T)
print(f"Sampled {len(events)} track events over {T}")
