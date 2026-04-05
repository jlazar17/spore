"""
Neutrino source from dark matter annihilation in the Galactic halo.

The differential neutrino flux per unit solid angle is:

    dΦ/dE dΩ = (σv / (8π m_χ²)) × (dN/dE)(E) × J(RA, Dec)

where J is the line-of-sight integral of ρ² (the J-factor) and dN/dE
is the neutrino energy spectrum per annihilation for a given channel.

This source has a 2D angular dependence — the J-factor varies across
the sky because the Galactic Centre is not at the celestial pole.
Use GalacticHaloEventSampler rather than ExtendedSourceEventSampler to
correctly account for this.

Units
-----
DM parameters are accepted in conventional astrophysical units:
    m_chi   [GeV]
    sigma_v [cm³ s⁻¹]
The flux returned by __call__ is in spore natural units (eV basis):
    [eV⁻¹ eV² eV] = [eV²]  (differential flux per solid angle)

Spectral models
---------------
The `spectrum` argument must be a callable f(e_eV) returning dN/dE in
units of eV⁻¹ (number of ν per annihilation per eV).  Two built-in
models are provided:

    BoxSpectrum(m_chi_eV)
        Flat spectrum from 0 to m_chi; appropriate for χχ → νν̄ smeared
        over detector energy resolution.

    SoftSpectrum(m_chi_eV, a=1.5, b=1.0)
        Power-law with kinematic cutoff: ∝ (E/m_χ)^a (1 − E/m_χ)^b.
        Suitable as a rough proxy for hadronic channels.  Normalised so
        that ∫₀^{m_chi} dN/dE dE = n_nu.
"""

import numpy as np
from scipy.integrate import quad

from ..conventions.units import units
from ..physics import Neutrino, neutrinos
from .source import Source
from .jfactor import j_factor, psi_from_radec, NFWProfile, EinastoProfile

# Conventional unit conversions to eV
_GEV_TO_EV = 1.0e9
_CM3_PER_S_TO_EV_M2 = units.cm**3 / units.sec   # [eV⁻²] per [cm³ s⁻¹]

# J-factor unit conversion: GeV² cm⁻⁵ → eV⁵ (all in natural units)
# GeV² → (1e9 eV)² = 1e18 eV²
# cm⁻⁵ → (eV / units.cm)⁵  [since 1 cm = units.cm eV⁻¹ → cm⁻¹ = (units.cm)⁻¹ eV]
_J_GEV2_CM5_TO_EV5 = (1.0e9) ** 2 * (1.0 / units.cm) ** 5


# ---------------------------------------------------------------------------
# Spectral models
# ---------------------------------------------------------------------------

class LineSpectrum:
    """
    Monochromatic (delta-function) neutrino spectrum for χχ → νν̄.

    Each annihilation produces ``n_nu`` neutrinos at a fixed energy ``E_0``.
    Because the spectrum is a true delta function, the energy dimension is
    removed from the MCMC entirely: the sampler draws directions only and
    assigns every event the true energy ``E_0``.

    Parameters
    ----------
    E_0_eV : float
        Injection energy in eV.  For the direct channel χχ → νν̄, set this
        to ``m_chi_eV`` (Majorana fermion) or ``m_chi_eV / 2`` if needed.
    n_nu : float
        Number of neutrinos per annihilation. Default 2.
    """

    is_monochromatic: bool = True

    def __init__(self, E_0_eV: float, n_nu: float = 2.0):
        self.E_0 = E_0_eV
        self.n_nu = n_nu

    def __call__(self, e_eV: float) -> float:
        raise NotImplementedError(
            "LineSpectrum cannot be evaluated pointwise. "
            "Use GalacticHaloEventSampler, which handles monochromatic "
            "injection by fixing the energy to E_0 and sampling only in "
            "direction space."
        )


class BoxSpectrum:
    """
    Flat (box) neutrino spectrum for the direct channel χχ → νν̄.

    Each annihilation produces exactly 2 neutrinos at E = m_chi (rest-frame
    monoenergetic).  In practice detector energy resolution smears this; here
    we represent it as a uniform distribution from 0 to m_chi as a simple
    proxy for studies where the exact line shape is not critical.

    Parameters
    ----------
    m_chi_eV : float
        DM mass in eV.
    n_nu : float
        Number of neutrinos per annihilation. Default 2.
    """

    def __init__(self, m_chi_eV: float, n_nu: float = 2.0):
        self.m_chi_eV = m_chi_eV
        self.n_nu = n_nu
        self._norm = n_nu / m_chi_eV   # dN/dE so that ∫ dE = n_nu

    def __call__(self, e_eV: float) -> float:
        """dN/dE at energy e_eV [eV⁻¹]."""
        if e_eV <= 0.0 or e_eV >= self.m_chi_eV:
            return 0.0
        return self._norm


class SoftSpectrum:
    r"""
    Parametric soft neutrino spectrum for hadronic/leptonic channels.

    Shape: dN/dE ∝ (E / m_χ)^a (1 − E / m_χ)^b  for 0 < E < m_χ.

    Normalised so that ∫₀^{m_chi} dN/dE dE = n_nu.

    Parameters
    ----------
    m_chi_eV : float
        DM mass in eV.
    a : float
        Low-energy power index. Must be > −1 for integrability. Default 1.5
        (approximate for bb̄ channel at m_chi ≳ 100 GeV).
    b : float
        Kinematic suppression index near cutoff. Default 1.0.
    n_nu : float
        Mean number of neutrinos (of one flavour) per annihilation.
        Default 1.0; user should set based on channel and branching ratio.
    """

    def __init__(
        self,
        m_chi_eV: float,
        a: float = 1.5,
        b: float = 1.0,
        n_nu: float = 1.0,
    ):
        self.m_chi_eV = m_chi_eV
        self.a = a
        self.b = b
        unnorm, _ = quad(
            lambda e: (e / m_chi_eV) ** a * (1.0 - e / m_chi_eV) ** b,
            0.0,
            m_chi_eV,
            limit=200,
        )
        self._norm = n_nu / unnorm if unnorm > 0 else 1.0

    def __call__(self, e_eV: float) -> float:
        """dN/dE at energy e_eV [eV⁻¹]."""
        if e_eV <= 0.0 or e_eV >= self.m_chi_eV:
            return 0.0
        return self._norm * (e_eV / self.m_chi_eV) ** self.a * (1.0 - e_eV / self.m_chi_eV) ** self.b


# ---------------------------------------------------------------------------
# Source class
# ---------------------------------------------------------------------------

class GalacticHaloSource:
    """
    Neutrino source from dark matter annihilation in the Galactic halo.

    Parameters
    ----------
    profile : NFWProfile or EinastoProfile
        DM density profile providing a ``density(r_kpc)`` method.
    spectrum : callable
        f(e_eV) → dN/dE [eV⁻¹] per annihilation, for one neutrino flavour
        (or summed over all flavours if ``same_for_all_flavours=True``).
    m_chi_gev : float
        DM particle mass [GeV].
    sigma_v_cm3s : float
        Thermally averaged annihilation cross section [cm³ s⁻¹].
    same_for_all_flavours : bool
        If True (default), the same spectrum is applied to all six neutrino
        flavours.  Set to False and pass a dict as ``spectrum`` if the
        flavour composition differs across channels.
    d_max_kpc : float
        Line-of-sight truncation for J-factor integrals [kpc]. Default 200.

    Notes
    -----
    The returned flux when called as ``src(nu, e_eV, dec_rad, ra_rad)`` is
    the differential neutrino flux per unit solid angle in spore natural
    units (eV basis), consistent with the values expected by
    GalacticHaloEventSampler.
    """

    def __init__(
        self,
        profile,
        spectrum,
        m_chi_gev: float,
        sigma_v_cm3s: float,
        same_for_all_flavours: bool = True,
        d_max_kpc: float = 200.0,
    ):
        self._profile = profile
        self._d_max_kpc = d_max_kpc

        # Convert parameters to natural units
        m_chi_eV = m_chi_gev * _GEV_TO_EV
        sigma_v_eV2 = sigma_v_cm3s * _CM3_PER_S_TO_EV_M2

        # Pre-factor: σv / (8π m_chi²)  [eV⁻²]
        self._prefactor = sigma_v_eV2 / (8.0 * np.pi * m_chi_eV**2)

        # Spectral callables keyed by Neutrino
        if same_for_all_flavours:
            self._spectra = {nu: spectrum for nu in neutrinos}
        else:
            if not isinstance(spectrum, dict):
                raise ValueError(
                    "If same_for_all_flavours=False, spectrum must be a dict "
                    "mapping Neutrino → callable."
                )
            self._spectra = spectrum

    @property
    def is_monochromatic(self) -> bool:
        """True if the source spectrum is a delta function in energy."""
        spectrum = next(iter(self._spectra.values()))
        return getattr(spectrum, "is_monochromatic", False)

    @property
    def E_0_eV(self) -> float:
        """Injection energy [eV] for monochromatic sources."""
        if not self.is_monochromatic:
            raise AttributeError("Source is not monochromatic; E_0_eV is undefined.")
        return next(iter(self._spectra.values())).E_0

    def __call__(self, nu: Neutrino, e_eV: float, dec_rad: float, ra_rad: float) -> float:
        """
        Differential neutrino flux per solid angle at (nu, E, RA, Dec).

        Parameters
        ----------
        nu : Neutrino
        e_eV : float
            Neutrino energy [eV].
        dec_rad : float
            Declination [radians].
        ra_rad : float
            Right ascension [radians].

        Returns
        -------
        float
            dΦ/dE dΩ in spore natural units (eV basis).
        """
        psi = psi_from_radec(ra_rad, dec_rad)
        j = j_factor(self._profile, psi, d_max_kpc=self._d_max_kpc)
        # Convert J from GeV² cm⁻⁵ to natural units (eV⁵)
        j_nat = j * _J_GEV2_CM5_TO_EV5
        dNdE = self._spectra[nu](e_eV)
        return self._prefactor * dNdE * j_nat
