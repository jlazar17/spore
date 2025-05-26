import numpy as np

from scipy.integrate import quad
from typing import List, Dict, Tuple

from ..physics import CrossSection, Neutrino, InteractionType
from ..conventions import Event, units, sample_cone
from ..source import Source
from ..detector import Detector

equal_fraction_dict = {
    Neutrino.NuE: 1 / 6,
    Neutrino.NuMu: 1 / 6,
    Neutrino.NuTau: 1 / 6,
    Neutrino.NuEBar: 1 / 6,
    Neutrino.NuMuBar: 1 / 6,
    Neutrino.NuTauBar: 1 / 6,
}

def is_numu(nu: Neutrino):
    """
    Find out if neutrino is muonflavored

    params
    ______
    nu: Neutrino object to check

    returns
    _______
    b: whether neutrino is muon flavored
    """
    return abs(int(nu))==14

# TODO this assumes that \sigma_{nu}==\sigma_{\nubar}
#def sample_track_type(
#    ff_dict: Dict
#) -> Tuple[Neutrino, InteractionType]:
#    u = np.random.rand()
#    r = ff_dict[Neutrino.NuMu] / sum([f for nu,f in ff_dict.items() if is_numu(nu)])
#    if u < r:
#        return Neutrino.NuMu, InteractionType.ChargedCurrent
#    else:
#        return Neutrino.NuMuBar, InteractionType.ChargedCurrent

# TODO this assumes that \sigma_{nu}==\sigma_{\nubar} and \sigma{CC}==3\sigma{NC}
#def sample_cascade_type(
#    ff_dict: Dict[Neutrino, float]
#) -> Tuple[Neutrino, InteractionType]:
#
#    t, a = 0, []
#    for nu, f in ff_dict.items():
#        a.append((nu, t))
#        r = 4 / 3
#        if abs(int(nu))==14:
#            r = 1/3
#        t += f * r
#    a = [(nu, x/t) for nu, x in a]
#
#    u, v = np.random.rand(), 1
#    while u < v:
#        nu, v = a.pop()
#
#    interaction = InteractionType.NeutralCurrent
#    u = np.random.rand()
#    if not is_numu(nu) and u > 0.25:
#        interaction = InteractionType.ChargedCurrent
#
#    return nu, interaction


def get_

def get_new_event(
    source: Source,
    detector: Detector,
    nu: Neutrino,
    interaction: InteractionType,
) -> List[Event]:
    ang_sampler = detector.response.angular_response[(nu, interaction)]
    e_sampler = detector.response.energy_response[(nu, interaction)]
    true_energy = source.flux.sample_energy()
    reco_energy = np.exp(e_sampler(np.random.rand())) * true_energy
    true_dir = source.location
    psi = ang_sampler(true_energy, np.random.rand())
    reco_dir = sample_cone(true_dir, psi)
    event = Event(
        interaction,
        nu,
        true_dir,
        reco_dir,
        true_energy,
        reco_energy
    )
    return event


def sample_events(
    detector: Detector,
    source: Source,
    deltat: float,
    flavor_fraction_dict: Dict=None,
    xs: CrossSection=None
) -> List[Event]:
    if deltat < units.day:
        from warnings import warn
        warn("`deltat` is less than one day, but code currently assumes daily averaged effective area")
    if xs is None:
        from ..physics import DummyCrossSection
        xs = DummyCrossSection()

    if flavor_fraction_dict is None:
        flavor_fraction_dict = equal_fraction_dict

    events = []
    for morphology, sampler in zip(["track", "cascade"], [sample_track_type, sample_cascade_type]):
        effa_f = lambda e: detector.response.effective_area[morphology](source.location.declination, e)
        f = lambda e: source.flux(e) * effa_f(e)
        i = lambda le: np.exp(le) * f(np.exp(le))
        I, err = quad(
            i,
            np.log(source.flux.energy_distribution.emin),
            np.log(source.flux.energy_distribution.emax)
        )
        nevent = np.random.poisson(I * deltat)
        types = [sampler(flavor_fraction_dict) for _ in range(nevent)]
        for nu, interaction in types:
            event = get_new_event(source, detector, nu, interaction)
            events.append(event)
    return events
