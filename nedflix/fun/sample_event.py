import numpy as np

from scipy.integrate import quad
from typing import List, Dict

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

def get_new_events(
    source: Source,
    detector: Detector,
    nu: Neutrino,
    interaction: InteractionType,
    nevent: int,
) -> List[Event]:
    ang_sampler = detector.response.angular_response[(nu, interaction)]
    e_sampler = detector.response.energy_response[(nu, interaction)]
    events = []
    for idx in range(nevent):
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
        events.append(event)
    return events

        

def sample_events(
    detector: Detector,
    source: Source,
    deltat: float,
    flavor_fraction_dict: Dict=equal_fraction_dict,
    xs: CrossSection=None
) -> List[Event]:
    if deltat < units.day:
        from warnings import warn
        warn("`deltat` is less than one day, but code currently assumes daily averaged effective area")
    if xs is None:
        from ..physics import DummyCrossSection
        xs = DummyCrossSection()
    events = []
    for nu in Neutrino:
        flavor_fraction = flavor_fraction_dict[nu]
        for interaction in InteractionType:
    
            p_interaction = lambda e: xs[interaction][0](e) / (xs[InteractionType.NeutralCurrent][0](e) + xs[InteractionType.ChargedCurrent][0](e))
            effa_f = lambda e: detector.response.effective_area[(nu, interaction)](source.location.declination, e)
            f = lambda e: source.flux(e) * effa_f(e) * p_interaction(e) * flavor_fraction
            i = lambda le: np.exp(le) * f(np.exp(le))
            I, err = quad(
                i,
                np.log(source.flux.energy_distribution.emin),
                np.log(source.flux.energy_distribution.emax)
            )
            nevent = np.random.poisson(I * deltat)
            new_events = get_new_events(source, detector, nu, interaction, nevent)
            events += new_events
    return events
