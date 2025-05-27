import numpy as np

from ..conventions import SkyCoordinate, sample_cone
from ..detector import Detector

def new_sample(bounds):
    return np.array([np.random.uniform(lb, ub) for lb, ub in bounds])

def smear_truth(
    true_direction: SkyCoordinate,
    true_energy: float,
    detector: Detector,
    morphology: str
):
    if morphology not in "track cascade".split():
        raise ValueError(f"Invalid morphology {morphology}")
    ang_sampler = detector.response.angular_response[morphology]
    e_sampler = detector.response.energy_response[morphology]
    reco_energy = np.exp(e_sampler(np.random.rand())) * true_energy
    psi = ang_sampler(true_energy, np.random.rand())
    reco_direction = sample_cone(true_direction, psi)
    return reco_direction, reco_energy
