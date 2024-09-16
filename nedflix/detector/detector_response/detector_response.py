import numpy as np
import h5py as h5

from dataclasses import dataclass
from typing import Tuple, Callable, Dict
from scipy.interpolate import RegularGridInterpolator, interp1d

from .. import units
from .. import Neutrino, InteractionType

@dataclass(frozen=True)
class DetectorResponse:
    effective_area: Dict[Tuple[Neutrino, InteractionType], Callable]
    angular_response: Dict[Tuple[Neutrino, InteractionType], Callable]
    energy_response: Dict[Tuple[Neutrino, InteractionType], Callable]

def effa_spline_from_group(gp: h5.Group) -> Callable:
    from .utils import effa_helper
    decs = gp["declinations"]
    es = gp["es"]
    tabulated_values = gp["tabulated_values"]
    lower_bounds = gp["lower_bounds"]
    upper_bounds = gp["upper_bounds"]
    effa_fxn = effa_helper(decs[:], es[:], tabulated_values[:], lower_bounds[:], upper_bounds[:])
    return effa_fxn

def ang_spline_from_group(gp: h5.Group) -> Callable:
    i = RegularGridInterpolator(
        (np.log(gp["es"][:]), gp["us"][:]),
        gp["inv_cdfs"][:]
    )
    fxn = lambda e, u: i((np.log(e), u))
    return fxn

def energy_spline_from_group(gp: h5.Group) -> Callable:
    i = interp1d(
        gp["us"][:],
        gp["inv_cdf"][:]
    )
    return i

def detector_response_from_config(config: Dict) -> DetectorResponse:

    with h5.File(config["detector_response_file"]) as h5f:
        track_effa = effa_spline_from_group(h5f["track_effective_area"])
        cscd_effa = effa_spline_from_group(h5f["cascade_effective_area"])
        track_ang = ang_spline_from_group(h5f["track_angular_response"])
        cscd_ang = ang_spline_from_group(h5f["cascade_angular_response"])
        track_energy = energy_spline_from_group(h5f["track_energy_resolution"])
        cscd_energy = energy_spline_from_group(h5f["cascade_energy_resolution"])

    # I know this is psycho. Don't @ me
    effective_area = {
        (Neutrino.NuE, InteractionType.ChargedCurrent): cscd_effa,
        (Neutrino.NuMu, InteractionType.ChargedCurrent): track_effa,
        (Neutrino.NuTau, InteractionType.ChargedCurrent): cscd_effa,
        (Neutrino.NuEBar, InteractionType.ChargedCurrent): cscd_effa,
        (Neutrino.NuMuBar, InteractionType.ChargedCurrent): track_effa,
        (Neutrino.NuTauBar, InteractionType.ChargedCurrent): cscd_effa,
        (Neutrino.NuE, InteractionType.NeutralCurrent): cscd_effa,
        (Neutrino.NuMu, InteractionType.NeutralCurrent): cscd_effa,
        (Neutrino.NuTau, InteractionType.NeutralCurrent): cscd_effa,
        (Neutrino.NuEBar, InteractionType.NeutralCurrent): cscd_effa,
        (Neutrino.NuMuBar, InteractionType.NeutralCurrent): cscd_effa,
        (Neutrino.NuTauBar, InteractionType.NeutralCurrent): cscd_effa,
    }

    angular_response = {
        (Neutrino.NuE, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuMu, InteractionType.ChargedCurrent): track_ang,
        (Neutrino.NuTau, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuEBar, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuMuBar, InteractionType.ChargedCurrent): track_ang,
        (Neutrino.NuTauBar, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuE, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuMu, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuTau, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuEBar, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuMuBar, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuTauBar, InteractionType.NeutralCurrent): cscd_ang,
    }

    energy_response = {
        (Neutrino.NuE, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuMu, InteractionType.ChargedCurrent): track_ang,
        (Neutrino.NuTau, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuEBar, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuMuBar, InteractionType.ChargedCurrent): track_ang,
        (Neutrino.NuTauBar, InteractionType.ChargedCurrent): cscd_ang,
        (Neutrino.NuE, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuMu, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuTau, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuEBar, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuMuBar, InteractionType.NeutralCurrent): cscd_ang,
        (Neutrino.NuTauBar, InteractionType.NeutralCurrent): cscd_ang,
    }

    return DetectorResponse(effective_area, angular_response, energy_response)
