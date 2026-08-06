"""
Compute and cache event skymap data for two detectors.

Samples 200 atmospheric track events from IceCube (South Pole) and KM3NeT
(Mediterranean) in both instantaneous and diurnal-average modes, then writes
the reconstructed (RA, sin δ) arrays to resources/plotting_data.h5 under
the group ``multidetector_skymap``.

Plotting is handled by paper/plots.py.
"""
import logging
import os
import numpy as np
import h5py as h5

from spore.conventions import ureg
from spore.detector import Detector
from spore.source import ExtendedSource
from spore.event_sampling import SourceSampler

logging.basicConfig(level=logging.INFO)

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
ATM_H5  = os.path.join(REPO, "resources", "atmo_flux_models.h5")
OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")

N_EVENTS = 200


def _ras(events):
    return np.array([ev.reco_direction.right_ascension for ev in events])


def _sindecs(events):
    return np.sin([ev.reco_direction.declination for ev in events])


if __name__ == "__main__":
    south_polar = Detector.from_config({
        "properties": {"latitude": -90.0, "longitude": 0.0, "depth": 1945, "medium": "Ice"},
        "response": {"detector_response_file": os.path.join(REPO, "resources", "configs", "ps10yr_detector_response.h5")},
    })

    mediterranean = Detector.from_config({
        "properties": {"latitude": 36.3, "longitude": 16.1, "depth": 3500, "medium": "Water"},
        "response": {"detector_response_file": os.path.join(REPO, "resources", "configs", "ps10yr_detector_response.h5")},
    })

    atmo_src = ExtendedSource.from_config(
        {"flux": {"location": f"{ATM_H5}:mceq_h4a_sibyll23d"}}
    )

    ic_quick_atmo_sampler   = SourceSampler(south_polar, atmo_src, n_dec=100, n_ra=100, n_e=100)
    km3_quick_atmo_sampler  = SourceSampler(mediterranean,  atmo_src, n_dec=100, n_ra=100, n_e=100)
    ic_steady_atmo_sampler  = SourceSampler(south_polar, atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
    km3_steady_atmo_sampler = SourceSampler(mediterranean,  atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)

    logging.info("Sampling instantaneous events")
    ic_inst  = ic_quick_atmo_sampler.sample_events("track",  nevent=N_EVENTS)
    km3_inst = km3_quick_atmo_sampler.sample_events("track", nevent=N_EVENTS)

    logging.info("Sampling diurnal-average events")
    ic_steady  = ic_steady_atmo_sampler.sample_events("track",  nevent=N_EVENTS)
    km3_steady = km3_steady_atmo_sampler.sample_events("track", nevent=N_EVENTS)

    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w"):
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "multidetector_skymap" in h5f:
            del h5f["multidetector_skymap"]
        gp = h5f.create_group("multidetector_skymap")
        gp["ic_instant_ra"]     = _ras(ic_inst)
        gp["ic_instant_sindec"] = _sindecs(ic_inst)
        gp["km3_instant_ra"]    = _ras(km3_inst)
        gp["km3_instant_sindec"] = _sindecs(km3_inst)
        gp["ic_steady_ra"]      = _ras(ic_steady)
        gp["ic_steady_sindec"]  = _sindecs(ic_steady)
        gp["km3_steady_ra"]     = _ras(km3_steady)
        gp["km3_steady_sindec"] = _sindecs(km3_steady)

    logging.info("Wrote multidetector_skymap to %s", OUTFILE)
