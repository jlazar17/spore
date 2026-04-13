"""
Compute and cache effective area data for two detectors.

RA-averages the track effective area over 100 random pointings at five
declinations for IceCube (South Pole) and KM3NeT (Mediterranean), then writes
the results to resources/plotting_data.h5 under the group ``effective_area``.

Requires: pip install colorspacious

Plotting is handled by paper/plots.py.
"""
import logging
import os
import numpy as np
import h5py as h5

from spore.conventions import ureg, SkyCoordinate, sky_to_local
from spore.detector import Detector

logging.basicConfig(level=logging.INFO)

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")

icecube = Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "depth": 1945, "medium": "Ice"},
    "response": {"detector_response_toml": os.path.join(REPO, "resources", "configs", "ps10yr_response.toml")},
})

km3net = Detector.from_config({
    "properties": {"latitude": 36.3, "longitude": 16.1, "depth": 3500, "medium": "Water"},
    "response": {"detector_response_toml": os.path.join(REPO, "resources", "configs", "ps10yr_response.toml")},
})

T_MJD    = 60_355.83
ES       = np.logspace(2, 6, 41)           # 100 GeV – 1 PeV
DECS_RAD = np.radians([90, 45, 0, -45, -90])
N_RA     = 100

if __name__ == "__main__":
    effas = np.zeros((2, len(DECS_RAD), len(ES)))

    for idx, dec_rad in enumerate(DECS_RAD):
        logging.info("Effective area: declination %d / %d (%.0f deg)", idx + 1, len(DECS_RAD), np.degrees(dec_rad))
        for _ in range(N_RA):
            ra = np.random.rand() * 2 * np.pi
            sc = SkyCoordinate(dec_rad, ra)
            ic_local  = sky_to_local(sc, icecube.location, T_MJD)
            km3_local = sky_to_local(sc, km3net.location,  T_MJD)
            effas[0, idx, :] += icecube.response.effective_area["track"](ic_local.zenith,  ES)
            effas[1, idx, :] += km3net.response.effective_area["track"](km3_local.zenith, ES)
    effas /= N_RA

    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w"):
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "effective_area" in h5f:
            del h5f["effective_area"]
        gp = h5f.create_group("effective_area")
        gp["effas"]    = effas       # shape (2, n_dec, n_e)
        gp["es"]       = ES
        gp["decs_rad"] = DECS_RAD

    logging.info("Wrote effective_area to %s", OUTFILE)
