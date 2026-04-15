"""
Compute and cache point source ψ² template data.

Samples 10 years of IceCube atmospheric and astrophysical background events
plus signal events from a soft power-law point source, builds signal and
background ψ² templates from 1000 pseudo-experiments, then writes everything
to resources/plotting_data.h5 under the group ``point_source_demo``.

Plotting is handled by paper/plots.py.
"""
import logging
import os
import numpy as np
import h5py as h5

from tqdm import tqdm

from spore.conventions import ureg, SkyCoordinate
from spore.detector import Detector
from spore.source import PointSource, ExtendedSource
from spore.event_sampling import PointSourceEventSampler, ExtendedSourceEventSampler

logging.basicConfig(level=logging.INFO)

HERE    = os.path.abspath(os.path.dirname(__file__))
REPO    = os.path.join(HERE, "..", "..")
COMB_H5 = os.path.join(REPO, "resources", "ps10yr_combined_flux.h5")
OUTFILE = os.path.join(REPO, "resources", "plotting_data.h5")

det = Detector.from_config({
    "properties": {"latitude": -90.0, "longitude": 0.0, "depth": 1945, "medium": "Ice"},
    "response": {"detector_response_file": os.path.join(REPO, "resources", "configs", "ps10yr_detector_response.h5")},
})

# Benchmark point source: soft power law near NGC 1068
SOURCE_CONFIG = {
    "flux": {
        "norm":  2.5e-14,   # GeV^{-1} cm^{-2} s^{-1}
        "gamma": 3.2,
        "pivot": 1000.0,    # GeV
        "emin":  1e2,       # GeV
        "emax":  1e7,       # GeV
    },
    "location": {
        "right_ascension": 40.67,    # deg
        "declination":    -0.01329,  # deg
    },
}
SRC_SC = SkyCoordinate(
    np.radians(SOURCE_CONFIG["location"]["declination"]),
    np.radians(SOURCE_CONFIG["location"]["right_ascension"]),
)

src       = PointSource.from_config(SOURCE_CONFIG)
atmo_src  = ExtendedSource.from_config({"flux": {"location": f"{COMB_H5}:atmospheric"}})
astro_src = ExtendedSource.from_config({"flux": {"location": f"{COMB_H5}:astrophysical"}})

# IC86 10-year livetime (https://arxiv.org/pdf/2211.09972)
LIVETIME = ureg.Quantity(3_186, "day")

sampler       = PointSourceEventSampler(det, src, n_time_samples=50)
atmo_sampler  = ExtendedSourceEventSampler(det, atmo_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)
astro_sampler = ExtendedSourceEventSampler(det, astro_src, n_time_samples=50, n_dec=100, n_ra=100, n_e=100)

PSI2_BINS  = np.linspace(-1, 50, 154)
PSI2_CENTS = (PSI2_BINS[1:] + PSI2_BINS[:-1]) / 2
N_PSEUDO   = 1000


def _separations_rad(decs: np.ndarray, ras: np.ndarray,
                     src_dec: float = SRC_SC.declination,
                     src_ra: float  = SRC_SC.right_ascension) -> np.ndarray:
    """Vectorized great-circle separations from a fixed source to N events."""
    cos_s = np.cos(src_dec)
    ax = cos_s * np.cos(src_ra)
    ay = cos_s * np.sin(src_ra)
    az = np.sin(src_dec)
    cos_e = np.cos(decs)
    bx = cos_e * np.cos(ras)
    by = cos_e * np.sin(ras)
    bz = np.sin(decs)
    cross = np.sqrt((ay*bz - az*by)**2 + (az*bx - ax*bz)**2 + (ax*by - ay*bx)**2)
    return np.arctan2(cross, ax*bx + ay*by + az*bz)


def _event_coords(events):
    """Return (decs, ras) as plain float arrays from a list of Events."""
    decs = np.array([ev.reco_direction.declination     for ev in events])
    ras  = np.array([ev.reco_direction.right_ascension for ev in events])
    return decs, ras


if __name__ == "__main__":
    logging.info("Sampling background events")
    ngc_events   = sampler.sample_events("track",  deltat=LIVETIME)
    atmo_events  = atmo_sampler.sample_events("track", deltat=LIVETIME)
    astro_events = astro_sampler.sample_events("track", deltat=LIVETIME)
    all_events   = atmo_events + astro_events + ngc_events
    logging.info("Total background events: %d", len(all_events))

    # Pre-extract coordinates once — avoids per-iteration object overhead
    all_decs, all_ras = _event_coords(all_events)

    logging.info("Building signal template (%d pseudo-experiments)", N_PSEUDO)
    ngc_template = np.zeros(PSI2_CENTS.shape)
    for i in range(N_PSEUDO):
        if i % 100 == 0:
            logging.info("  signal pseudo-experiment %d / %d", i, N_PSEUDO)
        new_ngc = sampler.sample_events("track", deltat=LIVETIME)
        if new_ngc:
            decs, ras = _event_coords(new_ngc)
            psi2 = np.degrees(_separations_rad(decs, ras)) ** 2
            ngc_template += np.histogram(psi2, bins=PSI2_BINS)[0]
    ngc_template /= N_PSEUDO

    logging.info("Building background template (%d pseudo-experiments)", N_PSEUDO)
    bg_template = np.zeros(PSI2_CENTS.shape)
    for i in range(N_PSEUDO):
        if i % 100 == 0:
            logging.info("  background pseudo-experiment %d / %d", i, N_PSEUDO)
        rand_ras = np.random.rand(len(all_events)) * 2 * np.pi
        psi2 = np.degrees(_separations_rad(all_decs, rand_ras)) ** 2
        bg_template += np.histogram(psi2, bins=PSI2_BINS)[0]
    bg_template /= N_PSEUDO

    h_data, _ = np.histogram(
        np.degrees(_separations_rad(all_decs, all_ras)) ** 2, bins=PSI2_BINS
    )

    if not os.path.exists(OUTFILE):
        with h5.File(OUTFILE, "w"):
            pass

    with h5.File(OUTFILE, "r+") as h5f:
        if "point_source_demo" in h5f:
            del h5f["point_source_demo"]
        gp = h5f.create_group("point_source_demo")
        gp["psi2_bins"]    = PSI2_BINS
        gp["psi2_cents"]   = PSI2_CENTS
        gp["ngc_template"] = ngc_template
        gp["bg_template"]  = bg_template
        gp["h_data"]       = h_data

    logging.info("Wrote point_source_demo to %s", OUTFILE)
