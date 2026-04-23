"""
Serialization helpers for lists of Event objects.

Primary format: HDF5 via h5py (always available).
Each call to write_events stores one group per event list; multiple lists
(e.g., signal + background) can be written to the same file under different
group names.

Quick usage
-----------
    from spore.event_sampling.io import write_events, read_events

    write_events(signal_events, "run.h5", group="signal")
    write_events(bg_events,     "run.h5", group="background")

    signal = read_events("run.h5", group="signal")
    bg     = read_events("run.h5", group="background")

Pandas / parquet
----------------
    import pandas as pd
    df = pd.DataFrame([e.to_dict() for e in events])
    df.to_parquet("events.parquet")
    # round-trip:
    df = pd.read_parquet("events.parquet")
"""

import numpy as np
import h5py as h5

from .event import Event
from ..conventions import SkyCoordinate

_FLOAT_COLS = (
    "true_dec", "true_ra",
    "reco_dec", "reco_ra",
    "true_energy", "reco_energy",
    "time", "ang_err",
    "zenith", "azimuth",
)
_STR_COLS = ("morphology", "detector_id")


def write_events(events, path: str, group: str = "events", mode: str = "a") -> None:
    """
    Write a list of Event objects to an HDF5 file.

    Parameters
    ----------
    events : list[Event]
        Events to serialize.
    path : str
        HDF5 file path.
    group : str
        HDF5 group name to write into.  Multiple groups can coexist in one
        file (e.g. "signal", "background").  An existing group of the same
        name is overwritten.
    mode : str
        h5py open mode.  "a" appends/creates; "w" truncates the whole file.
    """
    if len(events) == 0:
        # Write an empty group so the file is still valid
        with h5.File(path, mode) as f:
            if group in f:
                del f[group]
            f.create_group(group).attrs["n_events"] = 0
        return

    rows = [e.to_dict() for e in events]

    with h5.File(path, mode) as f:
        if group in f:
            del f[group]
        gp = f.create_group(group)
        gp.attrs["n_events"] = len(rows)

        for col in _FLOAT_COLS:
            gp.create_dataset(col, data=np.array([r[col] for r in rows], dtype=np.float64))
        dt = h5.string_dtype()
        for col in _STR_COLS:
            gp.create_dataset(
                col,
                data=np.array([str(r[col]) for r in rows], dtype=object),
                dtype=dt,
            )


def list_groups(path: str):
    """Return the names of event groups stored in an HDF5 file."""
    with h5.File(path, "r") as f:
        return [k for k in f.keys() if "n_events" in f[k].attrs]


def read_events(path: str, group: str = "events"):
    """
    Read a list of Event objects from an HDF5 file written by write_events.

    Parameters
    ----------
    path : str
        HDF5 file path.
    group : str
        HDF5 group name to read from.

    Returns
    -------
    list[Event]
    """
    with h5.File(path, "r") as f:
        gp = f[group]
        n = gp.attrs["n_events"]
        if n == 0:
            return []

        true_decs     = gp["true_dec"][:]
        true_ras      = gp["true_ra"][:]
        reco_decs     = gp["reco_dec"][:]
        reco_ras      = gp["reco_ra"][:]
        true_energies = gp["true_energy"][:]
        reco_energies = gp["reco_energy"][:]
        times         = gp["time"][:]
        ang_errs      = gp["ang_err"][:] if "ang_err" in gp else np.zeros(n)
        zeniths       = gp["zenith"][:]   if "zenith"  in gp else np.full(n, np.nan)
        azimuths      = gp["azimuth"][:]  if "azimuth" in gp else np.full(n, np.nan)
        morphologies  = gp["morphology"][:]
        detector_ids  = gp["detector_id"][:]

    def _decode(v):
        return v.decode() if isinstance(v, bytes) else str(v)

    def _decode_id(v):
        s = v.decode() if isinstance(v, bytes) else str(v)
        try:
            return int(s)
        except ValueError:
            return s

    events = []
    for i in range(n):
        events.append(Event(
            true_direction=SkyCoordinate(float(true_decs[i]), float(true_ras[i])),
            reco_direction=SkyCoordinate(float(reco_decs[i]), float(reco_ras[i])),
            true_energy=float(true_energies[i]),
            reco_energy=float(reco_energies[i]),
            time=float(times[i]),
            morphology=_decode(morphologies[i]),
            detector_id=_decode_id(detector_ids[i]),
            ang_err=float(ang_errs[i]),
            zenith=float(zeniths[i]),
            azimuth=float(azimuths[i]),
        ))
    return events
