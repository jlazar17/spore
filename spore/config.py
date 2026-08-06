"""Configuration loading shared by the ``from_config`` constructors.

Every component is configured from a plain mapping.  The same constructors
also accept a path to a TOML file holding that mapping, so a source, flux, or
detector can be described in a file and version-controlled alongside an
analysis.  TOML parsing uses :mod:`tomllib` from the standard library, so this
adds no dependency on the supported Python versions (>= 3.11).
"""
import os
import tomllib
from pathlib import Path
from typing import Mapping, Optional, Union

ConfigLike = Union[Mapping, str, "os.PathLike[str]"]


def load_config(config: ConfigLike, section: Optional[str] = None) -> dict:
    """Return a config mapping, reading it from a TOML file when given a path.

    Args:
        config: A mapping, which is returned as a plain dict, or a path to a
            TOML file to parse.
        section: Optional table name to unwrap.  When the parsed document has
            a table of this name at the top level, that table is returned
            instead of the whole document.  Lets a flux be given either as a
            bare TOML file or as the ``[flux]`` table of a source file.

    Returns:
        The configuration as a dict.

    Raises:
        FileNotFoundError: If a path is given that does not exist.
        TypeError: If *config* is neither a mapping nor a path.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    if isinstance(config, Mapping):
        return dict(config)

    if isinstance(config, (str, Path, os.PathLike)):
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"No such configuration file: {path}")
        with open(path, "rb") as f:
            parsed = tomllib.load(f)
        if section is not None and isinstance(parsed.get(section), Mapping):
            return dict(parsed[section])
        return parsed

    raise TypeError(
        "Configuration must be a mapping or a path to a TOML file, got "
        f"{type(config).__name__}."
    )
