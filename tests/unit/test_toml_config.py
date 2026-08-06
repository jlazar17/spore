"""TOML configuration loading, as documented in the paper's appendix."""
import numpy as np
import pytest

from spore.config import load_config
from spore.detector import Detector
from spore.physics import neutrinos
from spore.source import PointSource
from spore.source.flux import Flux

# Verbatim from the TOML Configuration Reference appendix.
PAPER_EXAMPLE = """[location]
right_ascension = 50.0   # degrees, J2000
declination     =  5.0   # degrees, J2000

[flux]
norm_per_species = 1e-18  # GeV^-1 cm^-2 s^-1 per species
gamma      = 2.0
pivot      = 1e5          # GeV
emin       = 1e2          # GeV
emax       = 1e6          # GeV
"""


@pytest.fixture
def paper_toml(tmp_path):
    p = tmp_path / "source.toml"
    p.write_text(PAPER_EXAMPLE)
    return str(p)


class TestLoadConfig:
    def test_mapping_passes_through(self):
        assert load_config({"a": 1}) == {"a": 1}

    def test_section_unwrapped_when_present(self, paper_toml):
        assert load_config(paper_toml, section="flux")["gamma"] == 2.0

    def test_section_ignored_when_absent(self, tmp_path):
        p = tmp_path / "bare.toml"
        p.write_text("gamma = 3.0\n")
        assert load_config(str(p), section="flux")["gamma"] == 3.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/no/such/file.toml")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            load_config(42)


class TestConstructorsAcceptToml:
    def test_point_source(self, paper_toml):
        src = PointSource.from_config(paper_toml)
        assert np.degrees(src.location.declination) == pytest.approx(5.0)
        assert np.degrees(src.location.right_ascension) == pytest.approx(50.0)
        # Power law normalised per species at the pivot.
        assert src(neutrinos[2], 1e5) == pytest.approx(1e-18)

    def test_flux_unwraps_flux_table(self, paper_toml):
        assert Flux.from_config(paper_toml)(neutrinos[2], 1e5) == pytest.approx(1e-18)

    def test_detector(self, tmp_path, synthetic_detector_response_file):
        p = tmp_path / "det.toml"
        p.write_text(
            '[properties]\n'
            'latitude = -90.0\n'
            'longitude = 0.0\n'
            'medium = "Ice"\n\n'
            '[response]\n'
            f'detector_response_file = "{synthetic_detector_response_file}"\n'
        )
        det = Detector.from_config(str(p))
        assert np.degrees(det.location.latitude) == pytest.approx(-90.0)
        assert "track" in det.response.available_morphologies

    def test_dict_and_toml_agree(self, paper_toml):
        from_toml = PointSource.from_config(paper_toml)
        from_dict = PointSource.from_config({
            "location": {"right_ascension": 50.0, "declination": 5.0},
            "flux": {"norm_per_species": 1e-18, "gamma": 2.0,
                     "pivot": 1e5, "emin": 1e2, "emax": 1e6},
        })
        for e in (1e3, 1e5, 1e6):
            assert from_toml(neutrinos[2], e) == pytest.approx(from_dict(neutrinos[2], e))
