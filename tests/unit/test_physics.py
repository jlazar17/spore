import pytest

from spore.physics import Neutrino, neutrinos
from spore.physics.neutrino import Flavor, NeutrinoType


class TestNeutrinoEnum:
    def test_six_members(self):
        assert len(Neutrino) == 6

    def test_pdg_ids(self):
        assert int(Neutrino.NuE) == 12
        assert int(Neutrino.NuMu) == 14
        assert int(Neutrino.NuTau) == 16
        assert int(Neutrino.NuEBar) == -12
        assert int(Neutrino.NuMuBar) == -14
        assert int(Neutrino.NuTauBar) == -16

    def test_particle_flavors(self):
        assert Neutrino.NuE.value.flavor == Flavor.Electron
        assert Neutrino.NuMu.value.flavor == Flavor.Muon
        assert Neutrino.NuTau.value.flavor == Flavor.Tau

    def test_antiparticle_flavors(self):
        assert Neutrino.NuEBar.value.flavor == Flavor.Electron
        assert Neutrino.NuMuBar.value.flavor == Flavor.Muon
        assert Neutrino.NuTauBar.value.flavor == Flavor.Tau

    def test_particle_types(self):
        assert Neutrino.NuE.value.nutype == NeutrinoType.Nu
        assert Neutrino.NuMu.value.nutype == NeutrinoType.Nu
        assert Neutrino.NuTau.value.nutype == NeutrinoType.Nu

    def test_antiparticle_types(self):
        assert Neutrino.NuEBar.value.nutype == NeutrinoType.NuBar
        assert Neutrino.NuMuBar.value.nutype == NeutrinoType.NuBar
        assert Neutrino.NuTauBar.value.nutype == NeutrinoType.NuBar


class TestNeutrinosList:
    def test_has_six_elements(self):
        assert len(neutrinos) == 6

    def test_contains_all_neutrino_types(self):
        assert set(neutrinos) == set(Neutrino)


