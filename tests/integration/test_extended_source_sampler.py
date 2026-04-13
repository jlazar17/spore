import pytest

from spore.conventions import ureg
from spore.physics import neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.source.extended_source import ExtendedSource
from spore.event_sampling.extended_source_event_sampler import ExtendedSourceEventSampler


@pytest.fixture
def extended_source(powerlaw_dist):
    """Minimal ExtendedSource with a power-law flux independent of declination."""
    norm = 1e-18  # GeV⁻¹ cm⁻² s⁻¹
    flux = Flux(
        {nu: norm for nu in neutrinos},
        {nu: powerlaw_dist for nu in neutrinos},
    )
    return ExtendedSource(flux)


@pytest.fixture
def extended_source_sampler(detector, extended_source):
    return ExtendedSourceEventSampler(detector, extended_source, n_dec=5, n_ra=5, n_e=5)


@pytest.mark.slow
class TestExtendedSourceEventSampler:
    def test_nevent_returns_list(self, extended_source_sampler):
        events = extended_source_sampler.sample_events("track", nevent=3)
        assert isinstance(events, list)

    def test_nevent_zero_returns_empty_list_not_ndarray(self, extended_source_sampler):
        # Fix 2: the old code returned np.array([]) instead of [].
        events = extended_source_sampler.sample_events("track", nevent=0)
        assert isinstance(events, list)
        assert events == []

    def test_nevent_returns_correct_count(self, extended_source_sampler):
        events = extended_source_sampler.sample_events("track", nevent=5)
        assert len(events) == 5

    def test_invalid_morphology_raises(self, extended_source_sampler):
        with pytest.raises(ValueError):
            extended_source_sampler.sample_events("muon", nevent=3)

    def test_both_nevent_and_deltat_raises(self, extended_source_sampler):
        with pytest.raises(ValueError):
            extended_source_sampler.sample_events("track", nevent=3, deltat=1e20)

    def test_neither_nevent_nor_deltat_raises(self, extended_source_sampler):
        with pytest.raises(ValueError):
            extended_source_sampler.sample_events("track")
