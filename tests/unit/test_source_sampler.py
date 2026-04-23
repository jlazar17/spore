import numpy as np
import pytest

from spore.conventions import SkyCoordinate
from spore.physics import neutrinos
from spore.source.flux.distributions.power_laws import PowerLaw
from spore.source.flux.flux import Flux
from spore.source.point_source import PointSource
from spore.source.extended_source import ExtendedSource
from spore.event_sampling import SourceSampler
from spore.event_sampling.point_source_event_sampler import PointSourceEventSampler
from spore.event_sampling.extended_source_event_sampler import ExtendedSourceEventSampler


@pytest.fixture
def extended_source(powerlaw_dist):
    norm = 1e-18
    flux = Flux(
        {nu: norm for nu in neutrinos},
        {nu: powerlaw_dist for nu in neutrinos},
    )
    return ExtendedSource(flux)


class TestSourceSamplerDispatch:
    def test_point_source_returns_point_source_sampler(self, detector, point_source):
        sampler = SourceSampler(detector, point_source)
        assert isinstance(sampler, PointSourceEventSampler)

    def test_extended_source_returns_extended_source_sampler(self, detector, extended_source):
        sampler = SourceSampler(detector, extended_source, n_dec=3, n_ra=3, n_e=3)
        assert isinstance(sampler, ExtendedSourceEventSampler)

    def test_unknown_source_type_raises(self, detector):
        with pytest.raises(TypeError, match="PointSource or ExtendedSource"):
            SourceSampler(detector, object())

    def test_error_message_includes_type_name(self, detector):
        class WeirdSource:
            pass
        with pytest.raises(TypeError, match="WeirdSource"):
            SourceSampler(detector, WeirdSource())


class TestSourceSamplerKwargs:
    def test_n_time_samples_accepted_for_point_source(self, detector, point_source):
        sampler = SourceSampler(detector, point_source, n_time_samples=10)
        assert sampler._steady_state is True

    def test_n_time_samples_accepted_for_extended_source(self, detector, extended_source):
        sampler = SourceSampler(detector, extended_source, n_time_samples=10,
                                n_dec=3, n_ra=3, n_e=3)
        assert sampler._steady_state is True

    def test_extended_only_kwargs_ignored_for_point_source(self, detector, point_source):
        # n_dec, n_ra, and adaptive_energy_grid do not apply to point sources;
        # passing them must not raise.
        sampler = SourceSampler(detector, point_source, n_dec=5, n_ra=5,
                                adaptive_energy_grid=False)
        assert isinstance(sampler, PointSourceEventSampler)

    def test_e_min_e_max_forwarded_for_point_source(self, detector, point_source):
        # Passing explicit energy bounds must not raise.
        sampler = SourceSampler(detector, point_source, e_min=1e3, e_max=1e6)
        assert isinstance(sampler, PointSourceEventSampler)

    def test_e_min_e_max_forwarded_for_extended_source(self, detector, extended_source):
        sampler = SourceSampler(detector, extended_source,
                                n_dec=3, n_ra=3, n_e=3, e_min=1e3, e_max=1e6)
        assert isinstance(sampler, ExtendedSourceEventSampler)


class TestSourceSamplerMultiDetector:
    def test_list_of_detectors_accepted_for_point_source(self, detector, point_source):
        sampler = SourceSampler([detector, detector], point_source)
        assert isinstance(sampler, PointSourceEventSampler)
        assert sampler._multi is True

    def test_list_of_detectors_accepted_for_extended_source(self, detector, extended_source):
        sampler = SourceSampler([detector, detector], extended_source,
                                n_dec=3, n_ra=3, n_e=3)
        assert isinstance(sampler, ExtendedSourceEventSampler)
        assert sampler._multi is True
