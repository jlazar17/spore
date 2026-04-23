import numpy as np
import pytest

from spore.event_sampling.good_run_list import GoodRunList
from spore.conventions import ureg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_uptime_csv(path, runs):
    """Write an IceCube-style two-column uptime CSV."""
    np.savetxt(path, runs, header="MJD_start[days]  MJD_stop[days]", comments="# ")


# ---------------------------------------------------------------------------
# Construction from a numpy array
# ---------------------------------------------------------------------------

class TestGoodRunListInit:
    def test_valid_array_accepted(self):
        runs = np.array([[1.0, 2.0], [3.0, 4.0]])
        grl = GoodRunList(runs)
        assert grl.runs.shape == (2, 2)

    def test_single_run(self):
        runs = np.array([[0.0, 1.0]])
        grl = GoodRunList(runs)
        assert len(grl.runs) == 1

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            GoodRunList(np.array([1.0, 2.0, 3.0]))

    def test_wrong_columns_raises(self):
        with pytest.raises(ValueError):
            GoodRunList(np.ones((3, 3)))

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            GoodRunList(np.array([[5.0, 3.0]]))  # stop < start

    def test_zero_duration_run_accepted(self):
        # A zero-duration run is degenerate but not invalid.
        grl = GoodRunList(np.array([[1.0, 1.0], [2.0, 3.0]]))
        assert grl.runs.shape == (2, 2)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestGoodRunListProperties:
    @pytest.fixture
    def grl(self):
        return GoodRunList(np.array([
            [100.0, 102.0],   # 2 days
            [105.0, 108.0],   # 3 days
            [110.0, 111.0],   # 1 day
        ]))

    def test_total_livetime_value(self, grl):
        assert grl.total_livetime.magnitude == pytest.approx(6.0)

    def test_total_livetime_units(self, grl):
        assert str(grl.total_livetime.units) == "day"

    def test_t_start_is_minimum(self, grl):
        assert grl.t_start == pytest.approx(100.0)

    def test_runs_returns_array(self, grl):
        assert isinstance(grl.runs, np.ndarray)
        assert grl.runs.shape == (3, 2)


# ---------------------------------------------------------------------------
# Construction from paths
# ---------------------------------------------------------------------------

class TestGoodRunListFromPath:
    @pytest.fixture
    def uptime_dir(self, tmp_path):
        """Directory with two IceCube-style uptime CSV files."""
        _write_uptime_csv(
            tmp_path / "IC86_I_exp.csv",
            np.array([[55694.0, 55695.0], [55696.0, 55697.0]]),
        )
        _write_uptime_csv(
            tmp_path / "IC86_II_exp.csv",
            np.array([[55698.0, 55699.5]]),
        )
        return tmp_path

    def test_single_file(self, tmp_path):
        path = tmp_path / "IC86_I_exp.csv"
        _write_uptime_csv(path, np.array([[55694.0, 55695.0], [55696.0, 55697.0]]))
        grl = GoodRunList.from_path(str(path))
        assert grl.runs.shape == (2, 2)

    def test_single_file_pathlib(self, tmp_path):
        path = tmp_path / "IC86_I_exp.csv"
        _write_uptime_csv(path, np.array([[55694.0, 55695.0]]))
        grl = GoodRunList.from_path(path)
        assert grl.runs.shape == (1, 2)

    def test_directory_loads_all_csv_files(self, uptime_dir):
        grl = GoodRunList.from_path(uptime_dir)
        assert grl.runs.shape == (3, 2)

    def test_directory_total_livetime(self, uptime_dir):
        grl = GoodRunList.from_path(uptime_dir)
        assert grl.total_livetime.magnitude == pytest.approx(3.5)

    def test_list_of_files(self, tmp_path):
        f1 = tmp_path / "IC86_I_exp.csv"
        f2 = tmp_path / "IC86_II_exp.csv"
        _write_uptime_csv(f1, np.array([[55694.0, 55695.0]]))
        _write_uptime_csv(f2, np.array([[55696.0, 55698.0]]))
        grl = GoodRunList.from_path([str(f1), str(f2)])
        assert grl.runs.shape == (2, 2)
        assert grl.total_livetime.magnitude == pytest.approx(3.0)

    def test_list_mixing_file_and_directory(self, tmp_path):
        subdir = tmp_path / "season2"
        subdir.mkdir()
        f1 = tmp_path / "IC86_I_exp.csv"
        f2 = subdir / "IC86_II_exp.csv"
        _write_uptime_csv(f1, np.array([[55694.0, 55695.0]]))
        _write_uptime_csv(f2, np.array([[55696.0, 55697.0], [55698.0, 55699.0]]))
        grl = GoodRunList.from_path([str(f1), str(subdir)])
        assert grl.runs.shape == (3, 2)

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            GoodRunList.from_path("/nonexistent/path/to/uptime.csv")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No \\*_exp\\*\\.csv"):
            GoodRunList.from_path(tmp_path)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            GoodRunList.from_path(42)


# ---------------------------------------------------------------------------
# sample_times
# ---------------------------------------------------------------------------

class TestGoodRunListSampleTimes:
    @pytest.fixture
    def grl(self):
        return GoodRunList(np.array([
            [0.0, 1.0],
            [5.0, 6.0],
        ]))

    def test_returns_correct_length(self, grl):
        rng = np.random.default_rng(0)
        times = grl.sample_times(100, rng)
        assert len(times) == 100

    def test_n_zero_returns_empty(self, grl):
        rng = np.random.default_rng(0)
        times = grl.sample_times(0, rng)
        assert len(times) == 0

    def test_all_times_within_good_runs(self, grl):
        rng = np.random.default_rng(0)
        times = grl.sample_times(1000, rng)
        for t in times:
            in_run = any(r[0] <= t <= r[1] for r in grl.runs)
            assert in_run, f"time {t} falls outside all good runs"

    def test_no_times_in_gaps(self):
        # Two runs with a clear gap in [2, 4].
        grl = GoodRunList(np.array([[0.0, 2.0], [4.0, 6.0]]))
        rng = np.random.default_rng(42)
        times = grl.sample_times(2000, rng)
        assert not np.any((times > 2.0) & (times < 4.0))

    def test_sampling_weighted_by_duration(self):
        # Run 0 is 9 days, run 1 is 1 day.  Run 0 should get ~90% of samples.
        grl = GoodRunList(np.array([[0.0, 9.0], [10.0, 11.0]]))
        rng = np.random.default_rng(7)
        times = grl.sample_times(5000, rng)
        fraction_in_run0 = np.mean(times < 9.0)
        assert fraction_in_run0 == pytest.approx(0.9, abs=0.03)
