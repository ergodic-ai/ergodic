# SPDX-License-Identifier: Apache-2.0
"""Tests for the rolling-origin backtester.

The cutoff arithmetic is pinned by hand. The leakage claim is checked with a
probe model that records exactly what each fold handed it: the probe is a stub
of a *model*, so what is under test here (the harness) is never mocked. The
scoring claims run on the same seeded seasonal world the baseline tests use.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ergodic.data import Hierarchy, hts, panel, tabular, timeseries
from ergodic.forecast import (
    BacktestFold,
    BacktestReport,
    ForecastResult,
    Naive,
    SeasonalNaive,
    backtest,
)
from ergodic.forecast._backtest import _cutoffs

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

PERIOD = 12


def _seasonal(seed: int, n: int = 72) -> TimeSeriesData:
    """The world of criterion 5: y = 10 + 3 sin(2 pi t / 12) + eps(0.3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10.0 + 3.0 * np.sin(2.0 * np.pi * t / PERIOD) + rng.normal(0.0, 0.3, n)
    return timeseries({"t": t, "y": y}, time="t")


# -- criterion 8: the cutoff arithmetic ---------------------------------------


def test_expanding_cutoffs_are_evenly_spaced_and_end_at_the_last_origin() -> None:
    assert _cutoffs(60, "expanding", 12, 3, 24) == [24, 36, 48]
    assert _cutoffs(60, "single", 12, 3, 24) == [48]
    assert _cutoffs(60, "expanding", 12, 1, 24) == [48]
    # Uneven spacing still lands on the last origin.
    assert _cutoffs(61, "expanding", 12, 3, 24) == [25, 37, 49]
    # A longer history simply starts later, since min_train_size stops binding.
    assert _cutoffs(120, "expanding", 12, 3, 24) == [84, 96, 108]


def test_cutoffs_refuse_a_history_that_is_too_short() -> None:
    with pytest.raises(ValueError, match="not enough data"):
        _cutoffs(30, "single", 12, 3, 24)
    with pytest.raises(ValueError, match="not enough data for 3 expanding splits"):
        _cutoffs(30, "expanding", 12, 3, 24)
    with pytest.raises(ValueError, match="mode must be"):
        _cutoffs(60, "sliding", 12, 3, 24)
    with pytest.raises(ValueError, match="test_size must be"):
        _cutoffs(60, "expanding", 0, 3, 24)


def test_the_most_recent_origin_is_never_dropped() -> None:
    """The freshest fold is the one worth most, so the count gives way, not the origin.

    The source counted origins forwards and truncated to ``n_splits``, which
    could drop the last one. We count backwards from the last instead.
    """
    assert _cutoffs(20, "expanding", 5, 3, 10) == [11, 13, 15]  # 15 was reachable
    assert _cutoffs(50, "expanding", 12, 5, 24) == [26, 29, 32, 35, 38]


def test_cutoffs_hold_their_contract_across_the_grid() -> None:
    """Every accepted configuration ends on the last origin and respects its bounds."""
    checked = 0
    for n_rows in range(10, 130, 3):
        for test_size in range(1, 20, 2):
            for n_splits in range(1, 7):
                min_train = 2 * test_size
                try:
                    cutoffs = _cutoffs(n_rows, "expanding", test_size, n_splits, min_train)
                except ValueError:
                    continue
                checked += 1
                assert cutoffs[-1] == n_rows - test_size, (n_rows, test_size, n_splits)
                assert len(cutoffs) <= n_splits
                assert sorted(set(cutoffs)) == cutoffs
                assert min(cutoffs) >= min_train
    assert checked > 500, checked


def test_backtest_runs_the_folds_the_arithmetic_names() -> None:
    report = backtest(SeasonalNaive(PERIOD), _seasonal(0), test_size=12, n_splits=3)
    assert report.cutoffs == (36, 48, 60)
    assert [fold.index for fold in report.folds] == [0, 1, 2]
    assert report.mode == "expanding"
    assert report.model == "SeasonalNaive"
    assert report.nodes == ("y",)
    assert "mase" in report.metrics


# -- criteria 9 and 10: leakage and isolation ---------------------------------


class _Recorder:
    """A shared log the folds write to. Deep copy returns it unchanged on purpose."""

    def __init__(self) -> None:
        self.fits: list[dict[str, Any]] = []
        self.forecast_seeds: list[int] = []

    def __deepcopy__(self, memo: dict[int, Any]) -> _Recorder:
        return self


@dataclass
class _Probe:
    """A stub model that records what each fold handed it.

    It stubs a model so the harness can be probed; the harness itself is real.
    """

    recorder: _Recorder = field(default_factory=_Recorder)
    n_fits: int = 0
    node: str = ""
    last_value: float = 0.0

    def fit(self, data: TimeSeriesData, *, seed: int = 0) -> _Probe:
        self.n_fits += 1
        self.recorder.fits.append(
            {
                "seed": seed,
                "n_rows": data.n_rows,
                "max_time": float(data.times[-1]),
                "fits_at_entry": self.n_fits - 1,
                "own_fits": self.n_fits,
                "type": type(data).__name__,
            }
        )
        self.node = data.variables[0]
        self.last_value = float(data.column(self.node)[-1])
        return self

    def forecast(self, horizon: int, *, seed: int = 0) -> ForecastResult:
        self.recorder.forecast_seeds.append(seed)
        rng = np.random.default_rng(seed)
        draws: npt.NDArray[np.float64] = self.last_value + rng.normal(0.0, 1.0, (64, horizon))
        return ForecastResult(series={self.node: draws}, horizon=horizon)


def test_no_fold_ever_sees_data_at_or_past_its_cutoff() -> None:
    """Criterion 9: the probe's largest observed time is the step before the cutoff."""
    data = _seasonal(1)
    probe = _Probe()
    report = backtest(probe, data, test_size=12, n_splits=3)

    assert len(probe.recorder.fits) == 3
    for fold, record in zip(report.folds, probe.recorder.fits, strict=True):
        assert record["n_rows"] == fold.cutoff
        assert record["max_time"] == fold.cutoff - 1  # the time index is 0, 1, 2, ...
        # The first held-out timestamp is the cutoff itself, and it was not seen.
        assert record["max_time"] < fold.cutoff


def test_each_fold_gets_its_own_copy_of_the_model() -> None:
    """Criterion 10: fitted state never crosses a fold, and the caller's model stays raw."""
    probe = _Probe()
    report = backtest(probe, _seasonal(2), test_size=12, n_splits=3)

    assert [record["fits_at_entry"] for record in probe.recorder.fits] == [0, 0, 0]
    assert probe.n_fits == 0, "the model handed in must come back unfitted"
    assert probe.last_value == 0.0
    # Each fold's copy fitted exactly once. A shared object would count up.
    assert [record["own_fits"] for record in probe.recorder.fits] == [1, 1, 1]
    # And the fold records agree with what the probe was actually handed.
    assert [record["seed"] for record in probe.recorder.fits] == [0, 1, 2]
    assert probe.recorder.forecast_seeds == [1000, 1001, 1002]
    assert [fold.fit_seed for fold in report.folds] == [0, 1, 2]
    assert [fold.forecast_seed for fold in report.folds] == [1000, 1001, 1002]
    assert len({fold.fit_seed for fold in report.folds}) == 3


def test_the_base_seed_shifts_every_fold() -> None:
    probe = _Probe()
    report = backtest(probe, _seasonal(2), test_size=12, n_splits=2, seed=7)
    assert [fold.fit_seed for fold in report.folds] == [7, 8]
    assert [fold.forecast_seed for fold in report.folds] == [1007, 1008]
    assert probe.recorder.forecast_seeds == [1007, 1008]


def test_folds_carry_their_forecasts_and_actuals() -> None:
    data = _seasonal(3)
    report = backtest(Naive(), data, test_size=12, n_splits=2)
    for fold in report.folds:
        assert fold.forecasts["y"].shape == (500, 12)
        expected = data.column("y")[fold.cutoff : fold.cutoff + 12]
        assert np.allclose(fold.actuals["y"], expected)
        assert fold.horizon == 12


# -- criterion 11: comparing two models ---------------------------------------


def test_compare_marks_the_better_model_improved() -> None:
    """Criterion 11: on the seasonal world, SeasonalNaive beats Naive where it should."""
    data = _seasonal(4)
    seasonal = backtest(SeasonalNaive(PERIOD), data, test_size=12, n_splits=3)
    naive = backtest(Naive(), data, test_size=12, n_splits=3)
    frame = seasonal.compare(naive)

    for metric in ("mae", "rmse", "crps"):
        row = frame[(frame["node"] == "y") & (frame["metric"] == metric)].iloc[0]
        assert row["value"] < row["other"], metric
        assert row["delta"] < 0
        assert bool(row["improved"]) is True, metric
    # The direction flips for the accuracy family: bigger is better there.
    accuracy_row = frame[(frame["node"] == "y") & (frame["metric"] == "accuracy")].iloc[0]
    assert bool(accuracy_row["improved"]) is bool(accuracy_row["delta"] > 0)


def test_compare_survives_a_metric_only_one_side_carries() -> None:
    """A one-sided metric is reported as NaN rather than blowing the comparison up."""

    def _report(metrics: dict[str, float]) -> BacktestReport:
        fold = BacktestFold(
            index=0,
            cutoff=10,
            horizon=2,
            metrics={"y": metrics},
            forecasts={"y": np.zeros((4, 2))},
            actuals={"y": np.zeros(2)},
        )
        return BacktestReport(
            model="stub",
            mode="single",
            test_size=2,
            level=0.9,
            folds=(fold,),
            nodes=("y",),
            metrics=tuple(metrics),
        )

    rich = _report({"mae": 1.0, "mase": 0.5})
    plain = _report({"mae": 2.0})

    frame = rich.compare(plain)
    assert set(frame["metric"]) == {"mae", "mase"}
    mae_row = frame[frame["metric"] == "mae"].iloc[0]
    assert mae_row["delta"] == pytest.approx(-1.0)
    assert bool(mae_row["improved"]) is True
    mase_row = frame[frame["metric"] == "mase"].iloc[0]
    assert mase_row["value"] == pytest.approx(0.5)
    assert np.isnan(mase_row["other"])
    assert np.isnan(mase_row["delta"])
    assert bool(mase_row["improved"]) is False
    # And the other way round, where this report is the one missing the metric.
    flipped = plain.compare(rich)
    flipped_row = flipped[flipped["metric"] == "mase"].iloc[0]
    assert np.isnan(flipped_row["value"])
    assert flipped_row["other"] == pytest.approx(0.5)


def test_compare_judges_coverage_by_closeness_to_the_level() -> None:
    """Over-covering is not an improvement; landing nearer the nominal level is."""
    data = _seasonal(5)
    seasonal = backtest(SeasonalNaive(PERIOD), data, test_size=12, n_splits=3)
    naive = backtest(Naive(), data, test_size=12, n_splits=3)
    frame = seasonal.compare(naive)
    row = frame[(frame["node"] == "y") & (frame["metric"] == "coverage")].iloc[0]
    nearer = abs(row["value"] - 0.9) < abs(row["other"] - 0.9)
    assert bool(row["improved"]) is bool(nearer)


# -- the report tables --------------------------------------------------------


def test_summary_averages_over_folds_and_to_frame_is_tidy() -> None:
    data = _seasonal(6)
    report = backtest(SeasonalNaive(PERIOD), data, test_size=12, n_splits=3)

    summary = report.summary()
    assert list(summary.index) == ["y"]
    assert "crps" in summary.columns
    assert summary.loc["y", "mae"] == pytest.approx(
        float(np.mean([fold.metrics["y"]["mae"] for fold in report.folds]))
    )

    tidy = report.to_frame()
    assert list(tidy.columns) == ["fold", "cutoff", "node", "metric", "value"]
    assert len(tidy) == 3 * len(report.metrics)
    assert set(tidy["cutoff"]) == set(report.cutoffs)


def test_single_mode_runs_one_fold_at_the_end() -> None:
    data = _seasonal(7)
    report = backtest(Naive(), data, mode="single", test_size=12)
    assert report.cutoffs == (60,)
    assert len(report.folds) == 1
    assert report.mode == "single"


def test_backtest_validates_its_arguments() -> None:
    data = _seasonal(8)
    with pytest.raises(ValueError, match="level must be"):
        backtest(Naive(), data, test_size=12, level=1.5)
    with pytest.raises(ValueError, match="not enough data"):
        backtest(Naive(), data, test_size=40, n_splits=3)
    with pytest.raises(ValueError, match="not enough data"):
        backtest(Naive(), data, test_size=12, min_train_size=100)


def test_min_train_size_defaults_to_twice_the_test_window() -> None:
    report = backtest(Naive(), _seasonal(9, n=40), test_size=12, n_splits=5)
    assert min(report.cutoffs) >= 24


# -- the data the folds are cut from ------------------------------------------


def test_take_gives_the_backtester_a_real_prefix() -> None:
    """The backtester slices with Dataset.take, which has to keep the shape intact."""
    data = _seasonal(10)
    prefix = data.take(np.arange(30))
    assert type(prefix) is type(data)
    assert prefix.time == data.time
    assert prefix.schema == data.schema
    assert prefix.n_rows == 30
    assert np.array_equal(prefix.column("y"), data.column("y")[:30])
    assert np.array_equal(prefix.times, data.times[:30])


def test_backtest_reads_a_hierarchical_dataset_without_losing_the_hierarchy() -> None:
    """A HierarchicalTSData prefix is still hierarchical when the model sees it."""
    rng = np.random.default_rng(11)
    n = 48
    north = 100.0 + np.arange(n) * 0.5 + rng.normal(0.0, 2.0, n)
    south = 80.0 + np.arange(n) * 0.2 + rng.normal(0.0, 2.0, n)
    data = hts(
        {"t": np.arange(n), "north": north, "south": south, "total": north + south},
        time="t",
        hierarchy=Hierarchy.tree({"total": ["north", "south"]}),
    )
    probe = _Probe()

    class _HierarchyProbe(_Probe):
        def fit(self, data: TimeSeriesData, *, seed: int = 0) -> _Probe:
            assert getattr(data, "hierarchy", None) is not None
            assert data.hierarchy.children_of("total") == ("north", "south")  # type: ignore[attr-defined]
            assert data.is_coherent()  # type: ignore[attr-defined]
            return super().fit(data, seed=seed)

    hierarchical = _HierarchyProbe(recorder=probe.recorder)
    report = backtest(hierarchical, data, test_size=8, n_splits=2)
    assert len(report.folds) == 2
    assert [record["type"] for record in probe.recorder.fits] == [
        "HierarchicalTSData",
        "HierarchicalTSData",
    ]


def test_a_model_that_forecasts_nothing_scorable_raises() -> None:
    class _Elsewhere(_Probe):
        def forecast(self, horizon: int, *, seed: int = 0) -> ForecastResult:
            return ForecastResult(series={"other": np.zeros((4, horizon))}, horizon=horizon)

    with pytest.raises(ValueError, match="nothing to score"):
        backtest(_Elsewhere(), _seasonal(12), test_size=12, n_splits=2)


def test_a_forecast_node_with_no_column_is_recorded_not_dropped() -> None:
    """An aggregate the data does not carry cannot be scored, but it is not invisible."""

    class _WithAnExtra(_Probe):
        def forecast(self, horizon: int, *, seed: int = 0) -> ForecastResult:
            scored = super().forecast(horizon, seed=seed)
            return ForecastResult(
                series={**scored.series, "total": np.zeros((64, horizon))}, horizon=horizon
            )

    report = backtest(_WithAnExtra(), _seasonal(13), test_size=12, n_splits=2)
    assert report.nodes == ("y",)
    assert report.nodes_unscored == ("total",)
    assert all(fold.nodes_unscored == ("total",) for fold in report.folds)


def test_backtest_refuses_a_dataset_that_is_not_one_series() -> None:
    """Row-slicing a panel would mix entities, so it is refused rather than scored."""
    rng = np.random.default_rng(14)
    rows = 40
    entities = np.repeat(["north", "south"], rows // 2)
    frame = {
        "e": entities,
        "t": np.tile(np.arange(rows // 2), 2),
        "y": rng.normal(size=rows),
    }
    with pytest.raises(TypeError, match="mix entities"):
        backtest(Naive(), panel(frame, entity="e", time="t"), test_size=4, n_splits=2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TimeSeriesData"):
        backtest(Naive(), tabular({"y": rng.normal(size=20)}), test_size=4)  # type: ignore[arg-type]


def test_deepcopy_of_the_recorder_is_shared_on_purpose() -> None:
    """The probe's log survives the harness's deep copy, which is what makes it a probe."""
    recorder = _Recorder()
    assert copy.deepcopy(recorder) is recorder
