# SPDX-License-Identifier: Apache-2.0
"""The backtester driving the real forecaster, on S2's recovery world.

S1 built the harness against a protocol and scored baselines with it. This is
the other half: the Bayesian forecaster goes through the same harness, in the
same table as a naive baseline and as a version of itself that has been told the
wrong thing, and the winner is the one that should win. A backtest is where a
misspecification shows up as a number instead of as an opinion.

Every fold refits from scratch, so the cost is folds times nodes NUTS runs and
the budget here is deliberately small. Skipped where pymc is missing.
"""

from __future__ import annotations

import importlib.util
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from ergodic import backtest, forecast, timeseries
from ergodic.forecast import (
    BacktestReport,
    Forecaster,
    SeasonalNaive,
    StructuralMechanism,
    forecast_model,
    infer_period,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

    Array = npt.NDArray[np.float64]

requires_pymc = pytest.mark.skipif(
    importlib.util.find_spec("pymc") is None, reason="pymc is not installed"
)

WORLD_SEED = 20260805
TOTAL = 96
TEST_SIZE = 6
N_SPLITS = 2
DRAWS = 300
WARMUP = 300
SEED = 4


def world_a(seed: int = WORLD_SEED, n: int = TOTAL) -> Array:
    """S2's World A: a damped seasonal trend near 200 with 2 percent noise.

    The same parameters the mechanism sprint recovered from (phi 0.85, slope
    0.6, sigma_obs 4, a seasonal factor between 0.85 and 1.15), so what is being
    scored here is a model that is known to fit this world.
    """
    rng = np.random.default_rng(seed)
    factors = 1.0 + 0.15 * np.sin(2.0 * np.pi * np.arange(12) / 12.0)
    level, slope = 200.0, 0.6
    levels = np.empty(n, dtype=np.float64)
    for t in range(n):
        levels[t] = level
        level = level + slope + 0.4 * rng.normal()
        slope = 0.85 * slope + 0.15 * rng.normal()
    mu = levels * factors[np.arange(n) % 12]
    return np.asarray(mu + rng.normal(0.0, 4.0, n), dtype=np.float64)


@pytest.fixture(scope="module")
def sales() -> TimeSeriesData:
    frame = pd.DataFrame(
        {"t": pd.date_range("2018-01-31", periods=TOTAL, freq="ME"), "sales": world_a()}
    )
    return timeseries(frame, time="t", freq="ME")


def backtest_or_skip(*args: Any, **kwargs: Any) -> BacktestReport:
    """Backtest, or skip when this sandbox cannot compile a model."""
    try:
        return backtest(*args, **kwargs)
    except Exception as exc:  # pytensor needs a C compiler and Python headers
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise


def run(model: Any, data: TimeSeriesData) -> BacktestReport:
    return backtest_or_skip(
        model,
        data,
        test_size=TEST_SIZE,
        n_splits=N_SPLITS,
        seed=SEED,
        m=infer_period(data) or 1,
    )


@pytest.fixture(scope="module")
def reports(sales: TimeSeriesData) -> dict[str, BacktestReport]:
    """One table, three models: the truth, the same model told there is no season, and naive."""
    started = time.time()
    out = {
        "true spec": run(
            forecast_model(
                name="true spec",
                mechanisms={"sales": StructuralMechanism(period=12)},
                draws=DRAWS,
                warmup=WARMUP,
            ),
            sales,
        ),
        "no season": run(
            forecast_model(
                name="no season",
                mechanisms={"sales": StructuralMechanism(seasonality=None)},
                draws=DRAWS,
                warmup=WARMUP,
            ),
            sales,
        ),
        "seasonal naive": run(SeasonalNaive(period=12), sales),
    }
    out["elapsed"] = time.time() - started  # type: ignore[assignment]
    return out


def comparison(reports: dict[str, BacktestReport]) -> pd.DataFrame:
    """The three models side by side, mean over folds, one row each."""
    rows = {
        name: report.summary().loc["sales"]
        for name, report in reports.items()
        if isinstance(report, BacktestReport)
    }
    return pd.DataFrame(rows).T[["crps", "mae", "mase", "coverage"]]


# -- criterion 6: the true spec wins the table --------------------------------


@requires_pymc
def test_the_true_spec_beats_the_wrong_one_and_the_naive_baseline(
    reports: dict[str, BacktestReport],
) -> None:
    table = comparison(reports)
    assert list(table.index) == ["true spec", "no season", "seasonal naive"]
    assert table.loc["true spec", "crps"] < table.loc["no season", "crps"]
    assert table.loc["true spec", "crps"] < table.loc["seasonal naive", "crps"]
    # MASE is scaled by the in-sample seasonal naive error, so under one means
    # the forecast is worth more than repeating last year's month.
    assert table.loc["true spec", "mase"] < 1.0


@requires_pymc
def test_the_comparison_reads_the_same_way_pairwise(reports: dict[str, BacktestReport]) -> None:
    pairwise = reports["true spec"].compare(reports["seasonal naive"])
    crps = pairwise[pairwise["metric"] == "crps"].iloc[0]
    assert bool(crps["improved"])
    assert crps["value"] < crps["other"]


# -- criterion 7: seeds, diagnostics, and the cost ----------------------------


@requires_pymc
def test_every_fold_has_its_own_seeds(reports: dict[str, BacktestReport]) -> None:
    report = reports["true spec"]
    assert [fold.fit_seed for fold in report.folds] == [SEED, SEED + 1]
    assert [fold.forecast_seed for fold in report.folds] == [SEED + 1000, SEED + 1001]
    assert len({fold.fit_seed for fold in report.folds}) == N_SPLITS


@requires_pymc
def test_the_folds_carry_the_samplers_own_report(reports: dict[str, BacktestReport]) -> None:
    report = reports["true spec"]
    frame = report.diagnostics()
    assert len(frame) == N_SPLITS
    assert list(frame["cutoff"]) == list(report.cutoffs)
    assert set(frame["worst_node"]) == {"sales"}
    for fold in report.folds:
        assert fold.diagnostics is not None
        assert fold.diagnostics.n_nodes == 1
        assert np.isfinite(fold.diagnostics.min_ess)


@requires_pymc
def test_a_fold_is_exactly_what_the_facade_would_have_produced(
    sales: TimeSeriesData, reports: dict[str, BacktestReport]
) -> None:
    """The harness's seeds are the facade's seeds, checked by reproducing a fold.

    This is the whole seed contract in one assertion: refitting the fold's own
    training prefix at the fold's own fit seed gives back the draws the fold
    scored, bit for bit, and the diagnostics it recorded.
    """
    fold = reports["true spec"].folds[-1]
    train = sales.take(np.arange(fold.cutoff, dtype=np.int64))
    again = forecast(
        train,
        TEST_SIZE,
        mechanisms={"sales": StructuralMechanism(period=12)},
        draws=DRAWS,
        warmup=WARMUP,
        seed=fold.fit_seed,
    )
    assert np.array_equal(again.draws("sales"), fold.forecasts["sales"])
    assert fold.diagnostics is not None
    assert fold.diagnostics.min_ess == pytest.approx(again.diagnostics["sales"].min_ess)


@requires_pymc
def test_the_whole_table_fits_the_budget(reports: dict[str, BacktestReport]) -> None:
    # Two Bayesian models over two folds each: four NUTS runs plus their
    # predictive passes. The docs quote the cost model as folds times MCMC
    # nodes, and this is that arithmetic held to a number.
    assert float(reports["elapsed"]) < 180.0  # type: ignore[arg-type]


@requires_pymc
def test_a_baseline_reports_no_sampler_diagnostics(reports: dict[str, BacktestReport]) -> None:
    report = reports["seasonal naive"]
    assert all(fold.diagnostics is None for fold in report.folds)
    assert report.diagnostics().empty
    assert list(report.diagnostics().columns)[:2] == ["fold", "cutoff"]


# -- the wrapper itself -------------------------------------------------------


def test_the_model_wrapper_holds_configuration_and_data_apart() -> None:
    model = forecast_model(draws=17, name="tiny")
    assert isinstance(model, Forecaster)
    assert model.options == {"draws": 17}
    assert repr(model) == "tiny(draws=17)"
    with pytest.raises(ValueError, match="not fitted"):
        model.forecast(3)


def test_the_model_wrapper_refuses_an_option_the_facade_lacks() -> None:
    with pytest.raises(TypeError, match="do_calculus"):
        forecast_model(do_calculus=True)


def test_fitting_a_wrapper_returns_a_new_one(sales: TimeSeriesData) -> None:
    model = forecast_model(draws=17)
    fitted = model.fit(sales, seed=3)
    assert fitted is not model
    assert model.data is None
    assert fitted.seed == 3
    with pytest.raises(TypeError, match="TimeSeriesData"):
        model.fit(pd.DataFrame({"a": [1.0]}))  # type: ignore[arg-type]


@requires_pymc
def test_the_report_carries_the_name_the_model_gave_itself(
    reports: dict[str, BacktestReport],
) -> None:
    assert reports["true spec"].model == "true spec"
    assert reports["seasonal naive"].model == "SeasonalNaive"
