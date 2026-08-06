# SPDX-License-Identifier: Apache-2.0
"""The forecast() facade: schema dispatch, provenance, determinism, and refusals.

The refusals run everywhere, including on an install with no sampler, because
refusing well is most of what an unfinished keyword is for. The two tests that
actually sample are marked and skip when pymc is missing.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ergodic import forecast, hts, panel, tabular, timeseries
from ergodic.forecast import (
    ScenarioMechanism,
    StructuralMechanism,
    assign_mechanisms,
    infer_period,
)
from ergodic.forecast._mechanism import require_pymc

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

    Array = npt.NDArray[np.float64]

requires_pymc = pytest.mark.skipif(
    importlib.util.find_spec("pymc") is None, reason="pymc is not installed"
)

WORLD_SEED = 20260805


# -- little worlds ------------------------------------------------------------


def series(n: int = 48, seed: int = WORLD_SEED, level: float = 200.0) -> Array:
    """A short seasonal series in original units."""
    rng = np.random.default_rng(seed)
    factors = 1.0 + 0.15 * np.sin(2.0 * np.pi * np.arange(12) / 12.0)
    levels = level + 0.6 * np.arange(n) + np.cumsum(rng.normal(0.0, 0.4, n))
    return np.asarray(levels * factors[np.arange(n) % 12] + rng.normal(0.0, 4.0, n))


def monthly(n: int = 48, **kwargs: Any) -> TimeSeriesData:
    stamps = np.array([np.datetime64("2019-01-01") + np.timedelta64(31 * i, "D") for i in range(n)])
    return timeseries({"month": stamps, "sales": series(n, **kwargs)}, time="month")


def quarterly(n: int = 24) -> TimeSeriesData:
    stamps = np.array([np.datetime64("2019-01-01") + np.timedelta64(91 * i, "D") for i in range(n)])
    return timeseries({"quarter": stamps, "sales": series(n)}, time="quarter")


def weekly(n: int = 60) -> TimeSeriesData:
    stamps = np.array([np.datetime64("2019-01-07") + np.timedelta64(7 * i, "D") for i in range(n)])
    return timeseries({"week": stamps, "sales": series(n)}, time="week")


def stepped(n: int = 48) -> TimeSeriesData:
    return timeseries({"t": np.arange(n, dtype=float), "sales": series(n)}, time="t")


# -- criterion 9: schema dispatch and period inference ------------------------


def test_the_period_comes_from_the_index() -> None:
    """Monthly spacing gives 12, quarterly 4, weekly 52, a step index nothing."""
    assert infer_period(monthly()) == 12
    assert infer_period(quarterly()) == 4
    assert infer_period(weekly()) == 52
    assert infer_period(stepped()) is None


def test_auto_assignment_carries_the_period() -> None:
    """A continuous column gets the structural mechanism with the period the index implied."""
    assigned = assign_mechanisms(monthly())
    assert set(assigned) == {"sales"}
    mechanism = assigned["sales"]
    assert isinstance(mechanism, StructuralMechanism)
    assert mechanism.period == 12
    assert mechanism.seasonal_form() == ("monthly", 12)

    quarter = assign_mechanisms(quarterly())["sales"]
    assert isinstance(quarter, StructuralMechanism)
    assert quarter.period == 4
    assert quarter.seasonal_form() == ("fourier", 4)

    stepless = assign_mechanisms(stepped())["sales"]
    assert isinstance(stepless, StructuralMechanism)
    assert stepless.period is None
    assert stepless.seasonal_form() == (None, None)


def test_a_categorical_column_names_the_seam() -> None:
    """Criterion 9: a non-continuous column raises and points at the reserved seam."""
    stamps = np.array(
        [np.datetime64("2019-01-01") + np.timedelta64(31 * i, "D") for i in range(24)]
    )
    data = timeseries(
        {"month": stamps, "sales": series(24), "region": np.array(["north", "south"] * 12)},
        time="month",
    )
    with pytest.raises(NotImplementedError, match="reserved seam"):
        assign_mechanisms(data)


def test_an_override_outranks_the_schema() -> None:
    """A named mechanism is kept as it is, and robust= does not reach it."""
    given = StructuralMechanism(period=4, seasonality="fourier")
    kept = assign_mechanisms(monthly(), {"sales": given}, robust=True)["sales"]
    assert kept is given
    assert isinstance(kept, StructuralMechanism)
    assert kept.robust is False
    auto = assign_mechanisms(monthly(), robust=True)["sales"]
    assert isinstance(auto, StructuralMechanism)
    assert auto.robust is True


def test_overrides_are_checked() -> None:
    """A column that is not there, and an object that is not a mechanism, both raise."""
    with pytest.raises(KeyError, match="does not model"):
        assign_mechanisms(monthly(), {"revenue": StructuralMechanism()})
    with pytest.raises(TypeError, match="TemporalMechanism"):
        assign_mechanisms(monthly(), {"sales": "structural"})  # type: ignore[dict-item]


# -- the mechanism contract objects ------------------------------------------


def test_scenario_mechanism_is_resolved_not_sampled() -> None:
    """It has an identity scaler, a pinned path when given one, and no model to build."""
    mechanism = ScenarioMechanism(values=(1.0, 2.0, 3.0))
    assert np.array_equal(mechanism.path(), np.array([1.0, 2.0, 3.0]))
    identity = mechanism.scaler_for(np.array([100.0, 200.0]))
    assert float(identity.transform(np.array([5.0]))[0]) == 5.0
    with pytest.raises(NotImplementedError, match="resolved rather than"):
        mechanism.build("pmi", obs=None, parents={}, T=4, horizon=2, mask=np.ones(4, bool))
    with pytest.raises(ValueError, match="no pinned path"):
        ScenarioMechanism().path()


def test_the_naming_contract_is_one_function() -> None:
    """Every site name goes through site(), so there is one place it can be changed."""
    mechanism = StructuralMechanism()
    assert mechanism.site("sales", "phi") == "sales_phi"
    assert mechanism.site("sales", "beta_pmi") == "sales_beta_pmi"
    assert mechanism.predictive_site("sales") == "sales_y_fc"


def test_the_mechanism_validates_its_options() -> None:
    """The options that cannot work together say so at construction."""
    with pytest.raises(ValueError, match="trend must be"):
        StructuralMechanism(trend="linear")
    with pytest.raises(ValueError, match="seasonality must be"):
        StructuralMechanism(seasonality="weekly")
    with pytest.raises(ValueError, match="period must be at least"):
        StructuralMechanism(period=1)
    with pytest.raises(ValueError, match="twelve factors by definition"):
        StructuralMechanism(seasonality="monthly", period=4)
    with pytest.raises(ValueError, match="needs a period"):
        StructuralMechanism(seasonality="fourier").seasonal_form()
    with pytest.raises(TypeError, match="must be a Scaler"):
        StructuralMechanism(scaler="ratio")  # type: ignore[arg-type]


def test_the_base_contract_refuses_to_be_used_directly() -> None:
    """TemporalMechanism is a contract, not a model."""
    from ergodic.forecast import TemporalMechanism

    with pytest.raises(NotImplementedError, match="does not implement build"):
        TemporalMechanism().build(
            "sales", obs=None, parents={}, T=4, horizon=0, mask=np.ones(4, bool)
        )


# -- criterion 13: the facade's refusals --------------------------------------


def test_a_hierarchy_points_at_reconciliation() -> None:
    """A HierarchicalTSData needs the level machinery, and says so."""
    stamps = np.array(
        [np.datetime64("2019-01-01") + np.timedelta64(31 * i, "D") for i in range(24)]
    )
    north, south = series(24, seed=1), series(24, seed=2)
    data = hts(
        {"month": stamps, "north": north, "south": south, "total": north + south},
        time="month",
        hierarchy={"total": ["north", "south"]},
    )
    with pytest.raises(NotImplementedError, match="S3"):
        forecast(data, 3)


def test_a_graph_points_at_the_driver_sprint() -> None:
    from ergodic import dag

    with pytest.raises(NotImplementedError, match="S4"):
        forecast(monthly(), 3, graph=dag(["pmi[t-1] -> sales[t]"]))


def test_a_scenario_points_at_the_driver_sprint() -> None:
    with pytest.raises(NotImplementedError, match="S4"):
        forecast(monthly(), 3, scenario={"pmi": "hold"})


def test_reconciling_flat_data_is_a_value_error() -> None:
    """Criterion 13: there is no hierarchy to make coherent, so the keyword is wrong."""
    with pytest.raises(ValueError, match="needs a hierarchy"):
        forecast(monthly(), 3, reconcile="bottom_up")
    with pytest.raises(ValueError, match="needs a hierarchy"):
        forecast(monthly(), 3, reconcile="mint")


def test_staged_fitting_points_at_the_driver_sprint() -> None:
    with pytest.raises(NotImplementedError, match="S4"):
        forecast(monthly(), 3, fit="staged")
    with pytest.raises(ValueError, match="fit must be one of"):
        forecast(monthly(), 3, fit="magic")


def test_the_facade_refuses_the_wrong_shapes() -> None:
    """A tabular or panel dataset has no single time axis to walk forward."""
    with pytest.raises(TypeError, match="needs a TimeSeriesData"):
        forecast(tabular({"x": [1.0, 2.0, 3.0]}), 3)  # type: ignore[arg-type]
    frame = {
        "unit": ["a"] * 6 + ["b"] * 6,
        "t": list(range(6)) * 2,
        "y": list(np.arange(12, dtype=float)),
    }
    with pytest.raises(TypeError, match="needs a TimeSeriesData"):
        forecast(panel(frame, entity="unit", time="t"), 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon must be"):
        forecast(monthly(), 0)


def test_a_missing_engine_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 13: with pymc unimportable, the error names the extra."""
    monkeypatch.setitem(sys.modules, "pymc", None)
    with pytest.raises(ImportError, match=r"ergodic\[forecast\]"):
        require_pymc()
    with pytest.raises(ImportError, match=r"ergodic\[forecast\]"):
        forecast(monthly(24), 3, draws=10, warmup=10)


def test_importing_ergodic_does_not_import_the_engine() -> None:
    """The bare install never pays for pymc, which is why the imports live in methods."""
    import subprocess

    code = (
        "import sys; import ergodic; "
        "assert 'pymc' not in sys.modules, 'pymc leaked into import ergodic'; "
        "assert 'pytensor' not in sys.modules, 'pytensor leaked into import ergodic'; "
        "assert callable(ergodic.forecast)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_top_level_name_is_the_function() -> None:
    """`from ergodic import forecast` gives the callable; the module is still reachable."""
    from ergodic.forecast import backtest

    assert callable(forecast)
    assert callable(backtest)


# -- the sampled path ---------------------------------------------------------


@pytest.fixture(scope="module")
def small_forecast() -> Any:
    """One cheap facade run, reused by the tests that only need to look at it."""
    try:
        return forecast(monthly(48), 6, draws=300, warmup=300, chains=1, seed=3)
    except Exception as exc:  # pytensor needs a C compiler and Python headers
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise


@requires_pymc
def test_the_result_carries_its_provenance(small_forecast: Any) -> None:
    """Criterion 9: the chosen period is visible on the result, with the diagnostics."""
    result = small_forecast
    assert result.kind == "draws"
    assert result.nodes == ("sales",)
    assert result.draws("sales").shape == (300, 6)
    assert result.horizon == 6
    mechanism = result.mechanisms["sales"]
    assert isinstance(mechanism, StructuralMechanism)
    assert mechanism.period == 12
    diagnostics = result.diagnostics["sales"]
    assert diagnostics.node == "sales"
    assert diagnostics.n_draws == 300
    assert diagnostics.min_ess > 0
    assert result.start is not None
    lo, hi = result.interval("sales", level=0.9)
    assert bool(np.all(hi > lo))
    frame = result.to_frame()
    assert len(frame) == 6


@requires_pymc
def test_the_same_call_twice_is_bitwise_identical(small_forecast: Any) -> None:
    """Criterion 5: the determinism golden, and the pinned median at the first step."""
    again = forecast(monthly(48), 6, draws=300, warmup=300, chains=1, seed=3)
    assert np.array_equal(small_forecast.draws("sales"), again.draws("sales"))
    # Golden, measured and recorded in the sprint close-out: 223.047381. The
    # tolerance is loose enough that another machine's floating point does not
    # fail the suite and tight enough that a change in behavior does; the bitwise
    # claim above is the one that is exact.
    assert float(small_forecast.median("sales")[0]) == pytest.approx(223.0474, rel=0.005)


@requires_pymc
def test_a_different_seed_moves_the_draws(small_forecast: Any) -> None:
    """Seeds are real: the same data under another seed is not the same sample."""
    other = forecast(monthly(48), 6, draws=300, warmup=300, chains=1, seed=4)
    assert not np.array_equal(small_forecast.draws("sales"), other.draws("sales"))


@requires_pymc
def test_robust_reaches_the_fit_through_the_facade(small_forecast: Any) -> None:
    """The flag reaches the auto-assigned mechanism and the Student-t model samples.

    The mechanism-level comparison of the two likelihoods lives in
    `test_forecast_mechanisms.py`; what this pins is the wiring, that
    `robust=True` is not quietly dropped somewhere between the keyword and the
    model that gets built.
    """
    result = forecast(monthly(48), 6, draws=300, warmup=300, chains=1, seed=3, robust=True)
    assert result.mechanisms is not None
    mechanism = result.mechanisms["sales"]
    assert isinstance(mechanism, StructuralMechanism)
    assert mechanism.robust is True
    assert result.draws("sales").shape == (300, 6)
    assert bool(np.all(np.isfinite(result.draws("sales"))))
    # a different likelihood is a different model, so the draws cannot match the
    # Normal fit's under the same seed
    assert not np.array_equal(result.draws("sales"), small_forecast.draws("sales"))


@requires_pymc
def test_two_series_get_their_own_seeds_and_mechanisms() -> None:
    """Each column is fitted alone, with a seed of its own, and both land on the result."""
    stamps = np.array(
        [np.datetime64("2019-01-01") + np.timedelta64(31 * i, "D") for i in range(36)]
    )
    data = timeseries(
        {"month": stamps, "sales": series(36, seed=1), "visits": series(36, seed=2, level=90.0)},
        time="month",
    )
    result = forecast(data, 4, draws=300, warmup=300, chains=1, seed=0)
    assert result.diagnostics is not None
    assert set(result.nodes) == {"sales", "visits"}
    assert result.draws("sales").shape == (300, 4)
    assert not np.array_equal(result.draws("sales"), result.draws("visits"))
    assert set(result.diagnostics) == {"sales", "visits"}
    assert np.median(result.draws("sales")) > np.median(result.draws("visits"))
