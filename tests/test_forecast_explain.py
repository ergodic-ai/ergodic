# SPDX-License-Identifier: Apache-2.0
"""The attribution waterfall, against a world whose bars are computable in advance.

One world carries most of the criteria. A series is built as
``(level + 1.0 x1 - 0.5 x2) * season``, the two drivers are pinned to constants
over the horizon through ``scenario=``, and the seasonal factors are planted, so
every driver bar has an arithmetic target: the coefficient times the pinned
level times the seasonal factors it is composed through. Nothing about the
decomposition is checked against itself.

The bars say what each piece contributes **at its own level**, not what a change
in it would do: a driver pinned at 100 with a coefficient of 1.0 owns 100 units
of the forecast per step, season included. That is the composition the model
actually wrote, and it is what makes the bars add up.

Skipped where pymc is missing, per the ``test_scm_pymc.py`` precedent; the parts
that need no sampler run anyway.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from ergodic import forecast, hts, timeseries
from ergodic.data import Hierarchy
from ergodic.forecast import (
    Bass,
    CausalOU,
    ForecastResult,
    LogScaler,
    RatioScaler,
    SeasonalNaive,
    StructuralMechanism,
    Window,
    ZScoreScaler,
    regression_keys,
    window_graph,
)
from ergodic.forecast._attribution import _affine_terms, _states

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

    Array = npt.NDArray[np.float64]

requires_pymc = pytest.mark.skipif(
    importlib.util.find_spec("pymc") is None, reason="pymc is not installed"
)

WORLD_SEED = 20260805
T = 60
H = 6
DRAWS = 300
WARMUP = 300
SEED = 11

#: The planted coefficients, in the data's own units.
BETA1, BETA2 = 1.0, -0.5
#: What the drivers are pinned to over the horizon. Constant on purpose: it is
#: what makes a bar's expected value arithmetic rather than a simulation.
C1, C2 = 100.0, 20.0
#: The planted seasonal amplitude, as a fraction.
AMPLITUDE = 0.10

GRAPH = window_graph([("x1", "y", 0), ("x2", "y", 0)])
SCENARIO = {"x1": C1, "x2": C2}


def seasonal_factors(amplitude: float) -> Array:
    """Twelve multiplicative factors averaging exactly one."""
    raw = 1.0 + amplitude * np.sin(2 * np.pi * np.arange(12) / 12.0)
    return np.asarray(raw / raw.mean(), dtype=np.float64)


def two_driver_world(seed: int = WORLD_SEED, amplitude: float = AMPLITUDE) -> pd.DataFrame:
    """``y = (level + 1.0 x1 - 0.5 x2) * season + noise``, monthly, T steps.

    The drivers revert fast (persistence 0.5) so their movement looks nothing
    like the wandering level, which is what identifies the coefficients rather
    than letting the trend absorb them.
    """
    rng = np.random.default_rng(seed)
    season = seasonal_factors(amplitude)[np.arange(T) % 12]

    x1 = np.empty(T)
    x2 = np.empty(T)
    x1[0], x2[0] = C1, C2
    for t in range(1, T):
        x1[t] = C1 + 0.5 * (x1[t - 1] - C1) + rng.normal(0.0, 8.0)
        x2[t] = C2 + 0.5 * (x2[t - 1] - C2) + rng.normal(0.0, 4.0)

    level = np.empty(T)
    level[0] = 200.0
    for t in range(1, T):
        level[t] = level[t - 1] + 0.5 + rng.normal(0.0, 0.6)

    y = (level + BETA1 * x1 + BETA2 * x2) * season + rng.normal(0.0, 3.0, size=T)
    return pd.DataFrame(
        {
            "t": pd.date_range("2020-01-31", periods=T, freq="ME"),
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )


def horizon_factors(amplitude: float = AMPLITUDE) -> Array:
    """The planted seasonal factors over the forecast steps."""
    return seasonal_factors(amplitude)[np.arange(T, T + H) % 12]


def forecast_or_skip(*args: Any, **kwargs: Any) -> ForecastResult:
    """Forecast, or skip the test when this sandbox cannot compile a model."""
    try:
        return forecast(*args, **kwargs)
    except Exception as exc:  # pytensor needs a C compiler and Python headers
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise


def view_of(result: ForecastResult, node: str) -> Any:
    """One node's forecast view off the result, with the provenance asserted present."""
    assert result.views is not None
    return result.views[node]


def fit_of(result: ForecastResult, node: str) -> Any:
    """One node's fitted handle off the result, same reason."""
    assert result.fits is not None
    return result.fits[node]


@dataclass(frozen=True)
class OpaqueMechanism(StructuralMechanism):
    """A mechanism the pillar did not write.

    It is the workhorse in every other respect, so the two engines can be
    compared on one world and one posterior; what it does not do is offer a
    decomposition, which is the default for any mechanism a user writes. The
    attribution then falls back to replay, which finds the factors by the naming
    contract alone.
    """

    def factors(self, node: str, components: Any) -> tuple[str, ...] | None:
        return None


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def world_data() -> TimeSeriesData:
    return timeseries(two_driver_world(), time="t", freq="ME")


@pytest.fixture(scope="module")
def attributed(world_data: TimeSeriesData) -> ForecastResult:
    """The reference run: y driven by two pinned drivers, fitted staged."""
    return forecast_or_skip(
        world_data,
        H,
        graph=GRAPH,
        scenario=SCENARIO,
        fit="staged",
        draws=DRAWS,
        warmup=WARMUP,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def opaque(world_data: TimeSeriesData) -> ForecastResult:
    """The same world and the same seed, through a mechanism that hides its pieces."""
    return forecast_or_skip(
        world_data,
        H,
        graph=GRAPH,
        scenario=SCENARIO,
        mechanisms={"y": OpaqueMechanism(period=12)},
        fit="staged",
        draws=DRAWS,
        warmup=WARMUP,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def staged_frozen_opaque(world_data: TimeSeriesData) -> ForecastResult:
    """A staged handoff and an opaque mechanism at once: the two features crossed.

    The driver is modelled rather than pinned, so the child's forecast carries a
    real frozen parent (an injected ``x1_fc`` site), and the child's mechanism
    hides its pieces, so the attribution has to replay. Every other test picks one
    of those two; this is the pair.
    """
    return forecast_or_skip(
        world_data.select(["y", "x1"]),
        H,
        graph=window_graph([("x1", "y", 0)]),
        mechanisms={"y": OpaqueMechanism(period=12)},
        fit="staged",
        draws=200,
        warmup=200,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def seasonless() -> ForecastResult:
    """A world with no season at all, fitted with a monthly one anyway."""
    data = timeseries(two_driver_world(amplitude=0.0), time="t", freq="ME")
    return forecast_or_skip(
        data,
        H,
        graph=GRAPH,
        scenario=SCENARIO,
        fit="staged",
        draws=DRAWS,
        warmup=WARMUP,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def additive(world_data: TimeSeriesData) -> ForecastResult:
    """The seasonal world through the Fourier form, where the season adds rather than scales."""
    return forecast_or_skip(
        world_data,
        H,
        graph=GRAPH,
        scenario=SCENARIO,
        mechanisms={"y": StructuralMechanism(seasonality="fourier", period=12)},
        fit="staged",
        draws=DRAWS,
        warmup=WARMUP,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def joint_pair(world_data: TimeSeriesData) -> ForecastResult:
    """One model over the driver and the child, so the child reads a latent mean path."""
    return forecast_or_skip(
        world_data.select(["y", "x1"]),
        H,
        graph=window_graph([("x1", "y", 0)]),
        fit="joint",
        draws=200,
        warmup=200,
        seed=SEED,
    )


@pytest.fixture(scope="module")
def joint_opaque(world_data: TimeSeriesData) -> ForecastResult:
    """The same joint shape with a mechanism that neither decomposes nor replays."""
    return forecast_or_skip(
        world_data.select(["y", "x1"]),
        H,
        graph=window_graph([("x1", "y", 0)]),
        mechanisms={"y": OpaqueMechanism(period=12)},
        fit="joint",
        draws=150,
        warmup=150,
        seed=SEED,
    )


# -- criterion 1: the bars close ----------------------------------------------


@requires_pymc
def test_the_bars_close_on_the_total(attributed: ForecastResult) -> None:
    record = attributed.explain()["y"]
    assert record.engine == "structural"
    assert record.relative_closure_error < 1e-6
    # The closure is measured against the model's own recorded mean path, not
    # against the sum of the bars, so it can fail: it is what would catch a
    # composition rule that drifted away from build().
    assert abs(record.closure_error) < 1e-6 * abs(record.total)
    assert record.baseline.name == "Level and trend"
    assert [bar.name for bar in record.factors] == ["season", "x1", "x2"]


@requires_pymc
def test_every_window_closes_and_the_windows_add_up(attributed: ForecastResult) -> None:
    whole = attributed.explain()["y"]
    parts = attributed.explain(windows=[Window("early", 0, 3), Window("late", 3, H)])
    for record in parts:
        assert record.relative_closure_error < 1e-6
    for index, bar in enumerate(whole.bars):
        halves = sum(record.bars[index].value for record in parts)
        assert bar.value == pytest.approx(halves, rel=1e-9)


@requires_pymc
def test_loo_closes_too_by_booking_what_no_factor_owns(attributed: ForecastResult) -> None:
    record = attributed.explain(method="loo")["y"]
    assert record.relative_closure_error < 1e-6
    assert record.factors[-1].name == "Interaction"
    # A multiplicative season scales the drivers, so leaving each factor out one
    # at a time genuinely does not recover the whole. The bar is real structure.
    assert abs(record.factors[-1].value) > 1e-6 * abs(record.total)


# -- criterion 2: the bars land on the planted contributions -------------------


@requires_pymc
def test_each_driver_bar_lands_on_its_planted_contribution(attributed: ForecastResult) -> None:
    record = attributed.explain()["y"]
    bars = {bar.name: bar for bar in record.factors}
    season_sum = float(horizon_factors().sum())

    assert bars["x1"].value > 0.0
    assert bars["x2"].value < 0.0
    for name, beta, pinned in (("x1", BETA1, C1), ("x2", BETA2, C2)):
        target = beta * pinned * season_sum
        low, high = bars[name].credible_interval(0.9)
        assert low <= target <= high, f"{name}: {target} outside ({low}, {high})"


@requires_pymc
def test_the_mean_reduce_is_the_sum_over_the_windows_length(attributed: ForecastResult) -> None:
    summed = attributed.explain(reduce="sum")["y"]
    averaged = attributed.explain(reduce="mean")["y"]
    for one, other in zip(summed.bars, averaged.bars, strict=True):
        assert other.value == pytest.approx(one.value / H, rel=1e-9)


# -- criterion 3: replay agrees with the structural path ----------------------


@requires_pymc
def test_replay_agrees_with_the_structural_decomposition(
    attributed: ForecastResult, opaque: ForecastResult
) -> None:
    # Same world, same seed, same model: only the decomposition path differs.
    assert np.array_equal(attributed.draws("y"), opaque.draws("y"))

    replayed = opaque.explain()["y"]
    structural = attributed.explain()["y"]
    assert replayed.engine == "replay"
    assert structural.engine == "structural"
    assert replayed.n_replays == 4  # three factors, so four cumulative states

    # Closure here measures whether the re-run reproduced the forecast it is
    # explaining, which is the seed discipline seen from the outside.
    assert abs(replayed.closure_error) < 0.02 * abs(replayed.total)
    for one, other in zip(structural.bars, replayed.bars, strict=True):
        assert one.name == other.name
        assert abs(one.value - other.value) < 0.10 * abs(one.value)


@requires_pymc
def test_replay_finds_its_factors_by_the_naming_contract(opaque: ForecastResult) -> None:
    record = opaque.explain()["y"]
    assert [bar.name for bar in record.factors] == ["season", "x1", "x2"]


@requires_pymc
def test_the_replay_budget_is_a_ceiling(opaque: ForecastResult) -> None:
    with pytest.raises(ValueError, match="max_replays"):
        opaque.explain(max_replays=2)


@requires_pymc
def test_replay_works_through_a_staged_handoff(staged_frozen_opaque: ForecastResult) -> None:
    """The crossing case: a replayed node whose parent's future is a posterior.

    A replay rebuilds the child's forecast from a *zeroed copy* of its trace,
    which is a bare posterior rather than a full sampler result, and that copy
    then has to carry the parent's injected future through the same handoff the
    original forecast used. Both halves work on their own; this is the test that
    they work together, and it is the one the suite was missing, because
    criterion 3 pins the drivers with ``scenario=`` so its bar values stay
    computable and a pinned driver needs no handoff at all.
    """
    record = staged_frozen_opaque.explain()["y"]
    assert record.engine == "replay"
    assert [bar.name for bar in record.factors] == ["season", "x1"]
    # Three cumulative states for two factors, and the bars still telescope onto
    # the forecast the result actually holds.
    assert record.n_replays == 3
    assert abs(record.closure_error) < 0.02 * abs(record.total)
    assert view_of(staged_frozen_opaque, "y").replay.frozen[0].base == "x1"


@requires_pymc
def test_every_window_of_a_node_shares_one_set_of_replays(opaque: ForecastResult) -> None:
    """Asking four questions of one node costs what asking one does.

    A re-forecast produces the whole horizon; a window only decides how it is
    summed up. Spending a sampler pass per window would multiply the cost by the
    number of questions, and the budget would run out for the wrong reason.
    """
    whole = opaque.explain()["y"]
    parts = opaque.explain(windows=[Window("early", 0, 3), Window("late", 3, H)])
    assert [record.n_replays for record in parts] == [whole.n_replays, whole.n_replays]
    for index, bar in enumerate(whole.bars):
        halves = sum(record.bars[index].value for record in parts)
        assert bar.value == pytest.approx(halves, rel=1e-9)


# -- criterion 4: zeroing is sharp --------------------------------------------


@requires_pymc
def test_a_seasonless_world_gets_a_seasonality_bar_of_nothing(
    seasonless: ForecastResult,
) -> None:
    record = seasonless.explain()["y"]
    season = next(bar for bar in record.factors if bar.name == "season")
    assert abs(season.value) < 0.05 * abs(record.total)
    low, high = season.credible_interval(0.9)
    assert low < 0.0 < high


@requires_pymc
def test_an_additive_world_has_no_interaction(
    additive: ForecastResult, attributed: ForecastResult
) -> None:
    # The Fourier season is a term, not a factor, so leaving pieces out one at a
    # time recovers the whole exactly and there is nothing left to book.
    record = additive.explain(method="loo")["y"]
    interaction = next(bar for bar in record.factors if bar.name == "Interaction")
    assert abs(interaction.value) < 1e-9 * abs(record.total)

    # The contrast is what stops that from being vacuous: the same world with a
    # multiplicative season leaves a real interaction behind.
    multiplicative = attributed.explain(method="loo")["y"]
    other = next(bar for bar in multiplicative.factors if bar.name == "Interaction")
    assert abs(other.value) > 0.005 * abs(multiplicative.total)


# -- criterion 5: same-seed discipline ----------------------------------------


@requires_pymc
def test_two_explain_calls_are_bitwise_identical(attributed: ForecastResult) -> None:
    first = attributed.explain()["y"]
    second = attributed.explain()["y"]
    for one, other in zip(first.bars, second.bars, strict=True):
        assert np.array_equal(one.draws, other.draws)
    assert np.array_equal(first.total_draws, second.total_draws)


@requires_pymc
def test_a_replay_reuses_the_forecasts_own_seed(attributed: ForecastResult) -> None:
    from ergodic.forecast import forecast_node

    recipe = view_of(attributed, "y").replay
    # sorted nodes are x1, x2, y, so y fits at SEED + 2 and forecasts a thousand
    # past that, which is the facade's rule and what a replay has to reuse.
    assert recipe.seed == SEED + 2 + 1000
    again = forecast_node(
        fit_of(attributed, "y"),
        recipe.horizon,
        parents=recipe.parents,
        frozen=recipe.frozen,
        seed=recipe.seed,
    )
    assert np.array_equal(again.draws, view_of(attributed, "y").draws)


# -- the other paths ----------------------------------------------------------


@requires_pymc
def test_a_pinned_driver_has_no_model_to_explain(attributed: ForecastResult) -> None:
    assert attributed.explain().node_names == ("y",)
    with pytest.raises(ValueError, match="no model of its own"):
        attributed.explain(nodes="x1")
    with pytest.raises(KeyError, match="no series"):
        attributed.explain(nodes="nope")


@requires_pymc
def test_a_joint_fit_decomposes_structurally(joint_pair: ForecastResult) -> None:
    record = joint_pair.explain(nodes="y")["y"]
    assert record.engine == "structural"
    assert record.relative_closure_error < 1e-6
    assert [bar.name for bar in record.factors] == ["season", "x1"]


@requires_pymc
def test_a_joint_fit_cannot_replay_one_node(joint_opaque: ForecastResult) -> None:
    # A joint forecast is one predictive pass over every node at once, so there
    # is no per-node call to run again. A mechanism that decomposes itself never
    # needs one; one that does not is refused with the reason rather than with a
    # forecast built from a different posterior.
    assert view_of(joint_opaque, "y").replay is None
    with pytest.raises(NotImplementedError, match="joint fit"):
        joint_opaque.explain(nodes="y")


@requires_pymc
def test_the_reversion_decomposition_reproduces_its_own_mean_path() -> None:
    """The CausalOU filter is a second statement of the scan, so it is measured.

    Its drivers reach the series through the reversion recursion, which the
    decomposition re-walks in numpy. Closure is the guard: if the two ever drift
    apart, this is the number that moves.
    """
    rng = np.random.default_rng(WORLD_SEED)
    n = 60
    driver = np.repeat(rng.normal(0.0, 1.0, size=n // 5), 5)[:n]
    # Centered on its own window, so that pinning it above zero really is
    # pinning it above where it has been: a z-scored driver's bar reads against
    # its own mean, and a world whose driver happens to sit high would make the
    # sign of the pull a fact about the sample rather than about the mechanism.
    driver = driver - driver.mean()
    level = np.empty(n)
    level[0] = 20.0
    for t in range(1, n):
        target = 20.0 + 1.0 * driver[t]
        level[t] = level[t - 1] + 0.3 * (target - level[t - 1]) + rng.normal(0.0, 0.15)
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2020-01-31", periods=n, freq="ME"),
            "rate": level + rng.normal(0.0, 0.1, size=n),
            "z": driver,
        }
    )
    data = timeseries(frame, time="t", freq="ME")
    result = forecast_or_skip(
        data,
        H,
        graph=window_graph([("z", "rate", 0)]),
        scenario={"z": 2.0},
        mechanisms={"rate": CausalOU(seasonality=None)},
        fit="staged",
        draws=200,
        warmup=200,
        seed=SEED,
    )
    record = result.explain(nodes="rate")["rate"]
    assert record.engine == "structural"
    assert record.baseline.name == "Level and reversion"
    assert [bar.name for bar in record.factors] == ["z"]
    assert record.relative_closure_error < 1e-8
    # The driver was pinned above its own training mean, so its pull is positive.
    assert record.factors[0].value > 0.0


@requires_pymc
def test_a_diffusion_curve_is_all_baseline() -> None:
    rng = np.random.default_rng(WORLD_SEED)
    n = 40
    p, q, market = 0.03, 0.4, 1000.0
    cumulative = 0.0
    adoption = np.empty(n)
    for t in range(n):
        step = (p + q * cumulative) * (1.0 - cumulative)
        cumulative += step
        adoption[t] = market * step
    frame = pd.DataFrame({"t": np.arange(float(n)), "units": adoption + rng.normal(0, 6.0, n)})
    result = forecast_or_skip(
        timeseries(frame, time="t"),
        H,
        mechanisms={"units": Bass()},
        draws=200,
        warmup=200,
        seed=3,
    )
    record = result.explain()["units"]
    assert record.engine == "structural"
    assert record.factors == ()
    assert record.baseline.name == "Diffusion"
    assert record.baseline.value == pytest.approx(record.total, rel=1e-9)


@requires_pymc
def test_a_reconciled_result_says_what_reconciliation_moved() -> None:
    rng = np.random.default_rng(WORLD_SEED)
    n = 36
    step = np.arange(n, dtype=np.float64)
    leaf_a = 100.0 + 1.5 * step + rng.normal(0.0, 4.0, n)
    leaf_b = 60.0 + 0.5 * step + rng.normal(0.0, 3.0, n)
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2021-01-31", periods=n, freq="ME"),
            "leaf_a": leaf_a,
            "leaf_b": leaf_b,
            "total": leaf_a + leaf_b,
        }
    )
    tree = Hierarchy.tree({"total": ["leaf_a", "leaf_b"]})
    data = hts(frame, time="t", hierarchy=tree, freq="ME")
    result = forecast_or_skip(data, 4, reconcile="mint", draws=200, warmup=200, seed=SEED)
    record = result.explain(nodes="leaf_a")["leaf_a"]
    # The bars explain the model, and the model ran before reconciliation moved
    # the draws, so the two totals are allowed to differ and both are reported.
    assert record.relative_closure_error < 1e-6
    assert record.forecast_total != pytest.approx(record.total, rel=1e-12)
    assert abs(record.forecast_total - record.total) < 0.25 * abs(record.total)


# -- the parts that need no sampler -------------------------------------------


def test_a_window_reads_integer_positions() -> None:
    assert Window("all").resolve(6) == (0, 6)
    assert Window("early", 0, 3).resolve(6) == (0, 3)
    assert Window("late", 3).resolve(6) == (3, 6)
    with pytest.raises(ValueError, match="empty or out of range"):
        Window("backwards", 4, 2).resolve(6)
    with pytest.raises(ValueError, match="empty or out of range"):
        Window("past the end", 0, 9).resolve(6)


def test_a_window_reads_dates_off_the_result_index() -> None:
    index = tuple(np.datetime64(f"2026-{month:02d}-28") for month in range(1, 7))
    result = ForecastResult(series={"y": np.ones((4, 6))}, horizon=6, index=index)
    assert Window("q1", "2026-01-01", "2026-04-01").resolve(6, result.index) == (0, 3)
    assert Window("q2", "2026-04-01", None).resolve(6, result.index) == (3, 6)


def test_a_date_window_needs_a_time_axis() -> None:
    with pytest.raises(ValueError, match="carries no time axis"):
        Window("q1", "2026-01-01").resolve(6, None)


def test_a_bar_carries_a_posterior_rather_than_a_point() -> None:
    from ergodic.forecast import FactorContribution

    draws = np.linspace(-1.0, 3.0, 401)
    bar = FactorContribution("pmi", "driver", draws)
    assert bar.value == pytest.approx(1.0)
    low, high = bar.credible_interval(0.9)
    assert low == pytest.approx(-0.8, abs=0.02)
    assert high == pytest.approx(2.8, abs=0.02)
    # It owns its draws: the caller's array stays writable and cannot reach in.
    draws[0] = 1e6
    assert bar.draws[0] == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        bar.draws[0] = 0.0


def test_a_log_scale_cannot_carry_a_waterfall() -> None:
    assert _affine_terms(RatioScaler(median=4.0)) == (4.0, 0.0)
    assert _affine_terms(ZScoreScaler(mean=2.0, std=3.0)) == (3.0, 2.0)
    with pytest.raises(NotImplementedError, match="straight line"):
        _affine_terms(LogScaler())


def test_a_mechanism_that_recorded_nothing_falls_through_to_replay() -> None:
    # The dispatch reads what the forecast recorded, not what the mechanism is,
    # so a subclass that suppresses its deterministics lands on replay.
    assert StructuralMechanism(period=12).factors("y", {}) is None
    assert CausalOU().factors("y", {}) is None
    assert Bass().factors("units", {}) is None
    assert Bass().factors("units", {"units_mu": np.ones((2, 3))}) == ()


def test_a_driver_named_season_is_refused() -> None:
    with pytest.raises(ValueError, match="Rename the column"):
        regression_keys("y", {"y_reg_season": np.ones((2, 3))})


def test_the_replay_states_walk_the_subsets() -> None:
    assert _states(("season", "pmi"), "sequential") == ((), ("season",), ("season", "pmi"))
    assert _states(("season", "pmi"), "loo") == (
        (),
        ("pmi",),
        ("season",),
        ("season", "pmi"),
    )


def test_a_result_with_no_model_cannot_be_explained() -> None:
    frame = pd.DataFrame({"t": np.arange(24.0), "y": np.arange(24.0) + 1.0})
    data = timeseries(frame, time="t")
    baseline = SeasonalNaive(period=4).fit(data, seed=0).forecast(4, seed=1)
    with pytest.raises(ValueError, match="no fitted model"):
        baseline.explain()


def test_explain_refuses_an_option_it_does_not_have() -> None:
    result = ForecastResult(series={"y": np.ones((4, 6))}, horizon=6)
    with pytest.raises(ValueError, match="method must be"):
        result.explain(method="nope")
    with pytest.raises(ValueError, match="reduce must be"):
        result.explain(reduce="median")
    with pytest.raises(TypeError, match="Window objects"):
        result.explain(windows=[(0, 3)])
