# SPDX-License-Identifier: Apache-2.0
"""Drivers and staged fitting, against planted truths.

World D1 is the sprint's spine: a driver index that wanders and wiggles, and a
series that reads it one month later, plus its own trend and season. Everything
about propagation is asked of that one world, because the interesting questions
are all comparisons on it: does the coefficient come back, does the driver's
uncertainty reach the child, does staging agree with fitting jointly, does
pinning the driver narrow the band.

The driver's persistence is a design decision the sprint measured rather than
guessed; the note on :data:`AR_PERSISTENCE` carries the numbers. Nothing under
test is mocked. Skipped wholesale when pymc is absent, the ``test_scm_pymc.py``
precedent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("pymc")

from ergodic import discover, forecast, hts, timeseries
from ergodic.data import Hierarchy
from ergodic.forecast import (
    AUTO_JOINT_MAX_NODES,
    FrozenParent,
    ScenarioMechanism,
    StructuralMechanism,
    design_keys,
    lag_align,
    node_roles,
    schedule,
    validate_window_graph,
    window_graph,
)
from ergodic.forecast._fit import _audit_sites, _group_dataset, align_draws, inject_futures
from ergodic.forecast._schedule import NodeRole

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData
    from ergodic.forecast import ForecastResult

    Array = npt.NDArray[np.float64]

WORLD_SEED = 20260805
WARMUP = 300
DRAWS = 400
SMALL_WARMUP = 200
SMALL_DRAWS = 200

#: The planted lag-one coefficient of D1's child on its driver.
BETA = 0.8

#: How persistent the driver's own movement is, and why this number.
#:
#: Measured, on two world seeds, staged against joint (share = mean absolute
#: difference of the median paths as a fraction of the child's standard
#: deviation):
#:
#: | a    | corr          | share           | staged beta   | joint beta  |
#: |------|---------------|-----------------|---------------|-------------|
#: | 0.55 | 0.995, 0.994  | 0.086, 0.169    | 0.804, 0.803  | 0.93, 0.91  |
#: | 0.75 | 0.999, 0.999  | 0.053, 0.107    | 0.815, 0.793  | 0.90, 0.90  |
#: | 0.90 | 1.000, 0.999  | 0.025, 0.064    | 0.804, 0.758  | 0.90, 0.88  |
#:
#: The reason is worth keeping: a driver whose movement is nearly white noise
#: cannot be carried by its own damped-trend mechanism, which reads the wiggle as
#: observation noise and hands the child a smooth mean path. The joint fit then
#: regresses on that smoothed path and raises beta to compensate, and the two
#: strategies part company. A persistent driver is one its mechanism can actually
#: represent, and then they agree.
AR_PERSISTENCE = 0.9


# -- World D1 -----------------------------------------------------------------


def d1_world(seed: int, n: int = 100) -> tuple[Array, Array]:
    """A driver and the series that reads it one step later, in original units.

    The driver is an index near 100: a slow drift plus a persistent deviation
    plus a little measurement noise. The child is that driver at lag one, scaled
    by 0.8, on its own rising trend and a plus-or-minus-seven-percent season. Both
    live where a monthly business series lives, so nothing here is standardized
    toy data.
    """
    rng = np.random.default_rng(seed)
    drift = 100.0 + np.cumsum(rng.normal(0.05, 0.25, n))
    deviation = np.empty(n, dtype=np.float64)
    state = 0.0
    for t in range(n):
        state = AR_PERSISTENCE * state + rng.normal(0.0, 1.6)
        deviation[t] = state
    driver = drift + deviation + rng.normal(0.0, 0.4, n)
    lagged = np.concatenate([[driver[0]], driver[:-1]])
    season = 1.0 + 0.07 * np.sin(2.0 * np.pi * np.arange(n) / 12.0)
    child = (10.0 + BETA * lagged + 0.25 * np.arange(n)) * season + rng.normal(0.0, 1.2, n)
    return driver, child


def monthly(columns: dict[str, Array], n: int) -> TimeSeriesData:
    """Wrap planted columns as a monthly dataset of ``n`` steps."""
    stamps = np.array([np.datetime64("2016-01-01") + np.timedelta64(31 * i, "D") for i in range(n)])
    return timeseries({"month": stamps, **{k: v[:n] for k, v in columns.items()}}, time="month")


D1_TOTAL = 100
D1_HOLDOUT = 12
D1_TRAIN = D1_TOTAL - D1_HOLDOUT

D1_GRAPH = window_graph([("x", "y", 1)])


def forecast_or_skip(*args: Any, **kwargs: Any) -> ForecastResult:
    """Forecast, or skip the test when this sandbox cannot compile a model."""
    try:
        return forecast(*args, **kwargs)
    except Exception as exc:  # pytensor needs a C compiler and Python headers
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise


@pytest.fixture(scope="module")
def d1_truth() -> tuple[Array, Array]:
    return d1_world(WORLD_SEED, D1_TOTAL)


@pytest.fixture(scope="module")
def d1_data(d1_truth: tuple[Array, Array]) -> TimeSeriesData:
    x, y = d1_truth
    return monthly({"x": x, "y": y}, D1_TRAIN)


@pytest.fixture(scope="module")
def staged_d1(d1_data: TimeSeriesData) -> ForecastResult:
    """The sprint's reference run: both nodes staged, the driver forecast by its mechanism."""
    return forecast_or_skip(
        d1_data,
        D1_HOLDOUT,
        graph=D1_GRAPH,
        fit="staged",
        draws=DRAWS,
        warmup=WARMUP,
        seed=11,
    )


@pytest.fixture(scope="module")
def pinned_d1(d1_data: TimeSeriesData, d1_truth: tuple[Array, Array]) -> ForecastResult:
    """The same run with the driver's true future pinned, so nothing propagates."""
    x, _ = d1_truth
    return forecast_or_skip(
        d1_data,
        D1_HOLDOUT,
        graph=D1_GRAPH,
        scenario={"x": x[D1_TRAIN : D1_TRAIN + D1_HOLDOUT]},
        fit="staged",
        draws=DRAWS,
        warmup=WARMUP,
        seed=11,
    )


@pytest.fixture(scope="module")
def joint_d1(d1_data: TimeSeriesData) -> ForecastResult:
    """The same world fitted as one model over both nodes."""
    return forecast_or_skip(
        d1_data,
        D1_HOLDOUT,
        graph=D1_GRAPH,
        fit="joint",
        draws=DRAWS,
        warmup=WARMUP,
        seed=11,
    )


def posterior_mean(result: ForecastResult, node: str, site: str) -> float:
    """One site's posterior mean, off the fit the result carries."""
    assert result.fits is not None
    trace = result.fits[node].trace
    return float(np.asarray(_group_dataset(trace, "posterior")[site]).mean())


def unscale_beta(result: ForecastResult, child: str, parent: str, site: str) -> float:
    """A regression coefficient in the data's own units.

    The child reads its parent on the parent's scaled space and is itself scaled,
    so the fitted coefficient is the planted one times the ratio of the two
    scales. Both scalers are on the result's fits, so this is arithmetic rather
    than a second guess at what the code did.
    """
    assert result.fits is not None
    child_scale = getattr(result.fits[child].scaler, "median", 1.0)
    parent_scaler = result.fits[parent].scaler
    parent_scale = getattr(parent_scaler, "median", getattr(parent_scaler, "std", 1.0))
    return posterior_mean(result, child, site) * float(child_scale) / float(parent_scale)


# -- criterion 1: the coefficient comes back ----------------------------------


def test_the_staged_fit_recovers_the_planted_coefficient(staged_d1: ForecastResult) -> None:
    """Criterion 1: the posterior mean of ``y_beta_x`` lands within 0.2 of 0.8."""
    fitted = posterior_mean(staged_d1, "y", "y_beta_x")
    assert abs(fitted - BETA) <= 0.2, f"y_beta_x posterior mean {fitted:.4f}, planted {BETA}"
    # And in the data's own units, which is the same number here only because the
    # world puts the two series at the same scale on purpose.
    in_units = unscale_beta(staged_d1, "y", "x", "y_beta_x")
    assert abs(in_units - BETA) <= 0.2, f"unscaled beta {in_units:.4f}"


# -- criterion 2: the driver's uncertainty reaches the child -------------------


def mean_width(result: ForecastResult, node: str, level: float = 0.9) -> float:
    lo, hi = result.interval(node, level)
    return float(np.mean(hi - lo))


def test_forecasting_the_driver_widens_the_child(
    staged_d1: ForecastResult, pinned_d1: ForecastResult
) -> None:
    """Criterion 2: a forecast driver hands down uncertainty a pinned one does not."""
    propagated = mean_width(staged_d1, "y")
    pinned = mean_width(pinned_d1, "y")
    assert propagated > pinned, (
        f"the band with the driver forecast ({propagated:.4f}) should be wider than with it "
        f"pinned ({pinned:.4f}); if it is not, the frozen parent's draws are not reaching "
        "the child"
    )
    # The driver's own band is what the child is inheriting, so it had better be
    # a band rather than a line.
    assert mean_width(staged_d1, "x") > 0.0
    assert mean_width(pinned_d1, "x") == pytest.approx(0.0, abs=1e-12)


# -- criterion 3: the propagated band is calibrated ---------------------------


def test_the_propagated_band_is_calibrated() -> None:
    """Criterion 3: pooled 90% coverage over ten seeded worlds sits in [0.78, 0.98].

    Ten worlds because one holdout of six steps resolves coverage to a sixth, and
    a band is a claim about repetition rather than about one path.
    """
    covered = 0
    total = 0
    n_steps, train, horizon = 54, 48, 6
    for index in range(10):
        x, y = d1_world(4100 + index, n_steps)
        data = monthly({"x": x, "y": y}, train)
        result = forecast_or_skip(
            data,
            horizon,
            graph=D1_GRAPH,
            fit="staged",
            draws=SMALL_DRAWS,
            warmup=SMALL_WARMUP,
            seed=100 + index,
        )
        lo, hi = result.interval("y", 0.9)
        truth = y[train : train + horizon]
        covered += int(np.sum((truth >= lo) & (truth <= hi)))
        total += horizon
    share = covered / total
    assert 0.78 <= share <= 0.98, f"pooled 90% coverage {covered}/{total} = {share:.4f}"


# -- criterion 4: the injected site is audited --------------------------------


def frozen_x(result: ForecastResult, data: TimeSeriesData) -> FrozenParent:
    """The handoff the staged run made, rebuilt from the result's own pieces."""
    assert result.fits is not None
    scaler = result.fits["x"].scaler
    return FrozenParent(
        base="x",
        draws=np.asarray(scaler.transform(result.draws("x")), dtype=np.float64),
        history=np.asarray(scaler.transform(np.asarray(data.column("x"))), dtype=np.float64),
        columns=(("x", 1),),
    )


def child_model(result: ForecastResult, handoff: FrozenParent, horizon: int) -> Any:
    """The child's forecast model, with the frozen parent's site declared in it."""
    import pymc as pm

    assert result.fits is not None
    fit = result.fits["y"]
    with pm.Model() as model:
        future = pm.Normal(handoff.site, mu=0.0, sigma=1.0, shape=horizon)
        paths = {
            key: lag_align(handoff.history, future, lag, horizon) for key, lag in handoff.columns
        }
        fit.mechanism.build(
            "y",
            obs=fit.obs_scaled,
            parents=paths,
            T=fit.n_obs,
            horizon=horizon,
            mask=fit.mask,
        )
    return model


def test_the_child_ppc_runs_only_with_the_injected_future(
    staged_d1: ForecastResult, d1_data: TimeSeriesData
) -> None:
    """Criterion 4: ``x_fc`` has to be in the augmented trace, and be declared injected."""
    assert staged_d1.fits is not None
    handoff = frozen_x(staged_d1, d1_data)
    model = child_model(staged_d1, handoff, D1_HOLDOUT)
    trace = staged_d1.fits["y"].trace
    augmented = inject_futures(trace, {handoff.site: handoff.draws})

    # As the run made it: the site is there, and it is declared as replayed.
    _audit_sites(model, augmented, injected=(handoff.site,))

    # Take the future out of the trace and the audit stops the pass.
    stripped = augmented.drop_vars(handoff.site)
    with pytest.raises(ValueError, match="not in the trace"):
        _audit_sites(model, stripped, injected=(handoff.site,))

    # Forget to declare it injected and the audit still stops, from the other
    # side: a site in the trace that was meant to be simulated fresh.
    with pytest.raises(ValueError, match="would be replayed instead"):
        _audit_sites(model, augmented, injected=())

    # And a future injected for a site the model never declared is refused, which
    # is the case where a child would forecast without its driver.
    with pytest.raises(ValueError, match="never declared them"):
        _audit_sites(model, augmented, injected=(handoff.site, "z_fc"))

    # The unaugmented trace is unchanged: injection copies.
    assert handoff.site not in _group_dataset(trace, "posterior")


def test_the_injected_values_actually_reach_the_child(
    staged_d1: ForecastResult, d1_data: TimeSeriesData
) -> None:
    """The audit is a guard; this is the fact it guards. Shift the driver, the child moves.

    A driver future ten units above the fitted one should lift the child by about
    ``beta`` times ten. If the injected draws were being ignored (or drawn from
    their prior), the child would not move at all.
    """
    from ergodic.forecast import forecast_node

    assert staged_d1.fits is not None
    scaler = staged_d1.fits["x"].scaler
    child = staged_d1.fits["y"]
    history = np.asarray(scaler.transform(np.asarray(d1_data.column("x"))), dtype=np.float64)
    base_draws = np.asarray(scaler.transform(staged_d1.draws("x")), dtype=np.float64)
    lifted = np.asarray(scaler.transform(staged_d1.draws("x") + 10.0), dtype=np.float64)

    def run(draws: Array) -> Array:
        handoff = FrozenParent(base="x", draws=draws, history=history, columns=(("x", 1),))
        view = forecast_node(child, D1_HOLDOUT, parents={}, frozen=[handoff], seed=5)
        return np.asarray(np.median(view.draws, axis=0), dtype=np.float64)

    plain = run(base_draws)
    shifted = run(lifted)
    # The first horizon step still reads the last observed driver value (lag one),
    # so the lift arrives from step two onward.
    moved = float(np.mean(shifted[1:] - plain[1:]))
    expected = 10.0 * unscale_beta(staged_d1, "y", "x", "y_beta_x")
    assert moved == pytest.approx(expected, rel=0.25), (
        f"a driver ten units higher moved the child by {moved:.3f}, expected about {expected:.3f}"
    )
    assert shifted[0] == pytest.approx(plain[0], rel=0.05)


# -- criterion 5: staged against joint ----------------------------------------


def test_staged_and_joint_agree_on_a_small_graph(
    staged_d1: ForecastResult, joint_d1: ForecastResult, d1_data: TimeSeriesData
) -> None:
    """Criterion 5: the two strategies land in the same place on a two-node graph."""
    staged_median = staged_d1.median("y")
    joint_median = joint_d1.median("y")
    correlation = float(np.corrcoef(staged_median, joint_median)[0, 1])
    assert correlation > 0.98, f"median paths correlate at {correlation:.4f}"
    scale = float(np.std(np.asarray(d1_data.column("y"))))
    difference = float(np.mean(np.abs(staged_median - joint_median)))
    assert difference < 0.10 * scale, (
        f"mean absolute median difference {difference:.4f} against a series scale of "
        f"{scale:.4f} ({difference / scale:.1%})"
    )


def test_the_joint_fit_is_one_posterior_over_both_nodes(joint_d1: ForecastResult) -> None:
    """What "joint" means, checked: one trace, holding both nodes' sites."""
    assert joint_d1.fits is not None
    assert joint_d1.fits["x"].trace is joint_d1.fits["y"].trace
    posterior = _group_dataset(joint_d1.fits["y"].trace, "posterior")
    names = {str(name) for name in posterior.data_vars}
    assert "x_phi" in names and "y_phi" in names and "y_beta_x" in names
    # Diagnostics are still per node, off that one trace.
    assert joint_d1.diagnostics is not None
    assert set(joint_d1.diagnostics) == {"x", "y"}
    assert joint_d1.diagnostics["x"].node == "x"


# -- criteria 10 and 11: what a scenario means and what it refuses ------------


def small_world(n: int = 40) -> TimeSeriesData:
    x, y = d1_world(31, n)
    return monthly({"x": x, "y": y}, n)


def test_hold_is_the_explicit_last_value_array() -> None:
    """Criterion 10: ``"hold"`` is a convenience, not a different code path."""
    data = small_world()
    last = float(np.asarray(data.column("x"))[-1])
    common = {
        "graph": D1_GRAPH,
        "fit": "staged",
        "draws": 150,
        "warmup": 150,
        "seed": 4,
    }
    held = forecast_or_skip(data, 4, scenario={"x": "hold"}, **common)
    spelled = forecast_or_skip(data, 4, scenario={"x": np.full(4, last)}, **common)
    assert np.array_equal(held.draws("y"), spelled.draws("y"))
    assert np.array_equal(held.draws("x"), spelled.draws("x"))
    assert np.allclose(held.median("x"), last)
    # A scalar is the same thing said a third way.
    scalar = forecast_or_skip(data, 4, scenario={"x": last}, **common)
    assert np.array_equal(held.draws("y"), scalar.draws("y"))


def test_a_known_future_driver_with_no_scenario_raises_before_sampling() -> None:
    """Criterion 10: a ScenarioMechanism with nothing to resolve from is refused by name."""
    data = small_world()
    with pytest.raises(ValueError, match=r"driver 'x' needs a mechanism or scenario="):
        forecast(
            data,
            4,
            graph=D1_GRAPH,
            mechanisms={"x": ScenarioMechanism()},
            fit="staged",
            draws=10,
            warmup=10,
        )
    # Pinned at construction instead, it resolves without scenario=.
    pinned = forecast_or_skip(
        data,
        4,
        graph=D1_GRAPH,
        mechanisms={"x": ScenarioMechanism(values=(101.0, 102.0, 103.0, 104.0))},
        fit="staged",
        draws=150,
        warmup=150,
        seed=4,
    )
    assert np.allclose(pinned.median("x"), [101.0, 102.0, 103.0, 104.0])


def test_scenario_refusals() -> None:
    """Criterion 10: unknown keys, wrong lengths, and strings that are not ``"hold"``."""
    data = small_world()
    with pytest.raises(ValueError, match=r"scenario= names \['pmi'\]"):
        forecast(data, 4, graph=D1_GRAPH, scenario={"pmi": "hold"}, draws=10, warmup=10)
    with pytest.raises(ValueError, match="expected"):
        forecast(data, 4, graph=D1_GRAPH, scenario={"x": np.ones(3)}, draws=10, warmup=10)
    with pytest.raises(ValueError, match="only string a scenario takes"):
        forecast(data, 4, graph=D1_GRAPH, scenario={"x": "freeze"}, draws=10, warmup=10)


def test_a_scenario_on_an_endogenous_node_is_refused_as_an_intervention() -> None:
    """Criterion 11: pinning a driven node is an intervention, and the message says so."""
    data = small_world()
    with pytest.raises(ValueError, match="is an intervention") as raised:
        forecast(data, 4, graph=D1_GRAPH, scenario={"y": "hold"}, draws=10, warmup=10)
    message = str(raised.value)
    assert "x[t-1]" in message
    assert "identified" in message


# -- criterion 12: the schedule and the provenance ----------------------------


def test_the_schedule_reads_the_waves(d1_data: TimeSeriesData) -> None:
    """Criterion 12: the plan is [x] then [y], and it is inspectable without fitting."""
    plan = schedule(d1_data, D1_GRAPH)
    assert [stage.nodes for stage in plan.mcmc_stages] == [("x",), ("y",)]
    assert plan.stages[1].parent_edges == {"y": (("x", 1),)}
    assert "x" in plan.summary() and "mcmc" in plan.summary()

    # A scenario driver resolves first instead, in its own stage.
    with_pin = schedule(d1_data, D1_GRAPH, {"x": ScenarioMechanism()})
    assert with_pin.stages[0].nodes == ("x",)
    assert with_pin.stages[0].strategy.value == "resolved"
    assert [stage.nodes for stage in with_pin.mcmc_stages] == [("y",)]


def test_the_roles_come_from_the_mechanisms_and_the_graph(d1_data: TimeSeriesData) -> None:
    """The rule the schedule is built on, checked on its own."""
    from ergodic.forecast import assign_mechanisms

    assigned = assign_mechanisms(d1_data)
    plain = node_roles(d1_data, D1_GRAPH, assigned)
    assert plain == {"x": NodeRole.MODEL, "y": NodeRole.MODEL}

    pinned = node_roles(d1_data, D1_GRAPH, assigned, scenario={"x": "hold"})
    assert pinned["x"] is NodeRole.SCENARIO

    # Hand it a mechanism as well and it stays a model whose horizon is pinned.
    both = node_roles(
        d1_data,
        D1_GRAPH,
        assigned,
        given={"x": StructuralMechanism()},
        scenario={"x": "hold"},
    )
    assert both["x"] is NodeRole.MODEL


def test_auto_picks_joint_on_a_small_graph_and_staged_on_a_chain(
    d1_data: TimeSeriesData,
) -> None:
    """Criterion 12: the threshold is arithmetic on the schedule, recorded on the result."""
    small = forecast_or_skip(d1_data, 2, graph=D1_GRAPH, draws=60, warmup=60, seed=2)
    assert small.strategy is not None
    assert small.strategy.strategy == "joint"
    assert small.strategy.requested == "auto"
    assert small.strategy.n_mcmc_nodes == 2
    assert small.strategy.auto_threshold == AUTO_JOINT_MAX_NODES

    rng = np.random.default_rng(77)
    names = [f"n{i}" for i in range(6)]
    columns: dict[str, Array] = {}
    previous: Array | None = None
    for name in names:
        base = 100.0 + np.cumsum(rng.normal(0.1, 0.8, 30))
        if previous is not None:
            base = base + 0.3 * np.concatenate([[previous[0]], previous[:-1]])
        columns[name] = base
        previous = base
    chain = monthly(columns, 30)
    graph = window_graph([(names[i], names[i + 1], 1) for i in range(5)])
    plan = schedule(chain, graph)
    assert plan.num_mcmc_stages == 6
    long = forecast_or_skip(chain, 2, graph=graph, draws=60, warmup=60, seed=2)
    assert long.strategy is not None
    assert long.strategy.strategy == "staged"
    assert long.strategy.n_mcmc_nodes == 6


def test_the_result_carries_the_graph_and_the_schedule(staged_d1: ForecastResult) -> None:
    """Criterion 12: provenance enough to say how the numbers were made."""
    assert staged_d1.graph is D1_GRAPH
    assert staged_d1.schedule is not None
    assert staged_d1.schedule.num_mcmc_stages == 2
    assert staged_d1.strategy is not None
    assert staged_d1.strategy.strategy == "staged"
    assert staged_d1.strategy.requested == "staged"
    assert staged_d1.fits is not None
    assert set(staged_d1.fits) == {"x", "y"}
    assert staged_d1.mechanisms is not None


# -- criterion 13: discovery straight into a forecast -------------------------


def var_world(seed: int, n: int) -> dict[str, Array]:
    """Three series, one planted cross-link: ``x`` drives ``y`` one step later."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    x[0], y[0], z[0] = 100.0, 50.0, 20.0
    for t in range(1, n):
        x[t] = 15.0 + 0.85 * x[t - 1] + rng.normal(0.0, 2.0)
        y[t] = 8.0 + 0.55 * y[t - 1] + 0.35 * x[t - 1] + rng.normal(0.0, 1.5)
        z[t] = 3.0 + 0.85 * z[t - 1] + rng.normal(0.0, 1.0)
    return {"x": x, "y": y, "z": z}


def test_a_discovered_window_graph_forecasts_end_to_end() -> None:
    """Criterion 13: PCMCI's graph goes into forecast() unchanged and the link keeps its sign."""
    n = 160
    columns = var_world(WORLD_SEED, n)
    data = timeseries({"t": np.arange(n, dtype=float), **columns}, time="t")
    found = discover(data, method="pcmci", max_lag=2)
    edges = {edge.glyph() for edge in found.graph.edges}
    assert "x[t-1] --> y[t]" in edges, f"the planted link is not in {sorted(edges)}"

    # The zero-translation promise: the same object the discovery returned.
    assert validate_window_graph(found.graph) is found.graph
    plan = schedule(data, found.graph)
    assert plan.num_mcmc_stages == 3

    with pytest.warns(UserWarning, match="minimum bulk ESS"):
        result = forecast_or_skip(
            data, 6, graph=found.graph, fit="staged", draws=250, warmup=250, seed=13
        )
    assert result.strategy is not None
    assert result.strategy.strategy == "staged"
    assert set(result.nodes) == {"x", "y", "z"}
    planted = posterior_mean(result, "y", "y_beta_x")
    assert planted > 0.0, f"the planted link came back at {planted:.4f}, wrong sign"


# -- criterion 14: staging composed with reconciliation -----------------------


def test_staged_fitting_composes_with_bottom_up_reconciliation() -> None:
    """Criterion 14: a driver, two leaves, one aggregate, and the levels still agree."""
    n, train, horizon = 72, 60, 6
    tree = Hierarchy.tree({"total": ["leaf_a", "leaf_b"]})
    driver, first = d1_world(2201, n)
    _, second = d1_world(2202, n)
    columns = {
        "pmi": driver,
        "leaf_a": first,
        "leaf_b": 0.6 * second + 0.5 * driver,
    }
    stamps = np.array(
        [np.datetime64("2016-01-01") + np.timedelta64(31 * i, "D") for i in range(train)]
    )
    data = hts(
        {"month": stamps, **{k: v[:train] for k, v in columns.items()}},
        time="month",
        hierarchy=tree,
    )
    graph = window_graph([("pmi", "leaf_a", 1), ("pmi", "leaf_b", 1)])

    plan = schedule(data, graph)
    assert [stage.nodes for stage in plan.mcmc_stages] == [("pmi",), ("leaf_a",), ("leaf_b",)]
    assert plan.stages[-1].nodes == ("total",)

    result = forecast_or_skip(
        data,
        horizon,
        graph=graph,
        fit="staged",
        reconcile="bottom_up",
        draws=250,
        warmup=250,
        seed=17,
    )
    assert result.coherence_error(tree) <= 1e-8
    assert result.coherent is True
    assert result.reconciliation == "bottom_up"
    # The schedule survives reconciliation, which rebuilds the result.
    assert result.schedule is not None
    assert result.schedule.num_mcmc_stages == 3
    assert result.strategy is not None
    assert result.strategy.strategy == "staged"
    # The driver is not part of the tree and rides through untouched.
    assert "pmi" in result.nodes
    per_draw = result.draws("total") - (result.draws("leaf_a") + result.draws("leaf_b"))
    assert float(np.max(np.abs(per_draw))) == pytest.approx(0.0, abs=1e-9)


# -- the pieces, on their own -------------------------------------------------


def test_a_hierarchy_parent_with_its_own_column_is_a_joint_fit() -> None:
    """A modeled parent is a staging limit, not a graph limit, and the message says which.

    Staged fitting adds the aggregates up last, so a parent that carries its own
    model has no wave it could run in. A joint fit has no waves at all, so it
    fits the parent and its children together and the reconciliation menu applies
    to the result afterwards exactly as it does on the independent path. The
    refusal names that way out, and this checks the way out actually works rather
    than only that the sentence exists.
    """
    n, train, horizon = 44, 36, 3
    tree = Hierarchy.tree({"total": ["leaf_a", "leaf_b"]})
    driver, first = d1_world(3301, n)
    _, second = d1_world(3302, n)
    stamps = np.array(
        [np.datetime64("2016-01-01") + np.timedelta64(31 * i, "D") for i in range(train)]
    )
    columns: dict[str, Any] = {
        "month": stamps,
        "pmi": driver[:train],
        "leaf_a": first[:train],
        "leaf_b": (0.7 * second)[:train],
    }
    columns["total"] = columns["leaf_a"] + columns["leaf_b"]
    data = hts(columns, time="month", hierarchy=tree)
    graph = window_graph([("pmi", "leaf_a", 1)])

    for fit in ("staged", "joint"):
        with pytest.raises(ValueError, match="hierarchy parent with role model") as raised:
            schedule(data, graph, fit="staged")
        assert "fit='joint'" in str(raised.value)
        if fit == "staged":
            with pytest.raises(ValueError, match="hierarchy parent with role model"):
                forecast(data, horizon, graph=graph, fit="staged", draws=10, warmup=10)

    # The advice is real: the joint plan carries the parent as a model node.
    plan = schedule(data, graph, fit="joint")
    assert set(plan.all_nodes) == {"pmi", "leaf_a", "leaf_b", "total"}
    assert plan.num_mcmc_stages == 4

    result = forecast_or_skip(
        data, horizon, graph=graph, fit="joint", draws=150, warmup=150, seed=1
    )
    assert result.strategy is not None
    assert result.strategy.strategy == "joint"
    assert result.fits is not None
    assert "total" in result.fits, "the parent was aggregated rather than modeled"
    assert result.coherence_error(tree) <= 1e-8
    assert result.coherent is True

    # And the parent really was fitted on its own: its unreconciled draws
    # disagree with its children's sum, which is the disagreement reconciliation
    # exists to settle. An aggregated parent could not disagree at all.
    unreconciled = forecast_or_skip(
        data,
        horizon,
        graph=graph,
        fit="joint",
        reconcile="none",
        draws=150,
        warmup=150,
        seed=1,
    )
    assert unreconciled.coherent is False
    assert unreconciled.coherence_error(tree) > 1e-6


def test_both_strategies_are_bitwise_deterministic() -> None:
    """The same call twice gives the same draws, handoff and all.

    S2 pinned this for the independent path; a staged run adds a whole second
    mechanism for it to go wrong in (a parent's draws are transformed, aligned,
    written into a trace copy, and replayed), and a joint run adds a bigger model.
    Both are exact, and the medians below are pinned loosely enough that another
    machine's floating point does not fail the suite.
    """
    x, y = d1_world(WORLD_SEED, 52)
    data = monthly({"x": x, "y": y}, 48)
    options: dict[str, Any] = {
        "graph": D1_GRAPH,
        "draws": 150,
        "warmup": 150,
        "chains": 1,
        "seed": 9,
    }
    first = forecast_or_skip(data, 4, fit="staged", **options)
    second = forecast_or_skip(data, 4, fit="staged", **options)
    for node in first.nodes:
        assert np.array_equal(first.draws(node), second.draws(node))
    assert first.median("y") == pytest.approx(
        [104.05099, 106.755923, 111.066536, 111.7587], rel=0.005
    )

    third = forecast_or_skip(data, 4, fit="joint", **options)
    fourth = forecast_or_skip(data, 4, fit="joint", **options)
    for node in third.nodes:
        assert np.array_equal(third.draws(node), fourth.draws(node))
    assert third.median("y") == pytest.approx(
        [104.246123, 107.567608, 111.453376, 112.218051], rel=0.005
    )


def test_lag_align_front_pads_with_the_first_value() -> None:
    """The documented padding rule, and the arithmetic of the shift."""
    history = np.array([10.0, 11.0, 12.0, 13.0])
    future = np.array([14.0, 15.0])
    assert np.array_equal(lag_align(history, None, 0, 0), history)
    assert np.array_equal(lag_align(history, None, 1, 0), [10.0, 10.0, 11.0, 12.0])
    assert np.array_equal(lag_align(history, None, 2, 0), [10.0, 10.0, 10.0, 11.0])
    assert np.array_equal(lag_align(history, future, 0, 2), [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    assert np.array_equal(lag_align(history, future, 1, 2), [10.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    # A lag at or past the horizon never reads the future at all.
    assert np.array_equal(lag_align(history, None, 2, 2), [10.0, 10.0, 10.0, 11.0, 12.0, 13.0])
    with pytest.raises(ValueError, match="needs 1 step"):
        lag_align(history, None, 1, 2)
    with pytest.raises(ValueError, match="future step"):
        lag_align(history, np.array([14.0]), 0, 2)


def test_design_keys_keep_the_plain_name_for_the_shortest_lag() -> None:
    """The naming rule for a parent that drives at several lags."""
    assert design_keys([("pmi", 1)]) == {("pmi", 1): "pmi"}
    assert design_keys([("pmi", 2)]) == {("pmi", 2): "pmi"}
    assert design_keys([("pmi", 2), ("pmi", 1)]) == {
        ("pmi", 1): "pmi",
        ("pmi", 2): "pmi_lag2",
    }
    assert design_keys([("a", 1), ("b", 0)]) == {("a", 1): "a", ("b", 0): "b"}
    with pytest.raises(ValueError, match="would both be called"):
        design_keys([("pmi", 1), ("pmi", 2), ("pmi_lag2", 1)])


def test_align_draws_resamples_only_when_it_has_to() -> None:
    """Equal draw counts pass through untouched; unequal ones resample under a seed."""
    draws = np.arange(20, dtype=np.float64).reshape(5, 4)
    assert align_draws(draws, 5, seed=1) is draws
    resampled = align_draws(draws, 9, seed=1)
    assert resampled.shape == (9, 4)
    assert np.array_equal(resampled, align_draws(draws, 9, seed=1))
    assert not np.array_equal(resampled, align_draws(draws, 9, seed=2))
    for row in resampled:
        assert any(np.array_equal(row, original) for original in draws)


def test_injection_copies_rather_than_edits(staged_d1: ForecastResult) -> None:
    """A fitted posterior is a record: one node's handoff must not change another's replay."""
    assert staged_d1.fits is not None
    trace = staged_d1.fits["y"].trace
    before = set(_group_dataset(trace, "posterior").data_vars)
    draws = np.zeros((staged_d1.n_draws, D1_HOLDOUT))
    augmented = inject_futures(trace, {"x_fc": draws})
    assert "x_fc" in augmented
    assert set(_group_dataset(trace, "posterior").data_vars) == before
    with pytest.raises(ValueError, match="align the draw counts"):
        inject_futures(trace, {"x_fc": np.zeros((3, D1_HOLDOUT))})


def test_the_graph_has_to_name_real_columns_and_the_fit_has_to_read_it() -> None:
    """The facade's structural refusals, all before anything samples."""
    data = small_world()
    with pytest.raises(ValueError, match=r"names \['pmi'\]"):
        forecast(data, 4, graph=window_graph([("pmi", "y", 1)]), draws=10, warmup=10)
    with pytest.raises(ValueError, match="would ignore the graph"):
        forecast(data, 4, graph=D1_GRAPH, fit="independent", draws=10, warmup=10)
    with pytest.raises(ValueError, match="no graph says"):
        forecast(data, 4, fit="staged", draws=10, warmup=10)
