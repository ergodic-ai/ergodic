# SPDX-License-Identifier: Apache-2.0
"""Tests for window graphs and the training schedule.

The window graph is the one representation of predictive structure, so the
zero-translation promise is checked live: a real PCMCI run on a seeded VAR
world hands its graph straight to the validator, unchanged. The schedule is
checked on a hand graph with two waves, where the right answer can be read off
by eye, plus the invariant that makes staging correct at all.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ergodic import discover
from ergodic.data import Hierarchy, timeseries
from ergodic.discovery import TimeSeriesDiscoveryResult
from ergodic.forecast import (
    NodeRole,
    StageStrategy,
    TrainingSchedule,
    bases,
    max_lag,
    split_lag,
    validate_window_graph,
    window_graph,
)
from ergodic.graph import GraphKind, dag, mixed

# -- criterion 12: building a window graph ------------------------------------


def test_window_graph_builds_the_lagged_nodes_and_edges() -> None:
    graph = window_graph([("x", "y", 1), ("y", "y", 1), ("x", "y", 0)])

    assert graph.kind is GraphKind.DAG
    assert graph.nodes == frozenset({"x[t-1]", "x[t]", "y[t-1]", "y[t]"})
    assert {edge.glyph() for edge in graph.edges} == {
        "x[t-1] --> y[t]",
        "x[t] --> y[t]",
        "y[t-1] --> y[t]",
    }
    assert max_lag(graph) == 1
    assert bases(graph) == ("x", "y")
    for node in graph.nodes:
        base, lag = split_lag(node)
        assert node == (f"{base}[t]" if lag == 0 else f"{base}[t-{lag}]")


def test_window_graph_keeps_two_lags_of_one_pair_apart() -> None:
    """Two lags between the same variables are two node pairs, so both survive."""
    graph = window_graph([("pmi", "sales", 1), ("pmi", "sales", 2)])
    assert graph.has_edge("pmi[t-1]", "sales[t]")
    assert graph.has_edge("pmi[t-2]", "sales[t]")
    assert max_lag(graph) == 2
    assert bases(graph) == ("pmi", "sales")


def test_window_graph_refuses_a_lag_it_cannot_read() -> None:
    with pytest.raises(ValueError, match="lag must be >= 0"):
        window_graph([("x", "y", -1)])
    with pytest.raises(ValueError, match="self-loop"):
        window_graph([("x", "x", 0)])


def test_max_lag_of_a_purely_contemporaneous_graph_is_zero() -> None:
    graph = window_graph([("x", "y", 0)])
    assert max_lag(graph) == 0
    assert max_lag(dag()) == 0


# -- criterion 13: what the validator refuses ---------------------------------


def test_an_edge_pointing_backward_in_time_raises() -> None:
    with pytest.raises(ValueError, match="points backward in time"):
        validate_window_graph(dag(["y[t] -> x[t-1]"]))
    with pytest.raises(ValueError, match="points backward in time"):
        validate_window_graph(mixed(["y[t] <-> x[t-2]"]))


def test_a_node_name_that_is_not_a_window_node_raises() -> None:
    with pytest.raises(ValueError, match="not a lagged node name"):
        validate_window_graph(dag(["a -> b"]))
    with pytest.raises(ValueError, match="not a lagged node name"):
        validate_window_graph(dag(["x[t-1] -> y[t+1]"]))
    with pytest.raises(ValueError, match="not a lagged node name"):
        validate_window_graph(dag(["x[t-01] -> y[t]"]))


def test_a_contemporaneous_cycle_raises() -> None:
    cyclic = mixed(["x[t] -> y[t]", "y[t] -> z[t]", "z[t] -> x[t]"])
    with pytest.raises(ValueError, match="directed cycle"):
        validate_window_graph(cyclic)


def test_the_validator_accepts_what_temporal_discovery_can_produce() -> None:
    """An unoriented contemporaneous pair is fine; only time direction is enforced."""
    graph = mixed(["x[t-1] -> y[t]", "x[t] --- y[t]", "z[t-2] -> z[t]"])
    assert validate_window_graph(graph) is graph
    assert max_lag(graph) == 2
    assert bases(graph) == ("x", "y", "z")


# -- criterion 14: the discovery handoff, live --------------------------------


def test_pcmci_output_is_a_window_graph_the_forecaster_accepts() -> None:
    """A seeded VAR world with x -> y at lag 1: discovery to forecasting, no translation."""
    rng = np.random.default_rng(20260805)
    n = 200
    ex, ey = rng.normal(size=(2, n))
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + ex[t]
        y[t] = 0.8 * x[t - 1] + 0.5 * y[t - 1] + ey[t]
    data = timeseries({"t": np.arange(n), "x": x, "y": y}, time="t")

    result = discover(data, method="pcmci", max_lag=2)
    assert isinstance(result, TimeSeriesDiscoveryResult)
    graph = result.graph

    assert validate_window_graph(graph) is graph, "the graph has to be accepted unchanged"
    assert max_lag(graph) == 2
    assert bases(graph) == ("x", "y")

    planted = graph.get_edge("x[t-1]", "y[t]")
    assert planted is not None, sorted(e.glyph() for e in graph.edges)
    assert planted.directed_pair() == ("x[t-1]", "y[t]")
    assert ("x", "y", 1) in result.links

    # And the schedule reads that graph directly: y depends on x, so x is fitted first.
    schedule = TrainingSchedule.from_graph(graph, None, {"x": NodeRole.MODEL, "y": NodeRole.MODEL})
    stage_x, stage_y = schedule.stage_of("x"), schedule.stage_of("y")
    assert stage_x is not None and stage_y is not None
    assert stage_x.index < stage_y.index
    assert stage_y.parent_edges["y"] == (("x", 1),)


# -- criteria 15 and 16: the training schedule --------------------------------

# Two scenario bases, three first-wave models reading only scenarios, two
# second-wave models reading the first wave, and one aggregate over three of
# the first-wave nodes. Self-memory on north is deliberate: it must not count.
_HAND_EDGES = [
    ("pmi", "north", 1),
    ("north", "north", 1),
    ("pmi", "south", 2),
    ("fx", "east", 1),
    ("north", "services", 1),
    ("south", "retail", 0),
    ("east", "retail", 1),
]
_HAND_ROLES = {
    "pmi": NodeRole.SCENARIO,
    "fx": NodeRole.SCENARIO,
    "north": NodeRole.MODEL,
    "south": NodeRole.MODEL,
    "east": NodeRole.MODEL,
    "services": NodeRole.MODEL,
    "retail": NodeRole.MODEL,
    "total": NodeRole.AGGREGATE,
}
_HAND_HIERARCHY = Hierarchy.tree({"total": ["north", "south", "east"]})


def _hand_schedule(max_batch_size: int = 1) -> TrainingSchedule:
    return TrainingSchedule.from_graph(
        window_graph(_HAND_EDGES),
        _HAND_HIERARCHY,
        _HAND_ROLES,
        max_batch_size=max_batch_size,
    )


def test_the_schedule_resolves_then_fits_wave_by_wave_then_aggregates() -> None:
    """Criterion 15: one resolved stage, five MCMC stages in wave order, one aggregate."""
    schedule = _hand_schedule()

    assert [stage.strategy for stage in schedule.stages] == [
        StageStrategy.RESOLVED,
        *[StageStrategy.MCMC] * 5,
        StageStrategy.AGGREGATE,
    ]
    assert schedule.num_mcmc_stages == 5
    assert [stage.nodes for stage in schedule.stages] == [
        ("fx", "pmi"),
        ("east",),
        ("north",),
        ("south",),
        ("retail",),
        ("services",),
        ("total",),
    ]
    assert schedule.stages[-1].hierarchy_children == {"total": ("east", "north", "south")}
    assert schedule.all_nodes[:2] == ("fx", "pmi")
    assert len(schedule) == 7


def test_parent_edges_name_the_posteriors_a_stage_needs() -> None:
    schedule = _hand_schedule()
    edges = {stage.nodes[0]: stage.parent_edges for stage in schedule.mcmc_stages}

    assert edges["north"] == {"north": (("pmi", 1),)}  # self-memory is not a parent
    assert edges["south"] == {"south": (("pmi", 2),)}
    assert edges["east"] == {"east": (("fx", 1),)}
    assert edges["services"] == {"services": (("north", 1),)}
    assert edges["retail"] == {"retail": (("east", 1), ("south", 0))}


def _assert_invariant(schedule: TrainingSchedule, roles: dict[str, NodeRole]) -> None:
    """Every parent a stage names is resolved by an earlier stage, and none is an aggregate."""
    index_of = {node: stage.index for stage in schedule.stages for node in stage.nodes}
    for stage in schedule.mcmc_stages:
        for node, parents in stage.parent_edges.items():
            for base, _lag in parents:
                assert index_of[base] < index_of[node], (node, base)
                assert roles[base] is not NodeRole.AGGREGATE, (node, base)
                assert base not in stage.nodes  # no edges inside a wave


def test_every_modeled_parent_sits_in_a_strictly_earlier_stage() -> None:
    """The invariant that makes staged fitting correct, on the hand graph."""
    _assert_invariant(_hand_schedule(), _HAND_ROLES)
    _assert_invariant(_hand_schedule(max_batch_size=999), _HAND_ROLES)


def test_a_model_that_reads_an_aggregate_is_refused() -> None:
    """The aggregate resolves last, so its posterior would not exist when the reader runs."""
    graph = window_graph([("total", "share", 1)])
    roles = {
        "north": NodeRole.MODEL,
        "south": NodeRole.MODEL,
        "total": NodeRole.AGGREGATE,
        "share": NodeRole.MODEL,
    }
    hierarchy = Hierarchy.tree({"total": ["north", "south"]})
    with pytest.raises(ValueError, match="is an aggregate of its hierarchy children"):
        TrainingSchedule.from_graph(graph, hierarchy, roles)


def _fuzz_world(
    rng: np.random.Generator,
) -> tuple[list[tuple[str, str, int]], dict[str, NodeRole], Hierarchy | None]:
    """A random window graph with random roles, sometimes with an aggregate in it."""
    bases = ["a", "b", "c", "d", "e"]
    roles: dict[str, NodeRole] = {}
    hierarchy: Hierarchy | None = None
    pool = list(bases)
    if bool(rng.random() < 0.5):
        kids = [str(k) for k in rng.choice(bases, size=2, replace=False)]
        hierarchy = Hierarchy.tree({"agg": kids})
        roles["agg"] = NodeRole.AGGREGATE
        for kid in kids:
            roles[kid] = NodeRole.MODEL
        pool.append("agg")
    for base in bases:
        if base not in roles:
            roles[base] = NodeRole.MODEL if rng.random() < 0.7 else NodeRole.SCENARIO

    order = [str(node) for node in rng.permutation(pool)]
    edges: list[tuple[str, str, int]] = []
    used: set[frozenset[str]] = set()
    for _ in range(int(rng.integers(2, 9))):
        lag = int(rng.integers(0, 3))
        if lag == 0:
            left, right = sorted(int(i) for i in rng.choice(len(order), size=2, replace=False))
            source, target = order[left], order[right]
        else:
            source = str(rng.choice(pool))
            target = str(rng.choice(pool))
            if source == target:
                continue
        pair = frozenset({f"{source}[{lag}]", f"{target}[0]"})
        if len(pair) == 1 or pair in used:
            continue
        used.add(pair)
        edges.append((source, target, lag))
    return edges, roles, hierarchy


def test_the_schedule_invariant_holds_or_the_schedule_refuses() -> None:
    """Fuzz over random graphs and roles: either the invariant holds, or we said why."""
    rng = np.random.default_rng(20260806)
    accepted = refused_aggregate = refused_loop = 0
    for _ in range(200):
        edges, roles, hierarchy = _fuzz_world(rng)
        if not edges:
            continue
        graph = window_graph(edges)
        drives_aggregate = any(
            roles.get(source) is NodeRole.AGGREGATE
            and roles.get(target) is NodeRole.MODEL
            and source != target
            for source, target, _lag in edges
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                schedule = TrainingSchedule.from_graph(graph, hierarchy, roles)
        except ValueError as error:
            message = str(error)
            if "is an aggregate of its hierarchy children" in message:
                refused_aggregate += 1
                assert drives_aggregate, edges
            else:
                assert "drive each other in a loop" in message, message
                refused_loop += 1
            continue
        assert not drives_aggregate, ("an aggregate driver slipped through", edges)
        accepted += 1
        _assert_invariant(schedule, roles)
    assert accepted > 50, accepted
    assert refused_aggregate > 5, refused_aggregate
    assert refused_loop > 5, refused_loop


def test_a_big_batch_collapses_each_wave_into_one_stage() -> None:
    schedule = _hand_schedule(max_batch_size=999)
    assert [stage.nodes for stage in schedule.stages] == [
        ("fx", "pmi"),
        ("east", "north", "south"),
        ("retail", "services"),
        ("total",),
    ]
    assert schedule.num_mcmc_stages == 2
    assert schedule.mcmc_stages[1].parent_edges == {
        "retail": (("east", 1), ("south", 0)),
        "services": (("north", 1),),
    }


def test_a_hierarchy_parent_that_wants_its_own_model_raises() -> None:
    """Criterion 16: staged fitting is bottom-up, so a modeled parent is refused."""
    roles = dict(_HAND_ROLES)
    roles["total"] = NodeRole.MODEL
    with pytest.raises(ValueError, match="hierarchy parent with role model"):
        TrainingSchedule.from_graph(window_graph(_HAND_EDGES), _HAND_HIERARCHY, roles)

    roles["total"] = NodeRole.SCENARIO
    with pytest.raises(ValueError, match="hierarchy parent with role scenario"):
        TrainingSchedule.from_graph(window_graph(_HAND_EDGES), _HAND_HIERARCHY, roles)


def test_an_aggregate_with_nothing_to_add_up_raises() -> None:
    with pytest.raises(ValueError, match="no hierarchy children"):
        TrainingSchedule.from_graph(None, None, {"total": NodeRole.AGGREGATE})


def test_a_schedule_without_a_graph_puts_every_model_in_one_wave() -> None:
    schedule = TrainingSchedule.from_graph(
        None, None, {"a": NodeRole.MODEL, "b": NodeRole.MODEL}, max_batch_size=999
    )
    assert len(schedule.stages) == 1
    assert schedule.stages[0].nodes == ("a", "b")
    assert schedule.stages[0].parent_edges == {}


def test_the_schedule_reads_roles_written_as_plain_strings() -> None:
    schedule = TrainingSchedule.from_graph(None, None, {"a": NodeRole("model")})
    assert schedule.stages[0].strategy is StageStrategy.MCMC
    with pytest.raises(ValueError, match="is not a valid NodeRole"):
        TrainingSchedule.from_graph(None, None, {"a": NodeRole("forecast")})


def test_stage_of_and_summary_read_the_plan() -> None:
    schedule = _hand_schedule()
    stage = schedule.stage_of("retail")
    assert stage is not None
    assert stage.strategy is StageStrategy.MCMC
    assert schedule.stage_of("nobody") is None

    text = schedule.summary()
    assert "7 stages, 5 by mcmc" in text
    assert "[resolved ]: fx, pmi" in text
    assert "total = sum(east + north + south)" in text


def test_nodes_that_drive_each_other_are_refused_with_the_real_reason() -> None:
    """Feedback across lags is an ordinary process, and staging still cannot serve it."""
    feedback = window_graph([("x", "y", 1), ("y", "x", 1)])
    roles = {"x": NodeRole.MODEL, "y": NodeRole.MODEL}
    with pytest.raises(ValueError) as caught:
        TrainingSchedule.from_graph(feedback, None, roles)
    message = str(caught.value)
    assert "drive each other in a loop" in message
    assert "feedback across lags" in message.lower()
    assert "joint multivariate mechanism" in message
    # The graph itself is perfectly legal: there is no cycle in time.
    assert validate_window_graph(feedback) is feedback

    contemporaneous = mixed(["a[t] -> b[t]", "b[t] -> c[t]", "c[t] -> a[t]"])
    with pytest.raises(ValueError, match="drive each other in a loop"):
        TrainingSchedule.from_graph(
            contemporaneous,
            None,
            {"a": NodeRole.MODEL, "b": NodeRole.MODEL, "c": NodeRole.MODEL},
        )


def test_a_cross_slice_edge_counts_whatever_marks_it_carries() -> None:
    """Time orients a lagged edge, so an unresolved causal mark is not a blocker."""
    for glyph in ("x[t-1] -> y[t]", "x[t-1] o-> y[t]", "x[t-1] --- y[t]", "x[t-1] o-o y[t]"):
        schedule = TrainingSchedule.from_graph(
            mixed([glyph]), None, {"x": NodeRole.MODEL, "y": NodeRole.MODEL}
        )
        stage = schedule.stage_of("y")
        assert stage is not None
        assert stage.parent_edges == {"y": (("x", 1),)}, glyph
        assert schedule.n_unoriented_dropped == 0


def test_a_same_slice_edge_counts_only_when_it_is_oriented() -> None:
    """Inside one slice nothing says which way the arrow runs, so the pair is dropped."""
    directed = TrainingSchedule.from_graph(
        mixed(["x[t] -> y[t]"]), None, {"x": NodeRole.MODEL, "y": NodeRole.MODEL}
    )
    stage = directed.stage_of("y")
    assert stage is not None
    assert stage.parent_edges == {"y": (("x", 0),)}
    assert directed.unoriented_dropped == ()

    for glyph in ("x[t] --- y[t]", "x[t] o-o y[t]", "x[t] o-> y[t]"):
        with pytest.warns(UserWarning, match="unoriented"):
            schedule = TrainingSchedule.from_graph(
                mixed([glyph]), None, {"x": NodeRole.MODEL, "y": NodeRole.MODEL}
            )
        assert schedule.unoriented_dropped == (("x", "y"),), glyph
        assert schedule.n_unoriented_dropped == 1
        assert all(stage.parent_edges == {} for stage in schedule.mcmc_stages), glyph


def test_the_dropped_pairs_are_counted_and_named() -> None:
    graph = mixed(["x[t] --- y[t]", "x[t] o-> z[t]", "w[t-1] -> z[t]"])
    roles = dict.fromkeys(("x", "y", "z", "w"), NodeRole.MODEL)
    with pytest.warns(UserWarning, match="x --- y, x --- z"):
        schedule = TrainingSchedule.from_graph(graph, None, roles)
    assert schedule.unoriented_dropped == (("x", "y"), ("x", "z"))
    assert schedule.n_unoriented_dropped == 2
    stage = schedule.stage_of("z")
    assert stage is not None
    assert stage.parent_edges == {"z": (("w", 1),)}


def test_the_graph_cannot_name_a_variable_the_roles_do_not() -> None:
    with pytest.raises(ValueError, match=r"no role: \['pmi'\]"):
        TrainingSchedule.from_graph(
            window_graph([("pmi", "sales", 1)]), None, {"sales": NodeRole.MODEL}
        )


def test_nested_aggregates_are_added_up_children_first() -> None:
    """A three-level tree: r1 and r2 have to exist before all can sum them."""
    hierarchy = Hierarchy.tree({"all": ["r1", "r2"], "r1": ["a", "b"], "r2": ["c", "d"]})
    roles: dict[str, NodeRole] = dict.fromkeys(("a", "b", "c", "d"), NodeRole.MODEL)
    roles.update(dict.fromkeys(("r1", "r2", "all"), NodeRole.AGGREGATE))

    schedule = TrainingSchedule.from_graph(None, hierarchy, roles, max_batch_size=999)
    final = schedule.stages[-1]
    assert final.strategy is StageStrategy.AGGREGATE
    assert final.nodes == ("r1", "r2", "all"), "alphabetical order would hide the dependency"
    assert final.hierarchy_children["all"] == ("r1", "r2")
    position = {node: i for i, node in enumerate(final.nodes)}
    for parent, kids in final.hierarchy_children.items():
        for kid in kids:
            if kid in position:
                assert position[kid] < position[parent]
