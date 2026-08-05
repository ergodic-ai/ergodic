# SPDX-License-Identifier: Apache-2.0
"""Tests for window graphs and the training schedule.

The window graph is the one representation of predictive structure, so the
zero-translation promise is checked live: a real PCMCI run on a seeded VAR
world hands its graph straight to the validator, unchanged. The schedule is
checked on a hand graph with two waves, where the right answer can be read off
by eye, plus the invariant that makes staging correct at all.
"""

from __future__ import annotations

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


def test_every_modeled_parent_sits_in_a_strictly_earlier_stage() -> None:
    """The invariant that makes staged fitting correct, checked node by node."""
    schedule = _hand_schedule()
    index_of = {node: stage.index for stage in schedule.stages for node in stage.nodes}
    for stage in schedule.mcmc_stages:
        for node, parents in stage.parent_edges.items():
            for base, _lag in parents:
                assert index_of[base] < index_of[node], (node, base)
        # No edges inside a wave: every parent of every node in the stage is outside it.
        for node in stage.nodes:
            for base, _lag in stage.parent_edges.get(node, ()):
                assert base not in stage.nodes


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


def test_a_cycle_among_modeled_nodes_raises() -> None:
    """A cycle cannot come from a valid window graph, so the guard is belt and braces."""
    cyclic = mixed(["a[t] -> b[t]", "b[t] -> c[t]", "c[t] -> a[t]"])
    roles = {"a": NodeRole.MODEL, "b": NodeRole.MODEL, "c": NodeRole.MODEL}
    with pytest.raises(ValueError, match="depend on each other in a cycle"):
        TrainingSchedule.from_graph(cyclic, None, roles)
