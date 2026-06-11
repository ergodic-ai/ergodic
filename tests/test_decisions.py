# SPDX-License-Identifier: Apache-2.0
"""Tests for decision points: branch assignment, timing gates, positivity, recovery."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, estimate_effect, event_log
from ergodic.discovery import discover
from ergodic.process._decisions import (
    DecisionPoint,
    DecisionTable,
    decision_points,
    decision_table,
)
from ergodic.process._tree import TAU, ProcessTree, loop, seq, xor

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import EventLog

# -- known-truth worlds --------------------------------------------------------


def _routing_tree() -> ProcessTree:
    return seq("open", "triage", xor("expedite", "standard"), "resolve")


@cache
def _routing_world(
    slope: float = 3.0, n: int = 2000, seed: int = 7
) -> tuple[EventLog, npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Confounded routing: complexity drives both the branch and the remaining time.

    P(expedite) = sigmoid(slope * (complexity - 0.5)) with complexity uniform
    on (0, 1); remaining time after the decision is 10 + 8 * complexity -
    5 * expedite + N(0, 1) seconds, realized as the gap to resolve. The
    planted branch effect is -5.0. Arrivals spread over [0, 6000] so the
    workload features vary, and the open-triage-decision gaps are jittered so
    elapsed time at the landmark varies too.
    """
    rng = np.random.default_rng(seed)
    complexity = rng.uniform(0.0, 1.0, size=n)
    expedite = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-slope * (complexity - 0.5)))
    start = rng.uniform(0.0, 6000.0, size=n)
    triage_gap = rng.uniform(0.5, 1.5, size=n)
    decide_gap = rng.uniform(0.5, 2.0, size=n)
    remaining = 10.0 + 8.0 * complexity - 5.0 * expedite + rng.normal(0.0, 1.0, size=n)
    rows = []
    for i in range(n):
        case = f"c{i:04d}"
        t0 = start[i]
        decision = t0 + triage_gap[i] + decide_gap[i]
        rows.append((case, "open", t0, complexity[i]))
        rows.append((case, "triage", t0 + triage_gap[i], complexity[i]))
        rows.append((case, "expedite" if expedite[i] else "standard", decision, complexity[i]))
        rows.append((case, "resolve", decision + remaining[i], complexity[i]))
    df = pd.DataFrame(rows, columns=["case", "activity", "time", "complexity"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    return log, expedite, remaining


def _counts_tree() -> ProcessTree:
    """An outer xor routing past the inner split, with a loop above the inner xor."""
    return seq("start", xor(loop(xor("a", "b"), "retry"), "skip"), "end")


def _counts_log() -> EventLog:
    events = [
        ("k1", "start", 0.0),  # inner branch a
        ("k1", "a", 1.0),
        ("k1", "end", 2.0),
        ("k2", "start", 10.0),  # inner branch b
        ("k2", "b", 11.0),
        ("k2", "end", 12.0),
        ("k3", "start", 20.0),  # the loop replays the inner split: a then b, ambiguous
        ("k3", "a", 21.0),
        ("k3", "retry", 22.0),
        ("k3", "b", 23.0),
        ("k3", "end", 24.0),
        ("k4", "start", 30.0),  # routed away by the outer xor: skips the inner split
        ("k4", "skip", 31.0),
        ("k4", "end", 32.0),
        ("k5", "start", 40.0),  # inner branch b
        ("k5", "b", 41.0),
        ("k5", "end", 42.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    return event_log(df, case="case", activity="activity", timestamp="time")


def _gated_tree() -> ProcessTree:
    return seq("open", loop("prep", TAU), xor("fast", "slow"), "close")


def _gated_log(extra: dict[str, list[float]] | None = None) -> EventLog:
    """Three cases with hand-computable landmark features; C overlaps nobody."""
    events = [
        ("A", "open", 0.0),
        ("A", "prep", 2.0),
        ("A", "fast", 5.0),  # landmark 5: 2 events before, elapsed 5
        ("A", "close", 20.0),
        ("B", "open", 1.0),
        ("B", "prep", 2.0),
        ("B", "prep", 3.0),
        ("B", "slow", 4.0),  # landmark 4: 3 events before, elapsed 3
        ("B", "close", 30.0),
        ("C", "open", 30.0),
        ("C", "prep", 31.0),
        ("C", "fast", 33.0),  # landmark 33: 2 events before, elapsed 3
        ("C", "close", 40.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    df["priority"] = [4.0] * 4 + [7.0] * 5 + [1.0] * 4
    counts = {"A": 4, "B": 5, "C": 4}
    for name, per_case in (extra or {}).items():
        df[name] = [
            value for case, value in zip("ABC", per_case, strict=True) for _ in range(counts[case])
        ]
    return event_log(df, case="case", activity="activity", timestamp="time")


# -- 1. the anchor world ---------------------------------------------------------


def test_anchor_table_encodes_the_split() -> None:
    log, expedite, remaining = _routing_world()
    dt = decision_table(log, _routing_tree())
    assert isinstance(dt, DecisionTable)
    assert dt.treatment == "branch"
    assert dt.point.path == (2,)
    assert dt.point.branches == ("expedite", "standard")
    assert dt.n_cases == 2000
    assert dt.point.n_skipped == 0 and dt.n_dropped_ambiguous == 0
    # The pre-decision prefix is open then triage in every case, so the two
    # prefix counters are constant and dropped in silence.
    assert dt.covariates == (
        "complexity",
        "elapsed_at_landmark",
        "workload_at_landmark",
        "workload_at_start",
    )
    assert dt.data.names == (*dt.covariates, "branch", "remaining_time")
    assert dt.data.schema["branch"].is_binary
    branch = dt.data.column("branch")
    assert np.array_equal(branch == "expedite", expedite)
    # The landmark is the decision event itself, so remaining time from the
    # landmark to resolve is exactly the planted gap.
    assert np.allclose(dt.data.column("remaining_time"), remaining)


def test_anchor_naive_branch_gap_is_biased() -> None:
    log, _, _ = _routing_world()
    dt = decision_table(log, _routing_tree())
    y = dt.data.column("remaining_time")
    branch = dt.data.column("branch")
    naive = float(y[branch == "expedite"].mean() - y[branch == "standard"].mean())
    # Expedite goes to complex cases, which run long, masking the benefit.
    assert naive - (-5.0) > 1.0


def test_anchor_composed_pipeline_recovers_the_branch_effect() -> None:
    log, _, _ = _routing_world()
    dt = decision_table(
        log,
        _routing_tree(),
        covariates=["complexity", "elapsed_at_landmark", "workload_at_landmark"],
    )
    assert dt.covariates == ("complexity", "elapsed_at_landmark", "workload_at_landmark")
    result = discover(dt.data, knowledge=dt.knowledge, alpha=0.01)
    # The tier knowledge orients every edge, which puts the graph outside the
    # CPDAG class on purpose; re-facing it as a DAG is the documented next step.
    est = estimate_effect(
        result.graph.as_kind("dag"),
        dt.data,
        "branch",
        "remaining_time",
        strategy="dml",
        random_state=0,
        treated="expedite",
    )
    assert abs(est.value - (-5.0)) < 0.4


# -- 2. positivity -----------------------------------------------------------------


def test_positivity_flags_a_near_deterministic_split() -> None:
    log, _, _ = _routing_world(slope=20.0, n=800, seed=9)
    sharp = decision_table(log, _routing_tree()).positivity
    # Both branches are large (the routing is symmetric around 0.5), so the
    # flag comes from overlap alone: complexity nearly decides the branch.
    assert min(sharp.counts) >= 10 and sharp.min_share > 0.05
    assert sharp.overlap_share is not None and sharp.overlap_share < 0.8
    assert sharp.flagged


def test_positivity_passes_the_healthy_world() -> None:
    log, _, _ = _routing_world()
    healthy = decision_table(log, _routing_tree()).positivity
    assert healthy.branches == ("expedite", "standard")
    assert sum(healthy.counts) == 2000
    assert healthy.min_share > 0.4
    assert healthy.overlap_share is not None and healthy.overlap_share > 0.95
    assert not healthy.flagged


# -- 3. counts on a hand log --------------------------------------------------------


def test_counts_are_exact_with_outer_routing_and_a_loop_above() -> None:
    log = _counts_log()
    outer, inner = decision_points(log, _counts_tree())
    assert outer.path == (1,)
    assert outer.branches == ("a+b+retry", "skip")
    assert outer.alphabets == (("a", "b", "retry"), ("skip",))
    assert outer.counts == (4, 1)
    assert outer.shares == (0.8, 0.2)
    assert outer.n_cases == 5 and outer.n_skipped == 0 and outer.n_ambiguous == 0
    assert inner.path == (1, 0, 0)
    assert inner.branches == ("a", "b")
    assert inner.counts == (1, 2)  # k1 takes a; k2 and k5 take b
    assert inner.n_cases == 3
    assert inner.n_skipped == 1  # k4 went down the skip branch of the outer xor
    assert inner.n_ambiguous == 1  # k3 touched a and b through the loop

    dt = decision_table(log, _counts_tree(), point=(1, 0, 0))
    assert dt.n_cases == 3 and dt.n_dropped_ambiguous == 1
    assert np.array_equal(dt.data.column("branch"), np.array(["a", "b", "b"]))
    # Three cases that never overlap leave every derived covariate constant,
    # so they all drop and the propensity is the base rate.
    assert dt.covariates == ()
    assert dt.positivity.overlap_share == 1.0
    assert dt.positivity.flagged  # branch counts of 1 and 2 are far below 10


def test_point_selection() -> None:
    log = _counts_log()
    with pytest.raises(ValueError) as err:
        decision_table(log, _counts_tree())
    assert "(1,)" in str(err.value) and "(1, 0, 0)" in str(err.value)
    dt = decision_table(log, _counts_tree(), point=(1,))
    assert dt.point.branches == ("a+b+retry", "skip")
    assert dt.n_cases == 5  # k3 is unambiguous here: a, retry, b are one branch
    points = decision_points(log, _counts_tree())
    via_object = decision_table(log, _counts_tree(), point=points[1])
    assert via_object.point == points[1]
    with pytest.raises(ValueError, match="not an xor node"):
        decision_table(log, _counts_tree(), point=(0,))


# -- 4. silent branches ---------------------------------------------------------------


def test_silent_branch_is_reported_then_refused() -> None:
    tree = seq("a", xor(TAU, "b"), "c")
    events = [
        ("s1", "a", 0.0),
        ("s1", "b", 1.0),
        ("s1", "c", 2.0),
        ("s2", "a", 10.0),  # no event in the split: the silent branch
        ("s2", "c", 11.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    (point,) = decision_points(log, tree)
    # The split sits under a plain sequence, so it is always reached and a
    # case with no visible hit must have taken the silent branch.
    assert point.branches == ("TAU", "b")
    assert point.counts == (1, 1)
    assert point.n_cases == 2 and point.n_skipped == 0
    with pytest.raises(ValueError, match="silent branch"):
        decision_table(log, tree)


def test_silent_branch_under_an_xor_counts_as_skipped() -> None:
    tree = seq("a", xor(xor(TAU, "b"), "d"), "c")
    events = [
        ("u1", "a", 0.0),  # takes d: never reaches the inner split
        ("u1", "d", 1.0),
        ("u1", "c", 2.0),
        ("u2", "a", 10.0),
        ("u2", "b", 11.0),
        ("u2", "c", 12.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    _, inner = decision_points(log, tree)
    assert inner.path == (1, 0)
    assert inner.branches == ("TAU", "b")
    # u1 touched no inner branch, and the inner split sits under an xor, so
    # the log cannot tell tau-taken from never-reached: skipped, not TAU.
    assert inner.counts == (0, 1)
    assert inner.n_skipped == 1 and inner.n_ambiguous == 0


# -- 5. landmark gating -----------------------------------------------------------------


def test_landmark_features_are_exact_on_a_hand_log() -> None:
    dt = decision_table(_gated_log(), _gated_tree())
    assert dt.point.path == (2,)
    assert np.array_equal(dt.data.column("branch"), np.array(["fast", "slow", "fast"]))
    # Every distinct activity before the landmark is open or prep, so the
    # distinct-count feature is constant and drops in silence.
    assert dt.covariates == (
        "elapsed_at_landmark",
        "n_events_prefix",
        "priority",
        "workload_at_landmark",
        "workload_at_start",
    )
    assert np.allclose(dt.data.column("n_events_prefix"), [2.0, 3.0, 2.0])
    assert np.allclose(dt.data.column("elapsed_at_landmark"), [5.0, 3.0, 3.0])
    # Open cases at each landmark, the case itself excluded: at t=5 only B is
    # open, at t=4 only A, and at t=33 both A and B have closed.
    assert np.allclose(dt.data.column("workload_at_landmark"), [1.0, 1.0, 0.0])
    assert np.allclose(dt.data.column("workload_at_start"), [0.0, 1.0, 0.0])


def test_remaining_time_is_exact_on_a_hand_log() -> None:
    dt = decision_table(_gated_log(), _gated_tree())
    # Landmarks 5, 4, 33; last events 20, 30, 40.
    assert np.allclose(dt.data.column("remaining_time"), [15.0, 26.0, 7.0])
    assert dt.data.schema["remaining_time"].is_continuous


# -- 6. outcomes ----------------------------------------------------------------------


def test_outcome_variants() -> None:
    log = _gated_log()
    tree = _gated_tree()
    assert np.allclose(
        decision_table(log, tree, outcome="cycle_time").data.column("cycle_time"),
        [20.0, 29.0, 10.0],
    )
    assert np.allclose(
        decision_table(log, tree, outcome="n_events").data.column("n_events"),
        [4.0, 5.0, 4.0],
    )
    by_column = decision_table(log, tree, outcome="priority")
    assert np.allclose(by_column.data.column("priority"), [4.0, 7.0, 1.0])
    assert "priority" not in by_column.covariates
    with pytest.raises(ValueError, match="is not 'remaining_time'"):
        decision_table(log, tree, outcome="throughput")
    with pytest.raises(ValueError, match="treatment column"):
        decision_table(log, tree, outcome="branch")


def test_name_collisions_raise() -> None:
    tree = _gated_tree()
    with pytest.raises(ValueError, match="reserved derived name"):
        decision_table(_gated_log({"remaining_time": [1.0, 2.0, 3.0]}), tree)
    with pytest.raises(ValueError, match="collides with the treatment column"):
        decision_table(_gated_log({"branch": [1.0, 2.0, 3.0]}), tree)
    with pytest.raises(ValueError, match="cannot also be a covariate"):
        decision_table(_gated_log(), tree, outcome="priority", covariates=["priority"])


# -- 7. knowledge ------------------------------------------------------------------------


def test_knowledge_tiers_order_covariates_branch_outcome() -> None:
    dt = decision_table(_gated_log(), _gated_tree())
    knowledge = dt.knowledge
    assert knowledge.permits_edge("priority", "branch")
    assert knowledge.permits_edge("branch", "remaining_time")
    assert not knowledge.permits_edge("remaining_time", "branch")
    assert not knowledge.permits_edge("branch", "priority")
    assert not knowledge.permits_edge("remaining_time", "elapsed_at_landmark")
    assert knowledge.is_consistent(dag(["priority -> branch", "branch -> remaining_time"]))
    assert not knowledge.is_consistent(dag(["remaining_time -> branch"]))


# -- 8. the tree invariant ----------------------------------------------------------------


def test_duplicated_label_raises() -> None:
    tree = seq("a", xor("b", seq("a", "d")), "c")
    log = _counts_log()
    with pytest.raises(ValueError, match="'a' appears in more than one leaf"):
        decision_points(log, tree)
    with pytest.raises(ValueError, match="'a' appears in more than one leaf"):
        decision_table(log, tree)


def test_decision_point_repr_is_short() -> None:
    log = _counts_log()
    _, inner = decision_points(log, _counts_tree())
    assert isinstance(inner, DecisionPoint)
    assert repr(inner) == "DecisionPoint(path=(1, 0, 0), a=1, b=2, skipped=1, ambiguous=1)"
