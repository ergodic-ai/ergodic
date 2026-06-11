# SPDX-License-Identifier: Apache-2.0
"""Tests for the case-table encoder: timing gates, leakage, and the composed pipeline."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, estimate_effect, event_log
from ergodic.discovery import discover
from ergodic.process._case_table import CaseTable, case_table

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import EventLog

# -- known-truth worlds --------------------------------------------------------


@cache
def _complexity_world() -> tuple[EventLog, npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Confounded review decision: complexity drives review and cycle time.

    P(review) = sigmoid(1.3 * complexity); cycle = 20 + 6 * complexity +
    5.0 * review + N(0, 2) seconds; arrivals spread over [0, 5000] so the
    workload feature varies. The planted treatment effect is 5.0.
    """
    n = 1500
    rng = np.random.default_rng(11)
    complexity = rng.normal(size=n)
    review = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.3 * complexity))
    cycle = 20.0 + 6.0 * complexity + 5.0 * review.astype(float) + rng.normal(0.0, 2.0, size=n)
    start = rng.uniform(0.0, 5000.0, size=n)
    rows = []
    for i in range(n):
        case = f"c{i:04d}"
        rows.append((case, "open", start[i], complexity[i]))
        rows.append((case, "triage", start[i] + 1.0, complexity[i]))
        if review[i]:
            rows.append((case, "review", start[i] + 2.0, complexity[i]))
        rows.append((case, "close", start[i] + cycle[i], complexity[i]))
    df = pd.DataFrame(rows, columns=["case", "activity", "time", "complexity"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    return log, review, cycle


@cache
def _landmark_world() -> tuple[
    EventLog,
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
]:
    """A world for the after_events landmark at k = 2.

    The first 45 cases close after two events (short). Among the rest,
    P(review) = sigmoid(0.9 * severity); 10 percent of the reviewed cases get
    the review as event 2 (inside the prefix), the others as event 3. The
    planted effect among the at-risk cases is 4.0 seconds.
    """
    n = 900
    rng = np.random.default_rng(5)
    severity = rng.normal(size=n)
    start = rng.uniform(0.0, 3000.0, size=n)
    triage_gap = rng.uniform(0.5, 2.0, size=n)
    short = np.zeros(n, dtype=bool)
    short[:45] = True
    treated = ~short & (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.9 * severity)))
    early = treated & (rng.uniform(size=n) < 0.10)
    cycle = 20.0 + 2.5 * severity + 4.0 * treated.astype(float) + rng.normal(0.0, 1.0, size=n)
    rows = []
    for i in range(n):
        case = f"c{i:04d}"
        t0 = start[i]
        rows.append((case, "open", t0, severity[i]))
        if short[i]:
            rows.append((case, "close", t0 + cycle[i], severity[i]))
        elif early[i]:
            rows.append((case, "review", t0 + triage_gap[i], severity[i]))
            rows.append((case, "work", t0 + triage_gap[i] + 0.7, severity[i]))
            rows.append((case, "close", t0 + cycle[i], severity[i]))
        elif treated[i]:
            rows.append((case, "triage", t0 + triage_gap[i], severity[i]))
            rows.append((case, "review", t0 + triage_gap[i] + 0.7, severity[i]))
            rows.append((case, "close", t0 + cycle[i], severity[i]))
        else:
            rows.append((case, "triage", t0 + triage_gap[i], severity[i]))
            rows.append((case, "close", t0 + cycle[i], severity[i]))
    df = pd.DataFrame(rows, columns=["case", "activity", "time", "severity"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    return log, short, treated, early, triage_gap


def _small_log(extra: dict[str, list[float]] | None = None) -> EventLog:
    """Four cases, numeric timestamps, an optional case-constant extra column."""
    events = [
        ("a", "open", 0.0),
        ("a", "fix", 2.0),
        ("a", "close", 6.0),
        ("b", "open", 1.0),
        ("b", "close", 4.0),
        ("c", "open", 2.0),
        ("c", "fix", 3.0),
        ("c", "close", 10.0),
        ("d", "open", 3.0),
        ("d", "close", 5.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    counts = {"a": 3, "b": 2, "c": 3, "d": 2}
    for name, per_case in (extra or {}).items():
        df[name] = [
            value for case, value in zip("abcd", per_case, strict=True) for _ in range(counts[case])
        ]
    return event_log(df, case="case", activity="activity", timestamp="time")


# -- 1. the complexity world ---------------------------------------------------


def test_complexity_world_encodes_start_features_only() -> None:
    log, review, _ = _complexity_world()
    ct = case_table(log, "review", "cycle_time")
    assert isinstance(ct, CaseTable)
    assert ct.when == "anytime"
    assert ct.n_cases == 1500
    assert ct.n_dropped_short == 0 and ct.n_dropped_treated == 0
    # Case-start features only: the attribute and the workload, nothing post-treatment.
    assert ct.covariates == ("complexity", "workload_at_start")
    # Numeric timestamps carry no calendar, so the calendar features must be absent.
    assert "start_weekday" not in ct.data.names
    assert "start_hour" not in ct.data.names
    assert ct.data.names == ("complexity", "workload_at_start", "review", "cycle_time")
    treated = ct.data.column("review")
    assert treated.dtype == np.int64
    assert set(np.unique(treated).tolist()) == {0, 1}
    assert np.array_equal(treated, review.astype(np.int64))
    assert ct.data.schema["review"].is_binary


def test_complexity_naive_gap_overstates_the_effect() -> None:
    log, _, _ = _complexity_world()
    ct = case_table(log, "review", "cycle_time")
    a = ct.data.column("review")
    y = ct.data.column("cycle_time")
    naive = float(y[a == 1].mean() - y[a == 0].mean())
    assert naive - 5.0 > 1.5  # confounding by complexity is visible in the raw gap


def test_complexity_direct_estimation_recovers_the_planted_effect() -> None:
    log, _, _ = _complexity_world()
    ct = case_table(log, "review", "cycle_time")
    graph = dag(
        [
            "complexity -> review",
            "complexity -> cycle_time",
            "review -> cycle_time",
            "workload_at_start -> cycle_time",
        ]
    )
    est = estimate_effect(graph, ct.data, "review", "cycle_time", random_state=0, n_boot=200)
    assert abs(est.value - 5.0) < 0.6


def test_complexity_composed_pipeline_recovers_the_planted_effect() -> None:
    log, _, _ = _complexity_world()
    ct = case_table(log, "review", "cycle_time")
    result = discover(ct.data, knowledge=ct.knowledge, alpha=0.01)
    # The tier knowledge orients every edge, which puts the graph outside the
    # CPDAG class on purpose; re-facing it as a DAG is the documented next step.
    est = estimate_effect(
        result.graph.as_kind("dag"),
        ct.data,
        "review",
        "cycle_time",
        strategy="dml",
        random_state=0,
    )
    assert abs(est.value - 5.0) < 0.7


def test_complexity_knowledge_forbids_post_treatment_arrows() -> None:
    log, _, _ = _complexity_world()
    ct = case_table(log, "review", "cycle_time")
    assert ct.knowledge.permits_edge("complexity", "review")
    assert ct.knowledge.permits_edge("review", "cycle_time")
    assert not ct.knowledge.permits_edge("cycle_time", "review")
    assert not ct.knowledge.permits_edge("review", "complexity")
    assert ct.knowledge.is_consistent(
        dag(["complexity -> review", "review -> cycle_time", "workload_at_start -> cycle_time"])
    )
    assert not ct.knowledge.is_consistent(dag(["cycle_time -> review"]))


# -- 2. the after_events landmark ------------------------------------------------


def test_after_events_landmark_counts_features_and_recovery() -> None:
    log, short, treated, early, triage_gap = _landmark_world()
    ct = case_table(log, "review", "cycle_time", when=("after_events", 2))
    keep = ~short & ~early
    assert ct.when == "after_events"
    assert ct.n_dropped_short == 45
    assert ct.n_dropped_treated == int((~short & early).sum())
    assert ct.n_dropped_treated > 0
    assert ct.n_cases == int(keep.sum())
    # Prefix features at the landmark join the start features; the constant
    # ones (every kept prefix is open then triage) are dropped in silence.
    assert ct.covariates == (
        "elapsed_at_landmark",
        "severity",
        "workload_at_landmark",
        "workload_at_start",
    )
    assert np.array_equal(ct.data.column("review"), treated[keep].astype(np.int64))
    assert np.allclose(ct.data.column("elapsed_at_landmark"), triage_gap[keep])
    graph = dag(
        [
            "severity -> review",
            "severity -> cycle_time",
            "review -> cycle_time",
            "elapsed_at_landmark -> cycle_time",
            "workload_at_landmark -> cycle_time",
            "workload_at_start -> cycle_time",
        ]
    )
    est = estimate_effect(graph, ct.data, "review", "cycle_time", random_state=0, n_boot=200)
    assert abs(est.value - 4.0) < 0.5  # the effect among cases at risk at event 2


# -- 3. the after_elapsed landmark ------------------------------------------------


def test_after_elapsed_counts_are_exact_on_a_hand_log() -> None:
    events = [
        ("A", "start", 0.0),  # treated at 5 <= 10: dropped treated
        ("A", "treat", 5.0),
        ("A", "end", 20.0),
        ("B", "start", 0.0),  # closes at 8 <= 10: dropped short
        ("B", "end", 8.0),
        ("C", "start", 0.0),  # treated at 12 > 10: kept, treatment 1
        ("C", "treat", 12.0),
        ("C", "end", 30.0),
        ("D", "start", 0.0),  # never treated, open at 10: kept, treatment 0
        ("D", "end", 30.0),
        ("E", "start", 0.0),  # one event, closed at 0: dropped short
        ("F", "start", 0.0),  # treated exactly at the landmark: dropped treated
        ("F", "treat", 10.0),
        ("F", "end", 25.0),
        ("G", "start", 0.0),  # closes exactly at the landmark: dropped short
        ("G", "end", 10.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    ct = case_table(log, "treat", when=("after_elapsed", 10))
    assert ct.when == "after_elapsed"
    assert ct.n_dropped_short == 3  # B, E, G
    assert ct.n_dropped_treated == 2  # A, F
    assert ct.n_cases == 2  # C and D, in case order
    assert np.array_equal(ct.data.column("treat"), np.array([1, 0], dtype=np.int64))
    assert np.allclose(ct.data.column("cycle_time"), [30.0, 30.0])
    # elapsed_at_landmark always equals x here, so the zero-variance rule drops
    # it, along with every other derived feature this tiny log keeps constant.
    assert "elapsed_at_landmark" not in ct.covariates
    assert ct.covariates == ()


# -- calendar features -----------------------------------------------------------


def test_datetime_timestamps_yield_calendar_start_features() -> None:
    events = [
        ("a", "open", pd.Timestamp("2026-06-08 09:30")),  # a Monday
        ("a", "fix", pd.Timestamp("2026-06-08 10:30")),
        ("a", "close", pd.Timestamp("2026-06-08 11:30")),
        ("b", "open", pd.Timestamp("2026-06-13 23:10")),  # a Saturday
        ("b", "close", pd.Timestamp("2026-06-14 01:00")),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    ct = case_table(log, "fix")
    assert "start_weekday" in ct.covariates and "start_hour" in ct.covariates
    assert np.allclose(ct.data.column("start_weekday"), [0.0, 5.0])
    assert np.allclose(ct.data.column("start_hour"), [9.0, 23.0])
    assert np.allclose(ct.data.column("cycle_time"), [7200.0, 6600.0])


# -- 4. column treatment ----------------------------------------------------------


def test_column_treatment_works_anytime_and_rejects_landmarks() -> None:
    log = _small_log({"intervention": [1.0, 0.0, 1.0, 0.0], "cost": [4.0, 7.0, 1.0, 9.0]})
    ct = case_table(log, "intervention", "cycle_time")
    assert np.array_equal(ct.data.column("intervention"), np.array([1, 0, 1, 0], dtype=np.int64))
    assert ct.data.schema["intervention"].is_binary
    assert "cost" in ct.covariates and "intervention" not in ct.covariates
    with pytest.raises(ValueError, match="carries no event timing"):
        case_table(log, "intervention", "cycle_time", when=("after_events", 1))
    with pytest.raises(ValueError, match="carries no event timing"):
        case_table(log, "intervention", "cycle_time", when=("after_elapsed", 2.0))


def test_column_treatment_must_be_zero_one() -> None:
    log = _small_log({"dose": [0.0, 2.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="0/1 or booleans"):
        case_table(log, "dose", "cycle_time")


# -- 5. outcome modes --------------------------------------------------------------


def test_outcome_n_events_counts_events_per_case() -> None:
    log = _small_log()
    ct = case_table(log, "fix", "n_events")
    assert np.allclose(ct.data.column("n_events"), [3.0, 2.0, 3.0, 2.0])


def test_outcome_case_constant_column() -> None:
    log = _small_log({"cost": [4.0, 7.0, 1.0, 9.0]})
    ct = case_table(log, "fix", "cost")
    assert np.allclose(ct.data.column("cost"), [4.0, 7.0, 1.0, 9.0])
    assert ct.data.schema["cost"].is_continuous
    assert "cost" not in ct.covariates


def test_outcome_reserved_name_clashing_with_a_column_raises() -> None:
    log = _small_log({"cycle_time": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="reserved derived name"):
        case_table(log, "fix", "cycle_time")


def test_outcome_unknown_name_raises() -> None:
    log = _small_log()
    with pytest.raises(ValueError, match="not 'cycle_time', 'n_events', or a case-constant"):
        case_table(log, "fix", "throughput")


# -- 6. validation ------------------------------------------------------------------


def test_unknown_treatment_lists_what_it_searched() -> None:
    log = _small_log()
    with pytest.raises(ValueError, match="neither an activity nor a column"):
        case_table(log, "escalate")


def test_treatment_that_is_both_activity_and_column_raises() -> None:
    log = _small_log({"fix": [1.0, 0.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="both an activity and a column"):
        case_table(log, "fix")


def test_named_covariate_that_varies_within_a_case_raises() -> None:
    events = [
        ("a", "open", 0.0, 1.0),
        ("a", "close", 5.0, 2.0),  # queue varies inside case a
        ("b", "open", 1.0, 3.0),
        ("b", "fix", 2.0, 3.0),
        ("b", "close", 4.0, 3.0),
    ]
    df = pd.DataFrame(events, columns=["case", "activity", "time", "queue"])
    log = event_log(df, case="case", activity="activity", timestamp="time")
    with pytest.raises(ValueError, match="varies within case"):
        case_table(log, "fix", covariates=["queue"])
    # auto skips the varying column instead of raising
    ct = case_table(log, "fix")
    assert "queue" not in ct.covariates


def test_bad_when_raises() -> None:
    log = _small_log()
    with pytest.raises(ValueError, match="unknown when"):
        case_table(log, "fix", when="sometimes")
    with pytest.raises(ValueError, match="needs a parameter"):
        case_table(log, "fix", when="after_events")
    with pytest.raises(ValueError, match="k >= 1"):
        case_table(log, "fix", when=("after_events", 0))
    with pytest.raises(ValueError, match="integer event count"):
        case_table(log, "fix", when=("after_events", 1.5))
    with pytest.raises(ValueError, match="x > 0"):
        case_table(log, "fix", when=("after_elapsed", 0))
    with pytest.raises(ValueError, match="unknown landmark mode"):
        case_table(log, "fix", when=("later", 3))


def test_covariate_name_validation() -> None:
    log = _small_log({"cost": [4.0, 7.0, 1.0, 9.0]})
    with pytest.raises(ValueError, match="unknown covariate"):
        case_table(log, "fix", covariates=["cost", "nope"])
    with pytest.raises(ValueError, match="needs a datetime timestamp"):
        case_table(log, "fix", covariates=["start_weekday"])
    with pytest.raises(ValueError, match="only computed by the landmark modes"):
        case_table(log, "fix", covariates=["workload_at_landmark"])
    with pytest.raises(ValueError, match="cannot also be a covariate"):
        case_table(log, "fix", "cost", covariates=["cost"])
    ct = case_table(log, "fix", covariates=["cost"])
    assert ct.covariates == ("cost",)
    assert ct.data.names == ("cost", "fix", "cycle_time")


# -- 7. determinism -----------------------------------------------------------------


def test_identical_calls_build_identical_tables() -> None:
    log, _, _ = _complexity_world()
    first = case_table(log, "review", "cycle_time")
    second = case_table(log, "review", "cycle_time")
    assert first.covariates == second.covariates
    assert first.data.names == second.data.names
    assert np.array_equal(first.data.matrix(), second.data.matrix())
