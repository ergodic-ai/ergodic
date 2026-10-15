# SPDX-License-Identifier: Apache-2.0
"""Tests for the process map and the performance summaries."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ergodic.data import EventLog, event_log
from ergodic.process._map import END, START, ArcStats, process_map
from ergodic.process._performance import ActivityStats, performance

# -- fixtures ------------------------------------------------------------------


def _two_case_log() -> EventLog:
    """Case 1: A@0, B@10, C@30. Case 2: A@0, C@5."""
    df = pd.DataFrame(
        {
            "case": ["1", "1", "1", "2", "2"],
            "act": ["A", "B", "C", "A", "C"],
            "ts": [0, 10, 30, 0, 5],
        }
    )
    return event_log(df, case="case", activity="act", timestamp="ts")


def _skewed_log() -> EventLog:
    """Sixteen two-event cases: arc counts A->B 8, B->C 4, C->D 2, D->E 1, E->F 1."""
    traces = [("A", "B")] * 8 + [("B", "C")] * 4 + [("C", "D")] * 2 + [("D", "E"), ("E", "F")]
    case_ids: list[str] = []
    acts: list[str] = []
    stamps: list[int] = []
    for i, (a, b) in enumerate(traces):
        case_ids += [f"c{i:02d}", f"c{i:02d}"]
        acts += [a, b]
        stamps += [0, 1]
    df = pd.DataFrame({"case": case_ids, "act": acts, "ts": stamps})
    return event_log(df, case="case", activity="act", timestamp="ts")


def _lifecycle_log() -> EventLog:
    """Case k: triage start@0/complete@5, review start@8/complete@20. Case q: a lone start."""
    df = pd.DataFrame(
        {
            "case": ["k", "k", "k", "k", "q"],
            "act": ["triage", "triage", "review", "review", "triage"],
            "ts": [0, 5, 8, 20, 0],
            "stage": ["start", "complete", "start", "complete", "start"],
        }
    )
    return event_log(df, case="case", activity="act", timestamp="ts")


# -- process map ---------------------------------------------------------------


def test_process_map_counts_and_gaps() -> None:
    m = process_map(_two_case_log())
    assert m.n_cases == 2
    assert m.n_events == 5
    assert m.activities == ("A", "B", "C")
    expected = {(START, "A"): 2, ("A", "B"): 1, ("A", "C"): 1, ("B", "C"): 1, ("C", END): 2}
    assert {pair: stats.count for pair, stats in m.arcs.items()} == expected
    assert m.arcs[("A", "B")] == ArcStats(count=1, mean_gap=10.0, median_gap=10.0)
    assert m.arcs[("B", "C")].mean_gap == 20.0
    assert m.arcs[("A", "C")].mean_gap == 5.0
    assert math.isnan(m.arcs[(START, "A")].mean_gap)
    assert math.isnan(m.arcs[("C", END)].median_gap)
    assert m.frequency("A", "B") == 1
    assert m.frequency(START, "A") == 2
    assert m.frequency("C", "A") == 0
    assert m.frequency("A", END) == 0
    assert list(m.arcs) == sorted(m.arcs)
    assert m.dropped_arcs == 0
    assert m.dropped_share == 0.0


def test_process_map_datetime_gap_in_seconds() -> None:
    df = pd.DataFrame(
        {
            "case": ["c", "c"],
            "act": ["A", "B"],
            "ts": pd.to_datetime(["2024-01-01 09:00:00", "2024-01-01 10:00:00"]),
        }
    )
    m = process_map(event_log(df, case="case", activity="act", timestamp="ts"))
    assert m.arcs[("A", "B")] == ArcStats(count=1, mean_gap=3600.0, median_gap=3600.0)


def test_process_map_reserved_activity_name_raises() -> None:
    df = pd.DataFrame({"case": ["c", "c"], "act": ["__start__", "B"], "ts": [0, 1]})
    log = event_log(df, case="case", activity="act", timestamp="ts")
    with pytest.raises(ValueError, match="__start__"):
        process_map(log)


def test_simplify_keeps_heavy_arcs_and_accumulates() -> None:
    m = process_map(_skewed_log())
    assert m.n_cases == 16
    assert m.n_events == 32

    s1 = m.simplify(keep=0.7)  # threshold 11.2 of 16: keep A->B (8) and B->C (4)
    real1 = [pair for pair in s1.arcs if pair[0] != START and pair[1] != END]
    assert real1 == [("A", "B"), ("B", "C")]
    assert s1.dropped_arcs == 3
    assert s1.dropped_share == 0.25  # 4 of 16 traversals gone
    assert len(s1.arcs) == 12  # 2 real plus all 10 boundary arcs
    assert s1.frequency(START, "E") == 1
    assert s1.frequency("F", END) == 1
    assert s1.activities == m.activities  # isolated activities stay
    assert s1.n_cases == 16
    assert s1.n_events == 32

    s2 = s1.simplify(keep=0.5)  # threshold 6 of 12: keep A->B alone
    real2 = [pair for pair in s2.arcs if pair[0] != START and pair[1] != END]
    assert real2 == [("A", "B")]
    assert s2.dropped_arcs == 4  # accumulated: 3 then 1 more
    assert s2.dropped_share == 0.5  # accumulated: 4 then 4 more of the original 16
    assert len(s2.arcs) == 11


def test_simplify_keep_one_drops_nothing() -> None:
    m = process_map(_skewed_log())
    assert m.simplify(keep=1.0) == m


def test_simplify_keep_out_of_range_raises() -> None:
    m = process_map(_two_case_log())
    for bad in (0.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="keep"):
            m.simplify(keep=bad)


def test_to_dict_and_repr() -> None:
    m = process_map(_two_case_log())
    d = m.to_dict()
    assert d["activities"] == ["A", "B", "C"]
    assert len(d["arcs"]) == 5
    by_pair = {(row["source"], row["target"]): row["count"] for row in d["arcs"]}
    assert by_pair[("A", "B")] == 1
    assert by_pair[(START, "A")] == 2
    assert d["n_cases"] == 2
    assert d["n_events"] == 5
    assert "dropped" not in repr(m)

    s1 = process_map(_skewed_log()).simplify(keep=0.7)
    assert "dropped 3 arcs" in repr(s1)
    assert "25.0%" in repr(s1)


# -- performance ---------------------------------------------------------------


def test_performance_complete_only() -> None:
    perf = performance(_two_case_log())
    assert perf.n_cases == 2
    assert perf.n_events == 5
    assert perf.mean_duration == 17.5  # case durations 30 and 5
    assert perf.median_duration == 17.5
    assert perf.p90_duration == 27.5
    assert perf.lifecycle is False
    assert perf.n_unmatched == 0
    assert [s.activity for s in perf.activities] == ["A", "B", "C"]
    stats = {s.activity: s for s in perf.activities}
    assert stats["A"] == ActivityStats(activity="A", count=2, mean_wait=None, mean_service=None)
    assert stats["B"] == ActivityStats(activity="B", count=1, mean_wait=10.0, mean_service=None)
    assert stats["C"].mean_wait == 12.5  # gaps 20 and 5
    assert all(s.mean_service is None for s in perf.activities)


def test_performance_lifecycle_split() -> None:
    perf = performance(_lifecycle_log(), lifecycle="stage")
    assert perf.lifecycle is True
    assert perf.n_unmatched == 1  # case q's lone start
    assert perf.n_cases == 2
    assert perf.n_events == 5
    assert perf.mean_duration == 10.0  # case durations 20 and 0
    assert perf.median_duration == 10.0
    assert perf.p90_duration == 18.0
    assert [s.activity for s in perf.activities] == ["review", "triage"]
    stats = {s.activity: s for s in perf.activities}
    assert stats["review"] == ActivityStats(
        activity="review", count=2, mean_wait=3.0, mean_service=12.0
    )
    assert stats["triage"] == ActivityStats(
        activity="triage", count=3, mean_wait=None, mean_service=5.0
    )


def test_performance_lifecycle_other_values_ignored() -> None:
    df = pd.DataFrame(
        {
            "case": ["c", "c", "c"],
            "act": ["x", "x", "x"],
            "ts": [0, 1, 2],
            "stage": ["start", "pause", "complete"],
        }
    )
    log = event_log(df, case="case", activity="act", timestamp="ts", types={"stage": "categorical"})
    perf = performance(log, lifecycle="stage")
    assert perf.n_unmatched == 0
    stats = {s.activity: s for s in perf.activities}
    assert stats["x"] == ActivityStats(activity="x", count=3, mean_wait=None, mean_service=2.0)


def test_performance_unknown_lifecycle_column_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        performance(_two_case_log(), lifecycle="nope")


# -- determinism ---------------------------------------------------------------


def test_arc_stats_with_nan_gaps_compare_and_hash_equal() -> None:
    from ergodic.process._map import ArcStats

    # Distinct NaN objects, so equality cannot lean on identity.
    left = ArcStats(count=2, mean_gap=float("nan"), median_gap=float("nan"))
    right = ArcStats(count=2, mean_gap=float("nan"), median_gap=float("nan"))
    assert left == right and hash(left) == hash(right)
    assert left != ArcStats(count=3, mean_gap=float("nan"), median_gap=float("nan"))
    assert ArcStats(1, 2.0, 2.0) != ArcStats(1, 2.0, float("nan"))


def test_repeated_calls_give_equal_objects() -> None:
    plain = _two_case_log()
    skewed = _skewed_log()
    staged = _lifecycle_log()
    assert process_map(plain) == process_map(plain)
    assert process_map(skewed).simplify(keep=0.7) == process_map(skewed).simplify(keep=0.7)
    assert performance(plain) == performance(plain)
    assert performance(staged, lifecycle="stage") == performance(staged, lifecycle="stage")
