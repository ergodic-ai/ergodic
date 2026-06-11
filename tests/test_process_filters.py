# SPDX-License-Identifier: Apache-2.0
"""Tests for the event-log filters and the variant table."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic.data import EventLog, event_log
from ergodic.process._filters import (
    filter_activities,
    filter_endpoints,
    filter_follows,
    filter_performance,
    filter_time,
    filter_variants,
)
from ergodic.process._variants import Variant, variant_table

if TYPE_CHECKING:
    from collections.abc import Callable


def make_log() -> EventLog:
    """Five cases with known traces, numeric stamps in seconds.

    c1: A(0)   B(10)  C(20)    duration 20, 3 events
    c2: A(100) B(110) C(140)   duration 40, 3 events
    c3: A(200) C(230)          duration 30, 2 events
    c4: A(300) D(400)          duration 100, 2 events
    c5: B(1000) C(1010)        duration 10, 2 events
    """
    df = pd.DataFrame(
        {
            "case": ["c1"] * 3 + ["c2"] * 3 + ["c3"] * 2 + ["c4"] * 2 + ["c5"] * 2,
            "act": ["A", "B", "C", "A", "B", "C", "A", "C", "A", "D", "B", "C"],
            "ts": [0, 10, 20, 100, 110, 140, 200, 230, 300, 400, 1000, 1010],
        }
    )
    return event_log(df, case="case", activity="act", timestamp="ts")


def make_datetime_log() -> EventLog:
    df = pd.DataFrame(
        {
            "case": ["d1", "d1", "d2", "d2"],
            "act": ["A", "B", "A", "B"],
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"]),
        }
    )
    return event_log(df, case="case", activity="act", timestamp="ts")


# -- variant table -------------------------------------------------------------


def test_variant_table_ranking_and_stats() -> None:
    table = variant_table(make_log())
    assert [v.sequence for v in table] == [("A", "B", "C"), ("A", "C"), ("A", "D"), ("B", "C")]
    assert [v.n_cases for v in table] == [2, 1, 1, 1]
    assert [v.share for v in table] == [0.4, 0.2, 0.2, 0.2]
    assert [v.mean_duration for v in table] == [30.0, 30.0, 100.0, 10.0]
    assert [v.median_duration for v in table] == [30.0, 30.0, 100.0, 10.0]
    assert table[0] == Variant(("A", "B", "C"), 2, 0.4, 30.0, 30.0)


def test_variant_table_top_truncates() -> None:
    log = make_log()
    table = variant_table(log, top=2)
    assert [v.sequence for v in table] == [("A", "B", "C"), ("A", "C")]
    assert variant_table(log, top=10) == variant_table(log)
    with pytest.raises(ValueError, match="top"):
        variant_table(log, top=0)


# -- filter_time ---------------------------------------------------------------


def test_filter_time_modes_differ_on_the_same_window() -> None:
    log = make_log()
    contained = filter_time(log, start=100, end=350, mode="contained")
    assert contained.cases == ("c2", "c3")
    assert contained.n_rows == 5

    intersecting = filter_time(log, start=100, end=350, mode="intersecting")
    assert intersecting.cases == ("c2", "c3", "c4")
    assert intersecting.n_rows == 7
    assert intersecting.traces()["c4"] == ("A", "D")  # the whole case survives

    trimmed = filter_time(log, start=100, end=350, mode="trim")
    assert trimmed.cases == ("c2", "c3", "c4")
    assert trimmed.n_rows == 6
    assert trimmed.traces()["c4"] == ("A",)  # the event at 400 is outside


def test_filter_time_open_bounds_and_unknown_mode() -> None:
    log = make_log()
    assert filter_time(log, start=300).cases == ("c4", "c5")
    assert filter_time(log, end=20, mode="contained").cases == ("c1",)
    with pytest.raises(ValueError, match="mode"):
        filter_time(log, start=0, mode="overlapping")


def test_filter_time_datetime_bounds() -> None:
    log = make_datetime_log()
    assert filter_time(log, start=pd.Timestamp("2020-01-15")).cases == ("d2",)
    assert filter_time(log, start=np.datetime64("2020-01-15")).cases == ("d2",)
    trimmed = filter_time(log, end=pd.Timestamp("2020-01-01"), mode="trim")
    assert trimmed.traces() == {"d1": ("A",)}


# -- filter_variants -----------------------------------------------------------


def test_filter_variants_top_and_min_cases() -> None:
    log = make_log()
    assert filter_variants(log, top=1).cases == ("c1", "c2")
    assert filter_variants(log, min_cases=2).cases == ("c1", "c2")
    assert filter_variants(log, min_cases=1).cases == ("c1", "c2", "c3", "c4", "c5")


def test_filter_variants_requires_exactly_one_selector() -> None:
    log = make_log()
    with pytest.raises(ValueError, match="exactly one"):
        filter_variants(log)
    with pytest.raises(ValueError, match="exactly one"):
        filter_variants(log, top=1, min_cases=2)
    with pytest.raises(ValueError, match="top"):
        filter_variants(log, top=0)


# -- filter_endpoints ----------------------------------------------------------


def test_filter_endpoints() -> None:
    log = make_log()
    assert filter_endpoints(log, start_in={"A"}).cases == ("c1", "c2", "c3", "c4")
    assert filter_endpoints(log, end_in={"C"}).cases == ("c1", "c2", "c3", "c5")
    both = filter_endpoints(log, start_in={"A"}, end_in={"C"})
    assert both.cases == ("c1", "c2", "c3")


# -- filter_activities ---------------------------------------------------------


def test_filter_activities_semantics() -> None:
    log = make_log()
    assert filter_activities(log, mandatory={"B"}).cases == ("c1", "c2", "c5")
    assert filter_activities(log, forbidden={"D"}).cases == ("c1", "c2", "c3", "c5")

    kept = filter_activities(log, keep={"A", "C"})
    assert kept.cases == ("c1", "c2", "c3", "c4", "c5")  # cases survive on their kept events
    assert kept.n_rows == 8
    assert kept.traces() == {
        "c1": ("A", "C"),
        "c2": ("A", "C"),
        "c3": ("A", "C"),
        "c4": ("A",),
        "c5": ("C",),
    }


def test_filter_activities_validates() -> None:
    log = make_log()
    with pytest.raises(ValueError, match="at least one"):
        filter_activities(log)
    with pytest.raises(ValueError, match="Z"):
        filter_activities(log, keep={"A", "Z"})
    with pytest.raises(ValueError, match="Z"):
        filter_activities(log, mandatory={"Z"})


# -- filter_follows ------------------------------------------------------------


def test_filter_follows_directly_vs_eventually() -> None:
    log = make_log()
    assert filter_follows(log, "A", "C").cases == ("c3",)
    assert filter_follows(log, "A", "C", directly=False).cases == ("c1", "c2", "c3")


def test_filter_follows_within_excludes_a_slow_pair() -> None:
    log = make_log()
    # A-to-C gaps: c1 takes 20s, c2 takes 40s, c3 takes 30s.
    fast = filter_follows(log, "A", "C", directly=False, within=35)
    assert fast.cases == ("c1", "c3")


def test_filter_follows_unknown_activity_raises() -> None:
    with pytest.raises(ValueError, match="Z"):
        filter_follows(make_log(), "A", "Z")


# -- filter_performance --------------------------------------------------------


def test_filter_performance_bounds() -> None:
    log = make_log()
    assert filter_performance(log, min_duration=25).cases == ("c2", "c3", "c4")
    assert filter_performance(log, max_duration=35).cases == ("c1", "c3", "c5")
    assert filter_performance(log, min_events=3).cases == ("c1", "c2")
    assert filter_performance(log, max_events=2).cases == ("c3", "c4", "c5")
    assert filter_performance(log, min_duration=25, max_events=2).cases == ("c3", "c4")
    with pytest.raises(ValueError, match="at least one"):
        filter_performance(log)


# -- composition and hygiene ---------------------------------------------------


def test_filters_compose_into_a_working_log() -> None:
    log = make_log()
    complete = filter_endpoints(log, start_in={"A"}, end_in={"C"})
    busy = filter_performance(complete, min_events=3)
    assert busy.cases == ("c1", "c2")
    assert busy.n_rows == 6
    assert busy.traces() == {"c1": ("A", "B", "C"), "c2": ("A", "B", "C")}
    assert busy.variants() == Counter({("A", "B", "C"): 2})


def test_filter_removing_everything_raises() -> None:
    log = make_log()
    with pytest.raises(ValueError, match="removed every case"):
        filter_performance(log, min_duration=10_000)
    with pytest.raises(ValueError, match="removed every case"):
        filter_time(log, start=5_000, mode="trim")


def test_filters_are_deterministic_and_pure() -> None:
    log = make_log()
    before = log.traces()
    calls: list[Callable[[EventLog], EventLog]] = [
        lambda lg: filter_time(lg, start=100, end=350, mode="trim"),
        lambda lg: filter_variants(lg, top=1),
        lambda lg: filter_endpoints(lg, start_in={"A"}),
        lambda lg: filter_activities(lg, keep={"A", "C"}),
        lambda lg: filter_follows(lg, "A", "C", directly=False),
        lambda lg: filter_performance(lg, min_events=3),
    ]
    for call in calls:
        once, twice = call(log), call(log)
        assert once.traces() == twice.traces()
        assert once.n_rows == twice.n_rows
    assert log.n_rows == 12
    assert log.traces() == before
