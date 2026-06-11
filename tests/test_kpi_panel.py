# SPDX-License-Identifier: Apache-2.0
"""Tests for kpi_panel: bucketing, kpi columns, output shapes, and the did bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic.data import EventLog, PanelData, TimeSeriesData, event_log
from ergodic.inference.estimators._did import did
from ergodic.inference.estimators._its import interrupted_time_series
from ergodic.process._panel import kpi_panel

# -- hand logs -------------------------------------------------------------------


def _numeric_frame() -> pd.DataFrame:
    """Cases starting at t = 0, 3, 7, 13 with known durations and event counts."""
    return pd.DataFrame(
        {
            "case": ["a", "a", "a", "b", "b", "c", "c", "d", "d"],
            "activity": ["s", "m", "e", "s", "e", "s", "e", "s", "e"],
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 11.0, 13.0, 13.5],
        }
    )


def _numeric_log(frame: pd.DataFrame | None = None) -> EventLog:
    return event_log(
        frame if frame is not None else _numeric_frame(),
        case="case",
        activity="activity",
        timestamp="t",
    )


def _stamp_log(starts: list[str]) -> EventLog:
    """One case per start, two events an hour apart, datetime stamps."""
    cases: list[str] = []
    times: list[pd.Timestamp] = []
    for i, start in enumerate(starts):
        begin = pd.Timestamp(start)
        cases += [f"c{i}", f"c{i}"]
        times += [begin, begin + pd.Timedelta(hours=1)]
    frame = pd.DataFrame({"case": cases, "activity": ["open", "close"] * len(starts), "t": times})
    return event_log(frame, case="case", activity="activity", timestamp="t")


def _team_log() -> EventLog:
    """Team A starts cases in buckets 0 and 5, team B only in bucket 0 (width 5)."""
    frame = pd.DataFrame(
        {
            "case": ["a1", "a1", "a2", "a2", "b1", "b1"],
            "activity": ["s", "e", "s", "e", "s", "e"],
            "t": [0.0, 2.0, 6.0, 9.0, 1.0, 1.5],
            "team": ["A", "A", "A", "A", "B", "B"],
        }
    )
    return event_log(frame, case="case", activity="activity", timestamp="t")


# -- exact bucketing ---------------------------------------------------------------


def test_numeric_bucketing_is_exact() -> None:
    series = kpi_panel(_numeric_log(), 5.0, kpis=("cycle_time", "n_cases", "n_events"))
    assert isinstance(series, TimeSeriesData)
    assert series.time == "period"
    assert series.names == ("period", "cycle_time", "n_cases", "n_events")
    assert np.array_equal(series.column("period"), np.array([0.0, 5.0, 10.0]))
    # bucket 0 holds cases a (cycle 2, 3 events) and b (cycle 1, 2 events)
    assert np.array_equal(series.column("cycle_time"), np.array([1.5, 4.0, 0.5]))
    assert np.array_equal(series.column("n_cases"), np.array([2.0, 1.0, 1.0]))
    assert np.array_equal(series.column("n_events"), np.array([2.5, 2.0, 2.0]))


def test_datetime_bucketing_lands_on_days_mondays_and_months() -> None:
    # 2024-01-02 is a Tuesday, 2024-01-07 the Sunday after it, 2024-01-08 a Monday
    week_log = _stamp_log(["2024-01-02T09:00", "2024-01-07T23:00", "2024-01-08T09:00"])
    by_week = kpi_panel(week_log, "week", kpis=("n_cases",))
    assert by_week.column("period").dtype == np.dtype("datetime64[D]")
    assert np.array_equal(
        by_week.column("period"),
        np.array(["2024-01-01", "2024-01-08"], dtype="datetime64[D]"),
    )
    assert np.array_equal(by_week.column("n_cases"), np.array([2.0, 1.0]))

    boundary_log = _stamp_log(["2024-01-31T23:00", "2024-02-01T01:00"])
    by_month = kpi_panel(boundary_log, "month", kpis=("n_cases",))
    assert by_month.column("period").dtype == np.dtype("datetime64[M]")
    assert np.array_equal(
        by_month.column("period"), np.array(["2024-01", "2024-02"], dtype="datetime64[M]")
    )
    assert np.array_equal(by_month.column("n_cases"), np.array([1.0, 1.0]))

    by_day = kpi_panel(boundary_log, "day")
    assert by_day.column("period").dtype == np.dtype("datetime64[D]")
    assert np.array_equal(
        by_day.column("period"), np.array(["2024-01-31", "2024-02-01"], dtype="datetime64[D]")
    )
    assert np.array_equal(by_day.column("cycle_time"), np.array([3600.0, 3600.0]))


# -- validation --------------------------------------------------------------------


def test_reserved_name_collision_raises() -> None:
    frame = _numeric_frame()
    frame["cycle_time"] = 1.0
    log = _numeric_log(frame)
    with pytest.raises(ValueError, match="ambiguous"):
        kpi_panel(log, 5.0, kpis=("cycle_time",))


def test_non_numeric_kpi_column_raises() -> None:
    frame = _numeric_frame()
    frame["owner"] = ["ann", "ann", "ann", "bob", "bob", "cat", "cat", "dan", "dan"]
    log = _numeric_log(frame)
    with pytest.raises(ValueError, match="numeric"):
        kpi_panel(log, 5.0, kpis=("owner",))


def test_period_validation() -> None:
    numeric = _numeric_log()
    with pytest.raises(TypeError, match="numeric"):
        kpi_panel(numeric, "week")
    with pytest.raises(ValueError, match="positive"):
        kpi_panel(numeric, 0.0)
    with pytest.raises(ValueError, match="positive"):
        kpi_panel(numeric, -3.0)
    stamped = _stamp_log(["2024-01-02T09:00", "2024-01-03T09:00"])
    with pytest.raises(TypeError, match="datetime"):
        kpi_panel(stamped, 7.0)
    with pytest.raises(ValueError, match="day"):
        kpi_panel(stamped, "fortnight")


def test_output_name_collisions_raise() -> None:
    log = _team_log()
    with pytest.raises(ValueError, match="period"):
        kpi_panel(log, 5.0, kpis=("period",))
    with pytest.raises(ValueError, match="unit"):
        kpi_panel(log, 5.0, unit="team", kpis=("team",))
    with pytest.raises(ValueError, match="repeat"):
        kpi_panel(log, 5.0, kpis=("cycle_time", "cycle_time"))


# -- output shapes ------------------------------------------------------------------


def test_unit_gives_a_panel_and_empty_cells_are_absent() -> None:
    pnl = kpi_panel(_team_log(), 5.0, unit="team", kpis=("cycle_time", "n_cases"))
    assert isinstance(pnl, PanelData)
    assert pnl.entity == "team"
    assert pnl.time == "period"
    assert pnl.names == ("team", "period", "cycle_time", "n_cases")
    assert pnl.n_rows == 3  # the (B, 5.0) cell has no starting case, so no row
    assert not pnl.is_balanced
    frame = pnl.to_pandas()
    cells = list(zip(frame["team"], frame["period"], strict=True))
    assert cells == [("A", 0.0), ("A", 5.0), ("B", 0.0)]
    assert frame["cycle_time"].tolist() == [2.0, 3.0, 0.5]
    assert frame["n_cases"].tolist() == [1.0, 1.0, 1.0]


def test_no_unit_gives_a_timeseries() -> None:
    series = kpi_panel(_team_log(), 5.0)
    assert isinstance(series, TimeSeriesData)
    assert series.time == "period"
    assert series.names == ("period", "cycle_time")
    assert np.array_equal(series.column("period"), np.array([0.0, 5.0]))


# -- the bridge anchor: log -> kpi_panel -> did --------------------------------------


def _routing_frame(seed: int = 0) -> pd.DataFrame:
    """30 teams over 26 weeks of day stamps, about 10 cases per team-week.

    Teams 0-9 adopt the new routing at week 10, teams 10-19 at week 18, and
    teams 20-29 never. Ever-adopting teams sit 4.0 higher in base cycle time
    (selection on level), and adopted cases run 3.0 faster (the planted
    effect). Adoption aligns to week boundaries, so every case in a team-week
    shares the flag.
    """
    rng = np.random.default_rng(seed)
    n_teams, n_weeks = 30, 26
    adopt = np.array([10] * 10 + [18] * 10 + [n_weeks] * 10)
    base = 10.0 + 4.0 * (adopt < n_weeks) + rng.normal(0.0, 0.5, n_teams)
    shock = rng.normal(0.0, 0.3, n_weeks)
    cases: list[str] = []
    acts: list[str] = []
    stamps: list[float] = []
    teams: list[int] = []
    flags: list[float] = []
    for team in range(n_teams):
        for week in range(n_weeks):
            flag = 1.0 if week >= adopt[team] else 0.0
            n_cases = int(rng.integers(8, 13))
            starts = 7.0 * week + 6.9 * rng.random(n_cases)
            cycles = base[team] + shock[week] - 3.0 * flag + rng.normal(0.0, 1.0, n_cases)
            for i in range(n_cases):
                cid = f"t{team:02d}w{week:02d}c{i:02d}"
                cases += [cid, cid]
                acts += ["received", "resolved"]
                stamps += [float(starts[i]), float(starts[i] + cycles[i])]
                teams += [team, team]
                flags += [flag, flag]
    return pd.DataFrame(
        {"case": cases, "activity": acts, "t": stamps, "team": teams, "new_routing": flags}
    )


def test_bridge_log_to_did_recovers_the_planted_routing_effect() -> None:
    log = event_log(_routing_frame(), case="case", activity="activity", timestamp="t")
    pnl = kpi_panel(log, 7.0, unit="team", kpis=("cycle_time", "new_routing"))
    assert isinstance(pnl, PanelData)
    assert pnl.n_rows == 30 * 26
    routing = pnl.column("new_routing")
    assert bool(np.isin(routing, (0.0, 1.0)).all())  # adoption aligns to week boundaries

    est = did(pnl, "cycle_time", "new_routing", n_boot=200, random_state=0)
    assert est.flavor == "group_time"  # two adoption cohorts, so auto picks group_time
    assert abs(est.value - (-3.0)) < 0.5
    lo, hi = est.interval(0.95)
    assert lo <= -3.0 <= hi

    # the naive post-period cross-section absorbs the selection on level and misses
    frame = pnl.to_pandas()
    post = frame[frame["period"] >= 7.0 * 18]
    naive = float(post.loc[post["new_routing"] == 1.0, "cycle_time"].mean()) - float(
        post.loc[post["new_routing"] == 0.0, "cycle_time"].mean()
    )
    assert abs(naive - (-3.0)) > 1.0


# -- interrupted time series on one series -------------------------------------------


def test_its_covers_a_planted_level_shift() -> None:
    rng = np.random.default_rng(1)
    n_weeks, break_week = 40, 20
    cases: list[str] = []
    acts: list[str] = []
    stamps: list[float] = []
    for week in range(n_weeks):
        starts = 7.0 * week + 6.9 * rng.random(8)
        cycles = 12.0 - 4.0 * (week >= break_week) + rng.normal(0.0, 0.8, 8)
        for i in range(8):
            cid = f"w{week:02d}c{i}"
            cases += [cid, cid]
            acts += ["open", "close"]
            stamps += [float(starts[i]), float(starts[i] + cycles[i])]
    frame = pd.DataFrame({"case": cases, "activity": acts, "t": stamps})
    log = event_log(frame, case="case", activity="activity", timestamp="t")

    series = kpi_panel(log, 7.0)
    assert isinstance(series, TimeSeriesData)
    assert series.n_rows == n_weeks
    est = interrupted_time_series(series, "cycle_time", at=7.0 * break_week)
    assert abs(est.value - (-4.0)) < 1.0
    lo, hi = est.interval()
    assert lo <= -4.0 <= hi


# -- determinism ----------------------------------------------------------------------


def test_repeated_calls_are_identical() -> None:
    log = _team_log()
    one = kpi_panel(log, 5.0, unit="team", kpis=("cycle_time", "n_cases", "n_events"))
    two = kpi_panel(log, 5.0, unit="team", kpis=("cycle_time", "n_cases", "n_events"))
    assert one.names == two.names
    for name in one.names:
        assert np.array_equal(one.column(name), two.column(name))
