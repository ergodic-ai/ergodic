# SPDX-License-Identifier: Apache-2.0
"""Tests for difference-in-differences: canonical, TWFE, group-time, and event study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic.data import PanelData, panel
from ergodic.inference._target import EffectTarget
from ergodic.inference.estimators._did import DiDEffect, did, event_study

# -- known-truth simulations ---------------------------------------------------


def _panel(frame: pd.DataFrame) -> PanelData:
    # declare a numeric treatment continuous so schema inference stays quiet
    types = {"d": "continuous"} if frame["d"].dtype.kind in "biuf" else None
    return panel(frame, entity="unit", time="t", types=types)


def _single_cohort(seed: int = 0) -> pd.DataFrame:
    """120 units over periods 0..11; 40 adopt at period 6 with constant effect 3.0.

    Treated units sit 2.0 higher in level (selection on level), so the naive
    post-period cross-section is wrong by about 2 while the within-unit
    difference-in-differences is not.
    """
    rng = np.random.default_rng(seed)
    n_treated, n_never, n_periods, adopt_at = 40, 80, 12, 6
    n_units = n_treated + n_never
    is_treated = np.arange(n_units) < n_treated
    alpha = rng.normal(size=n_units) + 2.0 * is_treated
    lam = rng.normal(scale=0.5, size=n_periods)
    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    d = (is_treated[unit] & (t >= adopt_at)).astype(np.int64)
    y = alpha[unit] + lam[t] + 3.0 * d + rng.normal(scale=0.5, size=unit.size)
    return pd.DataFrame({"unit": unit, "t": t, "d": d, "y": y})


def _staggered(seed: int = 0, n_never: int = 40) -> pd.DataFrame:
    """Cohorts adopting at periods 4 and 8 (40 units each) plus never-treated units.

    The effect grows with time since adoption, tau(e) = 1.0 + 0.5 * e, which is
    the case where static TWFE mixes cells with negative weights and misses.
    The same selection-on-level confound as the single-cohort world applies.
    """
    rng = np.random.default_rng(seed)
    n_periods = 12
    adoption = np.asarray([4] * 40 + [8] * 40 + [-1] * n_never)
    n_units = adoption.size
    alpha = rng.normal(size=n_units) + 2.0 * (adoption >= 0)
    lam = rng.normal(scale=0.5, size=n_periods)
    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    adopt_at = adoption[unit]
    on = (adopt_at >= 0) & (t >= adopt_at)
    event = np.where(on, t - adopt_at, 0)
    d = on.astype(np.int64)
    y = alpha[unit] + lam[t] + (1.0 + 0.5 * event) * d + rng.normal(scale=0.5, size=unit.size)
    return pd.DataFrame({"unit": unit, "t": t, "d": d, "y": y})


def _trend_confounded(seed: int = 0) -> pd.DataFrame:
    """Single cohort whose units trend at 0.3 * x per period, with treated x higher.

    Without controls the canonical contrast absorbs the faster treated trend
    (bias about 0.3 * 1.0 * 6 = 1.8); the pre-period mean of x removes it.
    """
    rng = np.random.default_rng(seed)
    n_treated, n_never, n_periods, adopt_at = 40, 80, 12, 6
    n_units = n_treated + n_never
    is_treated = np.arange(n_units) < n_treated
    alpha = rng.normal(size=n_units)
    x = rng.normal(size=n_units) + 1.0 * is_treated
    lam = rng.normal(scale=0.5, size=n_periods)
    unit = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    d = (is_treated[unit] & (t >= adopt_at)).astype(np.int64)
    y = alpha[unit] + lam[t] + 0.3 * x[unit] * t + 3.0 * d + rng.normal(scale=0.5, size=unit.size)
    return pd.DataFrame({"unit": unit, "t": t, "d": d, "y": y, "x": x[unit]})


def _true_overall(groups: dict[int, int], n_periods: int) -> float:
    """The estimand the group-time aggregation targets, from the planted tau."""
    total = sum(groups.values())
    overall = 0.0
    for g, size in groups.items():
        taus = [1.0 + 0.5 * (t - g) for t in range(g, n_periods)]
        overall += (size / total) * float(np.mean(taus))
    return overall


# -- canonical -------------------------------------------------------------------


def test_canonical_recovers_where_the_naive_contrast_fails() -> None:
    frame = _single_cohort()
    est = did(_panel(frame), "y", "d", n_boot=200, random_state=0)
    assert isinstance(est, DiDEffect)
    assert est.flavor == "canonical"
    assert est.kind == "bootstrap"
    assert est.target is EffectTarget.ATT
    assert abs(est.value - 3.0) < 0.3
    lo, hi = est.interval(0.95)
    assert lo <= 3.0 <= hi
    post = frame[frame["t"] >= 6]
    naive = float(post.loc[post["d"] == 1, "y"].mean()) - float(
        post.loc[post["d"] == 0, "y"].mean()
    )
    assert abs(naive - 3.0) > 1.5  # selection on level fools the cross-section


def test_canonical_bootstrap_se_tracks_the_analytic_se() -> None:
    est = did(_panel(_single_cohort()), "y", "d", n_boot=200, random_state=0)
    analytic = est.diagnostics["analytic_se"]
    assert est.std_error is not None
    assert abs(est.std_error - analytic) / analytic < 0.4


def test_design_and_diagnostics_ride_along() -> None:
    est = did(_panel(_single_cohort()), "y", "d", n_boot=50, random_state=0)
    assert est.estimator == "did"
    assert est.learners == {}
    assert est.n == 120 * 12
    assert est.comparison == "never_treated"
    assert est.diagnostics["n_units"] == 120.0
    assert est.diagnostics["n_treated_units"] == 40.0
    assert est.diagnostics["n_comparison_units"] == 80.0
    assert est.diagnostics["n_periods"] == 12.0
    assert est.diagnostics["n_dropped_units"] == 0.0
    assert est.design is not None
    assert est.design.strategy == "did"
    assert est.design.outcome == "y"
    assert est.design.treatment == "d"
    assert any("parallel trends" in a for a in est.design.assumptions)
    assert "never_treated" in est.design.estimand


# -- validation --------------------------------------------------------------------


def test_non_absorbing_treatment_raises() -> None:
    frame = _single_cohort()
    flip = (frame["unit"] == 0) & (frame["t"] == 8)
    frame.loc[flip, "d"] = 0  # 1 at t=6,7 then back to 0: not absorbing
    with pytest.raises(ValueError, match="absorbing"):
        did(_panel(frame), "y", "d", n_boot=10, random_state=0)


def test_string_coded_treatment_raises() -> None:
    frame = _single_cohort()
    frame["d"] = np.where(frame["d"] == 1, "yes", "no")
    with pytest.raises(ValueError, match="0/1"):
        did(_panel(frame), "y", "d", n_boot=10, random_state=0)


def test_all_treated_panel_raises() -> None:
    frame = _single_cohort()
    frame = frame[frame["unit"] < 40]  # only the treated cohort remains
    with pytest.raises(ValueError, match="comparison"):
        did(_panel(frame), "y", "d", n_boot=10, random_state=0)


def test_input_validation() -> None:
    data = _panel(_single_cohort())
    with pytest.raises(ValueError, match="not a variable"):
        did(data, "nope", "d", n_boot=10)
    with pytest.raises(ValueError, match="exclude"):
        did(data, "y", "d", controls=["d"], n_boot=10)
    with pytest.raises(ValueError, match="comparison"):
        did(data, "y", "d", comparison="whatever", n_boot=10)
    with pytest.raises(ValueError, match="strategy"):
        did(data, "y", "d", strategy="magic", n_boot=10)
    with pytest.raises(ValueError, match="n_boot"):
        did(data, "y", "d", n_boot=0)
    with pytest.raises(ValueError, match="one adoption period"):
        did(_panel(_staggered()), "y", "d", strategy="canonical", n_boot=10)
    frame = _single_cohort()
    frame.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        did(_panel(frame), "y", "d", n_boot=10)


# -- staggered adoption --------------------------------------------------------------


def test_group_time_recovers_where_static_twfe_fails() -> None:
    data = _panel(_staggered())
    truth = _true_overall({4: 40, 8: 40}, 12)
    est = did(data, "y", "d", n_boot=200, random_state=0)
    assert est.flavor == "group_time"  # auto picks group_time for staggered adoption
    assert abs(est.value - truth) < 0.3
    assert est.by_event_time == ()
    cells = est.by_group_time
    assert cells
    assert sorted({c.group for c in cells}) == [4, 8]
    assert all(c.event_time != -1 for c in cells)  # the base period is never a cell
    assert list(cells) == sorted(cells, key=lambda c: (c.group, c.period))
    twfe = did(data, "y", "d", strategy="twfe", n_boot=200, random_state=0)
    assert twfe.flavor == "twfe"
    assert abs(twfe.value - truth) > 0.35  # negative Goodman-Bacon weights bite


def test_event_study_leads_cover_zero_and_lags_track_tau() -> None:
    data = _panel(_staggered())
    est = event_study(data, "y", "d", n_boot=200, random_state=0)
    assert est.flavor == "group_time"
    by_event = {entry.event_time: entry for entry in est.by_event_time}
    assert -1 not in by_event  # the reference period is omitted
    leads = [entry for entry in est.by_event_time if entry.event_time <= -2]
    assert leads
    for entry in leads:
        lo, hi = entry.interval
        assert lo <= 0.0 <= hi, f"lead at e={entry.event_time} excludes zero"
    for event in (0, 1, 2):
        truth = 1.0 + 0.5 * event
        assert abs(by_event[event].estimate - truth) < 0.45
        assert by_event[event].n_treated > 0
        assert by_event[event].std_error is not None


def test_event_study_window_clips_the_reported_event_times() -> None:
    data = _panel(_staggered())
    est = event_study(data, "y", "d", leads=2, lags=1, n_boot=50, random_state=0)
    assert [entry.event_time for entry in est.by_event_time] == [-2, 0, 1]
    assert est.by_group_time  # the cells still ride along in full


def test_not_yet_treated_comparison_when_nobody_stays_untreated() -> None:
    frame = _staggered(n_never=0)  # two cohorts, no never-treated pool
    data = _panel(frame)
    with pytest.raises(ValueError, match="never"):
        did(data, "y", "d", n_boot=10, random_state=0)
    est = did(data, "y", "d", comparison="not_yet_treated", n_boot=200, random_state=0)
    assert est.comparison == "not_yet_treated"
    # only the early cohort has cells, and only while the late cohort is untreated
    assert {c.group for c in est.by_group_time} == {4}
    assert all(c.period <= 7 for c in est.by_group_time)
    truth = float(np.mean([1.0 + 0.5 * e for e in range(4)]))  # e = 0..3 have a pool
    assert abs(est.value - truth) < 0.3


# -- controls ------------------------------------------------------------------------


def test_controls_absorb_a_trend_confounder() -> None:
    data = _panel(_trend_confounded())
    plain = did(data, "y", "d", n_boot=50, random_state=0)
    adjusted = did(data, "y", "d", controls=["x"], n_boot=200, random_state=0)
    assert adjusted.flavor == "canonical"
    assert abs(plain.value - 3.0) > 1.0  # x drives both trends and adoption
    assert abs(adjusted.value - 3.0) < 0.35
    assert "analytic_se" not in adjusted.diagnostics  # analytic SE is no-controls only


def test_controls_smoke_on_group_time() -> None:
    frame = _staggered()
    rng = np.random.default_rng(3)
    frame["x"] = rng.normal(size=len(frame))
    est = did(_panel(frame), "y", "d", controls=["x"], n_boot=50, random_state=0)
    truth = _true_overall({4: 40, 8: 40}, 12)
    assert est.flavor == "group_time"
    assert abs(est.value - truth) < 0.35


# -- unbalanced panels ----------------------------------------------------------------


def test_unbalanced_panel_recovers_and_warns_on_a_dropped_unit() -> None:
    frame = _single_cohort()
    keep = ~((frame["unit"] == 0) & (frame["t"] < 6))  # unit 0 loses every pre row
    frame = frame[keep]
    rng = np.random.default_rng(42)
    # thin ~5% of the remaining rows, sparing period 6 so every treated unit's
    # first observed treated period stays the common adoption period
    drop = (
        (rng.random(len(frame)) < 0.05)
        & (frame["unit"].to_numpy() != 0)
        & (frame["t"].to_numpy() != 6)
    )
    frame = frame.loc[~drop]
    data = _panel(frame)
    assert not data.is_balanced
    with pytest.warns(UserWarning, match="lacking a pre or post"):
        est = did(data, "y", "d", n_boot=200, random_state=0)
    assert abs(est.value - 3.0) < 0.4
    assert est.diagnostics["n_dropped_units"] == 1.0
    assert est.n < 120 * 12


# -- determinism ----------------------------------------------------------------------


def test_same_seed_gives_identical_value_and_draws() -> None:
    data = _panel(_staggered())
    one = did(data, "y", "d", n_boot=50, random_state=0)
    two = did(data, "y", "d", n_boot=50, random_state=0)
    assert one.value == two.value
    assert one.draws is not None and two.draws is not None
    assert np.array_equal(one.draws, two.draws)
