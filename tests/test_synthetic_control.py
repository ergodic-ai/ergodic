# SPDX-License-Identifier: Apache-2.0
"""Tests for synthetic control: weight recovery, placebo inference, and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import EffectTarget
from ergodic.data import panel
from ergodic.inference.estimators._synthetic_control import (
    SyntheticControlEffect,
    synthetic_control,
)

# -- a known-truth factor world ------------------------------------------------

N_DONORS = 25
N_PERIODS = 40
POST_FROM = 30


def _factor_world(effect: float = 5.0, treated_noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """25 donors with a factor structure; treated = 0.6 d01 + 0.4 d02 (+ effect post)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_PERIODS, dtype=np.float64)
    lam = 0.02 * t + 0.5 * np.sin(2 * np.pi * t / 20)
    f = np.cos(2 * np.pi * t / 9)
    mu = 0.5 * rng.normal(size=N_DONORS)
    phi = rng.uniform(-1.0, 1.0, size=N_DONORS)
    donor_paths = (
        mu[:, None]
        + lam[None, :]
        + phi[:, None] * f[None, :]
        + 0.1 * rng.normal(size=(N_DONORS, N_PERIODS))
    )
    treated_path = (
        0.6 * donor_paths[0]
        + 0.4 * donor_paths[1]
        + treated_noise * rng.normal(size=N_PERIODS)
        + effect * (t >= POST_FROM)
    )
    names = [f"d{j + 1:02d}" for j in range(N_DONORS)] + ["treated"]
    ymat = np.vstack([donor_paths, treated_path[None, :]])
    return pd.DataFrame(
        {
            "unit": np.repeat(np.array(names, dtype=object), N_PERIODS),
            "period": np.tile(np.arange(N_PERIODS), len(names)),
            "y": ymat.reshape(-1),
        }
    )


def _with_treatment_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["D"] = ((out["unit"] == "treated") & (out["period"] >= POST_FROM)).astype(int)
    return out


D_TYPES = {"D": "continuous"}  # keep the 0/1 column numeric, not inferred categorical


# -- recovery ------------------------------------------------------------------


def test_recovers_planted_weights_and_effect() -> None:
    data = panel(_factor_world(), entity="unit", time="period")
    est = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    assert isinstance(est, SyntheticControlEffect)
    assert 0.45 < est.weights["d01"] < 0.75
    assert 0.25 < est.weights["d02"] < 0.55
    others = [w for d, w in est.weights.items() if d not in ("d01", "d02")]
    assert all(w < 0.08 for w in others)
    assert abs(est.value - 5.0) < 0.6
    assert est.p_value is not None and est.p_value <= 0.12
    assert est.pre_rmspe < 0.5
    lo, hi = est.interval(0.95)
    assert lo <= 5.0 <= hi


def test_result_record_shape() -> None:
    data = panel(_factor_world(), entity="unit", time="period")
    est = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    assert est.target is EffectTarget.ATT
    assert est.estimator == "synthetic_control"
    assert est.kind == "analytic"
    assert est.std_error is None
    assert est.learners == {}
    assert est.n == (1 + N_DONORS) * N_PERIODS
    assert est.treated_unit == "treated"
    assert est.at == POST_FROM
    assert len(est.periods) == N_PERIODS
    assert len(est.observed) == len(est.synthetic) == len(est.gaps) == N_PERIODS
    assert np.isclose(float(np.mean(est.gaps[POST_FROM:])), est.value)
    assert est.diagnostics["n_donors"] == N_DONORS
    assert est.diagnostics["n_pre"] == POST_FROM
    assert est.diagnostics["n_post"] == N_PERIODS - POST_FROM
    assert est.diagnostics["n_placebos"] == N_DONORS
    assert est.diagnostics["n_dropped_donors"] == 0
    assert est.diagnostics["pre_rmspe"] == est.pre_rmspe
    assert set(est.placebo_gaps) == set(est.weights)
    assert est.design is not None
    assert est.design.strategy == "synthetic_control"
    assert est.design.treatment is None
    assert est.design.estimand.startswith("ATT")


def test_null_world_has_no_effect_and_a_large_p_value() -> None:
    data = panel(_factor_world(effect=0.0, treated_noise=0.1, seed=1), entity="unit", time="period")
    est = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    assert abs(est.value) < 0.6
    assert est.p_value is not None and est.p_value > 0.3


def test_named_donors_are_respected() -> None:
    data = panel(_factor_world(), entity="unit", time="period")
    named = [f"d{j:02d}" for j in range(1, 9)]
    est = synthetic_control(data, "y", treated="treated", at=POST_FROM, donors=named)
    assert set(est.weights) == set(named)
    assert "treated" not in est.weights
    assert "treated" not in est.placebo_gaps


# -- the treatment-column path ---------------------------------------------------


def test_treatment_column_matches_the_explicit_pair() -> None:
    data = panel(
        _with_treatment_column(_factor_world()), entity="unit", time="period", types=D_TYPES
    )
    by_column = synthetic_control(data, "y", treatment="D")
    explicit = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    assert by_column.treated_unit == "treated"
    assert by_column.at == POST_FROM
    assert by_column.value == explicit.value
    assert by_column.weights == explicit.weights
    assert by_column.p_value == explicit.p_value
    assert by_column.design is not None and by_column.design.treatment == "D"
    assert abs(by_column.value - 5.0) < 0.6


def test_two_treated_units_raise() -> None:
    frame = _with_treatment_column(_factor_world())
    frame.loc[(frame["unit"] == "d05") & (frame["period"] >= 35), "D"] = 1
    data = panel(frame, entity="unit", time="period", types=D_TYPES)
    with pytest.raises(ValueError, match="one call per treated unit"):
        synthetic_control(data, "y", treatment="D")


def test_non_absorbing_treatment_raises() -> None:
    frame = _with_treatment_column(_factor_world())
    frame.loc[(frame["unit"] == "d03") & (frame["period"] == 5), "D"] = 1
    data = panel(frame, entity="unit", time="period", types=D_TYPES)
    with pytest.raises(ValueError, match="absorbing"):
        synthetic_control(data, "y", treatment="D")


def test_exactly_one_of_treatment_or_the_pair() -> None:
    data = panel(
        _with_treatment_column(_factor_world()), entity="unit", time="period", types=D_TYPES
    )
    with pytest.raises(ValueError, match="exactly one"):
        synthetic_control(data, "y")
    with pytest.raises(ValueError, match="exactly one"):
        synthetic_control(data, "y", treated="treated")
    with pytest.raises(ValueError, match="exactly one"):
        synthetic_control(data, "y", treatment="D", treated="treated", at=POST_FROM)


# -- validation ------------------------------------------------------------------


def test_validation_raises_on_bad_inputs() -> None:
    data = panel(_factor_world(), entity="unit", time="period")
    with pytest.raises(ValueError, match="not an entity"):
        synthetic_control(data, "y", treated="nope", at=POST_FROM)
    with pytest.raises(ValueError, match="not one of the panel's periods"):
        synthetic_control(data, "y", treated="treated", at=29.5)
    with pytest.raises(ValueError, match="at least 2 pre periods"):
        synthetic_control(data, "y", treated="treated", at=1)
    with pytest.raises(ValueError, match="not a variable"):
        synthetic_control(data, "z", treated="treated", at=POST_FROM)


def test_incomplete_donor_is_dropped_with_a_warning() -> None:
    frame = _factor_world()
    frame = frame[~((frame["unit"] == "d10") & (frame["period"] == 5))]
    data = panel(frame, entity="unit", time="period")
    with pytest.warns(UserWarning, match="dropped 1 donor"):
        est = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    assert "d10" not in est.weights
    assert est.diagnostics["n_dropped_donors"] == 1
    assert est.diagnostics["n_donors"] == N_DONORS - 1


def test_nan_outcome_raises() -> None:
    frame = _factor_world()
    frame.loc[(frame["unit"] == "d04") & (frame["period"] == 3), "y"] = np.nan
    data = panel(frame, entity="unit", time="period")
    with pytest.raises(ValueError, match="missing values"):
        synthetic_control(data, "y", treated="treated", at=POST_FROM)


# -- determinism -------------------------------------------------------------------


def test_repeated_calls_are_identical() -> None:
    data = panel(_factor_world(), entity="unit", time="period")
    first = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    second = synthetic_control(data, "y", treated="treated", at=POST_FROM)
    assert first.value == second.value
    assert first.weights == second.weights
    assert first.p_value == second.p_value
    assert first.observed == second.observed
    assert first.synthetic == second.synthetic
