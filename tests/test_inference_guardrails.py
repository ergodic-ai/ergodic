# SPDX-License-Identifier: Apache-2.0
"""API-180: honest AIPW splitting, estimated overlap and qualified IV targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ergodic import EffectTarget, dag, estimate_effect, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import AIPW
from ergodic.inference._overlap import _propensity_diagnostics
from ergodic.inference._scores import dr_scores


def _fixture(seed: int = 7, n: int = 400) -> tuple[MixedGraph, TabularData]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 5))
    a = rng.binomial(1, 1 / (1 + np.exp(-z[:, 0]))).astype(float)
    y = 2 * a + z[:, 0] + rng.normal(size=n)
    graph = dag([f"Z{i} -> A" for i in range(5)] + [f"Z{i} -> Y" for i in range(5)] + ["A -> Y"])
    return graph, tabular({**{f"Z{i}": z[:, i] for i in range(5)}, "A": a, "Y": y})


def test_default_aipw_is_the_shared_crossfit_ate_and_score_se() -> None:
    graph, data = _fixture()
    scores = dr_scores(identify_effect(graph, "A", "Y"), data, folds=5, random_state=0)
    estimate = estimate_effect(graph, data, "A", "Y")
    assert estimate.value == pytest.approx(float(np.mean(scores.scores)))
    assert estimate.std_error == pytest.approx(
        float(np.std(scores.scores, ddof=1) / np.sqrt(data.n_rows))
    )
    assert estimate.diagnostics["cross_fitted"] == 1
    assert estimate.diagnostics["n_folds"] == 5
    assert estimate.diagnostics["crossfit_seed"] == 0
    assert estimate.identification == identify_effect(graph, "A", "Y")
    assert "arbitrary learners" in " ".join(estimate.assumptions)


def test_in_sample_is_explicit_and_different_from_crossfit() -> None:
    graph, data = _fixture()
    old = estimate_effect(graph, data, "A", "Y", mode="in_sample")
    new = estimate_effect(graph, data, "A", "Y", mode="cross_fitted", random_state=9)
    again = estimate_effect(graph, data, "A", "Y", mode="cross_fitted", random_state=9)
    assert old.diagnostics["cross_fitted"] == 0
    assert old.value != new.value
    assert new.value == again.value
    assert new.std_error == again.std_error


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_folds": 1},
        {"n_folds": 2.5},
        {"n_folds": True},
        {"clip": 0},
        {"clip": 0.5},
        {"mode": "magic"},
        {"overlap": "trim"},
    ],
)
def test_aipw_configuration_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AIPW(**kwargs)


def test_crossfit_refuses_one_small_arm_and_too_many_folds() -> None:
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    data = tabular({"Z": np.arange(10.0), "A": np.array([1.0] + [0.0] * 9), "Y": np.arange(10.0)})
    with pytest.raises(ValueError, match="two observations"):
        estimate_effect(graph, data, "A", "Y")
    graph, data = _fixture(n=20)
    with pytest.raises(ValueError, match="n_rows"):
        estimate_effect(graph, data, "A", "Y", n_folds=21)


def test_overlap_diagnostics_use_unclipped_predictions_and_actual_weights() -> None:
    a = np.array([0.0, 0, 0, 1, 1, 1])
    raw = np.array([0.0, 0.3, 0.4, 0.6, 0.7, 1.0])
    with pytest.warns(UserWarning, match="No rows were trimmed"):
        d = _propensity_diagnostics(a, raw, clip=0.01)
    assert d["propensity_min"] == 0
    assert d["propensity_max"] == 1
    assert d["propensity_clipped_fraction"] == pytest.approx(2 / 6)
    w = np.array([1 / 0.6, 1 / 0.7, 1 / 0.99])
    assert d["treated_effective_n"] == pytest.approx(float(w.sum() ** 2 / np.sum(w**2)))
    assert d["treated_weight_max"] == pytest.approx(float(w.max()))
    assert d["propensity_q05"] == pytest.approx(float(np.quantile(raw, 0.05)))
    with pytest.raises(ValueError, match="poor estimated overlap"):
        _propensity_diagnostics(a, raw, clip=0.01, overlap="raise")


def test_degenerate_propensities_refuse_by_default_and_override_is_explicit() -> None:
    a = np.array([0.0, 0, 1, 1])
    with pytest.raises(ValueError, match=r"100\.0% clipped"):
        _propensity_diagnostics(a, a, clip=0.001)
    assert _propensity_diagnostics(a, a, clip=0.001, overlap="ignore")["overlap_severe"] == 1
    with pytest.raises(ValueError, match="finite probabilities"):
        _propensity_diagnostics(a, np.full(4, np.nan), clip=0.001, overlap="ignore")


def test_original_interpolating_tree_fixture_is_refused() -> None:
    tree = pytest.importorskip("sklearn.tree")
    graph, data = _fixture(seed=0)
    with pytest.raises(ValueError, match=r"100\.0% clipped"):
        estimate_effect(
            graph,
            data,
            "A",
            "Y",
            learner={
                "outcome": tree.DecisionTreeRegressor(random_state=0),
                "propensity": tree.DecisionTreeClassifier(random_state=0),
            },
        )


def test_deterministic_assignment_does_not_get_an_unqualified_interval() -> None:
    rng = np.random.default_rng(4)
    z = rng.normal(size=400)
    a = (z > 0).astype(float)
    data = tabular({"Z": z, "A": a, "Y": 2 * a + z + rng.normal(size=400)})
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises(ValueError, match="poor estimated overlap"):
        estimate_effect(graph, data, "A", "Y")


def _binary_iv() -> tuple[MixedGraph, TabularData]:
    rng = np.random.default_rng(10)
    z = rng.binomial(1, 0.5, 2000)
    u = rng.uniform(size=2000)
    a = (u < 0.2 + 0.6 * z).astype(float)
    return dag(["Z -> A", "A -> Y", "U -> A", "U -> Y"], latent=["U"]), tabular(
        {"Z": z, "A": a, "Y": 2 * a + u + rng.normal(size=2000)}
    )


@pytest.mark.parametrize("strategy", ["2sls", "wald", "pliv"])
def test_generic_iv_default_preserves_identification_and_strength(strategy: str) -> None:
    graph, data = _binary_iv()
    result = estimate_effect(graph, data, "A", "Y", strategy=strategy, random_state=0)
    assert result.target is EffectTarget.LINEAR_IV
    assert result.identification == identify_effect(graph, "A", "Y")
    assert result.diagnostics["first_stage_f"] > 10
    assert 0 < result.diagnostics["first_stage_partial_r2"] < 1
    assert "no automatic" in " ".join(result.assumptions)
    assert result.value == pytest.approx(2, abs=0.15)


def test_late_requires_explicit_assumptions_and_binary_design() -> None:
    graph, data = _binary_iv()
    with pytest.raises(ValueError, match="late_assumptions=True"):
        estimate_effect(graph, data, "A", "Y", target="late")
    result = estimate_effect(graph, data, "A", "Y", target="late", late_assumptions=True)
    assert result.target is EffectTarget.LATE
    assert "monotonicity" in " ".join(result.assumptions)
    rng = np.random.default_rng(11)
    continuous = tabular(
        {"Z": rng.normal(size=500), "A": rng.normal(size=500), "Y": rng.normal(size=500)}
    )
    with (
        pytest.warns(UserWarning, match="weak first stage"),
        pytest.raises(ValueError, match="binary treatment"),
    ):
        estimate_effect(graph, continuous, "A", "Y", target="late", late_assumptions=True)
    with pytest.raises(ValueError, match="reports"):
        estimate_effect(
            graph, data, "A", "Y", strategy="pliv", target="late", late_assumptions=True
        )


def test_small_stratified_crossfit_skips_empty_folds_for_sklearn() -> None:
    linear = pytest.importorskip("sklearn.linear_model")
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    data = tabular(
        {"A": np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]), "Z": np.arange(6.0), "Y": np.arange(6.0)}
    )
    result = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        overlap="ignore",
        learner={"outcome": linear.LinearRegression(), "propensity": linear.LogisticRegression()},
    )
    assert result.diagnostics["n_folds"] == 3
    assert result.diagnostics["n_folds_requested"] == 5
    assert np.isfinite(result.value)


def test_first_stage_strength_is_invariant_to_treatment_units() -> None:
    from ergodic.inference.estimators._iv import _first_stage_diagnostics

    rng = np.random.default_rng(3)
    z = rng.normal(size=(200, 1))
    a = z[:, 0] + rng.normal(size=200)
    w = np.empty((200, 0))
    original = _first_stage_diagnostics(a, z, w)
    rescaled = _first_stage_diagnostics(a * 1e-10, z, w)
    assert rescaled["first_stage_f"] == pytest.approx(original["first_stage_f"])
    assert rescaled["first_stage_partial_r2"] == pytest.approx(original["first_stage_partial_r2"])
