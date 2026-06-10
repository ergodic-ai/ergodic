# SPDX-License-Identifier: Apache-2.0
"""Tests for effect estimation: the three families, the gates, and the result type."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ergodic import EffectTarget, admg, dag, estimate, estimate_effect, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import (
    OLS,
    candidates,
)

# -- known-truth simulations -------------------------------------------------


def _adjustment(n: int = 4000, seed: int = 0) -> tuple[TabularData, MixedGraph]:
    """Confounded binary treatment, continuous outcome, true ATE = 2.0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-0.8 * z))
    a = (rng.uniform(size=n) < propensity).astype(float)
    y = 2.0 * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"])


def _frontdoor(n: int = 8000, seed: int = 1) -> tuple[TabularData, MixedGraph, float]:
    """Unobserved A-Y confounding, mediator M; true effect = P(M|A=1) - P(M|A=0)."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-u))).astype(float)
    m = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(1.5 * a - 0.5)))).astype(float)
    y = 1.0 * m + 2.0 * u + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "M": m, "Y": y}))
    truth = 1.0 / (1.0 + np.exp(-1.0)) - 1.0 / (1.0 + np.exp(0.5))
    return data, admg(["A -> M", "M -> Y", "A <-> Y"]), truth


def _instrument(n: int = 5000, seed: int = 2) -> tuple[TabularData, MixedGraph]:
    """Instrument Z, confounded continuous treatment; true effect = 1.5."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    a = 0.8 * z + 1.0 * u + rng.normal(size=n)
    y = 1.5 * a + 2.0 * u + rng.normal(size=n)
    data = tabular(pd.DataFrame({"Z": z, "A": a, "Y": y}))
    return data, admg(["Z -> A", "A -> Y", "A <-> Y"])


# -- adjustment family -------------------------------------------------------


@pytest.mark.parametrize("strategy", ["aipw", "outcome_regression", "ipw"])
def test_adjustment_recovers_the_ate(strategy: str) -> None:
    data, graph = _adjustment()
    est = estimate_effect(graph, data, "A", "Y", strategy=strategy, random_state=0, n_boot=200)
    assert est.target is EffectTarget.ATE
    assert abs(est.value - 2.0) < 0.2


def test_default_estimator_is_doubly_robust() -> None:
    data, graph = _adjustment()
    est = estimate_effect(graph, data, "A", "Y")
    assert est.estimator == "aipw"


def test_aipw_ate_is_analytic_with_a_covering_interval() -> None:
    data, graph = _adjustment()
    est = estimate_effect(graph, data, "A", "Y", strategy="aipw")
    assert est.kind == "analytic"
    assert est.draws is None
    se = est.std_error
    assert se is not None and 0.0 < se < 0.1
    low, high = est.interval()
    assert low < 2.0 < high


def test_outcome_regression_is_bootstrap_with_draws() -> None:
    data, graph = _adjustment()
    est = estimate_effect(
        graph, data, "A", "Y", strategy="outcome_regression", random_state=0, n_boot=150
    )
    assert est.kind == "bootstrap"
    assert est.draws is not None
    assert est.draws.shape[0] == 150


def test_att_target() -> None:
    data, graph = _adjustment()
    est = estimate_effect(
        graph, data, "A", "Y", strategy="ipw", target="att", random_state=0, n_boot=200
    )
    assert est.target is EffectTarget.ATT
    assert abs(est.value - 2.0) < 0.3


def test_t_learner_recovers_the_ate() -> None:
    data, graph = _adjustment()
    est = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        strategy="outcome_regression",
        per_arm=True,
        random_state=0,
        n_boot=150,
    )
    assert abs(est.value - 2.0) < 0.2


def test_binary_outcome_runs() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(float)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.8 * a + z)))).astype(bool)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    est = estimate_effect(graph, data, "A", "Y", strategy="aipw")
    assert est.learners["outcome"] == "LogisticRegression"
    assert -1.0 <= est.value <= 1.0


# -- front-door --------------------------------------------------------------


def test_frontdoor_recovers_the_effect() -> None:
    data, graph, truth = _frontdoor()
    est = estimate_effect(graph, data, "A", "Y", random_state=0, n_boot=200)
    assert est.estimator == "two_stage_regression"
    assert abs(est.value - truth) < 0.1


# -- instruments -------------------------------------------------------------


def test_iv_recovers_the_effect_analytically() -> None:
    data, graph = _instrument()
    est = estimate_effect(graph, data, "A", "Y")
    assert est.target is EffectTarget.LATE
    assert est.kind == "analytic"
    assert abs(est.value - 1.5) < 0.15


def test_wald_matches_two_stage_least_squares() -> None:
    data, graph = _instrument()
    tsls = estimate_effect(graph, data, "A", "Y", strategy="2sls")
    wald = estimate_effect(graph, data, "A", "Y", strategy="wald")
    assert abs(tsls.value - wald.value) < 1e-9


# -- the gates ---------------------------------------------------------------


def test_propensity_slot_rejects_a_regressor() -> None:
    data, graph = _adjustment()
    with pytest.raises(TypeError, match="classifier"):
        estimate_effect(graph, data, "A", "Y", strategy="ipw", learner=OLS())


def test_iv_estimator_rejects_an_adjustment_estimand() -> None:
    data, graph = _adjustment()
    with pytest.raises(ValueError, match="consumes"):
        estimate_effect(graph, data, "A", "Y", strategy="2sls")


def test_unknown_estimator_and_learner_raise() -> None:
    data, graph = _adjustment()
    with pytest.raises(ValueError, match="unknown estimator"):
        estimate_effect(graph, data, "A", "Y", strategy="nope")
    with pytest.raises(ValueError, match="unknown learner"):
        estimate_effect(graph, data, "A", "Y", strategy="outcome_regression", learner="nope")


def test_unidentifiable_effect_raises() -> None:
    data = tabular(pd.DataFrame({"A": [0.0, 1.0, 0.0, 1.0], "Y": [1.0, 2.0, 1.5, 2.5]}))
    graph = admg(["A -> Y", "A <-> Y"])  # latent confounding, no adjustment or instrument
    with pytest.raises(ValueError, match="not identifiable"):
        estimate_effect(graph, data, "A", "Y")


def test_non_binary_treatment_is_not_yet_supported() -> None:
    rng = np.random.default_rng(4)
    n = 500
    z = rng.normal(size=n)
    a = rng.normal(size=n)  # continuous, not 0/1
    y = 2.0 * a + z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises(NotImplementedError, match="binary"):
        estimate_effect(graph, data, "A", "Y", strategy="ipw")


# -- the learner swap --------------------------------------------------------


def test_learner_alias_and_mapping() -> None:
    data, graph = _adjustment()
    by_alias = estimate_effect(graph, data, "A", "Y", strategy="outcome_regression", learner="ols")
    assert abs(by_alias.value - 2.0) < 0.2
    by_map = estimate_effect(graph, data, "A", "Y", strategy="aipw", learner={"outcome": OLS()})
    assert by_map.learners["outcome"] == "OLS"


def test_any_fit_predict_object_drops_in() -> None:
    class ConstantMean:
        """A deliberately constant regressor; proves the slot accepts any fit/predict."""

        def __init__(self) -> None:
            self.mean_ = 0.0

        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantMean:
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X: Any) -> Any:
            return np.full(np.asarray(X).shape[0], self.mean_)

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {}

        def set_params(self, **params: Any) -> ConstantMean:
            return self

    data, graph = _adjustment()
    est = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        strategy="outcome_regression",
        learner=ConstantMean(),
        random_state=0,
        n_boot=50,
    )
    # a constant learner predicts the same mean for both arms, so the contrast is ~0
    assert np.isfinite(est.value)
    assert abs(est.value) < 0.3


# -- the facade and the result type ------------------------------------------


def test_estimate_matches_estimate_effect() -> None:
    data, graph = _adjustment()
    ident = identify_effect(graph, "A", "Y")
    from_ident = estimate(ident, data, strategy="aipw")
    from_graph = estimate_effect(graph, data, "A", "Y", strategy="aipw")
    assert from_ident.value == pytest.approx(from_graph.value)


def test_candidates_lists_the_family() -> None:
    _, graph = _adjustment()
    ident = identify_effect(graph, "A", "Y")
    names = [c.name for c in candidates(ident)]
    assert names[0] == "aipw"
    assert "ipw" in names and "outcome_regression" in names


def test_interval_widens_with_level() -> None:
    data, graph = _adjustment()
    est = estimate_effect(graph, data, "A", "Y", strategy="aipw")
    lo90, hi90 = est.interval(0.90)
    lo99, hi99 = est.interval(0.99)
    assert (hi99 - lo99) > (hi90 - lo90)


def test_target_not_offered_raises() -> None:
    data, graph = _instrument()
    with pytest.raises(ValueError, match="reports"):
        estimate_effect(graph, data, "A", "Y", target="att")
