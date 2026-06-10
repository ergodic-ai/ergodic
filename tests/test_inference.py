# SPDX-License-Identifier: Apache-2.0
"""Tests for effect estimation: the three families, the gates, and the result type."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from ergodic import EffectTarget, admg, dag, estimate, estimate_effect, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.identification import Identification
from ergodic.inference import (
    OLS,
    OutcomeRegression,
    candidates,
)
from ergodic.inference._learner import _fresh
from ergodic.inference._result import EffectEstimate, _summary_fields

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ergodic.inference import Learner

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


def _drug_placebo(n: int = 3000, seed: int = 7) -> tuple[TabularData, MixedGraph]:
    """String-coded treatment: the drug's true effect over placebo is +2.0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    drug = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * z))
    a = np.where(drug, "drug", "placebo")
    y = 2.0 * drug.astype(float) + 1.5 * z + rng.normal(size=n)
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
    return data, dag(["Z -> A", "A -> Y", "U -> A", "U -> Y"], latent=["U"])


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


# -- treatment coding --------------------------------------------------------


def test_string_treatment_warns_and_codes_by_level_order() -> None:
    data, graph = _drug_placebo()
    with pytest.warns(UserWarning, match="coded 1 for level 'placebo'"):
        est = estimate_effect(graph, data, "A", "Y", strategy="aipw")
    # 'placebo' sorts after 'drug', so it codes to 1 and the sign flips
    assert abs(est.value + 2.0) < 0.25


def test_treated_picks_the_level_that_codes_to_one() -> None:
    data, graph = _drug_placebo()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        est = estimate_effect(graph, data, "A", "Y", strategy="aipw", treated="drug")
    assert not [w for w in captured if issubclass(w.category, UserWarning)]
    assert abs(est.value - 2.0) < 0.25


def test_treated_att_recovers_the_drug_effect() -> None:
    data, graph = _drug_placebo()
    att = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        strategy="aipw",
        target="att",
        treated="drug",
        random_state=0,
        n_boot=80,
    )
    assert att.target is EffectTarget.ATT
    assert abs(att.value - 2.0) < 0.3


def test_treated_must_be_an_observed_level() -> None:
    data, graph = _drug_placebo(n=200)
    with pytest.raises(ValueError, match="placebo"):
        estimate_effect(graph, data, "A", "Y", treated="aspirin")


def test_numeric_zero_one_treatment_does_not_warn() -> None:
    data, graph = _adjustment(n=800)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimate_effect(graph, data, "A", "Y", strategy="aipw")
    assert not [w for w in captured if issubclass(w.category, UserWarning)]


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
    graph = dag(["A -> Y", "U -> A", "U -> Y"], latent=["U"])  # confounded, no instrument
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


def test_empty_treatment_arm_raises() -> None:
    rng = np.random.default_rng(5)
    n = 200
    z = rng.normal(size=n)
    a = np.ones(n)  # everyone treated, no control arm
    y = 2.0 * a + z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    with pytest.raises(ValueError, match="200 treated and 0 control"):
        estimate_effect(graph, data, "A", "Y", strategy="aipw")


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


def test_fresh_clones_a_meta_estimator_with_nested_params() -> None:
    class _Inner:
        def __init__(self, alpha: float = 1.0) -> None:
            self.alpha = alpha

        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Inner:
            return self

        def predict(self, X: Any) -> Any:
            return np.zeros(np.asarray(X).shape[0])

    class _Meta:
        def __init__(self, estimator: _Inner | None = None) -> None:
            self.estimator = estimator if estimator is not None else _Inner()

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            params: dict[str, Any] = {"estimator": self.estimator}
            if deep:
                # sklearn meta-estimators add nested keys no constructor accepts
                params["estimator__alpha"] = self.estimator.alpha
            return params

        def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Meta:
            return self

        def predict(self, X: Any) -> Any:
            return self.estimator.predict(X)

    original = _Meta(_Inner(alpha=2.0))
    clone = _fresh(original)
    assert isinstance(clone, _Meta)
    assert clone.estimator is not original.estimator
    assert clone.estimator.alpha == 2.0


# -- the bootstrap loop ------------------------------------------------------


class _AlwaysFails(OutcomeRegression):
    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        raise ValueError("degenerate resample")


def test_bootstrap_with_no_surviving_draw_raises() -> None:
    data, graph = _adjustment(n=400)
    ident = identify_effect(graph, "A", "Y")
    model = _AlwaysFails(n_boot=10).fit(ident, data)
    with pytest.raises(RuntimeError, match="no finite draws"):
        model._bootstrap(model._state, EffectTarget.ATE, np.random.default_rng(0))


def test_bootstrap_records_its_draw_counts() -> None:
    data, graph = _adjustment(n=800)
    est = estimate_effect(
        graph, data, "A", "Y", strategy="outcome_regression", random_state=0, n_boot=50
    )
    assert est.diagnostics["n_boot_requested"] == 50.0
    assert est.diagnostics["n_boot_returned"] == 50.0


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


# -- the null family ----------------------------------------------------------


def test_null_strategy_estimates_an_exact_zero() -> None:
    data = tabular(pd.DataFrame({"X": [0.0, 1.0, 0.0, 1.0], "Y": [1.0, 2.0, 1.5, 2.5]}))
    ident = Identification(
        treatment=frozenset({"X"}),
        outcome=frozenset({"Y"}),
        identifiable=True,
        strategy="null",
        estimand="P(Y | do(X)) = P(Y)",
    )
    est = estimate(ident, data)
    assert est.estimator == "null_effect"
    assert est.kind == "analytic"
    assert est.value == 0.0
    assert est.std_error == 0.0


def test_null_strategy_checks_the_columns_exist() -> None:
    data = tabular(pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]}))
    ident = Identification(
        treatment=frozenset({"Q"}),
        outcome=frozenset({"Y"}),
        identifiable=True,
        strategy="null",
        estimand="P(Y | do(Q)) = P(Y)",
    )
    with pytest.raises(ValueError, match="not in the data"):
        estimate(ident, data)


def test_summary_fields_handle_a_point_only_estimate() -> None:
    point_only = EffectEstimate(
        target=EffectTarget.ATE, estimator="ipw", learners={}, n=10, kind="point", _point=1.5
    )
    fields = _summary_fields(point_only)
    assert fields["value"] == 1.5
    assert fields["ci_low"] is None
    assert fields["ci_high"] is None
    analytic = EffectEstimate(
        target=EffectTarget.ATE,
        estimator="ipw",
        learners={},
        n=10,
        kind="analytic",
        _point=1.5,
        _std_error=0.1,
    )
    lo = _summary_fields(analytic)["ci_low"]
    assert lo is not None and lo < 1.5
