# SPDX-License-Identifier: Apache-2.0
"""API-179: fitted encoding, method aliases, and unavailable honest rankings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from ergodic import cate, dag, estimate_effect, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.identification import Identification
from ergodic.inference import DML, DRLearner, OutcomeRegression, RLearner, XLearner, select_cate


def _world() -> tuple[TabularData, MixedGraph]:
    rng = np.random.default_rng(179)
    n = 400
    c = np.asarray(["a", "b", "c"])[np.arange(n) % 3]
    x = rng.normal(size=n)
    a = (rng.random(n) < 1 / (1 + np.exp(-1.5 - x))).astype(float)
    tau = 1 + 2 * (c == "b") + 4 * (c == "c") + 2 * x
    y = tau * a + x
    data = tabular(pd.DataFrame({"C": pd.Categorical(c), "X": x, "A": a, "Y": y}))
    graph = dag(["C -> A", "C -> Y", "X -> A", "X -> Y", "A -> Y"])
    return data, graph


def _query(categories: list[str], values: list[str]) -> TabularData:
    return tabular(
        pd.DataFrame({"C": pd.Categorical(values, categories=categories), "X": [0.2] * len(values)})
    )


@pytest.mark.parametrize("cls", [OutcomeRegression, DRLearner, RLearner, XLearner, DML])
def test_fitted_cate_preserves_category_design(cls: Any) -> None:
    data, graph = _world()
    kwargs = {"per_arm": True} if cls is OutcomeRegression else {}
    estimator = cls(random_state=179, n_boot=0, **kwargs).fit(
        identify_effect(graph, "A", "Y"), data
    )
    baseline = estimator.cate(_query(["a", "b", "c"], ["a", "b", "c"]), uncertainty=False)
    reordered = estimator.cate(_query(["c", "b", "a"], ["a", "b", "c"]), uncertainty=False)
    np.testing.assert_allclose(reordered.values, baseline.values)
    # The batch declares only one training level, but keeps every fitted dummy.
    absent = estimator.cate(_query(["b"], ["b"]), uncertainty=False)
    np.testing.assert_allclose(absent.values, baseline.values[1:2])
    reversed_columns = tabular(_query(["a", "b", "c"], ["a", "b", "c"]).to_pandas()[["X", "C"]])
    np.testing.assert_allclose(
        estimator.cate(reversed_columns, uncertainty=False).values, baseline.values
    )
    if cls is OutcomeRegression:
        np.testing.assert_allclose(baseline.values, [1.4, 3.4, 5.4], atol=1e-9)


@pytest.mark.parametrize("problem", ["unknown", "missing", "type", "null"])
def test_fitted_cate_rejects_invalid_covariates(problem: str) -> None:
    data, graph = _world()
    estimator = OutcomeRegression(per_arm=True).fit(identify_effect(graph, "A", "Y"), data)
    if problem == "unknown":
        query = _query(["new"], ["new"])
        message = "column 'C'.*outside its declared categories.*new"
    elif problem == "missing":
        query = tabular(pd.DataFrame({"X": [0.2]}))
        message = "missing fitted covariate columns.*C"
    elif problem == "type":
        query = tabular(pd.DataFrame({"X": [0.2], "C": [0.0]}))
        message = "covariate 'C'.*fitted type is 'categorical'"
    else:
        query = _query(["a"], ["a"])
        query = tabular(query.to_pandas().assign(X=np.nan))
        message = "columns.*X.*missing values"
    with pytest.raises(ValueError, match=message):
        estimator.cate(query, uncertainty=False)


def test_named_effect_aliases_fit_the_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    data, graph = _world()
    fitted_per_arm: list[bool] = []
    original_fit = OutcomeRegression.fit

    def fit(
        self: OutcomeRegression, identification: Identification, data: TabularData
    ) -> OutcomeRegression:
        result = original_fit(self, identification, data)
        fitted_per_arm.append(isinstance(self._cate_model(), tuple))
        return result

    monkeypatch.setattr(OutcomeRegression, "fit", fit)
    s = estimate_effect(graph, data, "A", "Y", strategy="s_learner", n_boot=2, random_state=179)
    t = estimate_effect(graph, data, "A", "Y", strategy="T_LEARNER", n_boot=2, random_state=179)
    explicit = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        strategy="outcome_regression",
        per_arm=True,
        n_boot=2,
        random_state=179,
    )
    assert fitted_per_arm == [False, True, True]
    assert t.value == pytest.approx(explicit.value)
    assert abs(s.value - t.value) > 0.3
    per_unit = cate(graph, data, "A", "Y", method="t_learner", uncertainty=False, random_state=179)
    assert t.value == pytest.approx(float(np.mean(per_unit.values)))


@pytest.mark.parametrize("method, per_arm", [("t_learner", False), ("s_learner", True)])
@pytest.mark.parametrize("facade", ["effect", "cate"])
def test_aliases_reject_contradictory_flags(method: str, per_arm: bool, facade: str) -> None:
    data, graph = _world()
    with pytest.raises(ValueError, match="requires per_arm"):
        if facade == "effect":
            estimate_effect(graph, data, "A", "Y", strategy=method, per_arm=per_arm)
        else:
            cate(graph, data, "A", "Y", method=method, per_arm=per_arm)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_failed_ranking_preserves_blp_and_refuses_inference(
    monkeypatch: pytest.MonkeyPatch, nonfinite: bool
) -> None:
    data, graph = _world()
    estimator = OutcomeRegression(per_arm=True, random_state=179).fit(
        identify_effect(graph, "A", "Y"), data
    )

    def fail(*args: Any) -> Any:
        if nonfinite:
            return np.full(len(args[1]), np.nan)
        raise ValueError("synthetic fold fit failure")

    monkeypatch.setattr(estimator, "_predict_cate" if nonfinite else "_fit_cate", fail)
    if nonfinite:
        # Only fold prediction fails; the already-fitted full-sample prediction works.
        original_predict = OutcomeRegression._predict_cate

        def predict(model: Any, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            if len(z) < data.n_rows:
                return np.full(len(z), np.nan)
            return original_predict(estimator, model, z)

        monkeypatch.setattr(estimator, "_predict_cate", predict)
    result = estimator.cate(uncertainty=False)
    assert result._scores is not None
    assert result._scores.tau_hat is None
    assert result._scores.ranking_error is not None
    message = (
        "non-finite predictions" if nonfinite else "fold 1: ValueError: synthetic fold fit failure"
    )
    assert message in result._scores.ranking_error
    summary = result.explain()
    assert summary.blp().frame()["X"]["ci_low"] > 0.0
    methods: list[Callable[[], object]] = [
        summary.gates,
        summary.rate,
        summary.calibration_test,
        summary.clan,
    ]
    for method in methods:
        with pytest.raises(RuntimeError, match=message):
            method()


def test_missing_ranking_cannot_be_laundered_through_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, graph = _world()
    result = cate(graph, data, "A", "Y", uncertainty=False, random_state=179)
    assert result._scores is not None
    broken = replace(
        result, _scores=replace(result._scores, tau_hat=None, ranking_error="fold failed")
    )
    monkeypatch.setattr("ergodic.inference._estimate.cate", lambda *args, **kwargs: broken)
    with pytest.raises(RuntimeError, match="no out-of-fold CATE proxy: fold failed"):
        select_cate(graph, data, "A", "Y", candidates=["dr_learner"], random_state=179)


def test_dml_has_no_ranking_and_preserves_covariate_blp() -> None:
    data, graph = _world()
    result = cate(graph, data, "A", "Y", method="dml", uncertainty=False, random_state=179)
    assert result._scores is not None
    assert result._scores.tau_hat is None
    summary = result.explain()
    with pytest.raises(RuntimeError, match="DML assumes a homogeneous effect"):
        summary.calibration_test()
    assert summary.blp().frame()["X"]["ci_low"] > 0.0
