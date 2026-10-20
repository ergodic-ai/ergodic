# SPDX-License-Identifier: Apache-2.0
"""Tests for E2: cross-fitting, the orthogonal estimators, and heterogeneous effects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ergodic import EffectTarget, dag, estimate_effect, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import (
    DRLearner,
    KNNClassifier,
    KNNRegressor,
    OutcomeRegression,
    RLearner,
    XLearner,
    candidates,
)
from ergodic.inference._crossfit import _folds, _train_mask


def _hetero(n: int = 4000, seed: int = 0) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """Heterogeneous effect tau(z) = 1 + z, so the ATE is 1.0 and the CATE tracks 1 + z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + z) * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"]), z


def _instrument(n: int = 5000, seed: int = 2) -> tuple[TabularData, MixedGraph]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    a = 0.8 * z + u + rng.normal(size=n)
    y = 1.5 * a + 2.0 * u + rng.normal(size=n)
    data = tabular(pd.DataFrame({"Z": z, "A": a, "Y": y}))
    return data, dag(["Z -> A", "A -> Y", "U -> A", "U -> Y"], latent=["U"])


# -- orthogonal estimators ---------------------------------------------------


@pytest.mark.parametrize("strategy", ["dml", "dr_learner"])
def test_orthogonal_ate_is_analytic(strategy: str) -> None:
    data, graph, _ = _hetero()
    est = estimate_effect(graph, data, "A", "Y", strategy=strategy)
    assert est.kind == "analytic"
    assert abs(est.value - 1.0) < 0.15


@pytest.mark.parametrize("strategy", ["r_learner", "x_learner"])
def test_meta_learner_ate(strategy: str) -> None:
    data, graph, _ = _hetero()
    est = estimate_effect(graph, data, "A", "Y", strategy=strategy, random_state=0, n_boot=80)
    assert abs(est.value - 1.0) < 0.2


def test_stratification_ate_and_att() -> None:
    data, graph, _ = _hetero()
    ate = estimate_effect(
        graph, data, "A", "Y", strategy="stratification", random_state=0, n_boot=80
    )
    assert abs(ate.value - 1.0) < 0.25
    att = estimate_effect(
        graph, data, "A", "Y", strategy="stratification", target="att", random_state=0, n_boot=80
    )
    assert att.target is EffectTarget.ATT


def test_dml_handles_continuous_treatment() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    z = rng.normal(size=n)
    a = 0.6 * z + rng.normal(size=n)  # continuous treatment
    y = 1.5 * a + 2.0 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    est = estimate_effect(graph, data, "A", "Y", strategy="dml")
    assert est.kind == "analytic"
    assert abs(est.value - 1.5) < 0.1


def test_pliv_recovers_linear_iv_analytically() -> None:
    data, graph = _instrument()
    est = estimate_effect(graph, data, "A", "Y", strategy="pliv")
    assert est.target is EffectTarget.LINEAR_IV
    assert est.kind == "analytic"
    assert abs(est.value - 1.5) < 0.2


# -- heterogeneous effects ---------------------------------------------------


@pytest.mark.parametrize("cls", [DRLearner, RLearner, XLearner])
def test_cate_recovers_heterogeneity(cls: type) -> None:
    data, graph, z = _hetero()
    ident = identify_effect(graph, "A", "Y")
    model = cls(random_state=0).fit(ident, data)
    cate = model.cate()
    assert np.corrcoef(cate, 1.0 + z)[0, 1] > 0.6


def test_t_learner_recovers_heterogeneity() -> None:
    data, graph, z = _hetero()
    ident = identify_effect(graph, "A", "Y")
    model = OutcomeRegression(per_arm=True).fit(ident, data)
    assert np.corrcoef(model.cate(), 1.0 + z)[0, 1] > 0.6


def test_cate_predicts_out_of_sample() -> None:
    train, graph, _ = _hetero(seed=0)
    fresh, _, z_fresh = _hetero(seed=5)
    ident = identify_effect(graph, "A", "Y")
    model = DRLearner(random_state=0).fit(ident, train)
    cate = model.cate(fresh)
    assert cate.shape[0] == fresh.n_rows
    assert np.corrcoef(cate, 1.0 + z_fresh)[0, 1] > 0.6


def test_cate_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        DRLearner().cate()


# -- native flexible learner -------------------------------------------------


def test_knn_learner_drops_into_cross_fitting() -> None:
    data, graph, _ = _hetero()
    est = estimate_effect(graph, data, "A", "Y", strategy="dml", learner="knn", random_state=0)
    assert abs(est.value - 1.0) < 0.3


def test_knn_regressor_and_classifier() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    reg = KNNRegressor(k=10).fit(x, x[:, 0])
    assert np.corrcoef(reg.predict(x), x[:, 0])[0, 1] > 0.8
    clf = KNNClassifier(k=10).fit(x, (x[:, 0] > 0.0).astype(float))
    proba = clf.predict_proba(x)
    assert proba.shape == (300, 2)
    assert np.corrcoef(proba[:, 1], (x[:, 0] > 0.0).astype(float))[0, 1] > 0.7


def test_knn_zero_features_fall_back_to_the_training_mean() -> None:
    x = np.empty((100, 0))
    y = np.arange(100, dtype=np.float64)
    reg = KNNRegressor(k=5).fit(x, y)
    assert np.allclose(reg.predict(np.empty((3, 0))), 49.5)
    labels = (np.arange(100) < 30).astype(np.float64)
    clf = KNNClassifier(k=5).fit(np.empty((100, 0)), labels)
    proba = clf.predict_proba(np.empty((2, 0)))
    assert np.allclose(proba[:, 1], 0.3)


# -- cross-fitting folds -----------------------------------------------------


def test_folds_stratify_an_imbalanced_treatment() -> None:
    n = 60
    a = np.zeros(n, dtype=np.float64)
    a[:6] = 1.0
    folds = _folds(n, 5, np.random.default_rng(0), stratify=a)
    assert sorted(np.concatenate(folds).tolist()) == list(range(n))
    for test in folds:
        assert a[test].sum() >= 1.0  # each test fold sees the rare arm
        train = _train_mask(n, test)
        assert a[train].sum() > 0.0
        assert (1.0 - a[train]).sum() > 0.0


def test_r_learner_final_slot_requires_sample_weight() -> None:
    class _NoWeight:
        def fit(self, X: Any, y: Any) -> _NoWeight:
            return self

        def predict(self, X: Any) -> Any:
            return np.zeros(np.asarray(X).shape[0])

    data, graph, _ = _hetero(n=400)
    ident = identify_effect(graph, "A", "Y")
    with pytest.raises(TypeError, match="sample_weight"):
        RLearner({"final": _NoWeight()}, random_state=0).fit(ident, data)


# -- registry ----------------------------------------------------------------


def test_candidates_include_the_e2_estimators() -> None:
    ident = identify_effect(dag(["Z -> A", "Z -> Y", "A -> Y"]), "A", "Y")
    names = [c.name for c in candidates(ident)]
    assert names[0] == "aipw"  # default unchanged
    for expected in ("dml", "dr_learner", "stratification", "r_learner", "x_learner"):
        assert expected in names


def test_e2_aliases_resolve() -> None:
    data, graph, _ = _hetero()
    by_alias = estimate_effect(graph, data, "A", "Y", strategy="dr")
    assert by_alias.estimator == "dr_learner"
    assert abs(by_alias.value - 1.0) < 0.2
