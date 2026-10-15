# SPDX-License-Identifier: Apache-2.0
"""The native learners honour the sklearn surface they claim (API-149 regressions).

KNN used to drop ``sample_weight`` silently (the R-learner's final stage passes
it), raise ``AssertionError`` when used before ``fit``, and, like the linear
learners, accept any name in ``set_params``. ``LogisticRegression`` stopped at
``max_iter`` without a word. ``PyMCLearner`` ignored weights too.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ergodic import cate, dag, tabular
from ergodic.inference.learners import (
    OLS,
    ConvergenceWarning,
    KNNClassifier,
    KNNRegressor,
    LogisticRegression,
    PyMCLearner,
)


def test_knn_regressor_honours_sample_weight() -> None:
    x = np.zeros((4, 1))  # one tie group: every row is a neighbor of every query
    y = np.array([0.0, 0.0, 0.0, 10.0])
    plain = KNNRegressor(k=4).fit(x, y)
    assert plain.predict(np.zeros((1, 1)))[0] == pytest.approx(2.5)
    weighted = KNNRegressor(k=4).fit(x, y, sample_weight=np.array([1.0, 1.0, 1.0, 7.0]))
    assert weighted.predict(np.zeros((1, 1)))[0] == pytest.approx(7.0)
    equal = KNNRegressor(k=4).fit(x, y, sample_weight=np.full(4, 3.0))
    assert equal.predict(np.zeros((1, 1)))[0] == pytest.approx(2.5)


def test_knn_classifier_honours_sample_weight() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    y = (rng.uniform(size=50) < 0.5).astype(float)
    weight = np.where(y == 1.0, 9.0, 1.0)
    clf = KNNClassifier(k=50).fit(x, y, sample_weight=weight)
    expected = float(np.sum(weight * y) / np.sum(weight))
    assert clf.predict_proba(x[:1])[0, 1] == pytest.approx(expected)


def test_knn_rejects_bad_weights() -> None:
    x = np.zeros((3, 1))
    with pytest.raises(ValueError, match="entries"):
        KNNRegressor().fit(x, np.zeros(3), sample_weight=np.ones(2))
    with pytest.raises(ValueError, match="non-negative"):
        KNNRegressor().fit(x, np.zeros(3), sample_weight=np.array([1.0, -1.0, 1.0]))


def test_knn_before_fit_raises_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        KNNRegressor().predict(np.zeros((2, 1)))
    with pytest.raises(RuntimeError, match="not fitted"):
        KNNClassifier().predict_proba(np.zeros((2, 1)))


@pytest.mark.parametrize("learner", [KNNRegressor(), KNNClassifier(), OLS(), LogisticRegression()])
def test_set_params_refuses_unknown_names(learner: Any) -> None:
    with pytest.raises(ValueError, match="invalid parameter"):
        learner.set_params(n_neighbours=3)
    name, value = next(iter(learner.get_params().items()))
    assert learner.set_params(**{name: value}) is learner


def test_logistic_regression_warns_when_it_does_not_converge() -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])  # separable: the coefficient runs off
    with pytest.warns(ConvergenceWarning, match="max_iter=3"):
        LogisticRegression(l2=0.0, max_iter=3).fit(x, y)
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(400, 2))
    ys = (rng.uniform(size=400) < 1.0 / (1.0 + np.exp(-xs[:, 0]))).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        LogisticRegression().fit(xs, ys)


class _Mean(PyMCLearner):
    def build(self, X: Any, y: Any) -> Any:  # pragma: no cover - never fitted here
        raise AssertionError("the weighted slot must refuse before any fit")


def test_pymc_learner_refuses_weights() -> None:
    assert PyMCLearner.supports_sample_weight is False
    with pytest.raises(TypeError, match="sample_weight"):
        _Mean().fit(np.zeros((3, 1)), np.zeros(3), sample_weight=np.ones(3))
    rng = np.random.default_rng(5)
    z = rng.normal(size=120)
    t = (rng.normal(size=120) + z > 0).astype(float)
    y = 2.0 * t + z + rng.normal(size=120)
    data = tabular(pd.DataFrame({"Z": z, "T": t, "Y": y}), types={"T": "categorical"})
    graph = dag(["Z -> T", "Z -> Y", "T -> Y"])
    with pytest.raises(TypeError, match="sample_weight"):
        cate(graph, data, "T", "Y", method="r_learner", learner={"final": _Mean()})


def test_r_learner_with_a_knn_final_uses_the_weights() -> None:
    """The R-learner runs end to end with a weighted KNN final stage."""
    rng = np.random.default_rng(7)
    n = 600
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.5 * z))).astype(float)
    y = (1.0 + z) * t + z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"Z": z, "T": t, "Y": y}), types={"T": "categorical"})
    graph = dag(["Z -> T", "Z -> Y", "T -> Y"])
    res = cate(
        graph,
        data,
        "T",
        "Y",
        method="r_learner",
        learner={"final": KNNRegressor(k=50)},
        random_state=0,
        uncertainty=False,
    )
    assert np.corrcoef(res.values, 1.0 + z)[0, 1] > 0.5
