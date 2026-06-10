# SPDX-License-Identifier: Apache-2.0
"""sklearn interop: the advertised drop-in learner surface, against real sklearn.

The learner protocol claims any sklearn estimator fills a nuisance slot with no
adapter, including pipelines, and that cross-fitting clones it per fold through
``get_params``. These tests exercise that claim against the real library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, estimate_effect, tabular
from ergodic.inference._learner import _fresh

pytest.importorskip("sklearn")

if TYPE_CHECKING:
    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph


def _confounded(n: int = 1500, seed: int = 0) -> tuple[TabularData, MixedGraph]:
    """Binary treatment confounded by z; the true effect is 2.0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * z))).astype(float)
    y = 2.0 * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"])


def test_fresh_clones_a_pipeline() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
    clone: Any = _fresh(pipe)
    assert clone is not pipe
    assert clone.named_steps["ridge"] is not pipe.named_steps["ridge"]
    assert clone.named_steps["ridge"].alpha == 0.5


def test_pipeline_fills_a_slot_end_to_end() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data, graph = _confounded()
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.5))])
    est = estimate_effect(graph, data, "A", "Y", strategy="outcome_regression", learner=pipe)
    assert abs(est.value - 2.0) < 0.25


def test_sklearn_models_fill_the_aipw_slots() -> None:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    data, graph = _confounded()
    est = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        strategy="aipw",
        learner={
            "outcome": GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=0),
            "propensity": GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0),
        },
    )
    assert abs(est.value - 2.0) < 0.4


def test_sklearn_learners_clone_per_fold_in_dml() -> None:
    from sklearn.linear_model import LogisticRegression, Ridge

    data, graph = _confounded()
    est = estimate_effect(
        graph,
        data,
        "A",
        "Y",
        strategy="dml",
        random_state=0,
        learner={"outcome": Ridge(alpha=0.5), "treatment": LogisticRegression(max_iter=200)},
    )
    assert np.isfinite(est.value)
    assert abs(est.value - 2.0) < 0.3
