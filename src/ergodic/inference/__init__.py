# SPDX-License-Identifier: Apache-2.0
"""Causal inference: estimate causal effects from data and a causal model.

An estimator factors into independent pieces: the estimand family (from
:mod:`ergodic.identification`), the strategy that turns it into a number (a
:class:`CausalEstimator`), and the learner in each nuisance slot. ``estimate_effect``
runs identification then estimation and returns an :class:`EffectEstimate`.

    >>> from ergodic import dag, tabular
    >>> from ergodic.inference import estimate_effect
    >>> est = estimate_effect(graph, data, "T", "Y")          # doctest: +SKIP
    >>> est.value, est.interval()                              # doctest: +SKIP

The learner is structurally a sklearn estimator, so ``estimate_effect(..., learner="knn")``
or a slot mapping like ``learner={"outcome": GradientBoostingRegressor()}`` drops
any sklearn or xgboost model into the slots with no adapter. The result is
distribution-shaped: a point estimate with an analytic standard error, or
bootstrap draws when there is no closed form.
"""

from __future__ import annotations

from ._estimate import estimate, estimate_effect
from ._estimator import CausalEstimator, Slot, Task
from ._learner import BayesianLearner, Learner
from ._registry import candidates
from ._result import EffectEstimate
from ._target import EffectTarget
from .estimators import (
    AIPW,
    DML,
    IPW,
    PLIV,
    DRLearner,
    NullEffect,
    OutcomeRegression,
    RLearner,
    Stratification,
    TwoStageLeastSquares,
    TwoStageRegression,
    Wald,
    XLearner,
)
from .learners import (
    OLS,
    KNNClassifier,
    KNNRegressor,
    LogisticRegression,
    PyMCClassifier,
    PyMCLearner,
)

__all__ = [
    "AIPW",
    "DML",
    "IPW",
    "OLS",
    "PLIV",
    "BayesianLearner",
    "CausalEstimator",
    "DRLearner",
    "EffectEstimate",
    "EffectTarget",
    "KNNClassifier",
    "KNNRegressor",
    "Learner",
    "LogisticRegression",
    "NullEffect",
    "OutcomeRegression",
    "PyMCClassifier",
    "PyMCLearner",
    "RLearner",
    "Slot",
    "Stratification",
    "Task",
    "TwoStageLeastSquares",
    "TwoStageRegression",
    "Wald",
    "XLearner",
    "candidates",
    "estimate",
    "estimate_effect",
]
