# SPDX-License-Identifier: Apache-2.0
"""Resolve an estimand and a request into a concrete estimator.

The compatibility gate from the graph to the estimator: an estimator consumes a
family, and ``Identification.strategy`` says which family holds. The first entry
per family is the default (the doubly-robust one for adjustment).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._estimator import ADJUSTMENT, FRONTDOOR, IV, CausalEstimator
from ._target import EffectTarget
from .estimators import (
    AIPW,
    DML,
    IPW,
    PLIV,
    DRLearner,
    OutcomeRegression,
    RLearner,
    Stratification,
    TwoStageLeastSquares,
    TwoStageRegression,
    Wald,
    XLearner,
)

if TYPE_CHECKING:
    from ..identification import Identification

_BY_FAMILY: dict[str, list[type[CausalEstimator]]] = {
    ADJUSTMENT: [AIPW, DML, DRLearner, OutcomeRegression, IPW, Stratification, RLearner, XLearner],
    FRONTDOOR: [TwoStageRegression],
    IV: [TwoStageLeastSquares, PLIV, Wald],
}

_BY_NAME: dict[str, type[CausalEstimator]] = {
    "aipw": AIPW,
    "augmented_ipw": AIPW,
    "doubly_robust": AIPW,
    "dml": DML,
    "double_ml": DML,
    "partialling_out": DML,
    "dr_learner": DRLearner,
    "drlearner": DRLearner,
    "dr": DRLearner,
    "outcome_regression": OutcomeRegression,
    "regression": OutcomeRegression,
    "g_formula": OutcomeRegression,
    "s_learner": OutcomeRegression,
    "t_learner": OutcomeRegression,
    "ipw": IPW,
    "inverse_propensity_weighting": IPW,
    "stratification": Stratification,
    "strata": Stratification,
    "r_learner": RLearner,
    "rlearner": RLearner,
    "x_learner": XLearner,
    "xlearner": XLearner,
    "two_stage_regression": TwoStageRegression,
    "frontdoor": TwoStageRegression,
    "front_door": TwoStageRegression,
    "two_stage_least_squares": TwoStageLeastSquares,
    "2sls": TwoStageLeastSquares,
    "tsls": TwoStageLeastSquares,
    "iv": TwoStageLeastSquares,
    "wald": Wald,
    "pliv": PLIV,
    "dml_iv": PLIV,
}

_DEFAULT_TARGET: dict[str, EffectTarget] = {
    ADJUSTMENT: EffectTarget.ATE,
    FRONTDOOR: EffectTarget.ATE,
    IV: EffectTarget.LATE,
}


def estimator_for(
    identification: Identification, strategy: str | None = None
) -> type[CausalEstimator]:
    """The estimator class for the identified family, or a named one if it fits."""
    family = identification.strategy
    if strategy is not None:
        cls = _BY_NAME.get(strategy.lower())
        if cls is None:
            allowed = ", ".join(sorted(_BY_NAME))
            raise ValueError(f"unknown estimator {strategy!r}; use one of {allowed}")
        if family not in cls.consumes:
            raise ValueError(
                f"estimator {strategy!r} consumes {sorted(cls.consumes)}, "
                f"but the effect was identified by {family!r}"
            )
        return cls
    options = _BY_FAMILY.get(family)
    if not options:
        raise ValueError(f"no estimator for the {family!r} family")
    return options[0]


def candidates(identification: Identification) -> list[type[CausalEstimator]]:
    """Every estimator that consumes the identified family, default first."""
    return list(_BY_FAMILY.get(identification.strategy, []))


def default_target(identification: Identification) -> EffectTarget:
    return _DEFAULT_TARGET.get(identification.strategy, EffectTarget.ATE)
