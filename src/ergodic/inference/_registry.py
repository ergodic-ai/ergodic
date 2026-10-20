# SPDX-License-Identifier: Apache-2.0
"""Resolve an estimand and a request into a concrete estimator.

The compatibility gate from the graph to the estimator: an estimator consumes a
family, and ``Identification.strategy`` says which family holds. The first entry
per family is the default (the doubly-robust one for adjustment).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ._estimator import ADJUSTMENT, FRONTDOOR, IV, NULL, CausalEstimator
from ._target import EffectTarget
from .estimators import (
    AIPW,
    DML,
    IPW,
    PLIV,
    CausalForest,
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

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from ..data import TabularData
    from ..identification import Identification
    from ._perunit import CATEResult


@runtime_checkable
class HasCate(Protocol):
    """A fitted estimator that reports a per-unit CATE result.

    The CATE facade casts the resolved (and fitted) estimator to this Protocol so
    ``cate`` is callable while the library stays mypy-strict. The meta-learners and
    DML satisfy it.
    """

    def cate(self, data: TabularData | None = ..., *, uncertainty: bool = ...) -> CATEResult: ...


_BY_FAMILY: dict[str, list[type[CausalEstimator]]] = {
    ADJUSTMENT: [
        AIPW,
        DML,
        DRLearner,
        OutcomeRegression,
        IPW,
        Stratification,
        RLearner,
        XLearner,
        CausalForest,
    ],
    FRONTDOOR: [TwoStageRegression],
    IV: [TwoStageLeastSquares, PLIV, Wald],
    NULL: [NullEffect],
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
    "causal_forest": CausalForest,
    "grf": CausalForest,
    "causalforest": CausalForest,
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
    "null": NullEffect,
    "null_effect": NullEffect,
}

_DEFAULT_TARGET: dict[str, EffectTarget] = {
    ADJUSTMENT: EffectTarget.ATE,
    FRONTDOOR: EffectTarget.ATE,
    IV: EffectTarget.LINEAR_IV,
    NULL: EffectTarget.ATE,
}

# The CATE-capable estimators, by user-facing name. The doubly-robust DR-learner is
# the default (the robust choice, and it shares the doubly-robust score helper that
# the explainability layer reads).
_CATE_DEFAULT = "dr_learner"
_CATE_BY_NAME: dict[str, type[CausalEstimator]] = {
    "dr_learner": DRLearner,
    "drlearner": DRLearner,
    "dr": DRLearner,
    "r_learner": RLearner,
    "rlearner": RLearner,
    "x_learner": XLearner,
    "xlearner": XLearner,
    "s_learner": OutcomeRegression,
    "t_learner": OutcomeRegression,
    "outcome_regression": OutcomeRegression,
    "g_formula": OutcomeRegression,
    "dml": DML,
    "double_ml": DML,
    "causal_forest": CausalForest,
    "grf": CausalForest,
    "causalforest": CausalForest,
}

# Extra constructor knobs a method name implies (the S/T-learner split is one
# class with a flag, so the name carries the flag).
_METHOD_KWARGS: dict[str, dict[str, object]] = {
    "t_learner": {"per_arm": True},
    "s_learner": {"per_arm": False},
}

# Reserved method names for the neural and foundation-model CATE learners. Nothing
# implements them yet, and selecting one raises a clear "not available" error
# rather than a generic unknown-name one.
_CATE_NEURAL = ("tarnet", "dragonnet", "cfr", "drnet")
_CATE_FOUNDATION = ("causal_pfn", "causalfm", "causal_pfm")


def method_kwargs_for(method: str | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve named-method settings for both facades, rejecting contradictions."""
    implied = _METHOD_KWARGS.get((method or "").lower(), {})
    for key, value in implied.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f"method {method!r} requires {key}={value!r}; got {kwargs[key]!r}")
    return {**implied, **kwargs}


def cate_estimator_for(strategy: str | None = None) -> type[CausalEstimator]:
    """The CATE-capable estimator class for a name, or the doubly-robust default.

    Recognizes the reserved neural and foundation names and says they are not
    available in this version, rather than raising a generic unknown-name error.
    """
    name = (strategy or _CATE_DEFAULT).lower()
    if name in ("auto", ""):
        name = _CATE_DEFAULT
    if name in _CATE_BY_NAME:
        return _CATE_BY_NAME[name]
    if name in _CATE_NEURAL:
        raise NotImplementedError(
            f"the neural CATE learner {name!r} is not available in this version of "
            f"ergodic; use one of {', '.join(sorted(_CATE_BY_NAME))}"
        )
    if name in _CATE_FOUNDATION:
        raise NotImplementedError(
            f"the foundation-model CATE learner {name!r} is not available in this "
            f"version of ergodic; use one of {', '.join(sorted(_CATE_BY_NAME))}"
        )
    allowed = ", ".join(sorted(_CATE_BY_NAME))
    raise ValueError(f"unknown CATE method {strategy!r}; use one of {allowed}, or 'auto'")


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
