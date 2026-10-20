# SPDX-License-Identifier: Apache-2.0
"""The shared doubly-robust scores: one orthogonal object, computed once.

The cross-fit AIPW signal

    Gamma_i = mu1(X_i) - mu0(X_i)
              + W_i (Y_i - mu1(X_i)) / e(X_i)
              - (1 - W_i) (Y_i - mu0(X_i)) / (1 - e(X_i)),

whose conditional mean given the covariates is the CATE and whose sample mean is
the AIPW point estimate of the ATE. The doubly-robust meta-learner regresses these
on the covariates; the Tier-B explainability layer (BLP, GATES, the calibration
test, RATE) projects low-dimensional summaries off the same scores. Computing them
in one place keeps the cross-fitting and the orthogonality consistent across both.

This is the same construction the :class:`~ergodic.inference.DRLearner` and the
:class:`AIPW` estimator already use, lifted to a reusable function so the
explainability layer reads the identical signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._crossfit import _folds, _train_mask
from ._estimator import (
    _AdjustmentBase,
    _covariate_columns,
    _covariate_matrix,
    _outcome_vector,
    _require_complete,
    _single,
)
from ._learner import _mean_predict
from .learners import OLS, LogisticRegression

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ..data import TabularData
    from ..identification import Identification
    from ._learner import Learner


@dataclass(frozen=True)
class DRScores:
    """The cross-fit doubly-robust object: the AIPW signal plus its nuisances.

    ``scores`` is the per-unit ``Gamma_i`` (its mean is the AIPW ATE). ``mu0``,
    ``mu1``, and ``propensity`` are the cross-fit out-of-fold nuisances, kept so the
    calibration and projection tools reuse them without refitting. ``treatment``,
    ``outcome``, and ``covariates`` carry the design back to the result.
    ``ranking_error`` retains a failed cross-fit proxy diagnostic. Without
    ``tau_hat``, covariate projections remain available but ranking summaries refuse.
    """

    scores: npt.NDArray[np.float64]
    mu0: npt.NDArray[np.float64]
    mu1: npt.NDArray[np.float64]
    propensity: npt.NDArray[np.float64]
    covariates: npt.NDArray[np.float64]
    treatment: npt.NDArray[np.float64]
    outcome: npt.NDArray[np.float64]
    covariate_names: tuple[str, ...]
    tau_hat: npt.NDArray[np.float64] | None = None
    covariate_variables: tuple[str, ...] = ()
    ranking_error: str | None = None

    def __post_init__(self) -> None:
        width = np.asarray(self.covariates).shape[1] if np.ndim(self.covariates) == 2 else 0
        if width != len(self.covariate_names):
            raise ValueError(
                f"the covariate design has {width} columns but {len(self.covariate_names)} "
                f"names {list(self.covariate_names)}; name every design column (a "
                "categorical with three or more levels is one column per non-first level)"
            )
        if self.covariate_variables and len(self.covariate_variables) != width:
            raise ValueError("covariate_variables must name the variable of every column")

    @property
    def column_variables(self) -> tuple[str, ...]:
        """The variable each design column encodes (the column name when not given)."""
        return self.covariate_variables or self.covariate_names


class _ScoreHelper(_AdjustmentBase):
    """A minimal adjustment estimator used only to extract the arrays the same way.

    The score helper reuses the family ``_extract`` so the treatment coding,
    covariate design, and completeness checks match every other adjustment
    estimator. It never reports an effect, so its abstract ``_point`` is a stub.
    """

    name = "dr_scores"
    slots = ()
    targets = frozenset()

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: object,
        make: dict[str, Callable[[], Learner]],
    ) -> float:  # pragma: no cover - never called
        raise NotImplementedError


def _crossfit_nuisances(
    a: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    *,
    make_outcome: Callable[[], Learner],
    make_propensity: Callable[[], Learner],
    n_folds: int,
    clip: float,
    random_state: int | None,
    raw_propensity: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Out-of-fold ``mu0``, ``mu1``, ``e``: outcome on ``[a, z]``, propensity on ``z``.

    The same cross-fit construction the DR-learner uses. The outcome model carries
    the treatment as a feature and is read at both arms; the propensity is clipped
    to keep the inverse weights finite.
    """
    n = a.shape[0]
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or not 2 <= n_folds <= n:
        raise ValueError("n_folds must be an integer in [2, n_rows]")
    if not 0.0 < clip < 0.5:
        raise ValueError("clip must be in (0, 0.5)")
    if set(np.unique(a).tolist()) != {0.0, 1.0} or min(np.sum(a == 0), np.sum(a == 1)) < 2:
        raise ValueError("cross-fitting needs at least two observations in each treatment arm")
    mu0 = np.empty(n, dtype=np.float64)
    mu1 = np.empty(n, dtype=np.float64)
    e = np.empty(n, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    for test in _folds(n, n_folds, rng, stratify=a):
        if test.size == 0:
            continue
        train = _train_mask(n, test)
        design = np.hstack([a[train].reshape(-1, 1), z[train]])
        outcome = make_outcome().fit(design, y[train])
        ones = np.ones((test.shape[0], 1), dtype=np.float64)
        zeros = np.zeros((test.shape[0], 1), dtype=np.float64)
        mu1[test] = _mean_predict(outcome, np.hstack([ones, z[test]]))
        mu0[test] = _mean_predict(outcome, np.hstack([zeros, z[test]]))
        propensity = make_propensity().fit(z[train], a[train])
        predicted = _mean_predict(propensity, z[test])
        if not np.all(np.isfinite(predicted)) or np.any((predicted < 0) | (predicted > 1)):
            raise ValueError("propensity predictions must be finite probabilities in [0, 1]")
        if raw_propensity is not None:
            raw_propensity[test] = predicted
        e[test] = np.clip(predicted, clip, 1.0 - clip)
    return mu0, mu1, e


def aipw_scores(
    a: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    mu1: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The per-unit AIPW score ``Gamma_i`` from the nuisances."""
    scores = (mu1 - mu0) + a * (y - mu1) / e - (1.0 - a) * (y - mu0) / (1.0 - e)
    return np.asarray(scores, dtype=np.float64)


def dr_scores(
    identification: Identification,
    data: TabularData,
    *,
    learners: object = None,
    folds: int = 5,
    clip: float = 1e-3,
    treated: object = None,
    random_state: int | None = None,
) -> DRScores:
    """Cross-fit doubly-robust scores for an adjustment-identified effect.

    ``learners`` is an alias, a learner, or a slot mapping over ``outcome`` and
    ``propensity`` (the same surface every estimator takes); the defaults are the
    native :class:`OLS` (or logistic for a binary outcome) and logistic propensity.
    ``folds`` sets the cross-fit count. Returns a :class:`DRScores` carrying the
    signal and the out-of-fold nuisances.
    """
    treatment_name = _single(identification.treatment, "treatment")
    outcome_name = _single(identification.outcome, "outcome")
    covariates = identification.adjustment_set or frozenset()
    _require_complete(data, {treatment_name, outcome_name, *covariates})
    names, variables = _covariate_columns(data, covariates)

    helper = _ScoreHelper(treated=treated, random_state=random_state)
    a = helper._treatment_vector(data, treatment_name)
    y, y_binary = _outcome_vector(data, outcome_name)
    z = _covariate_matrix(data, covariates)

    make_outcome, make_propensity = _resolve_score_learners(learners, y_binary)
    mu0, mu1, e = _crossfit_nuisances(
        a,
        y,
        z,
        make_outcome=make_outcome,
        make_propensity=make_propensity,
        n_folds=folds,
        clip=clip,
        random_state=random_state,
    )
    scores = aipw_scores(a, y, mu0, mu1, e)
    return DRScores(
        scores=scores,
        mu0=mu0,
        mu1=mu1,
        propensity=e,
        covariates=z,
        treatment=a,
        outcome=y,
        covariate_names=names,
        covariate_variables=variables,
    )


def _resolve_score_learners(
    learners: object, y_binary: bool
) -> tuple[Callable[[], Learner], Callable[[], Learner]]:
    """Per-slot fresh-learner factories for the outcome and propensity nuisances."""
    from collections.abc import Mapping
    from functools import partial

    from ._learner import _as_learner, _fresh, _is_classifier

    outcome_spec: object = None
    propensity_spec: object = None
    if isinstance(learners, Mapping):
        unknown = set(learners) - {"outcome", "propensity"}
        if unknown:
            raise ValueError(
                f"dr_scores has slots ['outcome', 'propensity']; got {sorted(unknown)}"
            )
        outcome_spec = learners.get("outcome")
        propensity_spec = learners.get("propensity")
    elif learners is not None:
        outcome_spec = learners
        propensity_spec = learners

    if outcome_spec is None:
        outcome: Learner = LogisticRegression() if y_binary else OLS()
    else:
        outcome = _as_learner(outcome_spec)
    if propensity_spec is None:
        propensity: Learner = LogisticRegression()
    else:
        propensity = _as_learner(propensity_spec, want_classifier=True)
    if not _is_classifier(propensity):
        raise TypeError(
            "slot 'propensity' needs a classifier with predict_proba; "
            f"got {type(propensity).__name__}"
        )
    return partial(_fresh, outcome), partial(_fresh, propensity)
