# SPDX-License-Identifier: Apache-2.0
"""The learner protocol and the helpers that bind a learner to a nuisance slot.

A learner is structurally a sklearn estimator: ``fit(X, y)`` then ``predict(X)``,
with ``predict_proba`` for a classifier. We never import sklearn; anything that
quacks right is a :class:`Learner`. The native :class:`OLS` and
:class:`LogisticRegression` implement the same surface, so the zero-dependency
install works and any sklearn or xgboost estimator drops in with no adapter.

What the slot contract adds, and what sklearn leaves loose, is read here: whether
a slot is a regressor or a classifier (the presence of ``predict_proba``), and a
fresh clone per bootstrap resample or per arm (through ``get_params``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

from .learners import OLS, KNNClassifier, KNNRegressor, LogisticRegression

if TYPE_CHECKING:
    import numpy.typing as npt


@runtime_checkable
class Learner(Protocol):
    """The sklearn estimator surface the estimators rely on."""

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> Any: ...

    def predict(self, X: npt.NDArray[np.float64]) -> Any: ...


@runtime_checkable
class BayesianLearner(Protocol):
    """A learner that also carries a posterior over its mean function.

    Structurally a :class:`Learner` (``fit`` then ``predict``, where ``predict`` is
    the posterior predictive mean), plus ``predict_draws`` returning ``(S, n)``
    draws of the mean function. A classifier variant exposes ``predict_proba`` and
    ``predict_proba_draws`` (draws of the class-1 probability) instead. Detection is
    duck-typed in :func:`_is_bayesian`, the same way a classifier is spotted by
    ``predict_proba``, so any object that quacks this way counts.

    Because ``predict`` stays the posterior mean, a Bayesian learner is also a valid
    point learner in every estimator; the posterior path is an addition, not a fork.
    """

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> Any: ...

    def predict(self, X: npt.NDArray[np.float64]) -> Any: ...

    def predict_draws(self, X: npt.NDArray[np.float64]) -> Any: ...


_ALIASES: dict[str, type] = {
    "ols": OLS,
    "linear": OLS,
    "linear_regression": OLS,
    "logistic": LogisticRegression,
    "logit": LogisticRegression,
}


def _as_learner(spec: Any, *, want_classifier: bool = False) -> Learner:
    """Turn an alias string or a learner-like object into a learner instance.

    ``want_classifier`` picks the classifier variant for aliases that have one
    (``knn``), so the same alias fills a regression or a classification slot.
    """
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("knn", "neighbors"):
            return KNNClassifier() if want_classifier else KNNRegressor()
        if key not in _ALIASES:
            allowed = ", ".join([*sorted(_ALIASES), "knn"])
            raise ValueError(f"unknown learner {spec!r}; use one of {allowed}, or pass an object")
        return cast("Learner", _ALIASES[key]())
    if hasattr(spec, "fit") and hasattr(spec, "predict"):
        return cast("Learner", spec)
    raise TypeError(f"a learner needs fit and predict; got {type(spec)!r}")


def _fresh(learner: Learner) -> Learner:
    """An unfitted clone, so a resample or an arm never reuses fitted state."""
    import copy

    if hasattr(learner, "get_params"):
        try:
            # deep=True would add nested "step__param" keys that no constructor takes
            params = learner.get_params(deep=False)
        except TypeError:
            params = learner.get_params()
        return type(learner)(**{key: copy.deepcopy(value) for key, value in params.items()})
    return copy.deepcopy(learner)


def _is_classifier(learner: Learner) -> bool:
    return hasattr(learner, "predict_proba")


def _mean_predict(learner: Learner, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The conditional mean: ``predict_proba(.)[:, 1]`` for a classifier, else ``predict``."""
    proba = getattr(learner, "predict_proba", None)
    if proba is not None:
        return np.asarray(proba(X), dtype=np.float64)[:, 1]
    return np.asarray(learner.predict(X), dtype=np.float64)


def _is_bayesian(learner: Learner) -> bool:
    """A learner is Bayesian if it can hand back posterior draws of its mean."""
    return hasattr(learner, "predict_draws") or hasattr(learner, "predict_proba_draws")


def _mean_predict_draws(learner: Learner, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Posterior draws of the conditional mean, shape ``(S, n)``.

    ``predict_proba_draws`` (the class-1 probability) for a classifier, else
    ``predict_draws``. The row axis is the posterior draw, parallel to
    :func:`_mean_predict` but one dimension richer.
    """
    proba_draws = getattr(learner, "predict_proba_draws", None)
    if proba_draws is not None:
        return np.asarray(proba_draws(X), dtype=np.float64)
    draws = getattr(learner, "predict_draws", None)
    if draws is None:
        raise TypeError(f"{type(learner).__name__} has no posterior draws")
    return np.asarray(draws(X), dtype=np.float64)
