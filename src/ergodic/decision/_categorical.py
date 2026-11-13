# SPDX-License-Identifier: Apache-2.0
"""Cross-fitted multi-arm DR scores over the complete context population.

For arm a: m_a(X) + I(A=a)/p_a(X) * (Y-m_a(X)). Contrasts difference
paired scores before averaging. Query predictions regress the paired scores.
See Linden et al. (2016), https://doi.org/10.1002/sim.6768.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..inference._crossfit import _folds, _train_mask
from ..inference._learner import _fresh
from ..inference.learners import OLS, LogisticRegression

FloatArray = npt.NDArray[np.float64]


class _Propensity:
    """Normalized one-vs-rest logistic probabilities; a declared working model."""

    def fit(self, x: FloatArray, a: FloatArray) -> _Propensity:
        self.classes_ = np.unique(a)
        self.models = [LogisticRegression().fit(x, (a == k).astype(float)) for k in self.classes_]
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        p = np.column_stack([m.predict_proba(x)[:, 1] for m in self.models])
        return np.asarray(p / p.sum(axis=1, keepdims=True))

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(self.classes_[np.argmax(self.predict_proba(x), axis=1)])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


def _prediction(model: Any, x: FloatArray) -> FloatArray:
    values = np.asarray(model.predict(x), dtype=float)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError("outcome/final learner must return one finite regression mean per row")
    return values


class ArmDR:
    """Cross-fitted categorical-treatment DR with pluggable regression/propensity learners.

    Arm labels passed to ``fit`` are contiguous integer codes. The caller owns
    graph identification and label encoding. No arm or outcome is silently dropped.
    Positivity failures raise instead of clipping weights. Population intervals
    are asymptotic and do not include graph selection or multiple-comparison error.
    Query predictions are conditional expectations with no individual intervals.
    """

    def __init__(
        self,
        *,
        outcome: Any = None,
        propensity: Any = None,
        final: Any = None,
        n_folds: int = 5,
        seed: int = 0,
        min_propensity: float = 0.01,
    ) -> None:
        if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError("n_folds must be an integer >= 2")
        if not 0 < min_propensity < 0.5:
            raise ValueError("min_propensity must be between 0 and 0.5")
        self.outcome = outcome if outcome is not None else OLS()
        self.propensity: Any = propensity if propensity is not None else _Propensity()
        self.final = final if final is not None else OLS()
        if hasattr(self.outcome, "predict_proba") or hasattr(self.final, "predict_proba"):
            raise ValueError("outcome and final slots require regression means, not classifiers")
        self.n_folds, self.seed, self.min_propensity = n_folds, seed, min_propensity
        self._scores: FloatArray | None = None
        self._finals: dict[tuple[int, int], Any] = {}
        self._propensities: list[Any] = []

    def fit(self, x: FloatArray, a: FloatArray, y: FloatArray) -> ArmDR:
        """Fit fresh fold-specific nuisances; retain scores over every context row."""
        self._scores = None
        self._finals = {}
        self._propensities = []
        x, a, y = np.asarray(x, float), np.asarray(a, float), np.asarray(y, float)
        if (
            x.ndim != 2
            or a.shape != (len(x),)
            or y.shape != a.shape
            or not all(np.isfinite(v).all() for v in (x, a, y))
        ):
            raise ValueError("fit requires finite aligned features, treatment and outcome")
        arms, counts = np.unique(a, return_counts=True)
        if len(arms) < 2 or not np.array_equal(arms, np.arange(len(arms))):
            raise ValueError("treatment requires contiguous arm codes 0..K-1")
        if np.min(counts) < self.n_folds or len(x) < 2 * self.n_folds:
            raise ValueError("each treatment arm needs at least n_folds rows")
        scores = np.empty((len(x), len(arms)))
        minimum = 1.0
        for held in _folds(len(x), self.n_folds, np.random.default_rng(self.seed), stratify=a):
            train = _train_mask(len(x), held)
            model = _fresh(self.propensity).fit(x[train], a[train])
            self._propensities.append(model)
            classes = np.asarray(model.classes_)
            if len(classes) != len(arms) or set(classes.tolist()) != set(arms.tolist()):
                raise ValueError("propensity learner classes differ from treatment arms")
            p = np.asarray(model.predict_proba(x[held]), float)
            if (
                p.shape != (len(held), len(arms))
                or not np.isfinite(p).all()
                or np.any(p < 0)
                or np.any(p > 1)
                or not np.allclose(p.sum(axis=1), 1, atol=1e-7)
            ):
                raise ValueError("propensity learner must return normalized arm probabilities")
            p = p[:, [int(np.flatnonzero(classes == arm)[0]) for arm in arms]]
            minimum = min(minimum, float(p.min()))
            if minimum < self.min_propensity:
                raise ValueError("insufficient treatment overlap: propensity below min_propensity")
            for j, arm in enumerate(arms):
                selected = train & (a == arm)
                outcome = _fresh(self.outcome).fit(x[selected], y[selected])
                m = _prediction(outcome, x[held])
                scores[held, j] = m + (a[held] == arm) / p[:, j] * (y[held] - m)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite doubly robust scores")
        self._x = x.copy()
        self._scores = scores
        self.minimum_propensity_ = minimum
        return self

    def contrast(
        self, baseline: int, value: int, query: FloatArray | None = None
    ) -> tuple[float, float, FloatArray | None]:
        """Return population mean, paired-score standard error and optional query effects."""
        if self._scores is None:
            raise RuntimeError("ArmDR is not fitted")
        k = self._scores.shape[1]
        if not 0 <= baseline < k or not 0 <= value < k:
            raise ValueError("contrast arm is outside the fitted labels")
        score = self._scores[:, value] - self._scores[:, baseline]
        predicted = None
        if query is not None:
            query = np.asarray(query, float)
            if (
                query.ndim != 2
                or query.shape[1] != self._x.shape[1]
                or not np.isfinite(query).all()
            ):
                raise ValueError("query features must match the fitted design")
            if self._x.shape[1] and (
                np.any(query < self._x.min(axis=0)) or np.any(query > self._x.max(axis=0))
            ):
                raise ValueError("query covariates are outside observed context support")
            for propensity in self._propensities:
                probabilities = np.asarray(propensity.predict_proba(query), float)
                if (
                    probabilities.shape != (len(query), k)
                    or not np.isfinite(probabilities).all()
                    or np.any(probabilities < self.min_propensity)
                    or np.any(probabilities > 1)
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-7)
                ):
                    raise ValueError("insufficient treatment overlap on query rows")
            key = baseline, value
            if key not in self._finals:
                self._finals[key] = _fresh(self.final).fit(self._x, score)
            predicted = _prediction(self._finals[key], query)
        return float(score.mean()), float(score.std(ddof=1) / np.sqrt(len(score))), predicted
