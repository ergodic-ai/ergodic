# SPDX-License-Identifier: Apache-2.0
"""Orthogonal estimators for the adjustment family: DML and the DR-learner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._crossfit import _fit_predict, _folds, _train_mask
from .._estimator import (
    Slot,
    Task,
    _AdjustmentBase,
    _covariate_matrix,
    _outcome_vector,
    _require_complete,
    _single,
)
from .._learner import _mean_predict
from .._target import EffectTarget
from ..learners import OLS, LogisticRegression

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._learner import Learner


class DML(_AdjustmentBase):
    """Double machine learning, the partially linear model (partialling-out).

    Residualize the outcome and the treatment against the covariates with
    cross-fitting, then regress one residual on the other. The treatment may be
    binary or continuous; a continuous treatment reports the average partial
    effect. The ATE standard error comes from the orthogonal moment.
    """

    name = "dml"
    slots = (Slot("outcome", Task.REGRESSION), Slot("treatment", Task.REGRESSION))
    targets = frozenset({EffectTarget.ATE})

    def __init__(self, learners: Any = None, *, n_folds: int = 5, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)
        self.n_folds = n_folds
        self._t_binary = False
        self._cache: (
            tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], float] | None
        ) = None

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        covariates = identification.adjustment_set or frozenset()
        _require_complete(data, {treatment, outcome, *covariates})
        self._covariates = tuple(sorted(covariates))
        a = self._treatment_vector(data, treatment, allow_continuous=True)
        y, y_binary = _outcome_vector(data, outcome)
        z = _covariate_matrix(data, covariates)
        self._t_binary = set(np.unique(a).tolist()) <= {0.0, 1.0}
        arrays = {"A": a, "Y": y, "Z": z}
        diagnostics = {"n_covariates": float(z.shape[1]), "treatment_binary": float(self._t_binary)}
        return arrays, y_binary, diagnostics

    def _default_learner(self, slot: Slot, y_binary: bool) -> Learner:
        if slot.name == "treatment":
            return LogisticRegression() if self._t_binary else OLS()
        return super()._default_learner(slot, y_binary)

    def _residuals(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        n = a.shape[0]
        res_y = np.empty(n, dtype=np.float64)
        res_t = np.empty(n, dtype=np.float64)
        rng = np.random.default_rng(self._random_state)
        stratify = a if self._t_binary else None
        for test in _folds(n, self.n_folds, rng, stratify=stratify):
            train = _train_mask(n, test)
            res_y[test] = y[test] - _fit_predict(make["outcome"], z[train], y[train], z[test])
            res_t[test] = a[test] - _fit_predict(make["treatment"], z[train], a[train], z[test])
        return res_y, res_t

    def _solve(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        res_y, res_t = self._residuals(arrays, make)
        denom = float(np.sum(res_t * res_t))
        if denom <= 0.0:
            raise ValueError("no treatment variation after residualizing")
        theta = float(np.sum(res_t * res_y) / denom)
        return theta, res_y, res_t, denom

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        state = self._state
        self._cache = self._solve(state.arrays, state.make)
        return self

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        return self._solve(arrays, make)[0]

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        if target is not EffectTarget.ATE or self._cache is None:
            return None
        theta, res_y, res_t, denom = self._cache
        n = res_t.shape[0]
        influence = res_t * (res_y - theta * res_t) / (denom / n)
        return theta, influence


class DRLearner(_AdjustmentBase):
    """Doubly-robust learning of the ATE and the CATE.

    Cross-fit the AIPW score into a per-unit pseudo-outcome, average it for the
    ATE (with the influence-function standard error), and regress it on the
    covariates for the CATE through ``cate``.
    """

    name = "dr_learner"
    slots = (
        Slot("outcome", Task.REGRESSION),
        Slot("propensity", Task.CLASSIFICATION),
        Slot("final", Task.REGRESSION),
    )
    targets = frozenset({EffectTarget.ATE})

    def __init__(
        self, learners: Any = None, *, n_folds: int = 5, clip: float = 1e-3, **kwargs: Any
    ) -> None:
        super().__init__(learners, **kwargs)
        self.n_folds = n_folds
        self.clip = clip
        self._final: Learner | None = None
        self._pseudo: npt.NDArray[np.float64] | None = None

    def _crossfit(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        n = a.shape[0]
        mu0 = np.empty(n, dtype=np.float64)
        mu1 = np.empty(n, dtype=np.float64)
        e = np.empty(n, dtype=np.float64)
        rng = np.random.default_rng(self._random_state)
        for test in _folds(n, self.n_folds, rng, stratify=a):
            train = _train_mask(n, test)
            design = np.hstack([a[train].reshape(-1, 1), z[train]])
            outcome = make["outcome"]().fit(design, y[train])
            ones = np.ones((test.shape[0], 1), dtype=np.float64)
            zeros = np.zeros((test.shape[0], 1), dtype=np.float64)
            mu1[test] = _mean_predict(outcome, np.hstack([ones, z[test]]))
            mu0[test] = _mean_predict(outcome, np.hstack([zeros, z[test]]))
            propensity = make["propensity"]().fit(z[train], a[train])
            e[test] = np.clip(_mean_predict(propensity, z[test]), self.clip, 1.0 - self.clip)
        return mu0, mu1, e

    def _pseudo_outcome(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> npt.NDArray[np.float64]:
        a = arrays["A"]
        y = arrays["Y"]
        mu0, mu1, e = self._crossfit(arrays, make)
        return (mu1 - mu0) + a * (y - mu1) / e - (1.0 - a) * (y - mu0) / (1.0 - e)

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        state = self._state
        self._pseudo = self._pseudo_outcome(state.arrays, state.make)
        self._final = state.make["final"]().fit(state.arrays["Z"], self._pseudo)
        return self

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        return float(np.mean(self._pseudo_outcome(arrays, make)))

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        if target is not EffectTarget.ATE or self._pseudo is None:
            return None
        return float(np.mean(self._pseudo)), self._pseudo

    def cate(self, data: TabularData | None = None) -> npt.NDArray[np.float64]:
        """Per-unit CATE, in-sample when ``data`` is ``None``, else on its covariates."""
        if self._final is None:
            raise RuntimeError("dr_learner is not fitted; call fit first")
        return _mean_predict(self._final, self._cate_covariates(data))
