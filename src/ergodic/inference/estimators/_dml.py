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
    _MetaLearnerBase,
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
    from .._perunit import CATEResult
    from .._scores import DRScores


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

    def cate(self, data: TabularData | None = None) -> CATEResult:
        """The constant per-unit effect for the partially-linear model: ``theta``.

        The partially-linear DML assumes a homogeneous effect, so every unit's CATE
        is the single slope ``theta``, with the same analytic standard error from the
        orthogonal moment. Returned as a :class:`~ergodic.inference.CATEResult` for a
        uniform surface; a binary treatment additionally caches the doubly-robust
        scores so ``explain`` can test for the heterogeneity the model assumes away.
        """
        from .._perunit import CATEResult

        if self._cache is None:
            raise RuntimeError("dml is not fitted; call fit first")
        theta = self._cache[0]
        z = self._cate_covariates(data)
        n = z.shape[0]
        values = np.full(n, theta, dtype=np.float64)
        analytic = self._analytic(self._state.arrays, EffectTarget.ATE, self._state.make)
        std_error: npt.NDArray[np.float64] | None = None
        kind = "point"
        if analytic is not None:
            influence = analytic[1]
            se = float(np.std(influence, ddof=1) / np.sqrt(influence.shape[0]))
            std_error = np.full(n, se, dtype=np.float64)
            kind = "analytic"
        scores = self._dml_scores() if data is None else None
        return CATEResult(
            treatment=self._treatment_name,
            outcome=self._outcome_name,
            covariates=self._covariates,
            n=n,
            kind=kind,
            source=self.name,
            diagnostics=dict(self._state.diagnostics),
            identification=self._identification,
            _values=values,
            _index=np.arange(n, dtype=np.int64),
            _std_error=std_error,
            _scores=scores,
        )

    def _dml_scores(self) -> DRScores | None:
        """Doubly-robust scores for explainability, only for a binary treatment."""
        if not self._t_binary or self._identification is None:
            return None
        from .._scores import DRScores
        from .._scores import aipw_scores as _scores_fn

        state = self._state
        a, y, z = state.arrays["A"], state.arrays["Y"], state.arrays["Z"]
        try:
            mu0, mu1, e = _dml_crossfit_nuisances(
                a, y, z, state.make, self.n_folds, self._random_state
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        return DRScores(
            scores=_scores_fn(a, y, mu0, mu1, e),
            mu0=mu0,
            mu1=mu1,
            propensity=e,
            covariates=z,
            treatment=a,
            outcome=y,
            covariate_names=self._covariates,
        )


def _dml_crossfit_nuisances(
    a: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    make: dict[str, Callable[[], Learner]],
    n_folds: int,
    random_state: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cross-fit AIPW nuisances for DML's explainability scores (binary treatment)."""
    from .._scores import _crossfit_nuisances

    return _crossfit_nuisances(
        a,
        y,
        z,
        make_outcome=make["outcome"],
        make_propensity=lambda: LogisticRegression(),
        n_folds=n_folds,
        clip=1e-3,
        random_state=random_state,
    )


class _DRModel:
    """The DR-learner's fitted CATE handle: the pseudo-outcome and the final model."""

    def __init__(self, pseudo: npt.NDArray[np.float64], final: Learner) -> None:
        self.pseudo = pseudo
        self.final = final


class DRLearner(_MetaLearnerBase):
    """Doubly-robust learning of the ATE and the CATE.

    Cross-fit the AIPW score into a per-unit pseudo-outcome, average it for the
    ATE (with the influence-function standard error), and regress it on the
    covariates for the CATE through ``cate``. The CATE result carries the cached
    doubly-robust scores, so its ``explain`` runs the Tier-B characterization.
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

    def _dr_nuisances(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # reuse the estimator's own cross-fit for the explainability scores
        return self._crossfit(arrays, make)

    def _fit_cate(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> _DRModel:
        pseudo = self._pseudo_outcome(arrays, make)
        final = make["final"]().fit(arrays["Z"], pseudo)
        return _DRModel(pseudo, final)

    def _predict_cate(self, model: _DRModel, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _mean_predict(model.final, z)

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        self._pseudo = self._cate_model().pseudo
        return self

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        if target is not EffectTarget.ATE or self._pseudo is None:
            return None
        return float(np.mean(self._pseudo)), self._pseudo

    def _cate_analytic_se(
        self, model: _DRModel, z: npt.NDArray[np.float64], data: TabularData | None
    ) -> npt.NDArray[np.float64] | None:
        """Per-unit standard errors from the linear final's sandwich covariance.

        When the final stage is the native linear :class:`OLS`, the pseudo-outcome
        regression ``tau_hat(z) = [1, z] beta`` has a heteroskedasticity-robust
        coefficient covariance, and each unit's variance is the quadratic form
        ``x' Cov(beta) x``. Returns ``None`` for any other (nonlinear) final, where the
        per-unit point predictions have no closed-form interval and the bootstrap
        path takes over.
        """
        final = model.final
        if not isinstance(final, OLS) or final._coef is None:
            return None
        z_train = self._state.arrays["Z"]
        x_train = _ols_design(final, z_train)
        x_eval = _ols_design(final, z)
        resid = model.pseudo - x_train @ final._coef
        xtx_inv = np.linalg.pinv(x_train.T @ x_train)
        meat = (x_train * resid[:, None]).T @ (x_train * resid[:, None])
        cov = xtx_inv @ meat @ xtx_inv
        var = np.einsum("ij,jk,ik->i", x_eval, cov, x_eval)
        return np.asarray(np.sqrt(np.clip(var, 0.0, np.inf)), dtype=np.float64)


def _ols_design(model: OLS, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The OLS design matrix (a leading ones column when the model fits an intercept)."""
    if model.fit_intercept:
        ones = np.ones((z.shape[0], 1), dtype=np.float64)
        return np.asarray(np.hstack([ones, z]), dtype=np.float64)
    return np.asarray(z, dtype=np.float64)
