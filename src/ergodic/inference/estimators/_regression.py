# SPDX-License-Identifier: Apache-2.0
"""Outcome regression for the adjustment family (g-formula, S- and T-learner)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._estimator import Slot, Task, _MetaLearnerBase
from .._learner import _mean_predict, _mean_predict_draws
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._estimator import _FitState
    from .._learner import Learner


class OutcomeRegression(_MetaLearnerBase):
    """Fit a model for the outcome and average the predicted contrast.

    With ``per_arm=False`` (the default) one model carries the treatment as a
    feature (the S-learner / g-formula). With ``per_arm=True`` a separate model is
    fit on each arm (the T-learner). A Bayesian outcome learner flows its posterior
    into the per-unit CATE result and into ``cate_draws``/``cate_interval``.
    """

    name = "outcome_regression"
    slots = (Slot("outcome", Task.REGRESSION),)
    targets = frozenset({EffectTarget.ATE, EffectTarget.ATT})

    def __init__(self, learners: Any = None, *, per_arm: bool = False, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)
        self.per_arm = per_arm
        self._model: Learner | None = None
        self._arm_models: tuple[Learner, Learner] | None = None

    def _fit_cate(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> Learner | tuple[Learner, Learner]:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        if self.per_arm:
            model1 = make["outcome"]()
            model1.fit(z[a == 1.0], y[a == 1.0])
            model0 = make["outcome"]()
            model0.fit(z[a == 0.0], y[a == 0.0])
            return (model0, model1)
        model = make["outcome"]()
        model.fit(np.hstack([a.reshape(-1, 1), z]), y)
        return model

    def _predict_cate(
        self, model: Learner | tuple[Learner, Learner], z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if isinstance(model, tuple):
            model0, model1 = model
            return _mean_predict(model1, z) - _mean_predict(model0, z)
        n = z.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        zeros = np.zeros((n, 1), dtype=np.float64)
        return _mean_predict(model, np.hstack([ones, z])) - _mean_predict(
            model, np.hstack([zeros, z])
        )

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        model = self._cate_model()
        if isinstance(model, tuple):
            self._arm_models = model
        else:
            self._model = model
        return self

    def _cate_draws(
        self, model: Learner | tuple[Learner, Learner], z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64] | None:
        mu1, mu0 = self._outcome_draws(z)
        return np.asarray(mu1 - mu0, dtype=np.float64)

    # -- the posterior path (Bayesian outcome learner) -----------------------

    def _outcome_draws(
        self, z: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Posterior draws of mu1 and mu0 at ``z``, each ``(S, n)``.

        For the S-learner the two predictions read one coefficient draw, so the
        contrast is coherent within a draw; for the T-learner the arms are
        independent posteriors paired by index.
        """
        if self.per_arm:
            if self._arm_models is None:
                raise RuntimeError("outcome_regression is not fitted; call fit first")
            model0, model1 = self._arm_models
            mu1 = _mean_predict_draws(model1, z)
            mu0 = _mean_predict_draws(model0, z)
        else:
            if self._model is None:
                raise RuntimeError("outcome_regression is not fitted; call fit first")
            n = z.shape[0]
            ones = np.ones((n, 1), dtype=np.float64)
            zeros = np.zeros((n, 1), dtype=np.float64)
            mu1 = _mean_predict_draws(self._model, np.hstack([ones, z]))
            mu0 = _mean_predict_draws(self._model, np.hstack([zeros, z]))
        s = min(mu1.shape[0], mu0.shape[0])
        return mu1[:s], mu0[:s]

    def cate_draws(self, data: TabularData | None = None) -> npt.NDArray[np.float64]:
        """Per-unit CATE posterior (mu1 - mu0), shape ``(S, n)``.

        Needs a Bayesian outcome learner; in-sample when ``data`` is ``None``.
        """
        z = self._cate_covariates(data)
        mu1, mu0 = self._outcome_draws(z)
        return np.asarray(mu1 - mu0, dtype=np.float64)

    def cate_interval(
        self, data: TabularData | None = None, *, level: float = 0.95
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-unit credible interval for the CATE: the column quantiles of the draws."""
        draws = self.cate_draws(data)
        lo = np.quantile(draws, (1.0 - level) / 2.0, axis=0)
        hi = np.quantile(draws, (1.0 + level) / 2.0, axis=0)
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def _posterior(self, state: _FitState, target: EffectTarget) -> npt.NDArray[np.float64] | None:
        a = state.arrays["A"]
        mu1, mu0 = self._outcome_draws(state.arrays["Z"])
        diff = mu1 - mu0
        if target is EffectTarget.ATT:
            return np.asarray(diff[:, a == 1.0].mean(axis=1), dtype=np.float64)
        return np.asarray(diff.mean(axis=1), dtype=np.float64)

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        n = a.shape[0]
        if self.per_arm:
            treated = a == 1.0
            control = a == 0.0
            model1 = make["outcome"]().fit(z[treated], y[treated])
            model0 = make["outcome"]().fit(z[control], y[control])
            mu1 = _mean_predict(model1, z)
            mu0 = _mean_predict(model0, z)
        else:
            model = make["outcome"]().fit(np.hstack([a.reshape(-1, 1), z]), y)
            ones = np.ones((n, 1), dtype=np.float64)
            zeros = np.zeros((n, 1), dtype=np.float64)
            mu1 = _mean_predict(model, np.hstack([ones, z]))
            mu0 = _mean_predict(model, np.hstack([zeros, z]))
        diff = mu1 - mu0
        if target is EffectTarget.ATT:
            return float(np.mean(diff[a == 1.0]))
        return float(np.mean(diff))
