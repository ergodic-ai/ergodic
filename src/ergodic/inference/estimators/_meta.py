# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous-effect meta-learners for the adjustment family: R- and X-learner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._crossfit import _fit_predict, _folds, _train_mask
from .._estimator import Slot, Task, _AdjustmentBase
from .._learner import _mean_predict
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._learner import Learner


class RLearner(_AdjustmentBase):
    """The R-learner: residualize the outcome and treatment, then fit the CATE.

    Robinson's residual-on-residual moment with a heterogeneous final model,
    weighted by the squared treatment residual. The ATE is the average of the
    fitted CATE, with a bootstrap standard error.
    """

    name = "r_learner"
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

    def _residualize(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        n = a.shape[0]
        res_y = np.empty(n, dtype=np.float64)
        res_t = np.empty(n, dtype=np.float64)
        rng = np.random.default_rng(self._random_state)
        for test in _folds(n, self.n_folds, rng, stratify=a):
            train = _train_mask(n, test)
            res_y[test] = y[test] - _fit_predict(make["outcome"], z[train], y[train], z[test])
            e = np.clip(
                _mean_predict(make["propensity"]().fit(z[train], a[train]), z[test]),
                self.clip,
                1.0 - self.clip,
            )
            res_t[test] = a[test] - e
        return res_y, res_t

    def _fit_final(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> Learner:
        res_y, res_t = self._residualize(arrays, make)
        pseudo = res_y / res_t
        weight = res_t * res_t
        final = make["final"]()
        try:
            final.fit(arrays["Z"], pseudo, sample_weight=weight)
        except TypeError as exc:
            raise TypeError(
                "slot 'final' needs a learner whose fit accepts sample_weight; "
                f"got {type(final).__name__}"
            ) from exc
        return final

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        self._final = self._fit_final(self._state.arrays, self._state.make)
        return self

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        final = self._fit_final(arrays, make)
        return float(np.mean(_mean_predict(final, arrays["Z"])))

    def cate(self, data: TabularData | None = None) -> npt.NDArray[np.float64]:
        """Per-unit CATE, in-sample when ``data`` is ``None``, else on its covariates."""
        if self._final is None:
            raise RuntimeError("r_learner is not fitted; call fit first")
        return _mean_predict(self._final, self._cate_covariates(data))


class XLearner(_AdjustmentBase):
    """The X-learner: per-arm outcome models, imputed effects, propensity-weighted.

    Fit one outcome model per arm, impute each unit's effect against the other
    arm's model, regress those imputed effects per arm, and combine the two with
    the propensity. The ATE is the average of the fitted CATE, with a bootstrap
    standard error.
    """

    name = "x_learner"
    slots = (Slot("outcome", Task.REGRESSION), Slot("propensity", Task.CLASSIFICATION))
    targets = frozenset({EffectTarget.ATE})

    def __init__(self, learners: Any = None, *, clip: float = 1e-3, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)
        self.clip = clip
        self._models: tuple[Learner, Learner, Learner] | None = None

    def _fit_models(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[Learner, Learner, Learner]:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        treated = a == 1.0
        control = a == 0.0
        mu1 = make["outcome"]().fit(z[treated], y[treated])
        mu0 = make["outcome"]().fit(z[control], y[control])
        imputed_treated = y[treated] - _mean_predict(mu0, z[treated])
        imputed_control = _mean_predict(mu1, z[control]) - y[control]
        tau1 = make["outcome"]().fit(z[treated], imputed_treated)
        tau0 = make["outcome"]().fit(z[control], imputed_control)
        propensity = make["propensity"]().fit(z, a)
        return tau0, tau1, propensity

    def _combine(
        self, models: tuple[Learner, Learner, Learner], z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        tau0, tau1, propensity = models
        e = np.clip(_mean_predict(propensity, z), self.clip, 1.0 - self.clip)
        return e * _mean_predict(tau0, z) + (1.0 - e) * _mean_predict(tau1, z)

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        self._models = self._fit_models(self._state.arrays, self._state.make)
        return self

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        models = self._fit_models(arrays, make)
        return float(np.mean(self._combine(models, arrays["Z"])))

    def cate(self, data: TabularData | None = None) -> npt.NDArray[np.float64]:
        """Per-unit CATE, in-sample when ``data`` is ``None``, else on its covariates."""
        if self._models is None:
            raise RuntimeError("x_learner is not fitted; call fit first")
        return self._combine(self._models, self._cate_covariates(data))
