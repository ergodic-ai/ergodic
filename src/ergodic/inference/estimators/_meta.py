# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous-effect meta-learners for the adjustment family: R- and X-learner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._crossfit import _fit_predict, _folds, _train_mask
from .._estimator import Slot, Task, _MetaLearnerBase
from .._learner import _mean_predict
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from .._learner import Learner


class RLearner(_MetaLearnerBase):
    """The R-learner: residualize the outcome and treatment, then fit the CATE.

    Robinson's residual-on-residual moment with a heterogeneous final model,
    weighted by the squared treatment residual. The ATE is the average of the
    fitted CATE, with a bootstrap standard error.
    """

    name = "r_learner"
    slots = (
        Slot("outcome", Task.REGRESSION),
        Slot("propensity", Task.CLASSIFICATION),
        Slot("final", Task.REGRESSION, weighted=True),
    )
    targets = frozenset({EffectTarget.ATE})

    def __init__(
        self, learners: Any = None, *, n_folds: int = 5, clip: float = 1e-3, **kwargs: Any
    ) -> None:
        super().__init__(learners, **kwargs)
        self.n_folds = n_folds
        self.clip = clip

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

    def _fit_cate(
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

    def _predict_cate(self, model: Learner, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _mean_predict(model, z)


class XLearner(_MetaLearnerBase):
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

    def _fit_cate(
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

    def _predict_cate(
        self, model: tuple[Learner, Learner, Learner], z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        tau0, tau1, propensity = model
        e = np.clip(_mean_predict(propensity, z), self.clip, 1.0 - self.clip)
        return e * _mean_predict(tau0, z) + (1.0 - e) * _mean_predict(tau1, z)
