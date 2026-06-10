# SPDX-License-Identifier: Apache-2.0
"""Augmented inverse propensity weighting (AIPW), the doubly-robust estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._estimator import Slot, Task, _AdjustmentBase, _dirichlet_weights
from .._learner import _mean_predict, _mean_predict_draws
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from .._estimator import _FitState
    from .._learner import Learner


def _aipw_ate_scores(
    a: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    mu1: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Efficient influence scores for the ATE; their mean is the AIPW point."""
    scores = (mu1 - mu0) + a * (y - mu1) / e - (1.0 - a) * (y - mu0) / (1.0 - e)
    return np.asarray(scores, dtype=np.float64)


def _aipw_att_scores(
    a: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    p1: float,
) -> npt.NDArray[np.float64]:
    """Influence scores for the ATT; their mean is the AIPW-ATT point."""
    scores = (a * (y - mu0) - (1.0 - a) * (e / (1.0 - e)) * (y - mu0)) / p1
    return np.asarray(scores, dtype=np.float64)


class AIPW(_AdjustmentBase):
    """Combine an outcome model and a propensity model.

    Consistent if either model is right. The ATE standard error comes from the
    efficient influence function (Neyman-orthogonal, so first-order insensitive to
    nuisance error); the ATT standard error falls back to bootstrap.
    """

    name = "aipw"
    slots = (Slot("outcome", Task.REGRESSION), Slot("propensity", Task.CLASSIFICATION))
    targets = frozenset({EffectTarget.ATE, EffectTarget.ATT})

    def __init__(
        self,
        learners: Any = None,
        *,
        clip: float = 1e-3,
        bayesian_bootstrap: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(learners, **kwargs)
        self.clip = clip
        self.bayesian_bootstrap = bayesian_bootstrap

    def _nuisance(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        n = a.shape[0]
        outcome = make["outcome"]().fit(np.hstack([a.reshape(-1, 1), z]), y)
        ones = np.ones((n, 1), dtype=np.float64)
        zeros = np.zeros((n, 1), dtype=np.float64)
        mu1 = _mean_predict(outcome, np.hstack([ones, z]))
        mu0 = _mean_predict(outcome, np.hstack([zeros, z]))
        propensity = make["propensity"]().fit(z, a)
        e = np.clip(_mean_predict(propensity, z), self.clip, 1.0 - self.clip)
        return mu0, mu1, e

    def _scores(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> npt.NDArray[np.float64]:
        a = arrays["A"]
        y = arrays["Y"]
        mu0, mu1, e = self._nuisance(arrays, make)
        if target is EffectTarget.ATT:
            p1 = float(np.mean(a))
            if p1 <= 0.0:
                raise ValueError("no treated units for ATT")
            return _aipw_att_scores(a, y, mu0, e, p1)
        return _aipw_ate_scores(a, y, mu0, mu1, e)

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        return float(np.mean(self._scores(arrays, target, make)))

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        if target is not EffectTarget.ATE:
            return None
        psi = self._scores(arrays, target, make)
        return float(np.mean(psi)), psi

    def _posterior(self, state: _FitState, target: EffectTarget) -> npt.NDArray[np.float64] | None:
        """The doubly-robust posterior: the influence scores per nuisance draw.

        Fit the outcome and propensity once, read their ``(S, n)`` draws, and for
        each draw evaluate the same influence scores as the point path. With
        ``bayesian_bootstrap`` (the default) reweight the rows by a Dirichlet(1)
        draw, so the spread carries the nuisance posterior and the sampling
        uncertainty together. Needs both nuisances Bayesian; a mix returns ``None``.
        """
        if not all(state.bayesian.values()):
            return None
        arrays = state.arrays
        a, y, z = arrays["A"], arrays["Y"], arrays["Z"]
        n = a.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        zeros = np.zeros((n, 1), dtype=np.float64)
        outcome = state.make["outcome"]().fit(np.hstack([a.reshape(-1, 1), z]), y)
        mu1d = _mean_predict_draws(outcome, np.hstack([ones, z]))
        mu0d = _mean_predict_draws(outcome, np.hstack([zeros, z]))
        propensity = state.make["propensity"]().fit(z, a)
        ed = np.clip(_mean_predict_draws(propensity, z), self.clip, 1.0 - self.clip)
        n_draws = min(mu1d.shape[0], mu0d.shape[0], ed.shape[0])
        p1 = float(np.mean(a))
        if target is EffectTarget.ATT and p1 <= 0.0:
            raise ValueError("no treated units for ATT")
        rng = np.random.default_rng(self._random_state)
        draws = np.empty(n_draws, dtype=np.float64)
        for s in range(n_draws):
            if target is EffectTarget.ATT:
                phi = _aipw_att_scores(a, y, mu0d[s], ed[s], p1)
            else:
                phi = _aipw_ate_scores(a, y, mu0d[s], mu1d[s], ed[s])
            if self.bayesian_bootstrap:
                draws[s] = float(np.sum(_dirichlet_weights(n, rng) * phi))
            else:
                draws[s] = float(np.mean(phi))
        return draws
