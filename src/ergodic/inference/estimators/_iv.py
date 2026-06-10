# SPDX-License-Identifier: Apache-2.0
"""Instrumental-variable estimation: two-stage least squares and the Wald ratio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._crossfit import _fit_predict, _folds, _train_mask
from .._estimator import Slot, Task, _IVBase
from .._learner import _mean_predict
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._learner import Learner


def _two_sls(
    y: npt.NDArray[np.float64],
    endog: npt.NDArray[np.float64],
    instr: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve ``Y = endog @ beta`` with instruments ``instr``; return ``beta`` and influence.

    The influence rows give the robust (sandwich) variance: row ``i`` is
    ``(X_hat' X / n)^{-1} x_hat_i u_i``, so ``std(influence[:, j]) / sqrt(n)`` is the
    standard error of coefficient ``j``.
    """
    n = y.shape[0]
    pinv_ztz = np.linalg.pinv(instr.T @ instr)
    proj = instr @ (pinv_ztz @ (instr.T @ endog))  # X_hat = P_Z X
    cross = proj.T @ endog  # X_hat' X = X_hat' X_hat
    beta = np.asarray(np.linalg.solve(cross, proj.T @ y), dtype=np.float64)
    resid = y - endog @ beta
    bread = np.linalg.pinv(cross / n)
    influence = (proj * resid[:, None]) @ bread.T
    return beta, np.asarray(influence, dtype=np.float64)


class TwoStageLeastSquares(_IVBase):
    """Two-stage least squares. OLS in both stages, robust standard errors.

    The first stage regresses the treatment on the instruments and any
    conditioning covariates; the second regresses the outcome on the fitted
    treatment. Reports the local average treatment effect (LATE).
    """

    name = "two_stage_least_squares"
    targets = frozenset({EffectTarget.LATE})

    def __init__(self, learners: Any = None, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)

    def _design(
        self, arrays: dict[str, npt.NDArray[np.float64]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        a = arrays["A"]
        y = arrays["Y"]
        zi = arrays["Zi"]
        w = arrays["W"]
        n = y.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        endog = np.hstack([ones, w, a])
        instr = np.hstack([ones, w, zi])
        t_index = 1 + w.shape[1]
        return y, endog, instr, t_index

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        y, endog, instr, t_index = self._design(arrays)
        beta, _ = _two_sls(y, endog, instr)
        return float(beta[t_index])

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        y, endog, instr, t_index = self._design(arrays)
        beta, influence = _two_sls(y, endog, instr)
        return float(beta[t_index]), influence[:, t_index]


class Wald(TwoStageLeastSquares):
    """The Wald ratio: two-stage least squares with a single instrument."""

    name = "wald"

    def _iv_columns(
        self, instruments: list[str], conditioning: list[str]
    ) -> tuple[list[str], list[str]]:
        if len(instruments) != 1:
            raise NotImplementedError(
                "Wald needs exactly one instrument; use two_stage_least_squares for more"
            )
        return instruments, conditioning


class PLIV(_IVBase):
    """Orthogonal instrumental variables (the partially linear IV model).

    Residualize the outcome, treatment, and instrument against the covariates with
    cross-fitting, then the IV moment on the residuals. The flexible-learner
    instrument estimator that two-stage least squares is not. Reports LATE with an
    analytic standard error.
    """

    name = "pliv"
    slots = (
        Slot("outcome", Task.REGRESSION),
        Slot("treatment", Task.REGRESSION),
        Slot("instrument", Task.REGRESSION),
    )
    targets = frozenset({EffectTarget.LATE})

    def __init__(self, learners: Any = None, *, n_folds: int = 5, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)
        self.n_folds = n_folds
        self._cache: tuple[float, npt.NDArray[np.float64]] | None = None

    def _residualize(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        a = arrays["A"].reshape(-1)
        y = arrays["Y"]
        zi = arrays["Zi"]
        w = arrays["W"]
        n = y.shape[0]
        res_y = np.empty(n, dtype=np.float64)
        res_t = np.empty(n, dtype=np.float64)
        res_z = np.empty((n, zi.shape[1]), dtype=np.float64)
        rng = np.random.default_rng(self._random_state)
        has_w = w.shape[1] > 0
        for test in _folds(n, self.n_folds, rng):
            train = _train_mask(n, test)
            if has_w:
                res_y[test] = y[test] - _fit_predict(make["outcome"], w[train], y[train], w[test])
                res_t[test] = a[test] - _fit_predict(make["treatment"], w[train], a[train], w[test])
                for j in range(zi.shape[1]):
                    fitted = make["instrument"]().fit(w[train], zi[train, j])
                    res_z[test, j] = zi[test, j] - _mean_predict(fitted, w[test])
            else:
                res_y[test] = y[test] - float(np.mean(y[train]))
                res_t[test] = a[test] - float(np.mean(a[train]))
                for j in range(zi.shape[1]):
                    res_z[test, j] = zi[test, j] - float(np.mean(zi[train, j]))
        return res_y, res_t, res_z

    def _solve(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        res_y, res_t, res_z = self._residualize(arrays, make)
        beta, influence = _two_sls(res_y, res_t.reshape(-1, 1), res_z)
        return float(beta[0]), influence[:, 0]

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        self._cache = self._solve(self._state.arrays, self._state.make)
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
        return self._cache
