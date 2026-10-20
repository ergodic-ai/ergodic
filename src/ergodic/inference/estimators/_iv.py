# SPDX-License-Identifier: Apache-2.0
"""Instrumental-variable estimation: two-stage least squares and the Wald ratio."""

from __future__ import annotations

import warnings
from dataclasses import replace
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
    from .._result import EffectEstimate


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


def _first_stage_diagnostics(
    a: npt.NDArray[np.float64],
    zi: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Classical partial F and R-squared; descriptive, not weak-IV robust inference."""
    a = a.reshape(-1)
    base = np.column_stack([np.ones(a.size), w])
    full = np.column_stack([base, zi])
    q = int(np.linalg.matrix_rank(full) - np.linalg.matrix_rank(base))
    df = int(a.size - np.linalg.matrix_rank(full))
    restricted = a - base @ np.linalg.lstsq(base, a, rcond=None)[0]
    residual = a - full @ np.linalg.lstsq(full, a, rcond=None)[0]
    rss0, rss1 = float(restricted @ restricted), float(residual @ residual)
    if q <= 0 or df <= 0 or rss0 <= 0:
        raise ValueError(
            "first stage has no identified instrument variation or residual degrees of freedom"
        )
    improvement = max(0.0, rss0 - rss1)
    f = improvement / q / (rss1 / df) if rss1 > 0 else float("inf")
    if f < 10:
        warnings.warn(
            "weak first stage (classical partial F < 10); ordinary normal IV intervals "
            "can be unreliable. This diagnostic does not establish instrument validity "
            "or replace weak-IV robust inference.",
            UserWarning,
            stacklevel=3,
        )
    return {
        "first_stage_partial_r2": improvement / rss0,
        "first_stage_f": f,
        "first_stage_numerator_df": float(q),
        "first_stage_residual_df": float(df),
        "weak_first_stage": float(f < 10),
    }


class _IVInferenceBase(_IVBase):
    def __init__(
        self, learners: Any = None, *, late_assumptions: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(learners, **kwargs)
        if not isinstance(late_assumptions, bool):
            raise TypeError("late_assumptions must be a boolean acknowledgement")
        self.late_assumptions = late_assumptions

    def _pick_target(self, target: EffectTarget | str | None) -> EffectTarget:
        chosen = super()._pick_target(EffectTarget.LINEAR_IV if target is None else target)
        if chosen is EffectTarget.LATE:
            a, zi = self._state.arrays["A"], self._state.arrays["Zi"]
            if not self.late_assumptions:
                raise ValueError(
                    "target='late' requires late_assumptions=True: acknowledge instrument "
                    "exogeneity, exclusion, relevance and monotonicity"
                )
            if (
                set(np.unique(a).tolist()) != {0.0, 1.0}
                or zi.shape[1] != 1
                or set(np.unique(zi).tolist()) != {0.0, 1.0}
            ):
                raise ValueError(
                    "qualified LATE requires binary treatment and one binary instrument"
                )
            if self._state.arrays["W"].shape[1]:
                raise ValueError(
                    "qualified LATE currently requires an unconditional binary instrument; "
                    "use linear_iv for covariate-adjusted IV"
                )
        return chosen

    def effect(self, *, target: EffectTarget | str | None = None) -> EffectEstimate:
        result = super().effect(target=target)
        assumptions: tuple[str, ...] = (
            "Instrument exogeneity and exclusion are substantive assumptions, not verified "
            "by the graph or first-stage statistic.",
            "Instrument relevance is required; ordinary normal intervals are not weak-IV robust.",
        )
        if result.target is EffectTarget.LATE:
            assumptions += (
                "Binary instrument and treatment, monotonicity (no defiers), consistency "
                "and no interference: LATE for instrument-induced compliers; asserted by "
                "the caller.",
            )
        else:
            assumptions += (
                "Linear IV coefficient under the stated linear/partially linear structural "
                "model; no automatic complier-LATE interpretation.",
            )
        return replace(result, identification=self._identification, assumptions=assumptions)


class TwoStageLeastSquares(_IVInferenceBase):
    """Two-stage least squares. OLS in both stages, robust standard errors.

    The first stage regresses the treatment on the instruments and any
    conditioning covariates; the second regresses the outcome on the fitted
    treatment. Reports a linear IV coefficient by default. An unconditional binary
    instrument and treatment can report ``target="late"`` only with caller-asserted
    ``late_assumptions=True`` (exogeneity, exclusion, relevance and monotonicity).
    """

    name = "two_stage_least_squares"
    targets = frozenset({EffectTarget.LINEAR_IV, EffectTarget.LATE})

    def __init__(self, learners: Any = None, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        arrays = self._state.arrays
        self._state.diagnostics.update(
            _first_stage_diagnostics(arrays["A"], arrays["Zi"], arrays["W"])
        )
        return self

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


class PLIV(_IVInferenceBase):
    """Orthogonal instrumental variables (the partially linear IV model).

    Residualize the outcome, treatment, and instrument against the covariates with
    cross-fitting, then the IV moment on the residuals. The flexible-learner
    instrument estimator that two-stage least squares is not. Reports a partially
    linear IV coefficient with an analytic standard error, never an automatic LATE.
    """

    name = "pliv"
    slots = (
        Slot("outcome", Task.REGRESSION),
        Slot("treatment", Task.REGRESSION),
        Slot("instrument", Task.REGRESSION),
    )
    targets = frozenset({EffectTarget.LINEAR_IV})

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
        if arrays is self._state.arrays:
            self._state.diagnostics.update(
                _first_stage_diagnostics(res_t, res_z, np.empty((res_t.size, 0), dtype=np.float64))
            )
            self._state.diagnostics["first_stage_residualized"] = 1.0
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
