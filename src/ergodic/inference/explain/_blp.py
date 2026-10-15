# SPDX-License-Identifier: Apache-2.0
"""Best Linear Projection of the CATE on covariates (Semenova-Chernozhukov / grf).

Regress the doubly-robust scores ``Gamma_i`` on an intercept plus the covariates,

    Gamma_i ~ beta_0 + A_i' beta,

and each slope is the linear association between a covariate and the CATE. The
covariates enter as they are, not centred, so the intercept is the projected effect
at covariates all zero; it estimates the ATE only when every covariate has mean
zero. For the ATE read :meth:`~ergodic.inference.CATEResult.ate` or
:func:`~ergodic.inference.estimate_effect`. Inference is heteroskedasticity-robust (HC3, the
jackknife-shortcut White covariance), which is valid because ``Gamma_i`` is an
orthogonal signal: the second-stage projection is first-order insensitive to the
cross-fit nuisance error. "Which covariates associate with a bigger effect."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._linalg import _hc3_ols

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class BLPResult:
    """The best-linear-projection coefficients with HC3 confidence intervals.

    ``names`` are the regressors (``"(intercept)"`` first, then the covariates).
    ``coef``/``std_error`` are aligned to ``names``; the slopes are
    covariate-to-effect associations, and the intercept is the projected effect at
    covariates all zero (the ATE only when the covariates are centred).
    :meth:`interval` and :meth:`frame` read them.
    """

    names: tuple[str, ...]
    coef: npt.NDArray[np.float64]
    std_error: npt.NDArray[np.float64]
    n: int
    level: float = 0.9

    def interval(
        self, level: float | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Coefficient intervals at ``level`` (the construction default otherwise)."""
        from .._stats import _norm_ppf

        lvl = self.level if level is None else level
        if not 0.0 < lvl < 1.0:
            raise ValueError(f"level must be in (0, 1), got {lvl}")
        z = _norm_ppf((1.0 + lvl) / 2.0)
        return self.coef - z * self.std_error, self.coef + z * self.std_error

    def frame(self) -> dict[str, dict[str, float]]:
        """A per-coefficient dict of estimate/se/ci, handy for tables and tests."""
        lo, hi = self.interval()
        return {
            name: {
                "coef": float(self.coef[i]),
                "std_error": float(self.std_error[i]),
                "ci_low": float(lo[i]),
                "ci_high": float(hi[i]),
            }
            for i, name in enumerate(self.names)
        }


def best_linear_projection(
    scores: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64],
    covariate_names: tuple[str, ...],
    *,
    level: float = 0.9,
) -> BLPResult:
    """OLS of the DR scores on ``[1, covariates]`` with HC3 standard errors."""
    n = scores.shape[0]
    design = np.hstack([np.ones((n, 1), dtype=np.float64), covariates])
    names = ("(intercept)", *covariate_names)
    beta, se = _hc3_ols(design, scores)
    return BLPResult(names=names, coef=beta, std_error=se, n=n, level=level)
