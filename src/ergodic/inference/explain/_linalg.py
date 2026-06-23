# SPDX-License-Identifier: Apache-2.0
"""OLS with heteroskedasticity-robust (HC3) standard errors, numpy-only.

The Tier-B summaries are all least-squares projections of the doubly-robust scores
onto a low-dimensional design (covariates, group indicators, the prediction pair).
Each needs an honest, heteroskedasticity-robust covariance: the projection residual
variance is not constant across units. HC3 is the jackknife-shortcut White
covariance (MacKinnon-White 1985), the grf default for ``best_linear_projection``
and ``test_calibration``. Computed here with ``numpy.linalg`` so the explainability
layer stays dependency-light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _hc3_ols(
    design: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """OLS coefficients and HC3 standard errors for ``y ~ design``.

    Returns ``(beta[p], se[p])``. The design must already carry any intercept
    column. The HC3 sandwich inflates each residual by ``1 / (1 - h_ii)`` (the
    leverage), the standard heteroskedasticity-robust correction.
    """
    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    # leverages h_ii = x_i' (X'X)^{-1} x_i
    hat = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    hat = np.clip(hat, 0.0, 1.0 - 1e-10)
    adj = resid / (1.0 - hat)
    meat = (design * adj[:, None]).T @ (design * adj[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return np.asarray(beta, dtype=np.float64), np.asarray(se, dtype=np.float64)


def _weighted_mean_and_se(
    values: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """The sample mean of ``values`` and the standard error of that mean."""
    n = values.shape[0]
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se
