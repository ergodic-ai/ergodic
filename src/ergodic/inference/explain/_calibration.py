# SPDX-License-Identifier: Apache-2.0
"""The calibration / omnibus heterogeneity test (grf ``test_calibration``).

Regress the doubly-robust scores on two regressors built from the predicted effect:

    Gamma_i ~ beta_mean * mean.prediction + beta_diff * (tau_hat_i - mean.prediction),

where ``mean.prediction`` is the constant ``mean(tau_hat)`` and the differential
regressor is the centered prediction. ``beta_mean ~ 1`` says the average effect is
right; ``beta_diff ~ 1`` says the heterogeneity is well calibrated. The ONE-SIDED
p-value on ``beta_diff`` is an omnibus test for the presence of heterogeneity: if
``beta_diff`` is significantly above zero, reject "no usable heterogeneity." HC3
standard errors (a BLP on the DR scores). "Is there real heterogeneity, and is it
calibrated."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._linalg import _hc3_ols

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class CalibrationResult:
    """The two calibration coefficients and the one-sided HTE p-value.

    ``mean_coefficient`` (near 1 = the level is calibrated) and
    ``differential_coefficient`` (near 1 = the heterogeneity is calibrated) with
    their standard errors. ``heterogeneity_pvalue`` is the one-sided p for
    ``differential_coefficient > 0`` (small = reject "no heterogeneity").
    """

    mean_coefficient: float
    mean_se: float
    differential_coefficient: float
    differential_se: float
    heterogeneity_pvalue: float
    n: int

    def rejects(self, alpha: float = 0.05) -> bool:
        """True when the one-sided heterogeneity test rejects at ``alpha``."""
        return self.heterogeneity_pvalue < alpha


def _normal_sf(x: float) -> float:
    """The standard-normal survival function ``P(Z > x)`` via ``erfc``."""
    import math

    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _one_sided_pvalue(coef: float, se: float) -> float:
    """The one-sided ``P(coef > 0)`` p-value, degenerate when the SE is zero."""
    if se > 0.0:
        return _normal_sf(coef / se)
    return 1.0 if coef <= 0.0 else 0.0


def calibration_test(
    scores: npt.NDArray[np.float64],
    tau_hat: npt.NDArray[np.float64],
) -> CalibrationResult:
    """Regress the DR scores on the mean and differential predictions (HC3)."""
    n = scores.shape[0]
    mean_pred = float(np.mean(tau_hat))
    mean_col = np.full(n, mean_pred, dtype=np.float64)
    diff_col = tau_hat - mean_pred
    # A degenerate (constant) tau_hat has no differential signal: the regressor is
    # all-zero, so the differential coefficient is not identified. HC3 on the
    # rank-deficient design returns a float-noise coefficient with a near-zero SE,
    # whose sign (and thus the one-sided p) is arbitrary. Report the honest "no
    # usable heterogeneity in this ranking": a zero differential and p = 1.
    diff_scale = float(np.std(diff_col))
    if diff_scale <= 1e-12 * (abs(mean_pred) + 1.0):
        beta_mean, se_mean = _hc3_ols(mean_col.reshape(-1, 1), scores)
        return CalibrationResult(
            mean_coefficient=float(beta_mean[0]),
            mean_se=float(se_mean[0]),
            differential_coefficient=0.0,
            differential_se=float("nan"),
            heterogeneity_pvalue=1.0,
            n=n,
        )
    design = np.column_stack([mean_col, diff_col])
    beta, se = _hc3_ols(design, scores)
    diff_coef = float(beta[1])
    diff_se = float(se[1])
    pvalue = _one_sided_pvalue(diff_coef, diff_se)
    return CalibrationResult(
        mean_coefficient=float(beta[0]),
        mean_se=float(se[0]),
        differential_coefficient=diff_coef,
        differential_se=diff_se,
        heterogeneity_pvalue=pvalue,
        n=n,
    )
