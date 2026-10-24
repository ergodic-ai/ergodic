# SPDX-License-Identifier: Apache-2.0
"""The generalized covariance measure: any regression learner becomes a CI test.

The GCM turns a conditional-mean regressor into a conditional-independence
test: regress each endpoint on the conditioning block, multiply the two
residual series elementwise, and check that the product has mean zero. With a
flexible regressor this catches the nonlinear dependence the partial
correlation misses. It stays blind where every covariance-based test is
blind: dependence that leaves the conditional covariance at zero (a pure
variance or higher-moment link) does not move the statistic.

The learner slot is the same surface as an estimator's nuisance slot, so any
sklearn regressor drops in, as does an alias string like ``"ols"`` or
``"knn"``. The default is the native
:class:`~ergodic.inference.KNNRegressor`.

Reference: Shah and Peters (2020), The hardness of conditional independence
testing and the generalised covariance measure.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ...inference import KNNRegressor
from ...inference._learner import _as_learner, _fresh
from .._ci import CITest
from .._stats import _norm_two_sided

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData
    from ...inference._learner import Learner

# Refuse residuals indistinguishable from an exact fit at floating-point precision.
_DEGENERACY_RTOL = 100.0 * np.finfo(np.float64).eps


class GCM(CITest):
    """The generalized covariance measure over a regression learner.

    Regresses x and y on z with a fresh clone of the learner per endpoint,
    forms the elementwise residual product, and reports
    ``sqrt(n) * mean(R) / sd(R)`` against the standard normal, two-sided.
    Residuals are in-sample, as in the paper's basic form. With an empty
    conditioning set the endpoints are centered instead, so the statistic is
    a plain covariance test. Constant endpoints, exact or numerically exact
    fits, and constant residual products raise ``ValueError``: the normal
    reference statistic is undefined, so no independence decision is returned.
    The default KNN learner is not generally calibrated: in a 200-seed
    direct-CI diagnostic it rejected 7.5-12% of nulls at nominal 5% across
    linear, quadratic and sine conditional means (n=200/1000). OLS was not
    a safe general replacement. See the discovery guide for the protocol.

    An in-sample learner must leave nondegenerate residual variation; this
    check alone does not establish calibration for an arbitrary learner.

    Args:
        data: The dataset to bind, or ``None`` for a reusable config.
        learner: The regression slot, accepting exactly what an estimator's
            learner slot accepts: any sklearn-style regressor or an alias
            string like ``"ols"`` or ``"knn"``. ``None`` means the native
            :class:`~ergodic.inference.KNNRegressor`, constructed lazily at
            first use.
    """

    # Two-sided tails use erfc; below binary64 range (e.g. |z|=40), p is zero.
    name: ClassVar[str] = "gcm"
    needs: ClassVar[str] = "numeric"

    def __init__(self, data: TabularData | None = None, *, learner: Any = None) -> None:
        self.learner: Any = learner
        self._template: Learner | None = None
        super().__init__(data)

    def statistic(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> tuple[float, float, float | None]:
        if x.shape[1] != 1 or y.shape[1] != 1:
            raise TypeError(
                "GCM needs scalar endpoints; a categorical endpoint expands to a "
                "one-hot block, use Mixed for categorical endpoints"
            )
        n = x.shape[0]
        if n < 2:
            raise ValueError("GCM needs at least two complete observations")
        if z.shape[1] == 0:
            rx = x[:, 0] - float(np.mean(x[:, 0]))
            ry = y[:, 0] - float(np.mean(y[:, 0]))
        else:
            rx = self._residual(z, x[:, 0])
            ry = self._residual(z, y[:, 0])
        rx = _checked_residual(rx, x[:, 0], "x")
        ry = _checked_residual(ry, y[:, 0], "y")
        product = rx * ry
        product_scale = float(np.max(np.abs(product)))
        if product_scale == 0.0:
            raise ValueError("GCM residual product has zero variance; no valid p-value")
        # Positive rescaling leaves the statistic unchanged and avoids a
        # variance floor that depends on the endpoints' measurement units.
        product = product / product_scale
        mean_r = float(np.mean(product))
        var_r = float(np.var(product))
        if var_r <= _DEGENERACY_RTOL**2:
            raise ValueError(
                "GCM residual product has zero or numerically degenerate variance; no valid p-value"
            )
        stat = math.sqrt(n) * mean_r / math.sqrt(var_r)
        p = _norm_two_sided(stat)
        return stat, p, None

    # -- internals -----------------------------------------------------------

    def _slot(self) -> Learner:
        """The coerced learner template: resolved once, cloned fresh per fit."""
        if self._template is None:
            spec = KNNRegressor() if self.learner is None else self.learner
            self._template = _as_learner(spec)
        return self._template

    def _residual(
        self, z: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The in-sample residual of ``target`` regressed on ``z``."""
        model = _fresh(self._slot())
        model.fit(z, target.ravel())
        prediction = np.asarray(model.predict(z), dtype=np.float64).reshape(-1)
        return np.asarray(target.ravel() - prediction, dtype=np.float64)


def _checked_residual(
    residual: npt.NDArray[np.float64], target: npt.NDArray[np.float64], endpoint: str
) -> npt.NDArray[np.float64]:
    """Check residual variation and rescale without changing the GCM statistic."""
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(target)):
        raise ValueError(f"GCM endpoint {endpoint} and its residuals must be finite")
    target_scale = float(np.max(np.abs(target - np.mean(target))))
    if target_scale == 0.0:
        raise ValueError(f"GCM endpoint {endpoint} is constant; no valid p-value")
    residual_scale = float(np.max(np.abs(residual)))
    if residual_scale <= _DEGENERACY_RTOL * target_scale:
        raise ValueError(
            f"GCM endpoint {endpoint} has zero or numerically degenerate residual variance; "
            "the in-sample learner may interpolate the training targets. "
            "Use a less flexible learner that leaves residual variation; no valid p-value"
        )
    scaled = residual / residual_scale
    if float(np.var(scaled)) <= _DEGENERACY_RTOL**2:
        raise ValueError(
            f"GCM endpoint {endpoint} has zero or numerically degenerate residual variance; "
            "no valid p-value"
        )
    return np.asarray(scaled, dtype=np.float64)
