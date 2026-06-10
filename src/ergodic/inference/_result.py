# SPDX-License-Identifier: Apache-2.0
"""The estimation result: one shape for a point estimate and a posterior alike.

A point estimate is a degenerate distribution, so :class:`EffectEstimate` carries
draws and treats the point as the special case. ``value`` is the analytic point or
the mean of the draws, ``std_error`` is the analytic standard error or their
standard deviation, and ``interval`` is the normal interval or the draw quantiles.
The ``kind`` tag records the source. A bootstrap interval and a Bayesian credible
interval run the same code over the same array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._stats import _norm_ppf
from ._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt


@dataclass(frozen=True)
class EffectEstimate:
    """A causal effect with its uncertainty.

    Built by a :class:`CausalEstimator`. Either ``_point``/``_std_error`` are set
    (an analytic result) or ``_draws`` is set (bootstrap or posterior), and the
    accessors read whichever is present.
    """

    target: EffectTarget
    estimator: str
    learners: Mapping[str, str]
    n: int
    kind: str  # "analytic", "bootstrap", or "posterior"
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    _point: float | None = None
    _std_error: float | None = None
    _draws: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def value(self) -> float:
        """The point estimate, or the mean of the draws."""
        if self._point is not None:
            return self._point
        if self._draws is not None:
            return float(np.mean(self._draws))
        raise ValueError("estimate has neither a point nor draws")

    @property
    def std_error(self) -> float | None:
        """The analytic standard error, or the standard deviation of the draws."""
        if self._std_error is not None:
            return self._std_error
        if self._draws is not None:
            return float(np.std(self._draws, ddof=1))
        return None

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """A confidence or credible interval at ``level``.

        From the draw quantiles when draws are present, else the normal interval
        around the point estimate.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if self._draws is not None:
            lo = float(np.quantile(self._draws, (1.0 - level) / 2.0))
            hi = float(np.quantile(self._draws, (1.0 + level) / 2.0))
            return (lo, hi)
        if self._point is not None and self._std_error is not None:
            z = _norm_ppf((1.0 + level) / 2.0)
            return (self._point - z * self._std_error, self._point + z * self._std_error)
        raise ValueError("estimate has no uncertainty to form an interval from")

    @property
    def draws(self) -> npt.NDArray[np.float64] | None:
        """The bootstrap or posterior draws, or ``None`` for a pure analytic result."""
        return self._draws

    def __repr__(self) -> str:
        se = self.std_error
        body = f"{self.target}={self.value:.4g}"
        if se is not None:
            lo, hi = self.interval()
            body += f", se={se:.3g}, 95% CI [{lo:.3g}, {hi:.3g}]"
        return f"EffectEstimate({self.estimator}, {self.kind}: {body}, n={self.n})"


def _summary_fields(estimate: EffectEstimate) -> dict[str, Any]:
    """A flat dict of the estimate, handy for tables and tests."""
    lo: float | None = None
    hi: float | None = None
    if estimate._draws is not None or estimate._std_error is not None:
        lo, hi = estimate.interval()
    return {
        "target": str(estimate.target),
        "estimator": estimate.estimator,
        "value": estimate.value,
        "std_error": estimate.std_error,
        "ci_low": lo,
        "ci_high": hi,
        "n": estimate.n,
        "kind": estimate.kind,
    }
