# SPDX-License-Identifier: Apache-2.0
"""RATE: the Rank-Weighted Average Treatment Effect (AUTOC and Qini).

Order the units by a prioritization rule ``S`` (here the predicted effect
``tau_hat``), build the Targeting Operator Characteristic from the doubly-robust
scores,

    TOC(q) = mean of Gamma_i over the top q fraction by S, minus the overall mean,

and integrate it against a weight ``alpha(q)``: AUTOC takes ``alpha(q) = 1`` (most
powerful when a small subgroup has large effects), Qini takes ``alpha(q) = q`` (most
powerful when the heterogeneity is diffuse). The RATE is positive when the rule puts
the high-benefit units at the top; a one-sided test of ``RATE > 0`` is an omnibus
heterogeneity test that directly targets rank quality. Standard errors come from a
seeded half-sample bootstrap (the grf default). "Is this prioritization rule good,
and is there heterogeneity to exploit."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class RATEResult:
    """A rank-weighted ATE with a bootstrap standard error and a one-sided p-value.

    ``estimate`` is the AUTOC or Qini point; ``std_error`` its half-sample-bootstrap
    standard error; ``pvalue`` the one-sided ``estimate > 0`` p (an HTE test). ``toc``
    and ``toc_fractions`` are the TOC curve for plotting.
    """

    metric: str  # "autoc" or "qini"
    estimate: float
    std_error: float
    pvalue: float
    n: int
    toc: npt.NDArray[np.float64]
    toc_fractions: npt.NDArray[np.float64]

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """A normal confidence interval at ``level``."""
        from .._stats import _norm_ppf

        z = _norm_ppf((1.0 + level) / 2.0)
        return self.estimate - z * self.std_error, self.estimate + z * self.std_error

    def rejects(self, alpha: float = 0.05) -> bool:
        """True when the one-sided ``RATE > 0`` test rejects at ``alpha``."""
        return self.pvalue < alpha


def _toc_curve(
    scores: npt.NDArray[np.float64], priority: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The TOC at fractions ``j/n``: cumulative mean of the top-ranked scores minus the mean."""
    order = np.argsort(-priority, kind="stable")
    ordered = scores[order]
    n = ordered.shape[0]
    cumulative = np.cumsum(ordered) / np.arange(1, n + 1, dtype=np.float64)
    return np.asarray(cumulative - float(np.mean(scores)), dtype=np.float64)


def _rate_from_scores(
    scores: npt.NDArray[np.float64], priority: npt.NDArray[np.float64], metric: str
) -> float:
    """Integrate the TOC against the metric weight (a 1/n Riemann sum)."""
    toc = _toc_curve(scores, priority)
    n = toc.shape[0]
    fractions = np.arange(1, n + 1, dtype=np.float64) / n
    weight = fractions if metric == "qini" else np.ones(n, dtype=np.float64)
    return float(np.mean(weight * toc))


def rate(
    scores: npt.NDArray[np.float64],
    tau_hat: npt.NDArray[np.float64],
    *,
    metric: str = "autoc",
    n_boot: int = 200,
    random_state: int | None = None,
) -> RATEResult:
    """The AUTOC or Qini of the ``tau_hat`` ranking, with a half-sample-bootstrap SE."""
    if metric not in ("autoc", "qini"):
        raise ValueError(f"metric must be 'autoc' or 'qini', got {metric!r}")
    n = scores.shape[0]
    if n < 4:
        raise ValueError(f"rate needs at least 4 units, got {n}")
    estimate = _rate_from_scores(scores, tau_hat, metric)
    fractions = np.arange(1, n + 1, dtype=np.float64) / n
    toc = _toc_curve(scores, tau_hat)

    rng = np.random.default_rng(random_state)
    half = max(2, n // 2)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.choice(n, size=half, replace=False)
        boot[b] = _rate_from_scores(scores[idx], tau_hat[idx], metric)
    # half-sample draws have ~2x the variance of the full-sample estimator
    std_error = float(np.std(boot, ddof=1) / np.sqrt(2.0))
    from ._calibration import _one_sided_pvalue

    pvalue = _one_sided_pvalue(estimate, std_error)
    return RATEResult(
        metric=metric,
        estimate=estimate,
        std_error=std_error,
        pvalue=pvalue,
        n=n,
        toc=toc,
        toc_fractions=fractions,
    )
