# SPDX-License-Identifier: Apache-2.0
"""GATES: Sorted Group Average Treatment Effects.

Partition the units by quantiles of the predicted effect ``tau_hat`` into ``K``
groups, then report the TRUE effect within each group as the mean of the
doubly-robust scores ``Gamma_i`` over that group (a within-group average is a valid
estimate of that group's ATE because the grouping is a fixed proxy). The monotone
pattern ``gamma_1 <= ... <= gamma_K`` is the effect gradient; the headline contrast
is ``gamma_K - gamma_1`` (most affected minus least affected), significantly above
zero when heterogeneity is present. "Does the true effect rise across predicted
groups." The nonparametric, group-indicator special case of the BLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._linalg import _weighted_mean_and_se

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class GATESResult:
    """Per-group effects and the top-minus-bottom contrast.

    ``group_effect``/``group_se`` are length ``n_groups`` (group 1 the lowest
    predicted effect). ``difference``/``difference_se`` are the top-minus-bottom
    contrast and its standard error (the two groups are disjoint, so the variances
    add). ``counts`` are the group sizes.
    """

    n_groups: int
    group_effect: npt.NDArray[np.float64]
    group_se: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    difference: float
    difference_se: float
    level: float = 0.9

    def interval(self, level: float | None = None) -> tuple[float, float]:
        """The top-minus-bottom contrast interval at ``level``."""
        from .._stats import _norm_ppf

        lvl = self.level if level is None else level
        z = _norm_ppf((1.0 + lvl) / 2.0)
        return self.difference - z * self.difference_se, self.difference + z * self.difference_se

    def is_monotone(self) -> bool:
        """True when the group effects are non-decreasing across the predicted groups."""
        return bool(np.all(np.diff(self.group_effect) >= 0.0))


def _quantile_groups(tau_hat: npt.NDArray[np.float64], n_groups: int) -> npt.NDArray[np.int64]:
    """Assign each unit to a quantile group of ``tau_hat`` (0 lowest, n_groups-1 top).

    Ranks break ties so groups stay near-equal in size even with a degenerate
    (constant) ``tau_hat``, where every quantile edge collapses.
    """
    n = tau_hat.shape[0]
    order = np.argsort(tau_hat, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    edges = (ranks / n * n_groups).astype(np.int64)
    return np.asarray(np.clip(edges, 0, n_groups - 1), dtype=np.int64)


def gates(
    scores: npt.NDArray[np.float64],
    tau_hat: npt.NDArray[np.float64],
    *,
    n_groups: int = 5,
    level: float = 0.9,
) -> GATESResult:
    """Group by predicted-effect quantiles and average the DR scores within each."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    n = scores.shape[0]
    if n < n_groups:
        raise ValueError(f"need at least n_groups={n_groups} units, got {n}")
    groups = _quantile_groups(tau_hat, n_groups)
    effect = np.empty(n_groups, dtype=np.float64)
    se = np.empty(n_groups, dtype=np.float64)
    counts = np.empty(n_groups, dtype=np.int64)
    for g in range(n_groups):
        members = scores[groups == g]
        counts[g] = members.shape[0]
        effect[g], se[g] = _weighted_mean_and_se(members)
    difference = float(effect[-1] - effect[0])
    difference_se = float(np.sqrt(se[-1] ** 2 + se[0] ** 2))
    return GATESResult(
        n_groups=n_groups,
        group_effect=effect,
        group_se=se,
        counts=counts,
        difference=difference,
        difference_se=difference_se,
        level=level,
    )
