# SPDX-License-Identifier: Apache-2.0
"""CLAN: Classification ANalysis of the most- vs least-affected groups.

For each covariate, compare its average in the most-affected group (the top
predicted-effect quantile) against the least-affected group (the bottom quantile),
and report the difference with a two-sample standard error on the fixed groups. A
significant difference says the covariate PROFILES the heterogeneity. "Who is in
the high-effect group." Complements GATES (how much) with who.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._gates import _quantile_groups

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class CLANResult:
    """Per-covariate mean differences between the extreme predicted-effect groups.

    ``names`` are the covariates. ``high_mean``/``low_mean`` are their averages in
    the top and bottom groups; ``difference`` is high minus low with ``difference_se``
    a two-sample standard error. :meth:`frame` lays them out.
    """

    names: tuple[str, ...]
    high_mean: npt.NDArray[np.float64]
    low_mean: npt.NDArray[np.float64]
    difference: npt.NDArray[np.float64]
    difference_se: npt.NDArray[np.float64]
    n_high: int
    n_low: int
    level: float = 0.9

    def interval(
        self, level: float | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-covariate difference intervals at ``level``."""
        from .._stats import _norm_ppf

        lvl = self.level if level is None else level
        z = _norm_ppf((1.0 + lvl) / 2.0)
        return self.difference - z * self.difference_se, self.difference + z * self.difference_se

    def frame(self) -> dict[str, dict[str, float]]:
        """A per-covariate dict of high/low means and the difference with its CI."""
        lo, hi = self.interval()
        return {
            name: {
                "high_mean": float(self.high_mean[i]),
                "low_mean": float(self.low_mean[i]),
                "difference": float(self.difference[i]),
                "std_error": float(self.difference_se[i]),
                "ci_low": float(lo[i]),
                "ci_high": float(hi[i]),
            }
            for i, name in enumerate(self.names)
        }


def clan(
    covariates: npt.NDArray[np.float64],
    covariate_names: tuple[str, ...],
    tau_hat: npt.NDArray[np.float64],
    *,
    n_groups: int = 5,
    level: float = 0.9,
) -> CLANResult:
    """Mean covariate differences between the top and bottom ``tau_hat`` groups."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    if covariates.shape[1] != len(covariate_names):
        raise ValueError("covariate matrix width does not match the covariate names")
    groups = _quantile_groups(tau_hat, n_groups)
    high = covariates[groups == n_groups - 1]
    low = covariates[groups == 0]
    n_high, n_low = high.shape[0], low.shape[0]
    high_mean = np.asarray(high.mean(axis=0), dtype=np.float64)
    low_mean = np.asarray(low.mean(axis=0), dtype=np.float64)
    difference = high_mean - low_mean
    var_high = high.var(axis=0, ddof=1) / n_high if n_high > 1 else np.zeros(high.shape[1])
    var_low = low.var(axis=0, ddof=1) / n_low if n_low > 1 else np.zeros(low.shape[1])
    difference_se = np.sqrt(np.asarray(var_high + var_low, dtype=np.float64))
    return CLANResult(
        names=covariate_names,
        high_mean=high_mean,
        low_mean=low_mean,
        difference=difference,
        difference_se=difference_se,
        n_high=n_high,
        n_low=n_low,
        level=level,
    )
