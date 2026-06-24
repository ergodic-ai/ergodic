# SPDX-License-Identifier: Apache-2.0
"""The ``Heterogeneity`` record: Tier-B characterization off one orthogonal object.

Returned by :meth:`~ergodic.inference.CATEResult.explain`. Every method below reads
the SAME cached doubly-robust scores, covariate design, and fitted ``tau_hat``, so
the cross-fitting and orthogonality are consistent across the tools. This is the
load-bearing inference discipline from the literature: the CATE point predictions
are only inputs (a fixed proxy / ranking), never the object of inference; the honest
quantity is a low-dimensional summary of the AIPW signal.

The five tools:

- ``blp`` Best Linear Projection of the CATE on covariates (which covariates
  associate with a bigger effect).
- ``gates`` Sorted Group ATEs (does the true effect rise across predicted groups).
- ``clan`` Classification Analysis (who is in the high-effect group).
- ``calibration_test`` the omnibus heterogeneity test (is there real, calibrated HTE).
- ``rate`` the Rank-Weighted ATE, AUTOC or Qini (is the prioritization rule good).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._blp import BLPResult, best_linear_projection
from ._calibration import CalibrationResult, calibration_test
from ._clan import CLANResult, clan
from ._gates import GATESResult, gates
from ._rate import RATEResult, rate

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .._scores import DRScores
    from ._policy import PolicyTree


@dataclass(frozen=True)
class Heterogeneity:
    """Tier-B heterogeneity characterization on shared doubly-robust scores.

    Carries the cross-fit :class:`~ergodic.inference._scores.DRScores` and the fitted
    per-unit ``tau_hat``. The honesty register: the rankings and proxies are fixed
    (computed cross-fit), so the orthogonal summaries below inherit valid inference
    while the CATE point predictions themselves never carry a confidence interval.
    """

    scores: DRScores
    tau_hat: npt.NDArray[np.float64]
    treatment: str
    outcome: str

    def blp(self, *, level: float = 0.9) -> BLPResult:
        """Best Linear Projection of the CATE on the covariates (HC3 inference)."""
        return best_linear_projection(
            self.scores.scores,
            self.scores.covariates,
            self.scores.covariate_names,
            level=level,
        )

    def gates(self, *, n_groups: int = 5, level: float = 0.9) -> GATESResult:
        """Sorted Group ATEs across ``n_groups`` predicted-effect quantiles."""
        return gates(self.scores.scores, self.tau_hat, n_groups=n_groups, level=level)

    def clan(self, *, n_groups: int = 5, level: float = 0.9) -> CLANResult:
        """Covariate-mean differences between the most- and least-affected groups."""
        return clan(
            self.scores.covariates,
            self.scores.covariate_names,
            self.tau_hat,
            n_groups=n_groups,
            level=level,
        )

    def calibration_test(self) -> CalibrationResult:
        """The omnibus calibration / heterogeneity test (one-sided HTE p-value)."""
        return calibration_test(self.scores.scores, self.tau_hat)

    def rate(
        self, metric: str = "autoc", *, n_boot: int = 200, random_state: int | None = None
    ) -> RATEResult:
        """The Rank-Weighted ATE of the ``tau_hat`` ranking: ``"autoc"`` or ``"qini"``."""
        return rate(
            self.scores.scores,
            self.tau_hat,
            metric=metric,
            n_boot=n_boot,
            random_state=random_state,
        )

    def policy_tree(
        self,
        covariates: str | tuple[str, ...] | list[str] | None = None,
        *,
        depth: int = 2,
    ) -> PolicyTree:
        """The depth-limited, DR-welfare-optimal policy tree off these scores.

        A convenience that forwards the shared doubly-robust scores to
        :func:`~ergodic.inference.policy_tree`. ``covariates`` selects the split
        columns by name (``None`` uses all of them). The "act" companion to
        :meth:`rate`: where ``rate`` scores a ranking, this returns an interpretable
        treat-or-not rule and its estimated policy value.
        """
        from ._policy import policy_tree as _policy_tree

        return _policy_tree(self.scores, covariates, depth=depth)
