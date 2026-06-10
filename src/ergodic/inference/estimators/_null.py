# SPDX-License-Identifier: Apache-2.0
"""The null estimator: a graph with no directed path fixes the effect at zero."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._estimator import NULL, CausalEstimator, _single
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._learner import Learner


class NullEffect(CausalEstimator):
    """Report the exact zero effect for a ``null`` estimand.

    Identification returns ``strategy="null"`` when the graph has no directed
    path from the treatment to the outcome, so the intervention cannot move the
    outcome and the effect is zero by construction. No nuisance slots, no
    uncertainty: the point is 0.0 with a zero standard error.
    """

    name = "null_effect"
    consumes = frozenset({NULL})
    targets = frozenset({EffectTarget.ATE, EffectTarget.ATT})

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        missing = [name for name in (treatment, outcome) if name not in data.schema]
        if missing:
            raise ValueError(f"columns {missing} are not in the data")
        return {}, False, {}

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        return 0.0

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        return 0.0, np.zeros(self._state.n, dtype=np.float64)
