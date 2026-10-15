# SPDX-License-Identifier: Apache-2.0
"""Front-door estimation through a mediator (binary treatment and mediator)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._estimator import Slot, Task, _FrontdoorBase
from .._learner import _mean_predict
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from .._learner import Learner


class TwoStageRegression(_FrontdoorBase):
    """Estimate the front-door functional from a mediator model and an outcome model.

    For binary treatment ``A`` and binary mediator ``M``: a classifier gives
    ``P(M | A)``, a regression gives ``E[Y | M, A]``, and the effect is the
    front-door sum ``E[Y | do(A=1)] - E[Y | do(A=0)]``. Continuous and multi-level
    mediators are not supported.
    """

    name = "two_stage_regression"
    slots = (Slot("mediator", Task.CLASSIFICATION), Slot("outcome", Task.REGRESSION))
    targets = frozenset({EffectTarget.ATE})

    def __init__(self, learners: Any = None, **kwargs: Any) -> None:
        super().__init__(learners, **kwargs)

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        a = arrays["A"]
        m = arrays["M"]
        y = arrays["Y"]
        mediator = make["mediator"]().fit(a.reshape(-1, 1), m)
        p_m1_a1 = float(_mean_predict(mediator, np.array([[1.0]]))[0])
        p_m1_a0 = float(_mean_predict(mediator, np.array([[0.0]]))[0])
        outcome = make["outcome"]().fit(np.column_stack([a, m, a * m]), y)

        def e_outcome(m_value: float, a_value: float) -> float:
            row = np.array([[a_value, m_value, a_value * m_value]], dtype=np.float64)
            return float(_mean_predict(outcome, row)[0])

        p_a1 = float(np.mean(a))

        def do(a_value: float) -> float:
            p_m1 = p_m1_a1 if a_value == 1.0 else p_m1_a0
            total = 0.0
            for m_value, p_m in ((1.0, p_m1), (0.0, 1.0 - p_m1)):
                inner = e_outcome(m_value, 1.0) * p_a1 + e_outcome(m_value, 0.0) * (1.0 - p_a1)
                total += p_m * inner
            return total

        return do(1.0) - do(0.0)
