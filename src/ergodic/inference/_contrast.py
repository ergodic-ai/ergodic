# SPDX-License-Identifier: Apache-2.0
"""Explicit dose contrasts; existing estimators' slope contracts stay unchanged."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from ._perunit import CATEResult
from ._result import EffectEstimate
from ._target import EffectTarget

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class DoseContrast(EffectEstimate):
    """E[Y(value) - Y(baseline)] for the fitted context population.

    ``protocol`` identifies the numerical procedure, not a graph truth claim.
    Continuous DR intervals are pointwise asymptotic intervals: smoothing bias,
    graph selection and simultaneous comparisons are not included.
    """

    baseline: float = 0.0
    treatment_value: float = 1.0
    treatment: str = ""
    outcome: str = ""
    protocol: str = ""


@dataclass(frozen=True, eq=False)
class DoseEffects(CATEResult):
    """Conditional dose contrasts in query-row order, not structural ITEs."""

    baseline: float = 0.0
    treatment_value: float = 1.0
    protocol: str = ""
    _shared_effect: DoseContrast | None = field(default=None, repr=False)

    def ate(self) -> DoseContrast:
        """Average over these rows, preserving a shared PL slope's uncertainty.

        Continuous DR query predictions carry no interval. Their mean is a
        plug-in summary of these query rows, not the fitted population DR mean.
        """
        if not self.n:
            raise ValueError("cannot average an empty query population")
        if self._shared_effect is not None:
            return self._shared_effect
        return DoseContrast(
            target=EffectTarget.ATE,
            estimator=self.source,
            learners={},
            n=self.n,
            kind="point",
            diagnostics=dict(self.diagnostics),
            _point=float(np.mean(self.values)),
            baseline=self.baseline,
            treatment_value=self.treatment_value,
            treatment=self.treatment,
            outcome=self.outcome,
            protocol=self.protocol,
        )


def _levels(baseline: float, value: float) -> tuple[float, float]:
    for name, level in (("baseline", baseline), ("value", value)):
        if isinstance(level, (bool, np.bool_)) or not isinstance(level, Real):
            raise ValueError(f"{name} must be a finite numerical dose")
        if not np.isfinite(level):
            raise ValueError(f"{name} must be a finite numerical dose")
    return float(baseline), float(value)


def _in_range(a: npt.NDArray[np.float64], *levels: float) -> None:
    low, high = float(a.min()), float(a.max())
    if any(level < low or level > high for level in levels):
        raise ValueError(f"requested dose is outside observed treatment range [{low}, {high}]")
