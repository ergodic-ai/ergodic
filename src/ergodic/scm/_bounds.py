# SPDX-License-Identifier: Apache-2.0
"""The honesty layer: a non-identified effect returns bounds, not a point.

When the identification layer hands back a :class:`~ergodic.identification.Hedge`
(the effect is not a function of the data alone), a confident point estimate would
be dishonest: the pressure-test showed a "low confounding" prior returns twice the
truth with tight error bars. So the SCM returns a :class:`Sensitivity` instead.

The MVP engine is the analytic linear sensitivity curve. For a bow
(``U -> X``, ``U -> Y``, ``X -> Y`` with ``U`` latent), the observed margin fixes
``cov(X, Y) = b * var(X) + c`` where ``c = cov_U(X, Y)`` is the unidentified
confounding channel. So the causal effect is the exact line

    b(c) = (cov(X, Y) - c) / var(X)

a one-to-one map from the assumed confounding to the answer. ``c = 0`` (assume no
confounding) recovers the observational slope; the truth sits at the true channel
strength, which the data cannot pin. :class:`Sensitivity` reports this curve over a
range of ``c``, collapsing to the identified point as ``c -> 0``.

The deliberate design choice: :class:`Sensitivity` has **no** ``value`` or scalar
accessor. A caller cannot read a naked number off a non-identified effect; they get
the curve, the range, and the obstruction. That type distinction is the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class Sensitivity:
    """The effect as a function of the assumed confounding strength.

    Returned in place of a point when an effect is not identified. ``confounding``
    is the grid of assumed confounding-channel values ``c`` (a covariance on the
    treatment-outcome pair), ``effect`` the implied causal effect ``b(c)`` at each.
    ``identified_at_zero`` is ``b(0)``, the answer under the no-confounding
    assumption (the observational slope), which is where the curve meets the
    identified region. ``obstruction`` is the human-readable reason (the hedge).

    There is no scalar accessor on purpose: a non-identified effect is a range, not
    a number. Read :meth:`range`, :meth:`at`, or the arrays.
    """

    treatment: str
    outcome: str
    confounding: npt.NDArray[np.float64]
    effect: npt.NDArray[np.float64]
    identified_at_zero: float
    obstruction: str
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def identified(self) -> bool:
        """Always ``False``: a :class:`Sensitivity` exists because the effect is not."""
        return False

    def range(self) -> tuple[float, float]:
        """The (min, max) implied effect over the swept confounding grid."""
        if self.effect.size == 0:
            raise ValueError("sensitivity curve is empty")
        return (float(np.min(self.effect)), float(np.max(self.effect)))

    def at(self, c: float) -> float:
        """The implied causal effect if the confounding channel strength is ``c``.

        Linear interpolation on the swept grid; ``c`` outside the grid extrapolates
        along the line, which is exact for the linear-Gaussian case.
        """
        return float(np.interp(c, self.confounding, self.effect))

    def __repr__(self) -> str:
        lo, hi = self.range()
        return (
            f"Sensitivity({self.treatment} -> {self.outcome}: effect in "
            f"[{lo:.3g}, {hi:.3g}] over assumed confounding, "
            f"b(0)={self.identified_at_zero:.3g}; not identified: {self.obstruction})"
        )


def linear_sensitivity_curve(
    *,
    treatment: str,
    outcome: str,
    cov_xy: float,
    var_x: float,
    obstruction: str,
    c_grid: npt.NDArray[np.float64] | None = None,
    diagnostics: dict[str, float] | None = None,
) -> Sensitivity:
    """Build the analytic ``b(c) = (cov_xy - c) / var_x`` sensitivity curve.

    ``cov_xy`` and ``var_x`` are read off the data (the observed treatment-outcome
    covariance and treatment variance). ``c_grid`` is the range of assumed
    confounding to sweep; it defaults to ``[0, cov_xy]`` -- from "no confounding"
    (``b = cov / var``, the observational slope) to "all of the covariance is
    confounding" (``b = 0``). That honest band brackets the truth for a positively
    confounded effect; pass an explicit grid for a different sensitivity range. The
    curve is exact for a linear-Gaussian bow.
    """
    if var_x <= 0.0:
        raise ValueError(f"var_x must be positive, got {var_x}")
    if c_grid is None:
        # from no confounding (c=0) to all-covariance-is-confounding (c=cov_xy),
        # oriented so the band runs the right way whatever the sign of cov_xy.
        c_grid = np.linspace(min(0.0, cov_xy), max(0.0, cov_xy), 41)
    c = np.asarray(c_grid, dtype=np.float64)
    b = (cov_xy - c) / var_x
    return Sensitivity(
        treatment=treatment,
        outcome=outcome,
        confounding=c,
        effect=b,
        identified_at_zero=float(cov_xy / var_x),
        obstruction=obstruction,
        diagnostics=dict(diagnostics or {}),
    )
