# SPDX-License-Identifier: Apache-2.0
"""The distribution-shaped query result an SCM returns.

A query on a fitted :class:`~ergodic.scm.SCM` (a sample, an interventional mean,
a counterfactual, an identified effect) carries a posterior: draws of the answer,
not a bare number. :class:`SCMResult` is the same distribution shape the Bayesian
estimators emit (:class:`~ergodic.inference.EffectEstimate`): ``mean``, ``draws``,
``interval``, with the point as the degenerate one-draw case.

What it adds is the honesty flag. ``identified`` is ``True`` for a query that is a
function of the data alone: an interventional distribution that the identification
layer certified, or a counterfactual on invertible mechanisms. Today every
query that reaches an :class:`SCMResult` is identified -- the back-door effect, the
interventional distribution, and the exact invertible counterfactual (the SCM
refuses a non-invertible-mechanism counterfactual rather than return a result). The
flag is kept so a result can say ``False`` if a wider-abduction counterfactual is
ever returned, where the width of the posterior would be the message. A
*non-identified effect* never lands here at all; it returns a
:class:`~ergodic.scm.Sensitivity`, so a naked point read is impossible by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True)
class SCMResult:
    """Draws of a query's answer.

    ``query`` names what was asked (``"sample"``, ``"interventional"``,
    ``"counterfactual"``, ``"ate"``). ``target`` names the node read.
    ``_draws`` holds the draws (flattened); the accessors read them.
    ``identified`` records whether the answer is a function of the data alone.

    ``kind`` says what the draws are. ``"posterior"`` (``ate``,
    ``counterfactual``): draws of the answer itself, so ``interval`` is a
    credible interval. ``"predictive"`` (``sample``, ``marginal``,
    ``interventional``): posterior-predictive draws of the outcome, one unit
    each, so ``value`` estimates the mean outcome while ``std_error`` and
    ``interval`` give the spread of the outcome, not the uncertainty in its mean.
    """

    query: str
    target: str
    kind: str  # "posterior" or "predictive"
    identified: bool
    diagnostics: dict[str, float] = field(default_factory=dict)
    _draws: npt.NDArray[np.float64] = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def value(self) -> float:
        """The mean of the draws (the point summary)."""
        if self._draws.size == 0:
            raise ValueError("result carries no draws")
        return float(np.mean(self._draws))

    def mean(self) -> float:
        """The posterior mean of the answer."""
        return self.value

    @property
    def std_error(self) -> float | None:
        """The standard deviation of the draws (the outcome's spread when predictive)."""
        if self._draws.size < 2:
            return None
        return float(np.std(self._draws, ddof=1))

    def draws(self) -> npt.NDArray[np.float64]:
        """The posterior draws of the answer."""
        return np.array(self._draws, copy=True)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """An interval at ``level`` from the draw quantiles.

        A credible interval for a ``"posterior"`` result; for a ``"predictive"``
        result, the range a single unit's outcome falls in.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if self._draws.size == 0:
            raise ValueError("result carries no draws to form an interval from")
        lo = float(np.quantile(self._draws, (1.0 - level) / 2.0))
        hi = float(np.quantile(self._draws, (1.0 + level) / 2.0))
        return (lo, hi)

    def __repr__(self) -> str:
        flag = "identified" if self.identified else "not identified"
        se = self.std_error
        body = f"{self.target}={self.value:.4g}"
        if se is not None:
            lo, hi = self.interval()
            label = "95% predictive interval" if self.kind == "predictive" else "95% CrI"
            body += f", sd={se:.3g}, {label} [{lo:.3g}, {hi:.3g}]"
        return f"SCMResult({self.query}, {flag}: {body})"


@dataclass(frozen=True)
class CategoricalResult:
    """A posterior over a categorical query's answer.

    A categorical node's interventional or counterfactual answer is a
    distribution over its categories, not a single number, so this carries the
    per-category probabilities (``probabilities``, summing to one over
    ``categories``) and the per-unit draws when a counterfactual was asked of
    specific units. ``identified`` carries the same honesty flag as
    :class:`SCMResult`.
    """

    query: str
    target: str
    categories: tuple[object, ...]
    probabilities: npt.NDArray[np.float64]
    identified: bool
    diagnostics: dict[str, float] = field(default_factory=dict)
    _per_unit: npt.NDArray[np.int64] | None = field(repr=False, default=None)

    def mode(self) -> object:
        """The most probable category."""
        return self.categories[int(np.argmax(self.probabilities))]

    def per_unit(self) -> npt.NDArray[np.int64] | None:
        """Per-unit counterfactual category codes, when units were queried."""
        if self._per_unit is None:
            return None
        return np.array(self._per_unit, copy=True)

    def __repr__(self) -> str:
        flag = "identified" if self.identified else "not identified"
        probs = ", ".join(
            f"{cat}={p:.3g}" for cat, p in zip(self.categories, self.probabilities, strict=False)
        )
        return f"CategoricalResult({self.query}, {flag}: {self.target} [{probs}])"
