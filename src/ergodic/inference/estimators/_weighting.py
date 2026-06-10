# SPDX-License-Identifier: Apache-2.0
"""Inverse propensity weighting for the adjustment family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._estimator import Slot, Task, _AdjustmentBase
from .._learner import _mean_predict
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from .._learner import Learner


class IPW(_AdjustmentBase):
    """Weight by the inverse propensity of treatment.

    ``scheme="hajek"`` (the default) self-normalizes the weights; ``scheme="ht"``
    is the raw Horvitz-Thompson estimator. Propensities are clipped to
    ``[clip, 1 - clip]`` to bound the weights.
    """

    name = "ipw"
    slots = (Slot("propensity", Task.CLASSIFICATION),)
    targets = frozenset({EffectTarget.ATE, EffectTarget.ATT})

    def __init__(
        self, learners: Any = None, *, scheme: str = "hajek", clip: float = 1e-3, **kwargs: Any
    ) -> None:
        super().__init__(learners, **kwargs)
        if scheme not in ("hajek", "ht"):
            raise ValueError(f"scheme must be 'hajek' or 'ht', got {scheme!r}")
        self.scheme = scheme
        self.clip = clip

    def _propensity(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> npt.NDArray[np.float64]:
        a = arrays["A"]
        z = arrays["Z"]
        model = make["propensity"]().fit(z, a)
        e = _mean_predict(model, z)
        return np.clip(e, self.clip, 1.0 - self.clip)

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        a = arrays["A"]
        y = arrays["Y"]
        e = self._propensity(arrays, make)
        if target is EffectTarget.ATT:
            odds = e / (1.0 - e)
            w0 = (1.0 - a) * odds
            denom = float(np.sum(w0))
            if denom <= 0.0:
                raise ValueError("no control weight for ATT")
            mu0 = float(np.sum(w0 * y) / denom)
            return float(np.mean(y[a == 1.0])) - mu0
        if self.scheme == "hajek":
            w1 = a / e
            w0 = (1.0 - a) / (1.0 - e)
            s1 = float(np.sum(w1))
            s0 = float(np.sum(w0))
            if s1 <= 0.0 or s0 <= 0.0:
                raise ValueError("a treatment arm has no weight")
            return float(np.sum(w1 * y) / s1 - np.sum(w0 * y) / s0)
        return float(np.mean(a * y / e - (1.0 - a) * y / (1.0 - e)))


class Stratification(_AdjustmentBase):
    """Propensity stratification: bin on the propensity and difference within strata.

    The estimate is the stratum-size-weighted (ATE) or treated-size-weighted (ATT)
    average of the within-stratum mean difference. Standard error by bootstrap.
    """

    name = "stratification"
    slots = (Slot("propensity", Task.CLASSIFICATION),)
    targets = frozenset({EffectTarget.ATE, EffectTarget.ATT})

    def __init__(
        self, learners: Any = None, *, n_strata: int = 5, clip: float = 1e-3, **kwargs: Any
    ) -> None:
        super().__init__(learners, **kwargs)
        if n_strata < 2:
            raise ValueError(f"n_strata must be >= 2, got {n_strata}")
        self.n_strata = n_strata
        self.clip = clip

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        e = np.clip(_mean_predict(make["propensity"]().fit(z, a), z), self.clip, 1.0 - self.clip)
        edges = np.quantile(e, np.linspace(0.0, 1.0, self.n_strata + 1))
        bins = np.searchsorted(edges[1:-1], e, side="right")
        total = 0.0
        accum = 0.0
        for stratum in range(self.n_strata):
            here = bins == stratum
            treated = here & (a == 1.0)
            control = here & (a == 0.0)
            n_treated = int(np.sum(treated))
            n_control = int(np.sum(control))
            if n_treated == 0 or n_control == 0:
                continue
            diff = float(np.mean(y[treated]) - np.mean(y[control]))
            weight = (
                float(n_treated) if target is EffectTarget.ATT else float(n_treated + n_control)
            )
            accum += weight * diff
            total += weight
        if total <= 0.0:
            raise ValueError("no stratum has both a treated and a control unit")
        return accum / total
