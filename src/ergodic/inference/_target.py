# SPDX-License-Identifier: Apache-2.0
"""The causal quantity an estimator returns."""

from __future__ import annotations

from enum import Enum


class EffectTarget(Enum):
    """Which effect a :class:`CausalEstimator` reports.

    The estimators return ``ATE`` and ``ATT`` for the adjustment and front-door
    families and ``LATE`` for instruments; ``CATE`` and ``ITE`` tag the per-unit
    results' summaries. ``ATC`` is reserved and no estimator returns it.
    """

    ATE = "ate"
    ATT = "att"
    ATC = "atc"
    LATE = "late"
    CATE = "cate"
    ITE = "ite"

    def __str__(self) -> str:
        return self.value


def as_target(value: EffectTarget | str) -> EffectTarget:
    """Coerce a string or enum to an :class:`EffectTarget`."""
    if isinstance(value, EffectTarget):
        return value
    try:
        return EffectTarget(value.lower())
    except ValueError:
        allowed = ", ".join(t.value for t in EffectTarget)
        raise ValueError(f"unknown effect target {value!r}; use one of {allowed}") from None
