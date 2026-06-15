# SPDX-License-Identifier: Apache-2.0
"""Identification: turn a causal graph into an estimand.

Given a graph, a treatment, and an outcome, the question is whether the
interventional distribution P(Y | do(X)) can be written in terms of the
observational distribution, and if so, how. The work is graph-only and
deterministic: no data, no statistics, no estimator choices yet.

The named criteria (adjustment, front-door, instruments) live in
:mod:`._criteria`; the :class:`Identification` record in :mod:`._record`.
:func:`identify_effect` tries them in order and reports the result.
"""

from __future__ import annotations

from ._criteria import (
    adjustment_set,
    frontdoor_set,
    identify_effect,
    instruments,
    is_frontdoor_set,
    is_instrument,
    is_valid_adjustment_set,
    minimal_adjustment_set,
)
from ._record import Identification

__all__ = [
    "Identification",
    "adjustment_set",
    "frontdoor_set",
    "identify_effect",
    "instruments",
    "is_frontdoor_set",
    "is_instrument",
    "is_valid_adjustment_set",
    "minimal_adjustment_set",
]
