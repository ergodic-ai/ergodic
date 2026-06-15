# SPDX-License-Identifier: Apache-2.0
"""Identification: turn a causal graph into an estimand.

Given a graph, a treatment, and an outcome, the question is whether the
interventional distribution P(Y | do(X)) can be written in terms of the
observational distribution, and if so, how. The work is graph-only and
deterministic: no data, no statistics, no estimator choices yet.

The named criteria (adjustment, front-door, instruments) live in
:mod:`._criteria`; the :class:`Identification` record in :mod:`._record`.
:func:`identify_effect` tries them in order, falling through to the general
do-calculus ID algorithm on a single graph.

The general engine is exposed too: :func:`identify_distribution` runs ID (or
IDC) for any query and returns an :class:`IDResult` carrying the typed estimand
:class:`Functional` (or the :class:`Hedge` witness when not identifiable).
:func:`ida` covers an equivalence class by enumerating members.
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
from ._distribution import identify_distribution
from ._functional import Functional, Marginal, Prob, Product, Ratio
from ._id import Hedge, IDResult
from ._ida import IDA, ida
from ._mediation import MediationID, identify_mediation
from ._record import Identification

__all__ = [
    "IDA",
    "Functional",
    "Hedge",
    "IDResult",
    "Identification",
    "Marginal",
    "MediationID",
    "Prob",
    "Product",
    "Ratio",
    "adjustment_set",
    "frontdoor_set",
    "ida",
    "identify_distribution",
    "identify_effect",
    "identify_mediation",
    "instruments",
    "is_frontdoor_set",
    "is_instrument",
    "is_valid_adjustment_set",
    "minimal_adjustment_set",
]
