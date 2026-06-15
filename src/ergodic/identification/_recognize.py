# SPDX-License-Identifier: Apache-2.0
"""Recognition: route a general ID functional back to a named criterion.

When the named ladder (null, adjustment, front-door, instruments) misses on a
single graph but the general ID algorithm identifies the effect, the resulting
:class:`~ergodic.identification._functional.Functional` may still be one of the
named closed forms in disguise. Recognizing it lets today's estimators run with
no new estimation code.

The key fact that shapes this module: ``identify_effect`` runs the named ladder
*before* ID, and the adjustment criterion (:func:`adjustment_set`) is *complete*,
so any adjustment case is already caught upstream. The front-door search
(:func:`frontdoor_set`), by contrast, is non-exhaustive (it tries the full
mediator set and single mediators, not every subset), so it can miss a valid
front-door set that ID identifies. That ladder-missed front-door is recognition's
real prize.

Recognition never relabels a functional without validating the proposed set
against the *complete* criterion (:func:`is_valid_adjustment_set`,
:func:`is_frontdoor_set`). It proposes candidate sets *from the graph*, not by
pattern-matching the functional (ID functionals are correct but not minimized, so
their structure is not a reliable signal). If no named form validates, it returns
``None`` and the caller keeps ``strategy="id"``: correctness over coverage.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from ._criteria import (
    _causal_nodes,
    is_frontdoor_set,
    is_valid_adjustment_set,
)

if TYPE_CHECKING:
    from ..graph import MixedGraph
    from ._functional import Functional

# A small cap on the front-door candidate search: the mediator pool is the
# proper-causal-path nodes, already tiny in practice, but a hard ceiling keeps
# the subset enumeration from blowing up on a pathological graph. Past it,
# recognition gives up (returns None) and strategy="id" stands.
_MAX_MEDIATOR_POOL = 12


def recognize(
    graph: MixedGraph,
    x: frozenset[str],
    y: frozenset[str],
    functional: Functional,
) -> tuple[str, frozenset[str]] | None:
    """Reduce an ID functional to a named strategy, or return ``None``.

    Returns ``("adjustment", Z)`` or ``("frontdoor", M)`` when a valid named set
    is found and verified by the corresponding complete criterion, else ``None``.
    The ``functional`` argument is accepted but not pattern-matched: ID
    functionals are correct yet not minimized, so the graph is the reliable
    source of candidate sets. Every returned set has passed its complete check,
    so recognition can never disagree with the criteria.
    """
    adjustment = _recognize_adjustment(graph, x, y)
    if adjustment is not None:
        return ("adjustment", adjustment)
    frontdoor = _recognize_frontdoor(graph, x, y)
    if frontdoor is not None:
        return ("frontdoor", frontdoor)
    return None


def _recognize_adjustment(
    graph: MixedGraph, x: frozenset[str], y: frozenset[str]
) -> frozenset[str] | None:
    """The cheap adjustment check, for safety.

    The named ladder already runs the complete adjustment criterion before ID,
    so this essentially never fires post-ladder. It is kept so recognition is
    self-contained and correct even if the call order ever changes: the
    canonical adjustment set over ancestors is validated by the complete
    criterion.
    """
    from ._criteria import adjustment_set

    candidate = adjustment_set(graph, x, y)
    if candidate is not None and is_valid_adjustment_set(graph, x, y, candidate):
        return candidate
    return None


def _recognize_frontdoor(
    graph: MixedGraph, x: frozenset[str], y: frozenset[str]
) -> frozenset[str] | None:
    """A thorough front-door candidate search, validated by the complete criterion.

    The mediator pool is the proper-causal-path nodes between ``x`` and ``y``
    (latents excluded). Search its non-empty subsets, smallest first, and return
    the first that :func:`is_frontdoor_set` accepts. This catches the valid
    front-door sets the non-exhaustive :func:`frontdoor_set` search can miss. The
    pool is tiny in practice; a hard cap guards the pathological case.
    """
    pool = sorted(_causal_nodes(graph, x, y) - y - x - graph.latent)
    if not pool or len(pool) > _MAX_MEDIATOR_POOL:
        return None
    for size in range(1, len(pool) + 1):
        for combo in combinations(pool, size):
            candidate = frozenset(combo)
            if is_frontdoor_set(graph, x, y, candidate):
                return candidate
    return None
