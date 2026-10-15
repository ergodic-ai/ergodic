# SPDX-License-Identifier: Apache-2.0
"""The stable adjacency search PC and FCI share, plus the constraint helpers.

The skeleton phase is the stable variant (Colombo and Maathuis 2014): at each
conditioning size it snapshots the adjacencies and tests every ordered pair
against subsets of the snapshot neighbours, so the result does not depend on
the order edges are visited. Knowledge enters here as prune (a pair forbidden
in both directions is never tested, unless it is confounded: a hidden common
cause keeps it adjacent though neither causes the other) and protect (required
and confounded pairs are never tested or removed).
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...knowledge import DomainKnowledge
    from .._ci import CITest


def _permits_all(cause: str, effect: str) -> bool:
    """The open default when no knowledge is given."""
    return True


def _complement(permits: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
    """Turn a permits-edge projection into the forbidden callback Meek expects."""

    def forbids(u: str, v: str) -> bool:
        return not permits(u, v)

    return forbids


def _protected_pairs(knowledge: DomainKnowledge | None) -> frozenset[frozenset[str]]:
    """The pairs a skeleton search must never test or remove."""
    if knowledge is None:
        return frozenset()
    required = frozenset(frozenset(pair) for pair in knowledge.required)
    return required | knowledge.confounded


def stable_skeleton(
    oracle: CITest,
    nodes: list[str],
    knowledge: DomainKnowledge | None,
    *,
    alpha: float,
    max_cond_size: int | None,
) -> tuple[dict[str, set[str]], dict[frozenset[str], frozenset[str]]]:
    """PC-stable adjacency search: the surviving adjacencies and the sepsets."""
    protected = _protected_pairs(knowledge)
    permits = _permits_all if knowledge is None else knowledge.permits_edge
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for x, y in combinations(nodes, 2):
        if not permits(x, y) and not permits(y, x) and frozenset((x, y)) not in protected:
            # prune: forbidden in both directions, never tested. A confounded pair
            # stays: neither causes the other, but a hidden cause joins them.
            continue
        adj[x].add(y)
        adj[y].add(x)
    sepsets: dict[frozenset[str], frozenset[str]] = {}
    level = 0
    while max_cond_size is None or level <= max_cond_size:
        snapshot = {n: frozenset(adj[n]) for n in nodes}
        if all(len(snapshot[n]) <= level for n in nodes):
            break
        for x in nodes:
            for y in sorted(snapshot[x]):
                if y not in adj[x]:
                    continue  # removed earlier at this level
                if frozenset((x, y)) in protected:
                    continue  # protect: a required edge is never tested
                candidates = sorted(snapshot[x] - {y})
                if len(candidates) < level:
                    continue
                for subset in combinations(candidates, level):
                    if oracle.independent(x, y, subset, alpha=alpha):
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sepsets[frozenset((x, y))] = frozenset(subset)
                        break
        level += 1
    return adj, sepsets
