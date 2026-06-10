# SPDX-License-Identifier: Apache-2.0
"""m-separation for mixed graphs (and d-separation as the DAG special case).

A path is active given a conditioning set Z when every non-collider on it sits
outside Z and every collider on it is in Z or has a descendant in Z. Two nodes
are m-separated given Z when no active path joins them. The search below walks
the graph by (node, arriving-mark) state and applies the collider rule at each
step, so it never enumerates paths explicitly.

Circle marks (PAGs) need the separate notion of definite vs possible separation,
which is not built yet, so a graph carrying any circle is rejected here.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from ._marks import Mark

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._graph import Edge, MixedGraph


def _as_set(value: str | Iterable[str]) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


def m_connected(
    graph: MixedGraph,
    x: str | Iterable[str],
    y: str | Iterable[str],
    given: str | Iterable[str] = frozenset(),
) -> bool:
    """Return ``True`` if some active path joins ``x`` and ``y`` given ``given``."""
    sources = _as_set(x)
    targets = _as_set(y)
    conditioned = _as_set(given)

    if any(e.has_circle for e in graph.edges):
        raise NotImplementedError(
            "m-separation needs definite/possible reasoning for circle marks (PAGs); "
            "it is defined here for DAG, ADMG, and MAG only"
        )
    overlap = (sources & targets) | (sources & conditioned) | (targets & conditioned)
    if overlap:
        raise ValueError(f"x, y, and given must be disjoint; shared nodes: {sorted(overlap)}")

    ancestors_of_z: set[str] = set(conditioned)
    for z in conditioned:
        ancestors_of_z |= graph.ancestors(z)

    incident: dict[str, list[Edge]] = {n: [] for n in graph.nodes}
    for edge in graph.edges:
        incident[edge.u].append(edge)
        incident[edge.v].append(edge)

    visited: set[tuple[str, Mark]] = set()
    queue: deque[tuple[str, Mark]] = deque()

    def push(node: str, arrived: Mark) -> bool:
        if node in targets:
            return True
        state = (node, arrived)
        if state not in visited:
            visited.add(state)
            queue.append(state)
        return False

    for source in sources:
        for edge in incident[source]:
            nbr = edge.other(source)
            if push(nbr, edge.mark_at(nbr)):
                return True

    while queue:
        node, arrived = queue.popleft()
        for edge in incident[node]:
            out_mark = edge.mark_at(node)
            collider = arrived is Mark.ARROW and out_mark is Mark.ARROW
            passes = node in ancestors_of_z if collider else node not in conditioned
            if not passes:
                continue
            nbr = edge.other(node)
            if push(nbr, edge.mark_at(nbr)):
                return True
    return False


def m_separated(
    graph: MixedGraph,
    x: str | Iterable[str],
    y: str | Iterable[str],
    given: str | Iterable[str] = frozenset(),
) -> bool:
    """Return ``True`` if ``x`` and ``y`` are m-separated given ``given``."""
    return not m_connected(graph, x, y, given)


def d_separated(
    graph: MixedGraph,
    x: str | Iterable[str],
    y: str | Iterable[str],
    given: str | Iterable[str] = frozenset(),
) -> bool:
    """d-separation, the DAG special case of :func:`m_separated`."""
    if not graph.is_dag():
        raise ValueError("d_separated requires a DAG; use m_separated for mixed graphs")
    return m_separated(graph, x, y, given)
