# SPDX-License-Identifier: Apache-2.0
"""m-separation for mixed graphs (and d-separation as the DAG special case).

A path is active given a conditioning set Z when every non-collider on it sits
outside Z and every collider on it is in Z or has a descendant in Z. Two nodes
are m-separated given Z when no active path joins them. The search below walks
the graph by (node, arriving-mark) state and applies the collider rule at each
step, so it never enumerates paths explicitly.

Circle marks (PAGs) get Zhang's definite-status reading: a vertex on a path is
a collider when both path marks at it are arrowheads, and a definite
non-collider when a path edge carries a tail at it, or both path marks at it
are circles and its path neighbours are non-adjacent. A definite m-connecting
path given Z is a path of definite-status vertices whose definite non-colliders
avoid Z and whose colliders are ancestors of Z (ancestry through directed edges
only). On a valid complete PAG this answers for the whole equivalence class:
every member MAG agrees. On a partially oriented graph that is not closed under
the orientation rules it is the conservative reading (vertices without definite
status block), and on circle-free graphs it reduces exactly to plain
m-separation.
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


def _ancestors_of(graph: MixedGraph, conditioned: frozenset[str]) -> set[str]:
    out: set[str] = set(conditioned)
    for z in conditioned:
        out |= graph.ancestors(z)
    return out


def _incident_map(graph: MixedGraph) -> dict[str, list[Edge]]:
    incident: dict[str, list[Edge]] = {n: [] for n in graph.nodes}
    for edge in graph.edges:
        incident[edge.u].append(edge)
        incident[edge.v].append(edge)
    return incident


def _plainly_connected(
    graph: MixedGraph,
    sources: frozenset[str],
    targets: frozenset[str],
    conditioned: frozenset[str],
) -> bool:
    """The circle-free walker: (node, arriving-mark) states, the collider rule."""
    ancestors_of_z = _ancestors_of(graph, conditioned)
    incident = _incident_map(graph)
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


def _definitely_connected(
    graph: MixedGraph,
    sources: frozenset[str],
    targets: frozenset[str],
    conditioned: frozenset[str],
) -> bool:
    """The definite-status walker for circle-bearing graphs: (prev, node) states.

    A vertex's status depends on the marks at it on both path edges and, in
    the circle-circle case, on whether its path neighbours are adjacent, so
    the state carries the previous vertex. Immediate backtracking along the
    same edge is skipped; revisiting a vertex from a different neighbour is a
    different state.
    """
    ancestors_of_z = _ancestors_of(graph, conditioned)
    incident = _incident_map(graph)
    adjacent: set[frozenset[str]] = {frozenset((e.u, e.v)) for e in graph.edges}
    visited: set[tuple[str, str]] = set()
    queue: deque[tuple[str, str]] = deque()

    def push(prev: str, node: str) -> bool:
        if node in targets:
            return True
        state = (prev, node)
        if state not in visited:
            visited.add(state)
            queue.append(state)
        return False

    for source in sources:
        for edge in incident[source]:
            if push(source, edge.other(source)):
                return True

    while queue:
        prev, node = queue.popleft()
        edge_in = graph.get_edge(prev, node)
        if edge_in is None:  # pragma: no cover - states come from edges
            continue
        in_mark = edge_in.mark_at(node)
        for edge_out in incident[node]:
            nxt = edge_out.other(node)
            if nxt == prev:
                continue
            out_mark = edge_out.mark_at(node)
            if in_mark is Mark.ARROW and out_mark is Mark.ARROW:
                passes = node in ancestors_of_z  # a collider, definite by its marks
            else:
                definite = (
                    in_mark is Mark.TAIL
                    or out_mark is Mark.TAIL
                    or (
                        in_mark is Mark.CIRCLE
                        and out_mark is Mark.CIRCLE
                        and frozenset((prev, nxt)) not in adjacent
                    )
                )
                passes = definite and node not in conditioned
            if not passes:
                continue
            if push(node, nxt):
                return True
    return False


def m_connected(
    graph: MixedGraph,
    x: str | Iterable[str],
    y: str | Iterable[str],
    given: str | Iterable[str] = frozenset(),
) -> bool:
    """Return ``True`` if some active path joins ``x`` and ``y`` given ``given``.

    Circle-free graphs (DAG, ADMG, MAG) use plain m-connection; a graph with
    circle marks (a PAG) uses definite m-connection, which on a valid complete
    PAG is the class-invariant answer.
    """
    sources = _as_set(x)
    targets = _as_set(y)
    conditioned = _as_set(given)

    unknown = sorted((sources | targets | conditioned) - graph.nodes)
    if unknown:
        raise ValueError(f"unknown node names (not in the graph): {unknown}")
    overlap = (sources & targets) | (sources & conditioned) | (targets & conditioned)
    if overlap:
        raise ValueError(f"x, y, and given must be disjoint; shared nodes: {sorted(overlap)}")

    if any(e.has_circle for e in graph.edges):
        return _definitely_connected(graph, sources, targets, conditioned)
    return _plainly_connected(graph, sources, targets, conditioned)


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
