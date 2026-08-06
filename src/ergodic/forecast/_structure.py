# SPDX-License-Identifier: Apache-2.0
"""Window graphs: predictive structure written as a plain DAG over lagged nodes.

A window graph is the ordinary :class:`~ergodic.graph.MixedGraph` over node names
like ``"sales[t]"`` and ``"pmi[t-2]"``. An edge ``pmi[t-2] -> sales[t]`` says the
driver acts with a two-step delay. Nothing about the graph object changes: two
lags between the same pair of variables are two different node pairs, so the
one-edge-per-pair rule holds without being bent, and domain knowledge, plotting,
and the separation machinery all work as they already do.

This is the same convention temporal discovery already returns, down to the
format string (:func:`ergodic.discovery.lagged_node`). A
:class:`~ergodic.discovery.TimeSeriesDiscoveryResult` hands its ``.graph``
straight to :func:`validate_window_graph` and to the forecaster: discovery to
forecasting with no translation step in between.

:func:`window_graph` is the hand-built path, for structure you already know.
:func:`validate_window_graph` is the gate: node names must parse, and no edge may
point backward in time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..discovery import lagged_node, parse_lagged
from ..graph import Mark, dag

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..graph import MixedGraph

__all__ = ["bases", "max_lag", "split_lag", "validate_window_graph", "window_graph"]


def split_lag(node: str) -> tuple[str, int]:
    """Split a window node into its variable and its lag: ``"x[t-2]"`` to ``("x", 2)``."""
    return parse_lagged(node)


def window_graph(edges: Iterable[tuple[str, str, int]]) -> MixedGraph:
    """Build a window graph from ``(source, target, lag)`` triples.

    Each triple becomes the edge ``source[t-lag] -> target[t]``. Lag zero is a
    contemporaneous edge, and a variable may point at itself at lag one or more
    (self-memory), which is a legal edge because the two lagged copies are
    different nodes.

    ``window_graph([("pmi", "sales", 2), ("sales", "sales", 1)])`` gives
    ``pmi[t-2] -> sales[t]`` and ``sales[t-1] -> sales[t]``.
    """
    specs: list[tuple[str, str]] = []
    for source, target, lag in edges:
        if int(lag) < 0:
            raise ValueError(f"lag must be >= 0, got {lag} for {source!r} -> {target!r}")
        if source == target and int(lag) == 0:
            raise ValueError(
                f"{source!r} -> {target!r} at lag 0 is a self-loop; "
                "self-memory needs a lag of 1 or more"
            )
        specs.append((lagged_node(str(source), int(lag)), lagged_node(str(target), 0)))
    return validate_window_graph(dag(specs))


def bases(graph: MixedGraph) -> tuple[str, ...]:
    """The variables behind the window nodes, sorted and deduplicated."""
    return tuple(sorted({split_lag(node)[0] for node in graph.nodes}))


def max_lag(graph: MixedGraph) -> int:
    """The deepest lag any node reaches back to, zero for a purely contemporaneous graph."""
    return max((split_lag(node)[1] for node in graph.nodes), default=0)


def validate_window_graph(graph: MixedGraph) -> MixedGraph:
    """Check ``graph`` is a window graph and return it unchanged.

    Three things have to hold. Every node name parses as ``"v[t]"`` or
    ``"v[t-k]"``. No edge points backward in time, meaning no arrowhead lands on
    the earlier of two nodes at different lags. And the directed edges are
    acyclic, which for nodes inside one time slice is the ordinary requirement
    and across slices is time's own doing.
    """
    lags = {node: split_lag(node)[1] for node in graph.nodes}
    for edge in graph.edge_list():
        if lags[edge.u] == lags[edge.v]:
            continue
        earlier = edge.u if lags[edge.u] > lags[edge.v] else edge.v
        if edge.mark_at(earlier) is Mark.ARROW:
            raise ValueError(
                f"edge {edge.glyph()!r} points backward in time: it puts an arrowhead on "
                f"{earlier!r}, which is the earlier of the two"
            )
    _check_acyclic(graph)
    return graph


def _check_acyclic(graph: MixedGraph) -> None:
    """Raise when the directed edges of ``graph`` close a cycle."""
    children: dict[str, list[str]] = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        pair = edge.directed_pair()
        if pair is not None:
            children.setdefault(pair[0], []).append(pair[1])
            children.setdefault(pair[1], [])
    state: dict[str, int] = {}
    for start in sorted(children):
        if state.get(start, 0) != 0:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = []
        while stack:
            node, position = stack.pop()
            if position == 0:
                if state.get(node, 0) != 0:
                    continue
                state[node] = 1
                path.append(node)
                stack.append((node, 1))
                for child in children.get(node, ()):
                    if state.get(child, 0) == 1:
                        cycle = " -> ".join([*path[path.index(child) :], child])
                        raise ValueError(f"the window graph has a directed cycle: {cycle}")
                    if state.get(child, 0) == 0:
                        stack.append((child, 0))
            else:
                state[node] = 2
                path.pop()
