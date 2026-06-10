# SPDX-License-Identifier: Apache-2.0
"""Graph surgery: interventions and subgraphs.

Each function takes a graph and returns a new one. Removing edges or nodes keeps
the declared kind valid (a sub-DAG is still a DAG), so these preserve ``kind``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ._marks import Mark

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._graph import MixedGraph


def _as_set(value: str | Iterable[str]) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


def intervene(graph: MixedGraph, targets: str | Iterable[str]) -> MixedGraph:
    """Apply the do-operator: cut every edge with an arrowhead into a target.

    Intervening on a node sets it from outside the system, so incoming directed
    edges (its parents) and bidirected edges (latent confounding) both drop.
    Edges where the target has a tail stay, since the target still affects them.
    """
    cut = _as_set(targets)
    kept = frozenset(
        e
        for e in graph.edges
        if not (e.u in cut and e.u_mark is Mark.ARROW)
        and not (e.v in cut and e.v_mark is Mark.ARROW)
    )
    return replace(graph, edges=kept)


def induced_subgraph(graph: MixedGraph, nodes: Iterable[str]) -> MixedGraph:
    """Keep only ``nodes`` and the edges among them."""
    keep = _as_set(nodes) & graph.nodes
    edges = frozenset(e for e in graph.edges if e.u in keep and e.v in keep)
    return replace(
        graph,
        nodes=keep,
        edges=edges,
        latent=graph.latent & keep,
        selection=graph.selection & keep,
    )


def ancestral_subgraph(graph: MixedGraph, nodes: Iterable[str]) -> MixedGraph:
    """Keep ``nodes`` together with all their ancestors, and the edges among them."""
    keep = set(_as_set(nodes) & graph.nodes)
    for node in list(keep):
        keep |= graph.ancestors(node)
    return induced_subgraph(graph, keep)
