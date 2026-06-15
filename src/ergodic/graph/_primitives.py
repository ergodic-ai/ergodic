# SPDX-License-Identifier: Apache-2.0
"""Graph primitives the identification algorithms read.

Districts (the c-components of the bidirected subgraph), a deterministic
topological order over the directed edges, and the ADMG latent projection
that turns ``latent=`` nodes into ``<->`` edges over the observed margin.

The latent projection is the ADMG projection, not the MAG ancestral
projection in :func:`mag_of_dag`: it preserves m-separation over the kept
margin, which is the property the ID algorithm needs of its input graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._graph import Edge, GraphKind, MixedGraph
from ._marks import Mark

if TYPE_CHECKING:
    from collections.abc import Iterable


# -- districts (c-components) -------------------------------------------------


def districts(graph: MixedGraph) -> frozenset[frozenset[str]]:
    """The c-components: connected components of the bidirected (``<->``) subgraph.

    Two observed nodes share a district when a ``<->`` path joins them; a node
    with no bidirected edge is its own singleton. Every node lands in exactly
    one district and the districts partition ``graph.nodes``. On a DAG (no
    bidirected edges) every district is a singleton. Defined over the nodes as
    they are: project latents first (:func:`latent_projection`) if the graph
    still carries ``latent=`` designations.
    """
    seen: set[str] = set()
    out: set[frozenset[str]] = set()
    for start in sorted(graph.nodes):
        if start in seen:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            for nbr in graph.spouses(node):
                if nbr not in component:
                    stack.append(nbr)
        seen |= component
        out.add(frozenset(component))
    return frozenset(out)


def district_of(graph: MixedGraph, node: str) -> frozenset[str]:
    """Return the district (c-component) containing ``node``."""
    if node not in graph.nodes:
        raise ValueError(f"unknown node name: {node!r}")
    component: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur in component:
            continue
        component.add(cur)
        for nbr in graph.spouses(cur):
            if nbr not in component:
                stack.append(nbr)
    return frozenset(component)


# -- topological order -------------------------------------------------------


def topological_order(graph: MixedGraph) -> tuple[str, ...]:
    """A topological order over the directed edges, tie-broken by node name.

    For every directed edge ``a -> b``, ``a`` precedes ``b``. Bidirected edges
    impose no order. The name tie-break makes the order stable across runs, so
    estimands that read it render reproducibly. Raises on a graph whose
    directed part has a cycle.
    """
    order, acyclic = graph._kahn()
    if not acyclic:
        raise ValueError("topological_order needs an acyclic directed part; the graph has a cycle")
    return tuple(order)


# -- ADMG latent projection --------------------------------------------------


def latent_projection(graph: MixedGraph, keep: Iterable[str] | None = None) -> MixedGraph:
    """Project a DAG or ADMG with latents onto the observed margin, as an ADMG.

    ``keep`` defaults to ``graph.nodes - graph.latent``. The projected-out set
    is ``H = nodes - keep``. Over ``keep`` the result is an ADMG (``kind=ADMG``,
    empty ``latent``):

    - ``a -> b`` iff a directed path ``a -> ... -> b`` has every intermediate
      node in ``H``.
    - ``a <-> b`` iff a path between ``a`` and ``b`` has every intermediate node
      in ``H`` and a non-collider, with arrowheads into both ``a`` and ``b``.

    Computed by iterative single-node elimination of each node in ``H``, which
    is equivalent. The defining contract is m-separation preservation: for
    disjoint observed ``A``, ``B``, ``Z`` (none in ``H``), the projection and
    ``graph`` agree on ``m_separated(A, B, Z)``. This is the ADMG projection
    the ID algorithm needs, distinct from the MAG ancestral projection
    :func:`mag_of_dag`.
    """
    if not graph.is_admg():
        raise ValueError(
            "latent_projection needs a DAG or ADMG (directed and bidirected "
            "edges, no directed cycle); project the graph you have onto one first"
        )
    keep_set = (graph.nodes - graph.latent) if keep is None else frozenset(keep)
    unknown = keep_set - graph.nodes
    if unknown:
        raise ValueError(f"unknown node names to keep: {', '.join(sorted(unknown))}")
    eliminate = graph.nodes - keep_set

    # Track directed and bidirected relations separately, since the projection
    # of a DAG/ADMG can produce a bow (both ``a -> b`` and ``a <-> b`` on one
    # pair) and m-separation needs both: the directed edge makes ``a`` a
    # non-collider for pass-through paths, the bidirected edge makes it a
    # collider that conditioning opens. They are kept as two edges; the
    # one-edge-per-pair invariant only constrains the glyph builders.
    directed: set[tuple[str, str]] = set()
    bidir: set[frozenset[str]] = set()
    for e in graph.edges:
        pair = e.directed_pair()
        if pair is not None:
            directed.add(pair)
        elif e.is_bidirected:
            bidir.add(frozenset((e.u, e.v)))

    def parents_of(node: str) -> set[str]:
        return {p for (p, c) in directed if c == node}

    def children_of(node: str) -> set[str]:
        return {c for (p, c) in directed if p == node}

    def spouses_of(node: str) -> set[str]:
        return {next(iter(pr - {node})) for pr in bidir if node in pr}

    for h in sorted(eliminate):
        pa = parents_of(h)
        ch = children_of(h)
        sp = spouses_of(h)
        # A path through h is a non-collider iff one side is a child of h
        # (the ``h -> c`` edge gives h a tail). Reconnect each child to every
        # other neighbour, arrowhead into the child, the neighbour keeping its
        # mark: a parent stays a tail (directed), a child or spouse an arrow
        # (bidirected).
        for c in ch:
            for p in pa:
                if p != c:
                    directed.add((p, c))
            for other in (ch | sp) - {c}:
                bidir.add(frozenset((c, other)))
        directed = {(p, c) for (p, c) in directed if p != h and c != h}
        bidir = {pr for pr in bidir if h not in pr}

    edges: list[Edge] = []
    for p, c in sorted(directed):
        if p in keep_set and c in keep_set:
            edges.append(Edge.make(p, Mark.TAIL, c, Mark.ARROW))
    for pr in sorted(sorted(pr) for pr in bidir):
        a, b = pr
        if a in keep_set and b in keep_set:
            edges.append(Edge.make(a, Mark.ARROW, b, Mark.ARROW))
    out = MixedGraph(nodes=keep_set, edges=frozenset(edges), kind=GraphKind.ADMG)
    out.validate()
    return out
