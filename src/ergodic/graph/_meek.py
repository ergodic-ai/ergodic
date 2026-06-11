# SPDX-License-Identifier: Apache-2.0
"""Meek's orientation rules and the Dor-Tarsi consistent extension.

Pure graph algebra over PDAGs, graphs whose edges are all directed or
undirected. :func:`meek_closure` applies Meek's rules R1 to R4 (Meek 1995)
until no more undirected edges can be oriented, optionally blocking
orientations through a ``forbidden`` callback. :func:`cpdag_of_dag` builds the
CPDAG of a DAG by the standard construction: undirected skeleton, restored
v-structures, then the closure. :func:`consistent_extension` and
:func:`extendable` implement Dor and Tarsi's (1992) algorithm: a PDAG is
extendable when some DAG has exactly its skeleton and v-structures.

Knowledge never enters here; an algorithm that wants to respect constraints
passes them in as the ``forbidden`` callable.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING

from ._graph import Edge, GraphKind, MixedGraph
from ._marks import Mark

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

_Maps = tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]


def _pdag_maps(graph: MixedGraph, caller: str) -> _Maps:
    """Split a PDAG into ``(parents, children, undirected)`` neighbour maps."""
    parents: dict[str, set[str]] = {n: set() for n in graph.nodes}
    children: dict[str, set[str]] = {n: set() for n in graph.nodes}
    undirected: dict[str, set[str]] = {n: set() for n in graph.nodes}
    for edge in graph.edges:
        pair = edge.directed_pair()
        if pair is not None:
            children[pair[0]].add(pair[1])
            parents[pair[1]].add(pair[0])
        elif edge.is_undirected:
            undirected[edge.u].add(edge.v)
            undirected[edge.v].add(edge.u)
        else:
            raise ValueError(
                f"{caller} needs a PDAG (directed and undirected edges only); "
                f"edge {edge.glyph()!r} is neither"
            )
    return parents, children, undirected


def _adjacent(x: str, y: str, maps: _Maps) -> bool:
    parents, children, undirected = maps
    return y in parents[x] or y in children[x] or y in undirected[x]


def _compelled(a: str, b: str, maps: _Maps, ambiguous: frozenset[tuple[str, str, str]]) -> bool:
    """Return ``True`` when some Meek rule orients the undirected edge ``a -- b`` as ``a -> b``.

    ``ambiguous`` gates R1 and R3, the two rules whose premise is a definite
    non-collider; see :func:`meek_closure`.
    """
    parents, children, undirected = maps
    # R1: c -> a, a -- b, c and b nonadjacent  =>  a -> b,
    #     unless (c, a, b) is ambiguous: b -> a could then be the collider.
    for c in parents[a]:
        if not _adjacent(c, b, maps) and (min(c, b), a, max(c, b)) not in ambiguous:
            return True
    # R2: a -> c -> b with a -- b  =>  a -> b (else a cycle).
    if children[a] & parents[b]:
        return True
    # R3: a -- c, a -- d, c -> b, d -> b, c and d nonadjacent  =>  a -> b,
    #     unless (c, a, d) is ambiguous: a could then collide its two ends.
    for c, d in combinations(sorted(undirected[a] & parents[b]), 2):
        if not _adjacent(c, d, maps) and (c, a, d) not in ambiguous:
            return True
    # R4: a -- b, c -> b, a and c adjacent, e -- a, e -> c, e and b nonadjacent
    #     =>  a -> b (b -> a would force either a cycle or a new v-structure at a).
    for c in parents[b]:
        if c == a or not _adjacent(a, c, maps):
            continue
        for e in parents[c] & undirected[a]:
            if e != b and not _adjacent(e, b, maps):
                return True
    return False


def meek_closure(
    pdag: MixedGraph,
    *,
    forbidden: Callable[[str, str], bool] | None = None,
    ambiguous: Collection[tuple[str, str, str]] | None = None,
) -> MixedGraph:
    """Apply Meek's rules R1 to R4 to a PDAG until fixpoint.

    Each rule turns an undirected edge into a directed one whenever the
    reverse orientation would create a cycle or a new v-structure in every
    consistent extension. ``forbidden(u, v)`` returning ``True`` blocks the
    orientation ``u -> v``: the rule simply does not fire that way and the
    edge stays undirected unless another rule orients it.

    ``ambiguous`` lists unshielded triples ``(x, z, y)``, ``z`` the middle
    node and the ends unordered, that the caller cannot certify as definite
    non-colliders (PC's conservative and majority collider rules leave such
    triples behind). A definite non-collider is exactly what R1 and R3
    consume: R1 reads ``c -> a -- b`` with ``c`` and ``b`` nonadjacent as
    compelling ``a -> b`` only because the collider ``c -> a <- b`` was ruled
    out, and R3 rests on its two converging parents never colliding at the
    undirected partner. Both rules skip a gated triple; R2 (a cycle rule)
    and R4 run unchanged. Default ``None`` gates nothing.

    The result is returned with kind ``mixed``; re-face it with
    :meth:`MixedGraph.as_kind` when more is known.
    """
    maps = _pdag_maps(pdag, "meek_closure")
    parents, children, undirected = maps
    blocked: Callable[[str, str], bool] = forbidden if forbidden is not None else lambda u, v: False
    gated = frozenset(
        (min(x, y), z, max(x, y)) for x, z, y in (ambiguous if ambiguous is not None else ())
    )
    graph = pdag
    changed = True
    while changed:
        changed = False
        pairs = sorted({tuple(sorted((x, y))) for x in undirected for y in undirected[x]})
        for u, v in pairs:
            if v not in undirected[u]:
                continue  # oriented earlier in this pass
            for a, b in ((u, v), (v, u)):
                if blocked(a, b) or not _compelled(a, b, maps, gated):
                    continue
                graph = graph.orient(a, b)
                undirected[a].discard(b)
                undirected[b].discard(a)
                children[a].add(b)
                parents[b].add(a)
                changed = True
                break
    return replace(graph, kind=GraphKind.MIXED)


def cpdag_of_dag(dag: MixedGraph) -> MixedGraph:
    """The CPDAG of a DAG: undirected skeleton, restored v-structures, Meek closure.

    Every DAG in the Markov equivalence class of ``dag`` maps to the same
    CPDAG, which keeps an edge directed exactly when every member orients it
    the same way.
    """
    if not dag.is_dag():
        raise ValueError("cpdag_of_dag needs a DAG: every edge directed and no cycle")
    skeleton = frozenset(Edge.make(e.u, Mark.TAIL, e.v, Mark.TAIL) for e in dag.edges)
    out = MixedGraph(nodes=dag.nodes, edges=skeleton, kind=GraphKind.MIXED)
    for z in sorted(dag.nodes):
        for x, y in combinations(sorted(dag.parents(z)), 2):
            if not dag.has_edge(x, y):
                out = out.orient(x, z).orient(y, z)
    return replace(meek_closure(out), kind=GraphKind.CPDAG)


def dags_of_cpdag(cpdag: MixedGraph, *, max_members: int | None = None) -> tuple[MixedGraph, ...]:
    """Every DAG in the CPDAG's Markov equivalence class, in a deterministic order.

    Branch on the first undirected edge, orient it each way, close under
    Meek's rules, and recurse. Distinct branches fix that edge differently,
    so no member appears twice and none is missed (the rules are sound, so
    every member agrees with the closure of its own branch). Each leaf is
    verified against the class definition, ``cpdag_of_dag(leaf)`` equal to
    the input, so a branch that closes into a non-member dies quietly and
    the answer is exact. ``max_members`` raises once the count would pass
    it; there is no silent truncation.
    """
    if not cpdag.is_cpdag():
        raise ValueError("dags_of_cpdag needs a CPDAG; build one with cpdag() or cpdag_of_dag()")
    found: list[MixedGraph] = []

    def expand(graph: MixedGraph) -> None:
        undirected = sorted((e.u, e.v) for e in graph.edges if e.is_undirected)
        if not undirected:
            candidate = replace(graph, kind=GraphKind.DAG)
            if not candidate.is_dag() or cpdag_of_dag(candidate).edges != cpdag.edges:
                return
            if max_members is not None and len(found) >= max_members:
                raise ValueError(
                    f"the class has more than {max_members} members; raise max_members "
                    "or work on the CPDAG itself"
                )
            found.append(candidate)
            return
        u, v = undirected[0]
        for a, b in ((u, v), (v, u)):
            expand(meek_closure(graph.orient(a, b)))

    expand(replace(cpdag, kind=GraphKind.MIXED))
    return tuple(found)


def consistent_extension(pdag: MixedGraph) -> MixedGraph | None:
    """A DAG with exactly the PDAG's skeleton and v-structures, or ``None``.

    Dor and Tarsi (1992): repeatedly find a node ``x`` with no outgoing
    directed edges whose undirected neighbours are adjacent to every other
    neighbour of ``x``, orient all undirected edges at ``x`` into ``x``, and
    retire ``x``. When every node retires the oriented graph is the
    extension; when no candidate exists none does.
    """
    maps = _pdag_maps(pdag, "consistent_extension")
    parents, children, undirected = maps
    remaining = set(pdag.nodes)
    graph = pdag
    while remaining:
        target = None
        for x in sorted(remaining):
            if children[x] & remaining:
                continue
            local_und = undirected[x] & remaining
            local_all = (parents[x] | undirected[x]) & remaining
            if all(_adjacent(u, w, maps) for u in local_und for w in local_all if w != u):
                target = x
                break
        if target is None:
            return None
        for u in sorted(undirected[target] & remaining):
            graph = graph.orient(u, target)
        remaining.discard(target)
    return graph.as_kind(GraphKind.DAG)


def extendable(pdag: MixedGraph) -> bool:
    """Return ``True`` when the PDAG admits a consistent extension."""
    return consistent_extension(pdag) is not None
