# SPDX-License-Identifier: Apache-2.0
"""The mixed-graph core: ``Edge``, ``MixedGraph``, and the typed faces.

A single representation, a mixed graph with typed endpoint marks, covers every
causal graph we care about. Each named kind (DAG, ADMG, MAG, CPDAG, PAG) is that
core under a different set of allowed marks plus a few structural rules.

Graphs are immutable value objects. Every mutator returns a new graph and leaves
the original untouched, which keeps graph surgery (the do-operator, orienting
edges during discovery) easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from ._marks import Mark

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy as np


class GraphKind(Enum):
    """The kind of a mixed graph, which fixes the allowed marks and rules."""

    MIXED = "mixed"
    DAG = "dag"
    ADMG = "admg"
    MAG = "mag"
    CPDAG = "cpdag"
    PAG = "pag"


class InvalidGraphError(ValueError):
    """A graph violates the rules of its declared :class:`GraphKind`."""


@dataclass(frozen=True)
class Edge:
    """One edge between two distinct nodes, with a mark at each end.

    Stored canonically with ``u < v`` (string order) so an undirected pair maps
    to exactly one ``Edge``. ``u_mark`` is the mark at ``u``, ``v_mark`` at ``v``.
    """

    u: str
    v: str
    u_mark: Mark
    v_mark: Mark

    @staticmethod
    def make(a: str, a_mark: Mark, b: str, b_mark: Mark) -> Edge:
        """Build an edge between ``a`` and ``b``, normalizing endpoint order."""
        if a == b:
            raise ValueError(f"self-loops are not allowed: {a!r}")
        if a < b:
            return Edge(a, b, a_mark, b_mark)
        return Edge(b, a, b_mark, a_mark)

    def mark_at(self, node: str) -> Mark:
        """Return the mark this edge carries at ``node``."""
        if node == self.u:
            return self.u_mark
        if node == self.v:
            return self.v_mark
        raise KeyError(node)

    def other(self, node: str) -> str:
        """Return the endpoint that is not ``node``."""
        if node == self.u:
            return self.v
        if node == self.v:
            return self.u
        raise KeyError(node)

    @property
    def is_directed(self) -> bool:
        return {self.u_mark, self.v_mark} == {Mark.TAIL, Mark.ARROW}

    @property
    def is_bidirected(self) -> bool:
        return self.u_mark is Mark.ARROW and self.v_mark is Mark.ARROW

    @property
    def is_undirected(self) -> bool:
        return self.u_mark is Mark.TAIL and self.v_mark is Mark.TAIL

    @property
    def has_circle(self) -> bool:
        return Mark.CIRCLE in (self.u_mark, self.v_mark)

    def directed_pair(self) -> tuple[str, str] | None:
        """Return ``(parent, child)`` if this edge is directed, else ``None``."""
        if not self.is_directed:
            return None
        if self.u_mark is Mark.TAIL:
            return (self.u, self.v)
        return (self.v, self.u)

    def glyph(self) -> str:
        """Render the edge, e.g. ``"X o-> Y"``."""
        left = self.u_mark.char(left=True)
        right = self.v_mark.char(left=False)
        return f"{self.u} {left}-{right} {self.v}"

    def __repr__(self) -> str:
        return f"Edge({self.glyph()!r})"


EdgeSpec = Edge | str | tuple[Any, ...]
"""How a caller can name an edge: an ``Edge``, a glyph string like ``"X o-> Y"``,
a 2-tuple ``(u, v)`` (read as ``u -> v``), or a 4-tuple ``(u, u_mark, v, v_mark)``."""

_LEFT = {"-": Mark.TAIL, "<": Mark.ARROW, "o": Mark.CIRCLE}
_RIGHT = {"-": Mark.TAIL, ">": Mark.ARROW, "o": Mark.CIRCLE}


def _parse_edge_string(spec: str) -> Edge:
    parts = spec.split()
    if len(parts) != 3:
        raise ValueError(f"cannot parse edge {spec!r}; expected 'A <mark> B'")
    a, mid, b = parts
    left, right = mid[0], mid[-1]
    if left not in _LEFT or right not in _RIGHT:
        raise ValueError(f"cannot parse edge marks in {spec!r}")
    return Edge.make(a, _LEFT[left], b, _RIGHT[right])


def _parse_spec(spec: EdgeSpec) -> Edge:
    if isinstance(spec, Edge):
        return spec
    if isinstance(spec, str):
        return _parse_edge_string(spec)
    if isinstance(spec, tuple):
        if len(spec) == 2:
            u, v = spec
            return Edge.make(u, Mark.TAIL, v, Mark.ARROW)
        if len(spec) == 4:
            a, a_mark, b, b_mark = spec
            return Edge.make(a, a_mark, b, b_mark)
    raise TypeError(f"unsupported edge spec: {spec!r}")


def _build_edges(specs: Iterable[EdgeSpec]) -> frozenset[Edge]:
    if isinstance(specs, str):
        raise TypeError("pass an iterable of edge specs, not a single string")
    by_pair: dict[frozenset[str], Edge] = {}
    for spec in specs:
        edge = _parse_spec(spec)
        by_pair[frozenset((edge.u, edge.v))] = edge
    return frozenset(by_pair.values())


@dataclass(frozen=True)
class MixedGraph:
    """An immutable mixed graph over named nodes.

    Use the :func:`dag`, :func:`admg`, :func:`cpdag`, :func:`pag`, or
    :func:`mixed` builders rather than constructing this directly.
    """

    nodes: frozenset[str] = field(default_factory=frozenset)
    edges: frozenset[Edge] = field(default_factory=frozenset)
    latent: frozenset[str] = field(default_factory=frozenset)
    selection: frozenset[str] = field(default_factory=frozenset)
    kind: GraphKind = GraphKind.MIXED

    # -- lookups -------------------------------------------------------------

    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def node_list(self) -> list[str]:
        return sorted(self.nodes)

    def edge_list(self) -> list[Edge]:
        return sorted(self.edges, key=lambda e: (e.u, e.v))

    def has_edge(self, a: str, b: str) -> bool:
        return self.get_edge(a, b) is not None

    def get_edge(self, a: str, b: str) -> Edge | None:
        pair = frozenset((a, b))
        for edge in self.edges:
            if frozenset((edge.u, edge.v)) == pair:
                return edge
        return None

    def mark_at(self, a: str, b: str, node: str) -> Mark | None:
        edge = self.get_edge(a, b)
        return None if edge is None else edge.mark_at(node)

    # -- neighbourhoods ------------------------------------------------------

    def _incident(self, node: str) -> Iterator[Edge]:
        return (e for e in self.edges if node in (e.u, e.v))

    def neighbors(self, x: str) -> frozenset[str]:
        return frozenset(e.other(x) for e in self._incident(x))

    def parents(self, x: str) -> frozenset[str]:
        out: set[str] = set()
        for e in self._incident(x):
            pair = e.directed_pair()
            if pair is not None and pair[1] == x:
                out.add(pair[0])
        return frozenset(out)

    def children(self, x: str) -> frozenset[str]:
        out: set[str] = set()
        for e in self._incident(x):
            pair = e.directed_pair()
            if pair is not None and pair[0] == x:
                out.add(pair[1])
        return frozenset(out)

    def spouses(self, x: str) -> frozenset[str]:
        return frozenset(e.other(x) for e in self._incident(x) if e.is_bidirected)

    def ancestors(self, x: str) -> frozenset[str]:
        reverse: dict[str, set[str]] = {n: set() for n in self.nodes}
        for e in self.edges:
            pair = e.directed_pair()
            if pair is not None:
                reverse[pair[1]].add(pair[0])
        return self._reach(x, reverse)

    def descendants(self, x: str) -> frozenset[str]:
        forward: dict[str, set[str]] = {n: set() for n in self.nodes}
        for e in self.edges:
            pair = e.directed_pair()
            if pair is not None:
                forward[pair[0]].add(pair[1])
        return self._reach(x, forward)

    @staticmethod
    def _reach(start: str, adj: dict[str, set[str]]) -> frozenset[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        seen.discard(start)
        return frozenset(seen)

    def markov_blanket(self, x: str) -> frozenset[str]:
        """Parents, children, and the other parents of children (DAG only)."""
        if not self.is_dag():
            raise NotImplementedError("markov_blanket is implemented for DAGs only")
        blanket: set[str] = set(self.parents(x)) | set(self.children(x))
        for child in self.children(x):
            blanket |= self.parents(child)
        blanket.discard(x)
        return frozenset(blanket)

    # -- immutable edits -----------------------------------------------------

    def add_nodes(self, *names: str) -> MixedGraph:
        return replace(self, nodes=self.nodes | frozenset(names))

    def add_edge(self, spec: EdgeSpec) -> MixedGraph:
        edge = _parse_spec(spec)
        existing = self.get_edge(edge.u, edge.v)
        rest = self.edges - ({existing} if existing else frozenset())
        return replace(
            self,
            nodes=self.nodes | {edge.u, edge.v},
            edges=rest | {edge},
            kind=GraphKind.MIXED,
        )

    def remove_edge(self, a: str, b: str) -> MixedGraph:
        edge = self.get_edge(a, b)
        if edge is None:
            return self
        return replace(self, edges=self.edges - {edge})

    def orient(self, parent: str, child: str) -> MixedGraph:
        """Return a graph with the edge between the two nodes set to ``parent -> child``."""
        return self.add_edge((parent, Mark.TAIL, child, Mark.ARROW))

    def as_kind(self, kind: GraphKind) -> MixedGraph:
        """Re-tag the graph as ``kind`` after checking it satisfies that kind."""
        out = replace(self, kind=kind)
        out.validate()
        return out

    # -- kind checks ---------------------------------------------------------

    def is_acyclic(self) -> bool:
        forward: dict[str, set[str]] = {n: set() for n in self.nodes}
        for e in self.edges:
            pair = e.directed_pair()
            if pair is not None:
                forward[pair[0]].add(pair[1])
        color: dict[str, int] = {}

        def visit(node: str) -> bool:
            color[node] = 1
            for nxt in forward.get(node, ()):
                state = color.get(nxt, 0)
                if state == 1:
                    return False
                if state == 0 and not visit(nxt):
                    return False
            color[node] = 2
            return True

        return all(visit(n) for n in self.nodes if color.get(n, 0) == 0)

    def is_dag(self) -> bool:
        return all(e.is_directed for e in self.edges) and self.is_acyclic()

    def is_admg(self) -> bool:
        marks_ok = all(e.is_directed or e.is_bidirected for e in self.edges)
        return marks_ok and self.is_acyclic()

    def is_cpdag(self) -> bool:
        marks_ok = all(e.is_directed or e.is_undirected for e in self.edges)
        return marks_ok and self.is_acyclic()

    def is_mag(self) -> bool:
        if not self.is_admg():
            return False
        for e in self.edges:
            if e.is_bidirected and (e.u in self.ancestors(e.v) or e.v in self.ancestors(e.u)):
                return False
        return True

    def is_pag(self) -> bool:
        return self.is_acyclic()

    def validate(self, kind: GraphKind | None = None) -> None:
        """Check the graph against ``kind`` (default: its own), raising on failure."""
        target = self.kind if kind is None else kind
        for e in self.edges:
            if e.u not in self.nodes or e.v not in self.nodes:
                raise InvalidGraphError(f"edge {e.glyph()!r} references an unknown node")
        checks = {
            GraphKind.MIXED: lambda: True,
            GraphKind.DAG: self.is_dag,
            GraphKind.ADMG: self.is_admg,
            GraphKind.MAG: self.is_mag,
            GraphKind.CPDAG: self.is_cpdag,
            GraphKind.PAG: self.is_pag,
        }
        if not checks[target]():
            raise InvalidGraphError(f"graph does not satisfy the rules of {target.value}")

    # -- algorithms (delegated, kept out of this class) ----------------------

    def m_separated(self, x: Any, y: Any, given: Any = frozenset()) -> bool:
        from ._separation import m_separated

        return m_separated(self, x, y, given)

    def d_separated(self, x: Any, y: Any, given: Any = frozenset()) -> bool:
        from ._separation import d_separated

        return d_separated(self, x, y, given)

    def do(self, targets: Any) -> MixedGraph:
        """Graph surgery for an intervention: cut every arrowhead into ``targets``."""
        from ._surgery import intervene

        return intervene(self, targets)

    def induced_subgraph(self, nodes: Any) -> MixedGraph:
        from ._surgery import induced_subgraph

        return induced_subgraph(self, nodes)

    def ancestral_subgraph(self, nodes: Any) -> MixedGraph:
        from ._surgery import ancestral_subgraph

        return ancestral_subgraph(self, nodes)

    # -- interop -------------------------------------------------------------

    def to_networkx(self) -> Any:
        from ._io import to_networkx

        return to_networkx(self)

    def to_adjacency(self, order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        from ._io import to_adjacency

        return to_adjacency(self, order)

    def to_dict(self) -> dict[str, Any]:
        from ._io import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MixedGraph:
        from ._io import from_dict

        return from_dict(data)

    @classmethod
    def from_networkx(cls, graph: Any) -> MixedGraph:
        from ._io import from_networkx

        return from_networkx(graph)

    def __repr__(self) -> str:
        glyphs = ", ".join(e.glyph() for e in self.edge_list())
        return f"MixedGraph(kind={self.kind.value}, nodes={len(self.nodes)}, edges=[{glyphs}])"


# -- builders ----------------------------------------------------------------


def mixed(
    edges: Iterable[EdgeSpec] = (),
    *,
    nodes: Iterable[str] = (),
    latent: Iterable[str] = (),
    selection: Iterable[str] = (),
    kind: GraphKind = GraphKind.MIXED,
) -> MixedGraph:
    """Build a mixed graph of the given ``kind`` and check it satisfies that kind."""
    built = _build_edges(edges)
    node_set = frozenset(nodes) | frozenset(n for e in built for n in (e.u, e.v))
    graph = MixedGraph(
        nodes=node_set,
        edges=built,
        latent=frozenset(latent),
        selection=frozenset(selection),
        kind=kind,
    )
    graph.validate()
    return graph


def dag(edges: Iterable[EdgeSpec] = (), **kwargs: Any) -> MixedGraph:
    """Build a directed acyclic graph. Edges default to directed (``u -> v``)."""
    return mixed(edges, kind=GraphKind.DAG, **kwargs)


def admg(edges: Iterable[EdgeSpec] = (), **kwargs: Any) -> MixedGraph:
    """Build an acyclic directed mixed graph (directed and bidirected edges)."""
    return mixed(edges, kind=GraphKind.ADMG, **kwargs)


def cpdag(edges: Iterable[EdgeSpec] = (), **kwargs: Any) -> MixedGraph:
    """Build a CPDAG (directed and undirected edges)."""
    return mixed(edges, kind=GraphKind.CPDAG, **kwargs)


def pag(edges: Iterable[EdgeSpec] = (), **kwargs: Any) -> MixedGraph:
    """Build a partial ancestral graph (any marks, including circles)."""
    return mixed(edges, kind=GraphKind.PAG, **kwargs)
