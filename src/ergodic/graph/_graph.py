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
        """Whether the edge is ``->``: a tail at one end and an arrowhead at the other."""
        return {self.u_mark, self.v_mark} == {Mark.TAIL, Mark.ARROW}

    @property
    def is_bidirected(self) -> bool:
        """Whether the edge is ``<->``: an arrowhead at both ends."""
        return self.u_mark is Mark.ARROW and self.v_mark is Mark.ARROW

    @property
    def is_undirected(self) -> bool:
        """Whether the edge is ``--``: a tail at both ends."""
        return self.u_mark is Mark.TAIL and self.v_mark is Mark.TAIL

    @property
    def has_circle(self) -> bool:
        """Whether either end carries a circle mark (a PAG edge such as ``o->``)."""
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


def _pair_conflict_message(existing: Edge, incoming: Edge) -> str:
    message = (
        f"two different edges for the pair ({existing.u!r}, {existing.v!r}): "
        f"{existing.glyph()!r} and {incoming.glyph()!r}"
    )
    bow = (existing.is_directed and incoming.is_bidirected) or (
        existing.is_bidirected and incoming.is_directed
    )
    if bow:
        message += (
            "; the current data model holds one edge per pair, so a directed and a "
            "bidirected edge cannot share one"
        )
    return message


def _merge_edges(edges: Iterable[Edge]) -> frozenset[Edge]:
    """Collapse to one edge per node pair, raising when two edges disagree."""
    by_pair: dict[frozenset[str], Edge] = {}
    for edge in edges:
        pair = frozenset((edge.u, edge.v))
        existing = by_pair.get(pair)
        if existing is not None and existing != edge:
            raise ValueError(_pair_conflict_message(existing, edge))
        by_pair[pair] = edge
    return frozenset(by_pair.values())


def _build_edges(specs: Iterable[EdgeSpec]) -> frozenset[Edge]:
    if isinstance(specs, str):
        raise TypeError("pass an iterable of edge specs, not a single string")
    return _merge_edges(_parse_spec(spec) for spec in specs)


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
        """Return the node names, sorted."""
        return sorted(self.nodes)

    def edge_list(self) -> list[Edge]:
        """Return the edges, sorted by their stored ``(u, v)`` endpoints."""
        return sorted(self.edges, key=lambda e: (e.u, e.v))

    def has_edge(self, a: str, b: str) -> bool:
        """Return ``True`` if any edge joins ``a`` and ``b``, whatever its marks.

        The order of ``a`` and ``b`` does not matter.
        """
        return self.get_edge(a, b) is not None

    def get_edge(self, a: str, b: str) -> Edge | None:
        """Return the edge between ``a`` and ``b``, or ``None`` when they are not adjacent.

        The order of ``a`` and ``b`` does not matter. The edge keeps its stored
        orientation, so read its marks with :meth:`Edge.mark_at`.
        """
        pair = frozenset((a, b))
        for edge in self.edges:
            if frozenset((edge.u, edge.v)) == pair:
                return edge
        return None

    def mark_at(self, a: str, b: str, node: str) -> Mark | None:
        """Return the mark at ``node`` on the edge between ``a`` and ``b``.

        Args:
            a: One endpoint of the edge.
            b: The other endpoint.
            node: The endpoint whose mark you want; it must be ``a`` or ``b``.

        Returns:
            The :class:`Mark` at ``node``, or ``None`` when ``a`` and ``b`` are not adjacent.

        Raises:
            KeyError: ``node`` is not an endpoint of the edge.
        """
        edge = self.get_edge(a, b)
        return None if edge is None else edge.mark_at(node)

    # -- neighbourhoods ------------------------------------------------------

    def _incident(self, node: str) -> Iterator[Edge]:
        return (e for e in self.edges if node in (e.u, e.v))

    def neighbors(self, x: str) -> frozenset[str]:
        """Return every node that shares an edge with ``x``, whatever the marks."""
        return frozenset(e.other(x) for e in self._incident(x))

    def parents(self, x: str) -> frozenset[str]:
        """Return the nodes with a directed edge into ``x`` (``p -> x``)."""
        out: set[str] = set()
        for e in self._incident(x):
            pair = e.directed_pair()
            if pair is not None and pair[1] == x:
                out.add(pair[0])
        return frozenset(out)

    def children(self, x: str) -> frozenset[str]:
        """Return the nodes ``x`` points to with a directed edge (``x -> c``)."""
        out: set[str] = set()
        for e in self._incident(x):
            pair = e.directed_pair()
            if pair is not None and pair[0] == x:
                out.add(pair[1])
        return frozenset(out)

    def spouses(self, x: str) -> frozenset[str]:
        """Return the nodes joined to ``x`` by a bidirected edge (``x <-> s``)."""
        return frozenset(e.other(x) for e in self._incident(x) if e.is_bidirected)

    def ancestors(self, x: str) -> frozenset[str]:
        """Return the nodes with a directed path into ``x``, excluding ``x`` itself.

        Only directed edges count; undirected, bidirected and circle edges are not
        followed. An unknown node has no ancestors.
        """
        reverse: dict[str, set[str]] = {n: set() for n in self.nodes}
        for e in self.edges:
            pair = e.directed_pair()
            if pair is not None:
                reverse[pair[1]].add(pair[0])
        return self._reach(x, reverse)

    def descendants(self, x: str) -> frozenset[str]:
        """Return the nodes reachable from ``x`` along directed edges, excluding ``x``.

        Only directed edges count; undirected, bidirected and circle edges are not
        followed. An unknown node has no descendants.
        """
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
        """Return a graph with ``names`` added as isolated nodes; existing names are kept."""
        return replace(self, nodes=self.nodes | frozenset(names))

    def add_edge(self, spec: EdgeSpec) -> MixedGraph:
        """Add an edge on a free node pair; adding the edge already there is a no-op.

        Raises ``ValueError`` when the pair already carries a different edge.
        Call :meth:`remove_edge` first, or :meth:`orient` to redirect.
        """
        edge = _parse_spec(spec)
        existing = self.get_edge(edge.u, edge.v)
        if existing == edge:
            return self
        if existing is not None:
            raise ValueError(
                f"the pair ({edge.u!r}, {edge.v!r}) already has edge {existing.glyph()!r}; "
                f"call remove_edge first to replace it with {edge.glyph()!r}"
            )
        return replace(
            self,
            nodes=self.nodes | {edge.u, edge.v},
            edges=self.edges | {edge},
            kind=GraphKind.MIXED,
        )

    def remove_edge(self, a: str, b: str) -> MixedGraph:
        """Return a graph without the edge between ``a`` and ``b``.

        Both nodes stay in the graph and the graph's kind is unchanged. When the two
        nodes are not adjacent, the graph comes back as it is.
        """
        edge = self.get_edge(a, b)
        if edge is None:
            return self
        return replace(self, edges=self.edges - {edge})

    def orient(self, parent: str, child: str) -> MixedGraph:
        """Return a graph with the edge between the two nodes set to ``parent -> child``."""
        return self.remove_edge(parent, child).add_edge((parent, Mark.TAIL, child, Mark.ARROW))

    def set_mark(self, a: str, b: str, *, at: str, mark: Mark) -> MixedGraph:
        """Return a graph with the ``{a, b}`` edge's mark at endpoint ``at`` set to ``mark``.

        The finer-grained sibling of :meth:`orient`: one endpoint changes, the
        other keeps its mark (how the FCI rules orient one mark at a time).
        Raises ``KeyError`` when the pair carries no edge or ``at`` is not one
        of its endpoints.
        """
        edge = self.get_edge(a, b)
        if edge is None:
            raise KeyError(f"no edge between {a!r} and {b!r}")
        if at not in (edge.u, edge.v):
            raise KeyError(f"{at!r} is not an endpoint of the edge between {a!r} and {b!r}")
        other = edge.other(at)
        new = Edge.make(at, mark, other, edge.mark_at(other))
        if new == edge:
            return self
        return replace(self, edges=(self.edges - {edge}) | {new})

    def as_kind(self, kind: GraphKind | str) -> MixedGraph:
        """Re-tag the graph as ``kind`` (an enum or its name) after checking it fits."""
        if isinstance(kind, str):
            try:
                kind = GraphKind(kind.lower())
            except ValueError:
                names = ", ".join(k.value for k in GraphKind)
                raise ValueError(f"unknown graph kind {kind!r}; expected one of: {names}") from None
        out = replace(self, kind=kind)
        out.validate()
        return out

    # -- kind checks ---------------------------------------------------------

    def _kahn(self) -> tuple[list[str], bool]:
        """Kahn's algorithm over the directed edges: a topological order and acyclicity.

        Iterative so deep graphs do not hit the recursion limit. Ties (several
        ready nodes) break by name, so the order is deterministic. The second
        element is ``True`` when every node was emitted, i.e. the directed part
        is acyclic; on a cycle the order is the prefix that could be emitted.
        """
        forward: dict[str, set[str]] = {n: set() for n in self.nodes}
        for e in self.edges:
            pair = e.directed_pair()
            if pair is not None:
                forward.setdefault(pair[0], set()).add(pair[1])
                forward.setdefault(pair[1], set())
        indegree: dict[str, int] = dict.fromkeys(forward, 0)
        for targets in forward.values():
            for nxt in targets:
                indegree[nxt] += 1
        ready = sorted(n for n, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            newly: list[str] = []
            for nxt in forward[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    newly.append(nxt)
            if newly:
                ready = sorted(ready + newly)
        return order, len(order) == len(forward)

    def is_acyclic(self) -> bool:
        """Return ``True`` if the directed part of the graph has no cycle."""
        return self._kahn()[1]

    def is_dag(self) -> bool:
        """Every edge directed and no directed cycle. This check is complete."""
        return all(e.is_directed for e in self.edges) and self.is_acyclic()

    def is_admg(self) -> bool:
        """Directed or bidirected edges only, and no directed cycle. This check is complete."""
        marks_ok = all(e.is_directed or e.is_bidirected for e in self.edges)
        return marks_ok and self.is_acyclic()

    def is_cpdag(self) -> bool:
        """Return ``True`` when the graph really is a CPDAG. This check is complete.

        Three things must hold: every edge is directed or undirected, some DAG
        has exactly this skeleton and these v-structures (the Dor-Tarsi
        consistent extension exists), and the graph equals the CPDAG of that
        extension, so every compelled edge is directed and every reversible
        edge is left undirected.
        """
        if not all(e.is_directed or e.is_undirected for e in self.edges):
            return False
        if not self.is_acyclic():
            return False
        from ._meek import consistent_extension, cpdag_of_dag

        extension = consistent_extension(self)
        if extension is None:
            return False
        return cpdag_of_dag(extension).edges == self.edges

    def is_mag(self) -> bool:
        """The ancestral rules of Richardson and Spirtes (2002).

        Three edge types (directed, bidirected, undirected), no directed
        cycle, no almost-directed cycle (a bidirected edge between ancestors),
        and the endpoints of any undirected edge carry no arrowhead at all (no
        parents and no spouses), which is how selection bias shows up in a
        MAG. One gap: maximality (every non-adjacent pair separable) is not
        checked here; the :meth:`is_pag` round trip is what exercises it.
        """
        if not all(e.is_directed or e.is_bidirected or e.is_undirected for e in self.edges):
            return False
        if not self.is_acyclic():
            return False
        undirected = {n for e in self.edges if e.is_undirected for n in (e.u, e.v)}
        for e in self.edges:
            if e.is_bidirected and (e.u in self.ancestors(e.v) or e.v in self.ancestors(e.u)):
                return False
            if e.u_mark is Mark.ARROW and e.u in undirected:
                return False
            if e.v_mark is Mark.ARROW and e.v in undirected:
                return False
        return True

    def is_pag(self) -> bool:
        """Return ``True`` when the graph really is a complete PAG. This check is complete.

        The round trip decides it, exactly as :meth:`is_cpdag` does for
        CPDAGs: extract one member MAG (``mag_of_pag``), check it satisfies
        the MAG rules, and require the canonical PAG of that member
        (``pag_of_mag``) to be exactly this graph, so every invariant mark is
        oriented and every varying mark is a circle. Selection bias is in
        scope: undirected and circle-tail edges are read as Zhang's rules R5
        to R7 leave them. One known gap rides along from :meth:`is_mag`:
        maximality of the member is not checked.
        """
        if not self.is_acyclic():
            return False
        from ._pag import mag_of_pag, pag_of_mag

        try:
            member = mag_of_pag(self)
        except InvalidGraphError:
            return False
        return pag_of_mag(member).edges == self.edges

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
        """Return ``True`` if ``x`` and ``y`` are m-separated given ``given``.

        Each argument is a node name or an iterable of names. On a graph with circle
        marks (a PAG) the check reads definite-status paths, so on a complete PAG the
        answer holds in every member of the class. A CPDAG-kind graph gets the same
        class reading, with its undirected edges read as circles.

        Args:
            x: The first node or set of nodes.
            y: The second node or set of nodes.
            given: The conditioning set. Defaults to empty.

        Returns:
            ``True`` when no active path joins ``x`` and ``y`` given ``given``.

        Raises:
            ValueError: A name is not in the graph, or ``x``, ``y`` and ``given`` overlap.
        """
        from ._separation import m_separated

        return m_separated(self, x, y, given)

    def d_separated(self, x: Any, y: Any, given: Any = frozenset()) -> bool:
        """Return ``True`` if ``x`` and ``y`` are d-separated given ``given``.

        The DAG case of :meth:`m_separated`, with the same arguments.

        Raises:
            ValueError: The graph is not a DAG (use :meth:`m_separated`), a name is not
                in the graph, or the node sets overlap.
        """
        from ._separation import d_separated

        return d_separated(self, x, y, given)

    def do(self, targets: Any) -> MixedGraph:
        """Graph surgery for an intervention: cut every arrowhead into ``targets``."""
        from ._surgery import intervene

        return intervene(self, targets)

    def induced_subgraph(self, nodes: Any) -> MixedGraph:
        """Return the subgraph on ``nodes`` and the edges among them.

        Names not in the graph are ignored. Latent and selection flags carry over for
        the nodes that stay.
        """
        from ._surgery import induced_subgraph

        return induced_subgraph(self, nodes)

    def ancestral_subgraph(self, nodes: Any) -> MixedGraph:
        """Return the subgraph on ``nodes`` plus all their ancestors, with the edges among them.

        Ancestry follows directed edges only. Names not in the graph are ignored.
        """
        from ._surgery import ancestral_subgraph

        return ancestral_subgraph(self, nodes)

    # -- interop -------------------------------------------------------------

    def to_networkx(self) -> Any:
        """Return a ``networkx.MultiDiGraph`` that :meth:`from_networkx` reads back.

        Each edge becomes one ``u -> v`` arc with ``u_mark`` and ``v_mark`` data, and
        each node carries boolean ``latent`` and ``selection`` data. Needs networkx.

        Raises:
            ValueError: A node pair carries two edges (a projected bow), which the
                format cannot hold.
        """
        from ._io import to_networkx

        return to_networkx(self)

    def to_adjacency(self, order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Return the directed adjacency matrix and its node order.

        ``matrix[i, j] == 1`` when there is a directed edge from node ``i`` to node
        ``j``. Bidirected, undirected and circle edges do not appear.

        Args:
            order: The node order for rows and columns. Defaults to :meth:`node_list`.
                Nodes left out of ``order`` are dropped.

        Returns:
            A ``(matrix, order)`` pair; ``matrix`` is an integer array.

        Raises:
            ValueError: A node pair carries two edges (a projected bow).
        """
        from ._io import to_adjacency

        return to_adjacency(self, order)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict that :meth:`from_dict` reads back.

        The dict holds ``format`` (``"ergodic.graph/1"``), ``kind``, ``nodes``,
        ``edges`` (each with ``u``, ``v``, ``u_mark`` and ``v_mark``), ``latent`` and
        ``selection``.

        Raises:
            ValueError: A node pair carries two edges (a projected bow).
        """
        from ._io import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MixedGraph:
        """Rebuild a graph from :meth:`to_dict` output and validate it.

        A missing ``format`` field reads as ``"ergodic.graph/1"``. ``kind`` defaults to
        ``"mixed"``, and ``latent`` and ``selection`` default to empty.

        Args:
            data: The dict to read.

        Returns:
            The graph, checked against the rules of its kind.

        Raises:
            ValueError: The format is not ``"ergodic.graph/1"``, or two entries
                disagree about one node pair.
            InvalidGraphError: The graph breaks the rules of its declared kind.
        """
        from ._io import from_dict

        return from_dict(data)

    @classmethod
    def from_networkx(cls, graph: Any) -> MixedGraph:
        """Build a graph from a networkx graph.

        Edges with ``u_mark`` and ``v_mark`` data keep those marks; a plain edge reads
        as ``u -> v``. Node ``latent`` and ``selection`` flags carry over. The result
        is tagged DAG when every edge is directed and acyclic, otherwise mixed.

        Raises:
            ValueError: Two arcs disagree about one node pair, such as ``X -> Y`` and
                ``Y -> X``.
        """
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
    """Build a partial ancestral graph, validated as a complete PAG.

    The marks must be exactly what an FCI run would leave: every invariant
    mark oriented (closed under the orientation rules) and a circle wherever
    the equivalence class disagrees. Build with :func:`mixed` instead when
    you want circle marks without that guarantee.
    """
    return mixed(edges, kind=GraphKind.PAG, **kwargs)
