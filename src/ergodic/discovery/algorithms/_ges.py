# SPDX-License-Identifier: Apache-2.0
"""Greedy equivalence search (Chickering, 2002): two phases over CPDAGs.

GES never walks individual DAGs. Its state is an equivalence class, held as
a CPDAG, and its moves are Chickering's Insert and Delete operators, each of
which maps one class to a neighbouring class. The forward phase applies the
single best strictly positive Insert until none remains; the backward phase
then does the same with Delete. Every applied operator is followed by a
completion step (a consistent extension of the working PDAG, then the CPDAG
of that DAG), so the state stays a proper equivalence class throughout.

The optimality story is asymptotic. When the score is consistent and score
equivalent (every DAG in a class scores the same) and the data are faithful
to some DAG, the two phases land on the true equivalence class as ``n``
grows. At finite ``n`` GES is a greedy search like any other: fast, usually
right, and occasionally beaten by exact search. A score that declares
``score_equivalent = False`` draws a warning and the search runs anyway,
with no optimality claim at all.

Two honest caveats. First, enumerating the ``T`` subsets of an Insert (and
the ``H`` subsets of a Delete) is exponential in the size of the local
neighbourhood, so dense problems get expensive; ``max_degree`` is the
guard. Second, knowledge-constrained GES is a heuristic: required edges
seed the start and are never deleted, an Insert must be permitted by the
knowledge, and a final orientation pass enforces what the constraints
compel, but Chickering's optimality argument assumes an unconstrained
search and does not survive these gates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

from ...graph import Edge, GraphKind, InvalidGraphError, Mark, MixedGraph
from ...graph._meek import consistent_extension, cpdag_of_dag, meek_closure
from .._registry import resolve_score
from .._result import DiscoveryResult
from .._score import Score

if TYPE_CHECKING:
    from ...data import TabularData
    from ...knowledge import DomainKnowledge


@dataclass(frozen=True)
class _State:
    """Neighbour maps of the current class, rebuilt fresh before each sweep."""

    parents: dict[str, set[str]]
    children: dict[str, set[str]]
    undirected: dict[str, set[str]]
    adjacent: dict[str, set[str]]


def _state_of(graph: MixedGraph, nodes: list[str]) -> _State:
    """Split a completed class into parent, child, undirected, and adjacency maps."""
    parents: dict[str, set[str]] = {v: set() for v in nodes}
    children: dict[str, set[str]] = {v: set() for v in nodes}
    undirected: dict[str, set[str]] = {v: set() for v in nodes}
    for edge in graph.edges:
        pair = edge.directed_pair()
        if pair is not None:
            children[pair[0]].add(pair[1])
            parents[pair[1]].add(pair[0])
        else:  # a completed class carries only directed and undirected edges
            undirected[edge.u].add(edge.v)
            undirected[edge.v].add(edge.u)
    adjacent = {v: parents[v] | children[v] | undirected[v] for v in nodes}
    return _State(parents=parents, children=children, undirected=undirected, adjacent=adjacent)


def _is_clique(members: set[str], adjacent: dict[str, set[str]]) -> bool:
    """Every pair of ``members`` adjacent in some orientation."""
    return all(b in adjacent[a] for a, b in combinations(sorted(members), 2))


def _reaches_semidirected(start: str, target: str, blocked: set[str], state: _State) -> bool:
    """Whether a semi-directed path runs from ``start`` to ``target`` avoiding ``blocked``.

    A semi-directed path steps along undirected edges in either role and
    along directed edges only in their direction. Treating ``blocked`` as
    removed, ``target`` is reachable exactly when some such path dodges
    every blocked node, which is how an Insert's path condition is checked.
    """
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for nxt in state.children[node] | state.undirected[node]:
            if nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                stack.append(nxt)
    return False


def _complete(pdag: MixedGraph) -> MixedGraph:
    """PDAG to class: a consistent extension, then the CPDAG of that DAG."""
    extension = consistent_extension(pdag)
    if extension is None:  # pragma: no cover - a valid operator always extends
        raise RuntimeError(
            "a GES operator produced a PDAG with no consistent extension; this is a bug"
        )
    return cpdag_of_dag(extension)


@dataclass(frozen=True)
class _Move:
    """One scored operator: Insert(x, y, subset) or Delete(x, y, subset)."""

    delta: float
    x: str
    y: str
    subset: frozenset[str]


class GES:
    """Greedy equivalence search over a decomposable score.

    Args:
        score: A :class:`~ergodic.discovery.Score` instance (bound or
            unbound) or a name: ``"auto"`` (pick by the data schema),
            ``"bic"``, ``"bic_discrete"``, ``"bdeu"``, ``"bic_mixed"``.
        max_degree: A cap on the number of edges at any node; an Insert
            that touches a node already at the cap is never considered.
            ``None`` leaves degrees free.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    carries the CPDAG face when the final PDAG is a valid CPDAG and the
    ``mixed`` face otherwise (forced knowledge orientations can leave the
    class; ``diagnostics["cpdag_valid"]`` says which happened).
    """

    ingredient: ClassVar[str] = "score"

    def __init__(self, score: Score | str = "auto", *, max_degree: int | None = None) -> None:
        if not isinstance(score, Score | str):
            raise TypeError(f"score must be a Score or a string, got {type(score).__name__}")
        if max_degree is not None and max_degree < 1:
            raise ValueError(f"max_degree must be None or at least 1, got {max_degree}")
        self._score = score
        self._max_degree = max_degree

    # -- the public entry -------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Run the forward and backward phases and return the class found."""
        score = resolve_score(self._score, data, algorithm=type(self).__name__)
        variables = list(score.variables)
        if not type(score).score_equivalent:
            warnings.warn(
                f"{type(score).__name__} declares score_equivalent=False; GES assumes "
                "score equivalence when it moves between classes, so the search may "
                "not be optimal",
                UserWarning,
                stacklevel=2,
            )
        current = self._start(variables, knowledge)
        n_inserts = 0
        while (move := self._best_insert(score, current, variables, knowledge)) is not None:
            current = self._apply_insert(current, move)
            n_inserts += 1
        n_deletes = 0
        while (move := self._best_delete(score, current, variables, knowledge)) is not None:
            current = self._apply_delete(current, move)
            n_deletes += 1
        final, cpdag_valid = self._face(self._knowledge_tail(current, knowledge))
        if knowledge is not None and not knowledge.is_consistent(final):
            raise ValueError(
                "GES produced a graph that violates the domain knowledge; the data and "
                "the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics = dict(score.diagnostics())
        diagnostics["n_inserts"] = float(n_inserts)
        diagnostics["n_deletes"] = float(n_deletes)
        diagnostics["n_edges"] = float(len(final.edges))
        diagnostics["n_directed"] = float(sum(1 for e in final.edges if e.is_directed))
        diagnostics["n_undirected"] = float(sum(1 for e in final.edges if e.is_undirected))
        diagnostics["cpdag_valid"] = cpdag_valid
        return DiscoveryResult(
            graph=final,
            method="ges",
            score=score.name,
            alpha=None,
            n=score.n,
            diagnostics=diagnostics,
        )

    # -- the start state ----------------------------------------------------------

    @staticmethod
    def _start(variables: list[str], knowledge: DomainKnowledge | None) -> MixedGraph:
        """The empty class, with any required edges seeded and completed in."""
        empty = MixedGraph(nodes=frozenset(variables), edges=frozenset(), kind=GraphKind.MIXED)
        if knowledge is None:
            return empty
        names = frozenset(variables)
        seeds = frozenset(
            Edge.make(u, Mark.TAIL, v, Mark.ARROW)
            for u, v in sorted(knowledge.required)
            if u in names and v in names
        )
        if not seeds:
            return empty
        extension = consistent_extension(replace(empty, edges=seeds))
        if extension is None:
            raise ValueError(
                "the required edges admit no DAG over these variables; "
                "check the knowledge for required-edge cycles"
            )
        return cpdag_of_dag(extension)

    # -- the operators --------------------------------------------------------------

    def _best_insert(
        self,
        score: Score,
        graph: MixedGraph,
        nodes: list[str],
        knowledge: DomainKnowledge | None,
    ) -> _Move | None:
        """The best strictly positive Insert(X, Y, T), or ``None`` at convergence.

        The sweep order is deterministic (X ascending, Y ascending, T by
        size then name) and only a strictly larger delta displaces the
        incumbent, so ties break toward the first candidate and runs repeat.
        """
        state = _state_of(graph, nodes)
        best: _Move | None = None
        for x in nodes:
            for y in nodes:
                if x == y or y in state.adjacent[x]:
                    continue
                if knowledge is not None and not knowledge.permits_edge(x, y):
                    continue
                if self._max_degree is not None and (
                    len(state.adjacent[x]) >= self._max_degree
                    or len(state.adjacent[y]) >= self._max_degree
                ):
                    continue
                na_yx = state.undirected[y] & state.adjacent[x]
                pool = sorted(state.undirected[y] - state.adjacent[x])
                for size in range(len(pool) + 1):
                    for combo in combinations(pool, size):
                        cond = na_yx | set(combo)
                        if not _is_clique(cond, state.adjacent):
                            continue
                        if _reaches_semidirected(y, x, cond, state):
                            continue
                        base = frozenset(state.parents[y] | cond)
                        delta = score.local(y, base | {x}) - score.local(y, base)
                        if delta > 0.0 and (best is None or delta > best.delta):
                            best = _Move(delta=delta, x=x, y=y, subset=frozenset(combo))
        return best

    @staticmethod
    def _best_delete(
        score: Score,
        graph: MixedGraph,
        nodes: list[str],
        knowledge: DomainKnowledge | None,
    ) -> _Move | None:
        """The best strictly positive Delete(X, Y, H), or ``None`` at convergence.

        Candidates are the ordered pairs whose edge reads ``x -> y`` or
        ``x -- y`` (never against an arrow); a pair the knowledge requires
        in either direction is protected. The sweep order and tie-breaking
        match :meth:`_best_insert`.
        """
        state = _state_of(graph, nodes)
        best: _Move | None = None
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                if y not in state.children[x] and y not in state.undirected[x]:
                    continue
                if knowledge is not None and (
                    knowledge.requires_edge(x, y) or knowledge.requires_edge(y, x)
                ):
                    continue
                na_yx = state.undirected[y] & state.adjacent[x]
                pool = sorted(na_yx)
                for size in range(len(pool) + 1):
                    for combo in combinations(pool, size):
                        kept = na_yx - set(combo)
                        if not _is_clique(kept, state.adjacent):
                            continue
                        base = frozenset(state.parents[y] | kept)
                        delta = score.local(y, base - {x}) - score.local(y, base | {x})
                        if delta > 0.0 and (best is None or delta > best.delta):
                            best = _Move(delta=delta, x=x, y=y, subset=frozenset(combo))
        return best

    @staticmethod
    def _apply_insert(graph: MixedGraph, move: _Move) -> MixedGraph:
        """Add ``x -> y``, orient each ``t -- y`` as ``t -> y``, and complete."""
        working = replace(graph, kind=GraphKind.MIXED)
        working = working.add_edge((move.x, Mark.TAIL, move.y, Mark.ARROW))
        for t in sorted(move.subset):
            working = working.orient(t, move.y)
        return _complete(working)

    @staticmethod
    def _apply_delete(graph: MixedGraph, move: _Move) -> MixedGraph:
        """Drop the x-y edge, orient ``y -> h`` and any undirected x-h as ``x -> h``, complete."""
        working = replace(graph, kind=GraphKind.MIXED).remove_edge(move.x, move.y)
        for h in sorted(move.subset):
            working = working.orient(move.y, h)
            between = working.get_edge(move.x, h)
            if between is not None and between.is_undirected:
                working = working.orient(move.x, h)
        return _complete(working)

    # -- the knowledge tail and the face ------------------------------------------------

    @staticmethod
    def _knowledge_tail(graph: MixedGraph, knowledge: DomainKnowledge | None) -> MixedGraph:
        """PC's orientation tail: compelled directions, then the guarded Meek closure."""
        if knowledge is None:
            return graph
        out = graph
        for u, v in sorted(knowledge.required):
            if out.get_edge(u, v) is not None:
                out = out.orient(u, v)
        for edge in out.edge_list():
            if not edge.is_undirected:
                continue
            forward = knowledge.permits_edge(edge.u, edge.v)
            backward = knowledge.permits_edge(edge.v, edge.u)
            if forward and not backward:
                out = out.orient(edge.u, edge.v)
            elif backward and not forward:
                out = out.orient(edge.v, edge.u)

        def forbids(u: str, v: str) -> bool:
            return not knowledge.permits_edge(u, v)

        return meek_closure(out, forbidden=forbids)

    @staticmethod
    def _face(graph: MixedGraph) -> tuple[MixedGraph, float]:
        """The CPDAG face when the graph earns it, the mixed face otherwise."""
        try:
            return graph.as_kind(GraphKind.CPDAG), 1.0
        except InvalidGraphError:
            return graph.as_kind(GraphKind.MIXED), 0.0
