# SPDX-License-Identifier: Apache-2.0
"""Greedy hill climbing over DAGs, with a tabu list.

The search walks DAG space with three moves: add an edge, remove one, or
reverse one. Every move is priced by the decomposable ``local_diff``, so a
step costs a couple of cached local evaluations, and the best admissible
move is applied whenever it improves the total. A tabu ring holds the
inverses of the recently applied moves, which stops the search from undoing
itself, and it buys a bounded number of non-improving escape steps before
the run stops; the best state seen along the way is what comes back, not
the last one visited.

This is the fast, dependency-free baseline: a local search with no
optimality guarantee. Reach for GES when you want large-sample consistency
over equivalence classes, or for :class:`~ergodic.discovery.AStar` when the
node count is small enough for the exact optimum.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, ClassVar

from ...graph import dag
from ...knowledge import _refuse_soft
from .._registry import resolve_score
from .._result import DiscoveryResult
from .._score import Score

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...data import TabularData
    from ...knowledge import DomainKnowledge

    # ("add", u, v) and ("remove", u, v) act on the edge u -> v;
    # ("reverse", u, v) turns u -> v into v -> u.
    _Move = tuple[str, str, str]

_EPS = 1e-9


def _beats(delta: float, incumbent: float) -> bool:
    """Whether ``delta`` is a strictly better move than ``incumbent``, beyond rounding.

    Score-equivalent moves (adding ``u -> v`` versus ``v -> u``, reversing a
    covered edge) have exactly equal deltas in exact arithmetic, but the computed
    values differ in the last few bits, and which bits depends on the BLAS build.
    Comparing them raw let that noise pick the move, so the climb took different
    paths on different machines. A difference within a relative ``_EPS`` is a tie,
    and a tie keeps the move enumerated first.
    """
    return delta > incumbent + _EPS * max(1.0, abs(incumbent))


def _inverse(move: _Move) -> _Move:
    """The move that undoes ``move``; what the tabu ring stores."""
    kind, u, v = move
    if kind == "add":
        return ("remove", u, v)
    if kind == "remove":
        return ("add", u, v)
    return ("reverse", v, u)


def _reaches(
    children: Mapping[str, set[str]],
    source: str,
    target: str,
    drop: tuple[str, str] | None = None,
) -> bool:
    """Whether a directed path runs from ``source`` to ``target``, ignoring the ``drop`` edge."""
    stack = [source]
    seen = {source}
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for child in children[node]:
            if (node, child) != drop and child not in seen:
                seen.add(child)
                stack.append(child)
    return False


def _has_cycle(parents_of: Mapping[str, set[str]]) -> bool:
    """Whether the directed graph in ``parents_of`` has a cycle (Kahn's peel)."""
    indegree = {node: len(ps) for node, ps in parents_of.items()}
    children: dict[str, list[str]] = {node: [] for node in parents_of}
    for node, ps in parents_of.items():
        for parent in ps:
            children[parent].append(node)
    ready = [node for node, degree in indegree.items() if degree == 0]
    removed = 0
    while ready:
        node = ready.pop()
        removed += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return removed != len(parents_of)


class HillClimb:
    """Greedy DAG search: add, remove, and reverse moves with a tabu ring.

    Steepest ascent from the empty (or knowledge-seeded) graph, returning
    the best state seen. ``n_iterations`` in the diagnostics counts applied
    moves, so it always equals ``n_adds + n_removes + n_reverses``.

    Args:
        score: A :class:`~ergodic.discovery.Score` instance (bound or
            unbound) or a name: ``"auto"`` (pick by the data schema),
            ``"bic"``, ``"bic_discrete"``, ``"bdeu"``, ``"bic_mixed"``.
        tabu: The length of the tabu ring (inverses of recent moves that may
            not be replayed) and the bound on consecutive non-improving
            escape steps; ``0`` disables both, so the search stops at the
            first local optimum.
        max_iter: A hard cap on the number of applied moves.
        max_parents: An in-degree cap enforced on adds and reversals;
            ``None`` leaves the in-degree free. Required edges are seeded
            whatever the cap.
    """

    ingredient: ClassVar[str] = "score"

    def __init__(
        self,
        score: Score | str = "auto",
        *,
        tabu: int = 10,
        max_iter: int = 10_000,
        max_parents: int | None = None,
    ) -> None:
        if not isinstance(score, Score | str):
            raise TypeError(f"score must be a Score or a string, got {type(score).__name__}")
        if tabu < 0:
            raise ValueError(f"tabu must be non-negative, got {tabu}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        if max_parents is not None and max_parents < 0:
            raise ValueError(f"max_parents must be non-negative, got {max_parents}")
        self._score = score
        self._tabu = tabu
        self._max_iter = max_iter
        self._max_parents = max_parents

    # -- the public entry -------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Climb to a high-scoring DAG and return it with its provenance."""
        _refuse_soft(knowledge, "HillClimb")
        score = resolve_score(self._score, data, algorithm=type(self).__name__)
        variables = sorted(score.variables)
        parents_of: dict[str, set[str]] = {v: set() for v in variables}
        required: frozenset[tuple[str, str]] = frozenset()
        if knowledge is not None:
            known = set(variables)
            required = frozenset(
                pair for pair in knowledge.required if pair[0] in known and pair[1] in known
            )
            for u, v in required:
                parents_of[v].add(u)
            if _has_cycle(parents_of):
                raise ValueError(
                    "the required edges form a cycle, so no DAG can satisfy the "
                    "knowledge; drop one of the required edges"
                )
        total = float(sum(score.local(node, parents_of[node]) for node in variables))
        best_total = total
        best = {node: frozenset(ps) for node, ps in parents_of.items()}
        ring: deque[_Move] | None = deque(maxlen=self._tabu) if self._tabu > 0 else None
        iterations = 0
        stalls = 0
        applied = {"add": 0, "remove": 0, "reverse": 0}
        while iterations < self._max_iter:
            found = self._best_move(score, variables, parents_of, required, knowledge, ring)
            if found is None:
                break
            move, delta = found
            if delta > _EPS:
                stalls = 0
            elif ring is not None and stalls < self._tabu:
                stalls += 1  # a bounded escape: take the least bad move anyway
            else:
                break
            kind, u, v = move
            if kind == "add":
                parents_of[v].add(u)
            elif kind == "remove":
                parents_of[v].remove(u)
            else:
                parents_of[v].remove(u)
                parents_of[u].add(v)
            total += delta
            iterations += 1
            applied[kind] += 1
            if ring is not None:
                ring.append(_inverse(move))
            if _beats(total, best_total):
                best_total = total
                best = {node: frozenset(ps) for node, ps in parents_of.items()}
        edges = [f"{u} -> {v}" for v in variables for u in sorted(best[v])]
        graph = dag(edges, nodes=variables)
        if knowledge is not None:
            problems = knowledge.violations(graph)
            if problems:
                raise ValueError(
                    "the climbed DAG violates the domain knowledge; constraints a DAG "
                    "cannot express (confounded pairs) or required edges over variables "
                    "missing from the data can cause this: " + "; ".join(problems)
                )
        diagnostics = dict(score.diagnostics())
        diagnostics["n_iterations"] = float(iterations)
        diagnostics["n_adds"] = float(applied["add"])
        diagnostics["n_removes"] = float(applied["remove"])
        diagnostics["n_reverses"] = float(applied["reverse"])
        diagnostics["n_edges"] = float(sum(len(ps) for ps in best.values()))
        return DiscoveryResult(
            graph=graph,
            method="hc",
            score=score.name,
            alpha=None,
            n=score.n,
            diagnostics=diagnostics,
        )

    # -- move selection -----------------------------------------------------------

    def _best_move(
        self,
        score: Score,
        variables: list[str],
        parents_of: dict[str, set[str]],
        required: frozenset[tuple[str, str]],
        knowledge: DomainKnowledge | None,
        ring: deque[_Move] | None,
    ) -> tuple[_Move, float] | None:
        """The best admissible move off the tabu ring, or ``None`` when nothing is legal.

        Moves are enumerated in a fixed order (adds, then removals, then
        reversals, each over sorted node pairs) and a tie keeps the first,
        so a run is deterministic for a given score and knowledge. Required
        pairs are never removed or reversed, adds and reversals respect
        ``permits_edge``, the in-degree cap, and acyclicity by reachability.
        """
        children: dict[str, set[str]] = {node: set() for node in variables}
        for node, ps in parents_of.items():
            for parent in ps:
                children[parent].add(node)
        cap = self._max_parents
        best: tuple[_Move, float] | None = None

        def consider(move: _Move, delta: float) -> None:
            nonlocal best
            if ring is not None and move in ring:
                return
            if best is None or _beats(delta, best[1]):
                best = (move, delta)

        for u in variables:
            for v in variables:
                if (
                    u != v
                    and u not in parents_of[v]
                    and v not in parents_of[u]
                    and (knowledge is None or knowledge.permits_edge(u, v))
                    and (cap is None or len(parents_of[v]) < cap)
                    and not _reaches(children, v, u)
                ):
                    consider(("add", u, v), score.local_diff(v, parents_of[v], u))
        for u in variables:
            for v in variables:
                if (
                    u != v
                    and u in parents_of[v]
                    and (u, v) not in required
                    and (v, u) not in required
                ):
                    removal = score.local(v, parents_of[v] - {u}) - score.local(v, parents_of[v])
                    consider(("remove", u, v), removal)
        for u in variables:
            for v in variables:
                if (
                    u != v
                    and u in parents_of[v]
                    and (u, v) not in required
                    and (v, u) not in required
                    and (knowledge is None or knowledge.permits_edge(v, u))
                    and (cap is None or len(parents_of[u]) < cap)
                    and not _reaches(children, u, v, drop=(u, v))
                ):
                    removal = score.local(v, parents_of[v] - {u}) - score.local(v, parents_of[v])
                    gain = removal + score.local_diff(u, parents_of[u], v)
                    consider(("reverse", u, v), gain)
        return best
