# SPDX-License-Identifier: Apache-2.0
"""Greedy relaxations of the sparsest permutation (Lam, Andrews, and Ramsey,
2022): a depth-first search over tucks.

Like BOSS, GRaSP scores a permutation through its grow-shrink projection,
but its move is the tuck. For an edge ``j -> k`` of the projected DAG (so
``j`` sits before ``k``), ``tuck(j, k)`` moves ``k``, together with the
variables between them that are ancestors of ``k`` in the projection (order
preserved), to just before ``j``. The pair swaps sides, the edge gets a
chance to reverse, and the rest of the segment slides after ``j``.

The search is a depth-first walk over tucks. At the first level every edge
is a candidate; the first tuck that strictly improves the total (by more
than the shared ``1e-6`` slack) is accepted and the walk restarts from the
new permutation. A covered edge (``parents(j) == parents(k) - {j}``) whose
tuck ties the total opens a recursion, up to ``depth`` levels, because a
chain of score-neutral covered flips can unlock an improvement a single
move cannot see; non-covered tucks are tried only at the first level. A
history of visited permutations keeps tied walks from cycling. The search
stops when a full walk finds no improvement, and the output is the CPDAG of
the final projection, like BOSS without its BES step.

Knowledge enters as in BOSS: ``permits_edge`` gates grow, required edges
seed the grow set and pin the permutation's partial order (a tuck that
would put a required parent after its child is skipped), and the final
graph gets the compelled orientations and is checked. One honest limit:
tucks act only on edges of the projection, so a variable whose adjacencies
are gated off in the current ordering has no edge to tuck and can strand
with less than the knowledge allows. BOSS moves variables by position and
does not have this failure; prefer it when the knowledge is heavy.

The default run is fully deterministic: the sorted start, sorted edge
sweeps, and first-improvement acceptance leave nothing to chance.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

from ...graph import dag
from ...graph._meek import cpdag_of_dag
from ...knowledge import _refuse_soft
from .._registry import resolve_score
from .._result import DiscoveryResult
from .._score import Score
from ._boss import _SLACK, BOSS, _project, _required_consistent
from ._ges import GES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...data import TabularData
    from ...knowledge import DomainKnowledge

_ParentMap = dict[str, frozenset[str]]


def _ancestors(parents_of: _ParentMap, node: str) -> set[str]:
    """The ancestors of ``node`` in a projected DAG, excluding itself."""
    out: set[str] = set()
    stack = [node]
    while stack:
        for parent in parents_of[stack.pop()]:
            if parent not in out:
                out.add(parent)
                stack.append(parent)
    return out


def _tuck(order: Sequence[str], j: str, k: str, parents_of: _ParentMap) -> list[str]:
    """``order`` with ``k`` and its segment-ancestors moved to just before ``j``.

    The segment is everything strictly between ``j`` and ``k``; the part of
    it that is ancestral to ``k`` in the projection moves with ``k`` (order
    preserved), the rest stays behind ``j``.
    """
    pj = order.index(j)
    pk = order.index(k)
    segment = order[pj + 1 : pk]
    ancestral = _ancestors(parents_of, k)
    delta2 = [w for w in segment if w in ancestral]
    delta1 = [w for w in segment if w not in ancestral]
    return [*order[:pj], *delta2, k, j, *delta1, *order[pk + 1 :]]


def _respects_required(order: Sequence[str], req_parents: _ParentMap) -> bool:
    """Whether every required parent sits before its child in ``order``."""
    position = {name: i for i, name in enumerate(order)}
    return all(
        position[parent] < position[child]
        for child, parents in req_parents.items()
        for parent in parents
    )


class GRaSP:
    """Greedy relaxations of the sparsest permutation over a decomposable score.

    Args:
        score: A :class:`~ergodic.discovery.Score` instance (bound or
            unbound) or a name: ``"auto"`` (pick by the data schema),
            ``"bic"``, ``"bic_discrete"``, ``"bdeu"``, ``"bic_mixed"``.
        depth: How many levels of tied covered tucks the walk may chain
            before backtracking; the paper's experiments use 3.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    carries the CPDAG face when the final PDAG is a valid CPDAG and the
    ``mixed`` face otherwise (forced knowledge orientations can leave the
    class; ``diagnostics["cpdag_valid"]`` says which happened).
    """

    ingredient: ClassVar[str] = "score"

    def __init__(self, score: Score | str = "auto", *, depth: int = 3) -> None:
        if not isinstance(score, Score | str):
            raise TypeError(f"score must be a Score or a string, got {type(score).__name__}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self._score = score
        self._depth = depth

    # -- the public entry -------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Walk tucks to convergence and return the class found."""
        _refuse_soft(knowledge, "GRaSP")
        score = resolve_score(self._score, data, algorithm=type(self).__name__)
        variables = list(score.variables)
        if not type(score).score_equivalent:
            warnings.warn(
                f"{type(score).__name__} declares score_equivalent=False; GRaSP walks "
                "tied covered tucks as if every member of a class scored the same, "
                "so the search may not be optimal",
                UserWarning,
                stacklevel=2,
            )
        req_parents, _ = BOSS._required_maps(variables, knowledge)
        order = _required_consistent(variables, req_parents)
        parents_of, total = _project(score, order, knowledge)
        n_tucks = 0
        n_visited = 0
        while True:
            history = {tuple(order)}
            found = self._dfs(
                score, order, parents_of, total, knowledge, req_parents, self._depth, history
            )
            n_visited += len(history) - 1
            if found is None:
                break
            order, parents_of, total = found
            n_tucks += 1
        edges = [f"{u} -> {v}" for v in variables for u in sorted(parents_of[v])]
        state = cpdag_of_dag(dag(edges, nodes=variables))
        final, cpdag_valid = GES._face(GES._knowledge_tail(state, knowledge))
        if knowledge is not None and not knowledge.is_consistent(final):
            raise ValueError(
                "GRaSP produced a graph that violates the domain knowledge; the data and "
                "the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics = dict(score.diagnostics())
        diagnostics["n_tucks"] = float(n_tucks)
        diagnostics["n_visited"] = float(n_visited)
        diagnostics["cpdag_valid"] = cpdag_valid
        return DiscoveryResult(
            graph=final,
            method="grasp",
            score=score.name,
            alpha=None,
            n=score.n,
            diagnostics=diagnostics,
        )

    # -- the walk -------------------------------------------------------------------

    def _dfs(
        self,
        score: Score,
        order: list[str],
        parents_of: _ParentMap,
        total: float,
        knowledge: DomainKnowledge | None,
        req_parents: _ParentMap,
        depth_left: int,
        history: set[tuple[str, ...]],
    ) -> tuple[list[str], _ParentMap, float] | None:
        """The first strictly improving permutation reachable by tucks, or ``None``.

        ``total`` is the score of the permutation the walk started from;
        every state on a tied chain shares it. Edges are swept child-first
        in sorted name order, so the walk is deterministic. Non-covered
        tucks are tried only at the first level (``depth_left`` at its
        starting value); tied covered tucks recurse while depth remains.
        """
        at_root = depth_left == self._depth
        for k in sorted(order):
            for j in sorted(parents_of[k]):
                covered = parents_of[j] == parents_of[k] - {j}
                if not covered and not at_root:
                    continue
                candidate = _tuck(order, j, k, parents_of)
                key = tuple(candidate)
                if key in history or not _respects_required(candidate, req_parents):
                    continue  # seen on this walk, or a required parent would trail its child
                history.add(key)
                new_parents, new_total = _project(score, candidate, knowledge)
                if new_total > total + _SLACK:
                    return candidate, new_parents, new_total
                if covered and depth_left > 1 and abs(new_total - total) <= _SLACK:
                    found = self._dfs(
                        score,
                        candidate,
                        new_parents,
                        total,
                        knowledge,
                        req_parents,
                        depth_left - 1,
                        history,
                    )
                    if found is not None:
                        return found
        return None
