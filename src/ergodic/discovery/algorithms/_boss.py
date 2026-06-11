# SPDX-License-Identifier: Apache-2.0
"""Best order score search (Andrews, Ramsey, Sanchez-Romero, Camchong, and
Kummerfeld, 2023): a greedy search over variable orderings.

A permutation projects to a DAG by grow-shrink: each variable greedily grows
a parent set from the variables before it (add the best strictly improving
candidate, repeat), then greedily shrinks it (drop the best strictly
improving removal, repeat). The permutation's score is the sum of the
projected locals, and the search is a best-move sweep: each variable in turn
is tried at every position, and it moves to the best one only when that
beats its current position by more than a small slack (``1e-6``, the
threshold the published implementations use to keep float noise from
churning moves); ties resolve to the earliest position. Sweeps repeat until
one passes with no move. The paper caches projections in grow-shrink trees;
here the :class:`~ergodic.discovery.Score` cache of ``(node, parents)``
pairs plays that role, which is plenty at the node counts this library
searches.

The paper's correctness argument needs one more step: run the backward
equivalence step of GES (Chickering's delete phase) on the CPDAG of the
projected DAG. Their experiments ran without it; ``use_bes`` defaults on
here because this library holds its searches to the exact optimum. The
output is the equivalence class, as a CPDAG, like GES.

Knowledge enters in two ways. Forbidden directions and tiers arrive through
``permits_edge`` as plain gates on grow, and the permutation search stays
free: a tier-consistent optimum is always reachable because every
tier-consistent DAG has a tier-sorted topological order. Required edges
constrain more: they seed the grow set, shrink never drops them, and the
permutation keeps every required parent before its child (a free search
would dodge a required edge the data does not support by reordering the
pair). BES protects required pairs the way GES's delete does. As with GES,
the optimality story assumes an unconstrained search and does not survive
these gates.

The default run is fully deterministic: the start is the sorted variable
order, candidates are swept in sorted order, and only a strict improvement
displaces an incumbent. Restarts shuffle the start with a seeded generator
and keep the best final score, so they are deterministic too.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ...graph import dag
from ...graph._meek import cpdag_of_dag
from .._registry import resolve_score
from .._result import DiscoveryResult
from .._score import Score
from ._ges import GES

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ...data import TabularData
    from ...knowledge import DomainKnowledge

# The move threshold of the published implementations: a position must beat
# the current one by more than this before a variable moves.
_SLACK = 1e-6


def _grow_shrink(
    score: Score,
    node: str,
    prefix: Iterable[str],
    knowledge: DomainKnowledge | None,
) -> frozenset[str]:
    """Project one node onto its prefix: greedy grow, then greedy shrink.

    Grow starts from the required parents found in the prefix (empty without
    knowledge), adds the candidate whose addition scores highest whenever
    that strictly improves the current local, and stops otherwise. Shrink
    then drops the parent whose removal scores highest, again only on strict
    improvement. Candidates are gated by ``permits_edge`` and swept in
    sorted order, so exact ties go to the earliest name.
    """
    pool = frozenset(prefix)
    required: frozenset[str] = frozenset()
    if knowledge is not None:
        required = frozenset(w for w in pool if knowledge.requires_edge(w, node))
        pool = frozenset(w for w in pool if knowledge.permits_edge(w, node)) | required
    parents = set(required)
    current = score.local(node, parents)
    while True:
        addition: str | None = None
        best = current
        for w in sorted(pool - parents):
            value = score.local(node, parents | {w})
            if value > best:
                best = value
                addition = w
        if addition is None:
            break
        parents.add(addition)
        current = best
    while True:
        removal: str | None = None
        best = current
        for w in sorted(parents - required):
            value = score.local(node, parents - {w})
            if value > best:
                best = value
                removal = w
        if removal is None:
            break
        parents.remove(removal)
        current = best
    return frozenset(parents)


def _project(
    score: Score,
    order: Sequence[str],
    knowledge: DomainKnowledge | None,
) -> tuple[dict[str, frozenset[str]], float]:
    """The grow-shrink DAG of a permutation and its total score.

    The total sums the projected locals in sorted node order, so two
    permutations that project to the same DAG report bit-identical totals.
    """
    parents_of = {
        node: _grow_shrink(score, node, order[:i], knowledge) for i, node in enumerate(order)
    }
    total = float(sum(score.local(node, parents_of[node]) for node in sorted(parents_of)))
    return parents_of, total


class BOSS:
    """Best order score search over a decomposable score.

    Args:
        score: A :class:`~ergodic.discovery.Score` instance (bound or
            unbound) or a name: ``"auto"`` (pick by the data schema),
            ``"bic"``, ``"bic_discrete"``, ``"bdeu"``, ``"bic_mixed"``.
        use_bes: Finish with the backward equivalence step (GES's delete
            phase) on the CPDAG of the projected DAG. The paper's
            correctness proposition needs it, so it defaults on; the
            paper's experiments ran without it.
        restarts: Extra runs from shuffled starting permutations, on top
            of the sorted start. The best final score across starts wins,
            with the earlier start keeping ties.
        seed: The seed for the restart shuffles. ``None`` means a fixed
            default of 0, so a run is reproducible without choosing one;
            unused when ``restarts`` is 0.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    carries the CPDAG face when the final PDAG is a valid CPDAG and the
    ``mixed`` face otherwise (forced knowledge orientations can leave the
    class; ``diagnostics["cpdag_valid"]`` says which happened).
    """

    ingredient: ClassVar[str] = "score"

    def __init__(
        self,
        score: Score | str = "auto",
        *,
        use_bes: bool = True,
        restarts: int = 0,
        seed: int | None = None,
    ) -> None:
        if not isinstance(score, Score | str):
            raise TypeError(f"score must be a Score or a string, got {type(score).__name__}")
        if restarts < 0:
            raise ValueError(f"restarts must be non-negative, got {restarts}")
        self._score = score
        self._use_bes = use_bes
        self._restarts = restarts
        self._seed = seed

    # -- the public entry -------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Sweep permutations to convergence and return the class found."""
        score = resolve_score(self._score, data)
        variables = list(score.variables)
        if not type(score).score_equivalent:
            warnings.warn(
                f"{type(score).__name__} declares score_equivalent=False; BOSS scores "
                "orderings through projected DAGs and reports an equivalence class, "
                "so the search may not be optimal",
                UserWarning,
                stacklevel=2,
            )
        req_parents, req_children = self._required_maps(variables, knowledge)
        starts = [_required_consistent(variables, req_parents)]
        if self._restarts > 0:
            rng = np.random.default_rng(0 if self._seed is None else self._seed)
            for _ in range(self._restarts):
                shuffled = [variables[int(i)] for i in rng.permutation(len(variables))]
                starts.append(_required_consistent(shuffled, req_parents))
        best_order: list[str] = []
        best_total = -np.inf
        n_sweeps = 0
        n_moves = 0
        for start in starts:
            order, total, sweeps, moves = self._search(
                score, start, knowledge, req_parents, req_children
            )
            n_sweeps += sweeps
            n_moves += moves
            if total > best_total:
                best_order, best_total = order, total
        parents_of, _ = _project(score, best_order, knowledge)
        edges = [f"{u} -> {v}" for v in variables for u in sorted(parents_of[v])]
        state = cpdag_of_dag(dag(edges, nodes=variables))
        n_bes_deletes = 0
        if self._use_bes:
            while (move := GES._best_delete(score, state, variables, knowledge)) is not None:
                state = GES._apply_delete(state, move)
                n_bes_deletes += 1
        final, cpdag_valid = GES._face(GES._knowledge_tail(state, knowledge))
        if knowledge is not None and not knowledge.is_consistent(final):
            raise ValueError(
                "BOSS produced a graph that violates the domain knowledge; the data and "
                "the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics = dict(score.diagnostics())
        diagnostics["n_sweeps"] = float(n_sweeps)
        diagnostics["n_moves"] = float(n_moves)
        diagnostics["n_bes_deletes"] = float(n_bes_deletes)
        diagnostics["cpdag_valid"] = cpdag_valid
        return DiscoveryResult(
            graph=final,
            method="boss",
            score=score.name,
            alpha=None,
            n=score.n,
            diagnostics=diagnostics,
        )

    # -- the sweep ----------------------------------------------------------------

    @staticmethod
    def _search(
        score: Score,
        start: list[str],
        knowledge: DomainKnowledge | None,
        req_parents: dict[str, frozenset[str]],
        req_children: dict[str, frozenset[str]],
    ) -> tuple[list[str], float, int, int]:
        """Best-move sweeps from ``start`` until one passes with no move.

        A sweep visits every variable in sorted name order, scores the
        permutation with the variable inserted at every admissible position
        (required parents stay before it, required children after), and
        moves it only when the best position beats the current one by more
        than the slack; the earliest best position wins ties. Returns the
        converged order, its total, and the sweep and move counts.
        """
        order = list(start)
        n_sweeps = 0
        n_moves = 0
        moved = True
        while moved:
            moved = False
            n_sweeps += 1
            for node in sorted(order):
                rest = [u for u in order if u != node]
                current = order.index(node)
                lo = 0
                hi = len(rest)
                for i, u in enumerate(rest):
                    if u in req_parents[node]:
                        lo = i + 1
                    elif u in req_children[node]:
                        hi = min(hi, i)
                _, baseline = _project(score, order, knowledge)
                best_total = baseline
                best_pos = current
                for pos in range(lo, hi + 1):
                    if pos == current:
                        continue
                    candidate = [*rest[:pos], node, *rest[pos:]]
                    _, total = _project(score, candidate, knowledge)
                    if total > best_total:
                        best_total = total
                        best_pos = pos
                if best_pos != current and best_total > baseline + _SLACK:
                    order = [*rest[:best_pos], node, *rest[best_pos:]]
                    n_moves += 1
                    moved = True
        _, total = _project(score, order, knowledge)
        return order, total, n_sweeps, n_moves

    # -- required-edge structure -----------------------------------------------------

    @staticmethod
    def _required_maps(
        variables: list[str], knowledge: DomainKnowledge | None
    ) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
        """Per-node required parents and children, restricted to ``variables``."""
        req_parents: dict[str, set[str]] = {v: set() for v in variables}
        req_children: dict[str, set[str]] = {v: set() for v in variables}
        if knowledge is not None:
            names = set(variables)
            for u, v in sorted(knowledge.required):
                if u in names and v in names:
                    req_parents[v].add(u)
                    req_children[u].add(v)
        return (
            {v: frozenset(ps) for v, ps in req_parents.items()},
            {v: frozenset(cs) for v, cs in req_children.items()},
        )


def _required_consistent(
    preference: Sequence[str], req_parents: dict[str, frozenset[str]]
) -> list[str]:
    """``preference``, minimally repaired so required parents come first.

    Repeatedly places the earliest pending name whose required parents are
    all placed, so the result follows the preference order wherever the
    required edges allow and is deterministic. Raises when no name can be
    placed, which happens exactly when the required edges contain a cycle.
    """
    out: list[str] = []
    placed: set[str] = set()
    pending = list(preference)
    while pending:
        for i, node in enumerate(pending):
            if req_parents[node] <= placed:
                out.append(node)
                placed.add(node)
                del pending[i]
                break
        else:
            raise ValueError(
                "the required edges admit no ordering of these variables; "
                "check the knowledge for required-edge cycles"
            )
    return out
