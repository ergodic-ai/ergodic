# SPDX-License-Identifier: Apache-2.0
"""Exact score-based search over the order lattice (Yuan and Malone, 2013).

A DAG is an ordering plus, for each node, the best parent set drawn from its
predecessors. A* therefore walks the lattice of variable subsets: a state is
the set of nodes already placed, a successor places one more node ``v`` at
cost ``-bp_v(state)``, the negated best local score of ``v`` with parents
inside the state, and the goal is the full set. The heuristic relaxes the
ordering away and lets every unplaced node take its best parents from all of
its candidates at once; that never overestimates and is consistent, so the
first time the goal leaves the frontier the path is a highest-scoring DAG.

Two tables per node make the edge costs O(1): every admissible parent set is
scored up front (``enumerate_locals``), then a best-parent-set sweep over
subsets of the candidate pool turns "best parents inside this state" into an
array lookup. Domain knowledge enters only through its public projections:
``permits_edge`` shrinks a node's candidate pool, ``requires_edge`` forces
parents into every admissible set, and the finished graph is checked with
``is_consistent``. This is reference-quality exact search at small ``p``;
``node_limit`` keeps the lattice honest.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...graph import dag
from .._registry import resolve_score
from .._result import DiscoveryResult
from .._score import Score

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ...data import TabularData
    from ...knowledge import DomainKnowledge

    _Projector = Callable[[int, str], int]


@dataclass(frozen=True, eq=False)
class _NodeTable:
    """Per-node search tables, indexed by bitmasks over ``candidates``.

    ``bp[S]`` is the best local score achievable with parents inside the
    candidate subset ``S`` (``-inf`` when no admissible set fits), and
    ``choice[S]`` is the bitmask of a parent set attaining it (``-1`` when
    none does).
    """

    candidates: tuple[str, ...]
    bp: npt.NDArray[np.float64]
    choice: npt.NDArray[np.int64]


class AStar:
    """Exact structure search: the highest-BIC DAG, found with A*.

    Args:
        score: A :class:`~ergodic.discovery.Score` instance (bound or
            unbound) or a name: ``"auto"`` (pick by the data schema),
            ``"bic"``, ``"bic_discrete"``, ``"bdeu"``, ``"bic_mixed"``.
        max_parents: An in-degree cap; ``None`` enumerates every parent set.
        node_limit: Refuse problems with more variables than this; exact
            search is exponential in the node count.
    """

    def __init__(
        self,
        score: Score | str = "auto",
        *,
        max_parents: int | None = None,
        node_limit: int = 20,
    ) -> None:
        if not isinstance(score, Score | str):
            raise TypeError(f"score must be a Score or a string, got {type(score).__name__}")
        if max_parents is not None and max_parents < 0:
            raise ValueError(f"max_parents must be non-negative, got {max_parents}")
        if node_limit < 1:
            raise ValueError(f"node_limit must be at least 1, got {node_limit}")
        self._score = score
        self._max_parents = max_parents
        self._node_limit = node_limit

    # -- the public entry -------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Search for a highest-scoring DAG and return it with its provenance."""
        score = resolve_score(self._score, data)
        variables = list(score.variables)
        if len(variables) > self._node_limit:
            raise ValueError(
                f"exact search over {len(variables)} variables exceeds "
                f"node_limit={self._node_limit}; cap the in-degree with max_parents, "
                "shrink the problem with DomainKnowledge.restrict_to, or raise "
                "node_limit if you accept the cost"
            )
        tables = {v: self._node_table(score, v, variables, knowledge) for v in variables}
        parents_of, expanded = self._search(variables, tables)
        edges = [f"{u} -> {v}" for v in variables for u in sorted(parents_of[v])]
        graph = dag(edges, nodes=variables)
        if knowledge is not None:
            assert knowledge.is_consistent(graph), (
                "the searched DAG violates the domain knowledge; constraints a DAG "
                "cannot express (confounded pairs) or required edges over variables "
                "missing from the data can cause this"
            )
        diagnostics = dict(score.diagnostics())
        diagnostics["expanded_states"] = float(expanded)
        return DiscoveryResult(
            graph=graph,
            method="astar",
            score=score.name,
            alpha=None,
            n=score.n,
            diagnostics=diagnostics,
        )

    # -- per-node tables ------------------------------------------------------------

    def _node_table(
        self,
        score: Score,
        node: str,
        variables: list[str],
        knowledge: DomainKnowledge | None,
    ) -> _NodeTable:
        """Score every admissible parent set of ``node`` and sweep the best-set DP."""
        others = [u for u in variables if u != node]
        required: frozenset[str] = frozenset()
        if knowledge is None:
            candidates = others
        else:
            candidates = [u for u in others if knowledge.permits_edge(u, node)]
            required = frozenset(u for u in others if knowledge.requires_edge(u, node))
            stranded = sorted(required - set(candidates))
            if stranded:
                raise ValueError(
                    f"required parent(s) {stranded} of {node!r} are not permitted "
                    "by the same knowledge"
                )
        if self._max_parents is not None and len(required) > self._max_parents:
            raise ValueError(
                f"{node!r} has {len(required)} required parents, more than "
                f"max_parents={self._max_parents}; raise the cap or relax the knowledge"
            )
        position = {name: i for i, name in enumerate(candidates)}
        scored = score.enumerate_locals(node, candidates, self._max_parents)
        size = 1 << len(candidates)
        exact = np.full(size, -math.inf, dtype=np.float64)
        for parent_set, value in scored.items():
            if required <= parent_set:
                mask = 0
                for parent in parent_set:
                    mask |= 1 << position[parent]
                exact[mask] = value
        # bp[S] = max(exact[S], max over single-element removals of bp[S \ {s}]),
        # swept in increasing mask order so every submask is already final.
        bp = exact.copy()
        choice = np.where(np.isfinite(exact), np.arange(size, dtype=np.int64), np.int64(-1))
        for mask in range(size):
            best = float(bp[mask])
            pick = int(choice[mask])
            remaining = mask
            while remaining:
                low = remaining & -remaining
                value = float(bp[mask ^ low])
                if value > best:
                    best = value
                    pick = int(choice[mask ^ low])
                remaining ^= low
            bp[mask] = best
            choice[mask] = pick
        return _NodeTable(candidates=tuple(candidates), bp=bp, choice=choice)

    # -- the search --------------------------------------------------------------

    def _search(
        self, variables: list[str], tables: dict[str, _NodeTable]
    ) -> tuple[dict[str, frozenset[str]], int]:
        """A* from the empty set to the full set; returns the parent map and a counter."""
        bit_of = {name: 1 << i for i, name in enumerate(variables)}
        projections = {
            v: tuple((bit_of[u], 1 << j) for j, u in enumerate(tables[v].candidates))
            for v in variables
        }

        def project(state: int, node: str) -> int:
            """Restrict a placed-set state to ``node``'s candidate pool bitmask."""
            out = 0
            for variable_bit, candidate_bit in projections[node]:
                if state & variable_bit:
                    out |= candidate_bit
            return out

        # The heuristic term per unplaced node: its best local over ALL candidates.
        # bp is monotone in the subset, so h never overestimates and is consistent.
        h_terms = {v: -float(tables[v].bp[-1]) for v in variables}
        full = (1 << len(variables)) - 1
        g_best: dict[int, float] = {0: 0.0}
        came: dict[int, tuple[int, str]] = {}
        closed: set[int] = set()
        # Entries are (f, state); equal f breaks ties on the state integer, so
        # the expansion order is deterministic.
        frontier: list[tuple[float, int]] = [(sum(h_terms.values()), 0)]
        expanded = 0
        while frontier:
            f_value, state = heapq.heappop(frontier)
            if state in closed:
                continue
            closed.add(state)
            expanded += 1
            if state == full:
                return self._reconstruct(state, came, tables, project), expanded
            g_state = g_best[state]
            h_state = f_value - g_state
            for node in variables:
                bit = bit_of[node]
                if state & bit:
                    continue
                local_best = float(tables[node].bp[project(state, node)])
                if math.isinf(local_best):
                    continue  # required parents not yet placed
                child = state | bit
                if child in closed:
                    continue
                g_child = g_state - local_best
                if g_child < g_best.get(child, math.inf):
                    g_best[child] = g_child
                    came[child] = (state, node)
                    f_child = g_child + h_state - h_terms[node]
                    heapq.heappush(frontier, (f_child, child))
        raise ValueError(
            "the constraints admit no DAG over these variables; "
            "check the knowledge for required-edge cycles"
        )

    @staticmethod
    def _reconstruct(
        goal: int,
        came: dict[int, tuple[int, str]],
        tables: dict[str, _NodeTable],
        project: _Projector,
    ) -> dict[str, frozenset[str]]:
        """Walk the order back from the goal and read each node's parents off its DP."""
        parents_of: dict[str, frozenset[str]] = {}
        state = goal
        while state:
            previous, placed = came[state]
            table = tables[placed]
            pick = int(table.choice[project(previous, placed)])
            if pick < 0:  # pragma: no cover - the search never relaxes through -inf
                raise RuntimeError(f"no admissible parent set for {placed!r}; this is a bug")
            parents_of[placed] = frozenset(
                name for j, name in enumerate(table.candidates) if pick >> j & 1
            )
            state = previous
        return parents_of
