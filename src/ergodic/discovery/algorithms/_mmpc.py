# SPDX-License-Identifier: Apache-2.0
"""MMPC: the max-min parents-and-children skeleton screen.

MMPC (Tsamardinos, Brown, and Aliferis 2006, "The max-min hill-climbing
Bayesian network structure learning algorithm", Machine Learning 65) finds,
for each target ``t``, the set CPC(t) of variables that no subset of CPC(t)
can separate from ``t``. The forward phase admits candidates by the max-min
heuristic: each candidate's association is its worst case (largest p-value)
over conditioning subsets of the current CPC, and the candidate with the
strongest worst case enters while it stays dependent under every subset
tried. The backward phase then makes one sweep over the admitted members,
dropping any that some subset of the rest renders independent. That single
sweep is the classic formulation; removals shrink the pool the later members
are tested against, but no second pass runs.

An adjacency is kept only when it holds from both sides (the AND rule):
``x`` in CPC(t) and ``t`` in CPC(x). The output is an undirected skeleton, a
graph to hand to :meth:`~ergodic.knowledge.DomainKnowledge.restrict_to`, not
an equivalence class.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

from ...graph import Edge, GraphKind, Mark, MixedGraph
from .._registry import resolve_test
from .._result import DiscoveryResult
from ._skeleton import _permits_all, _protected_pairs

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ...data import TabularData
    from ...knowledge import DomainKnowledge
    from .._ci import CITest


class MMPC:
    """The max-min parents-and-children screen over a conditional-independence test.

    MMPC is a cheap skeleton screen to run before an expensive search: it
    returns the undirected adjacencies the data supports, and
    ``DomainKnowledge().restrict_to(result.graph)`` turns them into a support
    that PC, GES, or AStar then searches inside, with far fewer tests or
    score calls. Adjacencies are kept under the AND rule (each endpoint must
    sit in the other's parents-children set), the stricter of the two
    symmetry choices, so the screen errs toward sparsity.

    Args:
        test: A :class:`CITest` instance (bound or unbound, including
            :class:`DSeparationOracle`), or a name: ``"auto"`` (pick by the
            data schema), ``"linear"``/``"fisher_z"``, ``"discrete"``/
            ``"g_sq"``, ``"mixed"``/``"dg"``, ``"gcm"``.
        alpha: Significance level for the independence decisions.
        max_cond_size: Cap on the conditioning-set size, ``None`` for no
            cap. Subsets of a growing CPC multiply combinatorially, so the
            default keeps the standard practical cap of 3.

    Knowledge enters in three modes. Prune: a pair forbidden in both
    directions is never a candidate. Protect: required and confounded pairs
    enter every CPC untested, survive the backward sweep, and pass the AND
    rule unconditionally. Validate: the finished skeleton is checked at the
    adjacency level only, because a skeleton has no directions, so
    ``is_consistent`` (which demands required edges be directed) cannot
    apply; orientation constraints (tiers, single-direction forbids, roots,
    sinks) do not constrain an undirected adjacency and are left to the
    search that runs inside the support.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    holds one tail-tail edge per surviving pair, with the ``mixed`` face.
    """

    ingredient: ClassVar[str] = "test"

    def __init__(
        self,
        test: CITest | str = "auto",
        *,
        alpha: float = 0.05,
        max_cond_size: int | None = 3,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_cond_size is not None and max_cond_size < 0:
            raise ValueError(f"max_cond_size must be None or >= 0, got {max_cond_size}")
        self.test = test
        self.alpha = alpha
        self.max_cond_size = max_cond_size

    # -- the run ---------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Run MMPC and return the undirected skeleton with its provenance."""
        oracle = resolve_test(self.test, data, algorithm=type(self).__name__)
        nodes = sorted(oracle.variables)
        node_set = frozenset(nodes)
        permits = _permits_all if knowledge is None else knowledge.permits_edge
        protected = _protected_pairs(knowledge)
        cpc: dict[str, set[str]] = {}
        n_forward_added = 0
        n_backward_removed = 0
        for target in nodes:
            seeded = (
                frozenset(
                    partner for pair in protected if target in pair for partner in pair - {target}
                )
                & node_set
            )
            candidates = [
                x
                for x in nodes
                if x != target and x not in seeded and (permits(x, target) or permits(target, x))
            ]
            members, added, removed = _parents_children(
                oracle, target, candidates, seeded, alpha=self.alpha, cap=self.max_cond_size
            )
            cpc[target] = members
            n_forward_added += added
            n_backward_removed += removed
        pairs = _and_rule(nodes, cpc, protected)
        edges = set()
        for pair in sorted(pairs, key=sorted):
            a, b = sorted(pair)
            edges.add(Edge.make(a, Mark.TAIL, b, Mark.TAIL))
        graph = MixedGraph(nodes=node_set, edges=frozenset(edges), kind=GraphKind.MIXED)
        if knowledge is not None:
            _validate_skeleton(graph, knowledge)
        diagnostics: dict[str, float] = dict(oracle.diagnostics())
        diagnostics["n_edges"] = float(len(edges))
        diagnostics["n_forward_added"] = float(n_forward_added)
        diagnostics["n_backward_removed"] = float(n_backward_removed)
        return DiscoveryResult(
            graph=graph,
            method="mmpc",
            test=oracle.name,
            alpha=self.alpha,
            n=oracle.n,
            diagnostics=diagnostics,
            sepsets=None,
        )


def _worst_p(
    oracle: CITest,
    x: str,
    target: str,
    pool: Iterable[str],
    *,
    alpha: float,
    cap: int | None,
) -> float:
    """The worst-case (largest) p-value of ``x`` vs ``target`` over subsets of ``pool``.

    Subsets run from the empty set up to ``cap`` members (all of ``pool``
    when ``cap`` is ``None``). The enumeration stops at the first p-value
    past ``alpha``: the maximum can only grow, and a single subset past the
    threshold already disqualifies the candidate (forward) or separates the
    member (backward), so the exact maximum never matters beyond ``alpha``.
    """
    members = sorted(pool)
    top = len(members) if cap is None else min(cap, len(members))
    worst = 0.0
    for size in range(top + 1):
        for subset in combinations(members, size):
            p = oracle.test(x, target, subset).p_value
            if p > worst:
                worst = p
                if worst > alpha:
                    return worst
    return worst


def _parents_children(
    oracle: CITest,
    target: str,
    candidates: list[str],
    seeded: frozenset[str],
    *,
    alpha: float,
    cap: int | None,
) -> tuple[set[str], int, int]:
    """CPC(target): max-min forward admission, then one backward sweep.

    ``seeded`` holds the protected partners: in the CPC from the start,
    never tested, never removed. Each forward round recomputes every
    remaining candidate's worst-case p-value against the current CPC (the
    oracle caches, so revisited subsets cost a lookup) and admits the
    smallest, ties broken by name, while it stays at or below ``alpha``.
    The backward phase is the classic single sweep in sorted order: a
    member leaves when some subset of the current CPC without it tests
    independent, and later members see the shrunken set.
    """
    cpc: set[str] = set(seeded)
    pool = sorted(candidates)
    n_added = 0
    while pool:
        best_p = float("inf")
        best_x: str | None = None
        for x in pool:
            worst = _worst_p(oracle, x, target, cpc, alpha=alpha, cap=cap)
            if worst < best_p:
                best_p = worst
                best_x = x
        if best_x is None or best_p > alpha:
            break
        cpc.add(best_x)
        pool.remove(best_x)
        n_added += 1
    n_removed = 0
    for y in sorted(cpc - seeded):
        if _worst_p(oracle, y, target, cpc - {y}, alpha=alpha, cap=cap) > alpha:
            cpc.discard(y)
            n_removed += 1
    return cpc, n_added, n_removed


def _and_rule(
    nodes: list[str],
    cpc: Mapping[str, set[str]],
    protected: frozenset[frozenset[str]],
) -> frozenset[frozenset[str]]:
    """The surviving pairs: mutual CPC membership, or protection."""
    pairs: set[frozenset[str]] = set()
    for x, y in combinations(nodes, 2):
        pair = frozenset((x, y))
        if pair in protected or (y in cpc[x] and x in cpc[y]):
            pairs.add(pair)
    return frozenset(pairs)


def _validate_skeleton(graph: MixedGraph, knowledge: DomainKnowledge) -> None:
    """Check the skeleton against the knowledge at the adjacency level.

    A skeleton has no directions, so ``knowledge.is_consistent`` (which
    demands required edges be directed and confounded pairs be bidirected)
    cannot apply. What an undirected graph can honor is adjacency: required
    and confounded pairs must be adjacent, a pair forbidden in both
    directions must not be, and under a skeleton restriction no edge may sit
    outside the allow-list.
    """
    problems: list[str] = []
    for u, v in sorted(knowledge.required):
        if graph.get_edge(u, v) is None:
            problems.append(f"required pair {u}-{v} is not adjacent")
    for pair in sorted(knowledge.confounded, key=sorted):
        a, b = sorted(pair)
        if graph.get_edge(a, b) is None:
            problems.append(f"confounded pair {a}-{b} is not adjacent")
    reported: set[frozenset[str]] = set()
    for u, v in sorted(knowledge.forbidden):
        if (v, u) in knowledge.forbidden and graph.get_edge(u, v) is not None:
            pair = frozenset((u, v))
            if pair not in reported:
                reported.add(pair)
                a, b = sorted((u, v))
                problems.append(f"present edge between {a} and {b}, forbidden in both directions")
    if knowledge.allowed_adjacencies is not None:
        for edge in graph.edge_list():
            if frozenset((edge.u, edge.v)) not in knowledge.allowed_adjacencies:
                problems.append(
                    f"present edge between {edge.u} and {edge.v}, outside the skeleton restriction"
                )
    if problems:
        raise ValueError(
            "MMPC produced a skeleton that violates the domain knowledge; the data and "
            "the constraints disagree: " + "; ".join(problems)
        )
