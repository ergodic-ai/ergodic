# SPDX-License-Identifier: Apache-2.0
"""PC-stable: the constraint-based search for a CPDAG.

PC asks a conditional-independence oracle which adjacencies survive, then
orients what the independences compel. The skeleton phase is the stable
variant (Colombo and Maathuis 2014): at each conditioning size it snapshots
the adjacencies and tests every ordered pair against subsets of the
snapshot neighbours, so the result does not depend on the order edges are
visited. Orientation finds the v-structures from the recorded separating
sets and closes under Meek's rules.

Domain knowledge enters in four ways. Prune: a pair forbidden in both
directions never enters the graph and is never tested. Protect: a pair
carrying a required edge is never tested or removed. Orient: required edges
and tier-compelled directions are forced, v-structures skip arrowheads the
knowledge forbids, and the Meek closure never orients against the
knowledge. Validate: the final graph must satisfy the knowledge, or ``fit``
raises.

With a perfect oracle the output is a valid CPDAG. A noisy test (or forced
knowledge orientations) can leave a PDAG outside any equivalence class; the
result then keeps the ``mixed`` face and ``diagnostics["cpdag_valid"]`` is
``0.0``.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

from ...graph import Edge, GraphKind, InvalidGraphError, Mark, MixedGraph
from ...graph._meek import meek_closure
from .._registry import resolve_test
from .._result import DiscoveryResult
from ._skeleton import _complement, _permits_all, stable_skeleton

if TYPE_CHECKING:
    from ...data import TabularData
    from ...knowledge import DomainKnowledge
    from .._ci import CITest


class PC:
    """PC-stable over a conditional-independence test.

    Args:
        test: A :class:`CITest` instance (bound or unbound, including
            :class:`DSeparationOracle`), or a name: ``"auto"`` (pick by the
            data schema), ``"linear"``/``"fisher_z"``, ``"discrete"``/
            ``"g_sq"``, ``"mixed"``/``"dg"``, ``"gcm"``.
        alpha: Significance level for the independence decisions.
        max_cond_size: Cap on the conditioning-set size, ``None`` for no cap.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    carries the CPDAG face when the oriented PDAG is a valid CPDAG and the
    ``mixed`` face otherwise.
    """

    ingredient: ClassVar[str] = "test"

    def __init__(
        self,
        test: CITest | str = "auto",
        *,
        alpha: float = 0.05,
        max_cond_size: int | None = None,
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
        """Run PC-stable and return the discovered graph with its provenance."""
        oracle = resolve_test(self.test, data)
        nodes = sorted(oracle.variables)
        adj, sepsets = stable_skeleton(
            oracle, nodes, knowledge, alpha=self.alpha, max_cond_size=self.max_cond_size
        )
        graph, n_v_structures = _orient(nodes, adj, sepsets, knowledge)
        try:
            final = graph.as_kind(GraphKind.CPDAG)
            cpdag_valid = 1.0
        except InvalidGraphError:
            final = graph.as_kind(GraphKind.MIXED)
            cpdag_valid = 0.0
        if knowledge is not None and not knowledge.is_consistent(final):
            raise ValueError(
                "PC produced a graph that violates the domain knowledge; the data and "
                "the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics: dict[str, float] = dict(oracle.diagnostics())
        diagnostics["n_edges"] = float(len(final.edges))
        diagnostics["n_directed"] = float(sum(1 for e in final.edges if e.is_directed))
        diagnostics["n_undirected"] = float(sum(1 for e in final.edges if e.is_undirected))
        diagnostics["n_v_structures"] = float(n_v_structures)
        diagnostics["cpdag_valid"] = cpdag_valid
        return DiscoveryResult(
            graph=final,
            method="pc",
            test=oracle.name,
            alpha=self.alpha,
            n=oracle.n,
            diagnostics=diagnostics,
            sepsets=sepsets,
        )


def _orient(
    nodes: list[str],
    adj: dict[str, set[str]],
    sepsets: dict[frozenset[str], frozenset[str]],
    knowledge: DomainKnowledge | None,
) -> tuple[MixedGraph, int]:
    """Orient the skeleton: knowledge first, then v-structures, then Meek."""
    edges = frozenset(Edge.make(x, Mark.TAIL, y, Mark.TAIL) for x in nodes for y in adj[x] if x < y)
    graph = MixedGraph(nodes=frozenset(nodes), edges=edges, kind=GraphKind.MIXED)
    permits = _permits_all if knowledge is None else knowledge.permits_edge
    if knowledge is not None:
        for u, v in sorted(knowledge.required):
            if u in adj and v in adj[u]:
                graph = graph.orient(u, v)
    for edge in graph.edge_list():
        if not edge.is_undirected:
            continue
        forward = permits(edge.u, edge.v)
        backward = permits(edge.v, edge.u)
        if forward and not backward:
            graph = graph.orient(edge.u, edge.v)
        elif backward and not forward:
            graph = graph.orient(edge.v, edge.u)
    n_v_structures = 0
    for z in nodes:
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue  # shielded
            separator = sepsets.get(frozenset((x, y)))
            if separator is None or z in separator:
                continue
            if not permits(x, z) or not permits(y, z):
                continue  # a forbidden arrowhead skips the whole v-structure
            edge_xz = graph.get_edge(x, z)
            edge_yz = graph.get_edge(y, z)
            if edge_xz is None or edge_yz is None:  # pragma: no cover - adjacency holds
                continue
            if edge_xz.directed_pair() == (z, x) or edge_yz.directed_pair() == (z, y):
                continue  # already oriented the other way by the knowledge
            graph = graph.orient(x, z).orient(y, z)
            n_v_structures += 1
    forbidden = None if knowledge is None else _complement(permits)
    return meek_closure(graph, forbidden=forbidden), n_v_structures
