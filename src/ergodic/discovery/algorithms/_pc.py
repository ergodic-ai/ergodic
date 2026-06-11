# SPDX-License-Identifier: Apache-2.0
"""PC-stable: the constraint-based search for a CPDAG.

PC asks a conditional-independence oracle which adjacencies survive, then
orients what the independences compel. The skeleton phase is the stable
variant (Colombo and Maathuis 2014): at each conditioning size it snapshots
the adjacencies and tests every ordered pair against subsets of the
snapshot neighbours, so the result does not depend on the order edges are
visited. Orientation finds the v-structures from the recorded separating
sets and closes under Meek's rules.

The collider step comes in three rules. ``standard`` orients an unshielded
triple x - z - y as a v-structure when z is missing from the recorded
separating set. ``conservative`` and ``majority`` (Ramsey, Zhang, and
Spirtes 2006; Colombo and Maathuis 2014, section 4.2) re-test the pair
against every subset of either endpoint's neighbourhood and vote: a triple
orients only when z sits in none of the separating sets found
(conservative) or in fewer than half (majority). A split vote (or a tie,
or no separating set at all) marks the triple ambiguous, which blocks the
v-structure and gates Meek's R1 and R3. Under a perfect oracle on a
faithful world every separating set agrees, so no triple is ambiguous and
the three rules coincide.

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
        colliders: The v-structure rule: ``"standard"`` reads the recorded
            sepsets, ``"conservative"`` and ``"majority"`` re-test every
            unshielded triple and leave the ambiguous ones unoriented (see
            the module docstring).

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
        colliders: str = "standard",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_cond_size is not None and max_cond_size < 0:
            raise ValueError(f"max_cond_size must be None or >= 0, got {max_cond_size}")
        if colliders not in ("standard", "conservative", "majority"):
            raise ValueError(
                f"colliders must be 'standard', 'conservative', or 'majority', got {colliders!r}"
            )
        self.test = test
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.colliders = colliders

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
        collider_triples: frozenset[tuple[str, str, str]] | None = None
        ambiguous_triples: frozenset[tuple[str, str, str]] = frozenset()
        if self.colliders != "standard":
            collider_triples, ambiguous_triples = _vote_triples(
                oracle,
                nodes,
                adj,
                sepsets,
                rule=self.colliders,
                alpha=self.alpha,
                max_cond_size=self.max_cond_size,
            )
        graph, n_v_structures = _orient(
            nodes, adj, sepsets, knowledge, colliders=collider_triples, ambiguous=ambiguous_triples
        )
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
        diagnostics["n_ambiguous_triples"] = float(len(ambiguous_triples))
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
    *,
    colliders: frozenset[tuple[str, str, str]] | None = None,
    ambiguous: frozenset[tuple[str, str, str]] = frozenset(),
) -> tuple[MixedGraph, int]:
    """Orient the skeleton: knowledge first, then v-structures, then Meek.

    ``colliders`` is ``None`` on the standard path (orient every unshielded
    triple whose recorded sepset misses the middle node) and the voted set
    under the conservative or majority rule, where only decided colliders
    orient. ``ambiguous`` triples gate Meek's R1 and R3.
    """
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
            if colliders is None:
                separator = sepsets.get(frozenset((x, y)))
                if separator is None or z in separator:
                    continue
            elif (x, z, y) not in colliders:
                continue  # voted non-collider or ambiguous
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
    return meek_closure(graph, forbidden=forbidden, ambiguous=ambiguous), n_v_structures


def _classify_triple(rule: str, z: str, separating: list[frozenset[str]]) -> str:
    """One vote: ``"collider"``, ``"noncollider"``, or ``"ambiguous"``.

    No separating set at all is ambiguous under both rules (the skeleton
    removed the pair but the re-test cannot reproduce a separation).
    Conservative needs unanimity: z in no separating set is a collider, z in
    all of them a non-collider, anything between is ambiguous. Majority
    reads the share of separating sets containing z: under half collider,
    over half non-collider, a tie at exactly half ambiguous.
    """
    if not separating:
        return "ambiguous"
    share = sum(1 for s in separating if z in s) / len(separating)
    if rule == "conservative":
        if share == 0.0:
            return "collider"
        if share == 1.0:
            return "noncollider"
        return "ambiguous"
    if share < 0.5:
        return "collider"
    if share > 0.5:
        return "noncollider"
    return "ambiguous"


def _vote_triples(
    oracle: CITest,
    nodes: list[str],
    adj: dict[str, set[str]],
    sepsets: dict[frozenset[str], frozenset[str]],
    *,
    rule: str,
    alpha: float,
    max_cond_size: int | None,
) -> tuple[frozenset[tuple[str, str, str]], frozenset[tuple[str, str, str]]]:
    """Vote every unshielded triple: the decided colliders and the ambiguous ones.

    The conservative and majority rules (Ramsey, Zhang, and Spirtes 2006;
    Colombo and Maathuis 2014, section 4.2) re-test each unshielded triple
    (x, z, y) against every subset of adj(x) minus y and of adj(y) minus x,
    capped at ``max_cond_size`` like the skeleton. A conditioning set both
    neighbourhoods produce counts once (the Tetrad reading; pcalg counts the
    shared set twice, which can move a majority vote). The sepsets are
    rewritten to witness each decision, the way pcalg's checkTriple does: a
    voted collider's recorded sepset must exclude z, a voted non-collider's
    must include it. A triple whose pair has no recorded sepset (knowledge
    pruned the pair, so it was never tested) is skipped, exactly as the
    standard v-structure search skips it.
    """
    collider_triples: set[tuple[str, str, str]] = set()
    ambiguous_triples: set[tuple[str, str, str]] = set()
    for z in nodes:
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue  # shielded
            pair = frozenset((x, y))
            if pair not in sepsets:
                continue  # pruned by knowledge, never tested: orient nothing
            pool: set[frozenset[str]] = set()
            for side in (adj[x] - {y}, adj[y] - {x}):
                base = sorted(side)
                top = len(base) if max_cond_size is None else min(len(base), max_cond_size)
                for size in range(top + 1):
                    pool.update(frozenset(s) for s in combinations(base, size))
            ordered = sorted(pool, key=lambda s: (len(s), tuple(sorted(s))))
            separating = [s for s in ordered if oracle.independent(x, y, s, alpha=alpha)]
            verdict = _classify_triple(rule, z, separating)
            if verdict == "collider":
                collider_triples.add((x, z, y))
                if z in sepsets[pair]:
                    sepsets[pair] = next(s for s in separating if z not in s)
            elif verdict == "noncollider":
                if z not in sepsets[pair]:
                    sepsets[pair] = next(s for s in separating if z in s)
            else:
                ambiguous_triples.add((x, z, y))
    return frozenset(collider_triples), frozenset(ambiguous_triples)
