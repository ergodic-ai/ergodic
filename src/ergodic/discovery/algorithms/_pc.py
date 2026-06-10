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
from typing import TYPE_CHECKING, cast

from ...graph import Edge, GraphKind, InvalidGraphError, Mark, MixedGraph
from ...graph._meek import meek_closure
from .._ci import CITest
from .._result import DiscoveryResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...data import TabularData
    from ...knowledge import DomainKnowledge

_TEST_ALIASES: dict[str, tuple[str, str]] = {
    "linear": ("ergodic.discovery.ci._linear", "Linear"),
    "fisher_z": ("ergodic.discovery.ci._linear", "Linear"),
    "discrete": ("ergodic.discovery.ci._discrete", "Discrete"),
    "g_sq": ("ergodic.discovery.ci._discrete", "Discrete"),
    "gsq": ("ergodic.discovery.ci._discrete", "Discrete"),
    "mixed": ("ergodic.discovery.ci._mixed", "Mixed"),
    "dg": ("ergodic.discovery.ci._mixed", "Mixed"),
    "gcm": ("ergodic.discovery.ci._gcm", "GCM"),
}


def _auto_key(data: TabularData) -> str:
    """Pick a test family from the column schema."""
    kinds = [data.schema[name] for name in data.variables]
    if all(c.is_continuous for c in kinds):
        return "linear"
    if all(c.is_discrete for c in kinds):
        return "discrete"
    return "mixed"


def _test_class(key: str) -> type[CITest]:
    """Resolve a test name to its class, importing the module only when asked."""
    from importlib import import_module

    try:
        module_name, attr = _TEST_ALIASES[key]
    except KeyError:
        names = ", ".join(sorted(_TEST_ALIASES))
        raise ValueError(
            f"unknown test {key!r}; use 'auto', one of: {names}, or pass a CITest instance"
        ) from None
    return cast("type[CITest]", getattr(import_module(module_name), attr))


def _resolve_test(spec: CITest | str, data: TabularData | None) -> CITest:
    """Resolve the test spec and the data into one bound oracle.

    Data comes from exactly one place: an unbound spec needs ``data`` at fit,
    a bound test carries its own and must not be handed a different dataset.
    """
    if isinstance(spec, CITest):
        if spec.is_bound:
            if data is not None and data is not spec.data:
                raise ValueError(
                    f"{type(spec).__name__} is already bound to its own data; "
                    "omit data at fit, or pass an unbound test to bind here"
                )
            return spec
        if data is None:
            raise ValueError(
                f"{type(spec).__name__} is unbound and fit got no data; "
                "construct the test with data or pass data to fit"
            )
        return spec.bind(data)
    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"test {spec!r} needs data to bind at fit; none was given")
        key = spec.lower()
        if key == "auto":
            key = _auto_key(data)
        return _test_class(key)(data)
    raise TypeError(f"test must be a CITest or a string, got {type(spec).__name__}")


def _permits_all(cause: str, effect: str) -> bool:
    """The open default when no knowledge is given."""
    return True


def _complement(permits: Callable[[str, str], bool]) -> Callable[[str, str], bool]:
    """Turn a permits-edge projection into the forbidden callback Meek expects."""

    def forbids(u: str, v: str) -> bool:
        return not permits(u, v)

    return forbids


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
        oracle = _resolve_test(self.test, data)
        nodes = sorted(oracle.variables)
        adj, sepsets = self._skeleton(oracle, nodes, knowledge)
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

    def _skeleton(
        self, oracle: CITest, nodes: list[str], knowledge: DomainKnowledge | None
    ) -> tuple[dict[str, set[str]], dict[frozenset[str], frozenset[str]]]:
        """PC-stable adjacency search: the surviving adjacencies and the sepsets."""
        protected: frozenset[frozenset[str]] = frozenset()
        permits = _permits_all if knowledge is None else knowledge.permits_edge
        if knowledge is not None:
            protected = frozenset(frozenset(pair) for pair in knowledge.required)
        adj: dict[str, set[str]] = {n: set() for n in nodes}
        for x, y in combinations(nodes, 2):
            if not permits(x, y) and not permits(y, x):
                continue  # prune: forbidden in both directions, never tested
            adj[x].add(y)
            adj[y].add(x)
        sepsets: dict[frozenset[str], frozenset[str]] = {}
        level = 0
        while self.max_cond_size is None or level <= self.max_cond_size:
            snapshot = {n: frozenset(adj[n]) for n in nodes}
            if all(len(snapshot[n]) <= level for n in nodes):
                break
            for x in nodes:
                for y in sorted(snapshot[x]):
                    if y not in adj[x]:
                        continue  # removed earlier at this level
                    if frozenset((x, y)) in protected:
                        continue  # protect: a required edge is never tested
                    candidates = sorted(snapshot[x] - {y})
                    if len(candidates) < level:
                        continue
                    for subset in combinations(candidates, level):
                        if oracle.independent(x, y, subset, alpha=self.alpha):
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sepsets[frozenset((x, y))] = frozenset(subset)
                            break
            level += 1
        return adj, sepsets


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
