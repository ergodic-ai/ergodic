# SPDX-License-Identifier: Apache-2.0
"""Tests for GES, the greedy equivalence search over a decomposable score.

The executable anchor is agreement with the brute-force-verified A* optimum
on a shared warmed score; recovery, knowledge in all four modes, and the
resolution rules follow the PC test patterns. Every run uses a fixed seed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from ergodic.data import tabular
from ergodic.discovery.algorithms._astar import AStar
from ergodic.discovery.algorithms._ges import GES
from ergodic.discovery.scores._bic import BIC
from ergodic.graph import GraphKind, MixedGraph, consistent_extension, cpdag_of_dag, dag
from ergodic.knowledge import DomainKnowledge

if TYPE_CHECKING:
    from ergodic.data import TabularData

# -- helpers ---------------------------------------------------------------------


def _directed(graph: MixedGraph, parent: str, child: str) -> bool:
    edge = graph.get_edge(parent, child)
    return edge is not None and edge.directed_pair() == (parent, child)


def _undirected(graph: MixedGraph, a: str, b: str) -> bool:
    edge = graph.get_edge(a, b)
    return edge is not None and edge.is_undirected


def _skeleton(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((edge.u, edge.v)) for edge in graph.edges}


def _parent_map(graph: MixedGraph) -> dict[str, frozenset[str]]:
    return {node: graph.parents(node) for node in sorted(graph.nodes)}


def _random_dag_data(seed: int, n: int = 2000) -> TabularData:
    """A random 4-node linear SEM: edges in with p=0.5, signed weights in [0.5, 1.5]."""
    rng = np.random.default_rng(seed)
    names = ["a", "b", "c", "d"]
    values: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        column = rng.normal(size=n)
        for parent in names[:i]:
            if rng.random() < 0.5:
                weight = rng.uniform(0.5, 1.5) * (1.0 if rng.random() < 0.5 else -1.0)
                column = column + weight * values[parent]
        values[name] = column
    return tabular(values)


def _chain_data(n: int = 3000, seed: int = 6) -> TabularData:
    """X -> M -> Y, linear Gaussian with strong coefficients."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 1.1 * x + rng.normal(size=n)
    y = 0.9 * m + rng.normal(size=n)
    return tabular({"X": x, "M": m, "Y": y})


# -- exactness against A* on a shared warmed score ---------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_ges_matches_the_exact_optimum_at_four_nodes(seed: int) -> None:
    data = _random_dag_data(seed)
    bic = BIC(data)
    greedy = GES(score=bic).fit()
    exact = AStar(score=bic).fit()
    member = consistent_extension(greedy.graph)
    assert member is not None  # a valid CPDAG always extends
    found = bic.graph_score(_parent_map(member))
    best = bic.graph_score(_parent_map(exact.graph))
    assert math.isclose(found, best, rel_tol=1e-9)
    assert greedy.graph.kind is GraphKind.CPDAG
    assert greedy.graph == cpdag_of_dag(exact.graph)


# -- recovery -----------------------------------------------------------------------


def test_ges_recovers_the_canonical_five_node_class() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.7 * a + rng.normal(size=n)
    d = 0.9 * b + 0.8 * c + rng.normal(size=n)
    e = 0.7 * d + rng.normal(size=n)
    data = tabular({"A": a, "B": b, "C": c, "D": d, "E": e})
    truth = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    result = GES(score="bic").fit(data)
    assert result.graph == cpdag_of_dag(truth)
    assert result.graph.kind is GraphKind.CPDAG
    assert result.diagnostics["cpdag_valid"] == 1.0


def test_ges_returns_the_empty_graph_on_independent_data() -> None:
    rng = np.random.default_rng(4)
    data = tabular({name: rng.normal(size=500) for name in ("x", "y", "z")})
    result = GES("bic").fit(data)
    assert not result.graph.edges
    assert result.graph.nodes == frozenset({"x", "y", "z"})
    assert result.diagnostics["n_inserts"] == 0.0
    assert result.diagnostics["n_deletes"] == 0.0


# -- the score-equivalence warning ---------------------------------------------------


class _LopsidedScore(BIC):
    """A BIC clone that disclaims score equivalence, to exercise the warning."""

    name: ClassVar[str] = "lopsided"
    score_equivalent: ClassVar[bool] = False


def test_ges_warns_when_the_score_is_not_equivalent() -> None:
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    y = 1.3 * x + rng.normal(size=n)
    data = tabular({"x": x, "y": y})
    with pytest.warns(UserWarning, match="equival"):
        result = GES(score=_LopsidedScore(data)).fit()
    assert result.score == "lopsided"  # the search still ran
    assert result.graph.get_edge("x", "y") is not None


# -- knowledge in the four modes ------------------------------------------------------


def test_ges_knowledge_tiers_orient_the_chain_fully() -> None:
    data = _chain_data()
    plain = GES("bic").fit(data)
    assert _undirected(plain.graph, "X", "M")  # the chain class leaves both edges loose
    tiers = DomainKnowledge().with_tiers([["X"], ["M", "Y"]])
    result = GES("bic").fit(data, knowledge=tiers)
    assert _directed(result.graph, "X", "M")  # the tier order forces it
    assert _directed(result.graph, "M", "Y")  # and Meek carries it down the chain
    assert tiers.is_consistent(result.graph)


def test_ges_knowledge_forbidden_pair_never_appears() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.normal(size=n)
    y = 1.2 * x + rng.normal(size=n)
    z = 0.9 * y + rng.normal(size=n)
    data = tabular({"X": x, "Y": y, "Z": z})
    forbidden = DomainKnowledge().forbid_adjacency("X", "Y")
    result = GES("bic").fit(data, knowledge=forbidden)
    assert result.graph.get_edge("X", "Y") is None  # the ban beats the strong signal
    assert result.graph.get_edge("Y", "Z") is not None
    assert forbidden.is_consistent(result.graph)


def test_ges_knowledge_required_edge_survives_without_signal() -> None:
    rng = np.random.default_rng(8)
    data = tabular({"U": rng.normal(size=600), "V": rng.normal(size=600)})
    required = DomainKnowledge().require("U", "V")
    result = GES("bic").fit(data, knowledge=required)
    assert _directed(result.graph, "U", "V")
    assert required.is_consistent(result.graph)
    # a lone forced arrow is no CPDAG, so the face falls back to mixed
    assert result.diagnostics["cpdag_valid"] == 0.0
    assert result.graph.kind is GraphKind.MIXED


def test_ges_knowledge_skeleton_restriction_confines_the_search() -> None:
    rng = np.random.default_rng(9)
    n = 2000
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 1.1 * c + rng.normal(size=n)
    data = tabular({"a": a, "b": b, "c": c, "d": d})
    support = {frozenset(("a", "b")), frozenset(("b", "d"))}
    knowledge = DomainKnowledge().restrict_to([("a", "b"), ("b", "d")])
    result = GES("bic").fit(data, knowledge=knowledge)
    assert _skeleton(result.graph) <= support
    assert _skeleton(result.graph) == support  # both supported pairs carry real signal
    assert knowledge.is_consistent(result.graph)


def test_ges_required_edge_cycle_raises() -> None:
    rng = np.random.default_rng(10)
    data = tabular({name: rng.normal(size=100) for name in ("a", "b", "c")})
    cyclic = DomainKnowledge().require("a", "b").require("b", "c").require("c", "a")
    with pytest.raises(ValueError, match="required-edge cycle"):
        GES("bic").fit(data, knowledge=cyclic)


# -- the degree cap ---------------------------------------------------------------------


def test_ges_max_degree_caps_node_degrees() -> None:
    data = _chain_data()
    result = GES("bic", max_degree=1).fit(data)
    degrees = dict.fromkeys(result.graph.nodes, 0)
    for edge in result.graph.edges:
        degrees[edge.u] += 1
        degrees[edge.v] += 1
    assert max(degrees.values()) <= 1
    assert result.diagnostics["n_edges"] == 1.0  # one insert, then every node is capped


# -- resolution rules and the result record -----------------------------------------------


def test_ges_data_resolution_rules() -> None:
    rng = np.random.default_rng(11)
    data = tabular({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    other = tabular({"x": rng.normal(size=301), "y": rng.normal(size=301)})
    with pytest.raises(ValueError, match="data"):
        GES(score=BIC()).fit()  # unbound score and nothing to bind
    with pytest.raises(ValueError, match="different"):
        GES(score=BIC(data)).fit(other)  # bound score plus another dataset
    result = GES("auto").fit(data)  # continuous schema resolves to the Gaussian BIC
    assert result.score == "bic"
    assert result.n == 300


def test_ges_diagnostics_have_the_run_counters() -> None:
    data = _chain_data(n=1500, seed=12)
    result = GES("bic").fit(data)
    for key in (
        "n_score_evals",
        "cache_hits",
        "n_inserts",
        "n_deletes",
        "n_edges",
        "n_directed",
        "n_undirected",
        "cpdag_valid",
    ):
        assert key in result.diagnostics
    assert result.diagnostics["n_inserts"] >= 2.0  # the chain needs two inserts
    assert result.method == "ges"
    assert result.score == "bic"
    assert result.test is None
    assert result.alpha is None
    assert result.sepsets is None
    assert result.n == 1500


def test_ges_rejects_bad_arguments() -> None:
    with pytest.raises(TypeError, match="Score or a string"):
        GES(score=3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_degree"):
        GES(max_degree=0)
