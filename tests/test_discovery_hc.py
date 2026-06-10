# SPDX-License-Identifier: Apache-2.0
"""Tests for hill climbing with a tabu list."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import tabular
from ergodic.discovery.algorithms._astar import AStar
from ergodic.discovery.algorithms._hc import HillClimb
from ergodic.discovery.scores._bic import BIC
from ergodic.graph import cpdag_of_dag, dag
from ergodic.knowledge import DomainKnowledge

if TYPE_CHECKING:
    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph

# -- simulated structures ------------------------------------------------------


def _chain(n: int = 2000, seed: int = 21) -> tuple[TabularData, MixedGraph]:
    """a -> b -> c -> d, linear Gaussian with strong coefficients."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = 0.8 * b + rng.normal(size=n)
    d = 1.1 * c + rng.normal(size=n)
    truth = dag(["a -> b", "b -> c", "c -> d"])
    return tabular({"a": a, "b": b, "c": c, "d": d}), truth


def _diamond(n: int = 1500, seed: int = 5) -> tuple[TabularData, MixedGraph]:
    """a -> b, a -> c, b -> d, c -> d, linear Gaussian with strong coefficients."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 1.1 * c + rng.normal(size=n)
    truth = dag(["a -> b", "a -> c", "b -> d", "c -> d"])
    return tabular({"a": a, "b": b, "c": c, "d": d}), truth


def _five_node(n: int = 5000, seed: int = 6) -> tuple[TabularData, MixedGraph]:
    """The four-node diamond plus d -> e."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 1.1 * c + rng.normal(size=n)
    e = 0.8 * d + rng.normal(size=n)
    truth = dag(["a -> b", "a -> c", "b -> d", "c -> d", "d -> e"])
    return tabular({"a": a, "b": b, "c": c, "d": d, "e": e}), truth


def _independent(n: int = 1000, seed: int = 16) -> TabularData:
    """Four mutually independent Gaussians."""
    rng = np.random.default_rng(seed)
    return tabular({name: rng.normal(size=n) for name in ("w", "x", "y", "z")})


def _skeleton(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((edge.u, edge.v)) for edge in graph.edges}


def _parent_map(graph: MixedGraph) -> dict[str, frozenset[str]]:
    return {node: graph.parents(node) for node in sorted(graph.nodes)}


# -- recovery -------------------------------------------------------------------


def test_hill_climbing_finds_the_exact_optimum_on_an_easy_chain() -> None:
    data, truth = _chain()
    bic = BIC(data)
    climbed = HillClimb(score=bic).fit()
    exact = AStar(score=bic).fit()
    found = bic.graph_score(_parent_map(climbed.graph))
    optimum = bic.graph_score(_parent_map(exact.graph))
    assert math.isclose(found, optimum, rel_tol=1e-9)
    assert cpdag_of_dag(climbed.graph) == cpdag_of_dag(truth)


def test_hill_climbing_returns_the_empty_graph_on_independent_data() -> None:
    data = _independent()
    result = HillClimb("bic").fit(data)
    assert len(result.graph.edges) == 0
    assert result.graph.nodes == frozenset({"w", "x", "y", "z"})
    # With no ring there are no escapes: the first pass finds no improving
    # move, applies nothing, and stops, so the counter stays at zero.
    still = HillClimb("bic", tabu=0).fit(data)
    assert len(still.graph.edges) == 0
    assert still.diagnostics["n_iterations"] == 0.0


def test_the_result_is_a_dag_face() -> None:
    data, _ = _chain(n=400, seed=8)
    result = HillClimb("bic").fit(data)
    assert result.graph.is_dag()
    assert result.method == "hc"
    assert result.score == "bic"
    assert result.test is None
    assert result.alpha is None
    assert result.n == 400
    assert result.sepsets is None


# -- knowledge ------------------------------------------------------------------


def test_a_required_edge_is_seeded_and_kept_without_signal() -> None:
    rng = np.random.default_rng(7)
    n = 800
    w = rng.normal(size=n)  # independent of the rest; only knowledge connects it
    x = rng.normal(size=n)
    y = 1.2 * x + rng.normal(size=n)
    data = tabular({"w": w, "x": x, "y": y})
    knowledge = DomainKnowledge().require("w", "y")
    result = HillClimb("bic").fit(data, knowledge=knowledge)
    edge = result.graph.get_edge("w", "y")
    assert edge is not None
    assert edge.directed_pair() == ("w", "y")


def test_a_required_cycle_raises() -> None:
    data, _ = _chain(n=300, seed=9)
    knowledge = DomainKnowledge().require("a", "b").require("b", "a")
    with pytest.raises(ValueError, match="cycle"):
        HillClimb("bic").fit(data, knowledge=knowledge)


def test_a_forbidden_pair_stays_absent() -> None:
    data, _ = _diamond()
    knowledge = DomainKnowledge().forbid_adjacency("a", "b")
    result = HillClimb("bic").fit(data, knowledge=knowledge)
    assert result.graph.get_edge("a", "b") is None


def test_tiers_are_respected_on_a_chain() -> None:
    data, truth = _chain()
    knowledge = DomainKnowledge().with_tiers([["a"], ["b"], ["c"], ["d"]])
    result = HillClimb("bic").fit(data, knowledge=knowledge)
    tier = {"a": 0, "b": 1, "c": 2, "d": 3}
    for edge in result.graph.edges:
        pair = edge.directed_pair()
        assert pair is not None
        cause, effect = pair
        assert tier[cause] <= tier[effect]
    assert _skeleton(result.graph) == _skeleton(truth)


def test_restrict_to_a_skeleton_confines_the_adjacencies() -> None:
    data, _ = _diamond()
    allowed = [("a", "b"), ("b", "d"), ("c", "d")]
    knowledge = DomainKnowledge().restrict_to(allowed)
    result = HillClimb("bic").fit(data, knowledge=knowledge)
    support = {frozenset(pair) for pair in allowed}
    assert _skeleton(result.graph) <= support
    assert result.graph.get_edge("a", "c") is None  # strongly correlated, but off-support


# -- the controls ----------------------------------------------------------------


def test_the_tabu_ring_and_max_iter_are_honored() -> None:
    chain_data, _ = _chain()
    greedy = HillClimb("bic", tabu=0).fit(chain_data)  # no ring, no escapes
    assert greedy.graph.is_dag()
    one = HillClimb("bic", max_iter=1).fit(chain_data)
    assert one.diagnostics["n_iterations"] == 1.0
    assert len(one.graph.edges) == 1  # exactly the first applied add
    five_data, truth = _five_node()
    result = HillClimb().fit(five_data)  # default tabu and max_iter
    assert cpdag_of_dag(result.graph) == cpdag_of_dag(truth)


def test_max_parents_caps_the_in_degree() -> None:
    data, _ = _diamond()
    result = HillClimb("bic", max_parents=1).fit(data)
    assert result.graph.is_dag()
    assert all(len(result.graph.parents(node)) <= 1 for node in result.graph.nodes)


# -- resolution and guards ---------------------------------------------------------


def test_hc_data_resolution_rules() -> None:
    data, _ = _chain(n=300, seed=11)
    other, _ = _chain(n=301, seed=12)
    with pytest.raises(ValueError, match="data"):
        HillClimb("bic").fit()  # unbound score and nothing to bind
    bound = BIC(data)
    with pytest.raises(ValueError, match="different"):
        HillClimb(score=bound).fit(other)  # bound score plus another dataset
    resolved = HillClimb().fit(data)  # score="auto" picks BIC on continuous data
    assert resolved.score == "bic"
    assert resolved.n == 300


def test_hc_rejects_bad_arguments() -> None:
    with pytest.raises(TypeError, match="Score or a string"):
        HillClimb(3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tabu"):
        HillClimb(tabu=-1)
    with pytest.raises(ValueError, match="max_iter"):
        HillClimb(max_iter=0)
    with pytest.raises(ValueError, match="max_parents"):
        HillClimb(max_parents=-1)


def test_hc_diagnostics_have_the_run_counters() -> None:
    data, _ = _chain()
    result = HillClimb("bic").fit(data)
    counters = result.diagnostics
    for key in ("n_iterations", "n_adds", "n_removes", "n_reverses", "n_edges"):
        assert key in counters
    assert counters["n_iterations"] == (
        counters["n_adds"] + counters["n_removes"] + counters["n_reverses"]
    )
    assert counters["n_adds"] >= 3.0  # the chain needs at least its three edges
    assert counters["n_edges"] == float(len(result.graph.edges))
    assert counters["n_score_evals"] >= 1.0  # the score's own counters ride along
