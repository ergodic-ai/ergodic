# SPDX-License-Identifier: Apache-2.0
"""Tests for MMPC: the skeleton screen and its composition with the searches."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import DomainKnowledge, tabular
from ergodic.data import TabularData
from ergodic.discovery import PC, DSeparationOracle, Linear
from ergodic.discovery.algorithms._mmpc import MMPC
from ergodic.graph import GraphKind, dag


def _pairs(graph_edges: frozenset) -> set[frozenset[str]]:  # type: ignore[type-arg]
    return {frozenset((e.u, e.v)) for e in graph_edges}


def _chain_data(n: int = 2000, seed: int = 0) -> TabularData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = x + 0.5 * rng.normal(size=n)
    y = m + 0.5 * rng.normal(size=n)
    return tabular(pd.DataFrame({"X": x, "M": m, "Y": y}))


def _five_node_data(n: int = 3000, seed: int = 1) -> tuple[TabularData, set[frozenset[str]]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a + 0.5 * rng.normal(size=n)
    c = a + 0.5 * rng.normal(size=n)
    d = b + c + 0.5 * rng.normal(size=n)
    e = d + 0.5 * rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e}))
    skeleton = {
        frozenset(("A", "B")),
        frozenset(("A", "C")),
        frozenset(("B", "D")),
        frozenset(("C", "D")),
        frozenset(("D", "E")),
    }
    return data, skeleton


# -- exactness and data ----------------------------------------------------------


def test_mmpc_recovers_the_true_skeleton_under_a_perfect_oracle() -> None:
    graph = dag(["X -> Z", "Y -> Z", "Z -> W"])
    result = MMPC(test=DSeparationOracle(graph)).fit()
    assert _pairs(result.graph.edges) == {
        frozenset(("X", "Z")),
        frozenset(("Y", "Z")),
        frozenset(("Z", "W")),
    }
    assert all(e.is_undirected for e in result.graph.edges)
    assert result.graph.kind is GraphKind.MIXED


def test_the_backward_sweep_drops_the_indirect_neighbour() -> None:
    # W enters CPC(X) on its marginal association with X through the chain
    # X -> Z -> W, and the backward sweep removes it once Z is in.
    graph = dag(["X -> Z", "Z -> W"])
    result = MMPC(test=DSeparationOracle(graph)).fit()
    assert _pairs(result.graph.edges) == {frozenset(("X", "Z")), frozenset(("Z", "W"))}
    assert result.diagnostics["n_backward_removed"] >= 1.0


def test_mmpc_recovers_a_five_node_skeleton_from_data() -> None:
    data, skeleton = _five_node_data()
    result = MMPC(test="linear").fit(data)
    assert _pairs(result.graph.edges) == skeleton
    assert result.method == "mmpc"
    assert result.test == "linear"
    assert result.sepsets is None


# -- knowledge --------------------------------------------------------------------


def test_an_adjacency_ban_prunes_the_pair_and_its_tests() -> None:
    data = _chain_data()
    knowledge = DomainKnowledge().forbid_adjacency("X", "M")
    banned = MMPC().fit(data, knowledge=knowledge)
    assert frozenset(("X", "M")) not in _pairs(banned.graph.edges)
    free = MMPC().fit(data)
    assert banned.diagnostics["n_ci_tests"] < free.diagnostics["n_ci_tests"]


def test_protected_pairs_survive_without_being_tested() -> None:
    # A and B are independent in the truth; required and confounded pairs are
    # forced into the skeleton anyway, and adjacency-level validation passes.
    graph = dag(["A -> C", "B -> C"])
    required = MMPC(test=DSeparationOracle(graph)).fit(
        knowledge=DomainKnowledge().require("A", "B")
    )
    assert frozenset(("A", "B")) in _pairs(required.graph.edges)
    confounded = MMPC(test=DSeparationOracle(graph)).fit(
        knowledge=DomainKnowledge().confound("A", "B")
    )
    assert frozenset(("A", "B")) in _pairs(confounded.graph.edges)


def test_a_required_pair_over_a_missing_variable_raises_at_validation() -> None:
    data = _chain_data(n=400)
    knowledge = DomainKnowledge().require("X", "Q")
    with pytest.raises(ValueError, match="not adjacent"):
        MMPC().fit(data, knowledge=knowledge)


def test_a_skeleton_restriction_is_respected() -> None:
    data, _ = _five_node_data()
    support = dag(["A -> B", "B -> D", "D -> E"])
    knowledge = DomainKnowledge().restrict_to(support)
    result = MMPC().fit(data, knowledge=knowledge)
    allowed = {frozenset(("A", "B")), frozenset(("B", "D")), frozenset(("D", "E"))}
    assert _pairs(result.graph.edges) <= allowed


# -- composition -------------------------------------------------------------------


def test_the_screen_feeds_restrict_to_and_pc_keeps_the_skeleton_with_fewer_tests() -> None:
    # PC inside the support spends fewer tests and lands on the same
    # adjacencies. Orientation can be weaker: a pruned pair is never tested,
    # so it records no separating set and cannot witness a v-structure.
    data, _ = _five_node_data()
    screen = MMPC(test=Linear(data)).fit()
    support = DomainKnowledge().restrict_to(screen.graph)

    direct_oracle = Linear(data)
    direct = PC(test=direct_oracle).fit()
    restricted_oracle = Linear(data)
    restricted = PC(test=restricted_oracle).fit(knowledge=support)

    assert _pairs(restricted.graph.edges) == _pairs(direct.graph.edges)
    assert restricted_oracle.diagnostics()["n_ci_tests"] < direct_oracle.diagnostics()["n_ci_tests"]


def test_the_screen_feeds_a_score_search_and_the_optimum_survives() -> None:
    # The MMHC shape: a constraint screen feeding a score search. Exact
    # search inside the support equals the direct optimum whenever the
    # optimum's edges lie inside the screen, and a score search needs no
    # separating sets, so nothing is lost to pruning.
    from ergodic.discovery import AStar

    data, _ = _five_node_data()
    screen = MMPC(test=Linear(data)).fit()
    support = DomainKnowledge().restrict_to(screen.graph)

    direct = AStar(score="bic").fit(data)
    restricted = AStar(score="bic").fit(data, knowledge=support)
    from ergodic.graph import cpdag_of_dag
    print("PROBE mmpc", sorted(map(str, direct.graph.edges)), sorted(map(str, restricted.graph.edges)))
    assert cpdag_of_dag(restricted.graph) == cpdag_of_dag(direct.graph)


# -- resolution and diagnostics ------------------------------------------------------


def test_resolution_rules_mirror_pc() -> None:
    data = _chain_data(n=300)
    with pytest.raises(ValueError, match="unknown test"):
        MMPC(test="nope").fit(data)
    with pytest.raises(ValueError, match="no data"):
        MMPC(test=Linear()).fit()
    with pytest.raises(ValueError, match="alpha"):
        MMPC(alpha=0.0)
    with pytest.raises(ValueError, match="max_cond_size"):
        MMPC(max_cond_size=-2)


def test_diagnostics_count_the_phases() -> None:
    data, skeleton = _five_node_data()
    result = MMPC().fit(data)
    d = result.diagnostics
    assert d["n_edges"] == float(len(skeleton))
    assert d["n_forward_added"] >= d["n_edges"]
    assert d["n_backward_removed"] >= 0.0
    assert result.alpha == 0.05
    assert result.n == data.n_rows
