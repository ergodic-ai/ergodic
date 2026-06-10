# SPDX-License-Identifier: Apache-2.0
"""Tests for FCI: exactness against the canonical PAG, knowledge, and data runs."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ergodic import DomainKnowledge, tabular
from ergodic.data import TabularData
from ergodic.discovery import PC, DSeparationOracle, Linear
from ergodic.discovery.algorithms._fci import FCI
from ergodic.graph import (
    Edge,
    GraphKind,
    Mark,
    MixedGraph,
    admg,
    dag,
    mag_of_dag,
    pag_of_mag,
)

# -- fixtures ------------------------------------------------------------------


def _random_dag_with_latents(rng: np.random.Generator, p: int, n_latent: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        if rng.uniform() < 0.4:
            edges.append((names[i], names[j]))
    latent = list(rng.choice(names, size=n_latent, replace=False))
    return dag(edges, nodes=names, latent=latent)


def _random_admg(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        u = rng.uniform()
        if u < 0.25:
            edges.append(Edge.make(names[i], Mark.TAIL, names[j], Mark.ARROW))
        elif u < 0.50:
            edges.append(Edge.make(names[i], Mark.ARROW, names[j], Mark.ARROW))
    return admg(edges, nodes=names)


def _confounded_collider_data(n: int = 4000, seed: int = 0) -> TabularData:
    """A = L1 + e, B = L2 + e, C = L1 + L2 + e: the canonical PAG is A o-> C <-o B."""
    rng = np.random.default_rng(seed)
    l1 = rng.normal(size=n)
    l2 = rng.normal(size=n)
    a = l1 + 0.5 * rng.normal(size=n)
    b = l2 + 0.5 * rng.normal(size=n)
    c = l1 + l2 + 0.5 * rng.normal(size=n)
    return tabular(pd.DataFrame({"A": a, "B": b, "C": c}))


# -- exactness against the canonical PAG ----------------------------------------


def test_fci_over_the_oracle_equals_the_canonical_pag_on_latent_dags() -> None:
    rng = np.random.default_rng(7)
    for _ in range(40):
        graph = _random_dag_with_latents(rng, 7, 2)
        truth = pag_of_mag(mag_of_dag(graph))
        result = FCI(test=DSeparationOracle(graph)).fit()
        assert result.graph.edges == truth.edges
        assert result.graph.kind is GraphKind.PAG
        assert result.diagnostics["pag_valid"] == 1.0


def test_fci_over_the_oracle_equals_the_canonical_pag_on_admgs() -> None:
    for seed in range(30):
        graph = _random_admg(np.random.default_rng(seed), 7)
        truth = pag_of_mag(mag_of_dag(graph))
        result = FCI(test=DSeparationOracle(graph)).fit()
        assert result.graph.edges == truth.edges


def test_the_possible_d_sep_phase_removes_what_the_skeleton_cannot() -> None:
    # Found by scanning random ADMGs: the only pair separable here needs a
    # conditioning set outside both neighbourhoods, so PC keeps the edge and
    # only the possible-d-sep re-test drops it.
    graph = admg(
        [
            "V0 --> V4",
            "V0 <-> V1",
            "V0 <-> V3",
            "V0 <-> V5",
            "V1 --> V2",
            "V1 --> V6",
            "V2 --> V3",
            "V2 <-> V4",
            "V4 --> V5",
            "V5 --> V6",
        ]
    )
    oracle = DSeparationOracle(graph)
    by_fci = FCI(test=oracle).fit()
    by_pc = PC(test=oracle).fit()
    assert by_fci.diagnostics["n_pds_removed"] == 1.0
    assert by_fci.graph.edges == pag_of_mag(mag_of_dag(graph)).edges
    fci_pairs = {frozenset((e.u, e.v)) for e in by_fci.graph.edges}
    pc_pairs = {frozenset((e.u, e.v)) for e in by_pc.graph.edges}
    assert fci_pairs < pc_pairs  # strictly fewer adjacencies


def test_a_pure_confounder_pair_comes_back_all_circles() -> None:
    graph = dag(["L -> X", "L -> Y"], latent=["L"])
    result = FCI(test=DSeparationOracle(graph)).fit()
    assert {e.glyph() for e in result.graph.edges} == {"X o-o Y"}


def test_the_collider_keeps_its_arrowheads_and_nothing_else() -> None:
    graph = dag(["X -> Y", "Z -> Y"])
    result = FCI(test=DSeparationOracle(graph)).fit()
    assert {e.glyph() for e in result.graph.edges} == {"X o-> Y", "Y <-o Z"}


def test_the_y_structure_orients_a_causal_tail_from_observational_data() -> None:
    graph = dag(["X1 -> X3", "X2 -> X3", "X3 -> X4"])
    result = FCI(test=DSeparationOracle(graph)).fit()
    assert {e.glyph() for e in result.graph.edges} == {
        "X1 o-> X3",
        "X2 o-> X3",
        "X3 --> X4",
    }


def test_the_instrument_graph_is_uninformative_for_fci() -> None:
    graph = dag(["Z -> X", "X -> Y", "L -> X", "L -> Y"], latent=["L"])
    result = FCI(test=DSeparationOracle(graph)).fit()
    assert all(e.u_mark is Mark.CIRCLE and e.v_mark is Mark.CIRCLE for e in result.graph.edges)
    assert len(result.graph.edges) == 3


def test_two_latent_confounders_put_arrowheads_into_the_middle() -> None:
    graph = dag(["L1 -> A", "L1 -> C", "L2 -> B", "L2 -> C"], latent=["L1", "L2"])
    result = FCI(test=DSeparationOracle(graph)).fit()
    assert {e.glyph() for e in result.graph.edges} == {"A o-> C", "B o-> C"}
    assert result.sepsets is not None
    assert result.sepsets[frozenset(("A", "B"))] == frozenset()


# -- data ------------------------------------------------------------------------


def test_fci_recovers_the_confounded_collider_from_data() -> None:
    result = FCI(test="linear", alpha=0.01).fit(_confounded_collider_data())
    assert {e.glyph() for e in result.graph.edges} == {"A o-> C", "B o-> C"}
    assert result.graph.kind is GraphKind.PAG
    assert result.method == "fci"
    assert result.test == "linear"


def test_fci_orients_the_y_structure_tail_from_data() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = x1 + x2 + 0.5 * rng.normal(size=n)
    x4 = x3 + 0.5 * rng.normal(size=n)
    data = tabular(pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4}))
    result = FCI(test="linear", alpha=0.01).fit(data)
    edge = result.graph.get_edge("X3", "X4")
    assert edge is not None and edge.directed_pair() == ("X3", "X4")


# -- knowledge --------------------------------------------------------------------


def test_confounded_knowledge_orients_the_bidirected_edge() -> None:
    graph = dag(["L -> X", "L -> Y"], latent=["L"])
    knowledge = DomainKnowledge().confound("X", "Y")
    result = FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    edge = result.graph.get_edge("X", "Y")
    assert edge is not None and edge.is_bidirected
    # knowledge-seeded marks are not class invariants, so the PAG face is lost
    assert result.graph.kind is GraphKind.MIXED
    assert result.diagnostics["pag_valid"] == 0.0


def test_tier_knowledge_pins_only_the_arrowhead_at_the_later_end() -> None:
    graph = dag(["X -> Y"])
    knowledge = DomainKnowledge().order("X", "Y")
    result = FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    assert {e.glyph() for e in result.graph.edges} == {"X o-> Y"}


def test_required_knowledge_orients_the_full_edge() -> None:
    graph = dag(["X -> Y"])
    knowledge = DomainKnowledge().require("X", "Y")
    result = FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    edge = result.graph.get_edge("X", "Y")
    assert edge is not None and edge.directed_pair() == ("X", "Y")


def test_a_required_edge_against_a_data_collider_raises_at_validation() -> None:
    graph = dag(["V -> U", "W -> U"])
    knowledge = DomainKnowledge().require("U", "V")
    with pytest.raises(ValueError, match="violates the domain knowledge"):
        FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)


def test_an_adjacency_ban_prunes_the_pair_and_its_tests() -> None:
    graph = dag(["L1 -> A", "L1 -> C", "L2 -> B", "L2 -> C"], latent=["L1", "L2"])
    knowledge = DomainKnowledge().forbid_adjacency("A", "B")
    result = FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    assert result.graph.get_edge("A", "B") is None
    banned = FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    free = FCI(test=DSeparationOracle(graph)).fit()
    assert banned.diagnostics["n_ci_tests"] < free.diagnostics["n_ci_tests"]


def test_a_skeleton_restriction_confines_the_adjacencies() -> None:
    graph = dag(["X -> Y", "Y -> Z"])
    support = MixedGraph(
        nodes=frozenset({"X", "Y", "Z"}),
        edges=frozenset({Edge.make("X", Mark.TAIL, "Y", Mark.TAIL)}),
        kind=GraphKind.MIXED,
    )
    knowledge = DomainKnowledge().restrict_to(support)
    result = FCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    assert {frozenset((e.u, e.v)) for e in result.graph.edges} <= {frozenset(("X", "Y"))}


# -- the shared oracle and the resolution rules -----------------------------------


def test_one_warmed_oracle_serves_pc_then_fci() -> None:
    linear = Linear(_confounded_collider_data())
    PC(test=linear, alpha=0.01).fit()
    hits_after_pc = linear.diagnostics()["cache_hits"]
    result = FCI(test=linear, alpha=0.01).fit()
    assert linear.diagnostics()["cache_hits"] > hits_after_pc
    assert result.n == linear.n


def test_resolution_rules_mirror_pc() -> None:
    data = _confounded_collider_data(n=200)
    with pytest.raises(ValueError, match="unknown test"):
        FCI(test="nope").fit(data)
    with pytest.raises(ValueError, match="no data"):
        FCI(test=Linear()).fit()
    with pytest.raises(ValueError, match="alpha"):
        FCI(alpha=1.5)
    with pytest.raises(ValueError, match="max_cond_size"):
        FCI(max_cond_size=-1)


def test_diagnostics_carry_the_mark_census() -> None:
    graph = dag(["X1 -> X3", "X2 -> X3", "X3 -> X4"])
    result = FCI(test=DSeparationOracle(graph)).fit()
    d = result.diagnostics
    assert d["n_edges"] == 3.0
    assert d["n_v_structures"] == 1.0
    assert d["n_arrowheads"] == 3.0
    assert d["n_tails"] == 1.0
    assert d["n_circles"] == 2.0
    assert d["n_pds_removed"] == 0.0
