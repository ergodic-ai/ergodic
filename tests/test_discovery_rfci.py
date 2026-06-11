# SPDX-License-Identifier: Apache-2.0
"""Tests for RFCI: Theorem 3.2 soundness, the FCI contrast, and the extra-test stages."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ergodic import DomainKnowledge, tabular
from ergodic.data import TabularData
from ergodic.discovery import DSeparationOracle
from ergodic.discovery.algorithms._fci import FCI
from ergodic.discovery.algorithms._rfci import RFCI
from ergodic.graph import Edge, GraphKind, Mark, MixedGraph, admg, dag, mag_of_dag

# -- fixtures (the world builders the FCI tests use) ------------------------------


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


def _random_dag_with_margin(
    rng: np.random.Generator, p: int, n_latent: int, n_selection: int
) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        if rng.uniform() < 0.4:
            edges.append((names[i], names[j]))
    hidden = list(rng.choice(names, size=n_latent + n_selection, replace=False))
    return dag(edges, nodes=names, latent=hidden[:n_latent], selection=hidden[n_latent:])


def _confounded_collider_data(n: int = 4000, seed: int = 0) -> TabularData:
    """A = L1 + e, B = L2 + e, C = L1 + L2 + e: the canonical PAG is A o-> C <-o B."""
    rng = np.random.default_rng(seed)
    l1 = rng.normal(size=n)
    l2 = rng.normal(size=n)
    a = l1 + 0.5 * rng.normal(size=n)
    b = l2 + 0.5 * rng.normal(size=n)
    c = l1 + l2 + 0.5 * rng.normal(size=n)
    return tabular(pd.DataFrame({"A": a, "B": b, "C": c}))


def _pairs(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((e.u, e.v)) for e in graph.edges}


def _assert_sound_marks(found: MixedGraph, truth: MixedGraph) -> None:
    """Theorem 3.2 on shared edges: every definite RFCI mark is the true MAG's mark."""
    for e in found.edges:
        if truth.get_edge(e.u, e.v) is None:
            continue  # an extra edge RFCI kept; no shared mark to compare
        for node in (e.u, e.v):
            mark = e.mark_at(node)
            if mark is Mark.CIRCLE:
                continue
            assert truth.mark_at(e.u, e.v, node) is mark


# -- soundness on random worlds (the load-bearer) ----------------------------------


def test_rfci_is_sound_against_truth_and_never_sparser_than_fci() -> None:
    # Over random latent DAGs and random ADMGs, three facts at once: the RFCI
    # skeleton never drops an adjacency FCI keeps (an RFCI removal is a true
    # independence, so FCI removes it too), every definite mark on a shared
    # edge agrees with the true MAG (Theorem 3.2), and the equality count is
    # calibrated and pinned: at these seeds the possible-d-sep phase never
    # fires, so RFCI lands exactly on FCI in 30 of 30 latent-DAG worlds and
    # 30 of 30 ADMG worlds. The worlds where the two differ are pinned in the
    # dedicated tests below.
    rng = np.random.default_rng(7)
    n_equal = 0
    for _ in range(30):
        graph = _random_dag_with_latents(rng, 7, 2)
        truth = mag_of_dag(graph)
        oracle = DSeparationOracle(graph)
        by_rfci = RFCI(test=oracle).fit()
        by_fci = FCI(test=oracle).fit()
        assert _pairs(by_fci.graph) <= _pairs(by_rfci.graph)
        _assert_sound_marks(by_rfci.graph, truth)
        n_equal += by_rfci.graph.edges == by_fci.graph.edges
    assert n_equal >= 30
    n_equal_admg = 0
    for seed in range(30):
        graph = _random_admg(np.random.default_rng(seed), 7)
        truth = mag_of_dag(graph)
        oracle = DSeparationOracle(graph)
        by_rfci = RFCI(test=oracle).fit()
        by_fci = FCI(test=oracle).fit()
        assert _pairs(by_fci.graph) <= _pairs(by_rfci.graph)
        _assert_sound_marks(by_rfci.graph, truth)
        n_equal_admg += by_rfci.graph.edges == by_fci.graph.edges
    assert n_equal_admg >= 30


def test_rfci_is_sound_under_selection() -> None:
    # The selection is in the world, not the algorithm: the oracle answers
    # every query given S, mag_of_dag projects selection into the truth
    # marks, and the closure's R5 to R7 read what that implies. Calibrated at
    # seed 29: RFCI equals FCI in all 15 worlds.
    rng = np.random.default_rng(29)
    n_equal = 0
    for _ in range(15):
        graph = _random_dag_with_margin(rng, 7, 1, 2)
        truth = mag_of_dag(graph)
        oracle = DSeparationOracle(graph)
        by_rfci = RFCI(test=oracle).fit()
        by_fci = FCI(test=oracle).fit()
        assert _pairs(by_fci.graph) <= _pairs(by_rfci.graph)
        _assert_sound_marks(by_rfci.graph, truth)
        n_equal += by_rfci.graph.edges == by_fci.graph.edges
    assert n_equal == 15


# -- where RFCI and FCI part ways ---------------------------------------------------


def test_rfci_keeps_the_edge_only_possible_d_sep_would_remove() -> None:
    # The FCI suite's hand world: the pair (V3, V4) is separable only by
    # conditioning outside both neighbourhoods, so FCI's possible-d-sep
    # re-test removes it. RFCI has no such phase and the pair sits on no
    # unshielded triple or discriminating path that would expose it, so the
    # edge survives. The supergraph still happens to pass the complete-PAG
    # round trip, which is the honest report here: a valid PAG face standing
    # for a coarser class than FCI's (one RFCI-PAG can cover more than one
    # Markov equivalence class), with every definite mark still sound.
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
    by_rfci = RFCI(test=oracle).fit()
    by_fci = FCI(test=oracle).fit()
    assert by_fci.diagnostics["n_pds_removed"] == 1.0
    assert by_fci.graph.get_edge("V3", "V4") is None
    extra = by_rfci.graph.get_edge("V3", "V4")
    assert extra is not None and extra.glyph() == "V3 <-> V4"
    assert _pairs(by_rfci.graph) - _pairs(by_fci.graph) == {frozenset(("V3", "V4"))}
    assert by_rfci.graph.kind is GraphKind.PAG
    assert by_rfci.diagnostics["pag_valid"] == 1.0
    _assert_sound_marks(by_rfci.graph, mag_of_dag(graph))


def test_the_triple_rule_removes_an_edge_and_records_a_minimal_separator() -> None:
    # Found by scanning random ADMGs (p=8, seed 149): the skeleton keeps the
    # pair (V1, V7), and the Lemma 3.1 re-test of an unshielded triple's edge
    # finds the independence and removes it, recording an inclusion-minimal
    # separating set. Stage two catches here exactly what FCI's
    # possible-d-sep phase catches, so the two outputs coincide.
    graph = _random_admg(np.random.default_rng(149), 8)
    oracle = DSeparationOracle(graph)
    result = RFCI(test=oracle).fit()
    assert result.diagnostics["n_stage2_removed"] == 1.0
    assert result.diagnostics["n_stage3_removed"] == 0.0
    assert result.diagnostics["n_triple_tests"] > 0.0
    assert result.sepsets is not None
    recorded = result.sepsets[frozenset(("V1", "V7"))]
    assert recorded == frozenset(("V0", "V2", "V4"))
    assert oracle.independent("V1", "V7", sorted(recorded), alpha=0.05)
    for size in range(len(recorded)):  # minimal: no proper subset separates
        for subset in combinations(sorted(recorded), size):
            assert not oracle.independent("V1", "V7", subset, alpha=0.05)
    by_fci = FCI(test=oracle).fit()
    assert by_fci.diagnostics["n_pds_removed"] == 1.0
    assert result.graph.edges == by_fci.graph.edges


# -- data, knowledge, determinism ---------------------------------------------------


def test_rfci_recovers_the_confounded_collider_from_data() -> None:
    data = _confounded_collider_data()
    result = RFCI(test="linear", alpha=0.01).fit(data)
    assert {e.glyph() for e in result.graph.edges} == {"A o-> C", "B o-> C"}
    assert result.graph.kind is GraphKind.PAG
    assert result.method == "rfci"
    assert result.test == "linear"
    assert result.graph.edges == FCI(test="linear", alpha=0.01).fit(data).graph.edges


def test_confounded_knowledge_orients_and_the_face_falls_back_honestly() -> None:
    # Knowledge enters exactly as in FCI: confound seeds the bidirected edge,
    # and a knowledge-seeded mark is not a class invariant, so the PAG face
    # is lost and the diagnostics say so.
    graph = dag(["L -> X", "L -> Y"], latent=["L"])
    knowledge = DomainKnowledge().confound("X", "Y")
    result = RFCI(test=DSeparationOracle(graph)).fit(knowledge=knowledge)
    edge = result.graph.get_edge("X", "Y")
    assert edge is not None and edge.is_bidirected
    assert result.graph.kind is GraphKind.MIXED
    assert result.diagnostics["pag_valid"] == 0.0


def test_rfci_is_deterministic_run_to_run() -> None:
    # A world where stage two actually fires, so the removal bookkeeping is
    # exercised; two runs on one shared oracle must agree edge for edge and
    # counter for counter.
    graph = _random_admg(np.random.default_rng(149), 8)
    oracle = DSeparationOracle(graph)
    first = RFCI(test=oracle).fit()
    second = RFCI(test=oracle).fit()
    assert first.graph.edges == second.graph.edges
    for key in (
        "n_edges",
        "n_v_structures",
        "n_stage2_removed",
        "n_stage3_removed",
        "n_triple_tests",
        "n_path_tests",
    ):
        assert first.diagnostics[key] == second.diagnostics[key]
