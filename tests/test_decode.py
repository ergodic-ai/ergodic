# SPDX-License-Identifier: Apache-2.0
"""EdgeProbabilities and the explicit decoders (API-148, DEC-079).

The decoder tests are ported from the research library's
``test_probability_dag.py`` (exhaustive exact-MAP optimality, budgets, seeded
samplers, validation), plus knowledge, immutability and a parity check against
a few stored EXP-0626 rows (``tests/data/exp0626_decoder_parity.json``).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ergodic
from ergodic import DomainKnowledge, EdgeProbabilities, decode
from ergodic.discovery import DECODERS, BootstrapResult, Decoder, DecodeResult, DiscoveryResult

FIXTURE = Path(__file__).parent / "data" / "exp0626_decoder_parity.json"
NAMES = ("A", "B", "C", "D")


def research_matrix() -> np.ndarray:
    """The research suite's 4-node pair-state matrix."""
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 1], p[1, 0] = 0.80, 0.05
    p[1, 2], p[2, 1] = 0.70, 0.10
    p[2, 0], p[0, 2] = 0.60, 0.15
    p[2, 3], p[3, 2] = 0.65, 0.05
    p[0, 3], p[3, 0] = 0.25, 0.05
    p[1, 3], p[3, 1] = 0.10, 0.05
    return p


def pair_state(values: np.ndarray | None = None) -> EdgeProbabilities:
    matrix = research_matrix() if values is None else values
    return EdgeProbabilities(NAMES[: len(matrix)], matrix, kind="pair_state")


def random_pair_state(size: int, rng: np.random.Generator) -> np.ndarray:
    """A random pair-state matrix: three Dirichlet states per pair."""
    out = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            _, forward, reverse = rng.dirichlet([1.0, 1.0, 1.0])
            out[i, j], out[j, i] = forward, reverse
    return out


def random_independent(size: int, rng: np.random.Generator) -> np.ndarray:
    out = rng.uniform(size=(size, size))
    np.fill_diagonal(out, 0.0)
    return out


def adjacency(result: DiscoveryResult, names: tuple[str, ...]) -> np.ndarray:
    matrix, _ = result.graph.to_adjacency(order=list(names))
    return matrix.astype(bool)


def score(graph: np.ndarray, p: np.ndarray, floor: float = 1e-12) -> float:
    total = 0.0
    for i in range(len(graph)):
        for j in range(i + 1, len(graph)):
            if graph[i, j]:
                state = p[i, j]
            elif graph[j, i]:
                state = p[j, i]
            else:
                state = 1.0 - p[i, j] - p[j, i]
            total += math.log(max(floor, state))
    return total


def is_acyclic(graph: np.ndarray) -> bool:
    remaining = graph.copy()
    alive = np.ones(len(graph), dtype=bool)
    while alive.any():
        roots = alive & ~remaining[alive].any(axis=0)
        if not roots.any():
            return False
        alive &= ~roots
        remaining[roots] = False
    return True


def all_dags(size: int) -> list[np.ndarray]:
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    graphs = []
    for states in itertools.product((0, 1, 2), repeat=len(pairs)):
        graph = np.zeros((size, size), dtype=bool)
        for (i, j), state in zip(pairs, states, strict=True):
            if state == 1:
                graph[i, j] = True
            elif state == 2:
                graph[j, i] = True
        if is_acyclic(graph):
            graphs.append(graph)
    return graphs


# -- EdgeProbabilities ----------------------------------------------------------------


def test_edge_probabilities_store_a_read_only_copy_and_never_share_the_input() -> None:
    values = research_matrix()
    before = values.copy()
    probs = pair_state(values)
    assert probs.values is not values
    assert not probs.values.flags.writeable
    values[0, 1] = 0.0
    assert probs.values[0, 1] == 0.8
    np.testing.assert_array_equal(research_matrix(), before)
    with pytest.raises(ValueError):
        probs.values[0, 1] = 0.5
    assert probs.calibrated is False
    assert probs.kind == "pair_state"
    assert "posterior" not in repr(probs)
    assert repr(probs).startswith("EdgeProbabilities(pair_state, 4 variables")


@pytest.mark.parametrize(
    ("values", "kind", "error", "match"),
    [
        (np.zeros((2, 3)), "independent", ValueError, "2 x 2"),
        (np.eye(2), "independent", ValueError, "diagonal"),
        (np.array([[0.0, np.nan], [0.1, 0.0]]), "independent", ValueError, "finite"),
        (np.array([[0.0, 1.2], [0.1, 0.0]]), "independent", ValueError, r"\[0, 1\]"),
        (np.array([[0.0, -0.1], [0.1, 0.0]]), "independent", ValueError, r"\[0, 1\]"),
        (np.zeros((2, 2), dtype=bool), "independent", TypeError, "real-valued"),
        (np.array([[0.0, 0.7], [0.4, 0.0]]), "pair_state", ValueError, "<= 1"),
        (np.array([[0.0, 0.7], [0.4, 0.0]]), "frequency", ValueError, "<= 1"),
        (np.zeros((2, 2)), "posterior", ValueError, "kind"),
    ],
)
def test_edge_probabilities_validate_strictly(
    values: np.ndarray, kind: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        EdgeProbabilities(("A", "B"), values, kind=kind)


def test_edge_probabilities_validate_names_and_flags() -> None:
    with pytest.raises(ValueError, match="unique"):
        EdgeProbabilities(("A", "A"), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="at least one"):
        EdgeProbabilities((), np.zeros((0, 0)))
    with pytest.raises(TypeError, match="calibrated"):
        EdgeProbabilities(("A", "B"), np.zeros((2, 2)), calibrated=1)  # type: ignore[arg-type]
    # Independent marginals may put both directions of a pair high.
    both = EdgeProbabilities(("A", "B"), np.array([[0.0, 0.9], [0.8, 0.0]]))
    assert both.kind == "independent"


def test_provenance_is_a_frozen_copy() -> None:
    source = {"model": "avici", "decoder": {"threshold": 0.5}}
    probs = EdgeProbabilities(("A", "B"), np.zeros((2, 2)), provenance=source)
    source["decoder"]["threshold"] = 0.9  # type: ignore[index]
    assert probs.provenance["decoder"]["threshold"] == 0.5
    with pytest.raises(TypeError):
        probs.provenance["model"] = "cdfm"  # type: ignore[index]


def test_independent_marginals_condition_on_no_two_cycle() -> None:
    p, q = 0.8, 0.3
    probs = EdgeProbabilities(("A", "B"), np.array([[0.0, p], [q, 0.0]]))
    pair = probs.to_pair_states()
    total = (1 - p) * (1 - q) + p * (1 - q) + q * (1 - p)
    assert pair.kind == "pair_state"
    assert pair.values[0, 1] == pytest.approx(p * (1 - q) / total)
    assert pair.values[1, 0] == pytest.approx(q * (1 - p) / total)
    assert pair.provenance["pair_state_conversion"] == "no_reciprocal_conditioning_v1"
    # Both edges certain: the limit is undecided, so the pair splits evenly.
    certain = EdgeProbabilities(("A", "B"), np.array([[0.0, 1.0], [1.0, 0.0]])).to_pair_states()
    assert certain.values.tolist() == [[0.0, 0.5], [0.5, 0.0]]
    # A pair-state input is returned as it is; a frequency input keeps its numbers.
    already = pair_state()
    assert already.to_pair_states() is already
    freq = EdgeProbabilities(NAMES, research_matrix(), kind="frequency").to_pair_states()
    assert freq.kind == "pair_state"
    np.testing.assert_array_equal(freq.values, research_matrix())


def test_existence_reads_the_pair_state_view() -> None:
    probs = EdgeProbabilities(("A", "B", "C"), np.array([[0, 0.8, 0], [0.3, 0, 0], [0, 0, 0.0]]))
    existence = probs.existence()
    pair = probs.to_pair_states().values
    assert existence[0, 1] == existence[1, 0] == pytest.approx(pair[0, 1] + pair[1, 0])
    assert existence[0, 2] == 0.0 and np.all(np.diag(existence) == 0)
    assert not existence.flags.writeable


def test_threshold_is_a_diagnostic_mixed_graph_that_may_be_cyclic() -> None:
    values = np.zeros((3, 3))
    values[0, 1] = values[1, 2] = values[2, 0] = 0.9  # a three-cycle
    values[0, 2] = 0.95  # with values[2, 0] above too: a reciprocal pair
    graph = EdgeProbabilities(("A", "B", "C"), values).threshold(0.5)
    assert graph.kind.value == "mixed"
    glyphs = sorted(edge.glyph() for edge in graph.edges)
    assert glyphs == ["A --- C", "A --> B", "B --> C"]
    assert graph.nodes == frozenset("ABC")
    assert EdgeProbabilities(("A", "B", "C"), values).threshold(0.92).edges
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        EdgeProbabilities(("A", "B", "C"), values).threshold(1.0)


def test_frame_round_trip_and_column_reordering() -> None:
    probs = EdgeProbabilities(NAMES, research_matrix(), kind="pair_state", provenance={"k": 1})
    frame = probs.to_frame()
    assert frame.index.name == "source" and frame.columns.name == "target"
    assert frame.loc["A", "B"] == 0.8
    back = EdgeProbabilities.from_frame(frame, kind="pair_state", provenance={"k": 1})
    assert back == probs
    shuffled = frame[["D", "B", "A", "C"]]
    assert EdgeProbabilities.from_frame(shuffled, kind="pair_state", provenance={"k": 1}) == probs
    with pytest.raises(ValueError, match="same unique"):
        EdgeProbabilities.from_frame(pd.DataFrame([[0.0, 0.1]], index=["A"], columns=["A", "B"]))


def test_from_bootstrap_reads_directed_glyphs_and_splits_undirected_mass() -> None:
    result = BootstrapResult(
        method="pc",
        n_runs=10,
        n_failed=0,
        edges={frozenset("AB"): 1.0, frozenset("BC"): 0.6, frozenset("AC"): 0.2},
        marks={"A --> B": 0.7, "A <-- B": 0.1, "A --- B": 0.2, "B --> C": 0.6, "A <-> C": 0.2},
        variables=("A", "B", "C"),
    )
    probs = EdgeProbabilities.from_bootstrap(result)
    assert probs.kind == "frequency"
    assert probs.values[0, 1] == pytest.approx(0.8)  # 0.7 + half of 0.2
    assert probs.values[1, 0] == pytest.approx(0.2)  # 0.1 + half of 0.2
    assert probs.values[1, 2] == pytest.approx(0.6)
    assert probs.values[0, 2] == probs.values[2, 0] == 0.0
    assert probs.provenance["dropped_frequency"] == pytest.approx(0.2)
    assert probs.provenance["n_runs"] == 10
    dropped = EdgeProbabilities.from_bootstrap(result, undirected="drop")
    assert dropped.values[0, 1] == pytest.approx(0.7)
    assert dropped.provenance["dropped_frequency"] == pytest.approx(0.4)
    graph = decode(probs).graph
    assert {edge.glyph() for edge in graph.edges} == {"A --> B", "B --> C"}


# -- decoders: research ports -------------------------------------------------------------


@pytest.mark.parametrize("size", [3, 4])
def test_exact_map_matches_the_exhaustive_optimum(size: int) -> None:
    rng = np.random.default_rng(size)
    dags = all_dags(size)
    names = NAMES[:size]
    matrices = [research_matrix()[:size, :size]] + [random_pair_state(size, rng) for _ in range(6)]
    for values in matrices:
        probs = EdgeProbabilities(names, values, kind="pair_state")
        result = decode(probs, "exact_map")
        best = max(score(graph, values) for graph in dags)
        graph = adjacency(result, names)
        assert score(graph, values) == pytest.approx(best)
        assert result.diagnostics["objective"] == pytest.approx(best)
        assert result.diagnostics["search_expansions"][0] > 0
        # greedy never beats the exact optimum
        assert decode(probs).diagnostics["objective"] <= best + 1e-9


def test_greedy_reproduces_the_research_graph_on_the_research_matrix() -> None:
    result = decode(pair_state())
    assert isinstance(result, DecodeResult) and isinstance(result, DiscoveryResult)
    assert result.graph.kind.value == "dag"
    assert sorted(e.glyph() for e in result.graph.edges) == ["A --> B", "B --> C", "C --> D"]
    assert result.diagnostics["rejected_cycles"] == (1,)  # C -> A would close a cycle
    assert result.method == "decode:greedy_v1"


def test_cardinality_decoders_keep_exactly_the_budget() -> None:
    expected = decode(pair_state(), "expected_count")
    soft = decode(pair_state(), "soft_f1")
    assert expected.diagnostics["edge_budget"] == 4
    assert len(expected.graph.edges) == 4
    assert 0 <= soft.diagnostics["edge_budget"] <= 6
    assert len(soft.graph.edges) == soft.diagnostics["edge_budget"]
    assert expected.diagnostics["parameters"]["edge_budget_rule"] == "expected_count"


def test_samplers_are_seeded_diverse_and_leave_the_global_rng_alone() -> None:
    probs = pair_state()
    before = probs.values.copy()
    np.random.seed(1234)
    first = decode(probs, "pair_sampler", samples=20, seed=9)
    second = decode(probs, "pair_sampler", samples=20, seed=9)
    other = decode(probs, "pair_sampler", samples=20, seed=10)
    gumbel = decode(probs, "perturb_and_map", samples=8, seed=17)
    assert first.candidates == second.candidates
    assert first.diagnostics["rejected_cycles"] == second.diagnostics["rejected_cycles"]
    assert first.candidates != other.candidates
    assert len(first.candidates) == 20 and len(gumbel.candidates) == 8
    assert len({c.edges for c in gumbel.candidates}) > 1
    assert all(c.is_dag() for c in (*first.candidates, *gumbel.candidates))
    assert first.graph in first.candidates
    np.testing.assert_array_equal(probs.values, before)
    # The global stream continues exactly where seed(1234) left it.
    drawn = np.random.random(3)
    np.random.seed(1234)
    np.testing.assert_array_equal(drawn, np.random.random(3))


def test_the_consensus_is_the_medoid_candidate() -> None:
    result = decode(pair_state(), "pair_sampler", samples=30, seed=3)
    graphs = [adjacency(DiscoveryResult(graph=c, method="x"), NAMES) for c in result.candidates]

    def states(g: np.ndarray) -> list[int]:
        return [int(g[i, j]) + 2 * int(g[j, i]) for i in range(4) for j in range(i + 1, 4)]

    totals = [
        sum(np.sum(np.array(states(a)) != np.array(states(b))) for b in graphs) for a in graphs
    ]
    assert result.graph == result.candidates[int(np.argmin(totals))]


def test_decoding_is_deterministic_and_never_mutates_its_input() -> None:
    rng = np.random.default_rng(0)
    values = random_independent(6, rng)
    probs = EdgeProbabilities(tuple("UVWXYZ"), values)
    snapshot = probs.values.copy()
    for method in DECODERS:
        stochastic = method in ("pair_sampler_v1", "perturb_and_map_v1")
        kwargs: dict[str, Any] = {"samples": 4, "seed": 1} if stochastic else {}
        first = decode(probs, method, **kwargs)
        second = decode(probs, method, **kwargs)
        assert first.graph == second.graph
        assert first.candidates == second.candidates
        assert first.diagnostics == second.diagnostics
        assert first.graph.is_dag()
        assert first.graph.nodes == frozenset("UVWXYZ")
    np.testing.assert_array_equal(probs.values, snapshot)
    np.testing.assert_array_equal(values, snapshot)


def test_greedy_without_threshold_is_the_half_threshold_on_independent_marginals() -> None:
    rng = np.random.default_rng(5)
    for _ in range(20):
        probs = EdgeProbabilities(tuple("ABCDEFG"), random_independent(7, rng))
        assert decode(probs).graph == decode(probs, threshold=0.5).graph


def test_a_threshold_nominates_edges_above_it_and_orders_them_by_weight() -> None:
    values = np.zeros((3, 3))
    values[0, 1], values[1, 2], values[2, 0] = 0.45, 0.40, 0.35
    probs = EdgeProbabilities(("A", "B", "C"), values)
    assert not decode(probs).graph.edges  # nothing above 0.5
    low = decode(probs, threshold=0.3)
    # All three pass 0.3; the weakest (C -> A) would close the cycle and is dropped.
    assert sorted(e.glyph() for e in low.graph.edges) == ["A --> B", "B --> C"]
    assert low.diagnostics["rejected_cycles"] == (1,)
    assert low.diagnostics["parameters"]["threshold"] == 0.3
    # The cut is strict, as AVICI and CDFM threshold natively.
    assert sorted(e.glyph() for e in decode(probs, threshold=0.35).graph.edges) == [
        "A --> B",
        "B --> C",
    ]
    assert not decode(probs, threshold=0.45).graph.edges


def test_ties_break_by_variable_order() -> None:
    values = np.array([[0.0, 0.7, 0.0], [0.7, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = decode(EdgeProbabilities(("Q", "P", "R"), values))
    assert [e.glyph() for e in result.graph.edges] == ["P <-- Q"]  # Q comes first in the order


def test_decode_result_labels_the_graph_as_decoded() -> None:
    probs = EdgeProbabilities(NAMES, research_matrix(), provenance={"model": "fixture"})
    result = decode(probs)
    d = result.diagnostics
    assert d["decoded"] is True and d["native"] is False
    assert d["decoder"] == "greedy_v1" and d["decoder_port_of"] == "greedy_pair_map_v1"
    assert d["note"] == "not a graph posterior; orientation not certified"
    assert d["source_kind"] == "independent" and d["source_provenance"]["model"] == "fixture"
    assert d["seed"] is None and d["samples"] == 1
    assert d["n_edges"] == len(result.graph.edges)
    assert result.probabilities is probs
    assert result.decoder == "greedy_v1"
    assert "decoded dag" in repr(result)
    sampled = decode(probs, "perturb_and_map", samples=3, seed=2)
    assert sampled.diagnostics["seed"] == 2 and "consensus of 3" in repr(sampled)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "nope"}, "unknown decoder"),
        ({"method": "exact_map", "threshold": 0.3}, "greedy decoder only"),
        ({"threshold": 1.0}, r"\[0, 1\)"),
        ({"samples": 2}, "deterministic"),
        ({"seed": 3}, "takes no seed"),
        ({"method": "pair_sampler", "samples": 3}, "explicit integer seed"),
        ({"method": "pair_sampler", "samples": 0, "seed": 1}, "positive int"),
        ({"method": "perturb_and_map", "seed": -1}, "non-negative"),
        ({"method": "exact_map", "max_nodes": 3}, "max_nodes=3"),
    ],
)
def test_settings_that_do_not_apply_are_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decode(pair_state(), **kwargs)


def test_decode_needs_edge_probabilities_and_greedy_has_no_node_cap() -> None:
    with pytest.raises(TypeError, match="EdgeProbabilities"):
        decode(research_matrix())  # type: ignore[arg-type]
    rng = np.random.default_rng(1)
    names = tuple(f"x{i}" for i in range(30))
    big = EdgeProbabilities(names, random_independent(30, rng) * 0.6)
    assert decode(big).graph.is_dag()
    assert decode(big, "pair_sampler", samples=2, seed=0).graph.is_dag()
    with pytest.raises(ValueError, match="max_nodes"):
        decode(big, "exact_map")


def test_versioned_ids_and_aliases_resolve_alike() -> None:
    probs = pair_state()
    assert decode(probs, "greedy_v1").graph == decode(probs, "GREEDY").graph
    assert set(DECODERS) == {
        "greedy_v1",
        "exact_map_v1",
        "expected_count_v1",
        "expected_count_v2",
        "soft_f1_v1",
        "soft_f1_v2",
        "pair_sampler_v1",
        "perturb_and_map_v1",
    }
    with pytest.raises(TypeError):
        DECODERS["greedy_v1"] = "changed"  # type: ignore[index]
    assert all(isinstance(summary, str) for summary in DECODERS.values())
    # A short name reads as the latest version.
    assert Decoder("expected_count").method == "expected_count_v2"
    assert Decoder("soft_f1").method == "soft_f1_v2"
    assert Decoder("expected_count_v1").method == "expected_count_v1"
    assert "superseded by expected_count_v2" in DECODERS["expected_count_v1"]
    config = Decoder("exact_map")
    assert config.method == "exact_map_v1" and not config.stochastic
    assert config.decode(probs).graph == decode(probs, "exact_map").graph
    with pytest.raises(ValueError, match="explicit integer seed"):
        Decoder("pair_sampler")


# -- knowledge ---------------------------------------------------------------------------


ALL_METHODS: list[tuple[str, dict[str, Any]]] = [
    ("greedy", {}),
    ("greedy", {"threshold": 0.05}),
    ("exact_map", {}),
    ("expected_count", {}),
    ("soft_f1", {}),
    ("pair_sampler", {"samples": 12, "seed": 4}),
    ("perturb_and_map", {"samples": 4, "seed": 4}),
]


@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_every_decoder_honours_the_knowledge(method: str, kwargs: dict[str, Any]) -> None:
    probs = pair_state()
    knowledge = (
        DomainKnowledge()
        .forbid("A", "B")
        .require("D", "B")  # a weak edge, required anyway
        .root("C")
        .order("A", "D")
    )
    # D -> B has model probability 0.05: a strong conflict, warned about.
    with pytest.warns(UserWarning, match="D -> B"):
        result = decode(probs, method, knowledge=knowledge, **kwargs)
    assert knowledge.violations(result.graph) == []
    for candidate in result.candidates:
        assert knowledge.violations(candidate) == []
    edge = result.graph.get_edge("D", "B")
    assert edge is not None and edge.directed_pair() == ("D", "B")
    assert result.diagnostics["knowledge_applied"] is True


@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_tiers_and_a_skeleton_restriction_hold_for_every_decoder(
    method: str, kwargs: dict[str, Any]
) -> None:
    rng = np.random.default_rng(11)
    names = tuple("ABCDEF")
    probs = EdgeProbabilities(names, random_independent(6, rng))
    knowledge = DomainKnowledge().with_tiers([["E", "F"], ["A", "B"], ["C", "D"]])
    result = decode(probs, method, knowledge=knowledge, **kwargs)
    assert knowledge.violations(result.graph) == []
    restricted = DomainKnowledge().restrict_to([("A", "B"), ("B", "C"), ("C", "D")])
    # The random matrix is dense: the pair A, F (outside the skeleton) is adjacent with
    # probability above 0.9, a strong conflict.
    with pytest.warns(UserWarning, match=r"skeleton pairs \(pair A, F"):
        graph = decode(probs, method, knowledge=restricted, **kwargs).graph
    assert {frozenset((e.u, e.v)) for e in graph.edges} <= {
        frozenset("AB"),
        frozenset("BC"),
        frozenset("CD"),
    }


def test_exact_map_under_knowledge_is_the_constrained_optimum() -> None:
    values = research_matrix()
    knowledge = DomainKnowledge().forbid("A", "B").require("D", "A")
    with pytest.warns(UserWarning, match="D -> A"):  # model probability 0.05
        result = decode(pair_state(), "exact_map", knowledge=knowledge)
    feasible = [
        g
        for g in all_dags(4)
        if not g[0, 1] and g[3, 0]  # A -> B forbidden, D -> A required
    ]
    best = max(score(g, values) for g in feasible)
    assert score(adjacency(result, NAMES), values) == pytest.approx(best)


def test_knowledge_refusals() -> None:
    probs = pair_state()
    with pytest.raises(ValueError, match="banned from being adjacent"):
        decode(probs, knowledge=DomainKnowledge().confound("A", "B").forbid_adjacency("A", "B"))
    with pytest.raises(ValueError, match="lack"):
        decode(probs, knowledge=DomainKnowledge().require("A", "Z"))
    cyclic = DomainKnowledge().require("A", "B").require("B", "C").require("C", "A")
    with pytest.raises(ValueError, match="cycle"):
        decode(probs, knowledge=cyclic)


# -- parity with the research decoder on stored EXP-0626 rows ------------------------------


def test_greedy_reproduces_the_stored_exp0626_common_dags() -> None:
    document = json.loads(FIXTURE.read_text())
    rows = document["rows"]
    assert 1 <= len(rows) <= 10 and FIXTURE.stat().st_size < 64_000
    assert "Ergodic" in document["_provenance"]["owner"]
    for row in rows:
        values = np.asarray(row["directed_edge_probabilities"], dtype=np.float64)
        names = tuple(f"x{i}" for i in range(len(values)))
        probs = EdgeProbabilities(names, values, kind="independent")
        decoded = adjacency(decode(probs), names)
        expected = np.asarray(row["common_dag_adjacency"], dtype=bool)
        assert np.array_equal(decoded, expected), row["id"]


def test_top_level_exports() -> None:
    assert ergodic.EdgeProbabilities is EdgeProbabilities
    assert ergodic.decode is decode
    assert ergodic.DecodeResult is DecodeResult


def test_order_knowledge_is_one_time_order_stricter_than_violations() -> None:
    """order(a, b) and order(c, d) with strong b -> c and d -> a (the review's probe).

    {b -> c, d -> a} breaks neither statement on its own reading (no later variable is
    an ancestor of an earlier one), but no single time order puts a before b, c before
    d and every edge forward. The decoders enforce that one order, so they keep only
    one of the two edges and report the other as skipped for the order.
    """
    names = ("a", "b", "c", "d")
    values = np.full((4, 4), 0.01)
    np.fill_diagonal(values, 0.0)
    values[1, 2] = values[3, 0] = 0.95
    knowledge = DomainKnowledge().order("a", "b").order("c", "d")
    both = ergodic.dag([("b", "c"), ("d", "a")], nodes=names)
    assert knowledge.violations(both) == []
    probs = EdgeProbabilities(names, values, kind="pair_state")
    for method in ("greedy", "exact_map"):
        result = decode(probs, method, knowledge=knowledge)
        assert len(result.graph.edges) == 1
        assert knowledge.violations(result.graph) == []
    greedy = decode(probs, knowledge=knowledge)
    # Equal weights: the tie goes to the earlier source, b.
    assert [e.glyph() for e in greedy.graph.edges] == ["b --> c"]
    assert greedy.diagnostics["order_skipped"] == (1,)
    assert greedy.diagnostics["rejected_cycles"] == (0,)
    # Without the order statements both edges stay.
    assert len(decode(probs).graph.edges) == 2


def test_threshold_source_is_recorded_locally() -> None:
    assert decode(pair_state()).diagnostics["threshold_source"] is None
    assert decode(pair_state(), threshold=0.2).diagnostics["threshold_source"] == "decoder"


def test_from_bootstrap_reads_names_with_spaces() -> None:
    result = BootstrapResult(
        method="pc",
        n_runs=4,
        n_failed=0,
        edges={frozenset(("blood pressure", "heart rate")): 1.0},
        marks={"blood pressure --> heart rate": 1.0},
        variables=("blood pressure", "heart rate"),
    )
    probs = EdgeProbabilities.from_bootstrap(result)
    assert probs.values[0, 1] == 1.0
    window = BootstrapResult(
        method="pcmci",
        n_runs=1,
        n_failed=0,
        edges={},
        marks={"x[t-1] --> y[t]": 1.0},
        variables=("x", "y"),
    )
    with pytest.raises(ValueError, match="window graphs"):
        EdgeProbabilities.from_bootstrap(window)
