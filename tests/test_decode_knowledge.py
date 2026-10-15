# SPDX-License-Identifier: Apache-2.0
"""Decoding with domain knowledge: soft statements, confounders and the audit (API-151).

DEC-081: each builder's ``confidence=`` makes a statement soft, the decoders read
knowledge-weighted pair states, ``confound`` turns the result into an ADMG, and every
decode with knowledge audits each statement against the model.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from tests.test_decode import (
    ALL_METHODS,
    NAMES,
    adjacency,
    all_dags,
    pair_state,
    random_independent,
    random_pair_state,
    research_matrix,
    score,
)

from ergodic import DomainKnowledge, EdgeProbabilities, decode, identify_effect
from ergodic.graph import dag

Builder = Callable[[DomainKnowledge, float | None], DomainKnowledge]

# One statement of every soft-capable type, each changing the research matrix's decode.
STATEMENTS: dict[str, Builder] = {
    "require": lambda k, c: k.require("D", "B", confidence=c),
    "forbid": lambda k, c: k.forbid("A", "B", confidence=c),
    "forbid_adjacency": lambda k, c: k.forbid_adjacency("B", "C", confidence=c),
    "order": lambda k, c: k.order("D", "C", confidence=c),
    "tiers": lambda k, c: k.with_tiers([["C"], ["A", "B", "D"]], confidence=c),
    "root": lambda k, c: k.root("C", confidence=c),
    "sink": lambda k, c: k.sink("A", confidence=c),
    "forbid_between": lambda k, c: k.forbid_between(["A", "B"], ["C"], confidence=c),
    "restrict_to": lambda k, c: k.restrict_to([("A", "B"), ("C", "D"), ("A", "C")], confidence=c),
}
NEAR_ONE = 1.0 - 1e-9

# For tests that state knowledge the model rejects on purpose (require D -> B, which the
# model gives 0.05) or draw random matrices: the conflict warning is expected there and
# is itself tested in test_strong_conflicts_are_listed_and_warned_once.
EXPECTED_CONFLICTS = pytest.mark.filterwarnings(
    "ignore:.*have a pair the model gives a probability below:UserWarning"
)


def _edges(result: Any) -> list[str]:
    return sorted(edge.glyph() for edge in result.graph.edges)


# -- soft statements ---------------------------------------------------------------


@EXPECTED_CONFLICTS
@pytest.mark.parametrize("kind", sorted(STATEMENTS))
@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_one_half_decodes_exactly_as_no_knowledge(
    kind: str, method: str, kwargs: dict[str, Any]
) -> None:
    probs = pair_state()
    plain = decode(probs, method, **kwargs)
    neutral = decode(probs, method, knowledge=STATEMENTS[kind](DomainKnowledge(), 0.5), **kwargs)
    assert neutral.graph == plain.graph
    assert neutral.candidates == plain.candidates
    assert neutral.diagnostics["objective"] == plain.diagnostics["objective"]
    assert neutral.diagnostics["knowledge_objective"] == plain.diagnostics["objective"]
    assert neutral.diagnostics["knowledge_cost"] == 0.0


@pytest.mark.parametrize("kind", sorted(STATEMENTS))
@pytest.mark.parametrize(
    ("method", "kwargs"), [("greedy", {}), ("greedy", {"threshold": 0.05}), ("exact_map", {})]
)
@EXPECTED_CONFLICTS
def test_confidence_near_one_matches_the_hard_statement(
    kind: str, method: str, kwargs: dict[str, Any]
) -> None:
    probs = pair_state()
    hard = decode(probs, method, knowledge=STATEMENTS[kind](DomainKnowledge(), None), **kwargs)
    soft = decode(probs, method, knowledge=STATEMENTS[kind](DomainKnowledge(), NEAR_ONE), **kwargs)
    assert soft.graph == hard.graph, kind
    # Each statement changes the decode, so the test is not vacuous.
    assert hard.graph != decode(probs, method, **kwargs).graph, kind


def test_near_one_matches_hard_on_independent_marginals_with_a_threshold() -> None:
    rng = np.random.default_rng(5)
    probs = EdgeProbabilities(tuple("ABCDE"), random_independent(5, rng))
    for kind in ("forbid", "root", "sink", "forbid_adjacency"):
        hard = decode(probs, threshold=0.4, knowledge=STATEMENTS[kind](DomainKnowledge(), None))
        soft = decode(probs, threshold=0.4, knowledge=STATEMENTS[kind](DomainKnowledge(), NEAR_ONE))
        assert soft.graph == hard.graph, kind


@EXPECTED_CONFLICTS
@pytest.mark.parametrize("method", ["greedy", "exact_map"])
def test_a_soft_requirement_switches_on_once_and_stays_on(method: str) -> None:
    probs = pair_state()
    present = []
    for confidence in np.linspace(0.5, 0.999, 40):
        knowledge = DomainKnowledge().require("D", "B", confidence=float(confidence))
        edge = decode(probs, method, knowledge=knowledge).graph.get_edge("D", "B")
        present.append(edge is not None and edge.directed_pair() == ("D", "B"))
    assert not present[0] and present[-1]
    first = present.index(True)
    assert all(present[first:])


def test_a_soft_forbid_switches_off_once_and_stays_off() -> None:
    probs = pair_state()
    present = []
    for confidence in np.linspace(0.5, 0.999, 40):
        knowledge = DomainKnowledge().forbid("A", "B", confidence=float(confidence))
        present.append(decode(probs, knowledge=knowledge).graph.get_edge("A", "B") is not None)
    assert present[0] and not present[-1]
    assert not any(present[present.index(False) :])


@EXPECTED_CONFLICTS
def test_soft_statements_combine_and_pool_through_and() -> None:
    probs = pair_state()
    left = DomainKnowledge().forbid("A", "B", confidence=0.95)
    right = DomainKnowledge().require("D", "B", confidence=0.99)
    both = decode(probs, "exact_map", knowledge=left & right)
    assert both.diagnostics["n_soft_statements"] == 2
    assert both.graph.get_edge("A", "B") is None
    assert both.graph.get_edge("D", "B") is not None


def test_a_soft_statement_on_a_hard_pair_changes_nothing() -> None:
    probs = pair_state()
    hard = DomainKnowledge().forbid("A", "B")
    plus = hard.require("A", "B", confidence=0.99)
    for method in ("greedy", "exact_map"):
        assert (
            decode(probs, method, knowledge=plus).graph
            == decode(probs, method, knowledge=hard).graph
        )
    audit = decode(probs, knowledge=plus).diagnostics["knowledge_audit"]
    assert [(e["statement"], e["confidence"], e["holds"]) for e in audit] == [
        ("no A -> B", None, True),
        ("A -> B", 0.99, False),
    ]


@EXPECTED_CONFLICTS
@pytest.mark.parametrize("size", [3, 4])
def test_exact_map_is_exact_on_the_combined_score(size: int) -> None:
    rng = np.random.default_rng(100 + size)
    names = NAMES[:size]
    dags = all_dags(size)
    for trial in range(8):
        values = random_pair_state(size, rng)
        probs = EdgeProbabilities(names, values, kind="pair_state")
        a, b, c = (names[i] for i in rng.permutation(size)[:3])
        knowledge = (
            DomainKnowledge()
            .require(a, b, confidence=float(rng.uniform(0.5, 0.99)))
            .root(c, confidence=float(rng.uniform(0.5, 0.99)))
            .order(b, a, confidence=float(rng.uniform(0.5, 0.99)))
            .restrict_to([(a, b), (b, c)], confidence=float(rng.uniform(0.5, 0.99)))
        )
        if trial % 2:
            knowledge = knowledge.forbid(c, a)  # a hard constraint alongside

        def combined(
            graph: np.ndarray, values: np.ndarray = values, knowledge: DomainKnowledge = knowledge
        ) -> float:
            edges = [(names[i], names[j]) for i, j in zip(*np.nonzero(graph), strict=True)]
            return score(graph, values) + knowledge.log_prior(dag(edges, nodes=names))

        best = max(combined(g) for g in dags)
        exact = decode(probs, "exact_map", knowledge=knowledge)
        assert combined(adjacency(exact, names)) == pytest.approx(best)
        greedy = decode(probs, knowledge=knowledge)
        assert combined(adjacency(greedy, names)) <= best + 1e-9
        # knowledge_objective is the combined score up to a graph-free constant.
        for result in (exact, greedy):
            graph = adjacency(result, names)
            edges = [(names[i], names[j]) for i, j in zip(*np.nonzero(graph), strict=True)]
            prior = knowledge.log_prior(dag(edges, nodes=names))
            assert result.diagnostics["objective"] == pytest.approx(score(graph, values))
            offsets = result.diagnostics["knowledge_objective"] - result.diagnostics["objective"]
            if result is exact:
                constant = offsets - prior
            else:
                assert offsets - prior == pytest.approx(constant)


@EXPECTED_CONFLICTS
@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_soft_decoding_is_deterministic_pure_and_leaves_the_global_rng_alone(
    method: str, kwargs: dict[str, Any]
) -> None:
    rng = np.random.default_rng(3)
    values = random_independent(5, rng)
    copy = values.copy()
    probs = EdgeProbabilities(tuple("ABCDE"), values)
    knowledge = (
        DomainKnowledge()
        .root("A", confidence=0.8)
        .require("B", "C", confidence=0.9)
        .with_tiers([["A", "B"], ["C", "D", "E"]], confidence=0.7)
        .confound("D", "E")
    )
    np.random.seed(123)
    before = np.random.random_sample(3)
    np.random.seed(123)
    first = decode(probs, method, knowledge=knowledge, **kwargs)
    second = decode(probs, method, knowledge=knowledge, **kwargs)
    assert first.graph == second.graph and first.candidates == second.candidates
    assert first.diagnostics["knowledge_audit"] == second.diagnostics["knowledge_audit"]
    assert np.array_equal(values, copy) and np.array_equal(probs.values, copy)
    assert np.array_equal(np.random.random_sample(3), before)


# -- confounders and ADMGs ------------------------------------------------------------


def _matrix(names: tuple[str, ...], cells: dict[tuple[str, str], float]) -> EdgeProbabilities:
    values = np.full((len(names), len(names)), 0.02)
    np.fill_diagonal(values, 0.0)
    index = {name: i for i, name in enumerate(names)}
    for (a, b), value in cells.items():
        values[index[a], index[b]] = value
    return EdgeProbabilities(names, values)


@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_confound_adds_a_bidirected_edge_and_gives_an_admg(
    method: str, kwargs: dict[str, Any]
) -> None:
    probs = _matrix(("X", "M", "Y"), {("X", "M"): 0.9, ("M", "Y"): 0.9})
    knowledge = DomainKnowledge().confound("X", "Y")
    result = decode(probs, method, knowledge=knowledge, **kwargs)
    assert result.graph.kind.value == "admg"
    assert result.diagnostics["graph_kind"] == "admg"
    assert result.diagnostics["confounded_pairs"] == (("X", "Y"),)
    assert result.diagnostics["latent_confounders"] == {}
    assert all(c.kind.value == "admg" for c in result.candidates)
    assert "X <-> Y" in _edges(result)
    assert knowledge.violations(result.graph) == []
    assert repr(result).startswith(f"DecodeResult(decode:{result.decoder}: decoded admg")


def test_a_front_door_admg_is_identified() -> None:
    probs = _matrix(("X", "M", "Y"), {("X", "M"): 0.9, ("M", "Y"): 0.9})
    result = decode(probs, knowledge=DomainKnowledge().confound("X", "Y"))
    assert _edges(result) == ["M --> Y", "M <-- X", "X <-> Y"]
    assert identify_effect(result.graph, "X", "Y").strategy == "frontdoor"


def test_an_adjustment_admg_is_identified() -> None:
    probs = _matrix(("Z", "X", "Y"), {("Z", "X"): 0.9, ("X", "Y"): 0.9})
    result = decode(probs, knowledge=DomainKnowledge().confound("Z", "Y"))
    identification = identify_effect(result.graph, "X", "Y")
    assert identification.strategy == "adjustment"


def test_a_bow_is_an_explicit_latent_and_leaves_the_effect_unidentified() -> None:
    probs = _matrix(("X", "Y"), {("X", "Y"): 0.9})
    knowledge = DomainKnowledge().confound("Y", "X")
    result = decode(probs, knowledge=knowledge)
    assert result.graph.kind.value == "admg"
    assert result.graph.latent == {"L__X__Y"}
    assert _edges(result) == ["L__X__Y --> X", "L__X__Y --> Y", "X --> Y"]
    assert result.diagnostics["latent_confounders"] == {"L__X__Y": ("X", "Y")}
    assert result.diagnostics["n_edges"] == 1
    assert knowledge.violations(result.graph) == []
    assert identify_effect(result.graph, "X", "Y").strategy == "none"
    # The graph serializes: the latent node carries the confounder.
    assert result.graph.to_dict()["latent"] == ["L__X__Y"]


def test_a_latent_name_never_collides_with_a_variable() -> None:
    probs = _matrix(("X", "Y", "L__X__Y"), {("X", "Y"): 0.9})
    result = decode(probs, knowledge=DomainKnowledge().confound("X", "Y"))
    assert result.graph.latent == {"_L__X__Y"}


# -- the audit --------------------------------------------------------------------------


def test_the_audit_lists_each_statement_as_stated() -> None:
    p = research_matrix()
    knowledge = (
        DomainKnowledge()
        .with_tiers([["A"], ["B", "C", "D"]])
        .require("D", "B", confidence=0.9)
        .root("C", confidence=0.8)
        .confound("A", "D")
        .restrict_to([("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"), ("B", "D")], confidence=0.7)
    )
    with pytest.warns(UserWarning, match=r"D -> B \(pair B, D: 0\.05\)"):
        result = decode(pair_state(), "exact_map", knowledge=knowledge)
    audit = result.diagnostics["knowledge_audit"]
    assert [e["statement"] for e in audit] == [
        "tiers [A] < [B, C, D]",
        "A <-> D",
        "D -> B",
        "C is a root",
        "edges only on 5 skeleton pairs",
    ]
    by = {e["statement"]: e for e in audit}
    # Tiers: no edge from B, C or D into A.
    assert by["tiers [A] < [B, C, D]"]["probability"] == pytest.approx(
        (1 - p[1, 0]) * (1 - p[2, 0]) * (1 - p[3, 0])
    )
    assert by["tiers [A] < [B, C, D]"]["confidence"] is None
    # The weakest pair part: C -> A at 0.60 leaves 0.40 for "no edge into A from C".
    assert by["tiers [A] < [B, C, D]"]["min_pair_probability"] == pytest.approx(1 - p[2, 0])
    assert by["tiers [A] < [B, C, D]"]["conflict_pairs"] == ()
    assert by["D -> B"]["min_pair_probability"] == pytest.approx(p[3, 1])
    assert by["D -> B"]["conflict_pairs"] == (("B", "D"),)
    assert [e["statement"] for e in result.diagnostics["knowledge_conflicts"]] == ["D -> B"]
    assert by["D -> B"]["probability"] == pytest.approx(p[3, 1])
    assert by["C is a root"]["probability"] == pytest.approx(
        (1 - p[0, 2]) * (1 - p[1, 2]) * (1 - p[3, 2])
    )
    # Only A - C lies outside the skeleton.
    assert by["edges only on 5 skeleton pairs"]["probability"] == pytest.approx(
        1 - p[0, 2] - p[2, 0]
    )
    confound = by["A <-> D"]
    assert confound["probability"] is None and "cannot price" in confound["note"]
    assert confound["min_pair_probability"] is None and confound["conflict_pairs"] == ()
    assert confound["holds"] is True and confound["holds_without_knowledge"] is False
    assert by["tiers [A] < [B, C, D]"]["holds"] is True
    # Without knowledge the cycle edge C -> A is rejected, so the tiers hold there too.
    assert by["D -> B"]["holds_without_knowledge"] is False
    assert result.diagnostics["n_hard_statements"] == 2
    assert result.diagnostics["n_soft_statements"] == 3
    assert result.diagnostics["knowledge_cost"] >= 0.0


def test_knowledge_cost_is_the_score_given_up() -> None:
    probs = pair_state()
    knowledge = DomainKnowledge().forbid("A", "B")
    result = decode(probs, "exact_map", knowledge=knowledge)
    free = decode(probs, "exact_map")
    assert result.diagnostics["knowledge_cost"] == pytest.approx(
        free.diagnostics["objective"] - result.diagnostics["objective"]
    )
    assert result.diagnostics["knowledge_cost"] > 0.0


def test_strong_conflicts_are_listed_and_warned_once() -> None:
    probs = pair_state()
    knowledge = DomainKnowledge().require("D", "B").forbid("A", "B", confidence=0.6).root("A")
    with pytest.warns(UserWarning, match="1 knowledge statement") as caught:
        result = decode(probs, knowledge=knowledge)
    assert len([w for w in caught if "knowledge statement" in str(w.message)]) == 1
    (conflict,) = result.diagnostics["knowledge_conflicts"]
    assert conflict["statement"] == "D -> B"
    assert conflict["probability"] == pytest.approx(0.05)


def test_a_statement_over_many_pairs_conflicts_only_through_a_single_pair() -> None:
    """The rule is per pair: an unlikely whole made of likely parts is no conflict."""
    names = tuple(f"x{i}" for i in range(8))
    values = np.full((8, 8), 0.15)
    np.fill_diagonal(values, 0.0)
    probs = EdgeProbabilities(names, values, kind="pair_state")
    tiers = DomainKnowledge().with_tiers([names[:4], names[4:]], confidence=0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = decode(probs, knowledge=tiers)
    (entry,) = quiet.diagnostics["knowledge_audit"]
    assert entry["probability"] == pytest.approx(0.85**16)  # 0.07: unlikely as a whole
    assert entry["min_pair_probability"] == pytest.approx(0.85)
    assert entry["conflict_pairs"] == () and quiet.diagnostics["knowledge_conflicts"] == ()
    # One cross-tier edge against the order above 0.9 makes it a conflict.
    strong = values.copy()
    strong[5, 1], strong[1, 5] = 0.95, 0.01
    root = DomainKnowledge().root("x1")
    with pytest.warns(UserWarning, match=r"tiers .* \(pair x1, x5: 0\.05\)"):
        loud = decode(EdgeProbabilities(names, strong, kind="pair_state"), knowledge=tiers & root)
    by = {e["kind"]: e for e in loud.diagnostics["knowledge_audit"]}
    assert by["tiers"]["conflict_pairs"] == (("x1", "x5"),)
    assert by["root"]["conflict_pairs"] == (("x1", "x5"),)
    assert by["root"]["min_pair_probability"] == pytest.approx(0.05)
    assert len(loud.diagnostics["knowledge_conflicts"]) == 2


def test_no_warning_and_empty_audit_without_conflicts_or_knowledge() -> None:
    probs = pair_state()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = decode(probs, knowledge=DomainKnowledge().forbid("C", "A"))
        plain = decode(probs)
    assert result.diagnostics["knowledge_conflicts"] == ()
    assert plain.diagnostics["knowledge_audit"] == ()
    assert plain.diagnostics["knowledge_cost"] is None
    assert plain.diagnostics["n_hard_statements"] == plain.diagnostics["n_soft_statements"] == 0
    assert plain.diagnostics["graph_kind"] == "dag"


def test_soft_knowledge_naming_unknown_variables_is_refused() -> None:
    with pytest.raises(ValueError, match="lack"):
        decode(pair_state(), knowledge=DomainKnowledge().root("Z", confidence=0.9))


def test_the_threshold_path_weighs_the_input_odds() -> None:
    # A -> B at 0.45 sits below a 0.5 threshold; a 0.9 requirement multiplies its odds
    # by 9, to 0.88, which passes.
    probs = _matrix(("A", "B"), {("A", "B"): 0.45})
    assert decode(probs, threshold=0.5).graph.edges == frozenset()
    knowledge = DomainKnowledge().require("A", "B", confidence=0.9)
    assert _edges(decode(probs, threshold=0.5, knowledge=knowledge)) == ["A --> B"]
    weak = DomainKnowledge().require("A", "B", confidence=0.52)
    assert decode(probs, threshold=0.5, knowledge=weak).graph.edges == frozenset()
    assert math.isclose(0.45 * 9 / (0.45 * 9 + 0.55), 0.8804, abs_tol=1e-4)
