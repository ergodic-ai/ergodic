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
import pandas as pd
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

from ergodic import (
    DomainKnowledge,
    EdgeProbabilities,
    KnowledgeConflictWarning,
    decode,
    identify_effect,
)
from ergodic.discovery import Decoder
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
EXPECTED_CONFLICTS = pytest.mark.filterwarnings("ignore::ergodic.KnowledgeConflictWarning")


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


@EXPECTED_CONFLICTS
@pytest.mark.parametrize("kind", sorted(STATEMENTS))
@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_confidence_near_one_matches_the_hard_statement(
    kind: str, method: str, kwargs: dict[str, Any]
) -> None:
    probs = pair_state()
    hard = decode(probs, method, knowledge=STATEMENTS[kind](DomainKnowledge(), None), **kwargs)
    soft = decode(probs, method, knowledge=STATEMENTS[kind](DomainKnowledge(), NEAR_ONE), **kwargs)
    assert soft.diagnostics["edge_budget"] == hard.diagnostics["edge_budget"], kind
    if method == "pair_sampler" and kind in ("require", "order"):
        # The sampler draws each pair, then adds the drawn edges in random order and
        # drops any that closes a cycle. A hard required edge goes in first and a
        # hard order is enforced through paths; near 1 the soft forms draw the same
        # states, but a soft required edge can lose to a cycle and a soft order only
        # rules out the direct edge. So: the soft required edge is missing only where
        # a path already runs the other way, and the direct edge against the order
        # never appears.
        for candidate in soft.candidates:
            if kind == "require":
                edge = candidate.get_edge("D", "B")
                if edge is None:
                    assert "D" in candidate.descendants("B")
                else:
                    assert edge.directed_pair() == ("D", "B")
            else:
                assert candidate.get_edge("C", "D") is None or "D" not in candidate.children("C")
        return
    assert soft.candidates == hard.candidates, kind
    assert soft.graph == hard.graph, kind
    if method not in ("pair_sampler", "perturb_and_map"):
        # Each statement changes a deterministic decode, so the test is not vacuous.
        assert hard.graph != decode(probs, method, **kwargs).graph, kind


@EXPECTED_CONFLICTS
@pytest.mark.parametrize("kind", sorted(STATEMENTS))
@pytest.mark.parametrize(("method", "kwargs"), ALL_METHODS)
def test_a_hard_constraint_overrides_any_soft_statement_on_its_pair(
    kind: str, method: str, kwargs: dict[str, Any]
) -> None:
    """Soft weight on what a hard constraint settles changes nothing, budget included."""
    probs = pair_state()
    hard = STATEMENTS[kind](DomainKnowledge(), None)
    # Every soft statement aimed at the pairs and directions the hard one settles.
    against = DomainKnowledge()
    for a, b in hard.forbidden:
        against = against.require(a, b, confidence=0.999)
    for a, b in hard.required:
        against = against.require(b, a, confidence=0.999).forbid(a, b, confidence=0.999)
    for a, b in hard.precedence():
        against = against.require(b, a, confidence=0.999)
    for x in hard.roots:
        for v in set(NAMES) - {x}:
            against = against.require(v, x, confidence=0.999)
    for x in hard.sinks:
        for v in set(NAMES) - {x}:
            against = against.require(x, v, confidence=0.999)
    if hard.allowed_adjacencies is not None:
        outside = [
            (a, b)
            for i, a in enumerate(NAMES)
            for b in NAMES[i + 1 :]
            if frozenset((a, b)) not in hard.allowed_adjacencies
        ]
        for a, b in outside:
            against = against.require(a, b, confidence=0.999).require(b, a, confidence=0.999)
    assert against.soft
    alone = decode(probs, method, knowledge=hard, **kwargs)
    both = decode(probs, method, knowledge=hard & against, **kwargs)
    assert both.candidates == alone.candidates, kind
    assert both.diagnostics["edge_budget"] == alone.diagnostics["edge_budget"], kind


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


def test_the_audit_lists_each_statement() -> None:
    p = research_matrix()
    knowledge = (
        DomainKnowledge()
        .order("A", "B")
        .confound("A", "D")
        .with_tiers([["A"], ["B", "C", "D"]], confidence=0.9)
        .require("D", "B", confidence=0.9)
        .root("C", confidence=0.8)
        .restrict_to([("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"), ("B", "D")], confidence=0.7)
    )
    with pytest.warns(KnowledgeConflictWarning, match=r"D -> B \(pair B, D: 0\.05\)"):
        result = decode(pair_state(), "exact_map", knowledge=knowledge)
    audit = result.diagnostics["knowledge_audit"]
    assert [e["statement"] for e in audit] == [
        "A before B",
        "A <-> D",
        "D -> B",
        "tiers [A] < [B, C, D]",
        "C is a root",
        "edges only on 5 skeleton pairs",
    ]
    by = {e["statement"]: e for e in audit}
    tiers = by["tiers [A] < [B, C, D]"]
    # Tiers: no edge from B, C or D into A, one pair at a time.
    assert tiers["probability"] == pytest.approx((1 - p[1, 0]) * (1 - p[2, 0]) * (1 - p[3, 0]))
    assert tiers["confidence"] == 0.9
    # The weakest pair part: C -> A at 0.60 leaves 0.40 for "no edge into A from C".
    assert tiers["min_pair_probability"] == pytest.approx(1 - p[2, 0])
    assert tiers["conflict_pairs"] == ()
    assert "directed paths" in tiers["note"] and "direct edges" in tiers["note"]
    assert by["A before B"]["confidence"] is None
    assert by["A before B"]["probability"] == pytest.approx(1 - p[1, 0])
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
    assert tiers["holds"] is True
    assert by["D -> B"]["holds_without_knowledge"] is False
    assert result.diagnostics["n_hard_statements"] == 2
    assert result.diagnostics["n_soft_statements"] == 4
    assert result.diagnostics["knowledge_cost"] >= 0.0


@pytest.mark.parametrize("confidence", [None, 0.9])
def test_an_order_holds_or_fails_through_paths_not_only_direct_edges(
    confidence: float | None,
) -> None:
    """The model says c -> b -> a; "a before c" fails there through the chain."""
    probs = _matrix(("a", "b", "c"), {("c", "b"): 0.9, ("b", "a"): 0.9})
    knowledge = DomainKnowledge().order("a", "c", confidence=confidence)
    plain = decode(probs)
    assert _edges(plain) == ["a <-- b", "b <-- c"]
    result = decode(probs, knowledge=knowledge)
    (entry,) = result.diagnostics["knowledge_audit"]
    assert entry["holds_without_knowledge"] is False  # c is an ancestor of a
    # The probability reads the direct edge c -> a only, which the model makes unlikely.
    assert entry["probability"] == pytest.approx(1 - probs.to_pair_states().values[2, 0])
    assert entry["probability"] > 0.9
    assert entry["holds"] is ("a" not in result.graph.descendants("c"))
    if confidence is None:
        assert entry["holds"] is True


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
    with pytest.warns(KnowledgeConflictWarning, match="1 knowledge statement") as caught:
        result = decode(probs, knowledge=knowledge)
    assert len([w for w in caught if w.category is KnowledgeConflictWarning]) == 1
    # The warning points at this call, not inside ergodic.
    assert caught[0].filename == __file__
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
    with pytest.warns(KnowledgeConflictWarning, match=r"tiers .* \(pair x1, x5: 0\.05\)"):
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


# -- review follow-ups (API-151 independent review) -------------------------------------


def test_audit_false_skips_the_knowledge_free_decode_and_the_warning() -> None:
    probs = pair_state()
    knowledge = DomainKnowledge().require("D", "B")  # a strong conflict
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = decode(probs, "exact_map", knowledge=knowledge, audit=False)
        configured = Decoder("exact_map", audit=False).decode(probs, knowledge=knowledge)
    with pytest.warns(KnowledgeConflictWarning):
        audited = decode(probs, "exact_map", knowledge=knowledge)
    assert quiet.graph == audited.graph == configured.graph
    assert quiet.diagnostics["knowledge_audit"] == ()
    assert quiet.diagnostics["knowledge_cost"] is None
    assert quiet.diagnostics["knowledge_conflicts"] == ()
    assert quiet.diagnostics["knowledge_audited"] is False
    assert audited.diagnostics["knowledge_audited"] is True
    # Skipping the audit leaves the constrained decode itself unchanged.
    assert quiet.diagnostics["search_expansions"] == audited.diagnostics["search_expansions"]
    with pytest.raises(TypeError, match="audit"):
        decode(probs, knowledge=knowledge, audit="no")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="audit"):
        Decoder("greedy", audit=1)  # type: ignore[arg-type]


def test_the_repr_counts_observed_variables_and_names_latent_confounders() -> None:
    probs = _matrix(("X", "Y", "Z"), {("X", "Y"): 0.9})
    result = decode(probs, knowledge=DomainKnowledge().confound("X", "Y"))
    assert repr(result) == (
        "DecodeResult(decode:greedy_v1: decoded admg, 3 nodes, 1 edges, 1 latent confounder)"
    )


def test_confounded_and_neither_causes_the_other_decodes_to_a_bidirected_edge() -> None:
    # The model would draw X -> Y; the knowledge says only a hidden cause joins them.
    probs = _matrix(("Z", "X", "Y"), {("X", "Y"): 0.9, ("Z", "X"): 0.9})
    knowledge = DomainKnowledge().confound("X", "Y").forbid("X", "Y").forbid("Y", "X")
    assert knowledge.conflicts() == []
    for method, kwargs in ALL_METHODS:
        result = decode(probs, method, knowledge=knowledge, audit=False, **kwargs)
        assert result.graph.latent == frozenset()
        assert result.graph.get_edge("X", "Y").is_bidirected  # type: ignore[union-attr]
        assert knowledge.violations(result.graph) == []
    graph = decode(probs, knowledge=knowledge, audit=False).graph
    assert _edges_of(graph) == ["X <-- Z", "X <-> Y"]
    # X has no effect on Y here: no directed path, the zero-effect "null" strategy.
    assert identify_effect(graph, "X", "Y").strategy == "null"
    # Z's effect on X has no back-door path: plain adjustment.
    assert identify_effect(graph, "Z", "X").strategy == "adjustment"


def _edges_of(graph: Any) -> list[str]:
    return sorted(edge.glyph() for edge in graph.edges)


def test_soft_knowledge_is_refused_by_the_searches_not_dropped() -> None:
    from ergodic import discover, tabular
    from ergodic.discovery import PC, DSeparationOracle, bootstrap

    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    data = tabular(pd.DataFrame({"x": x, "y": x + rng.normal(size=200), "z": rng.normal(size=200)}))
    knowledge = DomainKnowledge().require("x", "y").root("z", confidence=0.99)
    for method in ("pc", "ges", "fci", "lingam", "mmpc", "boss", "glasso"):
        with pytest.raises(ValueError, match=r"reads hard knowledge only.*z is a root \(0\.99\)"):
            discover(data, method=method, knowledge=knowledge)
    with pytest.raises(ValueError, match="bootstrap reads hard knowledge only"):
        bootstrap(data, n_bootstrap=2, knowledge=knowledge)
    with pytest.raises(ValueError, match=r"PC reads hard knowledge only.*knowledge\.hard\(\)"):
        PC(DSeparationOracle(dag(["x -> y"], nodes=["z"]))).fit(knowledge=knowledge)
    # The hard part runs.
    result = PC(DSeparationOracle(dag(["x -> y"], nodes=["z"]))).fit(knowledge=knowledge.hard())
    assert result.graph.get_edge("x", "y") is not None


@pytest.mark.parametrize("name", ["FCI", "RFCI"])
def test_the_latent_searches_keep_a_confounded_pair_neither_side_causes(name: str) -> None:
    from ergodic import discovery

    truth = dag(["L -> X", "L -> Y", "Z -> X"], latent=["L"])
    knowledge = DomainKnowledge().confound("X", "Y").forbid("X", "Y").forbid("Y", "X")
    search = getattr(discovery, name)(test=discovery.DSeparationOracle(truth))
    graph = search.fit(knowledge=knowledge).graph
    assert graph.get_edge("X", "Y").is_bidirected
    assert knowledge.violations(graph) == []


# -- versioned cardinality decoders (DEC-079: an id never changes its results) -----------

# Recorded with ergodic 0.2.1 (main 3caba1a) on the research matrix with
# forbid_adjacency(A, B), root(C) and require(D, B): its expected_count and soft_f1.
V021_BUDGET = 4
V021_EDGES = ["A <-- C", "B <-- C", "B <-- D", "C --> D"]


@EXPECTED_CONFLICTS
@pytest.mark.parametrize("rule", ["expected_count", "soft_f1"])
def test_v1_reproduces_0_2_1_under_hard_knowledge_and_v2_settles_the_budget(rule: str) -> None:
    knowledge = DomainKnowledge().forbid_adjacency("A", "B").root("C").require("D", "B")
    v1 = decode(pair_state(), f"{rule}_v1", knowledge=knowledge)
    assert v1.diagnostics["edge_budget"] == V021_BUDGET
    assert _edges(v1) == V021_EDGES
    v2 = decode(pair_state(), f"{rule}_v2", knowledge=knowledge)
    # A -> B (0.80) and the edges into C no longer count toward the budget.
    assert v2.diagnostics["edge_budget"] == 3
    assert _edges(v2) == ["A <-- C", "B <-- D", "C --> D"]
    assert decode(pair_state(), rule, knowledge=knowledge).graph == v2.graph


@pytest.mark.parametrize("rule", ["expected_count", "soft_f1"])
def test_v1_and_v2_agree_without_knowledge(rule: str) -> None:
    rng = np.random.default_rng(21)
    for size in (3, 5, 7):
        for probs in (
            EdgeProbabilities(NAMES_7[:size], random_independent(size, rng)),
            EdgeProbabilities(NAMES_7[:size], random_pair_state(size, rng), kind="pair_state"),
        ):
            v1 = decode(probs, f"{rule}_v1")
            v2 = decode(probs, f"{rule}_v2")
            assert v1.graph == v2.graph
            assert v1.diagnostics["edge_budget"] == v2.diagnostics["edge_budget"]


NAMES_7 = tuple("ABCDEFG")
