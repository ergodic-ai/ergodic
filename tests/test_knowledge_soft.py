# SPDX-License-Identifier: Apache-2.0
"""Soft knowledge statements: storage, validation, log odds and the log prior (API-151)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ergodic.graph import admg, dag, mixed
from ergodic.knowledge import DomainKnowledge, KnowledgeStatement

L8 = math.log(0.8 / 0.2)  # the log ratio of a 0.8 statement


# -- validation and storage -----------------------------------------------------


@pytest.mark.parametrize("bad", [0.49, 1.01, float("nan"), -1.0, True, False])
def test_confidences_outside_half_to_one_are_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        DomainKnowledge().require("a", "b", confidence=bad)


def test_a_confidence_below_one_half_says_to_state_the_opposite() -> None:
    with pytest.raises(ValueError, match="state the opposite"):
        DomainKnowledge().forbid("a", "b", confidence=0.2)


def test_a_non_numeric_confidence_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="confidence"):
        DomainKnowledge().require("a", "b", confidence="high")  # type: ignore[arg-type]


def test_none_and_one_store_the_hard_constraint_exactly_as_before() -> None:
    plain = DomainKnowledge().require("a", "b").root("c").with_tiers([["a"], ["b"]])
    for level in (None, 1.0, 1):
        same = (
            DomainKnowledge()
            .require("a", "b", confidence=level)
            .root("c", confidence=level)
            .with_tiers([["a"], ["b"]], confidence=level)
        )
        assert same == plain
        assert hash(same) == hash(plain)
        assert repr(same) == repr(plain)
        assert same.soft == frozenset()


def test_hard_only_repr_and_equality_are_unchanged() -> None:
    knowledge = (
        DomainKnowledge()
        .with_tiers([["asia", "smoke"], ["tub", "lung", "bronc"], ["either"], ["xray", "dysp"]])
        .require("asia", "tub")
    )
    assert repr(knowledge) == (
        "DomainKnowledge(vars=8, required=1, forbidden=0, precedes=23, roots=0, sinks=0, "
        "confounded=0)"
    )
    # The statement log takes no part in equality: built another way, still equal.
    rebuilt = DomainKnowledge(
        variables=knowledge.variables,
        required=knowledge.required,
        precedes=knowledge.precedes,
    )
    assert rebuilt == knowledge
    assert hash(rebuilt) == hash(knowledge)


def test_soft_statements_are_stored_apart_and_counted_in_the_repr() -> None:
    knowledge = DomainKnowledge().require("a", "b", confidence=0.8).root("c", "d", confidence=0.9)
    assert knowledge.required == frozenset() and knowledge.roots == frozenset()
    assert knowledge.soft == {
        KnowledgeStatement("require", ("a", "b"), 0.8),
        KnowledgeStatement("root", ("c",), 0.9),
        KnowledgeStatement("root", ("d",), 0.9),
    }
    assert knowledge.variables == {"a", "b", "c", "d"}
    assert repr(knowledge).endswith("confounded=0, soft=3)")
    assert knowledge != DomainKnowledge().require("a", "b", confidence=0.7).root(
        "c", "d", confidence=0.9
    )


def test_one_half_is_stored_but_neutral() -> None:
    knowledge = DomainKnowledge().forbid("a", "b", confidence=0.5)
    assert len(knowledge.soft) == 1
    assert not knowledge.log_odds(["a", "b"]).any()


def test_confound_is_hard_only() -> None:
    with pytest.raises(ValueError, match="hard only"):
        DomainKnowledge().confound("a", "b", confidence=0.8)
    assert DomainKnowledge().confound("a", "b", confidence=1.0) == DomainKnowledge().confound(
        "a", "b"
    )


def test_a_soft_pair_statement_needs_two_variables() -> None:
    with pytest.raises(ValueError, match="distinct"):
        DomainKnowledge().order("a", "a", confidence=0.7)


def test_and_pools_soft_statements_and_keeps_the_hard_ones() -> None:
    left = DomainKnowledge().require("a", "b").forbid("b", "c", confidence=0.7)
    right = DomainKnowledge().sink("c", confidence=0.9).forbid("b", "c", confidence=0.7)
    both = left & right
    assert both.required == {("a", "b")}
    assert both.soft == left.soft | right.soft
    assert len(both.soft) == 2
    assert both.variables == {"a", "b", "c"}
    assert both == right & left


def test_soft_statements_leave_every_hard_projection_alone() -> None:
    soft = (
        DomainKnowledge()
        .require("a", "b", confidence=0.9)
        .forbid("b", "a", confidence=0.9)
        .root("a", confidence=0.9)
        .restrict_to([("a", "c")], confidence=0.9)
    )
    assert soft.conflicts() == []
    assert soft.violations(dag(["b -> a"], nodes=["c"])) == []
    assert soft.permits_edge("b", "a") and not soft.requires_edge("a", "b")
    mask, _ = soft.edge_mask(["a", "b", "c"])
    assert mask.sum() == 6
    graph = mixed(["a -- b"], nodes=["c"])
    assert soft.apply_to(graph) == graph


def test_statements_list_what_was_said_once_each() -> None:
    knowledge = (
        DomainKnowledge()
        .with_tiers([["x", "a"], ["b"]])
        .forbid_between(["a"], ["c", "b"])
        .root("r1", "r2")
        .confound("a", "c")
        .restrict_to([("a", "b"), ("b", "c")])
        .order("b", "c", confidence=0.6)
        .forbid_adjacency("a", "x", confidence=0.9)
    )
    texts = [str(s) for s in knowledge.statements()]
    assert texts == [
        "tiers [a, x] < [b]",
        "no edge from [a] into [b, c]",
        "r1 is a root",
        "r2 is a root",
        "a <-> c",
        "edges only on 2 skeleton pairs",
        "no edge between a and x",
        "b before c",
    ]
    assert [s.confidence for s in knowledge.statements()][-2:] == [0.9, 0.6]


def test_statements_cover_hard_content_set_without_a_builder() -> None:
    knowledge = DomainKnowledge(
        required=frozenset({("a", "b")}),
        forbidden=frozenset({("c", "d"), ("d", "c"), ("a", "c")}),
        precedes=frozenset({("a", "d")}),
        confounded=frozenset({frozenset({"b", "d"})}),
        allowed_adjacencies=frozenset({frozenset({"a", "b"})}),
    )
    assert [str(s) for s in knowledge.statements()] == [
        "a -> b",
        "no a -> c",
        "no edge between c and d",
        "a before d",
        "b <-> d",
        "edges only on 1 skeleton pair",
    ]


def test_a_confounded_pair_banned_from_adjacency_is_a_conflict() -> None:
    knowledge = DomainKnowledge().confound("a", "b").forbid_adjacency("b", "a")
    assert any("banned from being adjacent" in c for c in knowledge.conflicts())
    assert DomainKnowledge().confound("a", "b").forbid("a", "b").conflicts() == []


# -- log odds -----------------------------------------------------------------------


def _odds(knowledge: DomainKnowledge, names: str) -> dict[str, float]:
    matrix = knowledge.log_odds(list(names))
    return {
        f"{a}{b}": float(matrix[i, j])
        for i, a in enumerate(names)
        for j, b in enumerate(names)
        if matrix[i, j] != 0.0
    }


def test_log_odds_of_each_statement_type() -> None:
    k = DomainKnowledge()
    assert _odds(k.require("a", "b", confidence=0.8), "abc") == {"ab": pytest.approx(L8)}
    assert _odds(k.forbid("a", "b", confidence=0.8), "abc") == {"ab": pytest.approx(-L8)}
    assert _odds(k.forbid_adjacency("a", "b", confidence=0.8), "abc") == {
        "ab": pytest.approx(-L8),
        "ba": pytest.approx(-L8),
    }
    assert _odds(k.order("a", "b", confidence=0.8), "abc") == {"ba": pytest.approx(-L8)}
    assert _odds(k.root("a", confidence=0.8), "abc") == {
        "ba": pytest.approx(-L8),
        "ca": pytest.approx(-L8),
    }
    assert _odds(k.sink("a", confidence=0.8), "abc") == {
        "ab": pytest.approx(-L8),
        "ac": pytest.approx(-L8),
    }
    # Tiers: every edge from a later tier into an earlier one; nothing within a tier.
    assert _odds(k.with_tiers([["a"], ["b", "c"]], confidence=0.8), "abc") == {
        "ba": pytest.approx(-L8),
        "ca": pytest.approx(-L8),
    }
    assert _odds(k.forbid_between(["a", "b"], ["c"], confidence=0.8), "abc") == {
        "ac": pytest.approx(-L8),
        "bc": pytest.approx(-L8),
    }
    # restrict_to: both directions of every pair outside the skeleton.
    assert _odds(k.restrict_to([("a", "b")], confidence=0.8), "abc") == {
        "ac": pytest.approx(-L8),
        "ca": pytest.approx(-L8),
        "bc": pytest.approx(-L8),
        "cb": pytest.approx(-L8),
    }


def test_log_odds_add_up_and_ignore_hard_constraints() -> None:
    knowledge = (
        DomainKnowledge()
        .require("a", "b", confidence=0.8)
        .forbid("a", "b", confidence=0.6)
        .require("a", "b")  # hard: enforced elsewhere, not weighed
    )
    matrix = knowledge.log_odds(["b", "a"])
    assert matrix[1, 0] == pytest.approx(L8 - math.log(0.6 / 0.4))
    assert matrix[0, 1] == 0.0 and np.all(np.diag(matrix) == 0.0)


def test_soft_order_is_not_transitive() -> None:
    knowledge = DomainKnowledge().order("a", "b", confidence=0.9).order("b", "c", confidence=0.9)
    odds = _odds(knowledge, "abc")
    assert set(odds) == {"ba", "cb"}  # nothing on c -> a


def test_log_odds_refuses_unknown_names() -> None:
    with pytest.raises(ValueError, match="outside"):
        DomainKnowledge().require("a", "z", confidence=0.8).log_odds(["a", "b"])


# -- log prior and energy -----------------------------------------------------------


def test_log_prior_sums_log_c_or_log_one_minus_c_per_pair() -> None:
    knowledge = (
        DomainKnowledge()
        .require("a", "b", confidence=0.8)
        .root("c", confidence=0.9)
        .with_tiers([["a"], ["b", "c"]], confidence=0.7)
    )
    graph = dag(["a -> b"], nodes=["c"])
    expected = (
        math.log(0.8)  # a -> b present
        + 2 * math.log(0.9)  # nothing into c
        + 2 * math.log(0.7)  # nothing from b or c into a
    )
    assert knowledge.log_prior(graph) == pytest.approx(expected)
    assert knowledge.energy(graph) == pytest.approx(-expected)
    reversed_graph = dag(["b -> a", "a -> c"])
    # require fails; a -> c breaks the root, b -> c does not; b -> a breaks the tiers.
    expected = math.log(0.2) + math.log(0.1) + math.log(0.9) + math.log(0.3) + math.log(0.7)
    assert knowledge.log_prior(reversed_graph) == pytest.approx(expected)


def test_log_prior_differences_match_the_log_odds() -> None:
    knowledge = (
        DomainKnowledge()
        .forbid_adjacency("a", "b", confidence=0.75)
        .sink("b", confidence=0.6)
        .restrict_to([("a", "c")], confidence=0.85)
    )
    names = ["a", "b", "c"]
    odds = knowledge.log_odds(names)
    empty = dag([], nodes=names)
    for i, source in enumerate(names):
        for j, target in enumerate(names):
            if i == j:
                continue
            one = dag([(source, target)], nodes=names)
            gap = knowledge.log_prior(one) - knowledge.log_prior(empty)
            assert gap == pytest.approx(odds[i, j])


def test_log_prior_is_minus_infinity_off_the_hard_constraints() -> None:
    knowledge = DomainKnowledge().forbid("a", "b").require("b", "c", confidence=0.9)
    assert knowledge.log_prior(dag(["a -> b", "b -> c"])) == -math.inf
    assert knowledge.energy(dag(["a -> b", "b -> c"])) == math.inf
    assert knowledge.log_prior(dag(["b -> c"], nodes=["a"])) == pytest.approx(math.log(0.9))


def test_log_prior_reads_admgs_and_refuses_unoriented_edges() -> None:
    knowledge = DomainKnowledge().confound("a", "b").forbid("a", "b", confidence=0.8)
    assert knowledge.log_prior(admg(["a <-> b"])) == pytest.approx(math.log(0.8))
    assert knowledge.log_prior(dag(["a -> b"])) == -math.inf  # no bidirected edge
    with pytest.raises(ValueError, match="orient these first"):
        knowledge.log_prior(mixed(["a -- b"]))
    with pytest.raises(ValueError, match="outside"):
        DomainKnowledge().root("z", confidence=0.9).log_prior(dag(["a -> b"]))


def test_a_latent_parent_reads_as_hidden_confounding() -> None:
    bow = admg([("L", "a"), ("L", "b"), ("a", "b")], latent=["L"])
    knowledge = DomainKnowledge().confound("a", "b").root("a").restrict_to([("a", "b")])
    assert knowledge.violations(bow) == []
    assert knowledge.log_prior(bow) == 0.0
    soft = DomainKnowledge().root("a", confidence=0.9)
    # The latent node is not a variable: root(a) speaks about b -> a only.
    assert soft.log_prior(bow) == pytest.approx(math.log(0.9))
