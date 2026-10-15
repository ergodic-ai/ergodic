# SPDX-License-Identifier: Apache-2.0
"""Tests for DomainKnowledge: builders, composition, validation, and projections."""

from __future__ import annotations

import pytest

from ergodic.graph import admg, dag, mixed
from ergodic.knowledge import DomainKnowledge, InconsistentKnowledgeError

# -- builders and immutability -----------------------------------------------


def test_builders_accumulate_and_stay_immutable() -> None:
    base = DomainKnowledge()
    grown = base.require("X", "Y").forbid("Z", "Y")
    assert base.required == frozenset()
    assert ("X", "Y") in grown.required
    assert ("Z", "Y") in grown.forbidden
    assert {"X", "Y", "Z"} <= grown.variables


def test_tiers_become_precedence() -> None:
    knowledge = DomainKnowledge().with_tiers([["A"], ["B", "C"], ["D"]])
    assert ("A", "B") in knowledge.precedes
    assert ("B", "D") in knowledge.precedes
    assert ("B", "A") not in knowledge.precedes


def test_confound_rejects_identical_pair() -> None:
    with pytest.raises(ValueError, match="distinct"):
        DomainKnowledge().confound("A", "A")


# -- permits -----------------------------------------------------------------


def test_permits_edge_respects_constraints() -> None:
    knowledge = DomainKnowledge().forbid("A", "B").with_tiers([["C"], ["D"]]).root("R").sink("S")
    assert not knowledge.permits_edge("A", "B")
    assert knowledge.permits_edge("B", "A")
    assert not knowledge.permits_edge("D", "C")  # D is later than C
    assert knowledge.permits_edge("C", "D")
    assert not knowledge.permits_edge("X", "R")  # nothing enters a root
    assert not knowledge.permits_edge("S", "X")  # nothing leaves a sink


def test_precedence_is_transitive() -> None:
    knowledge = DomainKnowledge().order("A", "B").order("B", "C")
    assert not knowledge.permits_edge("C", "A")  # C is after A through B
    assert knowledge.permits_edge("A", "C")
    assert not knowledge.is_consistent(dag(["C -> A"], nodes=["A", "B", "C"]))
    mask, labels = knowledge.edge_mask()
    assert labels == ["A", "B", "C"]
    assert not mask[2, 0]  # C -> A blocked through the closure
    assert mask[0, 2]


def test_precedence_projection_exposes_the_closure() -> None:
    knowledge = DomainKnowledge().order("A", "B").order("B", "C")
    assert knowledge.precedence() == frozenset({("A", "B"), ("B", "C"), ("A", "C")})
    assert DomainKnowledge().precedence() == frozenset()


# -- conflict detection ------------------------------------------------------


def test_required_and_forbidden_conflict() -> None:
    knowledge = DomainKnowledge().require("X", "Y").forbid("X", "Y")
    assert knowledge.conflicts()
    with pytest.raises(InconsistentKnowledgeError):
        knowledge.validate()


def test_required_cycle_conflict() -> None:
    knowledge = DomainKnowledge().require("X", "Y").require("Y", "Z").require("Z", "X")
    assert any("cycle" in message for message in knowledge.conflicts())


def test_precedence_cycle_conflict() -> None:
    knowledge = DomainKnowledge().order("A", "B").order("B", "A")
    assert any("cycle" in message for message in knowledge.conflicts())


def test_required_into_root_conflict() -> None:
    knowledge = DomainKnowledge().require("X", "Y").root("Y")
    assert any("root" in message for message in knowledge.conflicts())


def test_required_against_order_conflict() -> None:
    knowledge = DomainKnowledge().require("X", "Y").order("Y", "X")
    assert knowledge.conflicts()


def test_required_and_precedence_union_cycle_conflict() -> None:
    knowledge = DomainKnowledge().require("U", "V").require("V", "W").order("W", "U")
    assert any("cycle" in message for message in knowledge.conflicts())


def test_deep_required_chain_validates_without_recursion_error() -> None:
    knowledge = DomainKnowledge()
    for i in range(2000):
        knowledge = knowledge.require(f"N{i}", f"N{i + 1}")
    assert knowledge.conflicts() == []


def test_empty_knowledge_is_consistent() -> None:
    DomainKnowledge().validate()


# -- composition -------------------------------------------------------------


def test_composition_unions_constraints() -> None:
    left = DomainKnowledge().forbid("A", "B")
    right = DomainKnowledge().require("C", "D")
    both = left & right
    assert ("A", "B") in both.forbidden
    assert ("C", "D") in both.required
    assert {"A", "B", "C", "D"} <= both.variables


# -- consistency against a graph ---------------------------------------------


def test_is_consistent_checks_a_graph() -> None:
    graph = dag(["X -> Y", "Y -> Z"])
    good = DomainKnowledge().require("X", "Y").forbid("Z", "Y").root("X")
    assert good.is_consistent(graph)

    missing = DomainKnowledge().require("X", "Z")
    assert not missing.is_consistent(graph)
    assert any("required" in v for v in missing.violations(graph))


def test_root_violation_detected() -> None:
    graph = dag(["A -> X"])
    assert not DomainKnowledge().root("X").is_consistent(graph)


def test_confounded_pair_needs_bidirected_edge() -> None:
    knowledge = DomainKnowledge().confound("X", "Y")
    assert knowledge.is_consistent(admg(["X <-> Y"]))
    assert not knowledge.is_consistent(dag(["X -> Y"]))


def test_forbid_adjacency_bans_every_edge_type() -> None:
    knowledge = DomainKnowledge().forbid_adjacency("A", "B")
    assert not knowledge.is_consistent(dag(["A -> B"]))
    assert not knowledge.is_consistent(admg(["A <-> B"]))
    assert not knowledge.is_consistent(mixed(["A -- B"]))
    assert knowledge.apply_to(admg(["A <-> B"])).get_edge("A", "B") is None
    assert knowledge.apply_to(mixed(["A -- B"])).get_edge("A", "B") is None


def test_single_direction_forbid_stays_directed_only() -> None:
    knowledge = DomainKnowledge().forbid("A", "B")
    assert knowledge.is_consistent(admg(["A <-> B"]))
    assert knowledge.is_consistent(mixed(["A -- B"]))
    oriented = knowledge.apply_to(mixed(["A -- B"]))
    edge = oriented.get_edge("A", "B")
    assert edge is not None
    assert edge.directed_pair() == ("B", "A")  # only the allowed direction remains


# -- projections -------------------------------------------------------------


def test_apply_to_orients_and_enforces() -> None:
    graph = mixed(["X -- Y", "A -- B"])
    knowledge = DomainKnowledge().require("A", "B").forbid("Y", "X")
    result = knowledge.apply_to(graph)
    assert result.get_edge("A", "B").directed_pair() == ("A", "B")  # type: ignore[union-attr]
    assert result.get_edge("X", "Y").directed_pair() == ("X", "Y")  # type: ignore[union-attr]


def test_edge_mask_shape_and_values() -> None:
    knowledge = DomainKnowledge().with_variables("A", "B", "C").forbid("A", "B").root("C")
    mask, labels = knowledge.edge_mask()
    assert labels == ["A", "B", "C"]
    assert mask.shape == (3, 3)
    assert not mask[0, 1]  # A -> B forbidden
    assert not mask[0, 2]  # nothing enters root C
    assert mask[1, 0]  # B -> A allowed
    assert not mask.diagonal().any()


# -- skeleton restriction ------------------------------------------------------


def test_restrict_to_from_graph_and_from_pairs_agree() -> None:
    skeleton = mixed(["A -> B", "B <-> C", "C -- D"])  # any mark kind is an adjacency
    from_graph = DomainKnowledge().restrict_to(skeleton)
    from_pairs = DomainKnowledge().restrict_to([("B", "A"), ("B", "C"), ("D", "C")])
    expected = frozenset({frozenset({"A", "B"}), frozenset({"B", "C"}), frozenset({"C", "D"})})
    assert from_graph.allowed_adjacencies == expected
    assert from_pairs.allowed_adjacencies == expected
    assert {"A", "B", "C", "D"} <= from_graph.variables
    assert {"A", "B", "C", "D"} <= from_pairs.variables


def test_restrict_to_returns_a_new_object() -> None:
    base = DomainKnowledge().require("A", "B")
    restricted = base.restrict_to([("A", "B")])
    assert base.allowed_adjacencies is None
    assert restricted.allowed_adjacencies == frozenset({frozenset({"A", "B"})})
    assert restricted.required == base.required


def test_restrict_to_rejects_self_pairs() -> None:
    with pytest.raises(ValueError, match="distinct"):
        DomainKnowledge().restrict_to([("A", "A")])


def test_restrict_to_gates_permits_edge_in_both_directions() -> None:
    knowledge = DomainKnowledge().restrict_to([("A", "B")])
    assert knowledge.permits_edge("A", "B")
    assert knowledge.permits_edge("B", "A")
    assert not knowledge.permits_edge("A", "C")
    assert not knowledge.permits_edge("C", "A")


def test_restrict_to_bans_every_edge_form_outside_the_list() -> None:
    knowledge = DomainKnowledge().restrict_to([("A", "B")])
    assert knowledge.is_consistent(dag(["A -> B"]))
    assert not knowledge.is_consistent(dag(["A -> C"], nodes=["A", "B", "C"]))
    assert not knowledge.is_consistent(admg(["A <-> C"]))
    assert not knowledge.is_consistent(mixed(["A -- C"]))
    messages = knowledge.violations(dag(["A -> B", "B -> C"]))
    assert any("outside the skeleton restriction" in m for m in messages)


def test_restrict_to_apply_to_drops_outside_edges() -> None:
    knowledge = DomainKnowledge().restrict_to([("A", "B")])
    graph = mixed(["A -> B", "B <-> C", "C -- D"])
    pruned = knowledge.apply_to(graph)
    assert pruned.get_edge("A", "B") is not None
    assert pruned.get_edge("B", "C") is None
    assert pruned.get_edge("C", "D") is None


def test_restrict_to_shows_in_edge_mask() -> None:
    knowledge = DomainKnowledge().with_variables("A", "B", "C").restrict_to([("A", "B")])
    mask, labels = knowledge.edge_mask()
    assert labels == ["A", "B", "C"]
    assert mask[0, 1] and mask[1, 0]  # the allowed pair, both directions
    assert not mask[0, 2] and not mask[2, 0]
    assert not mask[1, 2] and not mask[2, 1]


def test_required_edge_outside_skeleton_is_a_conflict() -> None:
    knowledge = DomainKnowledge().require("X", "Y").restrict_to([("A", "B")])
    assert any("required edge X->Y" in m for m in knowledge.conflicts())
    with pytest.raises(InconsistentKnowledgeError):
        knowledge.validate()
    inside = DomainKnowledge().require("A", "B").restrict_to([("A", "B")])
    assert inside.conflicts() == []


def test_confounded_pair_outside_skeleton_is_a_conflict() -> None:
    knowledge = DomainKnowledge().confound("X", "Y").restrict_to([("A", "B")])
    assert any("confounded pair X,Y" in m for m in knowledge.conflicts())
    inside = DomainKnowledge().confound("A", "B").restrict_to([("A", "B")])
    assert inside.conflicts() == []


def test_restriction_composes_by_intersection() -> None:
    left = DomainKnowledge().restrict_to([("A", "B"), ("B", "C")])
    right = DomainKnowledge().restrict_to([("B", "C"), ("C", "D")])
    both = left & right
    assert both.allowed_adjacencies == frozenset({frozenset({"B", "C"})})

    open_side = DomainKnowledge().forbid("A", "B")
    assert (open_side & left).allowed_adjacencies == left.allowed_adjacencies
    assert (left & open_side).allowed_adjacencies == left.allowed_adjacencies
    assert (open_side & open_side).allowed_adjacencies is None


def test_composed_restriction_still_validates() -> None:
    left = DomainKnowledge().require("A", "B").restrict_to([("A", "B"), ("B", "C")])
    right = DomainKnowledge().restrict_to([("B", "C")])
    assert left.conflicts() == []
    composed = left & right
    assert any("required edge A->B" in m for m in composed.conflicts())


def test_restricting_twice_intersects() -> None:
    knowledge = DomainKnowledge().restrict_to([("A", "B"), ("B", "C")])
    narrowed = knowledge.restrict_to([("B", "C"), ("C", "D")])
    assert narrowed.allowed_adjacencies == frozenset({frozenset({"B", "C"})})
    assert knowledge.allowed_adjacencies == frozenset(
        {frozenset({"A", "B"}), frozenset({"B", "C"})}
    )


def test_empty_restriction_allows_nothing() -> None:
    knowledge = DomainKnowledge().restrict_to([])
    assert knowledge.allowed_adjacencies == frozenset()
    assert not knowledge.permits_edge("A", "B")
    assert knowledge.is_consistent(dag([], nodes=["A", "B"]))
    assert not knowledge.is_consistent(dag(["A -> B"]))


def test_tiers_and_restriction_work_together() -> None:
    knowledge = (
        DomainKnowledge().with_tiers([["A"], ["B"], ["C"]]).restrict_to([("A", "B"), ("B", "C")])
    )
    assert knowledge.permits_edge("A", "B")
    assert not knowledge.permits_edge("B", "A")  # blocked by the tiers
    assert not knowledge.permits_edge("A", "C")  # blocked by the skeleton
    oriented = knowledge.apply_to(mixed(["A -- B", "B -- C", "A -- C"]))
    assert oriented.get_edge("A", "C") is None
    assert oriented.get_edge("A", "B").directed_pair() == ("A", "B")  # type: ignore[union-attr]
    assert oriented.get_edge("B", "C").directed_pair() == ("B", "C")  # type: ignore[union-attr]


def test_log_prior_of_hard_knowledge_is_zero_or_minus_infinity() -> None:
    knowledge = DomainKnowledge().require("X", "Y")
    assert knowledge.log_prior(dag(["X -> Y"])) == 0.0
    assert knowledge.energy(dag(["X -> Y"])) == 0.0
    assert knowledge.log_prior(dag(["Y -> X"])) == float("-inf")
    assert knowledge.energy(dag(["Y -> X"])) == float("inf")
    assert DomainKnowledge().log_prior(dag(["X -> Y"])) == 0.0
