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


def test_soft_hooks_not_implemented_yet() -> None:
    knowledge = DomainKnowledge()
    graph = dag(["X -> Y"])
    with pytest.raises(NotImplementedError):
        knowledge.energy(graph)
    with pytest.raises(NotImplementedError):
        knowledge.log_prior(graph)
