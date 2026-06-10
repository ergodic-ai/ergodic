# SPDX-License-Identifier: Apache-2.0
"""Tests for identification: adjustment, front-door, and instruments."""

from __future__ import annotations

import pytest

from ergodic import admg, dag, identify_effect, pag
from ergodic.identification import (
    adjustment_set,
    frontdoor_set,
    instruments,
    is_frontdoor_set,
    is_instrument,
    is_valid_adjustment_set,
    minimal_adjustment_set,
)

# -- adjustment --------------------------------------------------------------


def test_confounding_triangle_needs_the_confounder() -> None:
    g = dag(["Z -> X", "Z -> Y", "X -> Y"])
    assert adjustment_set(g, "X", "Y") == frozenset({"Z"})
    assert is_valid_adjustment_set(g, "X", "Y", {"Z"})
    assert not is_valid_adjustment_set(g, "X", "Y", set())  # back-door left open


def test_pure_mediator_needs_no_adjustment() -> None:
    g = dag(["X -> M", "M -> Y"])
    assert adjustment_set(g, "X", "Y") == frozenset()
    # a mediator is forbidden: adjusting for it blocks the effect we want
    assert not is_valid_adjustment_set(g, "X", "Y", {"M"})


def test_minimal_drops_a_redundant_node() -> None:
    g = dag(["W -> Z", "Z -> X", "Z -> Y", "X -> Y"])
    assert adjustment_set(g, "X", "Y") == frozenset({"Z", "W"})
    assert minimal_adjustment_set(g, "X", "Y") == frozenset({"Z"})


def test_unobserved_confounding_has_no_adjustment_set() -> None:
    g = admg(["X -> Y", "X <-> Y"])
    assert adjustment_set(g, "X", "Y") is None


# -- front-door --------------------------------------------------------------


def test_frontdoor_through_a_mediator() -> None:
    g = admg(["X -> M", "M -> Y", "X <-> Y"])
    assert adjustment_set(g, "X", "Y") is None  # confounder is unobserved
    assert is_frontdoor_set(g, "X", "Y", {"M"})
    assert frontdoor_set(g, "X", "Y") == frozenset({"M"})


def test_no_frontdoor_without_a_clean_mediator() -> None:
    g = dag(["Z -> X", "Z -> Y", "X -> Y"])
    assert frontdoor_set(g, "X", "Y") is None


# -- instruments -------------------------------------------------------------


def test_instrument_found() -> None:
    g = admg(["I -> X", "X -> Y", "X <-> Y"])
    assert is_instrument(g, "I", "X", "Y")
    assert ("I", frozenset()) in instruments(g, "X", "Y")


def test_instrument_rejected_when_it_reaches_outcome_directly() -> None:
    g = admg(["I -> X", "I -> Y", "X -> Y", "X <-> Y"])
    assert not is_instrument(g, "I", "X", "Y")
    assert instruments(g, "X", "Y") == []


# -- orchestration -----------------------------------------------------------


def test_identify_effect_picks_adjustment() -> None:
    result = identify_effect(dag(["Z -> X", "Z -> Y", "X -> Y"]), "X", "Y")
    assert result.identifiable
    assert bool(result) is True
    assert result.strategy == "adjustment"
    assert result.adjustment_set == frozenset({"Z"})
    assert "sum_z" in result.estimand


def test_identify_effect_picks_frontdoor() -> None:
    result = identify_effect(admg(["X -> M", "M -> Y", "X <-> Y"]), "X", "Y")
    assert result.strategy == "frontdoor"
    assert result.mediators == frozenset({"M"})


def test_identify_effect_picks_instrument() -> None:
    result = identify_effect(admg(["I -> X", "X -> Y", "X <-> Y"]), "X", "Y")
    assert result.strategy == "iv"
    assert result.instruments[0][0] == "I"


def test_identify_effect_reports_when_not_identifiable() -> None:
    result = identify_effect(admg(["X -> Y", "X <-> Y"]), "X", "Y")
    assert not result.identifiable
    assert result.strategy == "none"


def test_identification_rejects_pags() -> None:
    with pytest.raises(NotImplementedError):
        identify_effect(pag(["X o-> Y"]), "X", "Y")
