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
    g = dag(["X -> Y", "U -> X", "U -> Y"], latent=["U"])
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
    g = dag(["I -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
    assert is_instrument(g, "I", "X", "Y")
    assert ("I", frozenset()) in instruments(g, "X", "Y")


def test_instrument_rejected_when_it_reaches_outcome_directly() -> None:
    g = dag(["I -> X", "I -> Y", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
    assert not is_instrument(g, "I", "X", "Y")
    assert instruments(g, "X", "Y") == []


def test_conditional_instrument_needs_its_conditioning_set() -> None:
    g = dag(["W -> I", "W -> Y", "I -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
    assert not is_instrument(g, "I", "X", "Y")  # the open path I <- W -> Y
    assert is_instrument(g, "I", "X", "Y", "W")
    assert ("I", frozenset({"W"})) in instruments(g, "X", "Y")


def test_descendant_of_treatment_is_not_an_instrument() -> None:
    # Z sits downstream of X: cutting X's outgoing edges severs the X -> Z edge
    # that makes Z endogenous, so the exclusion check alone would pass it.
    g = admg(["X <-> Y", "X -> Z"])
    assert not is_instrument(g, "Z", "X", "Y")
    result = identify_effect(g, "X", "Y")
    assert result.strategy != "iv"
    assert result.strategy == "null"  # no directed path from X to Y: the effect is zero

    g2 = dag(["X -> Y", "U -> X", "U -> Y", "X -> Z"], latent=["U"])
    assert not is_instrument(g2, "Z", "X", "Y")
    result2 = identify_effect(g2, "X", "Y")
    assert not result2.identifiable
    assert result2.strategy == "none"


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
    g = dag(["I -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U"])
    result = identify_effect(g, "X", "Y")
    assert result.strategy == "iv"
    assert result.instruments[0][0] == "I"
    assert "linear (parametric)" in result.estimand


def test_identify_effect_reports_when_not_identifiable() -> None:
    g = dag(["X -> Y", "U -> X", "U -> Y"], latent=["U"])
    result = identify_effect(g, "X", "Y")
    assert not result.identifiable
    assert result.strategy == "none"


def test_identification_rejects_pags() -> None:
    with pytest.raises(NotImplementedError):
        identify_effect(pag(["X o-> Y"]), "X", "Y")


# -- latent nodes ------------------------------------------------------------


def test_latent_nodes_stay_out_of_adjustment_sets() -> None:
    butterfly = [
        "U1 -> Z",
        "U2 -> Z",
        "U1 -> X",
        "U2 -> Y",
        "Z -> X",
        "Z -> Y",
        "X -> Y",
    ]
    observed = dag(butterfly)
    assert adjustment_set(observed, "X", "Y") == frozenset({"U1", "U2", "Z"})
    hidden = dag(butterfly, latent=["U1", "U2"])
    # {Z} alone opens the U1 -> Z <- U2 collider, and U1, U2 cannot be used
    assert adjustment_set(hidden, "X", "Y") is None
    assert not is_valid_adjustment_set(hidden, "X", "Y", {"U1", "U2", "Z"})


def test_latent_mediator_is_no_frontdoor_set() -> None:
    g = admg(["X -> M", "M -> Y", "X <-> Y"], latent=["M"])
    assert not is_frontdoor_set(g, "X", "Y", {"M"})
    assert frontdoor_set(g, "X", "Y") is None


def test_latent_nodes_are_not_instruments_or_conditioning_nodes() -> None:
    g = dag(["U -> X", "X -> Y", "C -> X", "C -> Y"], latent=["U", "C"])
    assert not is_instrument(g, "U", "X", "Y")
    assert instruments(g, "X", "Y") == []
    # a conditional instrument whose conditioning node is latent is unusable
    h = dag(["W -> I", "W -> Y", "I -> X", "X -> Y", "U -> X", "U -> Y"], latent=["U", "W"])
    assert not is_instrument(h, "I", "X", "Y", "W")
    assert instruments(h, "X", "Y") == []


# -- input validation --------------------------------------------------------


def test_identify_effect_validates_inputs() -> None:
    g = dag(["X -> Y"])
    with pytest.raises(ValueError, match="disjoint"):
        identify_effect(g, "X", "X")
    with pytest.raises(ValueError, match="NOPE"):
        identify_effect(g, "X", "NOPE")
    with pytest.raises(ValueError, match="ZAP"):
        identify_effect(g, "ZAP", "Y")
    with pytest.raises(ValueError, match="treatment"):
        identify_effect(g, [], "Y")
    with pytest.raises(ValueError, match="outcome"):
        identify_effect(g, "X", [])


def test_helper_functions_validate_inputs() -> None:
    g = dag(["Z -> X", "Z -> Y", "X -> Y"])
    with pytest.raises(ValueError, match="NOPE"):
        is_valid_adjustment_set(g, "X", "Y", {"NOPE"})
    with pytest.raises(ValueError, match="NOPE"):
        adjustment_set(g, "X", "NOPE")
    with pytest.raises(ValueError, match="disjoint"):
        minimal_adjustment_set(g, "X", "X")
    with pytest.raises(ValueError, match="NOPE"):
        is_frontdoor_set(g, "X", "Y", {"NOPE"})
    with pytest.raises(ValueError, match="NOPE"):
        frontdoor_set(g, "NOPE", "Y")
    with pytest.raises(ValueError, match="NOPE"):
        is_instrument(g, "NOPE", "X", "Y")
    with pytest.raises(ValueError, match="NOPE"):
        instruments(g, "X", "NOPE")


# -- the null strategy -------------------------------------------------------


def test_identify_effect_null_when_outcome_is_upstream() -> None:
    result = identify_effect(dag(["Y -> X"]), "X", "Y")
    assert result.identifiable
    assert result.strategy == "null"
    assert "P(Y)" in result.estimand


def test_identify_effect_null_when_disconnected() -> None:
    result = identify_effect(dag(nodes=["X", "Y"]), "X", "Y")
    assert result.identifiable
    assert result.strategy == "null"


# -- set treatments ----------------------------------------------------------


def test_mediator_between_two_treatments_is_adjustable() -> None:
    # m sits between the two treatments; its only routes to y pass through x2,
    # so it is not on a proper causal path and may be adjusted for
    g = dag(["x1 -> m", "m -> x2", "x2 -> y"])
    assert is_valid_adjustment_set(g, {"x1", "x2"}, "y", {"m"})
    assert adjustment_set(g, {"x1", "x2"}, "y") == frozenset({"m"})
