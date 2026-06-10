# SPDX-License-Identifier: Apache-2.0
"""Identification on equivalence classes, anchored to all-members semantics.

"Z is an adjustment set relative to a CPDAG, MAG, or PAG" means: Z is valid
in every graph the class represents. The tests hold the generalized
criterion to that definition executably: exhaustively over candidate sets
against the enumerated members of small random classes, and for soundness
against the latent truths that generated them.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, estimate_effect, identify_effect, tabular
from ergodic.discovery import discover
from ergodic.graph import (
    MixedGraph,
    cpdag_of_dag,
    dags_of_cpdag,
    mag_of_dag,
    mags_of_pag,
    mixed,
    pag_of_mag,
)
from ergodic.identification import (
    adjustment_set,
    frontdoor_set,
    instruments,
    is_valid_adjustment_set,
    minimal_adjustment_set,
)


def _random_dag(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = [(names[i], names[j]) for i, j in combinations(range(p), 2) if rng.uniform() < 0.5]
    return dag(edges, nodes=names)


def _random_dag_with_latents(rng: np.random.Generator, p: int, n_latent: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        if rng.uniform() < 0.4:
            edges.append((names[i], names[j]))
    latent = list(rng.choice(names, size=n_latent, replace=False))
    return dag(edges, nodes=names, latent=latent)


def _subsets(rest: list[str]) -> list[frozenset[str]]:
    return [frozenset(zs) for size in range(len(rest) + 1) for zs in combinations(rest, size)]


# -- the all-members anchors ------------------------------------------------------


def test_the_cpdag_criterion_answers_for_every_member() -> None:
    # The spec for the generalized criterion: valid relative to the CPDAG
    # means valid in every consistent extension. Exhaustive over candidate
    # sets, with the existence theorem (some set works iff the canonical
    # set works) riding along.
    rng = np.random.default_rng(23)
    for _ in range(12):
        cp = cpdag_of_dag(_random_dag(rng, int(rng.choice([3, 4, 5]))))
        members = dags_of_cpdag(cp)
        nodes = sorted(cp.nodes)
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                rest = [n for n in nodes if n not in (x, y)]
                some_valid = False
                for z in _subsets(rest):
                    claim = is_valid_adjustment_set(cp, x, y, z)
                    spec = all(is_valid_adjustment_set(m, x, y, z) for m in members)
                    assert claim == spec, (sorted(e.glyph() for e in cp.edges), x, y, sorted(z))
                    some_valid |= spec
                got = adjustment_set(cp, x, y)
                assert (got is not None) == some_valid
                if got is not None:
                    assert all(is_valid_adjustment_set(m, x, y, got) for m in members)


def test_the_cpdag_criterion_handles_set_treatments() -> None:
    rng = np.random.default_rng(29)
    for _ in range(5):
        cp = cpdag_of_dag(_random_dag(rng, 5))
        members = dags_of_cpdag(cp)
        nodes = sorted(cp.nodes)
        for xs in combinations(nodes, 2):
            x = frozenset(xs)
            for y in [n for n in nodes if n not in x]:
                rest = [n for n in nodes if n not in x and n != y]
                for z in _subsets(rest):
                    claim = is_valid_adjustment_set(cp, x, y, z)
                    spec = all(is_valid_adjustment_set(m, x, y, z) for m in members)
                    assert claim == spec, (sorted(e.glyph() for e in cp.edges), sorted(x), y)


def test_the_pag_criterion_answers_for_every_member_mag() -> None:
    # The member MAGs represent exactly the DAGs the PAG represents, so
    # criterion-on-the-PAG must equal all(criterion-on-each-MAG).
    rng = np.random.default_rng(31)
    for _ in range(10):
        p = pag_of_mag(mag_of_dag(_random_dag_with_latents(rng, 4, int(rng.choice([1, 2])))))
        members = mags_of_pag(p)
        nodes = sorted(p.nodes)
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                rest = [n for n in nodes if n not in (x, y)]
                some_valid = False
                for z in _subsets(rest):
                    claim = is_valid_adjustment_set(p, x, y, z)
                    spec = all(is_valid_adjustment_set(m, x, y, z) for m in members)
                    assert claim == spec, (sorted(e.glyph() for e in p.edges), x, y, sorted(z))
                    some_valid |= spec
                got = adjustment_set(p, x, y)
                assert (got is not None) == some_valid


def test_the_mag_criterion_is_sound_against_the_generating_truth() -> None:
    # The latent truth is one member of the MAG's class, so a set the
    # criterion certifies on the MAG must be a valid adjustment set there.
    rng = np.random.default_rng(37)
    certified = 0
    for _ in range(40):
        truth = _random_dag_with_latents(rng, 5, int(rng.choice([1, 2])))
        mag = mag_of_dag(truth)
        nodes = sorted(mag.nodes)
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                rest = [n for n in nodes if n not in (x, y)]
                for z in _subsets(rest):
                    if is_valid_adjustment_set(mag, x, y, z):
                        certified += 1
                        assert is_valid_adjustment_set(truth, x, y, z), (
                            sorted(e.glyph() for e in truth.edges),
                            x,
                            y,
                            sorted(z),
                        )
    assert certified > 100  # the sweep is not vacuous


# -- hand cases ---------------------------------------------------------------------


def test_a_compelled_cpdag_edge_identifies_like_a_dag() -> None:
    # The v-structure compels A -> X <- B, and Meek's R1 compels X -> Y, so
    # the class identifies X -> Y with the same canonical set a DAG would.
    truth = dag(["A -> X", "B -> X", "X -> Y"])
    cp = cpdag_of_dag(truth)
    result = identify_effect(cp, "X", "Y")
    assert result.identifiable
    assert result.strategy == "adjustment"
    assert result.adjustment_set == frozenset({"A", "B"})
    assert minimal_adjustment_set(cp, "X", "Y") == frozenset()


def test_an_undirected_first_edge_is_not_amenable() -> None:
    # The chain class X - M - Y: members disagree about X's first edge, so
    # no set adjusts uniformly and the report names amenability.
    cp = cpdag_of_dag(dag(["X -> M", "M -> Y"]))
    result = identify_effect(cp, "X", "Y")
    assert not result.identifiable
    assert "not amenable" in result.estimand
    assert adjustment_set(cp, "X", "Y") is None
    assert not is_valid_adjustment_set(cp, "X", "Y", frozenset())


def test_member_enumeration_is_the_escape_hatch_when_the_class_disagrees() -> None:
    # The IDA pattern: when the class is not amenable, enumerate the members
    # and identify per member; the answers genuinely differ across them.
    cp = cpdag_of_dag(dag(["X -> M", "M -> Y"]))
    answers = {identify_effect(m, "X", "Y").strategy for m in dags_of_cpdag(cp)}
    assert answers == {"adjustment", "null"}


def test_null_on_a_class_when_no_member_has_a_causal_path() -> None:
    truth = dag(["A -> X", "B -> X", "X -> Y"])
    cp = cpdag_of_dag(truth)
    result = identify_effect(cp, "Y", "A")
    assert result.identifiable
    assert result.strategy == "null"
    assert "any member" in result.estimand
    p = pag_of_mag(mag_of_dag(truth))
    result = identify_effect(p, "Y", "A")
    assert result.strategy == "null"


def test_a_pag_identifies_through_its_visible_edge() -> None:
    # A o-> X <-o B with X --> Y: the circle marks tolerate confounding of
    # A and X, but X -> Y is visible (A is not adjacent to Y), so the class
    # is amenable and one estimand covers every member.
    p = pag_of_mag(mag_of_dag(dag(["A -> X", "B -> X", "X -> Y"])))
    glyphs = sorted(e.glyph() for e in p.edges)
    assert glyphs == ["A o-> X", "B o-> X", "X --> Y"]
    result = identify_effect(p, "X", "Y")
    assert result.identifiable
    assert result.strategy == "adjustment"
    chosen = result.adjustment_set
    assert chosen is not None
    members = mags_of_pag(p)
    assert len(members) > 1
    assert all(is_valid_adjustment_set(m, "X", "Y", chosen) for m in members)


def test_an_invisible_pag_edge_is_reported_not_amenable() -> None:
    # The two-node class X o-o Y: nothing certifies the direction or rules
    # out confounding, so adjustment cannot identify it.
    p = pag_of_mag(mag_of_dag(dag(["X -> Y"])))
    result = identify_effect(p, "X", "Y")
    assert not result.identifiable
    assert "not amenable" in result.estimand


# -- dispatch and errors --------------------------------------------------------------


def test_front_door_and_instruments_need_a_single_graph() -> None:
    cp = cpdag_of_dag(dag(["A -> X", "B -> X", "X -> Y"]))
    with pytest.raises(NotImplementedError, match="enumerate"):
        frontdoor_set(cp, "X", "Y")
    with pytest.raises(NotImplementedError, match="enumerate"):
        instruments(cp, "X", "Y")


def test_identification_rejects_class_faces_with_latent_designations() -> None:
    g = cpdag_of_dag(dag(["A -> X", "B -> X", "X -> Y"]))
    relabelled = mixed([e.glyph() for e in g.edges], latent=["A"]).as_kind("cpdag")
    with pytest.raises(ValueError, match="no latent or selection designations"):
        identify_effect(relabelled, "X", "Y")


# -- the bridge: discover, then estimate ----------------------------------------------


def _simulated(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    x = a + b + 0.5 * rng.normal(size=n)
    y = 2.0 * x + 0.5 * rng.normal(size=n)
    return pd.DataFrame({"A": a, "B": b, "X": x, "Y": y}), 2.0


def test_estimate_effect_flows_through_a_discovered_cpdag() -> None:
    frame, ate = _simulated()
    result = discover(tabular(frame), method="pc")
    assert result.graph.kind.value == "cpdag"
    estimate = estimate_effect(result.graph, tabular(frame), "X", "Y", strategy="dml")
    assert abs(estimate.value - ate) < 0.1


def test_estimate_effect_flows_through_a_discovered_pag() -> None:
    # FCI cannot rule out confounding of A and X (circles stay), but the
    # X -> Y edge it orients is visible, so the effect estimates anyway.
    frame, ate = _simulated(seed=1)
    result = discover(tabular(frame), method="fci")
    assert result.graph.kind.value == "pag"
    estimate = estimate_effect(result.graph, tabular(frame), "X", "Y", strategy="dml")
    assert abs(estimate.value - ate) < 0.1
