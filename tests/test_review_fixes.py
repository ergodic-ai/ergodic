# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the review fixes on the ID/mediation branch.

M1: ``_cut_outgoing`` must keep the bidirected edge of a projected bow.
M2: serialization/adjacency must raise (not corrupt) on a parallel-edge graph.
m1: ``identify_distribution`` must allow an empty do-set (the observational query).
"""

from __future__ import annotations

import pytest

from ergodic import admg, dag, identify_distribution
from ergodic.graph import MixedGraph, latent_projection
from ergodic.graph._surgery import _cut_outgoing


def _bow() -> MixedGraph:
    """The bow arc as a parallel-edge ADMG: X -> Y and X <-> Y."""
    return latent_projection(dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]))


def test_cut_outgoing_keeps_the_bidirected_edge_of_a_bow() -> None:
    # M1: cutting edges out of X drops the directed X -> Y but keeps X <-> Y.
    cut = _cut_outgoing(_bow(), {"X"})
    assert any(e.is_bidirected and {e.u, e.v} == {"X", "Y"} for e in cut.edges)
    assert not any(e.directed_pair() == ("X", "Y") for e in cut.edges)


def test_serialization_raises_on_a_projected_bow() -> None:
    # M2: a parallel-edge pair cannot be serialized or turned into an adjacency
    # matrix without losing the bidirected edge, so it raises instead.
    bow = _bow()
    with pytest.raises(ValueError, match="parallel-edge"):
        bow.to_dict()
    with pytest.raises(ValueError, match="parallel-edge"):
        bow.to_adjacency()
    nx = pytest.importorskip("networkx")
    assert nx is not None
    with pytest.raises(ValueError, match="parallel-edge"):
        bow.to_networkx()


def test_serialization_still_works_for_a_single_bidirected_edge() -> None:
    # The guard must not over-trigger: a normal ADMG has one edge per pair.
    g = admg(["X -> Y", "Z <-> Y"])
    assert g.to_dict()["edges"]  # round-trippable
    matrix, labels = g.to_adjacency()  # drops <-> as before, no raise
    assert matrix.shape == (len(labels), len(labels))


def test_identify_distribution_allows_an_empty_do() -> None:
    # m1: P(Y | do(), ...) is the observational query and the engine handles it.
    r = identify_distribution(dag(["X -> Y"]), "Y", [])
    assert r.identifiable
    assert r.functional is not None
    assert r.functional.free_vars() == frozenset({"Y"})


def test_identify_distribution_empty_do_with_conditioning() -> None:
    r = identify_distribution(dag(["X -> Y"]), "Y", [], given="X")
    assert r.identifiable
    assert r.functional is not None
    assert {"X", "Y"} <= r.functional.free_vars()
