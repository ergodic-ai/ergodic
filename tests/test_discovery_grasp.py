# SPDX-License-Identifier: Apache-2.0
"""Tests for GRaSP, the tuck walk over grow-shrink projections.

The executable anchor is the same one BOSS answers to: agreement with the
brute-force-verified A* optimum on a shared warmed score, over the same
twenty worlds. The tuck itself is held to a hand derivation, knowledge is
exercised where the edge-based move set supports it, and the documented
limit (a tier-gated variable can strand, because tucks only act on edges of
the projection) is pinned so a future fix has to update the docstring too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import tabular
from ergodic.discovery.algorithms._astar import AStar
from ergodic.discovery.algorithms._grasp import GRaSP, _tuck
from ergodic.discovery.scores._bic import BIC
from ergodic.discovery.scores._discrete_bic import DiscreteBIC
from ergodic.graph import GraphKind, MixedGraph, cpdag_of_dag, dag
from ergodic.knowledge import DomainKnowledge

if TYPE_CHECKING:
    from ergodic.data import TabularData

# -- helpers ---------------------------------------------------------------------


def _directed(graph: MixedGraph, parent: str, child: str) -> bool:
    edge = graph.get_edge(parent, child)
    return edge is not None and edge.directed_pair() == (parent, child)


def _random_dag_data(seed: int, p: int = 4, n: int = 2000) -> TabularData:
    """A random linear SEM on ``p`` nodes: edges in with 0.5, weights in [0.5, 1.5]."""
    rng = np.random.default_rng(seed)
    names = list("abcdef"[:p])
    values: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        column = rng.normal(size=n)
        for parent in names[:i]:
            if rng.random() < 0.5:
                weight = rng.uniform(0.5, 1.5) * (1.0 if rng.random() < 0.5 else -1.0)
                column = column + weight * values[parent]
        values[name] = column
    return tabular(values)


def _discrete_network() -> tuple[TabularData, MixedGraph]:
    """a -> b -> d <- c, binary, conditional probabilities far from uniform."""
    rng = np.random.default_rng(21)
    n = 3000
    a = rng.integers(0, 2, size=n)
    b = np.where(rng.random(n) < 0.85, a, 1 - a)
    c = rng.integers(0, 2, size=n)
    p_d = np.where(b & c, 0.9, np.where(b, 0.65, np.where(c, 0.35, 0.1)))
    d = (rng.random(n) < p_d).astype(np.int64)
    truth = dag(["a -> b", "b -> d", "c -> d"])
    data = tabular(
        {"a": a, "b": b, "c": c, "d": d},
        types=dict.fromkeys(("a", "b", "c", "d"), "categorical"),
    )
    return data, truth


# -- exactness against A* on a shared warmed score ---------------------------------


@pytest.mark.parametrize(
    ("seed", "p"), [(s, 4) for s in range(12)] + [(s, 5) for s in range(12, 20)]
)
def test_grasp_matches_the_exact_optimum(seed: int, p: int) -> None:
    data = _random_dag_data(seed, p=p)
    bic = BIC(data)
    found = GRaSP(score=bic).fit()
    exact = AStar(score=bic).fit()
    assert found.graph.kind is GraphKind.CPDAG
    assert found.graph == cpdag_of_dag(exact.graph)


# -- the tuck, held to a hand derivation -----------------------------------------------


def test_tuck_moves_the_segment_ancestors() -> None:
    order = ["a", "b", "c", "d", "e"]
    # c is a parent (so an ancestor) of d: it moves with d, ahead of b
    with_anc = {
        "a": frozenset(),
        "b": frozenset({"a"}),
        "c": frozenset(),
        "d": frozenset({"b", "c"}),
        "e": frozenset(),
    }
    assert _tuck(order, "b", "d", with_anc) == ["a", "c", "d", "b", "e"]
    # c is no ancestor of d: it stays in the segment, behind b
    without = {
        "a": frozenset(),
        "b": frozenset({"a"}),
        "c": frozenset(),
        "d": frozenset({"b"}),
        "e": frozenset(),
    }
    assert _tuck(order, "b", "d", without) == ["a", "d", "b", "c", "e"]


# -- a discrete world ------------------------------------------------------------------


def test_grasp_recovers_a_discrete_network_with_discrete_bic() -> None:
    data, truth = _discrete_network()
    result = GRaSP(score=DiscreteBIC(data)).fit()
    assert result.graph == cpdag_of_dag(truth)
    assert result.score == "bic_discrete"


# -- knowledge ---------------------------------------------------------------------------


def test_grasp_knowledge_required_edge_survives_in_both_directions() -> None:
    rng = np.random.default_rng(8)
    data = tabular({"U": rng.normal(size=600), "V": rng.normal(size=600)})
    for cause, effect in (("U", "V"), ("V", "U")):
        required = DomainKnowledge().require(cause, effect)
        result = GRaSP("bic").fit(data, knowledge=required)
        assert _directed(result.graph, cause, effect)
        assert required.is_consistent(result.graph)
        assert result.diagnostics["cpdag_valid"] == 0.0  # a lone forced arrow is no CPDAG
        assert result.graph.kind is GraphKind.MIXED


def test_grasp_knowledge_forbidden_pair_never_appears() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.normal(size=n)
    y = 1.2 * x + rng.normal(size=n)
    z = 0.9 * y + rng.normal(size=n)
    data = tabular({"X": x, "Y": y, "Z": z})
    forbidden = DomainKnowledge().forbid_adjacency("X", "Y")
    result = GRaSP("bic").fit(data, knowledge=forbidden)
    assert result.graph.get_edge("X", "Y") is None  # the ban beats the strong signal
    assert result.graph.get_edge("Y", "Z") is not None
    assert forbidden.is_consistent(result.graph)


def test_grasp_required_edge_cycle_raises() -> None:
    rng = np.random.default_rng(10)
    data = tabular({name: rng.normal(size=100) for name in ("a", "b", "c")})
    cyclic = DomainKnowledge().require("a", "b").require("b", "c").require("c", "a")
    with pytest.raises(ValueError, match="required-edge cycle"):
        GRaSP("bic").fit(data, knowledge=cyclic)


def test_grasp_tier_gated_variable_strands_as_documented() -> None:
    # The docstring's honest limit: tucks act only on edges of the projection,
    # so X, whose edges into the later tier never enter the sorted-start
    # projection, has nothing to tuck and strands. The result is consistent
    # with the knowledge but misses the X -> M edge BOSS finds; if a change
    # lifts this, update the GRaSP docstring along with this pin.
    rng = np.random.default_rng(6)
    n = 3000
    x = rng.normal(size=n)
    m = 1.1 * x + rng.normal(size=n)
    y = 0.9 * m + rng.normal(size=n)
    data = tabular({"X": x, "M": m, "Y": y})
    tiers = DomainKnowledge().with_tiers([["X"], ["M", "Y"]])
    result = GRaSP("bic").fit(data, knowledge=tiers)
    assert tiers.is_consistent(result.graph)
    assert result.graph.get_edge("X", "M") is None  # the documented strand
    assert result.graph.get_edge("M", "Y") is not None


# -- determinism ---------------------------------------------------------------------------


def test_grasp_is_deterministic() -> None:
    data = _random_dag_data(2)
    first = GRaSP("bic").fit(data)
    second = GRaSP("bic").fit(data)
    assert first.graph == second.graph
    assert dict(first.diagnostics) == dict(second.diagnostics)


# -- the result record and argument validation ------------------------------------------------


def test_grasp_result_record_and_diagnostics() -> None:
    data = _random_dag_data(5, p=4, n=1500)
    result = GRaSP("auto").fit(data)  # continuous schema resolves to the Gaussian BIC
    for key in ("n_score_evals", "cache_hits", "n_tucks", "n_visited", "cpdag_valid"):
        assert key in result.diagnostics
    assert result.method == "grasp"
    assert result.score == "bic"
    assert result.test is None
    assert result.alpha is None
    assert result.sepsets is None
    assert result.n == 1500


def test_grasp_rejects_bad_arguments() -> None:
    with pytest.raises(TypeError, match="Score or a string"):
        GRaSP(score=3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="depth"):
        GRaSP(depth=0)
