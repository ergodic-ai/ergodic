# SPDX-License-Identifier: Apache-2.0
"""Integration tests for discovery: the facade, the seams, and the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import DomainKnowledge, dag, discover, estimate_effect, tabular
from ergodic.data import TabularData
from ergodic.discovery import PC, DSeparationOracle, Linear
from ergodic.graph import MixedGraph, consistent_extension, cpdag_of_dag


def _chain(n: int = 4000, seed: int = 0) -> TabularData:
    """X -> M -> Y, linear Gaussian; the total effect of X on Y is 0.56."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(size=n)
    y = 0.7 * m + rng.normal(size=n)
    return tabular(pd.DataFrame({"X": x, "M": m, "Y": y}))


def _five_node(n: int = 5000, seed: int = 3) -> tuple[TabularData, MixedGraph]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = 0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 0.6 * c + rng.normal(size=n)
    e = 0.8 * d + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e}))
    truth = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    return data, truth


# -- the facade ---------------------------------------------------------------


def test_discover_defaults_to_pc_with_an_auto_test() -> None:
    result = discover(_chain())
    assert result.method == "pc"
    assert result.test == "linear"  # auto picked the continuous test
    assert result.graph == cpdag_of_dag(dag(["X -> M", "M -> Y"]))
    assert result.sepsets is not None
    assert result.sepsets[frozenset({"X", "Y"})] == frozenset({"M"})


def test_discover_auto_picks_the_test_from_the_schema() -> None:
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.integers(0, 3, size=n)
    y = (x + rng.integers(0, 2, size=n)) % 3
    discrete = tabular(pd.DataFrame({"X": x, "Y": y}).astype("category"))
    assert discover(discrete).test == "discrete"
    mixed = tabular(pd.DataFrame({"X": rng.normal(size=n), "C": pd.Categorical(x)}))
    assert discover(mixed).test == "mixed"


def test_discover_routes_to_exact_search() -> None:
    data, truth = _five_node()
    result = discover(data, method="exact")
    assert result.method == "astar"
    assert result.graph.is_dag()
    assert cpdag_of_dag(result.graph) == cpdag_of_dag(truth)


def test_discover_gates_the_ingredients() -> None:
    data = _chain(n=300)
    with pytest.raises(ValueError, match="consumes a test"):
        discover(data, method="pc", score="bic")
    with pytest.raises(ValueError, match="consumes a score"):
        discover(data, method="astar", test="linear")
    with pytest.raises(ValueError, match="unknown discovery method"):
        discover(data, method="nope")


# -- the seams ----------------------------------------------------------------


def test_a_warmed_oracle_serves_a_second_run_from_cache() -> None:
    data = _chain()
    linear = Linear(data)
    first = PC(test=linear, alpha=0.01).fit()
    second = PC(test=linear, alpha=0.10).fit()
    assert second.diagnostics["cache_hits"] >= first.diagnostics["n_ci_tests"]
    assert first.graph == second.graph  # same queries, comfortable margins


def test_pc_and_exact_search_agree_through_the_equivalence_class() -> None:
    data, truth = _five_node()
    by_pc = discover(data, method="pc")
    by_astar = discover(data, method="astar")
    assert by_pc.graph == cpdag_of_dag(by_astar.graph) == cpdag_of_dag(truth)


def test_the_oracle_run_is_the_recovery_guarantee() -> None:
    truth = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    result = PC(test=DSeparationOracle(truth)).fit()
    assert result.graph == cpdag_of_dag(truth)
    assert result.n == 0  # no data anywhere


# -- the pipeline -------------------------------------------------------------


def test_discover_then_estimate_end_to_end() -> None:
    data = _chain()
    tiers = DomainKnowledge().with_tiers([["X"], ["M"], ["Y"]])
    found = discover(data, method="pc", knowledge=tiers)
    extension = consistent_extension(found.graph)
    assert extension is not None
    learned = dag([e.glyph() for e in extension.edges])
    assert learned == dag(["X -> M", "M -> Y"])
    est = estimate_effect(learned, data, "X", "Y", strategy="dml")  # continuous treatment
    assert abs(est.value - 0.56) < 0.06


def test_skeleton_restriction_reaches_exact_search() -> None:
    data, truth = _five_node()
    support = DomainKnowledge().restrict_to(truth)
    result = discover(data, method="astar", knowledge=support)
    assert cpdag_of_dag(result.graph) == cpdag_of_dag(truth)
    found_pairs = {frozenset((e.u, e.v)) for e in result.graph.edges}
    truth_pairs = {frozenset((e.u, e.v)) for e in truth.edges}
    assert found_pairs <= truth_pairs
