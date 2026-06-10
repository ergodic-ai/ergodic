# SPDX-License-Identifier: Apache-2.0
"""Tests for the score stack: the Gaussian BIC and A* exact search."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.data import tabular
from ergodic.discovery.algorithms._astar import AStar
from ergodic.discovery.scores._bic import BIC
from ergodic.graph import dag
from ergodic.knowledge import DomainKnowledge

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph

# -- simulated structures ------------------------------------------------------


def _four_node(n: int = 1500, seed: int = 5) -> tuple[TabularData, MixedGraph]:
    """a -> b, a -> c, b -> d, c -> d, linear Gaussian with strong coefficients."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 1.1 * c + rng.normal(size=n)
    truth = dag(["a -> b", "a -> c", "b -> d", "c -> d"])
    return tabular({"a": a, "b": b, "c": c, "d": d}), truth


def _five_node(n: int = 5000, seed: int = 6) -> tuple[TabularData, MixedGraph]:
    """The four-node diamond plus d -> e."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 1.1 * c + rng.normal(size=n)
    e = 0.8 * d + rng.normal(size=n)
    truth = dag(["a -> b", "a -> c", "b -> d", "c -> d", "d -> e"])
    return tabular({"a": a, "b": b, "c": c, "d": d, "e": e}), truth


def _skeleton(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((edge.u, edge.v)) for edge in graph.edges}


def _parent_map(graph: MixedGraph) -> dict[str, frozenset[str]]:
    return {node: graph.parents(node) for node in sorted(graph.nodes)}


# -- BIC: closed form, two paths, cache ----------------------------------------


def test_bic_matches_the_closed_form() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(size=n)
    score = BIC(tabular({"x": x, "y": y}))
    xc = x - x.mean()
    yc = y - y.mean()
    syy = float(yc @ yc) / n
    sxx = float(xc @ xc) / n
    sxy = float(xc @ yc) / n
    residual = syy - sxy * sxy / sxx
    regression = -(n / 2) * math.log(residual) - (2 / 2) * math.log(n)  # k = 1
    assert math.isclose(score.local("y", ["x"]), regression, rel_tol=1e-10)
    marginal = -(n / 2) * math.log(syy) - (1 / 2) * math.log(n)  # k = 0
    assert math.isclose(score.local("y"), marginal, rel_tol=1e-10)


def test_bic_covariance_path_matches_least_squares() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    c = rng.integers(0, 3, size=n)
    y = 0.8 * x + 0.5 * (c == 2).astype(np.float64) + rng.normal(size=n)
    data = tabular({"x": x, "c": c, "y": y}, types={"c": "categorical"})
    score = BIC(data)
    onehot = (c[:, None] == np.arange(3)[None, :]).astype(np.float64)[:, 1:]
    blocks: dict[str, npt.NDArray[np.float64]] = {"x": x[:, None], "c": onehot}
    target = y[:, None]
    for parents in [(), ("x",), ("c",), ("c", "x")]:
        design = (
            np.hstack([blocks[p] for p in sorted(parents)])
            if parents
            else np.empty((n, 0), dtype=np.float64)
        )
        via_cov = score.local("y", parents)
        via_arrays = score.local_from_arrays(target, design)
        assert abs(via_cov - via_arrays) < 1e-8


def test_bic_caches_local_evaluations() -> None:
    rng = np.random.default_rng(2)
    data = tabular({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    score = BIC(data)
    first = score.local("y", ["x"])
    second = score.local("y", ["x"])
    assert first == second
    counters = score.diagnostics()
    assert counters["n_score_evals"] == 1.0
    assert counters["cache_hits"] == 1.0


def test_bic_rejects_a_multi_column_node_lazily() -> None:
    rng = np.random.default_rng(3)
    n = 120
    c = rng.integers(0, 3, size=n)
    y = (c == 1).astype(np.float64) + rng.normal(size=n)
    data = tabular({"c": c, "y": y}, types={"c": "categorical"})
    score = BIC(data)  # binding is fine; the reject happens at the first query
    assert math.isfinite(score.local("y", ["c"]))  # categorical parents are allowed
    with pytest.raises(TypeError, match="single-column"):
        score.local("c")


def test_bic_penalty_multiplies_the_complexity_term() -> None:
    rng = np.random.default_rng(4)
    n = 150
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    data = tabular({"x": x, "y": y})
    plain = BIC(data).local("y", ["x"])
    doubled = BIC(data, penalty=2.0).local("y", ["x"])
    assert math.isclose(plain - doubled, (2 / 2) * math.log(n), rel_tol=1e-10)  # k = 1


# -- A*: brute-force agreement ---------------------------------------------------


def _acyclic(parents: dict[str, set[str]]) -> bool:
    indegree = {node: len(ps) for node, ps in parents.items()}
    children: dict[str, list[str]] = {node: [] for node in parents}
    for node, ps in parents.items():
        for parent in ps:
            children[parent].append(node)
    ready = [node for node, degree in indegree.items() if degree == 0]
    removed = 0
    while ready:
        node = ready.pop()
        removed += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    return removed == len(parents)


def _all_parent_maps(names: tuple[str, ...]) -> Iterator[dict[str, frozenset[str]]]:
    """Every labeled DAG on ``names``, as a parent map."""
    ordered_pairs = [(u, v) for u in names for v in names if u != v]
    for mask in range(1 << len(ordered_pairs)):
        parents: dict[str, set[str]] = {node: set() for node in names}
        for i, (u, v) in enumerate(ordered_pairs):
            if mask >> i & 1:
                parents[v].add(u)
        if _acyclic(parents):
            yield {node: frozenset(ps) for node, ps in parents.items()}


def test_astar_matches_brute_force_at_four_nodes() -> None:
    data, _ = _four_node()
    score = BIC(data)
    best = -math.inf
    count = 0
    for parents in _all_parent_maps(("a", "b", "c", "d")):
        count += 1
        best = max(best, score.graph_score(parents))
    assert count == 543  # the number of labeled DAGs on four nodes
    result = AStar(score=score).fit()
    assert result.graph.is_dag()
    found = score.graph_score(_parent_map(result.graph))
    assert math.isclose(found, best, rel_tol=1e-12, abs_tol=1e-8)


def test_astar_result_record() -> None:
    data, _ = _four_node(n=400, seed=8)
    result = AStar("bic").fit(data)
    assert result.method == "astar"
    assert result.score == "bic"
    assert result.test is None
    assert result.alpha is None
    assert result.n == 400
    assert result.sepsets is None
    assert result.diagnostics["expanded_states"] >= 1.0
    assert result.diagnostics["n_score_evals"] >= 1.0


# -- A*: recovery -----------------------------------------------------------------


def test_astar_recovers_a_linear_gaussian_sem() -> None:
    data, truth = _five_node()
    score = BIC(data)
    result = AStar(score=score).fit()
    # Score equivalence allows any member of the Markov equivalence class, so
    # compare the skeleton and the total score, not the edge directions.
    assert _skeleton(result.graph) == _skeleton(truth)
    found = score.graph_score(_parent_map(result.graph))
    target = score.graph_score(_parent_map(truth))
    assert found >= target - 1e-8  # exact search never scores below the truth
    assert math.isclose(found, target, rel_tol=1e-9)


# -- A*: knowledge ------------------------------------------------------------------


def test_astar_forces_a_required_edge() -> None:
    rng = np.random.default_rng(7)
    n = 800
    w = rng.normal(size=n)  # independent of the rest; only knowledge connects it
    x = rng.normal(size=n)
    y = 1.2 * x + rng.normal(size=n)
    data = tabular({"w": w, "x": x, "y": y})
    knowledge = DomainKnowledge().require("w", "y")
    result = AStar("bic").fit(data, knowledge=knowledge)
    edge = result.graph.get_edge("w", "y")
    assert edge is not None
    assert edge.directed_pair() == ("w", "y")


def test_astar_never_returns_a_forbidden_edge() -> None:
    data, _ = _four_node()
    knowledge = DomainKnowledge().forbid("a", "b")
    result = AStar("bic").fit(data, knowledge=knowledge)
    edge = result.graph.get_edge("a", "b")
    assert edge is None or edge.directed_pair() != ("a", "b")


def test_astar_respects_tiers() -> None:
    data, truth = _four_node()
    knowledge = DomainKnowledge().with_tiers([["a"], ["b", "c"], ["d"]])
    result = AStar("bic").fit(data, knowledge=knowledge)
    tier = {"a": 0, "b": 1, "c": 1, "d": 2}
    for edge in result.graph.edges:
        pair = edge.directed_pair()
        assert pair is not None
        cause, effect = pair
        assert tier[cause] <= tier[effect]
    assert _skeleton(result.graph) == _skeleton(truth)


# -- A*: guards -----------------------------------------------------------------------


def test_astar_node_limit_guard_points_at_the_escape_hatches() -> None:
    data, _ = _four_node(n=200, seed=9)
    with pytest.raises(ValueError, match="max_parents") as caught:
        AStar("bic", node_limit=3).fit(data)
    assert "restrict_to" in str(caught.value)


def test_astar_rejects_required_parents_beyond_max_parents() -> None:
    data, _ = _four_node(n=200, seed=10)
    knowledge = DomainKnowledge().require("a", "d").require("b", "d")
    with pytest.raises(ValueError, match="max_parents"):
        AStar("bic", max_parents=1).fit(data, knowledge=knowledge)


def test_astar_data_resolution_rules() -> None:
    data, _ = _four_node(n=300, seed=11)
    other, _ = _four_node(n=301, seed=12)
    bound = BIC(data)
    with pytest.raises(ValueError, match="different"):
        AStar(score=bound).fit(other)  # bound score plus another dataset
    with pytest.raises(ValueError, match="data"):
        AStar("bic").fit()  # unbound score and nothing to bind
    same = AStar(score=bound).fit(data)  # the same dataset is fine
    assert same.n == 300
    rebound = AStar(score=BIC()).fit(data)  # an unbound instance binds at fit
    assert rebound.n == 300


def test_astar_rejects_an_unknown_score_name() -> None:
    with pytest.raises(ValueError, match="unknown score"):
        AStar("ges")
