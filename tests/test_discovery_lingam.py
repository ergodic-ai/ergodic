# SPDX-License-Identifier: Apache-2.0
"""Tests for DirectLiNGAM: order recovery, pruning, knowledge, and the guards.

The load-bearing checks are sampled: twenty random sparse linear worlds with
uniform noise, where the recovered DAG must equal the truth exactly in a
pinned count of worlds and the recovered order must agree with every true
edge. Everything is seeded, so every assertion is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from ergodic.data import TabularData, tabular, timeseries
from ergodic.discovery.algorithms._lingam import DirectLiNGAM
from ergodic.graph import GraphKind, MixedGraph, dag
from ergodic.knowledge import DomainKnowledge

# -- the sampled worlds ----------------------------------------------------------


def _random_world(
    rng: np.random.Generator, p: int = 5, n: int = 3000, edge_prob: float = 0.4
) -> tuple[MixedGraph, TabularData, list[tuple[str, str]]]:
    """A random sparse linear DAG with uniform noise, sampled at ``n`` rows.

    The causal order is a random permutation of the names (so the alphabet
    carries no signal), coefficients have magnitude in [0.5, 1.5] with a
    random sign, and every noise is uniform(-1, 1), strongly non-Gaussian.
    """
    names = [chr(ord("A") + k) for k in range(p)]
    causal = [names[i] for i in rng.permutation(p)]
    coefs: dict[tuple[str, str], float] = {}
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < edge_prob:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                coefs[(causal[i], causal[j])] = sign * rng.uniform(0.5, 1.5)
    cols: dict[str, np.ndarray] = {}
    for pos, v in enumerate(causal):
        x = rng.uniform(-1.0, 1.0, size=n)
        for u in causal[:pos]:
            if (u, v) in coefs:
                x = x + coefs[(u, v)] * cols[u]
        cols[v] = x
    truth = dag([f"{u} -> {v}" for u, v in coefs], nodes=names)
    return truth, tabular({v: cols[v] for v in names}), sorted(coefs)


def _order_of(full_triangular: MixedGraph) -> list[str]:
    """The unique topological order of a complete DAG, read off the in-degrees."""
    return sorted(full_triangular.nodes, key=lambda v: len(full_triangular.parents(v)))


def test_exact_recovery_on_random_nongaussian_worlds() -> None:
    # Seed 7, measured once and pinned: 19 of 20 worlds recovered exactly.
    # The one miss (world 18) keeps a single spurious edge under BIC with the
    # order still correct, the expected finite-sample failure mode.
    rng = np.random.default_rng(7)
    exact = 0
    for _ in range(20):
        truth, data, _ = _random_world(rng)
        result = DirectLiNGAM().fit(data)
        assert result.graph.kind is GraphKind.DAG
        if result.graph == truth:
            exact += 1
    assert exact == 19


def test_causal_order_agrees_with_every_true_edge() -> None:
    # Same twenty worlds (seed 7), measured once and pinned: in 20 of 20 the
    # recovered order puts every true cause before its effect.
    rng = np.random.default_rng(7)
    consistent = 0
    for _ in range(20):
        _, data, edges = _random_world(rng)
        full = DirectLiNGAM(prune="none").fit(data).graph
        assert sorted(len(full.parents(v)) for v in full.nodes) == [0, 1, 2, 3, 4]
        position = {v: k for k, v in enumerate(_order_of(full))}
        if all(position[u] < position[v] for u, v in edges):
            consistent += 1
    assert consistent == 20


# -- the hand chain ---------------------------------------------------------------


def _chain(n: int = 5000, seed: int = 11) -> tuple[TabularData, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = x + rng.uniform(-1.0, 1.0, n)
    z = y + rng.uniform(-1.0, 1.0, n)
    return tabular({"x": x, "y": y, "z": z}), x, y, z


def test_hand_chain_exact_with_full_provenance() -> None:
    data, _, _, _ = _chain()
    result = DirectLiNGAM().fit(data)
    assert result.graph == dag(["x -> y", "y -> z"])
    assert result.method == "lingam"
    assert result.test is None
    assert result.score is None
    assert result.alpha is None
    assert result.n == 5000
    assert result.sepsets is None
    assert result.diagnostics["n_root_picks"] == 3.0
    assert result.diagnostics["n_pruned_edges"] == 1.0  # x leaves z's regression
    assert result.diagnostics["n_knowledge_pruned"] == 0.0
    assert DirectLiNGAM.ingredient is None  # neither a test nor a score


def test_scale_robustness_one_column_times_100() -> None:
    _, x, y, z = _chain()
    scaled = tabular({"x": x, "y": 100.0 * y, "z": z})
    assert DirectLiNGAM().fit(scaled).graph == dag(["x -> y", "y -> z"])


def test_missing_rows_drop_before_the_fit() -> None:
    _, x, y, z = _chain()
    x = x.copy()
    x[0] = np.nan
    result = DirectLiNGAM().fit(tabular({"x": x, "y": y, "z": z}))
    assert result.n == 4999
    assert result.graph == dag(["x -> y", "y -> z"])


def test_gaussian_world_runs_but_the_direction_is_arbitrary() -> None:
    # Gaussian noise is the documented unidentifiable case: the fit must run
    # and return a DAG over the right nodes, with no claim about directions.
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    y = x + rng.normal(size=2000)
    z = y + rng.normal(size=2000)
    result = DirectLiNGAM().fit(tabular({"x": x, "y": y, "z": z}))
    assert result.graph.kind is GraphKind.DAG
    assert result.graph.nodes == frozenset({"x", "y", "z"})


# -- input guards -----------------------------------------------------------------


def test_categorical_column_raises_naming_it() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"a": rng.normal(size=40), "b": ["u", "v"] * 20})
    with pytest.raises(ValueError, match=r"\['b'\].*not continuous"):
        DirectLiNGAM().fit(data)


def test_excluded_columns_drop_like_elsewhere() -> None:
    rng = np.random.default_rng(11)
    x = rng.uniform(-1.0, 1.0, 1000)
    y = 1.2 * x + rng.uniform(-1.0, 1.0, 1000)
    data = tabular({"x": x, "y": y, "b": ["u", "v"] * 500}).exclude("b")
    result = DirectLiNGAM().fit(data)
    assert result.graph.nodes == frozenset({"x", "y"})


def test_temporal_data_raises_typeerror() -> None:
    rng = np.random.default_rng(0)
    ts = timeseries(
        {"t": np.arange(50), "X": rng.normal(size=50), "Y": rng.normal(size=50)}, time="t"
    )
    with pytest.raises(TypeError, match="TimeSeriesData"):
        DirectLiNGAM().fit(ts)  # type: ignore[arg-type]


# -- knowledge --------------------------------------------------------------------


def _pair_world(seed: int = 5, n: int = 3000) -> TabularData:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 1.2 * x + rng.uniform(-1.0, 1.0, n)
    return tabular({"x": x, "y": y})


def test_tiers_force_the_root_against_the_data_preference() -> None:
    data = _pair_world()
    assert DirectLiNGAM().fit(data).graph == dag(["x -> y"])  # the data preference
    tiers = DomainKnowledge().with_tiers([["y"], ["x"]])
    result = DirectLiNGAM().fit(data, knowledge=tiers)
    assert result.graph == dag(["y -> x"])  # the tier gate wins


def test_declared_roots_and_sinks_gate_the_root_choice() -> None:
    data = _pair_world()
    rooted = DirectLiNGAM().fit(data, knowledge=DomainKnowledge().root("y"))
    assert rooted.graph == dag(["y -> x"])
    sunk = DirectLiNGAM().fit(data, knowledge=DomainKnowledge().sink("x"))
    assert sunk.graph == dag(["y -> x"])


def test_forbidden_edge_is_absent_from_the_output() -> None:
    data = _pair_world()
    result = DirectLiNGAM().fit(data, knowledge=DomainKnowledge().forbid("x", "y"))
    assert not result.graph.edges
    assert result.diagnostics["n_knowledge_pruned"] == 1.0


def test_required_edge_survives_bic_elimination() -> None:
    # x -> w is strong, x -> y is far below what BIC keeps at this n.
    rng = np.random.default_rng(21)
    n = 2000
    x = rng.uniform(-1.0, 1.0, n)
    w = 1.5 * x + rng.uniform(-1.0, 1.0, n)
    y = 0.04 * x + rng.uniform(-1.0, 1.0, n)
    data = tabular({"x": x, "w": w, "y": y})
    base = DirectLiNGAM().fit(data)
    assert base.graph.get_edge("x", "y") is None  # BIC drops the tiny edge
    kept = DirectLiNGAM().fit(data, knowledge=DomainKnowledge().require("x", "y"))
    edge = kept.graph.get_edge("x", "y")
    assert edge is not None
    assert edge.directed_pair() == ("x", "y")


def test_required_edge_against_the_order_raises() -> None:
    data = _pair_world(seed=2)
    backwards = DomainKnowledge().require("y", "x")
    with pytest.raises(ValueError, match="contradicts the causal order"):
        DirectLiNGAM().fit(data, knowledge=backwards)


# -- pruning modes and determinism --------------------------------------------------


def test_prune_none_gives_the_full_triangular_dag_of_the_order() -> None:
    data, _, _, _ = _chain()
    result = DirectLiNGAM(prune="none").fit(data)
    assert result.graph == dag(["x -> y", "x -> z", "y -> z"])
    assert result.diagnostics["n_pruned_edges"] == 0.0


def test_unknown_prune_raises() -> None:
    with pytest.raises(ValueError, match="prune must be"):
        DirectLiNGAM(prune="bogus")


def test_two_fits_are_identical() -> None:
    rng = np.random.default_rng(7)
    _, data, _ = _random_world(rng)
    first = DirectLiNGAM().fit(data)
    second = DirectLiNGAM().fit(data)
    assert first.graph == second.graph
    assert first.diagnostics == second.diagnostics
    assert first.n == second.n
