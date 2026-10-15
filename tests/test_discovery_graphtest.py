# SPDX-License-Identifier: Apache-2.0
"""Tests for test_graph: basis sets, separation, calibration, the discrete path, errors."""

from __future__ import annotations

import json
import math
from itertools import combinations

import numpy as np
import pytest

import ergodic
from ergodic import GraphKind, admg, cpdag, dag, mixed, pag, tabular, timeseries
from ergodic.data import TabularData
from ergodic.datasets import load_earthquake
from ergodic.discovery import GCM, GraphTest, Linear, test_graph
from ergodic.discovery._ci import CITest, DSeparationOracle
from ergodic.discovery._graphtest import _fisher, _holm
from ergodic.discovery._stats import _chi2_sf
from ergodic.graph import (
    MixedGraph,
    cpdag_of_dag,
    dags_of_cpdag,
    mag_of_dag,
    mags_of_pag,
    pag_of_mag,
)

# -- helpers -----------------------------------------------------------------------


def _noise(names: frozenset[str] | list[str], n: int = 50, seed: int = 0) -> TabularData:
    """Independent normal columns: enough to name the variables."""
    rng = np.random.default_rng(seed)
    return tabular({v: rng.normal(size=n) for v in sorted(names)})


def _claims(graph: MixedGraph) -> set[tuple[str, str, tuple[str, ...], str]]:
    """The basis set, read through the graph's own oracle (no data needed)."""
    observed = graph.nodes - graph.latent - graph.selection
    result = test_graph(graph, _noise(observed), test=DSeparationOracle(graph))
    return {(i.x, i.y, i.given, i.rule) for i in result.implications}


def _linear_world(seed: int, n: int, p: int = 6) -> tuple[MixedGraph, TabularData, MixedGraph]:
    """A linear-Gaussian SEM on a random DAG with 6 to 8 edges, the calibration world.

    Weights are uniform on +-[0.5, 1.5] and the noise has unit variance. The third
    value is the true DAG minus one randomly chosen edge.
    """
    rng = np.random.default_rng(seed)
    names = [f"X{i}" for i in range(p)]
    order = [int(v) for v in rng.permutation(p)]
    pairs = [(order[i], order[j]) for i in range(p) for j in range(i + 1, p)]
    m = int(rng.integers(6, 9))
    chosen = [pairs[k] for k in rng.choice(len(pairs), size=m, replace=False)]
    weights = {pair: rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0]) for pair in chosen}
    values = np.zeros((n, p))
    for v in order:
        values[:, v] = rng.normal(size=n)
        for (a, b), w in weights.items():
            if b == v:
                values[:, v] += w * values[:, a]
    truth = dag([(names[a], names[b]) for a, b in chosen], nodes=names)
    data = tabular({names[i]: values[:, i] for i in range(p)})
    a, b = chosen[int(rng.integers(len(chosen)))]
    return truth, data, truth.remove_edge(names[a], names[b])


def _random_dag(rng: np.random.Generator, p: int, latent: int = 0) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = [(names[i], names[j]) for i, j in combinations(range(p), 2) if rng.uniform() < 0.4]
    hidden = list(rng.choice(names, size=latent, replace=False)) if latent else []
    return dag(edges, nodes=names, latent=hidden)


# -- basis sets on hand-checked graphs -------------------------------------------------


def test_a_chain_implies_its_ends_independent_given_the_middle() -> None:
    assert _claims(dag(["A -> B", "B -> C"])) == {("A", "C", ("B",), "parents")}


def test_a_collider_implies_its_causes_marginally_independent() -> None:
    assert _claims(dag(["A -> C", "B -> C"])) == {("A", "B", (), "parents")}


def test_a_fork_implies_its_effects_independent_given_the_cause() -> None:
    assert _claims(dag(["A -> B", "A -> C"])) == {("B", "C", ("A",), "parents")}


def test_a_five_node_dag_gets_shipleys_union_of_parents() -> None:
    graph = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    assert _claims(graph) == {
        ("A", "D", ("B", "C"), "parents"),
        ("A", "E", ("D",), "parents"),
        ("B", "C", ("A",), "parents"),
        ("B", "E", ("A", "D"), "parents"),
        ("C", "E", ("A", "D"), "parents"),
    }


def test_a_complete_graph_has_nothing_to_test() -> None:
    graph = dag(["A -> B", "A -> C", "B -> C"])
    result = test_graph(graph, _noise(graph.nodes))
    assert (result.k, result.dof, result.statistic, result.p_value) == (0, 0, 0.0, 1.0)
    assert not result.rejected
    assert any("Nothing to test" in note for note in result.notes)


def test_a_cpdag_reads_a_member_and_every_claim_holds_in_every_member() -> None:
    truth = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    klass = cpdag_of_dag(truth)
    claims = _claims(klass)
    assert {(x, y) if x < y else (y, x) for x, y, _, _ in claims} == {
        ("A", "D"),
        ("A", "E"),
        ("B", "C"),
        ("B", "E"),
        ("C", "E"),
    }
    members = dags_of_cpdag(klass)
    assert len(members) == 3
    for x, y, given, _ in claims:
        assert klass.m_separated(x, y, given)
        assert all(member.d_separated(x, y, given) for member in members)
    assert test_graph(klass, _noise(klass.nodes), test=DSeparationOracle(klass)).kind == "cpdag"


def test_an_admg_uses_ancestral_sets_and_skips_an_inducing_path() -> None:
    graph = admg(["u -> t", "t -> w", "w <-> v", "w -> s", "s -> v"])
    result = test_graph(graph, _noise(graph.nodes), test=DSeparationOracle(graph))
    assert {(i.x, i.y, i.given, i.rule) for i in result.implications} == {
        ("u", "w", ("t",), "parents"),
        ("u", "s", ("w",), "parents"),
        # The parents of v, {s}, open the collider at w (an ancestor of s).
        ("u", "v", ("t",), "ancestral"),
        ("t", "s", ("u", "w"), "parents"),
    }
    # t -> w <-> v: w is a collider and an ancestor of v, so no set separates them.
    assert result.untestable == (("t", "v"),)
    assert any("inducing path" in note for note in result.notes)
    assert result.kind == "admg"


def test_a_pag_reads_a_member_mag_and_claims_hold_class_wide() -> None:
    graph = pag(["A o-> B", "C o-> B", "B --> D"])
    assert _claims(graph) == {
        ("A", "C", (), "parents"),
        ("A", "D", ("B",), "parents"),
        ("C", "D", ("B",), "parents"),
    }
    for x, y, given, _ in _claims(graph):
        assert all(member.m_separated(x, y, given) for member in mags_of_pag(graph))


def test_latent_nodes_are_handled_by_m_separation_over_the_observed() -> None:
    # A <- L -> B, B -> C: L is unmeasured, so A and B are dependent and only A, C
    # carry a claim, given B.
    graph = dag(["L -> A", "L -> B", "B -> C"], latent=["L"])
    result = test_graph(graph, _noise(["A", "B", "C"]), test=DSeparationOracle(graph))
    assert [(i.x, i.y, i.given) for i in result.implications] == [("A", "C", ("B",))]
    assert result.untestable == (("A", "B"),)


# -- every conditioning set separates -----------------------------------------------


def test_every_claim_separates_on_random_dags_cpdags_and_latent_worlds() -> None:
    rng = np.random.default_rng(7)
    for _ in range(30):
        graph = _random_dag(rng, 6)
        for candidate in (graph, cpdag_of_dag(graph)):
            claims = _claims(candidate)
            all_pairs = combinations(graph.nodes, 2)
            non_adjacent = {frozenset(pair) for pair in all_pairs if not graph.has_edge(*pair)}
            assert {frozenset((x, y)) for x, y, _, _ in claims} == non_adjacent
            for x, y, given, _ in claims:
                assert graph.d_separated(x, y, given)
        hidden = _random_dag(rng, 7, latent=2)
        mag = mag_of_dag(hidden)
        klass = pag_of_mag(mag)
        for candidate in (hidden, mag, klass):
            claims = _claims(candidate)
            for x, y, given, _ in claims:
                # Every claim is true in the world the candidate stands for.
                assert hidden.m_separated(x, y, given)
                assert candidate.m_separated(x, y, given)
            # The MAG is maximal, so every non-adjacent observed pair is separable.
            pairs = {frozenset((x, y)) for x, y, _, _ in claims}
            expected = {
                frozenset(pair) for pair in combinations(mag.nodes, 2) if not mag.has_edge(*pair)
            }
            assert pairs == expected


def test_ancestral_sets_are_minimal() -> None:
    rng = np.random.default_rng(11)
    found = 0
    for _ in range(40):
        hidden = _random_dag(rng, 7, latent=2)
        for x, y, given, rule in _claims(hidden):
            if rule != "ancestral":
                continue
            found += 1
            for drop in given:
                assert not hidden.m_separated(x, y, set(given) - {drop})
    assert found > 0


# -- behaviour on data -------------------------------------------------------------------


def test_the_true_graph_passes_and_a_missing_edge_is_rejected() -> None:
    # Seed 0 gives p = 0.94 for the truth. Under the truth p is uniform, so some
    # seeds reject (seed 3 gives 0.049): one seed is an example, the size test
    # below is the calibration.
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.7 * b + rng.normal(size=n)
    d = 0.6 * a - 0.9 * c + rng.normal(size=n)
    data = tabular({"a": a, "b": b, "c": c, "d": d})
    truth = dag(["a -> b", "b -> c", "a -> d", "c -> d"])
    passed = test_graph(truth, data)
    assert passed.test == "linear"
    assert passed.k == 2
    assert passed.p_value > 0.05
    assert not passed.rejected and passed.n_rejected == 0
    failed = test_graph(truth.remove_edge("c", "d"), data)
    assert failed.rejected and failed.p_value < 1e-10
    # Without c -> d, both c and b (through c) carry false claims about d.
    rejected = {frozenset((i.x, i.y)) for i in failed.implications if i.rejected}
    assert rejected == {frozenset(("c", "d")), frozenset(("b", "d"))}
    ranked = [i.p_value for i in failed.implications]
    assert ranked == sorted(ranked)


def test_size_over_seeded_worlds_is_near_alpha_and_power_is_high() -> None:
    worlds = 300
    size = power = 0
    for seed in range(worlds):
        truth, data, missing = _linear_world(seed, n=200)
        size += test_graph(truth, data).rejected
        power += test_graph(missing, data).rejected
    # Measured 0.053 over these 300 worlds (0.055 over 3,000; 0.057 and 0.046 at
    # n = 1,000); the band is the calibration protocol's registered [0.02, 0.08].
    assert 0.02 <= size / worlds <= 0.08
    assert power / worlds > 0.9


def test_the_discrete_path_uses_g_squared() -> None:
    bunch = load_earthquake(n=5000, seed=1)
    truth = dag(list(bunch.target["edges"]))
    result = test_graph(truth, bunch.data)
    assert result.test == "discrete"
    assert result.k > 0
    assert not result.rejected
    assert all(i.dof is not None for i in result.implications)
    # Drop the alarm's edge to John: John and the alarm become a (false) claim.
    assert test_graph(truth.remove_edge("Alarm", "JohnCalls"), bunch.data).rejected


def test_mixed_columns_use_the_mixed_test_and_test_overrides() -> None:
    rng = np.random.default_rng(5)
    n = 800
    group = rng.choice(["a", "b", "c"], size=n)
    shift = {"a": 0.0, "b": 1.0, "c": -1.0}
    x = np.array([shift[g] for g in group]) + rng.normal(size=n)
    y = 0.8 * x + rng.normal(size=n)
    data = tabular({"group": group, "x": x, "y": y})
    graph = dag(["group -> x", "x -> y"])
    assert test_graph(graph, data).test == "mixed"
    continuous = data.select(["x", "y"])
    assert test_graph(dag(["x -> y"]), continuous).test == "linear"
    assert test_graph(graph, data, test="mixed").test == "mixed"
    assert test_graph(dag(["x -> y"], nodes=["z"]), _noise(["x", "y", "z"]), test=Linear()).k == 2


# -- the combination ----------------------------------------------------------------------


def test_holm_and_fisher_match_hand_computation() -> None:
    assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert _holm([0.5, 0.001]) == pytest.approx([0.5, 0.002])
    c, p = _fisher([0.2, 0.5])
    assert c == pytest.approx(-2.0 * (math.log(0.2) + math.log(0.5)))
    assert p == pytest.approx(_chi2_sf(c, 4.0))
    c0, p0 = _fisher([0.0, 0.5])
    assert math.isfinite(c0) and p0 == 0.0
    assert _fisher([]) == (0.0, 1.0)
    # A subnormal p-value is a real tail, not a zero.
    assert 0.0 < _fisher([1e-320])[1] < 1e-300
    for bad in ([float("nan")], [0.2, float("inf")], [1.5], [-0.1]):
        with pytest.raises(ValueError, match="finite numbers"):
            _fisher(bad)
        with pytest.raises(ValueError, match="finite numbers"):
            _holm(bad)


def test_the_result_serializes_and_reads_as_a_frame() -> None:
    graph = dag(["A -> B", "B -> C", "C -> D"])
    rng = np.random.default_rng(2)
    a = rng.normal(size=300)
    b = a + rng.normal(size=300)
    c = b + rng.normal(size=300)
    d = c + rng.normal(size=300)
    result = test_graph(graph, tabular({"A": a, "B": b, "C": c, "D": d, "E": d}))
    assert isinstance(result, GraphTest)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["k"] == 3 and payload["dof"] == 6 and payload["test"] == "linear"
    assert payload["implications"][0]["given"] in (["B"], ["C"], ["B", "C"])
    assert any("ignored" in note for note in payload["notes"])
    frame = result.to_frame()
    assert list(frame["p_value"]) == [i.p_value for i in result.implications]
    assert "GraphTest(dag, linear test, k=3" in repr(result)
    assert result.implications[0].p_holm >= result.implications[0].p_value


def test_test_graph_is_exported_and_not_collected_by_pytest() -> None:
    assert ergodic.test_graph is test_graph
    assert getattr(test_graph, "__test__", True) is False


# -- errors --------------------------------------------------------------------------------


def test_unknown_names_say_which_columns_are_missing() -> None:
    with pytest.raises(ValueError, match=r"\['Z'\] are not columns"):
        test_graph(dag(["A -> B", "B -> Z"]), _noise(["A", "B"]))


def test_cycles_and_unreadable_graphs_raise() -> None:
    with pytest.raises(ValueError, match="directed cycle"):
        test_graph(mixed(["A -> B", "B -> C", "C -> A"]), _noise(["A", "B", "C"]))
    with pytest.raises(ValueError, match="as_kind"):
        test_graph(mixed(["A o-> B", "B --> C"]), _noise(["A", "B", "C"]))
    with pytest.raises(ValueError, match="no latent"):
        test_graph(cpdag(["A --- B"], nodes=["L"], latent=["L"]), _noise(["A", "B"]))


def test_bad_inputs_raise() -> None:
    graph = dag(["A -> B"])
    data = _noise(["A", "B"])
    with pytest.raises(TypeError, match="MixedGraph"):
        test_graph("A -> B", data)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tabular"):
        test_graph(graph, {"A": [1.0], "B": [2.0]})  # type: ignore[arg-type]
    series = timeseries({"t": np.arange(10), "A": np.arange(10.0), "B": np.ones(10)}, time="t")
    with pytest.raises(TypeError, match="not exchangeable"):
        test_graph(graph, series)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        test_graph(graph, data, alpha=1.0)
    with pytest.raises(ValueError, match="already bound"):
        test_graph(graph, data, test=Linear(data))
    with pytest.raises(ValueError, match="unknown test"):
        test_graph(graph, data, test="nope")


# -- review fixes: non-finite input, calibration notes, vacuous claims ---------------------


def _chain_data(n: int = 400, seed: int = 4) -> TabularData:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a + rng.normal(size=n)
    c = b + rng.normal(size=n)
    return tabular({"a": a, "b": b, "c": c})


def test_an_infinite_value_is_refused_not_read_as_a_pass() -> None:
    rng = np.random.default_rng(1)
    n = 400
    a = rng.normal(size=n)
    b = a + rng.normal(size=n)
    c = 2.0 * b + rng.normal(size=n)
    graph = dag(["a -> b", "a -> c"])
    assert test_graph(graph, tabular({"a": a, "b": b, "c": c})).rejected
    c[7] = np.inf
    with pytest.raises(ValueError, match=r"\['c'\] hold infinite values"):
        test_graph(graph, tabular({"a": a, "b": b, "c": c}))
    # An infinite value outside the candidate's columns is not its business.
    clean = tabular({"a": a, "b": b, "c": 2.0 * b + rng.normal(size=n), "z": c})
    assert test_graph(graph, clean).rejected


def test_missing_values_are_dropped_with_a_note() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=300)
    b = a + rng.normal(size=300)
    c = b + rng.normal(size=300)
    c[:5] = np.nan
    result = test_graph(dag(["a -> b", "b -> c"]), tabular({"a": a, "b": b, "c": c}))
    assert result.n == 295
    assert any("5 row(s) with a missing value" in note for note in result.notes)


class _BrokenTest(CITest):
    name = "broken"
    needs = "numeric"

    def statistic(self, *args: object) -> tuple[float, float, float | None]:
        return float("nan"), float("nan"), None


def test_a_non_finite_p_value_from_a_test_is_an_error() -> None:
    with pytest.raises(ValueError, match="never a pass"):
        test_graph(dag(["a -> b", "b -> c"]), _chain_data(), test=_BrokenTest())


def test_gcm_with_nearest_neighbours_carries_a_calibration_note() -> None:
    graph = dag(["a -> b", "b -> c"])
    knn = test_graph(graph, _chain_data(), test="gcm")
    assert knn.test == "gcm"
    assert any("not calibrated" in note and "learner='ols'" in note for note in knn.notes)
    ols = test_graph(graph, _chain_data(), test=GCM(learner="ols"))
    assert not any("not calibrated" in note for note in ols.notes)
    assert any("has not been measured" in note for note in ols.notes)
    default = test_graph(graph, _chain_data())
    assert not any("measured" in note or "calibrated" in note for note in default.notes)


def test_g_squared_at_small_n_carries_a_size_note() -> None:
    truth = dag(list(load_earthquake(n=300, seed=1).target["edges"]))
    small = test_graph(truth, load_earthquake(n=300, seed=1).data)
    assert small.test == "discrete"
    assert any("can run over alpha" in note for note in small.notes)
    large = test_graph(truth, load_earthquake(n=5000, seed=1).data)
    assert not any("can run over alpha" in note for note in large.notes)


def test_zero_dof_claims_are_vacuous_and_left_out_of_k() -> None:
    rng = np.random.default_rng(3)
    n = 500
    a = rng.choice(["x", "y"], size=n)
    b = np.where(rng.uniform(size=n) < 0.8, a, np.where(a == "x", "y", "x"))
    data = tabular(
        {"a": a, "b": b, "c": np.array(["only"] * n)},
        types={"a": "categorical", "b": "categorical", "c": "categorical"},
    )
    result = test_graph(dag(["a -> b"], nodes=["c"]), data)
    assert result.test == "discrete"
    assert (result.k, result.dof, result.p_value) == (0, 0, 1.0)
    assert {(i.x, i.y) for i in result.vacuous} == {("a", "c"), ("b", "c")}
    assert all(i.dof == 0.0 for i in result.vacuous)
    assert any("zero degrees of freedom" in note for note in result.notes)
    assert len(result.to_dict()["vacuous"]) == 2


def test_linear_p_values_keep_their_order_far_in_the_tail() -> None:
    rng = np.random.default_rng(6)
    n = 1500
    a = rng.normal(size=n)
    b = a + 2.0 * rng.normal(size=n)
    c = b + 3.0 * rng.normal(size=n)
    result = test_graph(dag(["a -> b"], nodes=["c"]), tabular({"a": a, "b": b, "c": c}))
    p = [i.p_value for i in result.implications]
    stats = [abs(i.statistic) for i in result.implications]
    # Both |z| are above 8.3, where 1 - Phi underflowed to zero and tied them.
    assert min(stats) > 9.0 and len(stats) == 2
    assert all(value > 0.0 for value in p)
    assert stats == sorted(stats, reverse=True)


# -- review fixes: selection, MAGs, the few-rows and hidden-column notes ---------------------


def test_selection_nodes_condition_every_claim() -> None:
    # A -> S <- B with S selected: conditioning on S links A and B, so they are
    # untestable; A and C stay separable given B.
    graph = dag(["A -> S", "B -> S", "B -> C"], selection=["S"])
    result = test_graph(graph, _noise(["A", "B", "C"]), test=DSeparationOracle(graph))
    assert [(i.x, i.y, i.given) for i in result.implications] == [("A", "C", ("B",))]
    assert result.untestable == (("A", "B"),)
    assert graph.m_separated("A", "C", {"B", "S"})
    assert not graph.m_separated("A", "C", {"S"})


def test_a_mag_with_an_undirected_edge_is_read_by_m_separation() -> None:
    # A --- B: both are ancestors of selection. B --> C, and C <-> D.
    mag = mixed(["A --- B", "B --> C", "C <-> D"], kind=GraphKind.MAG)
    result = test_graph(mag, _noise(mag.nodes), test=DSeparationOracle(mag))
    assert result.kind == "mag"
    claims = {(i.x, i.y, i.given) for i in result.implications}
    assert claims == {("A", "C", ("B",)), ("A", "D", ()), ("B", "D", ())}
    for x, y, given in claims:
        assert mag.m_separated(x, y, given)


def _separable(graph: MixedGraph, x: str, y: str, observed: list[str]) -> bool:
    rest = [v for v in observed if v not in (x, y)]
    return any(
        graph.m_separated(x, y, set(given) | graph.selection)
        for size in range(len(rest) + 1)
        for given in combinations(rest, size)
    )


def test_untestable_pairs_match_an_exhaustive_search() -> None:
    rng = np.random.default_rng(21)
    checked = untestable = 0
    for _ in range(30):
        names = [f"V{i}" for i in range(7)]
        edges = [(names[i], names[j]) for i, j in combinations(range(7), 2) if rng.uniform() < 0.4]
        picks = list(rng.choice(names, size=3, replace=False))
        world = dag(edges, nodes=names, latent=picks[:2], selection=picks[2:])
        observed = sorted(world.nodes - world.latent - world.selection)
        for candidate in (world, mag_of_dag(world)):
            result = test_graph(candidate, _noise(observed), test=DSeparationOracle(candidate))
            claimed = {frozenset((i.x, i.y)) for i in result.implications}
            for x, y in combinations(observed, 2):
                if candidate.has_edge(x, y):
                    continue
                checked += 1
                separable = _separable(candidate, x, y, observed)
                assert (frozenset((x, y)) in claimed) == separable
                untestable += not separable
            assert {frozenset(pair) for pair in result.untestable} == {
                frozenset((x, y))
                for x, y in combinations(observed, 2)
                if not candidate.has_edge(x, y) and not _separable(candidate, x, y, observed)
            }
    assert checked > 100 and untestable > 0


def test_few_rows_and_hidden_columns_get_notes() -> None:
    graph = dag(["L -> a", "a -> b", "b -> c"], latent=["L"])
    data = _chain_data(n=30)
    noisy = tabular({**{v: data.column(v) for v in ("a", "b", "c")}, "L": data.column("a")})
    result = test_graph(graph, noisy)
    assert any("Few rows (30)" in note for note in result.notes)
    assert any("not tested, though in the data: ['L']" in note for note in result.notes)
