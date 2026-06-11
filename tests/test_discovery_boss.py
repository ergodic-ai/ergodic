# SPDX-License-Identifier: Apache-2.0
"""Tests for BOSS, the best order score search with grow-shrink projection.

The executable anchor is agreement with the brute-force-verified A* optimum
on a shared warmed score, mirroring the GES test. Order independence is
probed by renaming variables (the initial permutation is the sorted names,
so renaming is the axis that could bite), BES is pinned on a weak-signal
world where it strictly improves the sweep's result, and the grow-shrink
projection itself is held to a hand derivation. Every run uses a fixed seed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from ergodic.data import tabular
from ergodic.discovery.algorithms._astar import AStar
from ergodic.discovery.algorithms._boss import BOSS, _grow_shrink, _project
from ergodic.discovery.scores._bic import BIC
from ergodic.discovery.scores._discrete_bic import DiscreteBIC
from ergodic.graph import GraphKind, MixedGraph, consistent_extension, cpdag_of_dag, dag
from ergodic.knowledge import DomainKnowledge

if TYPE_CHECKING:
    from ergodic.data import TabularData

# -- helpers ---------------------------------------------------------------------


def _directed(graph: MixedGraph, parent: str, child: str) -> bool:
    edge = graph.get_edge(parent, child)
    return edge is not None and edge.directed_pair() == (parent, child)


def _undirected(graph: MixedGraph, a: str, b: str) -> bool:
    edge = graph.get_edge(a, b)
    return edge is not None and edge.is_undirected


def _skeleton(graph: MixedGraph) -> set[frozenset[str]]:
    return {frozenset((edge.u, edge.v)) for edge in graph.edges}


def _parent_map(graph: MixedGraph) -> dict[str, frozenset[str]]:
    return {node: graph.parents(node) for node in sorted(graph.nodes)}


def _class_score(score: BIC, graph: MixedGraph) -> float:
    """The shared score of a CPDAG's members, read off one extension."""
    member = consistent_extension(graph)
    assert member is not None  # a valid CPDAG always extends
    return score.graph_score(_parent_map(member))


def _canon(graph: MixedGraph, rename: dict[str, str]) -> set[tuple[str, ...]]:
    """A graph's edges as name-mapped tuples, comparable across relabelings."""
    out: set[tuple[str, ...]] = set()
    for edge in graph.edges:
        pair = edge.directed_pair()
        if pair is not None:
            out.add(("->", rename.get(pair[0], pair[0]), rename.get(pair[1], pair[1])))
        else:
            ends = sorted((rename.get(edge.u, edge.u), rename.get(edge.v, edge.v)))
            out.add(("--", ends[0], ends[1]))
    return out


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


def _weak_world(seed: int, p: int = 6, n: int = 80) -> TabularData:
    """The same SEM with weights in [0.3, 0.9] and few rows: noisy on purpose."""
    rng = np.random.default_rng(seed)
    names = list("abcdef"[:p])
    values: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        column = rng.normal(size=n)
        for parent in names[:i]:
            if rng.random() < 0.5:
                weight = rng.uniform(0.3, 0.9) * (1.0 if rng.random() < 0.5 else -1.0)
                column = column + weight * values[parent]
        values[name] = column
    return tabular(values)


def _chain_data(n: int = 3000, seed: int = 6) -> TabularData:
    """X -> M -> Y, linear Gaussian with strong coefficients."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 1.1 * x + rng.normal(size=n)
    y = 0.9 * m + rng.normal(size=n)
    return tabular({"X": x, "M": m, "Y": y})


def _five_node_world() -> tuple[dict[str, np.ndarray], MixedGraph]:
    """The canonical five-node class with strong coefficients."""
    rng = np.random.default_rng(3)
    n = 5000
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.7 * a + rng.normal(size=n)
    d = 0.9 * b + 0.8 * c + rng.normal(size=n)
    e = 0.7 * d + rng.normal(size=n)
    truth = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    return {"A": a, "B": b, "C": c, "D": d, "E": e}, truth


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
def test_boss_matches_the_exact_optimum(seed: int, p: int) -> None:
    data = _random_dag_data(seed, p=p)
    bic = BIC(data)
    found = BOSS(score=bic).fit()
    exact = AStar(score=bic).fit()
    assert math.isclose(
        _class_score(bic, found.graph),
        bic.graph_score(_parent_map(exact.graph)),
        rel_tol=1e-9,
    )
    assert found.graph.kind is GraphKind.CPDAG
    assert found.graph == cpdag_of_dag(exact.graph)


# -- order independence: renaming permutes the start and the tie-breaks ---------------


def test_boss_is_order_independent_under_renaming() -> None:
    columns, truth = _five_node_world()
    plain = BOSS("bic").fit(tabular(columns))
    mapping = {"A": "v4", "B": "v1", "C": "v3", "D": "v0", "E": "v2"}
    renamed = BOSS("bic").fit(tabular({mapping[name]: col for name, col in columns.items()}))
    inverse = {new: old for old, new in mapping.items()}
    assert _canon(plain.graph, {}) == _canon(renamed.graph, inverse)
    assert plain.graph == cpdag_of_dag(truth)


# -- a discrete world ------------------------------------------------------------------


def test_boss_recovers_a_discrete_network_with_discrete_bic() -> None:
    data, truth = _discrete_network()
    result = BOSS(score=DiscreteBIC(data)).fit()
    assert result.graph == cpdag_of_dag(truth)
    assert result.graph.kind is GraphKind.CPDAG
    assert result.score == "bic_discrete"


# -- BES: off is valid, and on a pinned weak world it strictly improves ------------------


def test_boss_without_bes_is_valid_and_agrees_on_a_clean_world() -> None:
    columns, truth = _five_node_world()
    data = tabular(columns)
    with_bes = BOSS("bic").fit(data)
    without = BOSS("bic", use_bes=False).fit(data)
    assert without.graph.kind is GraphKind.CPDAG
    assert without.diagnostics["cpdag_valid"] == 1.0
    assert without.diagnostics["n_bes_deletes"] == 0.0
    assert without.graph == with_bes.graph == cpdag_of_dag(truth)


def test_bes_improves_a_pinned_weak_world() -> None:
    # Calibrated: at seed 55 with 6 nodes and 80 rows the converged sweep
    # leaves one Chickering delete open, worth about 1.39 in score.
    data = _weak_world(55)
    bic = BIC(data)
    with_bes = BOSS(score=bic).fit()
    without = BOSS(score=bic, use_bes=False).fit()
    assert with_bes.diagnostics["n_bes_deletes"] >= 1.0
    gain = _class_score(bic, with_bes.graph) - _class_score(bic, without.graph)
    assert gain > 1.0
    assert with_bes.graph != without.graph


# -- knowledge ---------------------------------------------------------------------------


def test_boss_knowledge_tiers_orient_the_chain_fully() -> None:
    data = _chain_data()
    plain = BOSS("bic").fit(data)
    assert _undirected(plain.graph, "X", "M")  # the chain class leaves both edges loose
    tiers = DomainKnowledge().with_tiers([["X"], ["M", "Y"]])
    result = BOSS("bic").fit(data, knowledge=tiers)
    assert _directed(result.graph, "X", "M")  # the tier order forces it
    assert _directed(result.graph, "M", "Y")  # and Meek carries it down the chain
    assert tiers.is_consistent(result.graph)


def test_boss_knowledge_forbidden_pair_never_appears() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.normal(size=n)
    y = 1.2 * x + rng.normal(size=n)
    z = 0.9 * y + rng.normal(size=n)
    data = tabular({"X": x, "Y": y, "Z": z})
    forbidden = DomainKnowledge().forbid_adjacency("X", "Y")
    result = BOSS("bic").fit(data, knowledge=forbidden)
    assert result.graph.get_edge("X", "Y") is None  # the ban beats the strong signal
    assert result.graph.get_edge("Y", "Z") is not None
    assert forbidden.is_consistent(result.graph)


def test_boss_knowledge_required_edge_survives_without_signal() -> None:
    rng = np.random.default_rng(8)
    data = tabular({"U": rng.normal(size=600), "V": rng.normal(size=600)})
    required = DomainKnowledge().require("U", "V")
    result = BOSS("bic").fit(data, knowledge=required)
    assert _directed(result.graph, "U", "V")
    assert required.is_consistent(result.graph)
    # a lone forced arrow is no CPDAG, so the face falls back to mixed
    assert result.diagnostics["cpdag_valid"] == 0.0
    assert result.graph.kind is GraphKind.MIXED
    # the harder direction: the requirement runs against the sorted start,
    # so the order pinning is what keeps the parent in the prefix
    reversed_required = DomainKnowledge().require("V", "U")
    against = BOSS("bic").fit(data, knowledge=reversed_required)
    assert _directed(against.graph, "V", "U")
    assert reversed_required.is_consistent(against.graph)


def test_boss_knowledge_skeleton_restriction_confines_the_search() -> None:
    rng = np.random.default_rng(9)
    n = 2000
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.8 * a + rng.normal(size=n)
    d = 0.7 * b + 1.1 * c + rng.normal(size=n)
    data = tabular({"a": a, "b": b, "c": c, "d": d})
    support = {frozenset(("a", "b")), frozenset(("b", "d"))}
    knowledge = DomainKnowledge().restrict_to([("a", "b"), ("b", "d")])
    result = BOSS("bic").fit(data, knowledge=knowledge)
    assert _skeleton(result.graph) == support  # both supported pairs carry real signal
    assert knowledge.is_consistent(result.graph)


def test_boss_required_edge_cycle_raises() -> None:
    rng = np.random.default_rng(10)
    data = tabular({name: rng.normal(size=100) for name in ("a", "b", "c")})
    cyclic = DomainKnowledge().require("a", "b").require("b", "c").require("c", "a")
    with pytest.raises(ValueError, match="required-edge cycle"):
        BOSS("bic").fit(data, knowledge=cyclic)


# -- determinism ---------------------------------------------------------------------------


def test_boss_is_deterministic() -> None:
    data = _random_dag_data(2)
    first = BOSS("bic").fit(data)
    second = BOSS("bic").fit(data)
    assert first.graph == second.graph
    assert dict(first.diagnostics) == dict(second.diagnostics)


def test_boss_restarts_are_deterministic_given_a_seed() -> None:
    data = _random_dag_data(3)
    first = BOSS("bic", restarts=3, seed=11).fit(data)
    second = BOSS("bic", restarts=3, seed=11).fit(data)
    assert first.graph == second.graph
    assert dict(first.diagnostics) == dict(second.diagnostics)
    # seed=None means a fixed default seed, so even that is reproducible
    third = BOSS("bic", restarts=2).fit(data)
    fourth = BOSS("bic", restarts=2).fit(data)
    assert third.graph == fourth.graph


# -- the grow-shrink projection, held to a hand derivation -----------------------------------


def test_grow_shrink_projects_the_chain_by_hand() -> None:
    # x -> m -> y: given the prefix (x, m), grow on y adds m (the stronger
    # correlate) and then stops because x adds nothing beyond m; given
    # (x, y), grow on m needs both, since m depends on each given the other.
    rng = np.random.default_rng(6)
    n = 3000
    x = rng.normal(size=n)
    m = 1.1 * x + rng.normal(size=n)
    y = 0.9 * m + rng.normal(size=n)
    bic = BIC(tabular({"x": x, "m": m, "y": y}))
    assert _grow_shrink(bic, "y", ("x", "m"), None) == frozenset({"m"})
    assert _grow_shrink(bic, "m", ("x", "y"), None) == frozenset({"x", "y"})
    assert _grow_shrink(bic, "x", (), None) == frozenset()
    parents, total = _project(bic, ["x", "m", "y"], None)
    assert parents == {"x": frozenset(), "m": frozenset({"x"}), "y": frozenset({"m"})}
    assert math.isclose(total, bic.graph_score(parents), rel_tol=1e-12)
    # a misordered permutation projects to a denser, lower-scoring DAG
    misordered, worse = _project(bic, ["y", "x", "m"], None)
    assert misordered["m"] == frozenset({"x", "y"})
    assert worse < total


# -- the result record and argument validation ------------------------------------------------


def test_boss_result_record_and_diagnostics() -> None:
    data = _chain_data(n=1500, seed=12)
    result = BOSS("auto").fit(data)  # continuous schema resolves to the Gaussian BIC
    for key in (
        "n_score_evals",
        "cache_hits",
        "n_sweeps",
        "n_moves",
        "n_bes_deletes",
        "cpdag_valid",
    ):
        assert key in result.diagnostics
    assert result.diagnostics["cpdag_valid"] == 1.0
    assert result.method == "boss"
    assert result.score == "bic"
    assert result.test is None
    assert result.alpha is None
    assert result.sepsets is None
    assert result.n == 1500


class _LopsidedScore(BIC):
    """A BIC clone that disclaims score equivalence, to exercise the warning."""

    name: ClassVar[str] = "lopsided"
    score_equivalent: ClassVar[bool] = False


def test_boss_warns_when_the_score_is_not_equivalent() -> None:
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    y = 1.3 * x + rng.normal(size=n)
    data = tabular({"x": x, "y": y})
    with pytest.warns(UserWarning, match="equival"):
        result = BOSS(score=_LopsidedScore(data)).fit()
    assert result.score == "lopsided"  # the search still ran
    assert result.graph.get_edge("x", "y") is not None


def test_boss_rejects_bad_arguments() -> None:
    with pytest.raises(TypeError, match="Score or a string"):
        BOSS(score=3.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="restarts"):
        BOSS(restarts=-1)
