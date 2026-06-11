# SPDX-License-Identifier: Apache-2.0
"""Tests for Meek's rules, the Dor-Tarsi extension, the real is_cpdag, and PC-stable.

PC is checked for exactness against the d-separation oracle on the canonical
graphs, so every assertion is deterministic. One sampled linear-Gaussian run
goes through a self-contained partial-correlation stub defined here, keeping
this file independent of the concrete CI-test modules.
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Any, ClassVar

import numpy as np
import pytest

from ergodic.data import TabularData, tabular
from ergodic.discovery import CIResult, CITest, DSeparationOracle
from ergodic.discovery.algorithms._pc import PC, _classify_triple, _vote_triples
from ergodic.graph import GraphKind, InvalidGraphError, MixedGraph, cpdag, dag, mixed
from ergodic.graph._meek import (
    consistent_extension,
    cpdag_of_dag,
    extendable,
    meek_closure,
)
from ergodic.knowledge import DomainKnowledge


def _directed(graph: MixedGraph, parent: str, child: str) -> bool:
    edge = graph.get_edge(parent, child)
    return edge is not None and edge.directed_pair() == (parent, child)


def _undirected(graph: MixedGraph, a: str, b: str) -> bool:
    edge = graph.get_edge(a, b)
    return edge is not None and edge.is_undirected


# -- Meek rules, one minimal PDAG each -----------------------------------------


def test_meek_r1_orients_away_from_a_nonadjacent_parent() -> None:
    closed = meek_closure(mixed(["A -> B", "B -- C"]))
    assert _directed(closed, "B", "C")


def test_meek_r2_orients_to_avoid_a_cycle() -> None:
    closed = meek_closure(mixed(["A -> B", "B -> C", "A -- C"]))
    assert _directed(closed, "A", "C")


def test_meek_r3_orients_the_kite() -> None:
    closed = meek_closure(mixed(["A -- B", "A -- C", "A -- D", "C -> B", "D -> B"]))
    assert _directed(closed, "A", "B")
    assert _undirected(closed, "A", "C")
    assert _undirected(closed, "A", "D")


def test_meek_r4_orients_with_a_directed_chain() -> None:
    # b -- a, a -- c, a -- d, b -> c, c -> d, b and d nonadjacent  =>  a -> d
    closed = meek_closure(mixed(["A -- B", "A -- C", "A -- D", "B -> C", "C -> D"]))
    assert _directed(closed, "A", "D")
    assert _undirected(closed, "A", "B")
    assert _undirected(closed, "A", "C")


def test_meek_forbidden_callback_blocks_an_orientation() -> None:
    pdag = mixed(["A -> B", "B -- C"])
    closed = meek_closure(pdag, forbidden=lambda u, v: (u, v) == ("B", "C"))
    assert _undirected(closed, "B", "C")


def test_meek_closure_rejects_non_pdag_marks() -> None:
    with pytest.raises(ValueError, match="needs a PDAG"):
        meek_closure(mixed(["A <-> B"]))


def test_meek_ambiguous_triple_gates_r1() -> None:
    # with (A, B, C) uncertified, B <- C could still be the collider
    closed = meek_closure(mixed(["A -> B", "B -- C"]), ambiguous=[("A", "B", "C")])
    assert _undirected(closed, "B", "C")


def test_meek_ambiguous_triple_ends_read_unordered() -> None:
    closed = meek_closure(mixed(["A -> B", "B -- C"]), ambiguous=[("C", "B", "A")])
    assert _undirected(closed, "B", "C")


def test_meek_ambiguous_triple_gates_r3() -> None:
    # the kite orients A -> B only because (C, A, D) cannot collide at A
    kite = mixed(["A -- B", "A -- C", "A -- D", "C -> B", "D -> B"])
    closed = meek_closure(kite, ambiguous=[("C", "A", "D")])
    assert _undirected(closed, "A", "B")


def test_meek_r2_ignores_ambiguous_triples() -> None:
    # R2 breaks a cycle, no non-collider certificate involved
    closed = meek_closure(mixed(["A -> B", "B -> C", "A -- C"]), ambiguous=[("A", "B", "C")])
    assert _directed(closed, "A", "C")


# -- cpdag_of_dag on known examples --------------------------------------------


def test_cpdag_of_chain_is_all_undirected() -> None:
    assert cpdag_of_dag(dag(["X -> M", "M -> Y"])) == cpdag(["X -- M", "M -- Y"])


def test_cpdag_of_collider_keeps_both_arrows() -> None:
    assert cpdag_of_dag(dag(["X -> Z", "Y -> Z"])) == cpdag(["X -> Z", "Y -> Z"])


def test_cpdag_of_y_structure_compels_the_tail_edge() -> None:
    # the v-structure stays and Meek R1 compels Z -> W, a non-collider arrow
    got = cpdag_of_dag(dag(["X -> Z", "Y -> Z", "Z -> W"]))
    assert got == cpdag(["X -> Z", "Y -> Z", "Z -> W"])


def test_cpdag_of_diamond_mixes_directed_and_undirected() -> None:
    got = cpdag_of_dag(dag(["A -> B", "A -> C", "B -> D", "C -> D"]))
    assert got == cpdag(["A -- B", "A -- C", "B -> D", "C -> D"])
    assert got.kind is GraphKind.CPDAG


def test_cpdag_of_dag_rejects_non_dags() -> None:
    with pytest.raises(ValueError, match="needs a DAG"):
        cpdag_of_dag(mixed(["A -- B"]))


# -- Dor-Tarsi extension --------------------------------------------------------


def test_consistent_extension_of_a_cpdag_lands_in_its_class() -> None:
    pattern = cpdag(["A -- B", "A -- C", "B -> D", "C -> D"])
    extension = consistent_extension(pattern)
    assert extension is not None
    assert extension.is_dag()
    skeleton = {frozenset((e.u, e.v)) for e in pattern.edges}
    assert {frozenset((e.u, e.v)) for e in extension.edges} == skeleton
    assert cpdag_of_dag(extension) == pattern


def test_consistent_extension_of_a_dag_is_itself() -> None:
    truth = dag(["X -> M", "M -> Y"])
    assert consistent_extension(truth) == truth
    assert extendable(truth)


def test_undirected_square_is_not_extendable() -> None:
    square = mixed(["A -- B", "B -- C", "C -- D", "D -- A"])
    assert not extendable(square)
    assert consistent_extension(square) is None


# -- the tightened is_cpdag -----------------------------------------------------


def test_is_cpdag_accepts_real_cpdags() -> None:
    assert cpdag([]).is_cpdag()
    assert cpdag(["A -- B"]).is_cpdag()
    assert cpdag(["X -> Z", "Y -> Z"]).is_cpdag()
    assert mixed(["A -- B", "A -- C", "B -> D", "C -> D"]).is_cpdag()
    assert mixed(["A -- B", "B -- C", "A -- C"]).is_cpdag()  # complete, all reversible


def test_is_cpdag_rejects_non_cpdags() -> None:
    assert not mixed(["A -> B", "B -- C"]).is_cpdag()  # not closed under R1
    assert not mixed(["A -> B"]).is_cpdag()  # the arrow is not compelled
    assert not mixed(["A -> B", "A -> C", "B -> C"]).is_cpdag()  # shielded triangle
    assert not mixed(["A -- B", "B -- C", "C -- D", "D -- A"]).is_cpdag()  # no extension
    assert not mixed(["A <-> B"]).is_cpdag()  # wrong marks


def test_cpdag_builder_rejects_loose_constructions() -> None:
    with pytest.raises(InvalidGraphError):
        cpdag(["A -> B", "B -- C"])
    with pytest.raises(InvalidGraphError):
        cpdag(["A -> B"])


# -- PC exactness against the oracle --------------------------------------------


@pytest.mark.parametrize(
    "edges",
    [
        pytest.param(["X -> M", "M -> Y"], id="chain"),
        pytest.param(["Z -> X", "Z -> Y"], id="fork"),
        pytest.param(["X -> Z", "Y -> Z"], id="collider"),
        pytest.param(["T1 -> T3", "T2 -> T3", "T3 -> T4", "T4 -> T5"], id="y-structure"),
        pytest.param(["A -> B", "A -> C", "B -> D", "C -> D"], id="diamond"),
    ],
)
def test_pc_recovers_the_cpdag_exactly(edges: list[str]) -> None:
    truth = dag(edges)
    result = PC(DSeparationOracle(truth)).fit()
    assert result.graph == cpdag_of_dag(truth)
    assert result.graph.kind is GraphKind.CPDAG
    assert result.diagnostics["cpdag_valid"] == 1.0
    assert result.diagnostics["n_ambiguous_triples"] == 0.0  # always present
    assert result.method == "pc"
    assert result.test == "d_separation"
    assert result.score is None


class _ShuffledOracle(DSeparationOracle):
    """A d-separation oracle that reports its variables in a chosen order."""

    def __init__(self, graph: MixedGraph, order: tuple[str, ...]) -> None:
        super().__init__(graph)
        self._order = order

    @property
    def variables(self) -> tuple[str, ...]:
        return self._order


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pc_is_order_independent(seed: int) -> None:
    truth = dag(["A -> B", "A -> C", "B -> D", "C -> D", "D -> E"])
    expected = cpdag_of_dag(truth)
    order = sorted(truth.nodes)
    random.Random(seed).shuffle(order)
    result = PC(_ShuffledOracle(truth, tuple(order))).fit()
    assert result.graph == expected


# -- sepsets and the conditioning cap --------------------------------------------


def test_sepsets_are_recorded_for_removed_pairs() -> None:
    chain = PC(DSeparationOracle(dag(["X -> M", "M -> Y"]))).fit()
    assert chain.sepsets is not None
    assert chain.sepsets[frozenset({"X", "Y"})] == frozenset({"M"})

    collider = PC(DSeparationOracle(dag(["X -> Z", "Y -> Z"]))).fit()
    assert collider.sepsets is not None
    assert collider.sepsets[frozenset({"X", "Y"})] == frozenset()


def test_max_cond_size_is_honored() -> None:
    oracle = DSeparationOracle(dag(["X -> M", "M -> Y"]))
    capped = PC(oracle, max_cond_size=0).fit()
    # X and Y separate only given M, so the cap keeps the complete skeleton
    assert capped.diagnostics["n_edges"] == 3.0
    assert capped.diagnostics["max_cond_size"] == 0.0
    assert capped.sepsets is not None and not capped.sepsets

    full = PC(DSeparationOracle(dag(["X -> M", "M -> Y"])), max_cond_size=1).fit()
    assert full.diagnostics["n_edges"] == 2.0


# -- knowledge in the four modes -------------------------------------------------


def test_knowledge_prune_skips_the_pair_and_the_oracle() -> None:
    truth = dag(["A -> B"])
    forbidden = DomainKnowledge().forbid_adjacency("A", "B")
    result = PC(DSeparationOracle(truth)).fit(knowledge=forbidden)
    assert not result.graph.edges
    assert result.diagnostics["n_ci_tests"] == 0.0  # the pruned pair is never queried


def test_knowledge_protect_keeps_a_required_edge_untested() -> None:
    # X and Y are truly independent; without protection the edge would fall at level 0
    truth = dag([], nodes=["X", "Y"])
    required = DomainKnowledge().require("X", "Y")
    result = PC(DSeparationOracle(truth)).fit(knowledge=required)
    assert result.diagnostics["n_ci_tests"] == 0.0  # the protected pair is never tested
    assert _directed(result.graph, "X", "Y")
    assert required.is_consistent(result.graph)


def test_knowledge_required_edge_is_oriented_and_propagated() -> None:
    truth = dag(["X -> M", "M -> Y"])
    required = DomainKnowledge().require("X", "M")
    result = PC(DSeparationOracle(truth)).fit(knowledge=required)
    assert _directed(result.graph, "X", "M")
    assert _directed(result.graph, "M", "Y")  # Meek R1 propagates the forced arrow
    assert required.is_consistent(result.graph)
    # the knowledge-oriented chain is no CPDAG, so the face falls back to mixed
    assert result.diagnostics["cpdag_valid"] == 0.0
    assert result.graph.kind is GraphKind.MIXED


def test_knowledge_tiers_orient_an_otherwise_undirected_edge() -> None:
    truth = dag(["A -> B", "B -> C"])
    tiers = DomainKnowledge().with_tiers([["A"], ["B", "C"]])
    plain = PC(DSeparationOracle(truth)).fit()
    assert _undirected(plain.graph, "A", "B")
    result = PC(DSeparationOracle(truth)).fit(knowledge=tiers)
    assert _directed(result.graph, "A", "B")  # the tier order forces it
    assert _directed(result.graph, "B", "C")  # and Meek carries it down the chain
    assert tiers.is_consistent(result.graph)


def test_knowledge_validate_raises_when_unsatisfiable() -> None:
    # PC never produces a bidirected edge, so a confounding constraint must fail
    confounded = DomainKnowledge().confound("A", "B")
    with pytest.raises(ValueError, match="violates the domain knowledge"):
        PC(DSeparationOracle(dag(["A -> B"]))).fit(knowledge=confounded)


# -- test and data resolution -----------------------------------------------------


def test_pc_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PC(alpha=1.5)
    with pytest.raises(ValueError, match="max_cond_size"):
        PC(max_cond_size=-1)


def test_unbound_test_without_data_raises() -> None:
    with pytest.raises(ValueError, match="needs data"):
        PC().fit()


def test_bound_test_with_a_different_dataset_raises() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=8), "B": rng.normal(size=8)})
    oracle = DSeparationOracle(dag(["A -> B"]))
    with pytest.raises(ValueError, match="already bound"):
        PC(oracle).fit(data=data)


def test_unknown_test_name_raises() -> None:
    rng = np.random.default_rng(0)
    data = tabular({"A": rng.normal(size=8), "B": rng.normal(size=8)})
    with pytest.raises(ValueError, match="unknown test"):
        PC("not_a_test").fit(data=data)
    with pytest.raises(TypeError, match="CITest or a string"):
        PC(3.14).fit(data=data)  # type: ignore[arg-type]


# -- one sampled run through a self-contained numeric stub -------------------------


def _normal_sf(t: float) -> float:
    return 0.5 * (1.0 - math.erf(t / math.sqrt(2.0)))


class _StubLinear(CITest):
    """A self-contained partial-correlation Fisher-z test for the sampled run."""

    name: ClassVar[str] = "stub_linear"
    needs: ClassVar[str] = "numeric"

    def statistic(self, *args: Any) -> tuple[float, float, float | None]:
        x, y, z = args
        n = x.shape[0]
        design = np.hstack([np.ones((n, 1)), z])
        coeff_x, *_ = np.linalg.lstsq(design, x[:, 0], rcond=None)
        coeff_y, *_ = np.linalg.lstsq(design, y[:, 0], rcond=None)
        res_x = x[:, 0] - design @ coeff_x
        res_y = y[:, 0] - design @ coeff_y
        corr = float(res_x @ res_y / math.sqrt((res_x @ res_x) * (res_y @ res_y)))
        corr = min(max(corr, -0.999999), 0.999999)
        stat = math.sqrt(n - z.shape[1] - 3) * math.atanh(corr)
        return stat, 2.0 * _normal_sf(abs(stat)), None


def test_pc_recovers_a_sampled_linear_gaussian_chain() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.normal(size=n)
    m = 1.2 * x + 0.8 * rng.normal(size=n)
    y = -0.9 * m + 0.8 * rng.normal(size=n)
    data = tabular({"X": x, "M": m, "Y": y})
    result = PC(_StubLinear(), alpha=0.05).fit(data=data)
    assert result.graph == cpdag_of_dag(dag(["X -> M", "M -> Y"]))
    assert result.n == n
    assert result.test == "stub_linear"
    assert result.diagnostics["cpdag_valid"] == 1.0


# -- conservative and majority collider rules --------------------------------------


def test_pc_rejects_unknown_colliders_rule() -> None:
    with pytest.raises(ValueError, match="'standard', 'conservative', or 'majority'"):
        PC(colliders="bogus")


def _random_dag(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        if rng.uniform() < 0.4:
            edges.append((names[i], names[j]))
    return dag(edges, nodes=names)


@pytest.mark.parametrize("colliders", ["conservative", "majority"])
def test_collider_variants_equal_standard_under_the_oracle(colliders: str) -> None:
    # Colombo and Maathuis (2014, theorem 4): with a perfect oracle on a
    # faithful world every separating set agrees on each middle node, so no
    # triple is ambiguous and both variants land on the CPDAG of the truth
    rng = np.random.default_rng(11)
    for _ in range(20):
        truth = _random_dag(rng, 7)
        result = PC(DSeparationOracle(truth), colliders=colliders).fit()
        assert result.graph == cpdag_of_dag(truth)
        assert result.graph.kind is GraphKind.CPDAG
        assert result.diagnostics["cpdag_valid"] == 1.0
        assert result.diagnostics["n_ambiguous_triples"] == 0.0


# the classification helper on synthetic separating-set lists


def test_classify_triple_with_no_separating_set_is_ambiguous() -> None:
    assert _classify_triple("conservative", "Z", []) == "ambiguous"
    assert _classify_triple("majority", "Z", []) == "ambiguous"


def test_classify_triple_majority_tie_is_ambiguous() -> None:
    split = [frozenset(), frozenset({"Z"})]
    assert _classify_triple("majority", "Z", split) == "ambiguous"
    assert _classify_triple("conservative", "Z", split) == "ambiguous"
    # one more vote breaks the tie either way
    assert _classify_triple("majority", "Z", [*split, frozenset({"W"})]) == "collider"
    assert _classify_triple("majority", "Z", [*split, frozenset({"W", "Z"})]) == "noncollider"


def test_classify_triple_conservative_needs_unanimity() -> None:
    assert _classify_triple("conservative", "Z", [frozenset(), frozenset({"W"})]) == "collider"
    assert _classify_triple("conservative", "Z", [frozenset({"Z"})]) == "noncollider"
    mostly = [frozenset({"Z"}), frozenset({"W", "Z"}), frozenset()]
    assert _classify_triple("conservative", "Z", mostly) == "ambiguous"
    assert _classify_triple("majority", "Z", mostly) == "noncollider"  # two of three


# the vote sweep through a canned oracle: X - Z - Y unshielded, plus a leaf W at X


class _CannedTest(CITest):
    """A test answering from a fixed table: p = 1 for listed independences, else 0."""

    name: ClassVar[str] = "canned"

    def __init__(
        self, variables: tuple[str, ...], independent: set[tuple[str, str, frozenset[str]]]
    ) -> None:
        super().__init__(None)
        self._names = frozenset(variables)
        self._independent = independent

    @property
    def is_bound(self) -> bool:
        return True

    def _require_bound(self) -> None:
        return

    def _dispatch(self, x: str, y: str, given: frozenset[str]) -> CIResult:
        p = 1.0 if (x, y, given) in self._independent else 0.0
        return CIResult(statistic=0.0, p_value=p, dof=None)


_CHAIN_NODES = ("W", "X", "Y", "Z")
_CHAIN_ADJ = {"W": {"X"}, "X": {"W", "Z"}, "Y": {"Z"}, "Z": {"X", "Y"}}


def test_vote_triples_majority_decision_updates_the_sepset() -> None:
    # candidate sets {}, {W}, {Z}, {W, Z}; all but {W} separate, so Z sits in
    # two of three separating sets: majority says non-collider and rewrites
    # the recorded sepset to witness it, conservative stays ambiguous
    independent = {
        ("X", "Y", frozenset()),
        ("X", "Y", frozenset({"Z"})),
        ("X", "Y", frozenset({"W", "Z"})),
    }
    oracle = _CannedTest(_CHAIN_NODES, independent)
    sepsets: dict[frozenset[str], frozenset[str]] = {frozenset({"X", "Y"}): frozenset()}
    colliders, ambiguous = _vote_triples(
        oracle,
        sorted(_CHAIN_NODES),
        _CHAIN_ADJ,
        sepsets,
        rule="majority",
        alpha=0.05,
        max_cond_size=None,
    )
    assert colliders == frozenset()
    assert ambiguous == frozenset()
    assert sepsets[frozenset({"X", "Y"})] == frozenset({"Z"})
    # the two neighbourhoods share {} and {Z}: deduplication queries each set
    # once, and the sepset-less (W, Z) pair is skipped without a query
    assert oracle.diagnostics()["n_ci_tests"] == 4.0
    assert oracle.diagnostics()["cache_hits"] == 0.0

    sepsets = {frozenset({"X", "Y"}): frozenset()}
    colliders, ambiguous = _vote_triples(
        oracle,
        sorted(_CHAIN_NODES),
        _CHAIN_ADJ,
        sepsets,
        rule="conservative",
        alpha=0.05,
        max_cond_size=None,
    )
    assert colliders == frozenset()
    assert ambiguous == frozenset({("X", "Z", "Y")})
    assert sepsets[frozenset({"X", "Y"})] == frozenset()  # ambiguity leaves it alone


def test_vote_triples_collider_vote_rewrites_a_sepset_containing_z() -> None:
    # only {W} separates, so both rules vote collider; the stale skeleton
    # sepset {Z} would contradict the decision and is replaced
    oracle = _CannedTest(_CHAIN_NODES, {("X", "Y", frozenset({"W"}))})
    sepsets = {frozenset({"X", "Y"}): frozenset({"Z"})}
    colliders, ambiguous = _vote_triples(
        oracle,
        sorted(_CHAIN_NODES),
        _CHAIN_ADJ,
        sepsets,
        rule="conservative",
        alpha=0.05,
        max_cond_size=None,
    )
    assert colliders == frozenset({("X", "Z", "Y")})
    assert ambiguous == frozenset()
    assert sepsets[frozenset({"X", "Y"})] == frozenset({"W"})


def test_vote_triples_honors_max_cond_size() -> None:
    # capped at size one the pool loses {W, Z}, the vote drops to one of two
    # and the majority decision degrades to a tie, hence ambiguous
    independent = {
        ("X", "Y", frozenset()),
        ("X", "Y", frozenset({"Z"})),
        ("X", "Y", frozenset({"W", "Z"})),
    }
    oracle = _CannedTest(_CHAIN_NODES, independent)
    sepsets: dict[frozenset[str], frozenset[str]] = {frozenset({"X", "Y"}): frozenset()}
    colliders, ambiguous = _vote_triples(
        oracle,
        sorted(_CHAIN_NODES),
        _CHAIN_ADJ,
        sepsets,
        rule="majority",
        alpha=0.05,
        max_cond_size=1,
    )
    assert colliders == frozenset()
    assert ambiguous == frozenset({("X", "Z", "Y")})
    assert sepsets[frozenset({"X", "Y"})] == frozenset()


# sampled worlds where the vote actually splits


def _false_collider_world(seed: int = 16, n: int = 1000) -> TabularData:
    """A -> B <- C with a weak child D, sized so the A-D and C-D margins vanish.

    At this n the marginal A-D and C-D correlations (about 0.04) sit under
    the alpha threshold, so the skeleton drops both pairs with the empty
    sepset and standard PC orients the false v-structure D -> B. The re-test
    finds {} and {B} both separating, an even split on B, so both variants
    mark (A, B, D) and (C, B, D) ambiguous instead.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    c = rng.normal(size=n)
    b = 0.35 * a + 0.35 * c + 0.95 * rng.normal(size=n)
    d = 0.12 * b + rng.normal(size=n)
    return tabular({"A": a, "B": b, "C": c, "D": d})


def test_standard_pc_orients_the_false_v_structure() -> None:
    result = PC(_StubLinear(), alpha=0.05).fit(data=_false_collider_world())
    assert _directed(result.graph, "D", "B")  # the finite-sample mistake
    assert result.diagnostics["cpdag_valid"] == 1.0
    assert result.diagnostics["n_ambiguous_triples"] == 0.0


@pytest.mark.parametrize("colliders", ["conservative", "majority"])
def test_ambiguous_triples_stay_unoriented_with_an_honest_face(colliders: str) -> None:
    result = PC(_StubLinear(), alpha=0.05, colliders=colliders).fit(data=_false_collider_world())
    # the real collider still orients; the split votes refuse the false one
    assert _directed(result.graph, "A", "B")
    assert _directed(result.graph, "C", "B")
    assert _undirected(result.graph, "B", "D")
    assert result.diagnostics["n_ambiguous_triples"] == 2.0
    # with R1 gated the PDAG is not closed under Meek, so the face drops
    assert result.graph.kind is GraphKind.MIXED
    assert result.diagnostics["cpdag_valid"] == 0.0


@pytest.mark.parametrize("colliders", ["conservative", "majority"])
def test_an_ambiguous_lone_triple_keeps_the_valid_cpdag_face(colliders: str) -> None:
    # X -> Z <- Y with weak weights: at this n the collider leaks so little
    # that {Z} also separates X and Y, a split vote on the only triple;
    # nothing orients and the undirected chain is itself a valid CPDAG
    rng = np.random.default_rng(22)
    n = 300
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = 0.3 * x + 0.3 * y + rng.normal(size=n)
    data = tabular({"X": x, "Y": y, "Z": z})
    standard = PC(_StubLinear(), alpha=0.05).fit(data=data)
    assert _directed(standard.graph, "X", "Z") and _directed(standard.graph, "Y", "Z")
    result = PC(_StubLinear(), alpha=0.05, colliders=colliders).fit(data=data)
    assert _undirected(result.graph, "X", "Z")
    assert _undirected(result.graph, "Y", "Z")
    assert result.diagnostics["n_ambiguous_triples"] == 1.0
    assert result.graph.kind is GraphKind.CPDAG
    assert result.diagnostics["cpdag_valid"] == 1.0
