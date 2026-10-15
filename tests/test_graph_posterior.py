# SPDX-License-Identifier: Apache-2.0
"""The posterior over DAGs from edge probabilities (API-152, DEC-082 decision 1).

The exact subset recursion is held to brute-force enumeration (normalizer, edge
marginals, MAP, draws) with and without hard, soft and required knowledge; the
structure MCMC is held to the exact computation; and the facade's choices,
determinism, immutability and errors are pinned.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from tests.test_decode import all_dags, random_independent, random_pair_state, score

from ergodic import DomainKnowledge, EdgeProbabilities, graph_posterior
from ergodic.discovery import GraphSamples, PosteriorConvergenceWarning
from ergodic.discovery._evaluate import p_acyclic_exact
from ergodic.discovery._exact import PairWeights, SubsetDP, map_graph
from ergodic.discovery._target import GraphTarget
from ergodic.graph import dag

NAMES = ("A", "B", "C", "D", "E", "F", "G")


def _probs(size: int, seed: int, kind: str = "pair_state") -> EdgeProbabilities:
    rng = np.random.default_rng(seed)
    values = random_pair_state(size, rng) if kind == "pair_state" else random_independent(size, rng)
    return EdgeProbabilities(NAMES[:size], values, kind=kind)  # type: ignore[arg-type]


def _graph(names: tuple[str, ...], matrix: np.ndarray) -> object:
    sources, targets = np.nonzero(matrix)
    return dag([(names[s], names[t]) for s, t in zip(sources, targets, strict=True)], nodes=names)


def _brute(
    probs: EdgeProbabilities, knowledge: DomainKnowledge | None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Every DAG and its log target, from the public pieces: pair score plus log_prior."""
    pair = probs.to_pair_states().values
    graphs = all_dags(len(probs.variables))
    logs = []
    for graph in graphs:
        value = score(graph, pair)
        if knowledge is not None:
            value += knowledge.log_prior(_graph(probs.variables, graph))  # type: ignore[arg-type]
        logs.append(value)
    return graphs, np.asarray(logs)


def _marginals(graphs: list[np.ndarray], logs: np.ndarray) -> tuple[float, np.ndarray]:
    finite = np.isfinite(logs)
    top = logs[finite].max()
    weights = np.where(finite, np.exp(logs - top), 0.0)
    total = weights.sum()
    edges = sum(w * g for w, g in zip(weights, graphs, strict=True)) / total
    return float(top + math.log(total)), np.asarray(edges)


KNOWLEDGE = {
    "none": None,
    "hard": DomainKnowledge().forbid("A", "B").root("C"),
    "required": DomainKnowledge().require("B", "D").forbid_adjacency("A", "C"),
    "soft": DomainKnowledge().require("D", "A", confidence=0.8).sink("B", confidence=0.7),
    "mixed": DomainKnowledge()
    .require("A", "D")
    .with_tiers([["A", "B"], ["C", "D"]], confidence=0.75)
    .forbid("B", "C", confidence=0.9),
    "tiers": DomainKnowledge().with_tiers([["B"], ["A", "C"], ["D"]]),
}


@pytest.mark.parametrize("name", list(KNOWLEDGE))
@pytest.mark.parametrize("kind", ["pair_state", "independent"])
def test_the_subset_recursion_is_exact_against_enumeration(name: str, kind: str) -> None:
    probs = _probs(4, seed=len(name) + len(kind), kind=kind)
    knowledge = KNOWLEDGE[name]
    graphs, logs = _brute(probs, knowledge)
    log_z, edges = _marginals(graphs, logs)
    target = GraphTarget.from_probabilities(probs, knowledge)
    assert target.pair_factorized
    dp = SubsetDP.build(PairWeights.from_logs(target.log_none, target.log_edge))
    assert dp.log_z + target.constant == pytest.approx(log_z, abs=1e-10)
    np.testing.assert_allclose(dp.edge_marginals(), edges, atol=1e-10)
    best = map_graph(dp.weights)
    assert target.log_target(best) == pytest.approx(logs.max(), abs=1e-10)
    for graph, value in zip(graphs, logs, strict=True):
        assert target.log_target(graph) == pytest.approx(value, abs=1e-9) or (
            value == -math.inf and target.log_target(graph) == -math.inf
        )


def test_five_variables_exact_against_enumeration_through_the_facade() -> None:
    probs = _probs(5, seed=3)
    knowledge = DomainKnowledge().require("A", "B").forbid("C", "D").order("E", "A", confidence=0.7)
    graphs, logs = _brute(probs, knowledge)
    log_z, edges = _marginals(graphs, logs)
    samples = graph_posterior(probs, knowledge=knowledge, samples=200, seed=1)
    assert samples.diagnostics["engine"] == "subset_recursion"
    assert samples.method == "exact"
    assert samples.log_normalizer == pytest.approx(log_z, abs=1e-10)
    np.testing.assert_allclose(samples.edge_probabilities().values, edges, atol=1e-10)
    assert samples.best_log_target == pytest.approx(logs.max(), abs=1e-10)
    assert np.all(samples.edge_errors() == 0.0)


def test_exact_draws_follow_the_posterior() -> None:
    probs = _probs(4, seed=9)
    knowledge = DomainKnowledge().require("A", "C").forbid("B", "D")
    target = GraphTarget.from_probabilities(probs, knowledge)
    dp = SubsetDP.build(PairWeights.from_logs(target.log_none, target.log_edge))
    graphs, logs = _brute(probs, knowledge)
    probabilities = np.exp(logs - (dp.log_z + target.constant))
    sampler = dp.sampler()
    rng = np.random.default_rng(0)
    index = {graph.tobytes(): k for k, graph in enumerate(graphs)}
    draws = 20_000
    counts = np.zeros(len(graphs))
    for _ in range(draws):
        counts[index[sampler.draw(rng).tobytes()]] += 1
    support = probabilities > 0
    assert counts[~support].sum() == 0
    expected = draws * probabilities[support]
    chi2 = float(((counts[support] - expected) ** 2 / expected).sum())
    dof = int(support.sum()) - 1
    assert chi2 < dof + 5 * math.sqrt(2 * dof)


def test_the_target_is_the_pair_score_plus_the_log_prior() -> None:
    probs = _probs(5, seed=4)
    mixed = KNOWLEDGE["mixed"]
    assert mixed is not None
    knowledge = mixed.sink("E", confidence=0.6)
    samples = graph_posterior(probs, knowledge=knowledge, samples=50, seed=0)
    pair = probs.to_pair_states().values
    for index in range(min(20, samples.adjacency.shape[0])):
        graph = samples.graph(index)
        matrix = samples.adjacency[index]
        expected = score(matrix, pair) + knowledge.log_prior(graph)
        assert samples.log_target(graph) == pytest.approx(expected, abs=1e-9)
        assert samples.log_targets[index] == pytest.approx(expected, abs=1e-9)


def test_counting_dags_stays_exact_at_fifteen_variables() -> None:
    counts = [1]
    for n in range(1, 16):
        counts.append(
            sum(
                (-1) ** (k + 1) * math.comb(n, k) * 2 ** (k * (n - k)) * counts[n - k]
                for k in range(1, n + 1)
            )
        )
    for size in (6, 11, 15):
        zeros = np.zeros((size, size))
        dp = SubsetDP.build(PairWeights.from_logs(zeros, zeros))
        assert dp.log_z == pytest.approx(math.log(counts[size]), rel=1e-12)


def test_marginals_match_leave_one_edge_out_at_twelve_variables() -> None:
    rng = np.random.default_rng(2)
    probs = EdgeProbabilities(
        tuple(f"x{i}" for i in range(12)), random_pair_state(12, rng), kind="pair_state"
    )
    target = GraphTarget.from_probabilities(probs, None)
    dp = SubsetDP.build(PairWeights.from_logs(target.log_none, target.log_edge))
    edges = dp.edge_marginals()
    for i, j in ((0, 1), (7, 3), (11, 5)):
        banned = target.log_edge.copy()
        banned[i, j] = -np.inf
        without = SubsetDP.build(PairWeights.from_logs(target.log_none, banned)).log_z
        assert edges[i, j] == pytest.approx(1.0 - math.exp(without - dp.log_z), abs=1e-10)


# -- MCMC ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["none", "soft", "required"])
def test_mcmc_agrees_with_the_exact_computation(name: str) -> None:
    probs = _probs(5, seed=7)
    knowledge = KNOWLEDGE[name]
    exact = graph_posterior(probs, knowledge=knowledge, samples=100, seed=0)
    mcmc = graph_posterior(probs, knowledge=knowledge, method="mcmc", samples=4000, seed=0)
    assert mcmc.method == "mcmc"
    assert mcmc.diagnostics["edge_estimator"] == "rao_blackwell"
    assert mcmc.diagnostics["converged"]
    difference = np.abs(mcmc.edge_probabilities().values - exact.edge_probabilities().values)
    assert difference.max() < 0.02
    # The reported errors cover the differences.
    assert np.all(difference <= 5 * mcmc.edge_errors() + 1e-3)


def test_ancestral_order_constraints_run_by_enumeration_and_mcmc() -> None:
    probs = _probs(5, seed=11)
    # E is in no tier, so the path B -> E -> A could break "A before B".
    knowledge = DomainKnowledge().order("A", "B").order("C", "D")
    exact = graph_posterior(probs, knowledge=knowledge, seed=0)
    assert exact.diagnostics["engine"] == "enumeration"
    assert not exact.diagnostics["target"]["pair_factorized"]
    assert ("A", "B") in exact.diagnostics["target"]["ancestral_constraints"]
    for index in range(exact.adjacency.shape[0]):
        assert knowledge.is_consistent(exact.graph(index))
    mcmc = graph_posterior(probs, knowledge=knowledge, method="mcmc", samples=4000, seed=0)
    assert mcmc.diagnostics["edge_estimator"] == "frequency"
    for index in range(mcmc.adjacency.shape[0]):
        assert knowledge.is_consistent(mcmc.graph(index))
    difference = np.abs(mcmc.edge_probabilities().values - exact.edge_probabilities().values)
    assert difference.max() < 0.04
    with pytest.raises(ValueError, match="ancestral"):
        graph_posterior(_probs(6, seed=11), knowledge=knowledge, method="exact")


def test_tiers_covering_every_variable_stay_exact() -> None:
    probs = _probs(5, seed=12)
    knowledge = DomainKnowledge().with_tiers([["A", "B"], ["C"], ["D", "E"]])
    samples = graph_posterior(probs, knowledge=knowledge, samples=100, seed=0)
    assert samples.diagnostics["target"]["pair_factorized"]
    assert samples.diagnostics["engine"] == "subset_recursion"


@pytest.mark.filterwarnings("ignore::ergodic.discovery.PosteriorConvergenceWarning")
def test_draws_are_deterministic_given_the_seed_and_leave_inputs_alone() -> None:
    probs = _probs(5, seed=5, kind="independent")
    before = probs.values.copy()
    knowledge = KNOWLEDGE["soft"]
    state = np.random.get_state()[1].copy()  # type: ignore[index]
    first = graph_posterior(probs, knowledge=knowledge, method="mcmc", samples=400, seed=3)
    second = graph_posterior(probs, knowledge=knowledge, method="mcmc", samples=400, seed=3)
    np.testing.assert_array_equal(first.adjacency, second.adjacency)
    np.testing.assert_array_equal(first.draws, second.draws)
    third = graph_posterior(probs, knowledge=knowledge, samples=300, seed=3)
    fourth = graph_posterior(probs, knowledge=knowledge, samples=300, seed=3)
    np.testing.assert_array_equal(third.adjacency, fourth.adjacency)
    other = graph_posterior(probs, knowledge=knowledge, samples=300, seed=4)
    assert not np.array_equal(third.draws, other.draws)
    np.testing.assert_array_equal(probs.values, before)
    np.testing.assert_array_equal(np.random.get_state()[1], state)  # type: ignore[index]
    assert not first.adjacency.flags.writeable
    assert not first.weights.flags.writeable


def test_the_result_reads_as_pair_states_with_provenance() -> None:
    samples = graph_posterior(_probs(4, seed=1), seed=0)
    assert isinstance(samples, GraphSamples)
    assert samples.exhaustive and samples.diagnostics["engine"] == "enumeration"
    assert samples.weights.sum() == pytest.approx(1.0)
    assert samples.adjacency.shape[0] == 543
    edges = samples.edge_probabilities()
    assert edges.kind == "pair_state"
    assert edges.provenance["source"] == "graph_posterior"
    assert "posterior probability under the method" in edges.provenance["note"]
    graph, weight = samples.top(1)[0]
    assert graph == samples.best_graph
    assert weight == pytest.approx(float(samples.exact_probabilities.max()))  # type: ignore[union-attr]
    assert "543 distinct" in repr(samples)
    assert samples.probability(lambda g: True).probability == pytest.approx(1.0)


def test_feature_probabilities_carry_their_monte_carlo_error() -> None:
    probs = _probs(6, seed=8)
    exact = graph_posterior(probs, samples=2000, seed=0)
    feature = exact.probability(lambda g: "A" in g.ancestors("F"))
    assert not feature.exact and 0.0 < feature.mc_error < 0.02
    mcmc = graph_posterior(probs, method="mcmc", samples=2000, seed=0)
    estimate = mcmc.probability(lambda g: "A" in g.ancestors("F"))
    assert abs(estimate.probability - feature.probability) < 5 * (
        estimate.mc_error + feature.mc_error
    )


def test_confounders_are_ignored_and_listed() -> None:
    knowledge = DomainKnowledge().confound("A", "B")
    samples = graph_posterior(_probs(4, seed=2), knowledge=knowledge, seed=0)
    assert samples.diagnostics["ignored_confounders"] == (("A", "B"),)


def test_tempering_swaps_and_can_be_turned_off() -> None:
    probs = _probs(6, seed=3)
    tempered = graph_posterior(probs, method="mcmc", samples=2000, seed=0)
    ladder = tempered.diagnostics["temperatures"]
    assert ladder[0] == 1.0 and len(ladder) == 4
    assert all(0.05 < rate < 0.95 for rate in tempered.diagnostics["swap_acceptance"])
    plain = graph_posterior(probs, method="mcmc", samples=2000, temperatures=(1.0,), seed=0)
    assert plain.diagnostics["swap_acceptance"] == (0.0,) * 4
    exact = graph_posterior(probs, samples=100, seed=0).edge_probabilities().values
    for samples in (tempered, plain):
        assert np.abs(samples.edge_probabilities().values - exact).max() < 0.03
    for bad in ((0.5, 0.2), (1.0, 1.0), (1.0, 0.0), (1.0, 2.0), ()):
        with pytest.raises(ValueError, match="temperatures"):
            graph_posterior(probs, method="mcmc", temperatures=bad)


def test_a_short_run_warns_that_it_may_not_have_converged() -> None:
    with pytest.warns(PosteriorConvergenceWarning, match="may not have converged"):
        samples = graph_posterior(
            _probs(5, seed=3), method="mcmc", samples=40, warmup=0, thin=1, seed=0
        )
    assert samples.diagnostics["converged"] is False
    assert samples.diagnostics["convergence_problems"]


def test_p_acyclic_is_exact_against_brute_force() -> None:
    rng = np.random.default_rng(4)
    values = random_independent(4, rng)
    total = 0.0
    for graph in all_dags(4):
        mass = 1.0
        for i in range(4):
            for j in range(4):
                if i != j:
                    mass *= values[i, j] if graph[i, j] else 1.0 - values[i, j]
        total += mass
    assert p_acyclic_exact(values) == pytest.approx(total, abs=1e-12)
    forced = values.copy()
    forced[0, 1] = forced[1, 0] = 1.0
    assert p_acyclic_exact(forced) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"method": "magic"}, ValueError, "method"),
        ({"samples": 0}, ValueError, "samples"),
        ({"chains": 1.5}, ValueError, "chains"),
        ({"seed": -1}, ValueError, "seed"),
        ({"max_exact": 19}, ValueError, "max_exact"),
        ({"score": "bge"}, TypeError, "score"),
        ({"knowledge": DomainKnowledge().forbid("A", "Z")}, ValueError, "lacks"),
        (
            {"knowledge": DomainKnowledge().require("A", "B").require("B", "A")},
            ValueError,
            "cycle|contradict|conflict",
        ),
    ],
)
def test_bad_settings_raise_clearly(kwargs: dict[str, object], error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        graph_posterior(_probs(4, seed=0), **kwargs)  # type: ignore[arg-type]


def test_exact_refuses_where_it_does_not_apply() -> None:
    rng = np.random.default_rng(0)
    probs = EdgeProbabilities(
        tuple(f"x{i}" for i in range(7)), random_pair_state(7, rng), kind="pair_state"
    )
    with pytest.raises(ValueError, match="max_exact=6"):
        graph_posterior(probs, method="exact", max_exact=6)
    with pytest.raises(TypeError, match="EdgeProbabilities or a TabularData"):
        graph_posterior(np.zeros((3, 3)))  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert graph_posterior(probs, max_exact=6, samples=4000, seed=0).method == "mcmc"
