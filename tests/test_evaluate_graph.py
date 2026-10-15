# SPDX-License-Identifier: Apache-2.0
"""Evaluating a candidate graph against the posterior (API-152, DEC-082 decisions 2 and 3).

Every measure is held to brute-force enumeration on four variables: the log-ratio,
the credible level, each candidate type's claims, the equivalence-class
probability and the expected SHD. The exact-draw and MCMC estimates are held to
the exact values within their Monte Carlo errors.
"""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tests.test_decode import all_dags, random_independent, random_pair_state, score

from ergodic import DomainKnowledge, EdgeProbabilities, evaluate_graph, graph_posterior
from ergodic.discovery import GraphEvaluation
from ergodic.graph import admg, cpdag_of_dag, dag, mag_of_dag, mixed, pag_of_mag

if TYPE_CHECKING:
    from ergodic.graph import MixedGraph

NAMES = ("A", "B", "C", "D", "E")


def _probs(size: int = 4, seed: int = 0, kind: str = "pair_state") -> EdgeProbabilities:
    rng = np.random.default_rng(seed)
    values = random_pair_state(size, rng) if kind == "pair_state" else random_independent(size, rng)
    return EdgeProbabilities(NAMES[:size], values, kind=kind)  # type: ignore[arg-type]


def _dag(names: tuple[str, ...], matrix: np.ndarray) -> MixedGraph:
    sources, targets = np.nonzero(matrix)
    return dag([(names[s], names[t]) for s, t in zip(sources, targets, strict=True)], nodes=names)


def _posterior(probs: EdgeProbabilities) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Every DAG, its log target and its exact probability."""
    pair = probs.to_pair_states().values
    graphs = all_dags(len(probs.variables))
    logs = np.asarray([score(g, pair) for g in graphs])
    weights = np.exp(logs - logs.max())
    return graphs, logs, weights / weights.sum()


def _states(graph: np.ndarray) -> np.ndarray:
    low, high = np.triu_indices(graph.shape[0], k=1)
    states: np.ndarray = graph[low, high].astype(int) + 2 * graph[high, low].astype(int)
    return states


def test_dag_measures_match_enumeration() -> None:
    probs = _probs(seed=3)
    graphs, logs, weights = _posterior(probs)
    best = int(np.argmax(logs))
    for index in (best, int(np.argsort(logs)[len(logs) // 2]), int(np.argmin(logs))):
        candidate = _dag(probs.variables, graphs[index])
        result = evaluate_graph(candidate, probs)
        assert isinstance(result, GraphEvaluation)
        assert result.candidate_kind == "dag" and result.method == "exact"
        assert result.log_ratio_to_best == pytest.approx(logs[index] - logs[best], abs=1e-9)
        above = weights[logs > logs[index] + 1e-9].sum()
        tie = weights[np.abs(logs - logs[index]) <= 1e-9].sum()
        assert result.credible_level == pytest.approx(above + tie / 2, abs=1e-12)
        assert result.mass_above == pytest.approx(above, abs=1e-12)
        assert result.credible_level_error == 0.0
        states = np.stack([_states(g) for g in graphs])
        mine = _states(graphs[index])
        assert result.expected_shd == pytest.approx(
            float(weights @ (states != mine).sum(axis=1)), abs=1e-9
        )
        assert result.diagnostics["candidate_probability"] == pytest.approx(weights[index])
        assert len(result.claims) == 6
        for claim in result.claims:
            assert claim.exact and claim.mc_error == 0.0
    pairwise = (states[:, None, :] != states[None, :, :]).sum(axis=2)
    between = float(weights @ pairwise @ weights)
    assert result.expected_shd_between_draws == pytest.approx(between, abs=1e-9)


def test_the_map_has_nothing_above_it_and_a_terrible_graph_has_everything() -> None:
    probs = _probs(seed=5)
    samples = graph_posterior(probs, seed=0)
    best = evaluate_graph(samples.best_graph, samples)
    assert best.mass_above == 0.0
    assert best.log_ratio_to_best == 0.0
    top = float(samples.exact_probabilities.max())  # type: ignore[union-attr]
    assert best.credible_level == pytest.approx(top / 2)
    worst = samples.graph(int(np.argmin(samples.log_targets)))
    level = evaluate_graph(worst, samples).credible_level
    assert level is not None and level > 0.999


def test_dag_claims_are_the_pair_states() -> None:
    probs = _probs(seed=6)
    graphs, _, weights = _posterior(probs)
    candidate = dag(["A -> B", "C -> B", "C -> D"], nodes=probs.variables)
    result = evaluate_graph(candidate, probs)
    claims = {claim.claim: claim for claim in result.claims}
    index = {name: i for i, name in enumerate(probs.variables)}

    def edge(a: str, b: str) -> float:
        return float(sum(w for w, g in zip(weights, graphs, strict=True) if g[index[a], index[b]]))

    assert claims["A --> B"].probability == pytest.approx(edge("A", "B"))
    assert claims["C --> B"].kind == "edge"
    absent = claims["no edge between A and D"]
    assert absent.kind == "no_edge"
    assert absent.probability == pytest.approx(1 - edge("A", "D") - edge("D", "A"))
    frame = result.claims_frame()
    assert list(frame.columns) == ["claim", "kind", "probability", "mc_error", "exact", "note"]
    assert frame["probability"].is_monotonic_increasing
    assert "credible level" in repr(result)


def test_cpdag_claims_and_class_probability() -> None:
    probs = _probs(seed=7)
    graphs, _, weights = _posterior(probs)
    truth = dag(["A -> C", "B -> C", "C -> D"], nodes=probs.variables)
    candidate = cpdag_of_dag(truth)
    result = evaluate_graph(candidate, probs)
    assert result.candidate_kind == "cpdag"
    assert result.credible_level is None and result.log_ratio_to_best is None
    target = cpdag_of_dag(truth)
    same = sum(
        w
        for w, g in zip(weights, graphs, strict=True)
        if cpdag_of_dag(_dag(probs.variables, g)) == target
    )
    assert result.class_probability is not None
    assert result.class_probability.probability == pytest.approx(same, abs=1e-12)
    kinds = {claim.claim: claim.kind for claim in result.claims}
    assert kinds["A --> C"] == "edge" and kinds["C --> D"] == "edge"
    assert kinds["no edge between A and B"] == "no_edge"
    assert kinds["the graph's Markov equivalence class"] == "equivalence_class"
    undirected = cpdag_of_dag(dag(["A -> B", "B -> C"], nodes=probs.variables))
    claims = {c.claim: c for c in evaluate_graph(undirected, probs).claims}
    adjacency = claims["A and B adjacent"]
    assert adjacency.kind == "adjacent"
    index = {name: i for i, name in enumerate(probs.variables)}
    expected = sum(
        w
        for w, g in zip(weights, graphs, strict=True)
        if g[index["A"], index["B"]] or g[index["B"], index["A"]]
    )
    assert adjacency.probability == pytest.approx(expected)


def test_pag_marks_are_read_ancestrally_and_circles_claim_nothing() -> None:
    probs = _probs(seed=8)
    graphs, _, weights = _posterior(probs)
    candidate = pag_of_mag(mag_of_dag(dag(["A -> C", "B -> C", "C -> D"], nodes=probs.variables)))
    result = evaluate_graph(candidate, probs)
    assert result.candidate_kind == "pag"
    claims = {claim.claim: claim for claim in result.claims}
    index = {name: i for i, name in enumerate(probs.variables)}

    def ancestor(a: str, b: str) -> float:
        total = 0.0
        for w, g in zip(weights, graphs, strict=True):
            if a in _dag(probs.variables, g).ancestors(b):
                total += w
        return total

    # A o-> C: C is not an ancestor of A; the circle at A claims nothing.
    assert claims["C is not an ancestor of A"].probability == pytest.approx(1 - ancestor("C", "A"))
    assert "A is an ancestor of C" not in claims and "A is not an ancestor of C" not in claims
    # C --> D: a tail at C, an arrowhead at D.
    assert claims["C is an ancestor of D"].probability == pytest.approx(ancestor("C", "D"))
    assert claims["D is not an ancestor of C"].probability == pytest.approx(1 - ancestor("D", "C"))
    assert claims["A and C adjacent"].kind == "adjacent"
    assert claims["no edge between A and B"].kind == "no_edge"
    assert index  # the reading covers every pair: 3 edges and 3 absent pairs
    assert sum(c.kind in ("adjacent", "no_edge") for c in result.claims) == 6


def test_admg_directed_edges_are_claims_and_confounders_are_not_assessable() -> None:
    probs = _probs(seed=9)
    candidate = admg(["A -> B", "C <-> D", "B -> D"], nodes=probs.variables)
    result = evaluate_graph(candidate, probs)
    assert result.candidate_kind == "admg"
    kinds = {claim.claim: claim for claim in result.claims}
    assert kinds["A --> B"].probability is not None
    confounded = kinds["C <-> D"]
    assert confounded.probability is None and confounded.kind == "confounded"
    assert "cannot assess" in (confounded.note or "")
    assert result.not_assessable == ("C <-> D",)
    with_latent = admg(["L -> A", "L -> B", "A -> B"], nodes=(*probs.variables, "L"), latent=["L"])
    latent = evaluate_graph(with_latent, probs)
    assert latent.not_assessable == ("hidden common cause L of A, B",)


def test_a_graph_the_hard_knowledge_rules_out_has_everything_above_it() -> None:
    probs = _probs(seed=10)
    knowledge = DomainKnowledge().forbid("A", "B")
    candidate = dag(["A -> B"], nodes=probs.variables)
    result = evaluate_graph(candidate, probs, knowledge)
    assert result.log_ratio_to_best == -math.inf
    assert result.credible_level == 1.0
    assert "outside_support" in result.diagnostics


def test_exact_draws_and_mcmc_estimate_the_credible_level_within_their_errors() -> None:
    probs = _probs(size=5, seed=11)
    pair = probs.to_pair_states().values
    graphs = all_dags(5)
    logs = np.asarray([score(g, pair) for g in graphs])
    weights = np.exp(logs - logs.max())
    weights /= weights.sum()
    candidate_index = int(np.argsort(logs)[-40])
    candidate = _dag(probs.variables, graphs[candidate_index])
    exact = float(weights[logs > logs[candidate_index] + 1e-9].sum()) + 0.5 * float(
        weights[np.abs(logs - logs[candidate_index]) <= 1e-9].sum()
    )
    drawn = evaluate_graph(candidate, probs, samples=4000, seed=0)
    assert drawn.diagnostics["credible_level_method"] == "exact_draws"
    assert drawn.credible_level_error is not None and drawn.credible_level_error > 0
    assert abs(drawn.credible_level - exact) < 4 * drawn.credible_level_error  # type: ignore[operator]
    chained = evaluate_graph(candidate, probs, method="mcmc", samples=4000, seed=0)
    assert chained.diagnostics["credible_level_method"] == "mcmc"
    assert abs(chained.credible_level - exact) < 4 * chained.credible_level_error + 0.01  # type: ignore[operator]
    # Pair-state claims stay exact under exact draws; under MCMC they carry an error.
    assert all(claim.exact for claim in drawn.claims)
    assert not any(claim.exact for claim in chained.claims)


def test_exact_draws_use_the_candidates_exact_probability_in_its_level() -> None:
    probs = _probs(size=5, seed=15)
    samples = graph_posterior(probs, samples=800, seed=0)
    best = evaluate_graph(samples.best_graph, samples)
    own = best.diagnostics["candidate_probability"]
    assert best.mass_above == 0.0
    assert best.credible_level == pytest.approx(own / 2, abs=1e-12)
    assert best.diagnostics["credible_level_error_is_bound"]
    assert best.credible_level_error == pytest.approx(3 / 800)


def test_expected_shd_carries_its_monte_carlo_error() -> None:
    probs = _probs(size=5, seed=16)
    candidate = graph_posterior(probs, samples=100, seed=0).best_graph
    exact = evaluate_graph(candidate, probs, samples=100)
    assert exact.expected_shd_error == 0.0
    chained = evaluate_graph(candidate, probs, method="mcmc", samples=4000, seed=0)
    assert chained.expected_shd_error is not None and chained.expected_shd_error > 0
    assert abs(chained.expected_shd - exact.expected_shd) < 5 * chained.expected_shd_error + 1e-3  # type: ignore[operator]


def test_p_acyclic_is_reported_for_independent_marginals_only() -> None:
    independent = _probs(size=5, seed=12, kind="independent")
    candidate = dag([], nodes=independent.variables)
    exact = evaluate_graph(candidate, independent, samples=200)
    assert exact.diagnostics["p_acyclic_method"] == "exact"
    assert exact.p_acyclic_error == 0.0
    sampled = evaluate_graph(candidate, independent, max_exact=4, samples=2000)
    assert sampled.diagnostics["p_acyclic_method"] == "monte_carlo"
    assert abs(sampled.p_acyclic - exact.p_acyclic) < 4 * sampled.p_acyclic_error  # type: ignore[operator]
    assert evaluate_graph(dag([], nodes=NAMES[:4]), _probs()).p_acyclic is None


def test_samples_are_read_as_given() -> None:
    probs = _probs(seed=13)
    samples = graph_posterior(probs, seed=0)
    candidate = samples.best_graph
    assert evaluate_graph(candidate, samples).samples is samples
    with pytest.raises(ValueError, match="already carry their knowledge"):
        evaluate_graph(candidate, samples, DomainKnowledge().forbid("A", "B"))


@pytest.mark.parametrize(
    ("candidate", "error", "match"),
    [
        (dag(["A -> B"], nodes=("A", "B", "C")), ValueError, r"missing \['D'\]"),
        (dag(["A -> B"], nodes=("A", "B", "C", "D", "X")), ValueError, r"extra \['X'\]"),
        (mixed([("A", "-", "B", "-")], nodes=NAMES[:4]), ValueError, "as_kind"),
        ("A -> B", TypeError, "MixedGraph"),
    ],
)
def test_bad_candidates_raise_clearly(candidate: object, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        evaluate_graph(candidate, _probs())  # type: ignore[arg-type]


def test_a_bad_source_raises_clearly() -> None:
    with pytest.raises(TypeError, match="EdgeProbabilities or GraphSamples"):
        evaluate_graph(dag([], nodes=NAMES[:4]), np.zeros((4, 4)))  # type: ignore[arg-type]


def test_every_dag_on_three_variables_gets_a_level_in_order() -> None:
    probs = _probs(size=3, seed=14)
    samples = graph_posterior(probs, seed=0)
    levels = []
    for states in product(range(3), repeat=3):
        graph = np.zeros((3, 3), dtype=bool)
        for (i, j), state in zip(((0, 1), (0, 2), (1, 2)), states, strict=True):
            graph[i, j] = state == 1
            graph[j, i] = state == 2
        if not np.isfinite(samples._target.log_target(graph)):
            continue
        result = evaluate_graph(_dag(probs.variables, graph), samples)
        assert result.log_target is not None and result.credible_level is not None
        levels.append((result.log_target, result.credible_level))
    levels.sort()
    values = [level for _, level in levels]
    assert values == sorted(values, reverse=True)
    assert len(levels) == 25
