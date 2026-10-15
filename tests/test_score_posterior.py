# SPDX-License-Identifier: Apache-2.0
"""The BGe score and posteriors from a score (API-152, DEC-082 decision 5).

BGe is held to an independent derivation (the product of the normal-Wishart
model's sequential multivariate-t predictive densities) and to score
equivalence; score posteriors are held to enumeration, and MCMC to the
enumerated posterior on four variables.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tests.test_decode import all_dags

from ergodic import DomainKnowledge, discover, evaluate_graph, graph_posterior
from ergodic.data import tabular
from ergodic.discovery import BGe, PosteriorConvergenceWarning
from ergodic.graph import cpdag_of_dag, dag, dags_of_cpdag

if TYPE_CHECKING:
    from ergodic.data import TabularData
    from ergodic.graph import MixedGraph


def _gaussian(n: int = 200, seed: int = 0) -> tuple[TabularData, MixedGraph]:
    """a -> b -> d <- c, a -> c; linear Gaussian."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + rng.normal(size=n)
    c = -0.7 * a + rng.normal(size=n)
    d = 0.8 * b + 0.6 * c + rng.normal(size=n)
    truth = dag(["a -> b", "a -> c", "b -> d", "c -> d"])
    return tabular({"a": a, "b": b, "c": c, "d": d}), truth


def _sequential_log_likelihood(x: np.ndarray, alpha_mu: float, alpha_w: float, t: float) -> float:
    """log p(data) under the normal-Wishart model, one multivariate-t predictive at a time.

    Independent of BGe's closed form: the prior mean is the sample mean (as BGe
    takes it), and each row updates the posterior before the next is predicted.
    """
    dim = x.shape[1]
    nu = x.mean(axis=0)
    scale = t * np.eye(dim)
    am, aw = alpha_mu, alpha_w
    total = 0.0
    for row in x:
        dof = aw - dim + 1
        sigma = scale * (am + 1) / (am * dof)
        delta = row - nu
        quad = float(delta @ np.linalg.solve(sigma, delta))
        total += (
            math.lgamma((dof + dim) / 2)
            - math.lgamma(dof / 2)
            - dim / 2 * math.log(dof * math.pi)
            - 0.5 * np.linalg.slogdet(sigma)[1]
            - (dof + dim) / 2 * math.log1p(quad / dof)
        )
        scale = scale + am / (am + 1) * np.outer(delta, delta)
        nu = (am * nu + row) / (am + 1)
        am += 1
        aw += 1
    return total


def test_bge_matches_the_sequential_predictive_derivation() -> None:
    data, _ = _gaussian(n=40, seed=1)
    score = BGe(data)
    names = ["a", "b", "c", "d"]
    x = data.matrix(names)
    dim = 4
    assert score.alpha_w == dim + 2 and score.t == pytest.approx(0.5)
    full = sum(score.local(names[j], names[:j]) for j in range(dim))
    assert full == pytest.approx(_sequential_log_likelihood(x, 1.0, dim + 2.0, 0.5), abs=1e-9)
    # The empty graph: each variable's marginal model has alpha_w - n + 1 degrees of freedom.
    empty = sum(score.local(name) for name in names)
    marginals = sum(_sequential_log_likelihood(x[:, [j]], 1.0, 3.0, 0.5) for j in range(dim))
    assert empty == pytest.approx(marginals, abs=1e-9)
    y, parents = x[:, [3]], x[:, [1, 2]]
    assert score.local_from_arrays(y, parents) == pytest.approx(score.local("d", ["b", "c"]))


def test_bge_is_score_equivalent() -> None:
    data, truth = _gaussian(seed=2)
    score = BGe(data)
    totals = []
    for member in dags_of_cpdag(cpdag_of_dag(truth)):
        totals.append(sum(score.local(n, member.parents(n)) for n in sorted(member.nodes)))
    assert len(totals) > 1
    assert max(totals) - min(totals) < 1e-8
    chain = dag(["a -> b", "b -> c"], nodes=["a", "b", "c", "d"])
    reverse = dag(["c -> b", "b -> a"], nodes=["a", "b", "c", "d"])
    collider = dag(["a -> b", "c -> b"], nodes=["a", "b", "c", "d"])

    def total(graph: MixedGraph) -> float:
        return float(sum(score.local(n, graph.parents(n)) for n in sorted(graph.nodes)))

    assert total(chain) == pytest.approx(total(reverse), abs=1e-8)
    assert abs(total(chain) - total(collider)) > 1e-3


def test_bge_is_a_score_for_the_searches_and_checks_its_settings() -> None:
    data, truth = _gaussian(n=2000, seed=3)
    result = discover(data, "ges", score="bge")
    assert result.graph == cpdag_of_dag(truth)
    with pytest.raises(ValueError, match="alpha_mu"):
        BGe(alpha_mu=0.0)
    with pytest.raises(ValueError, match="exceed n \\+ 1"):
        BGe(data, alpha_w=5.0)
    rng = np.random.default_rng(0)
    categorical = tabular({"x": rng.normal(size=30), "g": rng.choice(["p", "q", "r"], size=30)})
    with pytest.raises(TypeError, match="single-column"):
        BGe(categorical)


def test_a_bge_posterior_is_exact_by_enumeration() -> None:
    data, _ = _gaussian(n=60, seed=4)
    samples = graph_posterior(data)
    assert samples.target == "score:bge" and samples.diagnostics["engine"] == "enumeration"
    assert samples.diagnostics["approximate"] is False
    score = BGe(data)
    names = samples.variables
    logs = []
    for graph in all_dags(4):
        logs.append(
            sum(
                score.local(names[j], [names[i] for i in np.flatnonzero(graph[:, j])])
                for j in range(4)
            )
        )
    values = np.asarray(logs)
    weights = np.exp(values - values.max())
    weights /= weights.sum()
    assert samples.adjacency.shape[0] == 543
    order = {g.tobytes(): w for g, w in zip(all_dags(4), weights, strict=True)}
    for graph, weight in zip(samples.adjacency, samples.weights, strict=True):
        assert weight == pytest.approx(order[graph.tobytes()], abs=1e-12)


def test_mcmc_on_a_score_agrees_with_enumeration() -> None:
    data, _ = _gaussian(n=60, seed=5)
    exact = graph_posterior(data, score="bge")
    mcmc = graph_posterior(data, score="bge", method="mcmc", samples=8000, seed=1)
    assert mcmc.diagnostics["edge_estimator"] == "frequency"
    assert mcmc.diagnostics["converged"]
    difference = np.abs(mcmc.edge_probabilities().values - exact.edge_probabilities().values)
    assert difference.max() < 0.03


def test_equivalent_dags_tie_in_the_credible_level() -> None:
    data, truth = _gaussian(n=500, seed=6)
    samples = graph_posterior(data)
    members = dags_of_cpdag(cpdag_of_dag(truth))
    result = evaluate_graph(truth, samples)
    probabilities = samples.exact_probabilities
    assert probabilities is not None
    member_matrices = {samples._target.matrix(m).tobytes() for m in members}
    tie = sum(
        p
        for g, p in zip(samples.adjacency, probabilities, strict=True)
        if g.tobytes() in member_matrices
    )
    assert result.credible_level == pytest.approx(result.mass_above + tie / 2, abs=1e-9)
    assert result.p_acyclic is None
    cpdag_result = evaluate_graph(cpdag_of_dag(truth), samples)
    assert cpdag_result.class_probability is not None
    assert cpdag_result.class_probability.probability == pytest.approx(tie, abs=1e-9)


def test_bic_posteriors_are_labelled_approximate_and_bdeu_runs_on_codes() -> None:
    data, _ = _gaussian(n=80, seed=7)
    samples = graph_posterior(data, score="bic")
    assert samples.diagnostics["approximate"] is True
    assert "approximation" in samples.diagnostics["note"]
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=300)
    y = (x ^ (rng.random(300) < 0.1)).astype(int)
    z = rng.integers(0, 3, size=300)
    discrete = tabular({"x": x.astype(str), "y": y.astype(str), "z": z.astype(str)})
    bdeu = graph_posterior(discrete)
    assert bdeu.target == "score:bdeu" and bdeu.diagnostics["approximate"] is False
    adjacent = bdeu.edge_probabilities().existence()
    assert adjacent[0, 1] > 0.9 and adjacent[0, 2] < 0.5


def test_knowledge_shapes_a_score_posterior() -> None:
    data, _ = _gaussian(n=60, seed=8)
    knowledge = DomainKnowledge().forbid("b", "d").require("a", "c", confidence=0.9)
    samples = graph_posterior(data, knowledge=knowledge)
    index = {name: i for i, name in enumerate(samples.variables)}
    assert not samples.adjacency[:, index["b"], index["d"]].any()
    for k in range(5):
        graph = samples.graph(k)
        bound = samples.score
        assert bound is not None
        base = float(sum(bound.local(n, graph.parents(n)) for n in sorted(graph.nodes)))
        assert samples.log_target(graph) == pytest.approx(base + knowledge.log_prior(graph))


def test_mixed_data_needs_an_explicit_score() -> None:
    rng = np.random.default_rng(1)
    data = tabular({"x": rng.normal(size=50), "g": rng.choice(["p", "q"], size=50).astype(object)})
    with pytest.raises(ValueError, match="mixed_bic"):
        graph_posterior(data)


@pytest.mark.filterwarnings("ignore::ergodic.discovery.PosteriorConvergenceWarning")
def test_score_mcmc_is_deterministic_given_the_seed() -> None:
    data, _ = _gaussian(n=60, seed=9)
    first = graph_posterior(data, method="mcmc", samples=200, seed=5)
    second = graph_posterior(data, method="mcmc", samples=200, seed=5)
    np.testing.assert_array_equal(first.draws, second.draws)
    assert PosteriorConvergenceWarning.__mro__[1] is UserWarning
