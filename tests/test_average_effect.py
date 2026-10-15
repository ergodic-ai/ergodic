# SPDX-License-Identifier: Apache-2.0
"""Effects over a posterior over DAGs (API-152, DEC-082 decision 6).

Small posteriors whose answers follow by hand: required edges pin ``t -> y`` and
``c -> y``, and the pair ``c, t`` keeps its three states with the model's own
probabilities (no state can close a cycle), so each adjustment question has a
closed-form mass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ergodic import (
    DomainKnowledge,
    EdgeProbabilities,
    adjustment_probability,
    average_effect,
    graph_posterior,
    tabular,
)
from ergodic.discovery import AveragedEffect, GraphSamples

P_CT, P_TC = 0.6, 0.15  # P(c -> t), P(t -> c); "no edge" gets the rest


def _posterior(knowledge: DomainKnowledge | None = None) -> GraphSamples:
    values = np.zeros((3, 3))
    names = ("c", "t", "y")
    values[0, 1], values[1, 0] = P_CT, P_TC
    values[0, 2], values[1, 2] = 0.5, 0.5  # required below, so these do not matter
    probs = EdgeProbabilities(names, values, kind="pair_state")
    pins = DomainKnowledge().require("t", "y").require("c", "y")
    return graph_posterior(probs, knowledge=pins if knowledge is None else pins & knowledge)


def _data(n: int = 2000, seed: int = 0) -> object:
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    t = (0.9 * c + rng.normal(size=n) > 0).astype(float)
    y = 2.0 * t + 1.5 * c + rng.normal(size=n)
    return tabular({"c": c, "t": t, "y": y}, types={"t": "continuous"})


def test_adjustment_probability_is_the_mass_where_the_set_is_valid() -> None:
    samples = _posterior()
    assert samples.exhaustive
    # {c} is valid unless t -> c makes c a mediator on t -> c -> y.
    confounder = adjustment_probability(samples, "t", "y", "c")
    assert confounder.exact and confounder.mc_error == 0.0
    assert confounder.probability == pytest.approx(1 - P_TC, abs=1e-9)
    # No adjustment is valid unless c -> t opens the back door.
    assert adjustment_probability(samples, "t", "y", []).probability == pytest.approx(
        1 - P_CT, abs=1e-9
    )


def test_adjustment_probability_from_draws_carries_an_error() -> None:
    rng = np.random.default_rng(1)
    values = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            _, forward, backward = rng.dirichlet([1.0, 1.0, 1.0])
            values[i, j], values[j, i] = forward, backward
    names = ("a", "b", "c", "d", "e", "f")
    samples = graph_posterior(EdgeProbabilities(names, values, kind="pair_state"), samples=2000)
    estimate = adjustment_probability(samples, "a", "f", ["b"])
    assert not estimate.exact and 0.0 < estimate.mc_error < 0.02
    with pytest.raises(ValueError, match="not variables"):
        adjustment_probability(samples, "a", "z", [])
    with pytest.raises(ValueError, match="overlap"):
        adjustment_probability(samples, "a", "f", ["a"])


def test_the_average_mixes_the_group_estimates_by_mass() -> None:
    samples = _posterior()
    data = _data()
    result = average_effect(samples, data, "t", "y")  # type: ignore[arg-type]
    assert isinstance(result, AveragedEffect) and result.exact
    groups = {g.adjustment_set: g for g in result.groups}
    assert set(groups) == {frozenset({"c"}), frozenset()}
    assert groups[frozenset({"c"})].mass == pytest.approx(1 - P_TC, abs=1e-9)
    assert groups[frozenset()].mass == pytest.approx(P_TC, abs=1e-9)
    assert groups[frozenset({"c"})].n_dags == 2
    adjusted, naive = groups[frozenset({"c"})], groups[frozenset()]
    assert abs(adjusted.value - 2.0) < 0.2 and naive.value > adjusted.value + 0.5
    mean = (1 - P_TC) * adjusted.value + P_TC * naive.value
    assert result.mean == pytest.approx(mean, abs=1e-9)
    between = (1 - P_TC) * (adjusted.value - mean) ** 2 + P_TC * (naive.value - mean) ** 2
    within = (1 - P_TC) * adjusted.std_error**2 + P_TC * naive.std_error**2  # type: ignore[operator]
    assert result.between_variance == pytest.approx(between, abs=1e-12)
    assert result.within_variance == pytest.approx(within, abs=1e-12)
    assert result.variance == pytest.approx(within + between, abs=1e-12)
    assert result.std == pytest.approx(math.sqrt(within + between))
    assert result.identified_mass == pytest.approx(1.0)
    assert result.zero_effect_mass == 0.0 and result.unidentified_mass == 0.0
    assert result.truncated_mass == 0.0
    assert result.mean_mc_error == 0.0  # every DAG enumerated
    frame = result.to_frame()
    assert list(frame["adjustment_set"]) == ["c", "(empty)"]
    assert "left out" in repr(result)


def test_no_path_mass_is_an_effect_of_zero_and_reported() -> None:
    values = np.array([[0.0, 0.7], [0.1, 0.0]])  # t -> y 0.7, y -> t 0.1, none 0.2
    samples = graph_posterior(EdgeProbabilities(("t", "y"), values, kind="pair_state"))
    rng = np.random.default_rng(2)
    t = (rng.random(1000) < 0.5).astype(float)
    data = tabular({"t": t, "y": 1.0 * t + rng.normal(size=1000)}, types={"t": "continuous"})
    result = average_effect(samples, data, "t", "y")
    assert result.zero_effect_mass == pytest.approx(0.3, abs=1e-9)
    zero = next(g for g in result.groups if g.adjustment_set is None)
    assert zero.value == 0.0 and zero.std_error == 0.0 and zero.estimate is None
    effect = next(g for g in result.groups if g.adjustment_set is not None)
    assert result.mean == pytest.approx(0.7 * effect.value, abs=1e-9)
    assert result.identified_mass == pytest.approx(1.0)


def test_a_variable_the_data_lacks_is_latent_and_its_confounding_is_unidentified() -> None:
    samples = _posterior()
    full = _data()
    observed = tabular(
        {"t": full.column("t"), "y": full.column("y")},  # type: ignore[attr-defined]
        types={"t": "continuous"},
    )
    result = average_effect(samples, observed, "t", "y")
    # c -> t with c hidden leaves t and y confounded: no valid set over {t, y}.
    assert result.unidentified_mass == pytest.approx(P_CT, abs=1e-9)
    assert result.identified_mass == pytest.approx(1 - P_CT, abs=1e-9)
    assert [g.adjustment_set for g in result.groups] == [frozenset()]
    assert result.mean == pytest.approx(result.groups[0].value)
    assert result.diagnostics["latent"] == ("c",)


def test_the_mean_from_draws_carries_a_monte_carlo_error() -> None:
    rng = np.random.default_rng(3)
    names = ("c", "t", "y", "m", "w")
    values = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            _, forward, backward = rng.dirichlet([1.0, 1.0, 1.0])
            values[i, j], values[j, i] = forward, backward
    pins = DomainKnowledge().require("t", "y")
    samples = graph_posterior(
        EdgeProbabilities(names, values, kind="pair_state"), knowledge=pins, samples=2000
    )
    x = rng.normal(size=(800, 5))
    x[:, 1] = (x[:, 1] > 0).astype(float)
    x[:, 2] += x[:, 1]
    data = tabular({name: x[:, k] for k, name in enumerate(names)})
    result = average_effect(samples, data, "t", "y")
    assert not result.exact and result.mean_mc_error > 0.0
    total = sum(g.mass for g in result.groups) + result.unidentified_mass + result.truncated_mass
    assert total == pytest.approx(1.0)


def test_sets_beyond_the_cap_are_truncated_and_reported() -> None:
    result = average_effect(_posterior(), _data(), "t", "y", max_sets=1)  # type: ignore[arg-type]
    assert result.truncated_mass == pytest.approx(P_TC, abs=1e-9)
    assert result.identified_mass == pytest.approx(1 - P_TC, abs=1e-9)
    assert len(result.groups) == 1
    assert result.mean == pytest.approx(result.groups[0].value)
    assert result.between_variance == 0.0


def test_the_average_is_deterministic_and_checks_its_inputs() -> None:
    samples, data = _posterior(), _data()
    first = average_effect(samples, data, "t", "y", strategy="outcome_regression", n_boot=50)  # type: ignore[arg-type]
    second = average_effect(samples, data, "t", "y", strategy="outcome_regression", n_boot=50)  # type: ignore[arg-type]
    assert first.mean == second.mean and first.variance == second.variance
    minimal = average_effect(samples, data, "t", "y", rule="minimal")  # type: ignore[arg-type]
    assert minimal.diagnostics["rule"] == "minimal"
    with pytest.raises(ValueError, match="rule"):
        average_effect(samples, data, "t", "y", rule="best")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_sets"):
        average_effect(samples, data, "t", "y", max_sets=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two different"):
        average_effect(samples, data, "t", "t")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GraphSamples"):
        average_effect(data, data, "t", "y")  # type: ignore[arg-type]
