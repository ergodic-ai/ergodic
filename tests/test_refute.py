# SPDX-License-Identifier: Apache-2.0
"""Refutation checks: a sound estimate survives them, a scrambled treatment does not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import MixedGraph, TabularData, dag, estimate_effect, refute_effect, tabular
from ergodic.inference import Refutation


def _confounded_world(seed: int = 0, n: int = 4000) -> tuple[MixedGraph, TabularData]:
    """A world where Z confounds T -> Y and the true effect is 2.0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    y = 2.0 * t + 1.5 * z + rng.normal(size=n)
    df = pd.DataFrame({"Z": z, "T": t, "Y": y})
    data = tabular(df, types={"Z": "continuous", "T": "categorical", "Y": "continuous"})
    graph = dag(["Z -> T", "Z -> Y", "T -> Y"])
    return graph, data


def test_placebo_treatment_collapses_the_effect() -> None:
    graph, data = _confounded_world()
    est = estimate_effect(graph, data, "T", "Y", strategy="aipw").value
    ref = refute_effect(
        graph, data, "T", "Y", method="placebo_treatment", n_simulations=30, random_state=1
    )
    assert isinstance(ref, Refutation)
    assert abs(est - 2.0) < 0.2  # the real estimate is near the truth
    assert abs(ref.refuted_value) < 0.3  # a scrambled treatment has no effect
    assert ref.passed
    assert ref.p_value is not None and ref.p_value < 0.1  # the real effect is special


def test_random_common_cause_leaves_the_effect_stable() -> None:
    graph, data = _confounded_world()
    ref = refute_effect(
        graph, data, "T", "Y", method="random_common_cause", n_simulations=20, random_state=2
    )
    assert ref.p_value is None
    assert abs(ref.refuted_value - ref.estimand_value) < 0.1
    assert ref.passed


def test_data_subset_leaves_the_effect_stable() -> None:
    graph, data = _confounded_world()
    ref = refute_effect(
        graph, data, "T", "Y", method="data_subset", n_simulations=20, random_state=3
    )
    assert abs(ref.refuted_value - ref.estimand_value) < 0.2
    assert ref.passed


def test_refute_validates_its_arguments() -> None:
    graph, data = _confounded_world()
    with pytest.raises(ValueError, match="unknown refutation method"):
        refute_effect(graph, data, "T", "Y", method="nonsense")
    with pytest.raises(ValueError, match="at least 2"):
        refute_effect(graph, data, "T", "Y", n_simulations=1)


def test_refute_is_deterministic_given_a_seed() -> None:
    graph, data = _confounded_world()
    a = refute_effect(graph, data, "T", "Y", method="data_subset", n_simulations=15, random_state=7)
    b = refute_effect(graph, data, "T", "Y", method="data_subset", n_simulations=15, random_state=7)
    assert a.refuted_value == b.refuted_value


def test_refute_runs_on_a_bundled_dataset() -> None:
    from ergodic.datasets import load_birthweight

    ds = load_birthweight()
    edges = [f"{c} -> {ds.treatment}" for c in ds.confounders]
    edges += [f"{c} -> {ds.outcome}" for c in ds.confounders]
    edges.append(f"{ds.treatment} -> {ds.outcome}")
    graph = dag(edges)
    ref = refute_effect(
        graph,
        ds.data,
        ds.treatment,
        ds.outcome,
        method="placebo_treatment",
        strategy="aipw",
        treated=ds.treated,
        n_simulations=8,
        random_state=0,
    )
    # smoking really does lower birthweight, so a scrambled treatment loses the signal
    assert abs(ref.refuted_value) < 0.5 * abs(ref.estimand_value)
    assert ref.passed
