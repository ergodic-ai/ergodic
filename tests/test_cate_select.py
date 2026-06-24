# SPDX-License-Identifier: Apache-2.0
"""Tests for CATE model selection and Q-aggregation.

Seeded, deterministic, against planted truths. A misspecification world makes one
candidate clearly right (a measurable DR tau-risk gap); the Frank-Wolfe convex
aggregate is checked to be at least as good as the best single (a guarantee) and
strictly better on complementary candidates; the ranking is stable under seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import CATEResult
from ergodic.inference._select import (
    SelectionReport,
    _frank_wolfe_simplex,
    _tau_risk,
    select_cate,
)


def _strong_interaction(n: int = 3000, seed: int = 7) -> tuple[TabularData, MixedGraph]:
    """RCT with a strong interaction tau(z) = 3 z and low noise.

    The S-learner pools the arms into one model and cannot represent the a*z
    interaction, so it is badly misspecified; the T- and DR-learners are right.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = (3.0 * z) * a + 0.5 * z + 0.3 * rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"])


def _linear_cate(n: int = 2500, seed: int = 2) -> tuple[TabularData, MixedGraph]:
    """Observational world with tau(z) = 1 + 1.5 z (every linear learner fits it)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(float)
    y = (1.0 + 1.5 * z) * a + 1.0 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"])


def test_picks_correctly_specified_learner() -> None:
    """The selector ranks the misspecified S-learner last and picks a right one."""
    data, g = _strong_interaction()
    report = select_cate(
        g,
        data,
        "A",
        "Y",
        candidates=["s_learner", "t_learner", "dr_learner"],
        random_state=0,
        n_boot=50,
    )
    risks = report.frame()
    # the pooled S-learner is far worse than the per-arm / doubly-robust learners
    assert risks["s_learner"] > 5.0 * risks["t_learner"]
    assert report.selected_method in ("t_learner", "dr_learner")
    assert report.ranking[-1].method == "s_learner"
    # the selected result is the fitted CATEResult of the lowest-risk candidate
    assert isinstance(report.selected, CATEResult)
    assert report.selected.source == report.selected_method
    assert report.best_single_tau_risk == report.ranking[0].tau_risk


def test_default_candidates_and_report_shape() -> None:
    """With no candidates given, the Phase-1 meta-learners are ranked and aggregated."""
    data, g = _linear_cate()
    report = select_cate(g, data, "A", "Y", random_state=0, n_boot=50)
    assert isinstance(report, SelectionReport)
    methods = {s.method for s in report.ranking}
    assert methods == {"dr_learner", "r_learner", "x_learner", "t_learner"}
    # the ranking is ascending in risk
    risks = [s.tau_risk for s in report.ranking]
    assert risks == sorted(risks)
    # the aggregate is a CATEResult that can run the Tier-B characterization
    assert isinstance(report.aggregate, CATEResult)
    assert report.aggregate.source == "q_aggregate"
    assert report.aggregate.explain() is not None
    # the weights are a convex combination over the aggregate methods
    assert report.weights.shape == (len(report.aggregate_methods),)
    assert report.weights.min() >= -1e-9
    assert float(report.weights.sum()) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_at_least_as_good_as_best_single() -> None:
    """The convex DR-loss aggregate's risk is <= the best single candidate's risk."""
    data, g = _linear_cate()
    report = select_cate(g, data, "A", "Y", random_state=0, n_boot=50)
    assert report.aggregate_tau_risk <= report.best_single_tau_risk + 1e-8


def test_aggregate_strictly_better_on_complementary_candidates() -> None:
    """On a mixed-specification (complementary-error) world the aggregate wins outright.

    Two candidates each unbiased on half the support and biased high on the other
    half: the convex combination cancels the complementary biases, so the
    Q-aggregation ensemble has strictly lower DR tau-risk than either single model.
    """
    rng = np.random.default_rng(0)
    n = 3000
    tau = rng.normal(size=n)
    gamma = tau + rng.normal(scale=2.0, size=n)  # E[gamma | x] = tau
    half = n // 2
    c1 = tau.copy()
    c1[half:] += 2.0  # biased on the second half
    c2 = tau.copy()
    c2[:half] += 2.0  # biased on the first half
    design = np.column_stack([c1, c2])
    per_model = np.array([_tau_risk(gamma, c1), _tau_risk(gamma, c2)])

    # nu = 0 is the pure convex DR-loss aggregate; it must beat both singles
    weights = _frank_wolfe_simplex(design, gamma, per_model, nu=0.0, iters=500)
    agg = design @ weights
    assert _tau_risk(gamma, agg) < per_model.min() - 1e-6
    # it spreads weight across both complementary models, not collapsing to one
    assert weights.min() > 0.2

    # the Q penalty (nu > 0) still keeps the aggregate at least as good as the best
    weights_q = _frank_wolfe_simplex(design, gamma, per_model, nu=0.5, iters=500)
    assert _tau_risk(gamma, design @ weights_q) <= per_model.min() + 1e-8


def test_ranking_stable_under_seed() -> None:
    """Two runs at the same seed give an identical ranking, risks, and weights."""
    data, g = _linear_cate()
    a = select_cate(g, data, "A", "Y", random_state=0, n_boot=50)
    b = select_cate(g, data, "A", "Y", random_state=0, n_boot=50)
    assert [s.method for s in a.ranking] == [s.method for s in b.ranking]
    for sa, sb in zip(a.ranking, b.ranking, strict=True):
        assert sa.tau_risk == pytest.approx(sb.tau_risk, abs=1e-12)
    assert a.selected_method == b.selected_method
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-12)
    assert a.aggregate_tau_risk == pytest.approx(b.aggregate_tau_risk, abs=1e-12)


def test_q_aggregate_appears_in_frame() -> None:
    """The frame maps each candidate and the aggregate to its DR tau-risk."""
    data, g = _linear_cate()
    report = select_cate(
        g, data, "A", "Y", candidates=["dr_learner", "t_learner"], random_state=0, n_boot=50
    )
    frame = report.frame()
    assert set(frame) == {"dr_learner", "t_learner", "q_aggregate"}
    assert frame["q_aggregate"] == pytest.approx(report.aggregate_tau_risk)
    assert report.risk("dr_learner") == report.frame()["dr_learner"]


def test_not_identifiable_and_empty_candidates_raise() -> None:
    """A non-back-door effect and an empty candidate list raise clear errors."""
    data, g = _linear_cate()
    with pytest.raises(ValueError, match="at least one CATE method"):
        select_cate(g, data, "A", "Y", candidates=[])

    # a latent-confounded effect with no adjustment set is not adjustment-identified
    rng = np.random.default_rng(0)
    n = 500
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-u))).astype(float)
    y = a + u + rng.normal(size=n)
    confounded_data = tabular(pd.DataFrame({"A": a, "Y": y}))
    confounded = dag(["A -> Y", "U -> A", "U -> Y"], latent=["U"])
    with pytest.raises(ValueError, match=r"not identifiable|adjustment-identified"):
        select_cate(confounded, confounded_data, "A", "Y", random_state=0)
