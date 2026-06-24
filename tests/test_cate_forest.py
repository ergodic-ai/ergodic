# SPDX-License-Identifier: Apache-2.0
"""Tests for the native honest causal forest (GRF) CATE estimator.

Seeded and deterministic, against planted truths. The forest is tested directly
(the ``cate`` facade is wired by the orchestrator, not here). The worlds plant a
known linear modifier ``tau(x) = a + b * x1`` so the per-unit effects, the ATE, the
honest interval coverage, and the variable importance can all be checked against
the truth and against a homogeneous null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, estimate_effect, identify_effect, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import CATEResult
from ergodic.inference.estimators._forest import (
    CausalForest,
    VariableImportance,
    _solve_local_moment,
)


def _linear_tau(
    n: int = 3000, seed: int = 0, *, a: float = 1.0, b: float = 1.0, n_decoy: int = 2
) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """Continuous outcome with tau(x) = a + b*x1, confounded through x1.

    The treatment is assigned by a propensity in x1 (so naive differencing is
    biased), and decoy covariates x2.. enter the outcome but not the effect. The ATE
    is a (x1 is mean-zero), and the CATE tracks a + b*x1.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    decoys = rng.normal(size=(n, n_decoy))
    propensity = 1.0 / (1.0 + np.exp(-0.8 * x1))
    w = (rng.uniform(size=n) < propensity).astype(float)
    tau = a + b * x1
    baseline = 1.5 * x1 + decoys @ np.linspace(0.5, 1.0, n_decoy)
    y = baseline + tau * w + rng.normal(size=n)
    cols: dict[str, np.ndarray] = {"W": w, "X1": x1, "Y": y}
    edges = ["X1 -> W", "X1 -> Y", "W -> Y"]
    for j in range(n_decoy):
        cols[f"X{j + 2}"] = decoys[:, j]
        edges.append(f"X{j + 2} -> Y")
    data = tabular(pd.DataFrame(cols))
    return data, dag(edges), x1


def _no_hte(
    n: int = 3000, seed: int = 0, *, ate: float = 1.0
) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """A constant-effect world: tau(x) = ate for all x, noise-only covariates.

    Used for the honest-vs-adaptive calibration check: the forest should not invent
    heterogeneity that is not there.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    w = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.5 * x1 + 0.5 * x2 + ate * w + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "X2": x2, "Y": y}))
    return data, dag(["X1 -> Y", "X2 -> Y", "W -> Y"]), x1


def _fit(data: TabularData, graph: MixedGraph, **kw: object) -> CausalForest:
    ident = identify_effect(graph, "W", "Y")
    return CausalForest(**kw).fit(ident, data)  # type: ignore[arg-type]


# -- recovering the planted CATE ---------------------------------------------


def test_recovers_linear_tau() -> None:
    """Per-unit CATE correlates with a + b*x1, and the ATE matches the truth."""
    data, graph, x1 = _linear_tau(n=3000, seed=0, a=1.0, b=1.0)
    res = _fit(data, graph, n_trees=200, min_leaf=10, random_state=0).cate()
    assert isinstance(res, CATEResult)
    assert res.estimand == "cate"
    truth = 1.0 + x1
    assert np.corrcoef(res.values, truth)[0, 1] > 0.6
    assert abs(res.ate().value - 1.0) < 0.2


def test_recovers_stronger_modifier() -> None:
    """A larger slope is still tracked, and the spread of the CATE rises with it."""
    data, graph, x1 = _linear_tau(n=3000, seed=2, a=0.0, b=2.0)
    res = _fit(data, graph, n_trees=200, min_leaf=10, random_state=2).cate()
    assert np.corrcoef(res.values, 2.0 * x1)[0, 1] > 0.6
    # the high-x1 units have a larger effect than the low-x1 units
    high = res.values[x1 > 1.0].mean()
    low = res.values[x1 < -1.0].mean()
    assert high - low > 1.0


# -- the ATE matches estimate_effect -----------------------------------------


def test_ate_matches_estimate_effect_and_truth() -> None:
    data, graph, _ = _linear_tau(n=3000, seed=0, a=1.0, b=1.0)
    forest = _fit(data, graph, n_trees=200, min_leaf=10, random_state=0)
    ate = forest.cate().ate()
    assert abs(ate.value - 1.0) < 0.2  # planted ATE
    direct = estimate_effect(graph, data, "W", "Y", strategy="dr_learner", random_state=0)
    assert abs(ate.value - direct.value) < 0.15


def test_effect_surface_reports_ate() -> None:
    """The estimator's own effect() (mean per-unit CATE) lands on the planted ATE."""
    data, graph, _ = _linear_tau(n=2500, seed=1, a=1.0, b=0.5)
    forest = _fit(data, graph, n_trees=150, min_leaf=10, random_state=1)
    eff = forest.effect()
    assert abs(eff.value - 1.0) < 0.25


# -- honest infinitesimal-jackknife coverage ---------------------------------


def test_ij_per_unit_interval_shape_and_kind() -> None:
    data, graph, _ = _linear_tau(n=2000, seed=0)
    res = _fit(data, graph, n_trees=200, min_leaf=10, random_state=0).cate()
    assert res.kind == "analytic"
    se = res.std_error
    assert se is not None and se.shape == (data.n_rows,)
    assert bool(np.all(se >= 0.0))
    assert bool(np.all(np.isfinite(se)))
    lo, hi = res.interval(0.9)
    assert bool(np.all(lo <= hi))


def test_ij_coverage_near_nominal() -> None:
    """Honest IJ 90% per-unit intervals cover a + b*x1 near the nominal rate.

    Averaged over several seeds, the empirical coverage clears a conservative floor.
    A native forest with a finite tree count is not expected to hit 0.90 exactly,
    so the floor is set below nominal but well above a degenerate (zero-width) band.
    """
    coverages: list[float] = []
    for seed in range(5):
        data, graph, x1 = _linear_tau(n=1500, seed=seed, a=1.0, b=1.0)
        res = _fit(data, graph, n_trees=300, min_leaf=15, random_state=seed).cate()
        lo, hi = res.interval(0.9)
        truth = 1.0 + x1
        coverages.append(float(np.mean((lo <= truth) & (truth <= hi))))
    mean_cov = float(np.mean(coverages))
    assert mean_cov > 0.6, f"mean coverage {mean_cov:.3f} below the floor"


# -- honest vs adaptive (calibration on a no-HTE world) ----------------------


def test_honest_does_not_overfit_noise() -> None:
    """On a constant-effect world the honest forest's CATE spread is small.

    Honesty is the discipline that keeps the forest from reading heterogeneity into
    noise. The honest forest's per-unit predictions vary far less than the adaptive
    forest's on a world with no true modifier.
    """
    data, graph, _ = _no_hte(n=3000, seed=0, ate=1.0)
    honest = _fit(data, graph, n_trees=200, min_leaf=15, honest=True, random_state=0).cate()
    adaptive = _fit(data, graph, n_trees=200, min_leaf=15, honest=False, random_state=0).cate()
    # both center on the truth
    assert abs(honest.ate().value - 1.0) < 0.2
    # honesty shrinks the spurious spread relative to the adaptive forest
    assert float(np.std(honest.values)) <= float(np.std(adaptive.values)) + 1e-9
    # and the honest spread is small in absolute terms (no real HTE to find)
    assert float(np.std(honest.values)) < 0.5


def test_no_hte_blp_slope_covers_zero() -> None:
    """With no modifier, the honest CATE has no systematic gradient in x1."""
    data, graph, x1 = _no_hte(n=3000, seed=1, ate=1.0)
    res = _fit(data, graph, n_trees=200, min_leaf=15, random_state=1).cate()
    # regress the per-unit effect on x1; the slope should be near zero
    slope = float(np.polyfit(x1, res.values, 1)[0])
    assert abs(slope) < 0.25


# -- variable importance ------------------------------------------------------


def test_variable_importance_fingers_modifier() -> None:
    """The depth-weighted split frequency favors the true modifier over decoys."""
    data, graph, _ = _linear_tau(n=3000, seed=0, a=1.0, b=2.0, n_decoy=3)
    forest = _fit(data, graph, n_trees=200, min_leaf=10, random_state=0)
    vim = forest.variable_importance()
    assert isinstance(vim, VariableImportance)
    importance = vim.to_dict()
    assert set(importance) == {"X1", "X2", "X3", "X4"}
    assert abs(sum(importance.values()) - 1.0) < 1e-9
    # the true modifier is the top driver, and clears every decoy
    assert vim.top(1)[0][0] == "X1"
    for decoy in ("X2", "X3", "X4"):
        assert importance["X1"] > importance[decoy]


def test_variable_importance_top_k() -> None:
    data, graph, _ = _linear_tau(n=2000, seed=3, b=1.5, n_decoy=2)
    vim = _fit(data, graph, n_trees=120, min_leaf=10, random_state=3).variable_importance()
    top = vim.top(2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]  # descending


# -- determinism --------------------------------------------------------------


def test_determinism_under_seed() -> None:
    """A fixed seed gives bit-identical per-unit effects and standard errors."""
    data, graph, _ = _linear_tau(n=1500, seed=0)
    a = _fit(data, graph, n_trees=100, min_leaf=10, random_state=42).cate()
    b = _fit(data, graph, n_trees=100, min_leaf=10, random_state=42).cate()
    assert np.array_equal(a.values, b.values)
    se_a, se_b = a.std_error, b.std_error
    assert se_a is not None and se_b is not None
    assert np.array_equal(se_a, se_b)


def test_different_seeds_differ() -> None:
    data, graph, _ = _linear_tau(n=1500, seed=0)
    a = _fit(data, graph, n_trees=100, min_leaf=10, random_state=1).cate()
    b = _fit(data, graph, n_trees=100, min_leaf=10, random_state=2).cate()
    assert not np.array_equal(a.values, b.values)


# -- out-of-sample prediction -------------------------------------------------


def test_predicts_out_of_sample() -> None:
    """A forest fit on one sample scores a fresh design and tracks its truth."""
    train, graph, _ = _linear_tau(n=3000, seed=0, a=1.0, b=1.0)
    fresh, _, x1_fresh = _linear_tau(n=1500, seed=7, a=1.0, b=1.0)
    forest = _fit(train, graph, n_trees=200, min_leaf=10, random_state=0)
    res = forest.cate(fresh)
    assert res.shape == (fresh.n_rows,)
    assert np.corrcoef(res.values, 1.0 + x1_fresh)[0, 1] > 0.5


# -- explainability hook ------------------------------------------------------


def test_cate_caches_scores_for_explain() -> None:
    """The forest caches the doubly-robust scores, so explain() runs."""
    data, graph, _ = _linear_tau(n=3000, seed=0, a=1.0, b=1.0)
    res = _fit(data, graph, n_trees=150, min_leaf=10, random_state=0).cate()
    h = res.explain()
    # the planted modifier shows up as a positive BLP slope on X1
    z = h.blp().frame()["X1"]
    assert z["coef"] > 0.3


# -- guards -------------------------------------------------------------------


def test_continuous_treatment_raises() -> None:
    """Binary treatment in this phase; a continuous treatment is a documented defer."""
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n)
    w = rng.normal(size=n)  # continuous treatment
    y = w + x1 + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "Y": y}))
    graph = dag(["X1 -> W", "X1 -> Y", "W -> Y"])
    forest = CausalForest(random_state=0)
    with pytest.raises(NotImplementedError, match="binary"):
        forest.fit(identify_effect(graph, "W", "Y"), data)


def test_cate_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        CausalForest().cate()


def test_variable_importance_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        CausalForest().variable_importance()


def test_bad_subsample_raises() -> None:
    with pytest.raises(ValueError, match="subsample"):
        CausalForest(subsample=0.0)
    with pytest.raises(ValueError, match="subsample"):
        CausalForest(subsample=1.5)


def test_bad_n_trees_raises() -> None:
    with pytest.raises(ValueError, match="n_trees"):
        CausalForest(n_trees=0)


# -- the local-moment solver --------------------------------------------------


def test_solve_local_moment_recovers_slope() -> None:
    """The closed-form local R-learner recovers a known Robinson slope."""
    rng = np.random.default_rng(0)
    n = 400
    w_res = rng.normal(size=n)
    y_res = 2.5 * w_res + 0.1 * rng.normal(size=n)
    alpha = np.ones((1, n), dtype=np.float64) / n  # one neighborhood = all rows
    tau = _solve_local_moment(alpha, w_res, y_res)
    assert tau.shape == (1,)
    assert abs(tau[0] - 2.5) < 0.1


def test_solve_local_moment_zero_without_variation() -> None:
    """A neighborhood with no treatment-residual spread returns a zero effect."""
    n = 50
    w_res = np.zeros(n, dtype=np.float64)
    y_res = np.ones(n, dtype=np.float64)
    alpha = np.ones((1, n), dtype=np.float64) / n
    tau = _solve_local_moment(alpha, w_res, y_res)
    assert tau[0] == 0.0
