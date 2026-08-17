# SPDX-License-Identifier: Apache-2.0
"""Adversarial stress tests for the Phase-2 meta-learners.

Harder cases than the per-feature suites: the causal forest's honesty discipline
and infinitesimal-jackknife variance, the policy tree's global optimality and
degenerate collapses, the Q-aggregation simplex and tau-risk selection, and the
wired public ``cate``/``select_cate``/``Heterogeneity.policy_tree`` surface. Every
world is seeded and deterministic, with planted truths or closed-form references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from ergodic import cate, dag, estimate_effect, identify_effect, select_cate, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.inference import CATEResult, PolicyTree, dr_scores, policy_tree
from ergodic.inference._scores import DRScores
from ergodic.inference._select import _frank_wolfe_simplex, _tau_risk
from ergodic.inference._select import select_cate as _select
from ergodic.inference.estimators._forest import CausalForest
from ergodic.inference.explain._policy import PolicyNode

if TYPE_CHECKING:
    import numpy.typing as npt

# ===========================================================================
# Worlds
# ===========================================================================


def _linear_tau(
    n: int = 3000, seed: int = 0, *, a: float = 1.0, b: float = 1.0, n_decoy: int = 1
) -> tuple[TabularData, MixedGraph, np.ndarray]:
    """tau(x) = a + b*x1, confounded through x1, decoys in the outcome only."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    decoys = rng.normal(size=(n, n_decoy))
    prop = 1.0 / (1.0 + np.exp(-0.8 * x1))
    w = (rng.uniform(size=n) < prop).astype(float)
    y = 1.5 * x1 + decoys @ np.linspace(0.5, 1.0, n_decoy) + (a + b * x1) * w + rng.normal(size=n)
    cols: dict[str, np.ndarray] = {"W": w, "X1": x1, "Y": y}
    edges = ["X1 -> W", "X1 -> Y", "W -> Y"]
    for j in range(n_decoy):
        cols[f"X{j + 2}"] = decoys[:, j]
        edges.append(f"X{j + 2} -> Y")
    return tabular(pd.DataFrame(cols)), dag(edges), x1


def _no_hte(n: int = 2500, seed: int = 0, *, ate: float = 1.0) -> tuple[TabularData, MixedGraph]:
    """Constant-effect world: tau(x) = ate for all x, noise-only covariates."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    w = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.5 * x1 + 0.5 * x2 + ate * w + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "X2": x2, "Y": y}))
    return data, dag(["X1 -> Y", "X2 -> Y", "W -> Y"])


def _fit(data: TabularData, graph: MixedGraph, **kw: object) -> CausalForest:
    ident = identify_effect(graph, "W", "Y")
    return CausalForest(**kw).fit(ident, data)  # type: ignore[arg-type]


def _signflip(n: int = 2500, seed: int = 0) -> tuple[TabularData, MixedGraph]:
    """RCT with tau(z) = 2 z, so the optimal rule is treat iff z > 0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = (2.0 * z) * a + 0.5 * z + rng.normal(size=n)
    return tabular(pd.DataFrame({"A": a, "Z": z, "Y": y})), dag(["Z -> A", "Z -> Y", "A -> Y"])


# ===========================================================================
# CAUSAL FOREST
# ===========================================================================


def test_honest_does_not_overfit_no_hte_across_seeds() -> None:
    """Honest forest keeps a small spurious CATE spread on a no-HTE world; adaptive does not.

    Repeated over several seeds: the honest forest's per-unit spread is consistently
    a fraction of the adaptive forest's, the discipline that stops it reading
    heterogeneity into noise.
    """
    for seed in range(4):
        data, graph = _no_hte(n=2000, seed=seed, ate=1.0)
        honest = _fit(data, graph, n_trees=150, min_leaf=15, honest=True, random_state=seed).cate()
        adaptive = _fit(
            data, graph, n_trees=150, min_leaf=15, honest=False, random_state=seed
        ).cate()
        # both center on the truth
        assert abs(honest.ate().value - 1.0) < 0.25
        # honesty shrinks the spurious spread by a clear margin
        assert float(np.std(honest.values)) < 0.6 * float(np.std(adaptive.values))


def test_no_hte_calibration_does_not_falsely_reject() -> None:
    """On a constant-effect world the heterogeneity calibration test does not fire.

    The omnibus calibration test should not report strong HTE where there is none;
    its one-sided p-value stays well above a strict alpha.
    """
    data, graph = _no_hte(n=3000, seed=1, ate=1.0)
    res = _fit(data, graph, n_trees=200, min_leaf=20, random_state=1).cate()
    cal = res.explain().calibration_test()
    # the omnibus heterogeneity test does not reject "no HTE" at a strict alpha
    assert not cal.rejects(0.05)


def test_ij_coverage_near_nominal_over_seeds() -> None:
    """Honest IJ 90% intervals cover a + b*x1 above a conservative floor, finite-width.

    A native forest with a finite tree count is not expected to hit 0.90 exactly, but
    coverage clears the floor and the bands are genuine (not a degenerate zero-width
    pass): mean width is bounded above.
    """
    covs: list[float] = []
    widths: list[float] = []
    for seed in range(5):
        data, graph, x1 = _linear_tau(n=1500, seed=seed, a=1.0, b=1.0)
        res = _fit(data, graph, n_trees=300, min_leaf=15, random_state=seed).cate()
        lo, hi = res.interval(0.9)
        truth = 1.0 + x1
        covs.append(float(np.mean((lo <= truth) & (truth <= hi))))
        widths.append(float(np.mean(hi - lo)))
    assert float(np.mean(covs)) > 0.6
    # the intervals are real, not a runaway-wide trivial cover
    assert float(np.mean(widths)) < 3.0


def test_ij_intervals_widen_where_sparse() -> None:
    """Per-unit IJ standard errors are larger in the sparse covariate tails than the dense core."""
    data, graph, x1 = _linear_tau(n=3000, seed=0, a=1.0, b=2.0)
    res = _fit(data, graph, n_trees=300, min_leaf=15, random_state=0).cate()
    se = res.std_error
    assert se is not None
    dense = np.abs(x1) < 0.5
    sparse = np.abs(x1) > 2.0
    assert sparse.sum() > 20 and dense.sum() > 20
    assert float(se[sparse].mean()) > float(se[dense].mean())


def test_vim_clears_decoy_under_correlation() -> None:
    """The true modifier outranks a correlated decoy and a pure-noise decoy in VIM."""
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = 0.8 * x1 + 0.6 * rng.normal(size=n)  # correlated with x1 but does not modify
    x3 = rng.normal(size=n)  # pure noise
    prop = 1.0 / (1.0 + np.exp(-0.5 * x1))
    w = (rng.uniform(size=n) < prop).astype(float)
    y = 1.5 * x1 + 0.5 * x2 + 0.5 * x3 + (1.0 + 2.0 * x1) * w + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "X2": x2, "X3": x3, "Y": y}))
    g = dag(["X1 -> W", "X1 -> Y", "X2 -> Y", "X3 -> Y", "W -> Y"])
    vim = _fit(data, g, n_trees=300, min_leaf=15, random_state=0).variable_importance()
    imp = vim.to_dict()
    assert vim.top(1)[0][0] == "X1"
    assert imp["X1"] > imp["X2"]  # beats the correlated decoy
    assert imp["X1"] > imp["X3"]  # beats the noise decoy
    assert abs(sum(imp.values()) - 1.0) < 1e-9


def test_ate_consistency_three_ways() -> None:
    """The forest ATE agrees across cate-mean, the AIPW effect(), estimate_effect, and the truth."""
    data, graph, _ = _linear_tau(n=3000, seed=0, a=1.0, b=1.0)
    forest = _fit(data, graph, n_trees=200, min_leaf=10, random_state=0)
    ate_mean = forest.cate().ate().value
    ate_aipw = forest.effect().value
    direct = estimate_effect(graph, data, "W", "Y", strategy="dr_learner", random_state=0)
    assert abs(ate_mean - 1.0) < 0.2  # planted ATE
    assert abs(ate_mean - ate_aipw) < 0.1  # the two ATE paths agree
    assert abs(ate_mean - direct.value) < 0.15  # matches the standalone DR-learner


def test_tiny_n_stays_finite() -> None:
    """At n barely above 2*min_leaf the forest produces finite effects and SEs, not nonsense."""
    rng = np.random.default_rng(0)
    for n in (12, 20, 40):
        x1 = rng.normal(size=n)
        w = np.zeros(n)
        w[: n // 2] = 1.0  # guaranteed both arms
        y = x1 + w + rng.normal(size=n)
        data = tabular(pd.DataFrame({"W": w, "X1": x1, "Y": y}))
        g = dag(["X1 -> W", "X1 -> Y", "W -> Y"])
        res = _fit(data, g, n_trees=30, min_leaf=5, n_folds=2, random_state=0).cate()
        assert res.values.shape == (n,)
        assert bool(np.all(np.isfinite(res.values)))
        se = res.std_error
        assert se is not None and bool(np.all(np.isfinite(se) & (se >= 0.0)))


def test_constant_covariate_no_split() -> None:
    """A zero-variance covariate cannot be split: the CATE is flat and finite (the ATE)."""
    rng = np.random.default_rng(1)
    n = 800
    x1 = np.full(n, 3.0)
    w = (rng.uniform(size=n) < 0.5).astype(float)
    y = w + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "Y": y}))
    g = dag(["X1 -> W", "X1 -> Y", "W -> Y"])
    res = _fit(data, g, n_trees=50, min_leaf=10, n_folds=3, random_state=0).cate()
    assert bool(np.all(np.isfinite(res.values)))
    assert float(np.std(res.values)) < 1e-6  # nothing to split on
    assert abs(res.ate().value - 1.0) < 0.25


def test_single_covariate_recovers_modifier() -> None:
    """With exactly one covariate (the modifier) the forest still tracks tau(x)."""
    data, _graph, x1 = _linear_tau(n=2500, seed=4, a=1.0, b=1.5, n_decoy=1)
    # drop the decoy: keep only X1
    df = data.to_pandas()[["W", "X1", "Y"]]
    data1 = tabular(df)
    g1 = dag(["X1 -> W", "X1 -> Y", "W -> Y"])
    res = _fit(data1, g1, n_trees=200, min_leaf=12, random_state=4).cate()
    assert np.corrcoef(res.values, 1.0 + 1.5 * x1)[0, 1] > 0.55


def test_near_violated_positivity_stays_finite() -> None:
    """An extreme propensity (overlap near 0/1) is clipped, not divided-by-zero into nan/inf."""
    rng = np.random.default_rng(0)
    n = 1200
    x1 = rng.normal(size=n)
    prop = 1.0 / (1.0 + np.exp(-8.0 * x1))  # nearly deterministic assignment
    w = (rng.uniform(size=n) < prop).astype(float)
    y = x1 + w + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "Y": y}))
    g = dag(["X1 -> W", "X1 -> Y", "W -> Y"])
    res = _fit(data, g, n_trees=80, min_leaf=10, n_folds=3, random_state=0).cate()
    assert bool(np.all(np.isfinite(res.values)))
    eff = _fit(data, g, n_trees=80, min_leaf=10, n_folds=3, random_state=0).effect()
    assert np.isfinite(eff.value)


def test_continuous_treatment_guard() -> None:
    """Binary treatment in this phase; a continuous treatment raises pointing at binary."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    w = rng.normal(size=n)  # continuous
    y = w + x1 + rng.normal(size=n)
    data = tabular(pd.DataFrame({"W": w, "X1": x1, "Y": y}))
    g = dag(["X1 -> W", "X1 -> Y", "W -> Y"])
    with pytest.raises(NotImplementedError, match="binary"):
        CausalForest(random_state=0).fit(identify_effect(g, "W", "Y"), data)


def test_determinism_values_and_se() -> None:
    """A fixed seed gives bit-identical per-unit effects and standard errors."""
    data, graph, _ = _linear_tau(n=1500, seed=0)
    a = _fit(data, graph, n_trees=100, min_leaf=10, random_state=42).cate()
    b = _fit(data, graph, n_trees=100, min_leaf=10, random_state=42).cate()
    assert np.array_equal(a.values, b.values)
    sa, sb = a.std_error, b.std_error
    assert sa is not None and sb is not None and np.array_equal(sa, sb)


# ===========================================================================
# POLICY TREES
# ===========================================================================


def test_recovers_planted_signflip_rule() -> None:
    """A depth-1 tree splits z near 0 and treats the positive side (the planted optimum)."""
    data, g = _signflip(seed=0)
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, "Z", depth=1)
    assert tree.root.feature == "Z"
    assert tree.root.threshold is not None and abs(tree.root.threshold) < 0.2
    assert tree.root.left is not None and tree.root.right is not None
    assert tree.root.left.action == 0 and tree.root.right.action == 1


@pytest.mark.parametrize("depth", [1, 2])
def test_depth_le_2_matches_brute_force(depth: int) -> None:
    """The exhaustive depth-<=2 search equals the brute-force global optimum on small data."""
    rng = np.random.default_rng(13)
    n = 70
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    tau = np.where((z1 > 0) & (z2 > 0), 3.0, -1.0)
    y = tau * a + 0.3 * z1 + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z1": z1, "Z2": z2, "Y": y}))
    g = dag(["Z1 -> A", "Z2 -> A", "Z1 -> Y", "Z2 -> Y", "A -> Y"])
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, depth=depth)
    assert tree.exhaustive
    learned = float(np.sum((2.0 * tree.assign().astype(np.float64) - 1.0) * scores.scores))
    optimum = _brute_force_welfare(scores.scores, scores.covariates, depth)
    assert learned == pytest.approx(optimum, abs=1e-9)


def _brute_force_welfare(scores: np.ndarray, design: np.ndarray, depth: int) -> float:
    n, p = design.shape

    def leaf(mask: np.ndarray) -> float:
        return abs(float(scores[mask].sum()))

    def thresholds(col: np.ndarray) -> list[float]:
        u = np.unique(col)
        return [float((u[i] + u[i + 1]) / 2.0) for i in range(len(u) - 1)]

    def best_child(mask: np.ndarray, d: int) -> float:
        best = leaf(mask)
        if d <= 0 or int(mask.sum()) <= 1:
            return best
        for j in range(p):
            col = design[:, j]
            for t in thresholds(col[mask]):
                left = mask & (col <= t)
                right = mask & (col > t)
                best = max(best, best_child(left, d - 1) + best_child(right, d - 1))
        return best

    return best_child(np.ones(n, dtype=bool), depth)


def test_greedy_reaches_requested_depth_and_is_fast() -> None:
    """The deep (greedy) fall-through genuinely grows to the requested depth, monotone in welfare.

    Regression guard: a depth-3+ request must not silently collapse to a depth-2 tree.
    The greedy path adds welfare with depth (a leaf is always in the search, so a split
    is never welfare-lowering) and reaches the requested maximum depth.
    """
    rng = np.random.default_rng(5)
    n = 800
    z = rng.normal(size=(n, 3))
    a = (rng.uniform(size=n) < 0.5).astype(float)
    tau = 2.0 * z[:, 0] - 1.5 * z[:, 1] + 1.0 * z[:, 2]
    y = tau * a + 0.3 * z[:, 0] + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z1": z[:, 0], "Z2": z[:, 1], "Z3": z[:, 2], "Y": y}))
    g = dag(["Z1 -> A", "Z2 -> A", "Z3 -> A", "Z1 -> Y", "Z2 -> Y", "Z3 -> Y", "A -> Y"])
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)

    def max_depth(node: PolicyNode, d: int = 0) -> int:
        if node.is_leaf:
            return d
        assert node.left is not None and node.right is not None
        return max(max_depth(node.left, d + 1), max_depth(node.right, d + 1))

    t3 = policy_tree(scores, depth=3)
    t4 = policy_tree(scores, depth=4)
    assert not t3.exhaustive and not t4.exhaustive
    assert max_depth(t3.root) == 3
    assert max_depth(t4.root) == 4
    # deeper greedy welfare is at least the shallower one (leaf always available)
    w3 = float(np.sum((2.0 * t3.assign().astype(np.float64) - 1.0) * scores.scores))
    w4 = float(np.sum((2.0 * t4.assign().astype(np.float64) - 1.0) * scores.scores))
    assert w4 >= w3 - 1e-9


def test_policy_value_beats_trivials_under_hte() -> None:
    """Under real HTE the learned policy value exceeds treat-all and treat-none; regret >= 0."""
    data, g = _signflip(seed=2)
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, "Z", depth=2)
    assert tree.value > tree.treat_all_value
    assert tree.value > tree.treat_none_value
    assert tree.regret("treat_all") >= -1e-9
    assert tree.regret("treat_none") >= -1e-9
    assert tree.regret("best_trivial") >= -1e-9
    lo, hi = tree.interval(0.9)
    assert lo <= tree.value <= hi and tree.std_error > 0.0


def test_homogeneous_effect_value_equals_treat_all() -> None:
    """A homogeneous positive effect: the policy value collapses to treat-all (no gain to split)."""
    rng = np.random.default_rng(2)
    n = 3000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.5 * a + 0.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    g = dag(["Z -> A", "Z -> Y", "A -> Y"])
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, "Z", depth=2)
    assert tree.value == pytest.approx(tree.treat_all_value, abs=0.1)
    assert tree.assign().mean() > 0.9


def test_all_positive_scores_treat_all() -> None:
    """Uniformly positive DR scores -> a treat-all leaf (and treat-none its mirror)."""
    n = 200
    pos = np.abs(np.random.default_rng(0).normal(size=n)) + 0.5  # all > 0
    design = np.random.default_rng(1).normal(size=(n, 2))
    scores_obj = _make_scores(pos, design)
    tree = policy_tree(scores_obj, depth=2)
    assert bool(np.all(tree.assign() == 1))
    assert tree.value == pytest.approx(tree.treat_all_value, abs=1e-9)
    # the all-negative mirror: treat none
    tree_neg = policy_tree(_make_scores(-pos, design), depth=2)
    assert bool(np.all(tree_neg.assign() == 0))
    assert tree_neg.value == pytest.approx(tree_neg.treat_none_value, abs=1e-9)


def test_covariate_ties_handled() -> None:
    """A covariate with heavy ties never splits between equal values; the tree stays valid."""
    rng = np.random.default_rng(7)
    n = 400
    z_tied = np.repeat(np.array([0.0, 1.0]), n // 2)  # only two distinct values
    scores = np.where(z_tied > 0.5, 1.0, -1.0) + 0.2 * rng.normal(size=n)
    design = z_tied.reshape(-1, 1)
    tree = policy_tree(_make_scores(scores, design, names=("Zt",)), depth=2)
    # the split cleanly separates the two tie groups: the threshold is the lower tie
    # value (the ``column <= threshold`` convention), never a value between equal rows
    assert set(tree.assign().tolist()) <= {0, 1}
    if not tree.root.is_leaf:
        assert tree.root.threshold == 0.0  # left = the z==0 group, right = the z==1 group
        z = design[:, 0]
        assign = tree.assign()
        # the two tie levels each get a single, consistent action (no split within a tie)
        assert len(set(assign[z == 0.0].tolist())) == 1
        assert len(set(assign[z == 1.0].tolist())) == 1
        # and the positive-score group (z==1) is the one assigned treatment
        assert assign[z == 1.0][0] == 1


def test_policy_value_ci_widens_with_level() -> None:
    """The policy-value interval is centered on the value and widens with the level."""
    data, g = _signflip(seed=1)
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, "Z", depth=1)
    lo80, hi80 = tree.interval(0.8)
    lo99, hi99 = tree.interval(0.99)
    assert lo99 < lo80 <= tree.value <= hi80 < hi99
    with pytest.raises(ValueError, match="level"):
        tree.interval(1.5)


def _make_scores(
    scores: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    names: tuple[str, ...] = ("Z1", "Z2"),
) -> DRScores:
    n, p = design.shape
    return DRScores(
        scores=scores.astype(np.float64),
        mu0=np.zeros(n),
        mu1=np.zeros(n),
        propensity=np.full(n, 0.5),
        covariates=design.astype(np.float64),
        treatment=np.zeros(n),
        outcome=np.zeros(n),
        covariate_names=names[:p],
    )


# ===========================================================================
# SELECT / Q-AGGREGATION
# ===========================================================================


def test_tau_risk_ranks_correct_learner_first() -> None:
    """On a strong-interaction world the misspecified S-learner ranks last, a right one first."""
    rng = np.random.default_rng(7)
    n = 3000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = (3.0 * z) * a + 0.5 * z + 0.3 * rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    g = dag(["Z -> A", "Z -> Y", "A -> Y"])
    report = select_cate(
        g,
        data,
        "A",
        "Y",
        candidates=["s_learner", "t_learner", "dr_learner"],
        random_state=0,
        n_boot=50,
    )
    assert report.ranking[-1].method == "s_learner"
    assert report.selected_method in ("t_learner", "dr_learner")
    assert report.frame()["s_learner"] > 3.0 * report.frame()["t_learner"]


def test_aggregate_le_best_single_always() -> None:
    """The Q-aggregate DR tau-risk is never worse than the best single candidate's."""
    rng = np.random.default_rng(2)
    n = 2500
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(float)
    y = (1.0 + 1.5 * z) * a + 1.0 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    g = dag(["Z -> A", "Z -> Y", "A -> Y"])
    report = select_cate(g, data, "A", "Y", random_state=0, n_boot=50)
    assert report.aggregate_tau_risk <= report.best_single_tau_risk + 1e-8


def test_aggregate_strictly_better_on_complementary_errors() -> None:
    """Complementary-error candidates: the convex combination strictly beats either single."""
    rng = np.random.default_rng(0)
    n = 4000
    tau = rng.normal(size=n)
    gamma = tau + rng.normal(scale=2.0, size=n)  # E[gamma|x] = tau
    half = n // 2
    c1 = tau.copy()
    c1[half:] += 2.0
    c2 = tau.copy()
    c2[:half] += 2.0
    design = np.column_stack([c1, c2])
    per = np.array([_tau_risk(gamma, c1), _tau_risk(gamma, c2)])
    w = _frank_wolfe_simplex(design, gamma, per, nu=0.0, iters=500)
    assert _tau_risk(gamma, design @ w) < per.min() - 1e-6
    assert w.min() > 0.2  # spreads weight, not collapsing


def test_simplex_weights_valid_all_nu() -> None:
    """The Frank-Wolfe weights are a valid simplex point for every Q mix nu."""
    rng = np.random.default_rng(3)
    n = 600
    for nu in (0.0, 0.25, 0.5, 1.0):
        design = rng.normal(size=(n, 4))
        target = rng.normal(size=n)
        per = np.array([_tau_risk(target, design[:, j]) for j in range(4)])
        w = _frank_wolfe_simplex(design, target, per, nu=nu, iters=300)
        assert w.min() >= -1e-12
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_single_candidate_aggregate_equals_it() -> None:
    """One candidate: the aggregate weight is 1 and the aggregate risk equals that candidate's."""
    data, g = _signflip(seed=0)
    report = select_cate(g, data, "A", "Y", candidates=["t_learner"], random_state=0, n_boot=30)
    assert report.weights.shape == (1,)
    assert float(report.weights[0]) == pytest.approx(1.0, abs=1e-9)
    assert report.aggregate_tau_risk == pytest.approx(report.best_single_tau_risk, abs=1e-9)
    assert report.selected_method == "t_learner"


def test_identical_candidates_aggregate_is_valid() -> None:
    """Two identical candidate predictions: the aggregate matches them, weights valid."""
    rng = np.random.default_rng(1)
    n = 800
    c = rng.normal(size=n)
    gamma = c + rng.normal(scale=1.5, size=n)
    design = np.column_stack([c, c])
    per = np.array([_tau_risk(gamma, c), _tau_risk(gamma, c)])
    w = _frank_wolfe_simplex(design, gamma, per, nu=0.5, iters=300)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-9)
    assert _tau_risk(gamma, design @ w) == pytest.approx(per[0], abs=1e-9)


def test_non_adjustment_graph_raises() -> None:
    """A latent-confounded (non-back-door) effect cannot be CATE-selected."""
    rng = np.random.default_rng(0)
    n = 500
    u = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-u))).astype(float)
    y = a + u + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Y": y}))
    g = dag(["A -> Y", "U -> A", "U -> Y"], latent=["U"])
    with pytest.raises(ValueError, match=r"not identifiable|adjustment-identified"):
        select_cate(g, data, "A", "Y", random_state=0)


def test_selection_seed_stable() -> None:
    """The selection ranking, risks, and weights are bit-identical at a fixed seed."""
    rng = np.random.default_rng(2)
    n = 2000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = (1.0 + z) * a + z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    g = dag(["Z -> A", "Z -> Y", "A -> Y"])
    r1 = select_cate(g, data, "A", "Y", random_state=0, n_boot=40)
    r2 = select_cate(g, data, "A", "Y", random_state=0, n_boot=40)
    assert [s.method for s in r1.ranking] == [s.method for s in r2.ranking]
    np.testing.assert_allclose(r1.weights, r2.weights, atol=1e-12)
    assert r1.aggregate_tau_risk == pytest.approx(r2.aggregate_tau_risk, abs=1e-12)
    # the public top-level import is the same function
    assert select_cate is _select


# ===========================================================================
# INTEGRATION (the wired public API)
# ===========================================================================


@pytest.mark.parametrize("method", ["causal_forest", "grf"])
def test_cate_facade_forest_full_surface(method: str) -> None:
    """cate(method=causal_forest|grf) returns a CATEResult whose explain/interval/ate all work."""
    data, graph, _ = _linear_tau(n=2500, seed=0, a=1.0, b=1.0)
    res = cate(graph, data, "W", "Y", method=method, random_state=0)
    assert isinstance(res, CATEResult)
    assert res.source == "causal_forest"
    assert res.kind == "analytic"
    assert abs(res.ate().value - 1.0) < 0.25
    lo, hi = res.interval(0.9)
    assert bool(np.all(lo <= hi))
    # explain runs (the forest caches the doubly-robust scores)
    blp = res.explain().blp().frame()["X1"]
    assert blp["coef"] > 0.2


def test_heterogeneity_policy_tree_matches_standalone() -> None:
    """Heterogeneity.policy_tree() equals the standalone policy_tree on the same scores."""
    data, g = _signflip(seed=0)
    res = cate(g, data, "A", "Y", method="dr_learner", random_state=0, n_boot=30)
    via_record = res.explain().policy_tree("Z", depth=2)
    via_fn = policy_tree(res, "Z", depth=2)
    assert isinstance(via_record, PolicyTree)
    assert np.array_equal(via_record.assign(), via_fn.assign())
    assert via_record.value == pytest.approx(via_fn.value, abs=1e-12)


def test_forest_cate_feeds_policy_tree() -> None:
    """A forest CATEResult carries scores, so policy_tree learns a rule off it directly."""
    data, g = _signflip(seed=0)
    res = cate(g, data, "A", "Y", method="grf", random_state=0)
    tree = policy_tree(res, "Z", depth=1)
    assert isinstance(tree, PolicyTree)
    assert tree.value >= tree.treat_none_value - 1e-9
