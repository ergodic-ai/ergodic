# SPDX-License-Identifier: Apache-2.0
"""Tests for the doubly-robust policy tree.

Seeded, deterministic, against planted optimal rules. A sign-flipping-effect world
plants ``treat iff z > 0``; an interaction world plants a depth-2 rectangle rule. The
exhaustive depth-<=2 search is checked against a brute-force optimum on small data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ergodic import dag, tabular
from ergodic.data import TabularData
from ergodic.graph import MixedGraph
from ergodic.identification import identify_effect
from ergodic.inference import cate, dr_scores
from ergodic.inference.explain._policy import PolicyNode, PolicyTree, policy_tree


def _signflip(n: int = 2500, seed: int = 0) -> tuple[TabularData, MixedGraph]:
    """RCT with tau(z) = 2 z, so the optimal rule is treat iff z > 0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = (2.0 * z) * a + 0.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"])


def _interaction(n: int = 1500, seed: int = 3) -> tuple[TabularData, MixedGraph]:
    """RCT where the effect is positive only when z1 > 0 AND z2 > 0 (a rectangle rule)."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    tau = np.where((z1 > 0) & (z2 > 0), 3.0, -1.0)
    y = tau * a + 0.3 * z1 + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z1": z1, "Z2": z2, "Y": y}))
    return data, dag(["Z1 -> A", "Z2 -> A", "Z1 -> Y", "Z2 -> Y", "A -> Y"])


def test_recovers_signflip_threshold() -> None:
    """A depth-1 tree splits z near 0 and treats the positive side."""
    data, g = _signflip()
    # policy_tree reads only the cached scores, so the bootstrap draw count is moot.
    res = cate(g, data, "A", "Y", method="dr_learner", random_state=0, n_boot=50)
    tree = policy_tree(res, "Z", depth=1)

    root = tree.root
    assert not root.is_leaf
    assert root.feature == "Z"
    assert root.threshold is not None
    assert abs(root.threshold) < 0.15  # the split is at z ~ 0
    # the high-z side is treated, the low-z side is not
    assert root.left is not None and root.right is not None
    assert root.left.action == 0
    assert root.right.action == 1
    # the learned rule assigns treatment to the positive-z units
    assign = tree.assign()
    z = data.to_pandas()["Z"].to_numpy()
    treated_z = z[assign == 1]
    untreated_z = z[assign == 0]
    assert treated_z.mean() > 0.0 > untreated_z.mean()


def test_policy_value_beats_treat_all_and_none() -> None:
    """The learned policy value exceeds both trivial baselines (regret >= 0)."""
    data, g = _signflip()
    res = cate(g, data, "A", "Y", method="dr_learner", random_state=0, n_boot=50)
    tree = policy_tree(res, "Z", depth=2)

    assert tree.value > tree.treat_all_value
    assert tree.value > tree.treat_none_value
    # regret against either trivial policy is non-negative (the rule is at least as good)
    assert tree.regret("treat_all") >= -1e-9
    assert tree.regret("treat_none") >= -1e-9
    assert tree.regret("best_trivial") >= -1e-9
    # the value has an honest interval from the influence terms
    lo, hi = tree.interval(0.9)
    assert lo <= tree.value <= hi
    assert tree.std_error > 0.0


def test_depth2_finds_interaction_rule() -> None:
    """A depth-2 tree recovers the z1>0 AND z2>0 rectangle and treats it positively."""
    data, g = _interaction()
    res = cate(g, data, "A", "Y", method="dr_learner", random_state=0, n_boot=50)
    tree = policy_tree(res, ("Z1", "Z2"), depth=2)

    df = data.to_pandas()
    z1 = df["Z1"].to_numpy()
    z2 = df["Z2"].to_numpy()
    assign = tree.assign()
    # the planted positive region should be treated, the rest mostly not
    planted = (z1 > 0) & (z2 > 0)
    assert assign[planted].mean() > 0.9
    assert assign[~planted].mean() < 0.1
    # both covariates are used (an interaction rule, not a single split)
    used = {node.feature for node in [tree.root, *_internal(tree)] if node.feature is not None}
    assert used == {"Z1", "Z2"}
    # the value beats treat-all by a clear margin (the rectangle carries the benefit)
    assert tree.value - tree.treat_all_value > 0.3


def test_value_and_regret_are_in_outcome_units() -> None:
    """The value is the mean outcome gain over treating no one; regret is not doubled.

    Regression (API-149): the value was ``mean[(2 pi - 1) Gamma]``, so ``regret()``
    reported twice the outcome gain its docstring promised (about 1.5 here). On the
    rectangle world treating the planted quarter gains ``0.25 * 3 = 0.75`` per unit
    over treating no one, and the ATE is ``0.25 * 3 - 0.75 * 1 = 0``.
    """
    data, g = _interaction(n=6000, seed=5)
    res = cate(g, data, "A", "Y", method="dr_learner", random_state=0, n_boot=20)
    tree = policy_tree(res, ("Z1", "Z2"), depth=2)
    assert tree.treat_none_value == 0.0
    assert tree.treat_all_value == pytest.approx(0.0, abs=0.1)
    assert tree.value == pytest.approx(0.75, abs=0.1)
    assert tree.regret("treat_none") == pytest.approx(0.75, abs=0.1)
    assert tree.regret("treat_all") == pytest.approx(0.75, abs=0.15)
    # the value is the influence mean mean[pi(X) Gamma] on the cached scores
    assert res._scores is not None
    scores = res._scores.scores
    assert tree.value == pytest.approx(float(np.mean(tree.assign() * scores)), abs=1e-12)
    lo, hi = tree.interval(0.9)
    assert lo < 0.75 < hi


def test_evaluate_scores_a_fixed_rule_on_held_out_units() -> None:
    """Regression (API-149): the in-sample value is optimistic; evaluate() is honest.

    The tree is chosen to maximize its value on the same scores, so the reported
    value runs high. Learning on one sample and evaluating on a fresh one removes
    the selection; the held-out interval covers the oracle value of the learned
    rule, and evaluating on the training scores reproduces the in-sample value.
    """
    train, g = _signflip(n=1500, seed=31)
    test, _ = _signflip(n=4000, seed=32)
    fit = cate(g, train, "A", "Y", method="dr_learner", random_state=0, n_boot=20)
    held = cate(g, test, "A", "Y", method="dr_learner", random_state=1, n_boot=20)
    tree = policy_tree(fit, "Z", depth=2)

    same = tree.evaluate(fit)
    assert same.value == pytest.approx(tree.value, abs=1e-12)
    assert same.std_error == pytest.approx(tree.std_error, abs=1e-12)

    out = tree.evaluate(held)
    assert out.n == 4000
    z = test.to_pandas()["Z"].to_numpy()
    oracle = float(np.mean(tree.predict(z.reshape(-1, 1)) * 2.0 * z))  # tau = 2 z
    lo, hi = out.interval(0.95)
    assert lo < oracle < hi
    assert out.regret("treat_none") == out.value
    assert out.regret("treat_all") == pytest.approx(out.value - out.treat_all_value)
    with pytest.raises(ValueError, match="split covariates"):
        PolicyTree(
            root=tree.root,
            covariates=("nope",),
            n=tree.n,
            depth=tree.depth,
            value=tree.value,
            std_error=tree.std_error,
            treat_all_value=tree.treat_all_value,
            treat_none_value=0.0,
            exhaustive=True,
        ).evaluate(held)


def test_in_sample_value_is_optimistic_on_average() -> None:
    """Across seeds the in-sample value exceeds the same rule's held-out value.

    A pure-noise world (no effect anywhere) makes the optimism plain: the true
    value of any rule is zero, but the search finds a positive in-sample value.
    """
    gaps = []
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        n = 800
        z = rng.normal(size=2 * n)
        a = (rng.uniform(size=2 * n) < 0.5).astype(float)
        y = 0.5 * z + rng.normal(size=2 * n)
        frame = pd.DataFrame({"A": a, "Z": z, "Y": y})
        g = dag(["Z -> A", "Z -> Y", "A -> Y"])
        fit = cate(g, tabular(frame.iloc[:n]), "A", "Y", random_state=0, uncertainty=False)
        held = cate(g, tabular(frame.iloc[n:]), "A", "Y", random_state=0, uncertainty=False)
        tree = policy_tree(fit, "Z", depth=2)
        gaps.append(tree.value - tree.evaluate(held).value)
    assert float(np.mean(gaps)) > 0.0


def _internal(tree: PolicyTree) -> list[PolicyNode]:
    """Internal (non-leaf) nodes of the tree."""
    out: list[PolicyNode] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if not node.is_leaf:
            out.append(node)
            assert node.left is not None and node.right is not None
            stack.extend([node.left, node.right])
    return out


def _brute_force_welfare(scores: np.ndarray, design: np.ndarray, depth: int) -> float:
    """The optimal depth-``depth`` welfare-sum by exhaustive enumeration (depth<=2)."""
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

    full = np.ones(n, dtype=bool)
    return best_child(full, depth)


@pytest.mark.parametrize("depth", [1, 2])
def test_matches_brute_force_optimum(depth: int) -> None:
    """The exhaustive search equals the brute-force global optimum on small data."""
    rng = np.random.default_rng(11)
    n = 60
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


def test_accepts_drscores_directly_and_selects_covariates() -> None:
    """policy_tree takes a DRScores and restricts to the named split covariates."""
    data, g = _interaction()
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    # only split on Z1: the tree's covariates are exactly the requested one
    tree = policy_tree(scores, "Z1", depth=1)
    assert tree.covariates == ("Z1",)
    assert tree.root.feature in (None, "Z1")
    # predict on a fresh design aligned to the selected covariates
    actions = tree.predict(np.array([[-2.0], [2.0]]))
    assert actions.shape == (2,)
    assert set(actions.tolist()) <= {0, 1}


def test_constant_effect_collapses_to_one_action() -> None:
    """A homogeneous-effect world yields a single dominant action (a leaf or near-leaf)."""
    rng = np.random.default_rng(2)
    n = 3000
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.5 * a + 0.5 * z + rng.normal(size=n)  # constant positive effect, treat everyone
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    g = dag(["Z -> A", "Z -> Y", "A -> Y"])
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, "Z", depth=2)
    # treating everyone is (near) optimal: the policy value is close to treat-all
    assert tree.value == pytest.approx(tree.treat_all_value, abs=0.1)
    # and almost every unit is assigned to treatment
    assert tree.assign().mean() > 0.9


def test_validation_errors() -> None:
    """Bad depth, unknown covariate names, and empty selections raise."""
    data, g = _signflip(n=200)
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    with pytest.raises(ValueError, match="depth must be at least 1"):
        policy_tree(scores, "Z", depth=0)
    with pytest.raises(ValueError, match="unknown split covariates"):
        policy_tree(scores, "nope")
    with pytest.raises(ValueError, match="at least one"):
        policy_tree(scores, [])
    with pytest.raises(TypeError, match="CATEResult or a DRScores"):
        policy_tree(object(), "Z")  # type: ignore[arg-type]


def test_cateresult_without_scores_raises() -> None:
    """A CATEResult that carries no scores cannot be policy-learned."""
    from ergodic.inference import CATEResult

    bare = CATEResult(
        treatment="A",
        outcome="Y",
        covariates=("Z",),
        n=10,
        kind="point",
        source="manual",
        _values=np.zeros(10),
    )
    with pytest.raises(ValueError, match="no doubly-robust scores"):
        policy_tree(bare, "Z")


def test_frozen_and_describe() -> None:
    """The PolicyTree is a frozen value object with a readable rendering."""
    import dataclasses

    data, g = _signflip(n=1500)
    scores = dr_scores(identify_effect(g, "A", "Y"), data, random_state=0)
    tree = policy_tree(scores, "Z", depth=1)
    assert isinstance(tree, PolicyTree)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tree.value = 0.0  # type: ignore[misc]
    text = tree.describe()
    assert "treat=" in text
    assert len(tree.leaves()) == 2


# -- categorical covariates with three or more levels (review F1) ----------------


def _three_level(n: int, seed: int, levels: tuple[str, ...] = ("a", "b", "c")) -> TabularData:
    """tau = 2 in level "c", -1 elsewhere; Z moves the outcome only."""
    rng = np.random.default_rng(seed)
    c = rng.choice(levels, size=n)
    z = rng.normal(size=n)
    a = (rng.uniform(size=n) < 0.5).astype(float)
    y = np.where(c == "c", 2.0, -1.0) * a + z + rng.normal(size=n)
    frame = pd.DataFrame({"A": a, "C": pd.Categorical(c), "Z": z, "Y": y})
    return tabular(frame)


_CAT_GRAPH = dag(["C -> A", "Z -> A", "C -> Y", "Z -> Y", "A -> Y"])


def test_a_three_level_covariate_is_named_per_column() -> None:
    """Regression: the design had one-hot columns but variable names, so names shifted.

    The probe's policy_tree reported a split on "Z" that was really the C==c
    indicator, and BLP returned four coefficients for three names.
    """
    fit = cate(_CAT_GRAPH, _three_level(2000, 1), "A", "Y", random_state=0, uncertainty=False)
    assert fit._scores is not None
    assert fit._scores.covariate_names == ("C==b", "C==c", "Z")
    assert fit._scores.column_variables == ("C", "C", "Z")
    tree = policy_tree(fit, depth=1)
    assert tree.root.feature == "C==c"  # the planted modifier, named for what it is
    blp = fit.explain().blp()
    assert blp.names == ("(intercept)", "C==b", "C==c", "Z")
    assert blp.coef.shape == (4,)
    assert blp.coef[2] == pytest.approx(3.0, abs=0.5)  # level c vs a: 2 - (-1)
    # naming the variable selects its whole block
    by_variable = policy_tree(fit, "C", depth=1)
    assert by_variable.covariates == ("C==b", "C==c")


def test_evaluate_refuses_a_differently_coded_categorical() -> None:
    """A held-out sample with other categories would map a column to another group."""
    fit = cate(_CAT_GRAPH, _three_level(2000, 1), "A", "Y", random_state=0, uncertainty=False)
    tree = policy_tree(fit, "C", depth=1)
    same = cate(_CAT_GRAPH, _three_level(2000, 2), "A", "Y", random_state=0, uncertainty=False)
    value = tree.evaluate(same)
    assert value.value == pytest.approx(2.0 / 3.0, abs=0.15)  # treat level c: 2 * 1/3
    assert np.isfinite(value.regret_std_error) and value.regret_std_error > 0.0
    missing = _three_level(2000, 3, levels=("b", "c"))
    held = cate(_CAT_GRAPH, missing, "A", "Y", random_state=0, uncertainty=False)
    with pytest.raises(ValueError, match="categories differ"):
        tree.evaluate(held)
    extra = _three_level(2000, 4, levels=("a", "b", "c", "d"))
    held = cate(_CAT_GRAPH, extra, "A", "Y", random_state=0, uncertainty=False)
    with pytest.raises(ValueError, match="categories differ"):
        tree.evaluate(held)
    # a tree on Z alone is unaffected by C's coding
    z_tree = policy_tree(fit, "Z", depth=1)
    assert np.isfinite(z_tree.evaluate(held).value)


def test_dr_scores_refuse_unnamed_columns() -> None:
    from ergodic.inference._scores import DRScores

    n = 5
    zeros = np.zeros(n)
    with pytest.raises(ValueError, match="3 columns but 2 names"):
        DRScores(
            scores=zeros,
            mu0=zeros,
            mu1=zeros,
            propensity=zeros + 0.5,
            covariates=np.zeros((n, 3)),
            treatment=zeros,
            outcome=zeros,
            covariate_names=("C", "Z"),
        )
