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
