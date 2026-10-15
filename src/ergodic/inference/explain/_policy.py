# SPDX-License-Identifier: Apache-2.0
"""Interpretable optimal-policy learning off the doubly-robust scores.

A shallow decision tree that assigns a binary treatment to maximize doubly-robust
welfare (Athey-Wager 2021; Zhou-Athey-Wager 2023). With the cross-fit AIPW score
``Gamma_i`` (its conditional mean is the CATE), the value of a policy
``pi: X -> {0, 1}``, measured against treating no one, is the empirical welfare
gain in outcome units

    V(pi) = mean_i[pi(X_i) Gamma_i]
          = mean_i[pi(X_i) Gamma_i^(1) + (1 - pi(X_i)) Gamma_i^(0)] - mean_i[Gamma_i^(0)].

The search maximizes the equivalent objective ``mean_i[(2 pi(X_i) - 1) Gamma_i] =
2 V(pi) - mean_i[Gamma_i]`` (the same argmax), so a leaf that covers a set of
units is best treated (action 1) when the sum of the scores over the leaf is
positive and best left untreated (action 0) otherwise; the leaf's contribution to
that objective is then ``|sum Gamma|``. Maximizing welfare over a
depth-``k`` axis-aligned tree is therefore maximizing the sum of those leaf
absolute-sums. At depth <= 2 the library searches every (feature, threshold) split
exhaustively, which is the globally optimal regime (the Zhou-Athey-Wager guarantee);
a deeper tree falls back to a greedy recursive best-split, which is documented and
is not guaranteed optimal.

The tree STRUCTURE is a point estimate (no interval on which split is "the"
split). The policy VALUE reported on the tree is the sample mean of the per-unit
terms ``pi(X_i) Gamma_i`` over the same scores the tree was chosen to maximize, so
it is optimistic: the search picked the rule that looks best on these units, noise
included, and the standard error and interval do not account for that selection.
For an honest value, learn the tree on one sample and score it on another with
:meth:`PolicyTree.evaluate`: there the rule is fixed before the scores are seen,
so the mean of ``pi(X_i) Gamma_i`` inherits a root-n central limit theorem from the
orthogonal score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from .._perunit import CATEResult
    from .._scores import DRScores


# The exhaustive (globally optimal) search is used up to this depth; deeper trees
# use the greedy recursive best-split (documented as not guaranteed optimal).
_EXHAUSTIVE_DEPTH = 2


@dataclass(frozen=True)
class PolicyNode:
    """One node of a policy tree: a split (internal) or an action (leaf).

    An internal node carries ``feature``/``feature_index``/``threshold`` and the two
    children (``left`` is ``feature <= threshold``); a leaf carries ``action`` (0 or
    1). ``n`` is the number of fit units that reached the node and ``value`` is the
    node's welfare contribution in score units (``sum_i (2 a_i - 1) Gamma_i`` over the
    units it covers under the subtree's actions).
    """

    n: int
    value: float
    action: int | None = None
    feature: str | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: PolicyNode | None = None
    right: PolicyNode | None = None

    @property
    def is_leaf(self) -> bool:
        """True when this node assigns an action rather than splitting."""
        return self.action is not None

    def predict(self, x: npt.NDArray[np.float64]) -> int:
        """The action this subtree assigns to one covariate row ``x``."""
        node = self
        while not node.is_leaf:
            assert node.feature_index is not None and node.threshold is not None
            child = node.left if x[node.feature_index] <= node.threshold else node.right
            assert child is not None
            node = child
        assert node.action is not None
        return node.action


@dataclass(frozen=True)
class PolicyValue:
    """A fixed policy's value on a set of units, from :meth:`PolicyTree.evaluate`.

    ``value`` is the mean outcome gain over treating no one (outcome units per
    unit), ``std_error`` its standard error, and ``treat_all_value`` the
    treat-everyone baseline (the ATE) on the same units.
    """

    value: float
    std_error: float
    treat_all_value: float
    n: int

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """A normal confidence interval for the value at ``level``."""
        from .._stats import _norm_ppf

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = _norm_ppf((1.0 + level) / 2.0)
        return self.value - z * self.std_error, self.value + z * self.std_error

    def regret(self, baseline: str = "treat_all") -> float:
        """``value`` minus the ``"treat_all"``, ``"treat_none"``, or ``"best_trivial"`` value."""
        if baseline == "treat_all":
            return self.value - self.treat_all_value
        if baseline == "treat_none":
            return self.value
        if baseline == "best_trivial":
            return self.value - max(self.treat_all_value, 0.0)
        raise ValueError(
            f"baseline must be 'treat_all', 'treat_none', or 'best_trivial', got {baseline!r}"
        )


@dataclass(frozen=True)
class PolicyTree:
    """A shallow doubly-robust optimal-policy tree with its value and regret.

    ``root`` is the learned tree over the named ``covariates``; :meth:`predict` and
    :meth:`assign` evaluate it. ``value``/``std_error`` are the estimated policy
    value ``V(pi_hat) = mean[pi_hat(X) Gamma]`` and its standard error from the
    per-unit influence terms, in outcome units per unit and measured against
    treating no one (the mean outcome gain of following the rule rather than
    treating nobody). On that scale ``treat_none_value`` is 0 and
    ``treat_all_value`` is the doubly-robust ATE, and :meth:`regret` is the value
    gap against either, also in outcome units per unit. These are measured on the
    units the tree was learned on, so they are optimistic (see
    :meth:`evaluate` for the value on held-out units). ``depth`` is the requested depth
    and ``exhaustive`` records whether the globally optimal search was used.
    """

    root: PolicyNode
    covariates: tuple[str, ...]
    n: int
    depth: int
    value: float
    std_error: float
    treat_all_value: float
    treat_none_value: float
    exhaustive: bool
    treatment: str = ""
    outcome: str = ""
    _scores: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0), repr=False)
    _design: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)), repr=False)

    # -- evaluation ----------------------------------------------------------

    def predict(self, covariates: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """The per-row action (0/1) for a covariate matrix aligned to ``covariates``."""
        x = np.asarray(covariates, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != len(self.covariates):
            raise ValueError(
                f"covariate matrix has {x.shape[1]} columns, expected {len(self.covariates)}"
            )
        return np.asarray([self.root.predict(row) for row in x], dtype=np.int64)

    def assign(self) -> npt.NDArray[np.int64]:
        """The fitted policy's action for each unit it was learned on."""
        return self.predict(self._design)

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """A normal interval for the in-sample policy value at ``level``.

        It ignores that the tree was chosen to maximize this value, so it is
        optimistic; :meth:`evaluate` gives a valid interval on held-out units.
        """
        from .._stats import _norm_ppf

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        z = _norm_ppf((1.0 + level) / 2.0)
        return self.value - z * self.std_error, self.value + z * self.std_error

    def regret(self, baseline: str = "treat_all") -> float:
        """The learned policy's welfare GAIN over a trivial baseline (the baseline's regret).

        ``baseline`` is ``"treat_all"``, ``"treat_none"``, or ``"best_trivial"`` (the
        better of the two). Returns ``value - baseline_value``: how much the mean
        outcome per unit rises under the learned rule compared with treating
        everyone / no one, in outcome units. The trivial constant
        policies are depth-0 leaves inside the tree's class, so the optimal tree is at
        least as good and this gain is non-negative up to sampling noise (it is the
        baseline's regret against the learned optimum, not the optimum's own regret,
        which would need the unknown oracle value).
        """
        if baseline == "treat_all":
            ref = self.treat_all_value
        elif baseline == "treat_none":
            ref = self.treat_none_value
        elif baseline == "best_trivial":
            ref = max(self.treat_all_value, self.treat_none_value)
        else:
            raise ValueError(
                f"baseline must be 'treat_all', 'treat_none', or 'best_trivial', got {baseline!r}"
            )
        return self.value - ref

    def evaluate(self, result_or_scores: CATEResult | DRScores) -> PolicyValue:
        """The learned rule's value on other units: the honest, out-of-sample value.

        ``result_or_scores`` carries doubly-robust scores for units the tree was not
        learned on (a :class:`~ergodic.inference.CATEResult` from a held-out sample,
        or a :class:`~ergodic.inference._scores.DRScores`); its covariates must
        include the tree's split covariates by name. Returns a :class:`PolicyValue`:
        the mean of ``pi(X_i) Gamma_i`` there (outcome units, against treating no
        one), its standard error, and the treat-all baseline on the same units.
        Because the rule was fixed before these scores were seen, the interval is
        valid, unlike the in-sample :attr:`value` the search maximized.
        """
        scores, design, names, _, _ = _extract(result_or_scores)
        index = {name: i for i, name in enumerate(names)}
        missing = [name for name in self.covariates if name not in index]
        if missing:
            raise ValueError(
                f"the evaluation scores lack the tree's split covariates {missing}; "
                f"they carry {list(names)}"
            )
        n = scores.shape[0]
        if n < 2:
            raise ValueError(f"evaluate needs at least 2 units, got {n}")
        x = design[:, [index[name] for name in self.covariates]]
        actions = self.predict(x).astype(np.float64)
        influence = actions * scores
        return PolicyValue(
            value=float(np.mean(influence)),
            std_error=float(np.std(influence, ddof=1) / np.sqrt(n)),
            treat_all_value=float(np.mean(scores)),
            n=n,
        )

    def leaves(self) -> list[PolicyNode]:
        """The leaf nodes, left to right (each carries its action, count, and value)."""
        out: list[PolicyNode] = []

        def walk(node: PolicyNode) -> None:
            if node.is_leaf:
                out.append(node)
                return
            assert node.left is not None and node.right is not None
            walk(node.left)
            walk(node.right)

        walk(self.root)
        return out

    def describe(self) -> str:
        """A readable ``if-else`` rendering of the tree (the interpretable rule)."""
        lines: list[str] = []

        def walk(node: PolicyNode, indent: str) -> None:
            if node.is_leaf:
                lines.append(f"{indent}-> treat={node.action} (n={node.n})")
                return
            feat = node.feature
            thr = node.threshold
            assert thr is not None
            lines.append(f"{indent}if {feat} <= {thr:.4g}:")
            assert node.left is not None and node.right is not None
            walk(node.left, indent + "  ")
            lines.append(f"{indent}else:  # {feat} > {thr:.4g}")
            walk(node.right, indent + "  ")

        walk(self.root, "")
        return "\n".join(lines)


# -- the search ----------------------------------------------------------------


def _leaf(scores: npt.NDArray[np.float64], n: int) -> tuple[float, PolicyNode]:
    """The best leaf for a set of scores: treat iff the score sum is positive."""
    total = float(np.sum(scores))
    action = 1 if total > 0.0 else 0
    value = abs(total)
    return value, PolicyNode(n=n, value=value, action=action)


def _reparent(node: PolicyNode, n: int, value: float) -> PolicyNode:
    """Copy a subtree's split/action onto a node with an updated count and value."""
    return PolicyNode(
        n=n,
        value=value,
        action=node.action,
        feature=node.feature,
        feature_index=node.feature_index,
        threshold=node.threshold,
        left=node.left,
        right=node.right,
    )


def _best_depth1(
    scores: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    names: tuple[str, ...],
) -> tuple[float, PolicyNode]:
    """The optimal depth-1 subtree (one split or a leaf), in ``O(p n log n)``.

    For each feature the threshold sweep is vectorized: with the scores in sorted
    order, splitting after the ``i``-th unit has welfare ``|prefix[i]| + |total -
    prefix[i]|``, evaluated for every boundary at once. Only boundaries between two
    distinct feature values are feasible. Returns the best split found, or the leaf
    when no split beats treating the whole subset as one.
    """
    n = scores.shape[0]
    best_value, best_node = _leaf(scores, n)
    if n <= 1:
        return best_value, best_node
    for j in range(design.shape[1]):
        column = design[:, j]
        order = np.argsort(column, kind="stable")
        col_sorted = column[order]
        prefix = np.cumsum(scores[order])
        total = float(prefix[-1])
        # boundary i (left = first i+1 units) is feasible iff the next value differs.
        feasible = col_sorted[:-1] != col_sorted[1:]
        if not np.any(feasible):
            continue
        left_sum = prefix[:-1]
        values = np.abs(left_sum) + np.abs(total - left_sum)
        values = np.where(feasible, values, -np.inf)
        i = int(np.argmax(values))
        split_value = float(values[i])
        if split_value > best_value:
            n_left = i + 1
            left_action = 1 if float(left_sum[i]) > 0.0 else 0
            right_action = 1 if (total - float(left_sum[i])) > 0.0 else 0
            left_val = abs(float(left_sum[i]))
            right_val = abs(total - float(left_sum[i]))
            best_value = split_value
            best_node = PolicyNode(
                n=n,
                value=split_value,
                feature=names[j],
                feature_index=j,
                threshold=float(col_sorted[i]),
                left=PolicyNode(n=n_left, value=left_val, action=left_action),
                right=PolicyNode(n=n - n_left, value=right_val, action=right_action),
            )
    return best_value, best_node


def _best_split(
    scores: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    names: tuple[str, ...],
    depth: int,
) -> tuple[float, PolicyNode]:
    """The globally optimal subtree on this subset: ``(welfare_sum, node)``.

    The base case (depth 0 or a constant subset) is a leaf whose welfare contribution
    is ``|sum Gamma|``; depth 1 is the vectorized :func:`_best_depth1`. At depth >= 2
    the root threshold sweep recurses optimally on both children, so the tree is
    globally optimal; the runtime is then ``O(p^2 n^2 log n)`` (the policytree growth,
    so a large ``n`` at depth 2 is costly). Used at ``depth <= 2``; the deeper greedy
    fall-through is :func:`_greedy_split`.
    """
    n = scores.shape[0]
    if depth <= 0 or n <= 1:
        return _leaf(scores, n)
    if depth == 1:
        return _best_depth1(scores, design, names)

    best_value, best_node = _leaf(scores, n)
    child_depth = depth - 1
    for j in range(design.shape[1]):
        column = design[:, j]
        order = np.argsort(column, kind="stable")
        col_sorted = column[order]
        boundaries = np.flatnonzero(col_sorted[:-1] != col_sorted[1:])
        for i in boundaries:
            threshold = float(col_sorted[i])
            left_mask = column <= threshold
            lv, lnode = _best_split(scores[left_mask], design[left_mask], names, child_depth)
            rv, rnode = _best_split(scores[~left_mask], design[~left_mask], names, child_depth)
            split_value = lv + rv
            if split_value > best_value:
                best_value = split_value
                best_node = PolicyNode(
                    n=n,
                    value=split_value,
                    feature=names[j],
                    feature_index=j,
                    threshold=threshold,
                    left=_reparent(lnode, int(left_mask.sum()), lv),
                    right=_reparent(rnode, int(n - left_mask.sum()), rv),
                )
    return best_value, best_node


def _greedy_split(
    scores: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    names: tuple[str, ...],
    depth: int,
) -> tuple[float, PolicyNode]:
    """The greedy depth-``depth`` subtree: best immediate split, then recurse.

    The deep fall-through (``depth > 2``). Picks the single welfare-best root split
    with :func:`_best_depth1`, then grows each child greedily to the remaining depth,
    so the tree genuinely reaches the requested depth in ``O(depth * p * n log n)``.
    Not guaranteed globally optimal (the documented deferral), but a leaf is always in
    the search, so a split is taken only when it does not lower welfare.
    """
    n = scores.shape[0]
    if depth <= 0 or n <= 1:
        return _leaf(scores, n)
    root_value, root = _best_depth1(scores, design, names)
    if root.is_leaf or depth == 1:
        return root_value, root
    assert root.feature_index is not None and root.threshold is not None
    column = design[:, root.feature_index]
    left_mask = column <= root.threshold
    lv, lnode = _greedy_split(scores[left_mask], design[left_mask], names, depth - 1)
    rv, rnode = _greedy_split(scores[~left_mask], design[~left_mask], names, depth - 1)
    value = lv + rv
    return value, PolicyNode(
        n=n,
        value=value,
        feature=root.feature,
        feature_index=root.feature_index,
        threshold=root.threshold,
        left=_reparent(lnode, int(left_mask.sum()), lv),
        right=_reparent(rnode, int(n - left_mask.sum()), rv),
    )


def _extract(
    result_or_scores: CATEResult | DRScores,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[str, ...], str, str]:
    """Pull the AIPW scores, the covariate design, and the names out of either input."""
    from .._perunit import CATEResult
    from .._scores import DRScores

    scores_obj: DRScores
    if isinstance(result_or_scores, DRScores):
        scores_obj = result_or_scores
    elif isinstance(result_or_scores, CATEResult):
        if result_or_scores._scores is None:
            raise ValueError(
                "this CATEResult carries no doubly-robust scores for policy learning; "
                "the cate() facade and the doubly-robust meta-learners cache them"
            )
        scores_obj = result_or_scores._scores
    else:
        raise TypeError(
            f"policy_tree needs a CATEResult or a DRScores, got {type(result_or_scores).__name__}"
        )
    return (
        np.asarray(scores_obj.scores, dtype=np.float64),
        np.asarray(scores_obj.covariates, dtype=np.float64),
        tuple(scores_obj.covariate_names),
        scores_obj.treatment if isinstance(scores_obj.treatment, str) else "",
        scores_obj.outcome if isinstance(scores_obj.outcome, str) else "",
    )


def _select_columns(
    design: npt.NDArray[np.float64],
    all_names: tuple[str, ...],
    covariates: str | tuple[str, ...] | list[str] | None,
) -> tuple[npt.NDArray[np.float64], tuple[str, ...]]:
    """Restrict the design to the requested split covariates by name."""
    if covariates is None:
        return design, all_names
    requested = (covariates,) if isinstance(covariates, str) else tuple(covariates)
    if not requested:
        raise ValueError("covariates is empty; pass at least one covariate name or None for all")
    index = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in requested if name not in index]
    if missing:
        raise ValueError(f"unknown split covariates {missing}; available: {list(all_names)}")
    cols = [index[name] for name in requested]
    return design[:, cols], requested


def policy_tree(
    result_or_scores: CATEResult | DRScores,
    covariates: str | tuple[str, ...] | list[str] | None = None,
    *,
    depth: int = 2,
) -> PolicyTree:
    """Fit a shallow doubly-robust optimal-policy tree of depth ``depth``.

    ``result_or_scores`` is a :class:`~ergodic.inference.CATEResult` (its cached
    cross-fit scores are read) or a :class:`~ergodic.inference._scores.DRScores`
    directly. ``covariates`` names the split covariates (a name, a list of names, or
    ``None`` for every covariate carried by the scores). The tree maximizes the
    doubly-robust welfare ``V(pi) = mean[pi(X) Gamma_i]`` (the mean outcome gain over
    treating no one, in outcome units); at ``depth <= 2``
    the search is exhaustive over thresholds and so globally optimal, and a deeper
    tree uses a greedy recursive best-split (documented, not guaranteed optimal).

    Returns a frozen :class:`PolicyTree` carrying the rule, the estimated policy
    value with a standard error from the per-unit influence terms, and the regret
    against treating everyone / no one.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    scores, full_design, all_names, treatment, outcome = _extract(result_or_scores)
    n = scores.shape[0]
    if n < 2:
        raise ValueError(f"policy_tree needs at least 2 units, got {n}")
    design, names = _select_columns(full_design, all_names, covariates)

    exhaustive = depth <= _EXHAUSTIVE_DEPTH
    if exhaustive:
        _, root = _best_split(scores, design, names, depth)
    else:
        _, root = _greedy_split(scores, design, names, depth)

    actions = np.asarray([root.predict(row) for row in design], dtype=np.float64)
    # The value against treating no one, in outcome units: mean[pi(X) Gamma]. (The
    # search's objective mean[(2 pi - 1) Gamma] is twice this minus the ATE.)
    influence = actions * scores
    value = float(np.mean(influence))
    std_error = float(np.std(influence, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    treat_all = float(np.mean(scores))
    treat_none = 0.0
    return PolicyTree(
        root=root,
        covariates=names,
        n=n,
        depth=depth,
        value=value,
        std_error=std_error,
        treat_all_value=treat_all,
        treat_none_value=treat_none,
        exhaustive=exhaustive,
        treatment=treatment,
        outcome=outcome,
        _scores=scores,
        _design=design,
    )
