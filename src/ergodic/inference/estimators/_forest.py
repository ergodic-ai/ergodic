# SPDX-License-Identifier: Apache-2.0
"""A native honest causal forest (GRF) for the CATE.

The generalized-random-forest estimator of Athey, Tibshirani, and Wager (2019),
built on the causal-forest inference theory of Wager and Athey (2018), in pure
numpy. The forest is the learner here: unlike the meta-learners, you cannot drop a
regressor into the CATE slot. The nuisance slots (the outcome mean and the
propensity) still take any learner; the forest itself is bespoke.

The construction, following the literature review's GRF section:

* Orthogonalize first (the forest R-learner). Cross-fit ``m(x) = E[Y | X = x]``
  and ``e(x) = E[W | X = x]`` and work on the residuals ``Y - m`` and ``W - e``, so
  the local moment is the partially-linear Robinson moment and confounding through
  ``X`` is removed before any split.
* Honest splitting. Each tree draws a subsample without replacement and partitions
  it into a SPLIT half (which places the splits) and an ESTIMATE half (which fills
  the leaves). A point's outcome is used for one job, never both, which is what
  makes the pointwise normal limit and the infinitesimal-jackknife variance valid.
* The split criterion is the GRF heterogeneity criterion: maximize the squared
  difference of the child node treatment-effect estimates, equivalently the
  labeled-pseudo-outcome sum-of-squares ``sum_j (sum_{i in C_j} rho_i)^2 / |C_j|``
  where ``rho_i`` is the influence-based pseudo-outcome of the parent node.
* The forest reads out a data-adaptive kernel ``alpha_i(x)`` (the average leaf
  co-membership across trees) and solves the within-neighborhood Robinson moment
  for ``tau(x)``, the closed-form local R-learner.
* Per-unit variance is the infinitesimal jackknife (the bootstrap-of-little-bags
  grouping), valid under honesty plus subsampling.

Binary treatment in this phase; a continuous treatment via the same local moment is
a documented deferral (it raises). The variable importance is a depth-weighted,
normalized split frequency, a driver signal exposed on the result and the estimator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._estimator import Slot, Task, _MetaLearnerBase
from .._scores import _crossfit_nuisances
from .._target import EffectTarget
from ..learners import LogisticRegression

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ...data import TabularData
    from .._learner import Learner
    from .._perunit import CATEResult


# -- the tree -----------------------------------------------------------------


@dataclass
class _Node:
    """One node of a causal tree.

    A leaf carries the estimate-half row indices that fell into it (``estimate``);
    an internal node carries a split (``feature``, ``threshold``) and its children.
    ``depth`` is the node's depth, used by the variable-importance weighting.
    """

    estimate: npt.NDArray[np.int64]
    depth: int
    feature: int = -1
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class _Tree:
    """A fitted honest causal tree: its root, and the subsample it was grown on.

    ``in_bag`` are the row indices of the subsample (split + estimate halves); the
    infinitesimal jackknife reads them as the per-tree inclusion counts.
    """

    root: _Node
    in_bag: npt.NDArray[np.int64]
    split_importance: npt.NDArray[np.float64]


def _node_effect(
    rows: npt.NDArray[np.int64],
    w_res: npt.NDArray[np.float64],
    y_res: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64]]:
    """The node treatment effect and per-row pseudo-outcomes on the residuals.

    On the residualized moment ``Y - m = (W - e) tau + eps`` the node effect is the
    Robinson slope ``sum (W~)(Y~) / sum (W~)^2`` over the node's rows, and the GRF
    pseudo-outcome is ``rho_i = (W~_i)(Y~_i - W~_i * theta) / (mean (W~)^2)`` (the
    influence function of that one-parameter moment). Splitting on the sum of squares
    of ``rho`` is the gradient approximation to re-fitting the effect in each child.
    A degenerate node (no treatment-residual variation) gets a zero effect and zero
    pseudo-outcomes, so it never drives a split.
    """
    wr = w_res[rows]
    yr = y_res[rows]
    denom = float(np.sum(wr * wr))
    if denom <= 1e-12:
        return 0.0, np.zeros(rows.shape[0], dtype=np.float64)
    theta = float(np.sum(wr * yr) / denom)
    var_w = denom / rows.shape[0]
    rho = wr * (yr - wr * theta) / var_w
    return theta, np.asarray(rho, dtype=np.float64)


class _TreeBuilder:
    """Grow one honest causal tree from a split half and place an estimate half."""

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        w_res: npt.NDArray[np.float64],
        y_res: npt.NDArray[np.float64],
        *,
        min_leaf: int,
        max_depth: int,
        n_features_try: int,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        self.x = x
        self.w_res = w_res
        self.y_res = y_res
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.n_features_try = n_features_try
        self.alpha = alpha
        self.rng = rng
        self.n_features = x.shape[1]
        self.importance = np.zeros(self.n_features, dtype=np.float64)

    def build(
        self, split_rows: npt.NDArray[np.int64], estimate_rows: npt.NDArray[np.int64]
    ) -> _Node:
        return self._grow(split_rows, estimate_rows, depth=0)

    def _grow(
        self,
        split_rows: npt.NDArray[np.int64],
        estimate_rows: npt.NDArray[np.int64],
        *,
        depth: int,
    ) -> _Node:
        node = _Node(estimate=estimate_rows, depth=depth)
        if (
            depth >= self.max_depth
            or split_rows.shape[0] < 2 * self.min_leaf
            or self.n_features == 0
        ):
            return node
        best = self._best_split(split_rows)
        if best is None:
            return node
        feature, threshold, left_split, right_split = best
        left_est = estimate_rows[self.x[estimate_rows, feature] <= threshold]
        right_est = estimate_rows[self.x[estimate_rows, feature] > threshold]
        # honest guard: every leaf must hold estimate-half rows in both arms-direction
        # so the local moment is solvable; otherwise do not take this split
        if left_est.shape[0] < self.min_leaf or right_est.shape[0] < self.min_leaf:
            return node
        self.importance[feature] += 1.0 / float(depth + 1)
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(left_split, left_est, depth=depth + 1)
        node.right = self._grow(right_split, right_est, depth=depth + 1)
        return node

    def _best_split(
        self, rows: npt.NDArray[np.int64]
    ) -> tuple[int, float, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
        _, rho = _node_effect(rows, self.w_res, self.y_res)
        if not np.any(rho):
            return None
        n = rows.shape[0]
        min_each = max(self.min_leaf, int(np.ceil(self.alpha * n)))
        if 2 * min_each > n:
            return None
        candidates = self._candidate_features()
        best_gain = 0.0
        best: tuple[int, float, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None = None
        for feature in candidates:
            values = self.x[rows, feature]
            order = np.argsort(values, kind="stable")
            sorted_vals = values[order]
            sorted_rho = rho[order]
            csum = np.cumsum(sorted_rho)
            total = csum[-1]
            # candidate cut after position k (k+1 rows go left), honoring the balance
            for k in range(min_each - 1, n - min_each):
                if sorted_vals[k] == sorted_vals[k + 1]:
                    continue
                n_left = k + 1
                left_sum = csum[k]
                right_sum = total - left_sum
                gain = (left_sum * left_sum) / n_left + (right_sum * right_sum) / (n - n_left)
                if gain > best_gain:
                    threshold = 0.5 * (sorted_vals[k] + sorted_vals[k + 1])
                    mask = values <= threshold
                    best_gain = gain
                    best = (feature, float(threshold), rows[mask], rows[~mask])
        return best

    def _candidate_features(self) -> npt.NDArray[np.int64]:
        if self.n_features_try >= self.n_features:
            return np.arange(self.n_features, dtype=np.int64)
        return np.asarray(
            self.rng.choice(self.n_features, size=self.n_features_try, replace=False),
            dtype=np.int64,
        )


def _leaf_rows(node: _Node, x_row: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Walk one query point to its leaf and return that leaf's estimate-half rows."""
    while not node.is_leaf:
        assert node.left is not None and node.right is not None
        node = node.left if x_row[node.feature] <= node.threshold else node.right
    return node.estimate


# -- the estimator ------------------------------------------------------------


class _ForestModel:
    """A fitted causal forest: the trees, the residuals, and the training design.

    The residuals ``w_res``/``y_res`` and the training covariates ``x_train`` close
    over the forest, so a query design ``z`` is scored by collecting forest-kernel
    weights against the training rows and solving the local Robinson moment.
    """

    def __init__(
        self,
        trees: list[_Tree],
        x_train: npt.NDArray[np.float64],
        w_res: npt.NDArray[np.float64],
        y_res: npt.NDArray[np.float64],
        importance: npt.NDArray[np.float64],
        covariate_names: tuple[str, ...],
    ) -> None:
        self.trees = trees
        self.x_train = x_train
        self.w_res = w_res
        self.y_res = y_res
        self.importance = importance
        self.covariate_names = covariate_names

    # -- the forest kernel and the local moment ------------------------------

    def weights(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The forest-kernel weight matrix ``alpha`` of shape ``(n_query, n_train)``.

        ``alpha[q, i]`` is the average over trees of ``1{i in leaf(z_q)} / |leaf|``,
        the GRF adaptive kernel. Only the estimate-half rows of a leaf carry weight,
        which is the honesty discipline.
        """
        n_query = z.shape[0]
        n_train = self.x_train.shape[0]
        alpha = np.zeros((n_query, n_train), dtype=np.float64)
        n_trees = len(self.trees)
        for tree in self.trees:
            for q in range(n_query):
                rows = _leaf_rows(tree.root, z[q])
                if rows.shape[0] == 0:
                    continue
                alpha[q, rows] += 1.0 / rows.shape[0]
        if n_trees:
            alpha /= n_trees
        return alpha

    def predict(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Per-unit CATE: solve the forest-weighted Robinson moment at each query row."""
        alpha = self.weights(z)
        return _solve_local_moment(alpha, self.w_res, self.y_res)

    def tree_predict(self, tree: _Tree, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """One tree's per-unit effect (its leaf Robinson slope), for the IJ variance."""
        n_query = z.shape[0]
        out = np.zeros(n_query, dtype=np.float64)
        for q in range(n_query):
            rows = _leaf_rows(tree.root, z[q])
            if rows.shape[0] == 0:
                continue
            out[q], _ = _node_effect(rows, self.w_res, self.y_res)
        return out


def _solve_local_moment(
    alpha: npt.NDArray[np.float64],
    w_res: npt.NDArray[np.float64],
    y_res: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The closed-form local R-learner over a kernel-weight matrix.

    ``tau(x) = sum_i a_i (W~_i - W_bar)(Y~_i - Y_bar) / sum_i a_i (W~_i - W_bar)^2``,
    with the forest-weighted local means of the residuals. Vectorized over the rows
    of ``alpha``. A query with no treatment-residual spread in its neighborhood gets
    a zero effect (the moment is unidentified there).
    """
    weight_total = alpha.sum(axis=1)
    safe = np.where(weight_total > 0, weight_total, 1.0)
    w_bar = (alpha @ w_res) / safe
    y_bar = (alpha @ y_res) / safe
    # centered cross-moment and second moment, both kernel-weighted
    cov = (alpha @ (w_res * y_res)) - weight_total * w_bar * y_bar
    var = (alpha @ (w_res * w_res)) - weight_total * w_bar * w_bar
    out = np.where(var > 1e-12, cov / np.where(var > 1e-12, var, 1.0), 0.0)
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class VariableImportance:
    """Depth-weighted, normalized split frequencies: a driver signal over covariates.

    ``names`` are the covariate columns and ``values`` the matching importances, which
    sum to one (or are all zero for a forest that never split). A split at depth ``d``
    contributes ``1 / (d + 1)``, so shallow splits, the ones that move the effect most,
    weigh more, following the GRF variable-importance convention.
    """

    names: tuple[str, ...]
    values: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))

    def top(self, k: int | None = None) -> list[tuple[str, float]]:
        """The covariates by descending importance (all of them when ``k`` is ``None``)."""
        order = np.argsort(self.values)[::-1]
        pairs = [(self.names[i], float(self.values[i])) for i in order]
        return pairs if k is None else pairs[:k]

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.values, strict=True)}


#: The share of zero-width per-unit bands above which ``cate`` warns.
_ZERO_SE_WARN = 0.05


class CausalForest(_MetaLearnerBase):
    """An honest causal forest (GRF) for the conditional average treatment effect.

    The forest is the CATE learner. The ``outcome`` and ``propensity`` nuisance slots
    take any learner (they residualize ``Y`` and ``W`` on the covariates before the
    forest grows); the forest machinery, honest splitting and the kernel readout, is
    native. The ATE is the average per-unit effect, and ``cate`` returns a
    :class:`~ergodic.inference.CATEResult` with per-unit infinitesimal-jackknife
    standard errors and intervals.

    Hyperparameters: ``n_trees`` (forest size), ``min_leaf`` (the minimum estimate-half
    rows per leaf), ``max_depth``, ``honest`` (split/estimate sample-splitting, on by
    default), ``subsample`` (the per-tree subsample fraction), and ``random_state``.
    Binary treatment only; a continuous treatment via the same local moment is not
    supported and raises.

    The per-unit intervals need many trees. The infinitesimal-jackknife variance
    subtracts a Monte-Carlo correction that shrinks only as ``n_trees`` grows, and
    where the correction exceeds the raw variance the standard error is clipped to
    zero (a zero-width band). Measured on a 4,000-row world with a linear effect
    (``min_leaf=15``, nominal 90%): 400 trees covered the true effect for 68% of
    units with 25% zero-width bands; 1,600 trees covered 87% with 1% zero-width.
    So use ``n_trees`` of the order of half the rows or more when the per-unit
    intervals matter; ``cate`` warns, and records
    ``diagnostics["ij_zero_se_share"]``, when more than 5% of the bands are
    zero-width. Coverage stays somewhat below nominal where the effect changes
    steeply (the forest's smoothing bias), so treat the bands as approximate.
    """

    name = "causal_forest"
    slots = (
        Slot("outcome", Task.REGRESSION),
        Slot("propensity", Task.CLASSIFICATION),
    )
    targets = frozenset({EffectTarget.ATE})

    def __init__(
        self,
        learners: Any = None,
        *,
        n_trees: int = 200,
        min_leaf: int = 5,
        max_depth: int = 8,
        honest: bool = True,
        subsample: float = 0.5,
        max_features: float | int | None = None,
        alpha: float = 0.05,
        n_folds: int = 5,
        clip: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(learners, **kwargs)
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if n_trees < 1:
            raise ValueError(f"n_trees must be at least 1, got {n_trees}")
        if min_leaf < 1:
            raise ValueError(f"min_leaf must be at least 1, got {min_leaf}")
        if not 0.0 <= alpha < 0.5:
            raise ValueError(f"alpha must be in [0, 0.5), got {alpha}")
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.honest = honest
        self.subsample = subsample
        self.max_features = max_features
        self.alpha = alpha
        self.n_folds = n_folds
        self.clip = clip

    # -- residualization (the forest R-learner front end) --------------------

    def _residualize(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Cross-fit ``m(x)`` and ``e(x)`` and return the residuals ``W - e``, ``Y - m``.

        Reuses the shared cross-fit nuisance routine: the outcome model carries the
        treatment as a feature (so ``m = e * mu1 + (1 - e) * mu0``), and the propensity
        is the treatment model. The residuals are what the forest splits on.
        """
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        mu0, mu1, e = _crossfit_nuisances(
            a,
            y,
            z,
            make_outcome=make["outcome"],
            make_propensity=make.get("propensity") or (lambda: LogisticRegression()),
            n_folds=self.n_folds,
            clip=self.clip,
            random_state=self._random_state,
        )
        m = e * mu1 + (1.0 - e) * mu0
        return a - e, y - m

    def _max_features_for(self, n_features: int) -> int:
        if self.max_features is None:
            # GRF default: ceil(sqrt(p) + 20), capped at p
            return min(n_features, int(np.ceil(np.sqrt(n_features) + 20)))
        if isinstance(self.max_features, int):
            return max(1, min(n_features, self.max_features))
        return max(1, min(n_features, int(np.ceil(self.max_features * n_features))))

    # -- the meta-learner hooks ----------------------------------------------

    def _fit_cate(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> _ForestModel:
        w_res, y_res = self._residualize(arrays, make)
        x = arrays["Z"]
        n, n_features = x.shape
        rng = np.random.default_rng(self._random_state)
        sub_n = max(2 * self.min_leaf, int(np.ceil(self.subsample * n)))
        sub_n = min(sub_n, n)
        n_try = self._max_features_for(n_features)
        trees: list[_Tree] = []
        importance = np.zeros(n_features, dtype=np.float64)
        for _ in range(self.n_trees):
            in_bag = np.asarray(rng.choice(n, size=sub_n, replace=False), dtype=np.int64)
            if self.honest and in_bag.shape[0] >= 2:
                shuffled = rng.permutation(in_bag)
                half = shuffled.shape[0] // 2
                split_rows = shuffled[:half]
                estimate_rows = shuffled[half:]
            else:
                split_rows = in_bag
                estimate_rows = in_bag
            builder = _TreeBuilder(
                x,
                w_res,
                y_res,
                min_leaf=self.min_leaf,
                max_depth=self.max_depth,
                n_features_try=n_try,
                alpha=self.alpha,
                rng=rng,
            )
            root = builder.build(split_rows, estimate_rows)
            trees.append(_Tree(root=root, in_bag=in_bag, split_importance=builder.importance))
            importance += builder.importance
        total = float(importance.sum())
        if total > 0:
            importance = importance / total
        return _ForestModel(
            trees=trees,
            x_train=x,
            w_res=w_res,
            y_res=y_res,
            importance=importance,
            covariate_names=self._covariates,
        )

    def _predict_cate(
        self, model: _ForestModel, z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return model.predict(z)

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        """The doubly-robust AIPW ATE with its influence-function standard error.

        GRF averages per-unit AIPW scores for a doubly-robust, efficient ATE rather
        than bootstrapping the forest. The score is
        ``Gamma_i = tau_hat(X_i) + (W_i - e_i) / (e_i (1 - e_i)) (Y_i - mu_hat_i)``
        with ``mu_hat_i = W_i mu1_i + (1 - W_i) mu0_i``; its mean is the ATE and its
        scaled standard deviation the standard error. Reusing this for ``effect()``
        keeps the scalar ATE path off the (prohibitively slow) forest-refit bootstrap.
        Computed only on the fit arrays (``_analytic`` on a resample returns ``None``).
        """
        if target is not EffectTarget.ATE:
            return None
        model = getattr(self, "_fitted_cate", None)
        if model is None or arrays["A"].shape[0] != model.x_train.shape[0]:
            return None
        a, y, z = arrays["A"], arrays["Y"], arrays["Z"]
        try:
            mu0, mu1, e = _crossfit_nuisances(
                a,
                y,
                z,
                make_outcome=make["outcome"],
                make_propensity=make.get("propensity") or (lambda: LogisticRegression()),
                n_folds=self.n_folds,
                clip=self.clip,
                random_state=self._random_state,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        tau_hat = model.predict(z)
        mu_hat = a * mu1 + (1.0 - a) * mu0
        gamma = tau_hat + (a - e) / (e * (1.0 - e)) * (y - mu_hat)
        return float(np.mean(gamma)), np.asarray(gamma, dtype=np.float64)

    # -- the per-unit result, with the infinitesimal-jackknife variance ------

    def cate(self, data: TabularData | None = None, *, uncertainty: bool = True) -> CATEResult:
        """Per-unit CATE with infinitesimal-jackknife standard errors and intervals.

        Overrides the base meta-learner ``cate`` (which would bootstrap-refit the
        whole forest). The point is the forest-kernel local effect; the per-unit
        variance is the infinitesimal jackknife over the trees (the
        bootstrap-of-little-bags grouping), valid because the forest is honest and
        subsampled. The result carries the cached doubly-robust scores for
        :meth:`~ergodic.inference.CATEResult.explain`.

        ``uncertainty=False`` returns the point effects with ``kind="point"`` and no
        standard error, skipping the infinitesimal-jackknife pass over the trees.
        """
        from .._perunit import CATEResult

        model = self._cate_model_forest()
        z = self._cate_covariates(data)
        point = np.asarray(model.predict(z), dtype=np.float64)
        n = point.shape[0]
        std_error = self._ij_standard_error(model, z, point) if uncertainty else None
        kind = "analytic" if uncertainty else "point"
        diagnostics = dict(self._state.diagnostics)
        if std_error is not None and std_error.size:
            zero_share = float(np.mean(std_error == 0.0))
            diagnostics["ij_zero_se_share"] = zero_share
            if zero_share > _ZERO_SE_WARN:
                warnings.warn(
                    f"{zero_share:.0%} of the per-unit standard errors are zero: with "
                    f"{len(model.trees)} trees for {model.x_train.shape[0]} rows the "
                    "infinitesimal-jackknife Monte-Carlo correction swamps the variance, "
                    "so those bands have zero width. Raise n_trees (of the order of half "
                    "the rows or more) before reading the per-unit intervals.",
                    UserWarning,
                    stacklevel=2,
                )
        scores = self._cate_scores(data)
        return CATEResult(
            treatment=self._treatment_name,
            outcome=self._outcome_name,
            covariates=self._covariates,
            n=n,
            kind=kind,
            source=self.name,
            diagnostics=diagnostics,
            identification=self._identification,
            _values=point,
            _index=self._cate_index(data, n),
            _std_error=std_error,
            _scores=scores,
        )

    def _cate_model_forest(self) -> _ForestModel:
        model = getattr(self, "_fitted_cate", None)
        if model is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit first")
        return model  # type: ignore[no-any-return]

    def _ij_standard_error(
        self,
        model: _ForestModel,
        z: npt.NDArray[np.float64],
        point: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Per-unit infinitesimal-jackknife standard errors over the trees.

        The IJ variance (Wager-Athey Eq. 8, the bootstrap-of-little-bags form)
        is ``V(x) = ((n - 1)/n)(n/(n - s))^2 sum_i Cov_b[t_b(x), N_ib]^2``, where
        ``t_b(x)`` is tree ``b``'s prediction at ``x``, ``N_ib`` is the inclusion count
        of training row ``i`` in tree ``b`` (here 0/1 under subsampling), and ``s`` is
        the subsample size. A Monte-Carlo correction subtracts the within-tree
        sampling noise so the estimate stays non-negative and is not inflated by a
        finite tree count. Honesty and subsampling are what make this valid.
        """
        n_trees = len(model.trees)
        n_query = z.shape[0]
        n_train = model.x_train.shape[0]
        if n_trees < 2 or n_train < 2:
            return np.zeros(n_query, dtype=np.float64)
        # per-tree predictions, (B, n_query)
        preds = np.empty((n_trees, n_query), dtype=np.float64)
        inbag = np.zeros((n_trees, n_train), dtype=np.float64)
        for b, tree in enumerate(model.trees):
            preds[b] = model.tree_predict(tree, z)
            inbag[b, tree.in_bag] = 1.0
        s = float(inbag.sum(axis=1).mean())
        pred_centered = preds - preds.mean(axis=0, keepdims=True)
        inbag_centered = inbag - inbag.mean(axis=0, keepdims=True)
        # Cov_b[t_b(x), N_ib] over trees, shape (n_query, n_train)
        cov = (pred_centered.T @ inbag_centered) / n_trees
        raw = np.sum(cov * cov, axis=1)
        # bootstrap-of-little-bags Monte-Carlo bias correction
        pred_var = pred_centered.var(axis=0)  # (n_query,)
        inbag_var = float(inbag_centered.var())
        correction = n_train * inbag_var * pred_var / n_trees
        var_unscaled = np.clip(raw - correction, 0.0, np.inf)
        if s >= n_train:
            scale = (n_train - 1.0) / n_train
        else:
            scale = ((n_train - 1.0) / n_train) * (n_train / (n_train - s)) ** 2
        var = scale * var_unscaled
        return np.asarray(np.sqrt(np.clip(var, 0.0, np.inf)), dtype=np.float64)

    # -- variable importance -------------------------------------------------

    def variable_importance(self) -> VariableImportance:
        """Depth-weighted, normalized split frequencies over the covariates.

        A driver signal: which covariates the forest splits on, weighted toward the
        shallow splits that move the effect most. Available after ``fit``.
        """
        model = self._cate_model_forest()
        return VariableImportance(names=model.covariate_names, values=model.importance)
