# SPDX-License-Identifier: Apache-2.0
"""DirectLiNGAM (Shimizu et al., 2011): a causal order for linear non-Gaussian data.

The model is ``x = Bx + e`` with ``B`` strictly lower triangular in some
variable order and the noises independent and non-Gaussian, and that
non-Gaussianity is what buys identifiability: the direction of every edge is
decided by the data, not just the skeleton, so the output is a single DAG
rather than an equivalence class. On Gaussian data the model is not
identifiable, the fit still runs, and the order it returns is arbitrary; the
directions mean nothing there.

The order is found one root at a time with the pairwise likelihood ratio of
Hyvarinen and Smith (2013): ``R(i -> j)`` compares the entropies of each
standardized variable and of the standardized residual of regressing one on
the other, every entropy taken with Hyvarinen's maxent approximation, and a
positive value favors ``i`` causing ``j``. The next root is the variable the
measure speaks against least, ``M(i) = -sum_j min(0, R(i -> j))^2``; the
chosen root is regressed out of the remaining variables and the step repeats
on the residuals.

Pruning diverges from the reference ``lingam`` package on purpose: that
package fits an adaptive lasso over ``LassoLarsIC``, while here each variable
is regressed on its predecessors in the order and backward-eliminated by BIC,
which is deterministic and dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ...data import TabularData
from ...graph import dag
from .._result import DiscoveryResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ...knowledge import DomainKnowledge

# The maxent approximation constants (Hyvarinen, 1998), exactly as Hyvarinen
# and Smith (2013) use them in the pairwise measure.
_K1 = 79.047
_K2 = 7.4129
_GAMMA = 0.37457


def _entropy(u: npt.NDArray[np.float64]) -> float:
    """The maxent entropy approximation of a standardized sample.

    ``H(u) = (1 + log 2 pi) / 2 - k1 (E[log cosh u] - gamma)^2
    - k2 (E[u exp(-u^2 / 2)])^2``. The log-cosh runs in its overflow-free
    form, ``|u| + log1p(exp(-2|u|)) - log 2``.
    """
    a = np.abs(u)
    log_cosh = float(np.mean(a + np.log1p(np.exp(-2.0 * a)))) - float(np.log(2.0))
    skew_like = float(np.mean(u * np.exp(-0.5 * u * u)))
    base = 0.5 * (1.0 + float(np.log(2.0 * np.pi)))
    return base - _K1 * (log_cosh - _GAMMA) ** 2 - _K2 * skew_like**2


def _standardized(matrix: npt.NDArray[np.float64], names: Sequence[str]) -> npt.NDArray[np.float64]:
    """Columns scaled to zero mean and unit variance, raising on a constant one."""
    centered = matrix - matrix.mean(axis=0)
    scale = centered.std(axis=0)
    flat = [names[int(k)] for k in np.flatnonzero(scale == 0.0)]
    if flat:
        raise ValueError(
            f"column(s) {flat} have zero variance (in the data, or after residualizing "
            "on earlier roots); DirectLiNGAM needs varying, linearly independent columns"
        )
    return np.asarray(centered / scale, dtype=np.float64)


def _root_scores(z: npt.NDArray[np.float64], names: Sequence[str]) -> npt.NDArray[np.float64]:
    """The aggregate measure ``M(i) = -sum_j min(0, R(i -> j))^2`` per column.

    ``R(i -> j) = H(x_j) + H(r_i_j) - H(x_i) - H(r_j_i)`` over the standardized
    columns of ``z``, where ``r_a_b`` is the residual of ``x_a`` on ``x_b``,
    standardized before its entropy is taken. This is the pairwise likelihood
    ratio: under the true direction ``i -> j``, the effect ``x_j`` and the
    backward residual ``r_i_j`` are the Gaussian-leaning mixtures (high
    entropy) while the cause ``x_i`` and the noise ``r_j_i`` are the
    non-Gaussian sides (low entropy), so a positive value favors ``i``
    causing ``j`` and the best root is the variable nothing speaks against.
    """
    n, m = z.shape
    corr = (z.T @ z) / float(n)
    own = [_entropy(z[:, k]) for k in range(m)]
    residual = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            r = z[:, j] - corr[i, j] * z[:, i]
            sd = float(r.std())
            if sd == 0.0:
                raise ValueError(
                    f"columns {names[i]!r} and {names[j]!r} are perfectly collinear; "
                    "DirectLiNGAM cannot order an exact linear copy"
                )
            residual[i, j] = _entropy(r / sd)
    scores = np.zeros(m, dtype=np.float64)
    for i in range(m):
        total = 0.0
        for j in range(m):
            if i == j:
                continue
            # residual[a, b] holds H(r_b_a), the entropy of b residualized on a.
            ratio = own[j] + residual[j, i] - own[i] - residual[i, j]
            total += min(0.0, ratio) ** 2
        scores[i] = -total
    return scores


def _root_candidates(remaining: list[str], knowledge: DomainKnowledge | None) -> list[str]:
    """The remaining variables admissible as the next root, in name order.

    Tiers gate first: while some remaining variable strictly precedes another,
    only the unpreceded ones (which includes every untiered variable, since
    nothing precedes those) may go next. This loses nothing: every
    tier-consistent DAG has a tier-sorted topological order, so confining the
    search to tier-sorted orders still reaches every admissible graph. Within
    that pool, declared sinks wait while a non-sink remains, and declared
    roots take priority while one is present.
    """
    if knowledge is None:
        return remaining
    present = frozenset(remaining)
    closure = knowledge.precedence()
    pool = [v for v in remaining if not any(u != v and (u, v) in closure for u in present)]
    if not pool:
        raise ValueError(
            "no admissible root: every remaining variable is preceded by another, so "
            "the precedence constraints contain a cycle (knowledge.validate() reports it)"
        )
    non_sinks = [v for v in pool if v not in knowledge.sinks]
    if non_sinks:
        pool = non_sinks
    rooted = [v for v in pool if v in knowledge.roots]
    if rooted:
        pool = rooted
    return pool


def _causal_order(
    matrix: npt.NDArray[np.float64],
    names: list[str],
    knowledge: DomainKnowledge | None,
) -> list[str]:
    """The order, one most-exogenous variable at a time.

    Each round standardizes the running residual matrix, scores every
    remaining variable with the pairwise measure, appends the best admissible
    root (exact ties go to the earliest name, since only a strictly larger
    score displaces the incumbent), and replaces every other variable by its
    OLS residual on that root.
    """
    order: list[str] = []
    remaining = list(names)
    working = np.array(matrix, dtype=np.float64)
    while len(remaining) > 1:
        z = _standardized(working, remaining)
        scores = _root_scores(z, remaining)
        candidates = _root_candidates(remaining, knowledge)
        best = candidates[0]
        best_score = float(scores[remaining.index(best)])
        for v in candidates[1:]:
            value = float(scores[remaining.index(v)])
            if value > best_score:
                best, best_score = v, value
        k = remaining.index(best)
        root = working[:, k] - working[:, k].mean()
        denom = float(root @ root)
        keep = [i for i in range(len(remaining)) if i != k]
        for i in keep:
            col = working[:, i]
            beta = float(root @ (col - col.mean())) / denom
            working[:, i] = col - col.mean() - beta * root
        working = working[:, keep]
        order.append(best)
        remaining.pop(k)
    order.extend(remaining)
    return order


def _check_required_order(order: list[str], knowledge: DomainKnowledge | None) -> None:
    """Raise when a required edge points against the data-driven order."""
    if knowledge is None:
        return
    position = {v: k for k, v in enumerate(order)}
    for cause, effect in sorted(knowledge.required):
        if cause in position and effect in position and position[effect] < position[cause]:
            raise ValueError(
                f"required edge {cause} -> {effect} contradicts the causal order the data "
                f"picked ({effect} comes out more exogenous than {cause}); the order is "
                "data-driven, so knowledge that contradicts it surfaces instead of bending it"
            )


def _ols_bic(y: npt.NDArray[np.float64], X: npt.NDArray[np.float64]) -> float:
    """Gaussian BIC of an OLS fit with intercept: ``n log(RSS / n) + k log n``.

    ``k`` counts the predictors; the always-present intercept shifts every
    candidate model equally, so leaving it out of ``k`` changes no comparison.
    """
    n = y.shape[0]
    design = np.column_stack([np.ones((n, 1), dtype=np.float64), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    rss = max(float(resid @ resid), float(np.finfo(np.float64).tiny))
    return float(n * np.log(rss / n) + X.shape[1] * np.log(n))


def _backward_bic(
    matrix: npt.NDArray[np.float64],
    column: dict[str, int],
    target: str,
    predecessors: list[str],
    required: frozenset[str],
) -> tuple[list[str], int]:
    """Backward elimination by BIC on the original data.

    Repeatedly drop the predecessor whose removal improves BIC the most (a
    strict improvement, swept in name order, so exact ties drop the earliest
    name) and stop when no removal improves. A required predecessor is never
    a removal candidate. Returns the survivors and the number dropped.
    """
    y = matrix[:, column[target]]

    def bic_of(chosen: list[str]) -> float:
        return _ols_bic(y, matrix[:, [column[p] for p in chosen]])

    current = sorted(predecessors)
    best = bic_of(current)
    dropped = 0
    while True:
        choice: str | None = None
        choice_bic = best
        for p in current:
            if p in required:
                continue
            trial = bic_of([q for q in current if q != p])
            if trial < choice_bic:
                choice, choice_bic = p, trial
        if choice is None:
            return current, dropped
        current = [q for q in current if q != choice]
        best = choice_bic
        dropped += 1


class DirectLiNGAM:
    """DirectLiNGAM over continuous tabular data.

    Args:
        prune: ``"bic"`` (the default) backward-eliminates each variable's
            predecessors by Gaussian BIC on the original data; ``"none"``
            keeps every predecessor, the full lower-triangular DAG of the
            found order.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    is a DAG: with independent non-Gaussian noises the whole orientation is
    identified, so there is no equivalence class to report. The class
    consumes neither a CI test nor a decomposable score (``ingredient`` is
    ``None``); the data is its only ingredient.
    """

    ingredient: ClassVar[str | None] = None

    def __init__(self, prune: str = "bic") -> None:
        if prune not in ("bic", "none"):
            raise ValueError(f"prune must be 'bic' or 'none', got {prune!r}")
        self.prune = prune

    # -- the run ---------------------------------------------------------

    def fit(self, data: TabularData, knowledge: DomainKnowledge | None = None) -> DiscoveryResult:
        """Find the causal order, prune the parents, and return the DAG."""
        if not isinstance(data, TabularData):
            raise TypeError(
                f"DirectLiNGAM reads TabularData, got {type(data).__name__}; temporal "
                "shapes have their own methods (method='pcmci' for time series and panels)"
            )
        complete = data.dropna()
        names = sorted(complete.variables)
        wrong = [v for v in names if not complete.schema[v].is_continuous]
        if wrong:
            raise ValueError(
                f"column(s) {wrong} are not continuous; the LiNGAM model is linear in "
                "continuous variables, so retype them or mark them excluded"
            )
        if complete.n_rows < 2:
            raise ValueError(f"DirectLiNGAM needs at least 2 complete rows, got {complete.n_rows}")
        matrix = complete.matrix(names)
        order = _causal_order(matrix, names, knowledge)
        _check_required_order(order, knowledge)
        parents, n_pruned, n_knowledge_pruned = self._parents_of(matrix, names, order, knowledge)
        graph = dag(
            [(p, child) for child in order for p in parents[child]],
            nodes=names,
        )
        if knowledge is not None and not knowledge.is_consistent(graph):
            raise ValueError(
                "DirectLiNGAM produced a graph that violates the domain knowledge; the "
                "data and the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics = {
            "n_root_picks": float(len(order)),
            "n_pruned_edges": float(n_pruned),
            "n_knowledge_pruned": float(n_knowledge_pruned),
        }
        return DiscoveryResult(
            graph=graph,
            method="lingam",
            test=None,
            score=None,
            alpha=None,
            n=complete.n_rows,
            diagnostics=diagnostics,
        )

    # -- pruning -----------------------------------------------------------

    def _parents_of(
        self,
        matrix: npt.NDArray[np.float64],
        names: list[str],
        order: list[str],
        knowledge: DomainKnowledge | None,
    ) -> tuple[dict[str, list[str]], int, int]:
        """Each variable's parents among its order predecessors, with counters.

        Knowledge filters the predecessor set first: a forbidden ``a -> b``
        never enters ``b``'s regression (counted in ``n_knowledge_pruned``)
        and a required edge can never leave it. ``prune="bic"`` then
        backward-eliminates; ``prune="none"`` keeps every survivor.
        """
        column = {v: k for k, v in enumerate(names)}
        parents: dict[str, list[str]] = {}
        n_pruned = 0
        n_knowledge_pruned = 0
        for t, target in enumerate(order):
            kept: list[str] = []
            required: set[str] = set()
            for p in order[:t]:
                if knowledge is not None and knowledge.requires_edge(p, target):
                    kept.append(p)
                    required.add(p)
                elif knowledge is None or knowledge.permits_edge(p, target):
                    kept.append(p)
                else:
                    n_knowledge_pruned += 1
            if self.prune == "bic":
                kept, dropped = _backward_bic(matrix, column, target, kept, frozenset(required))
                n_pruned += dropped
            parents[target] = sorted(kept)
        return parents, n_pruned, n_knowledge_pruned
