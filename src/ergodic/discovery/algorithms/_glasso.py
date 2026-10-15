# SPDX-License-Identifier: Apache-2.0
"""Graphical lasso: the sparse Gaussian precision matrix as a skeleton screen.

The graphical lasso (Friedman, Hastie, and Tibshirani 2008, "Sparse inverse
covariance estimation with the graphical lasso", Biostatistics 9(3)) puts an
l1 penalty on the precision matrix of a Gaussian model and maximizes the
penalized likelihood by block coordinate descent: cycle over the columns of
the working covariance ``W``, solve a lasso for each column against the rest,
write the solution back, and read the precision off at the end. A zero in the
precision is a conditional independence under Gaussianity, so the nonzero
pattern is an undirected conditional-independence skeleton. The penalty is
picked by the extended BIC over a warm-started descending path (Foygel and
Drton 2010, "Extended Bayesian information criteria for Gaussian graphical
models", NeurIPS 23), each candidate support scored at its restricted maximum
likelihood estimate, the way the criterion is defined; scoring the shrunk
path estimate instead biases the pick dense as ``n`` grows.

Two conventions worth naming. The screen works on the correlation matrix, not
the covariance: the deliverable is a graph, not a scale, and the penalty
should not weigh variables by their units. And the diagonal of the precision
is unpenalized, the scikit-learn and qgraph convention, which pins the
diagonal of ``W`` at the diagonal of the correlation; the 2008 paper
penalizes the diagonal too (there ``w_jj = s_jj + lam``), so published
solutions can differ by exactly that choice at the same lambda.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ...data import TabularData
from ...graph import Edge, GraphKind, Mark, MixedGraph
from ...knowledge import _refuse_soft
from .._result import DiscoveryResult

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...knowledge import DomainKnowledge

# An off-diagonal precision entry above this magnitude is an edge. Lasso zeros
# are exact zeros, so the threshold only has to clear arithmetic noise.
_EDGE_EPS = 1e-8

# Cap on inner coordinate-descent sweeps per column solve. The cap never hides
# a failure: an unsolved column keeps the working covariance moving, so the
# outer criterion is the one that raises.
_INNER_SWEEPS = 1000


def _default_factor(p: int) -> npt.NDArray[np.float64]:
    """Penalty multipliers: 1 everywhere off the diagonal."""
    return np.ones((p, p), dtype=np.float64)


def _default_free(p: int) -> npt.NDArray[np.bool_]:
    """Updatable coordinates: every off-diagonal entry."""
    return ~np.eye(p, dtype=bool)


def _initial_state(
    s: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The load-bearing start: off-diagonals of ``S`` shrunk by 0.95, diagonal kept.

    Mazumder and Hastie (2012, "The graphical lasso: new insights and
    alternatives") show arbitrary starts can fail to converge; the shrink
    keeps the working covariance positive definite (eigenvalues move toward
    the positive diagonal), so every column subproblem stays well posed. This
    is scikit-learn's choice given the unpenalized diagonal, matched here so
    the two solvers agree numerically. The betas start at zero.
    """
    w = 0.95 * s
    np.fill_diagonal(w, np.diag(s))
    return w, np.zeros_like(s)


def _lasso_gram(
    w11: npt.NDArray[np.float64],
    s12: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    pen: npt.NDArray[np.float64],
    active: npt.NDArray[np.bool_],
    inner_tol: float,
) -> None:
    """Coordinate descent on ``(1/2) b'W11 b - s12'b + sum_k pen_k |b_k|``, in place.

    The update is the soft threshold on the partial residual,
    ``b_k <- soft(s12_k - sum_{l != k} W11_kl b_l, pen_k) / W11_kk``, swept
    until the largest coefficient change drops under ``inner_tol``.
    Coordinates outside ``active`` are hard zeros and never update.
    """
    beta[~active] = 0.0
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return
    for _ in range(_INNER_SWEEPS):
        worst = 0.0
        for k in idx:
            r = float(s12[k]) - float(w11[k] @ beta) + float(w11[k, k]) * float(beta[k])
            slack = abs(r) - float(pen[k])
            new = math.copysign(slack, r) / float(w11[k, k]) if slack > 0.0 else 0.0
            step = abs(new - float(beta[k]))
            if step > worst:
                worst = step
            beta[k] = new
        if worst <= inner_tol:
            return


def _fit_at(
    s: npt.NDArray[np.float64],
    lam: float,
    *,
    factor: npt.NDArray[np.float64],
    free: npt.NDArray[np.bool_],
    w: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> int:
    """Block coordinate descent at one penalty; ``w`` and ``b`` update in place.

    Convergence is FHT's criterion: the mean absolute change of ``W`` over a
    full cycle, against ``tol`` times the mean absolute off-diagonal of
    ``S``. Past ``max_iter`` cycles it raises rather than return an
    unconverged answer. The primal objective is deliberately not used as a
    progress check: the iteration is dual ascent and the primal need not
    decrease monotonically (Mazumder and Hastie 2012).
    """
    p = s.shape[0]
    off = ~np.eye(p, dtype=bool)
    target = tol * float(np.mean(np.abs(s[off])))
    inner_tol = max(1e-14, 0.01 * target)
    columns = [np.flatnonzero(np.arange(p) != j) for j in range(p)]
    for cycle in range(1, max_iter + 1):
        before = w[off].copy()
        for j in range(p):
            others = columns[j]
            w11 = w[np.ix_(others, others)]
            beta = b[others, j].copy()
            pen = lam * factor[others, j]
            _lasso_gram(w11, s[others, j], beta, pen, free[others, j], inner_tol)
            b[others, j] = beta
            w12 = w11 @ beta
            w[others, j] = w12
            w[j, others] = w12
        if float(np.mean(np.abs(w[off] - before))) <= target:
            return cycle
    raise RuntimeError(
        f"graphical lasso did not converge in max_iter={max_iter} cycles at lam={lam:g}; "
        "raise max_iter or loosen tol rather than trust a half-converged precision"
    )


def _precision_of(
    w: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Recover the precision column by column from the converged ``(W, beta)``.

    ``theta_jj = 1 / (w_jj - w12'beta_j)`` and ``theta_12 = -beta_j * theta_jj``,
    then symmetrize. A non-positive denominator means the working covariance
    lost positive definiteness, which the initialization rules out, so it is
    a bug signal and raises.
    """
    p = w.shape[0]
    theta = np.zeros_like(w)
    for j in range(p):
        others = np.flatnonzero(np.arange(p) != j)
        beta = b[others, j]
        denom = float(w[j, j]) - float(w[others, j] @ beta)
        if denom <= 0.0:
            raise FloatingPointError(
                f"non-positive partial variance at column {j} (w_jj - w12'beta = {denom:.3e}); "
                "the working covariance lost positive definiteness, which the shrunk "
                "initialization rules out, so this signals a bug or a degenerate input"
            )
        theta[j, j] = 1.0 / denom
        theta[others, j] = -beta / denom
    return np.asarray((theta + theta.T) / 2.0, dtype=np.float64)


def _penalized_precision(
    s: npt.NDArray[np.float64],
    lam: float,
    *,
    factor: npt.NDArray[np.float64] | None = None,
    free: npt.NDArray[np.bool_] | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    state: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], int]:
    """One graphical-lasso solve at a fixed penalty on a correlation matrix.

    Returns ``(theta, (w, b), n_iter)``. ``state`` warm-starts from a previous
    solve (the path carries it down); ``None`` starts from the shrunk
    initialization. This is the numeric core the tests cross-check against
    ``sklearn.covariance.graphical_lasso``.
    """
    p = s.shape[0]
    factor = _default_factor(p) if factor is None else factor
    free = _default_free(p) if free is None else free
    w, b = _initial_state(s) if state is None else state
    n_iter = _fit_at(s, lam, factor=factor, free=free, w=w, b=b, tol=tol, max_iter=max_iter)
    return _precision_of(w, b), (w, b), n_iter


def _support(theta: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """The strict upper-triangle edge indicators at the |theta| > 1e-8 threshold."""
    iu = np.triu_indices(theta.shape[0], 1)
    return np.asarray(np.abs(theta[iu]) > _EDGE_EPS, dtype=np.bool_)


def _ebic(
    theta: npt.NDArray[np.float64], s: npt.NDArray[np.float64], n: int, gamma: float
) -> float:
    """Foygel-Drton extended BIC: ``-2 l_n + |E| log n + 4 |E| gamma log p``.

    ``l_n = (n/2)(log det theta - tr(S theta))`` and ``|E|`` counts nonzero
    off-diagonal pairs, never the diagonal.
    """
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0.0:
        raise FloatingPointError("the precision matrix is not positive definite")
    loglik = 0.5 * n * (float(logdet) - float(np.sum(s * theta)))
    p = s.shape[0]
    n_pairs = int(np.count_nonzero(_support(theta)))
    return -2.0 * loglik + n_pairs * math.log(n) + 4.0 * gamma * n_pairs * math.log(p)


def _diagonal_solution(
    s: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], int]:
    """The empty-graph endpoint: ``W = diag(S)``, ``theta = diag(1/s_jj)``, no cycles."""
    w = np.diag(np.diag(s)).astype(np.float64)
    theta = np.diag(1.0 / np.diag(s)).astype(np.float64)
    return theta, (w, np.zeros_like(s)), 0


def _restricted_mle(
    s: npt.NDArray[np.float64],
    support: npt.NDArray[np.bool_],
    *,
    tol: float,
    max_iter: int,
) -> npt.NDArray[np.float64]:
    """The unpenalized Gaussian MLE with the precision restricted to ``support``.

    This is Dempster's covariance selection, solved by the same block
    coordinate descent at zero penalty with the off-support coordinates
    excluded. It scores the EBIC: Foygel and Drton define the criterion at
    the maximum likelihood estimate of the model ``E``, not at the shrunk
    path estimate, whose penalization bias grows with ``n`` and drags the
    pick toward the dense end of the path.
    """
    p = s.shape[0]
    w, b = _initial_state(s)
    _fit_at(s, 0.0, factor=_default_factor(p), free=support, w=w, b=b, tol=tol, max_iter=max_iter)
    return _precision_of(w, b)


def _ebic_path(
    s: npt.NDArray[np.float64],
    *,
    n: int,
    factor: npt.NDArray[np.float64],
    free: npt.NDArray[np.bool_],
    gamma: float,
    n_lambdas: int,
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray[np.float64], float, float, int, int]:
    """The warm-started descending path; returns the EBIC minimizer.

    ``lambda_max = max |off-diagonal of S|`` (the empty graph, short-circuited
    to the diagonal solution unless a required pair is unpenalized there) down
    to one percent of it, log-spaced. Each support met on the path is scored
    once by EBIC at its restricted MLE (see :func:`_restricted_mle`); a
    support whose MLE does not exist or converge cannot be scored and cannot
    win. Ties go to the first minimum reached, which on a descending path is
    the sparser, larger lambda. Returns ``(theta, lam, ebic, n_iter,
    n_lambdas_tried)`` with ``theta`` the penalized solve at the winning
    lambda, whose nonzero pattern is the chosen support.
    """
    p = s.shape[0]
    off = np.abs(s[_default_free(p)])
    lam_max = float(off.max())
    if lam_max <= 0.0:
        # Every off-diagonal correlation is exactly zero: the graph is empty
        # at any penalty, so the path collapses to its one answer.
        theta, _, n_iter = _diagonal_solution(s)
        return theta, 0.0, _ebic(theta, s, n, gamma), n_iter, 1
    has_unpenalized = bool(np.any((factor == 0.0) & free))
    lams = np.geomspace(lam_max, 0.01 * lam_max, n_lambdas)
    state = _initial_state(s)
    iu = np.triu_indices(p, 1)
    scores: dict[bytes, float] = {}
    best: tuple[float, npt.NDArray[np.float64], float, int] | None = None
    for i, lam in enumerate(lams):
        if i == 0 and not has_unpenalized:
            # At lambda_max every lasso soft-thresholds to zero, so the
            # endpoint is the diagonal solution exactly; skip the cycles.
            theta, state, n_iter = _diagonal_solution(s)
        else:
            theta, state, n_iter = _penalized_precision(
                s, float(lam), factor=factor, free=free, tol=tol, max_iter=max_iter, state=state
            )
        sup = _support(theta)
        key = sup.tobytes()
        if key not in scores:
            if not bool(sup.any()):
                scores[key] = _ebic(_diagonal_solution(s)[0], s, n, gamma)
            else:
                mask = np.zeros((p, p), dtype=bool)
                mask[iu] = sup
                mask |= mask.T
                try:
                    refit = _restricted_mle(s, mask, tol=tol, max_iter=max_iter)
                    scores[key] = _ebic(refit, s, n, gamma)
                except (RuntimeError, FloatingPointError):
                    # The restricted MLE does not exist (for example a dense
                    # support with too few rows): this candidate cannot be
                    # scored, so it cannot be picked.
                    scores[key] = math.inf
        crit = scores[key]
        if math.isfinite(crit) and (best is None or crit < best[0]):
            best = (crit, theta, float(lam), n_iter)
    if best is None:
        raise RuntimeError(
            "no lambda on the EBIC path could be scored: every support's restricted MLE "
            "failed to exist or converge; the data has too few rows for the models on "
            "the path, so pass an explicit lam instead"
        )
    chosen_crit, chosen_theta, chosen_lam, chosen_iter = best
    return chosen_theta, chosen_lam, chosen_crit, chosen_iter, int(lams.size)


class Glasso:
    """The graphical-lasso screen: an undirected skeleton from a sparse precision.

    Like MMPC, this is a cheap screen to run before an expensive search:
    ``DomainKnowledge().restrict_to(result.graph)`` turns the recovered
    support into a constraint that PC, GES, or AStar then searches inside,
    with far fewer tests or score calls. The honesty note: a precision zero
    is a conditional independence under Gaussianity, so the graph is a
    skeleton source, not causal structure; it says which adjacencies the
    data supports, never which way they point.

    The solve is FHT 2008 block coordinate descent on the correlation matrix
    (the graph, not the scale, is the deliverable), with the diagonal
    unpenalized (the scikit-learn convention; the paper's differs, see the
    module docstring) and the shrunk initialization that keeps every column
    subproblem positive definite.

    Args:
        lam: The penalty. ``None`` (the default) runs the EBIC path:
            ``n_lambdas`` log-spaced values from ``max |off-diagonal
            correlation|`` (the empty graph) down to one percent of it,
            warm-started descending. Each support met on the path is scored
            by EBIC at its restricted maximum likelihood estimate (the
            Foygel-Drton definition; the shrunk path estimate would bias the
            pick dense as ``n`` grows), and the minimizer wins, ties to the
            sparser, larger lambda. An explicit value runs single-shot;
            ``0.0`` asks for the unpenalized MLE and raises when ``p >= n``.
        gamma: The EBIC sparsity weight (Foygel and Drton 2010), ``0.5`` by
            default; ``0.0`` recovers the plain BIC.
        n_lambdas: Path length when ``lam`` is ``None``.
        tol: Outer convergence, FHT's criterion: the mean absolute change of
            the working covariance over a full cycle must drop under ``tol``
            times the mean absolute off-diagonal correlation.
        max_iter: Cap on outer cycles per solve; past it the fit raises
            ``RuntimeError`` rather than return unconverged in silence.

    Knowledge enters through the penalty, and only its adjacency content
    binds, because the screen is undirected. A pair forbidden in both
    directions (an adjacency ban, including pairs outside a ``restrict_to``
    support) is a hard zero, excluded from the updates entirely. A required
    pair (either direction) is unpenalized, so it survives every lambda the
    data gives it any conditional association at. Directional knowledge
    (tiers, one-way forbids, roots, sinks) does not constrain an undirected
    adjacency and is left to the search that runs inside the support.

    ``fit(data, knowledge)`` needs a :class:`TabularData` whose variables are
    all continuous (a Gaussian model) and returns a :class:`DiscoveryResult`
    whose graph holds one tail-tail edge per nonzero precision pair, on the
    ``mixed`` face. The chosen penalty lives in the diagnostics, with the
    EBIC value when the path ran.
    """

    ingredient: ClassVar[str | None] = None

    def __init__(
        self,
        lam: float | None = None,
        *,
        gamma: float = 0.5,
        n_lambdas: int = 100,
        tol: float = 1e-3,
        max_iter: int = 100,
    ) -> None:
        if lam is not None and lam < 0.0:
            raise ValueError(f"lam must be None or >= 0, got {lam}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if n_lambdas < 1:
            raise ValueError(f"n_lambdas must be >= 1, got {n_lambdas}")
        if tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.lam = lam
        self.gamma = gamma
        self.n_lambdas = n_lambdas
        self.tol = tol
        self.max_iter = max_iter

    # -- the run ---------------------------------------------------------

    def fit(self, data: TabularData, knowledge: DomainKnowledge | None = None) -> DiscoveryResult:
        """Run the screen and return the undirected skeleton with its provenance."""
        _refuse_soft(knowledge, "Glasso")
        if not isinstance(data, TabularData):
            raise TypeError(
                f"glasso reads TabularData, got {type(data).__name__}; the Gaussian model "
                "assumes exchangeable rows, so temporal shapes do not fit here"
            )
        offenders = [name for name in data.variables if not data.schema[name].is_continuous]
        if offenders:
            raise ValueError(
                "glasso fits a Gaussian model, so every variable must be continuous; "
                f"not continuous: {offenders}"
            )
        names = list(data.variables)
        if len(names) < 2:
            raise ValueError(f"glasso needs at least two continuous variables, got {len(names)}")
        complete = data.dropna()
        n = complete.n_rows
        if n < 2:
            raise ValueError(f"glasso needs at least two complete rows, got {n}")
        x = complete.matrix(names)
        constant = [names[i] for i in np.flatnonzero(x.std(axis=0) == 0.0)]
        if constant:
            raise ValueError(f"constant columns have no correlation to screen: {constant}")
        s = np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)
        p = len(names)
        free, factor = self._knowledge_arrays(names, knowledge)
        ebic_value: float | None = None
        if self.lam is not None:
            if self.lam == 0.0 and p >= n:
                raise ValueError(
                    f"lam=0 asks for the unpenalized MLE, which does not exist with p={p} "
                    f"variables and n={n} rows; use a positive lam or the EBIC path"
                )
            theta, _, n_iter = _penalized_precision(
                s, self.lam, factor=factor, free=free, tol=self.tol, max_iter=self.max_iter
            )
            lam, n_tried = self.lam, 1
        else:
            theta, lam, ebic_value, n_iter, n_tried = _ebic_path(
                s,
                n=n,
                factor=factor,
                free=free,
                gamma=self.gamma,
                n_lambdas=self.n_lambdas,
                tol=self.tol,
                max_iter=self.max_iter,
            )
        edges = set()
        for i in range(p):
            for j in range(i + 1, p):
                if abs(float(theta[i, j])) > _EDGE_EPS:
                    a, b = sorted((names[i], names[j]))
                    edges.add(Edge.make(a, Mark.TAIL, b, Mark.TAIL))
        graph = MixedGraph(nodes=frozenset(names), edges=frozenset(edges), kind=GraphKind.MIXED)
        diagnostics: dict[str, float] = {
            "lam": float(lam),
            "n_edges": float(len(edges)),
            "n_iterations": float(n_iter),
            "n_lambdas_tried": float(n_tried),
        }
        if ebic_value is not None:
            diagnostics["ebic"] = float(ebic_value)
        return DiscoveryResult(
            graph=graph,
            method="glasso",
            test=None,
            score=None,
            alpha=None,
            n=n,
            diagnostics=diagnostics,
            sepsets=None,
        )

    @staticmethod
    def _knowledge_arrays(
        names: list[str], knowledge: DomainKnowledge | None
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        """The per-entry mask and penalty multipliers the knowledge implies.

        ``free`` is False where a pair is banned in both directions (the
        coordinate never updates, so the entry is a hard zero); ``factor``
        is 0 on required pairs (unpenalized) and 1 elsewhere.
        """
        p = len(names)
        free = _default_free(p)
        factor = _default_factor(p)
        if knowledge is None:
            return free, factor
        for i in range(p):
            for j in range(i + 1, p):
                a, b = names[i], names[j]
                if not knowledge.permits_edge(a, b) and not knowledge.permits_edge(b, a):
                    free[i, j] = free[j, i] = False
        index = {name: i for i, name in enumerate(names)}
        for u, v in knowledge.required:
            if u in index and v in index:
                i, j = index[u], index[v]
                factor[i, j] = factor[j, i] = 0.0
        return free, factor
