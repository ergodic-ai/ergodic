# SPDX-License-Identifier: Apache-2.0
"""CATE model selection and Q-aggregation on the doubly-robust tau-risk.

The van der Schaar lab's finding is that model selection, not the estimator family,
drives end-to-end CATE accuracy, and the Mahajan et al. (ICLR 2024) study showed the
doubly-robust tau-risk dominates the surrogate selection metrics. With the cross-fit
AIPW score ``Gamma_i`` (its conditional mean is the CATE), the DR tau-risk of a
candidate ``tau_hat`` is

    R(tau_hat) = mean_i[(Gamma_i - tau_hat(X_i))^2],

whose minimizer over functions of ``X`` is the true CATE, so a smaller risk is a
better CATE estimate. :func:`select_cate` fits each candidate through the ``cate``
facade, scores it against ONE shared ``Gamma`` (so the comparison is fair and the
candidate's own out-of-fold ``tau_hat`` keeps the ranking honest), and returns a
ranked report with the selected result.

:func:`select_cate` also returns a Q-aggregation ensemble (Lan-Syrgkanis, AISTATS
2024): convex weights over the candidates minimizing the DR loss of the ensemble
with a Q penalty that pulls weight toward the individually-better models. The
ensemble attains the oracle regret rate without any single candidate being close to
the truth. Phase-2 lean: the convex DR-loss aggregate with the entropic Q mix;
richer aggregation (cross-fit weight selection, the exact entropic dual) is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from ..data import TabularData
    from ..graph import MixedGraph
    from ._perunit import CATEResult


# The default candidate set: the Phase-1 registered adjustment-family CATE methods.
# "causal_forest"/"grf" is deliberately omitted (the orchestrator wires it later).
_DEFAULT_CANDIDATES: tuple[str, ...] = ("dr_learner", "r_learner", "x_learner", "t_learner")


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's selection score: its method name and DR tau-risk."""

    method: str
    tau_risk: float

    def __repr__(self) -> str:
        return f"CandidateScore({self.method!r}, tau_risk={self.tau_risk:.5g})"


@dataclass(frozen=True)
class SelectionReport:
    """The ranked CATE model-selection result with the chosen and aggregated models.

    ``ranking`` is the candidates sorted by ascending DR tau-risk (best first).
    ``selected`` is the fitted :class:`~ergodic.inference.CATEResult` of the lowest
    risk; ``selected_method`` names it. ``aggregate`` is the Q-aggregation ensemble
    result and ``weights`` its convex weights (aligned to ``aggregate_methods``),
    with ``aggregate_tau_risk`` the ensemble's DR tau-risk.
    """

    ranking: tuple[CandidateScore, ...]
    selected_method: str
    selected: CATEResult
    aggregate: CATEResult
    aggregate_methods: tuple[str, ...]
    weights: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    aggregate_tau_risk: float = float("nan")

    def risk(self, method: str) -> float:
        """The DR tau-risk of a candidate by name (raises when it was not scored)."""
        for score in self.ranking:
            if score.method == method:
                return score.tau_risk
        raise KeyError(f"{method!r} is not among the scored candidates")

    @property
    def best_single_tau_risk(self) -> float:
        """The DR tau-risk of the best single candidate (the first in the ranking)."""
        return self.ranking[0].tau_risk

    def frame(self) -> dict[str, float]:
        """A method-to-risk mapping (the aggregate included), handy for tables/tests."""
        out = {score.method: score.tau_risk for score in self.ranking}
        out["q_aggregate"] = self.aggregate_tau_risk
        return out


def _tau_risk(scores: npt.NDArray[np.float64], tau_hat: npt.NDArray[np.float64]) -> float:
    """The doubly-robust tau-risk ``mean[(Gamma - tau_hat)^2]``."""
    diff = scores - tau_hat
    return float(np.mean(diff * diff))


def _frank_wolfe_simplex(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    per_model_risk: npt.NDArray[np.float64],
    *,
    nu: float,
    iters: int,
) -> npt.NDArray[np.float64]:
    """Convex weights minimizing the Q-aggregation DR objective over the simplex.

    Minimizes ``(1 - nu) * mean[(target - design @ w)^2] + nu * sum_j w_j risk_j``
    subject to ``w >= 0`` and ``sum w = 1`` by Frank-Wolfe (conditional gradient):
    each step moves toward the simplex vertex of steepest descent, so the iterate
    stays feasible and the convex objective converges. The Q penalty (the second
    term) pulls weight toward the individually lower-risk models, the entropic-mix
    role of Lan-Syrgkanis Q-aggregation. Starts from the best single model.
    """
    n, m = design.shape
    best = int(np.argmin(per_model_risk))
    w = np.zeros(m, dtype=np.float64)
    w[best] = 1.0
    for step in range(iters):
        resid = design @ w - target
        # gradient of (1-nu)/n * ||design w - target||^2 + nu * risk' w
        grad = (1.0 - nu) * (2.0 / n) * (design.T @ resid) + nu * per_model_risk
        vertex = int(np.argmin(grad))
        direction = -w.copy()
        direction[vertex] += 1.0
        gap = -float(grad @ direction)
        if gap <= 1e-12:
            break
        gamma = 2.0 / (step + 2.0)
        w = w + gamma * direction
    w = np.clip(w, 0.0, None)
    total = float(np.sum(w))
    return w / total if total > 0.0 else w


def select_cate(
    graph: MixedGraph,
    data: TabularData,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    *,
    candidates: Sequence[str] | None = None,
    learner: Any = None,
    treated: Any | None = None,
    random_state: int | None = None,
    n_boot: int = 1000,
    folds: int = 5,
    nu: float = 0.5,
    q_iters: int = 200,
) -> SelectionReport:
    """Fit and rank candidate CATE learners by the doubly-robust tau-risk.

    Identifies the effect (the same back-door gate as :func:`~ergodic.inference.cate`),
    fits each candidate in ``candidates`` (default the Phase-1 registered meta-learners
    ``dr_learner``/``r_learner``/``x_learner``/``t_learner``) through the ``cate``
    facade, and scores each against one shared cross-fit AIPW signal ``Gamma`` by the
    DR tau-risk ``mean[(Gamma - tau_hat)^2]``. Returns a :class:`SelectionReport` with
    the ranked candidates (best risk first), the selected fitted
    :class:`~ergodic.inference.CATEResult`, and a Q-aggregation convex-ensemble result.

    ``learner`` fills every candidate's nuisance slots; ``nu`` is the Q-aggregation
    mix (the penalty pulling weight toward lower-risk models). The shared ``Gamma`` is
    computed once with :func:`~ergodic.inference.dr_scores`, so the comparison is fair
    across candidates and each candidate's own out-of-fold ``tau_hat`` keeps the
    ranking honest.
    """
    from ..identification import identify_effect
    from ._estimate import cate
    from ._scores import dr_scores

    names = tuple(candidates) if candidates is not None else _DEFAULT_CANDIDATES
    if not names:
        raise ValueError("candidates is empty; pass at least one CATE method name")

    identification = identify_effect(graph, treatment, outcome)
    if not identification.identifiable:
        raise ValueError(
            f"effect of {treatment} on {outcome} is not identifiable: {identification.estimand}"
        )
    if identification.strategy != "adjustment":
        raise ValueError(
            "select_cate needs an adjustment-identified (back-door) effect; the effect "
            f"was identified by {identification.strategy!r}, which has no CATE meta-learner"
        )

    # One shared orthogonal signal: the comparison currency and the aggregation target.
    shared = dr_scores(
        identification,
        data,
        learners=learner,
        folds=folds,
        treated=treated,
        random_state=random_state,
    )
    gamma = np.asarray(shared.scores, dtype=np.float64)

    fitted: dict[str, CATEResult] = {}
    tau_hats: dict[str, npt.NDArray[np.float64]] = {}
    scored: list[CandidateScore] = []
    for method in names:
        result = cate(
            graph,
            data,
            treatment,
            outcome,
            method=method,
            learner=learner,
            treated=treated,
            random_state=random_state,
            n_boot=n_boot,
        )
        # the candidate's out-of-fold proxy ranking when cached (the honest choice),
        # else its in-sample fit. Scored against the shared Gamma.
        cached = result._scores
        if cached is not None and cached.tau_hat is not None:
            tau = np.asarray(cached.tau_hat, dtype=np.float64)
        else:
            tau = np.asarray(result.values, dtype=np.float64)
        fitted[method] = result
        tau_hats[method] = tau
        scored.append(CandidateScore(method=method, tau_risk=_tau_risk(gamma, tau)))

    ranking = tuple(sorted(scored, key=lambda s: (s.tau_risk, s.method)))
    selected_method = ranking[0].method
    selected = fitted[selected_method]

    # Q-aggregation: convex DR-loss ensemble with the Q penalty over the candidates.
    agg_methods = tuple(names)
    design = np.column_stack([tau_hats[m] for m in agg_methods])
    per_model_risk = np.asarray([_tau_risk(gamma, tau_hats[m]) for m in agg_methods])
    weights = _frank_wolfe_simplex(design, gamma, per_model_risk, nu=nu, iters=q_iters)
    agg_tau = np.asarray(design @ weights, dtype=np.float64)
    aggregate = _aggregate_result(selected, agg_tau, shared)
    return SelectionReport(
        ranking=ranking,
        selected_method=selected_method,
        selected=selected,
        aggregate=aggregate,
        aggregate_methods=agg_methods,
        weights=weights,
        aggregate_tau_risk=_tau_risk(gamma, agg_tau),
    )


def _aggregate_result(
    template: CATEResult,
    tau_agg: npt.NDArray[np.float64],
    shared: Any,
) -> CATEResult:
    """Wrap the convex-combination per-unit CATE in a CATEResult that can explain.

    Carries the shared cross-fit scores with ``tau_hat`` set to the aggregated
    prediction, so :meth:`~ergodic.inference.CATEResult.explain` runs the Tier-B
    characterization on the ensemble ranking. A point result (the per-unit
    uncertainty of the ensemble is a documented Phase-2 deferral).
    """
    from dataclasses import replace

    from ._perunit import CATEResult
    from ._scores import DRScores

    agg_scores: DRScores = replace(shared, tau_hat=tau_agg)
    return CATEResult(
        treatment=template.treatment,
        outcome=template.outcome,
        covariates=template.covariates,
        n=tau_agg.shape[0],
        kind="point",
        source="q_aggregate",
        identification=template.identification,
        _values=np.asarray(tau_agg, dtype=np.float64),
        _scores=agg_scores,
    )
