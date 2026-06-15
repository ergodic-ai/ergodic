# SPDX-License-Identifier: Apache-2.0
"""Mediation: decompose a total effect into a direct and an indirect part.

The total effect of a treatment on an outcome splits into the part that flows
through a mediator (the natural indirect effect, NIE) and the part that does not
(the natural direct effect, NDE). The controlled direct effect (CDE) is a third,
plainer quantity: the effect of the treatment with the mediator held fixed by
intervention.

:func:`identify_mediation` answers the graphical question. NDE and NIE are
identified by the mediation formula (Pearl, 2001) under a covariate set ``W`` of
non-descendants of the treatment that satisfies three adjustment conditions; when
no such ``W`` exists the effects are not identifiable, which is the canonical
post-treatment-confounder (cross-world) failure. CDE is a plain interventional
contrast on ``{X, M}``, so it routes through the general ID engine
(:func:`~ergodic.identification.identify_distribution`) and needs no cross-world
assumption.

Single graphs only (a DAG or ADMG). A class face raises with a pointer, like the
rest of identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from ._criteria import (
    _ancestors,
    _as_set,
    _check_single_graph,
    _descendants,
    _join,
    _validate,
    is_valid_adjustment_set,
)
from ._distribution import identify_distribution

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..graph import MixedGraph


@dataclass(frozen=True)
class MediationID:
    """The result of trying to identify a mediation decomposition from a graph.

    ``identifiable`` is the verdict for the natural direct and indirect effects:
    ``True`` when a covariate set ``covariates`` makes the mediation formula
    valid, ``False`` when none exists (the cross-world / post-treatment-confounder
    failure). ``nde_estimand`` and ``nie_estimand`` render the mediation formula
    with that set; they are empty strings when not identifiable.

    The controlled direct effect is tracked separately, since it is a plain
    interventional contrast (no cross-world assumption): ``cde_identifiable`` is
    its verdict and ``cde_estimand`` its formula, decided through the general ID
    algorithm on ``P(Y | do(X), do(M))``. CDE can be identified when NDE/NIE are
    not, and vice versa.

    ``assumptions`` states, in words, the (untestable) claims the decomposition
    rests on, the same honesty the :class:`~ergodic.inference.Design` records carry
    for the quasi-experimental designs.
    """

    treatment: frozenset[str]
    outcome: frozenset[str]
    mediators: frozenset[str]
    identifiable: bool
    nde_estimand: str
    nie_estimand: str
    cde_identifiable: bool
    cde_estimand: str
    covariates: frozenset[str] | None
    assumptions: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.identifiable

    def __repr__(self) -> str:
        arrow = (
            f"{sorted(self.treatment)} -> [{', '.join(sorted(self.mediators))}] "
            f"-> {sorted(self.outcome)}"
        )
        if not self.identifiable:
            return f"MediationID(NDE/NIE not identifiable, {arrow}, cde={self.cde_identifiable})"
        return (
            f"MediationID(identifiable, {arrow}, "
            f"W={{{_join(self.covariates or frozenset())}}}, cde={self.cde_identifiable})"
        )


# -- the mediation-formula estimands -----------------------------------------


def _w_clause(w: frozenset[str]) -> tuple[str, str]:
    """The ``sum_w ... P(w)`` wrapper and the conditioning tail for a covariate set."""
    if not w:
        return "", ""
    return "sum_w ", f", {_join(w)}"


def _nde_formula(x: frozenset[str], y: frozenset[str], m: frozenset[str], w: frozenset[str]) -> str:
    """``NDE = sum_w sum_m [E[Y|x,m,w] - E[Y|x',m,w]] P(m|x',w) P(w)``."""
    sw, tail = _w_clause(w)
    xj, yj, mj = _join(x), _join(y), _join(m)
    body = f"sum_m [E[{yj} | {xj}, {mj}{tail}] - E[{yj} | {xj}', {mj}{tail}]] P({mj} | {xj}'{tail})"
    tail_p = f" P({_join(w)})" if w else ""
    return f"NDE = {sw}{body}{tail_p}"


def _nie_formula(x: frozenset[str], y: frozenset[str], m: frozenset[str], w: frozenset[str]) -> str:
    """``NIE = sum_w sum_m E[Y|x',m,w] [P(m|x,w) - P(m|x',w)] P(w)``."""
    sw, tail = _w_clause(w)
    xj, yj, mj = _join(x), _join(y), _join(m)
    body = f"sum_m E[{yj} | {xj}', {mj}{tail}] [P({mj} | {xj}{tail}) - P({mj} | {xj}'{tail})]"
    tail_p = f" P({_join(w)})" if w else ""
    return f"NIE = {sw}{body}{tail_p}"


# -- the search over covariate sets ------------------------------------------


def _candidate_w(
    graph: MixedGraph, x: frozenset[str], y: frozenset[str], m: frozenset[str]
) -> frozenset[str]:
    """The canonical covariate candidate: non-descendants of X among the pre-treatment ancestors.

    Mirrors the adjustment criterion's canonical set (the ancestors of the nodes
    of interest), then drops the treatment, outcome, mediators, and any descendant
    of the treatment (the mediation formula's ``W`` must be non-descendants of
    ``X``), and the latent nodes (never conditionable).
    """
    pool = _ancestors(graph, x | y | m) | x | y | m
    forbidden = x | y | m | _descendants(graph, x) | graph.latent
    return frozenset(pool - forbidden)


def _conditions_hold(
    graph: MixedGraph,
    x: frozenset[str],
    y: frozenset[str],
    m: frozenset[str],
    w: frozenset[str],
) -> bool:
    """The three mediation adjustment conditions M1-M3 for a covariate set ``W``.

    M1: ``W`` is a valid adjustment set for ``X -> Y``.
    M2: ``W`` is a valid adjustment set for ``X -> M``.
    M3: ``W u {X}`` is a valid adjustment set for ``M -> Y``.

    The cross-world condition (no mediator-outcome confounder that is itself a
    descendant of the treatment) is enforced implicitly: such a confounder would
    have to enter the ``M -> Y`` set to block its path, but it is a descendant of
    ``X`` and so cannot be a non-descendant ``W``; M3 then has no valid set and
    fails. That is the canonical non-identification.
    """
    if not is_valid_adjustment_set(graph, x, y, w):
        return False
    if not is_valid_adjustment_set(graph, x, m, w):
        return False
    return is_valid_adjustment_set(graph, m, y, w | x)


def _minimize(
    graph: MixedGraph,
    x: frozenset[str],
    y: frozenset[str],
    m: frozenset[str],
    w: frozenset[str],
) -> frozenset[str]:
    """Greedily drop covariates while the three conditions still hold (one minimal set)."""
    keep = set(w)
    changed = True
    while changed:
        changed = False
        for node in sorted(keep):
            if _conditions_hold(graph, x, y, m, frozenset(keep - {node})):
                keep.discard(node)
                changed = True
    return frozenset(keep)


def _find_w(
    graph: MixedGraph,
    x: frozenset[str],
    y: frozenset[str],
    m: frozenset[str],
) -> frozenset[str] | None:
    """Search for a covariate set satisfying M1-M3: the canonical set, then any subset.

    Tries the canonical candidate first (and minimizes it when it works, like
    :func:`~ergodic.identification.minimal_adjustment_set`). If the canonical set
    fails, searches its subsets from small to large, since a smaller set can block
    a path the canonical set re-opens by conditioning on a collider. Returns the
    found set (minimized) or ``None`` when no subset works.
    """
    candidate = _candidate_w(graph, x, y, m)
    if _conditions_hold(graph, x, y, m, candidate):
        return _minimize(graph, x, y, m, candidate)
    pool = sorted(candidate)
    for size in range(len(pool)):
        for combo in combinations(pool, size):
            sub = frozenset(combo)
            if _conditions_hold(graph, x, y, m, sub):
                return sub
    return None


# -- the public entry --------------------------------------------------------


def _check_mediators(
    graph: MixedGraph, x: frozenset[str], y: frozenset[str], m: frozenset[str]
) -> None:
    """Mediators must be nonempty, disjoint from X and Y, and on a causal path X -> Y.

    A mediator that does not sit on a directed path from a treatment to an outcome
    carries none of the effect, so a decomposition through it is meaningless; this
    catches that mistake loudly instead of returning a vacuous answer.
    """
    if not m:
        raise ValueError("mediators must name at least one node")
    if x & m:
        raise ValueError(f"treatment and mediators must be disjoint; both contain {_join(x & m)}")
    if y & m:
        raise ValueError(f"outcome and mediators must be disjoint; both contain {_join(y & m)}")
    if m & graph.latent:
        raise ValueError(f"mediators must be observed; {_join(m & graph.latent)} is latent")
    downstream = _descendants(graph, x)
    upstream = _ancestors(graph, y)
    on_path = downstream & upstream
    stranded = m - on_path
    if stranded:
        raise ValueError(
            f"mediator(s) {_join(stranded)} do not lie on a directed path from "
            f"{_join(x)} to {_join(y)} (a mediator must be a descendant of the "
            "treatment and an ancestor of the outcome)"
        )


def _cde_through_id(
    graph: MixedGraph, x: frozenset[str], y: frozenset[str], m: frozenset[str]
) -> tuple[bool, str]:
    """Identify the controlled direct effect through the general ID engine.

    The CDE is the contrast of ``P(Y | do(X), do(M))`` between treatment levels at
    a fixed mediator level, so its identifiability is exactly that of the joint
    interventional ``P(Y | do(X, M))``. Returns ``(identifiable, estimand)``;
    the estimand is the ID functional's render, or the hedge reason on failure.
    """
    res = identify_distribution(graph, y, x | m)
    if res.identifiable and res.functional is not None:
        estimand = (
            f"CDE(m) = E[{_join(y)} | do({_join(x)}=1), do({_join(m)}=m)] - "
            f"E[{_join(y)} | do({_join(x)}=0), do({_join(m)}=m)], "
            f"with P({_join(y)} | do({_join(x)}), do({_join(m)})) = {res.functional.render()}"
        )
        return True, estimand
    reason = res.hedge.render() if res.hedge is not None else "not identifiable"
    return False, f"CDE not identifiable: {reason}"


def identify_mediation(
    graph: MixedGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    mediators: str | Iterable[str],
) -> MediationID:
    """Identify the mediation decomposition of ``treatment -> outcome`` through ``mediators``.

    Searches for a covariate set ``W`` of non-descendants of the treatment under
    which the natural direct and indirect effects are identified by the mediation
    formula (Pearl, 2001): ``W`` is a valid adjustment set for ``X -> Y`` and for
    ``X -> M``, and ``W u {X}`` is a valid adjustment set for ``M -> Y``. When such
    a ``W`` exists, NDE and NIE are identifiable and their estimands render in the
    mediation-formula style; when none exists, they are not identifiable, the
    canonical case being a mediator-outcome confounder that is itself affected by
    the treatment (a cross-world obstruction).

    The controlled direct effect is identified separately, through the general ID
    algorithm on ``P(Y | do(X), do(M))``: a plain interventional contrast that
    needs no cross-world assumption, and may identify when NDE/NIE do not.

    Single graphs only (a DAG or ADMG). A CPDAG, MAG, or PAG raises (extract a
    member or enumerate the class).
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    m = _as_set(mediators)
    _validate(graph, x, y, m)
    _check_single_graph(graph, "mediation")
    _check_mediators(graph, x, y, m)

    cde_identifiable, cde_estimand = _cde_through_id(graph, x, y, m)

    w = _find_w(graph, x, y, m)
    if w is None:
        assumptions: tuple[str, ...] = (
            "the natural direct and indirect effects are not identifiable: no set "
            "of non-descendants of the treatment satisfies the three adjustment "
            "conditions. This is the cross-world obstruction, typically a "
            "mediator-outcome confounder that is itself affected by the treatment "
            "(a recanting witness); the controlled direct effect may still be "
            "identifiable. Consider a path-specific effect or report the CDE.",
        )
        return MediationID(
            treatment=x,
            outcome=y,
            mediators=m,
            identifiable=False,
            nde_estimand="",
            nie_estimand="",
            cde_identifiable=cde_identifiable,
            cde_estimand=cde_estimand,
            covariates=None,
            assumptions=assumptions,
        )

    if w:
        seq_ignorability = (
            f"sequential ignorability given W = {{{_join(w)}}}: conditional on these "
            "covariates, the treatment is as-if randomized for both the mediator and "
            "the outcome, and the mediator is as-if randomized for the outcome."
        )
    else:
        seq_ignorability = (
            "sequential ignorability with no covariates needed: the treatment is "
            "as-if randomized for both the mediator and the outcome, and the "
            "mediator is as-if randomized for the outcome given the treatment."
        )
    assumptions = (
        seq_ignorability,
        "the cross-world independence Y(x, m) is independent of M(x') given W "
        "(equivalently, no mediator-outcome confounder is affected by the "
        "treatment). This is graphically certified here but is untestable from "
        "data: no observational or experimental design can refute it.",
    )
    return MediationID(
        treatment=x,
        outcome=y,
        mediators=m,
        identifiable=True,
        nde_estimand=_nde_formula(x, y, m, w),
        nie_estimand=_nie_formula(x, y, m, w),
        cde_identifiable=cde_identifiable,
        cde_estimand=cde_estimand,
        covariates=w,
        assumptions=assumptions,
    )
