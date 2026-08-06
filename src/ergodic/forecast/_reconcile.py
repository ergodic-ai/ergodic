# SPDX-License-Identifier: Apache-2.0
"""Reconciliation: a forecast at every level, and the levels agree.

A hierarchy is an arithmetic promise. The total is the sum of the regions, a
region is the sum of its products, and that holds in the data by construction.
Forecasts made level by level break the promise, because five independent models
have no reason to add up. Reconciliation puts it back.

Which strategy does the putting back is a modeling decision, not a formality. It
says which level you believe:

- :data:`bottom_up` believes the leaves. Every parent becomes the sum of its
  children's draws, and a parent's own model is discarded from the answer.
- :data:`top_down` believes the top. The root keeps its draws and they are split
  downward, either by the shares the training window showed or, per draw and per
  step, by the shares the children's own forecasts imply.
- :data:`middle_out` believes a level in between, and composes the two.
- :data:`mint` believes everybody in proportion to how sure they are. It is the
  minimum-trace projection: the smallest weighted move from the unreconciled
  beliefs onto the coherent subspace, applied one draw at a time so the
  uncertainty survives.
- :data:`none` believes nobody and changes nothing, which is how you see what
  reconciliation actually did.

Everything here is numpy over draws. The models have already run by the time any
of it happens, so a strategy costs no sampling and switching strategies costs no
refit. Every allocation is per draw: a share applied to a median would hand back
a point forecast wearing a band it did not earn.

Sum hierarchies only. A mean-aggregation tree raises, because a mean constraint
is a different projection and the strategies below all assume the sum.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ._result import coherence_gap

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from ..data import HierarchicalTSData, Hierarchy
    from ._result import ForecastResult

    Array = npt.NDArray[np.float64]

__all__ = [
    "PROPORTION_RULES",
    "RECONCILE_METHODS",
    "WEIGHT_RULES",
    "constraint_matrix",
    "historical_shares",
    "mint_weights",
    "reconcile",
]

#: The strategy menu, in the order the module docstring introduces it.
RECONCILE_METHODS = ("bottom_up", "top_down", "middle_out", "mint", "none")
#: How ``top_down`` and ``middle_out`` split a parent between its children.
PROPORTION_RULES = ("historical", "forecast")
#: Where ``mint`` gets the diagonal of its weight matrix.
WEIGHT_RULES = ("posterior", "mint_shrink")

#: Coherence counts as reached below this absolute gap. Reconciliation is exact
#: arithmetic, so what is left is rounding: a sum of float64 terms at the scale
#: of the data, which stays around 1e-12 for a business series.
COHERENCE_TOL = 1e-8
#: A sibling sum this close to zero cannot carry a share, so that draw and step
#: fall back to the historical split.
_SHARE_FLOOR = 1e-12
#: Relative nudge on the diagonal of ``A W A'`` before the Cholesky, so a node
#: with no predictive spread cannot make the system singular. Relative, because
#: an absolute jitter means one thing on a series in units and another on the
#: same series in millions.
_JITTER = 1e-10


# -- the tree, as arithmetic ---------------------------------------------------


def _ordered_nodes(hierarchy: Hierarchy) -> tuple[str, ...]:
    """Every node of the tree in one deterministic order: the vector index."""
    return tuple(sorted(hierarchy.nodes))


def _leaves_up(hierarchy: Hierarchy) -> tuple[str, ...]:
    """The internal nodes, deepest first, so a parent is reached after its children."""
    depth = hierarchy.levels()
    return tuple(sorted(hierarchy.internal_nodes, key=lambda node: (-depth[node], node)))


def _root_down(hierarchy: Hierarchy) -> tuple[str, ...]:
    """The internal nodes, shallowest first, so a parent is reached before its children."""
    depth = hierarchy.levels()
    return tuple(sorted(hierarchy.internal_nodes, key=lambda node: (depth[node], node)))


def constraint_matrix(hierarchy: Hierarchy, nodes: tuple[str, ...] | None = None) -> Array:
    """The constraint matrix ``A``: one row per parent, reading ``parent - sum(children) = 0``.

    A coherent forecast is exactly a vector ``x`` over ``nodes`` with ``A x = 0``.
    Multi-level trees need no extra rows: the direct-parent rows imply every
    grandparent identity, because a parent that equals its children and children
    that equal theirs leaves nothing to add.
    """
    order = nodes if nodes is not None else _ordered_nodes(hierarchy)
    index = {node: position for position, node in enumerate(order)}
    parents = _root_down(hierarchy)
    matrix = np.zeros((len(parents), len(order)), dtype=np.float64)
    for row, parent in enumerate(parents):
        matrix[row, index[parent]] = 1.0
        for child in hierarchy.children_of(parent):
            matrix[row, index[child]] = -1.0
    return matrix


def _require_sum(hierarchy: Hierarchy) -> None:
    """Refuse a mean-aggregation tree, and say what would have to happen instead."""
    if hierarchy.aggregation != "sum":
        raise NotImplementedError(
            f"reconciliation is built for sum hierarchies and this one aggregates by "
            f"{hierarchy.aggregation!r}. A mean constraint is a different projection "
            "(the rows of A carry 1/k rather than 1) and every strategy here assumes the "
            "sum. Rebuild the hierarchy with aggregation='sum', carrying counts as their "
            "own series if the mean is what you need."
        )


def _require_coverage(result: ForecastResult, hierarchy: Hierarchy) -> None:
    """Every node of the tree has to be in the result; there is nothing to infer."""
    missing = sorted(set(hierarchy.nodes) - set(result.nodes))
    if missing:
        raise ValueError(
            f"the result does not cover the hierarchy: {missing} have no draws. Every node "
            "needs a forecast before the levels can be made to agree, either from its own "
            "mechanism or by aggregating its children; forecast() fills the aggregate-only "
            "nodes for you."
        )


# -- historical shares ---------------------------------------------------------


def historical_shares(
    data: HierarchicalTSData, hierarchy: Hierarchy | None = None
) -> dict[str, float]:
    """Each child's average share of its parent over the training window.

    The share is the mean of the ratio, step by step, rather than the ratio of
    the means: a month where the parent was small counts as much as a month where
    it was large, which is what "share" usually means to the person asking. The
    shares in a sibling set are then renormalized to sum to one, so an allocation
    can never leak or invent volume.

    A node's series is its own column when it has one and
    :meth:`~ergodic.data.HierarchicalTSData.aggregate` when it does not, which is
    the same definition the data layer's coherence check uses.
    """
    tree = hierarchy if hierarchy is not None else data.hierarchy
    if tree != data.hierarchy:
        raise ValueError(
            "the hierarchy passed in is not the one the data carries, and the shares are "
            "read through the data's own aggregation; pass the dataset's hierarchy"
        )
    series = {node: _series_of(data, node) for node in tree.nodes}
    shares: dict[str, float] = {}
    for parent in tree.internal_nodes:
        top = series[parent]
        usable = np.abs(top) > _SHARE_FLOOR
        if not usable.any():
            raise ValueError(
                f"{parent!r} is zero over the whole training window, so its children have no "
                "share of it; historical proportions cannot be read from this series"
            )
        raw = {
            child: float(np.mean(series[child][usable] / top[usable]))
            for child in tree.children_of(parent)
        }
        total = float(sum(raw.values()))
        if abs(total) < _SHARE_FLOOR:
            raise ValueError(
                f"the children of {parent!r} have shares summing to {total:.3g}, which cannot "
                "be renormalized; check that the series belong to the same hierarchy"
            )
        for child, value in raw.items():
            shares[child] = value / total
    return shares


def _series_of(data: HierarchicalTSData, node: str) -> Array:
    """One node's training series: its column when it has one, its children's sum when not."""
    if node in data.variables:
        return np.asarray(data.column(node), dtype=np.float64)
    return np.asarray(data.aggregate(node), dtype=np.float64)


# -- MinT weights --------------------------------------------------------------


def mint_weights(
    result: ForecastResult,
    hierarchy: Hierarchy,
    *,
    weights: str = "posterior",
    residual_var: Mapping[str, float] | None = None,
    trust: Mapping[str, float] | None = None,
) -> tuple[tuple[str, ...], Array]:
    """The diagonal of ``W``, one row per forecast step: ``(nodes, (H, N))``.

    Two rules, both diagonal. ``"posterior"`` reads each node's own predictive
    variance at that step straight off its unreconciled draws, so a node that
    already says it is unsure gets moved further and nothing extra has to be
    estimated. ``"mint_shrink"`` reads the node's in-sample one-step residual
    variance, the classic MinT weighting, which is flat across the horizon and
    comes either from ``residual_var`` or from the result's own
    :attr:`~ergodic.forecast.ForecastResult.residual_var`, since it is a property
    of the fit rather than of the draws.

    ``trust`` divides a node's entry, so a node you trust ten times as much moves
    about a tenth as far. That is the ``ForecastBelief.trust`` idea from the
    source, kept.
    """
    if weights not in WEIGHT_RULES:
        raise ValueError(f"weights must be one of {WEIGHT_RULES}, got {weights!r}")
    nodes = _ordered_nodes(hierarchy)
    horizon = int(result.horizon)

    if weights == "posterior":
        diagonal = np.stack([np.var(result.draws(node), axis=0, ddof=1) for node in nodes], axis=1)
    else:
        # A residual variance is a property of the fit, so it cannot be read off
        # the draws. The result carries it when a fitter computed it, which is
        # what keeps this weighting available without sampling anything again.
        residual_var = residual_var if residual_var is not None else result.residual_var
        if residual_var is None:
            raise ValueError(
                "weights='mint_shrink' needs the in-sample residual variance of every node, "
                "which lives with the fits rather than with the draws. This result carries "
                "none, so pass residual_var={node: variance}, or produce the forecast with "
                "forecast() on a HierarchicalTSData, which computes it from the fitted paths"
            )
        missing = sorted(set(nodes) - set(residual_var))
        if missing:
            raise ValueError(f"residual_var has no entry for {missing}")
        row = np.array([float(residual_var[node]) for node in nodes], dtype=np.float64)
        if not np.all(np.isfinite(row)) or np.any(row < 0.0):
            raise ValueError("residual_var must be finite and non-negative for every node")
        diagonal = np.repeat(row[None, :], horizon, axis=0)

    if trust:
        unknown = sorted(set(trust) - set(nodes))
        if unknown:
            raise ValueError(f"trust names nodes outside the hierarchy: {unknown}")
        scale = np.ones(len(nodes), dtype=np.float64)
        for node, weight in trust.items():
            value = float(weight)
            if value <= 0.0:
                raise ValueError(f"trust[{node!r}] is {value}, and trust must be positive")
            scale[nodes.index(node)] = value
        diagonal = diagonal / scale[None, :]
    return nodes, np.asarray(diagonal, dtype=np.float64)


# -- the strategies ------------------------------------------------------------


def _stack(series: Mapping[str, Array], nodes: tuple[str, ...]) -> Array:
    """The draws of every node as one ``(S, H, N)`` block."""
    return np.stack([series[node] for node in nodes], axis=2)


def _bottom_up(series: dict[str, Array], hierarchy: Hierarchy) -> dict[str, Array]:
    """Every parent becomes the sum of its children, deepest parent first."""
    for parent in _leaves_up(hierarchy):
        kids = hierarchy.children_of(parent)
        series[parent] = np.sum([series[kid] for kid in kids], axis=0)
    return series


def _allocate(
    series: dict[str, Array],
    unreconciled: Mapping[str, Array],
    hierarchy: Hierarchy,
    parents: tuple[str, ...],
    shares: Mapping[str, float],
    proportions: str,
) -> dict[str, Array]:
    """Split each parent's draws between its children, parent by parent, top first."""
    for parent in parents:
        kids = hierarchy.children_of(parent)
        top = series[parent]
        weights = np.array([shares[kid] for kid in kids], dtype=np.float64)
        if proportions == "historical":
            for kid, share in zip(kids, weights, strict=True):
                series[kid] = top * share
            continue
        # Forecast proportions: the children's own draws say how to split, one
        # draw and one step at a time, so a scenario where a child ran hot keeps
        # it hot. Where the siblings sum to nothing there is no ratio to take and
        # the historical split stands in.
        own = np.stack([np.asarray(unreconciled[kid], dtype=np.float64) for kid in kids], axis=0)
        denominator = own.sum(axis=0)
        usable = np.abs(denominator) > _SHARE_FLOOR
        safe = np.where(usable, denominator, 1.0)
        fractions = np.where(usable[None, :, :], own / safe[None, :, :], weights[:, None, None])
        for index, kid in enumerate(kids):
            series[kid] = top * fractions[index]
    return series


def _middle_out(
    series: dict[str, Array],
    unreconciled: Mapping[str, Array],
    hierarchy: Hierarchy,
    level: int,
    shares: Mapping[str, float],
    proportions: str,
) -> dict[str, Array]:
    """Keep one level's own draws, sum upward from it, allocate downward from it."""
    depth = hierarchy.levels()
    deepest = max(depth.values())
    if not 0 <= level <= deepest:
        raise ValueError(
            f"level must be between 0 (the root) and {deepest} (the deepest leaf), got {level}"
        )
    for parent in _leaves_up(hierarchy):
        if depth[parent] < level:
            kids = hierarchy.children_of(parent)
            series[parent] = np.sum([series[kid] for kid in kids], axis=0)
    below = tuple(node for node in _root_down(hierarchy) if depth[node] >= level)
    return _allocate(series, unreconciled, hierarchy, below, shares, proportions)


def _mint(
    series: dict[str, Array],
    hierarchy: Hierarchy,
    nodes: tuple[str, ...],
    diagonal: Array,
) -> dict[str, Array]:
    """Project every draw onto the coherent subspace, step by step.

    ``x_rec = x - W A' (A W A')^-1 A x``. The bracketed term is the constraint
    violation, and the rest spreads the fix over the nodes in proportion to their
    weight, which is what makes a node you trust move less. The projection runs
    per step because the posterior weights change with the horizon, and over all
    draws at once because the linear algebra does not care how many right-hand
    sides it gets.
    """
    matrix = constraint_matrix(hierarchy, nodes)
    parents = _root_down(hierarchy)
    block = _stack(series, nodes)
    out = np.array(block, dtype=np.float64, copy=True)
    for step in range(block.shape[1]):
        weights = diagonal[step]
        weighted = matrix * weights[None, :]
        gram = weighted @ matrix.T
        # A node moves in proportion to its weight, so a constraint whose every
        # node has weight zero is one that nothing is allowed to fix. Saying so
        # beats handing back a projection that quietly did not project.
        stuck = [parents[row] for row in range(gram.shape[0]) if gram[row, row] <= 0.0]
        if stuck:
            raise ValueError(
                f"every node under {stuck} has weight zero at step {step + 1}, so nothing can "
                "move and the levels cannot be made to agree. A weight of zero means the "
                "forecast is beyond question: check for a residual variance of zero, or trust "
                "so large it rounds the weight away."
            )
        chol = _factor(gram)
        current = block[:, step, :]
        # One refinement pass. The first solve leaves a residual violation of the
        # order of the conditioning (and of the jitter, when a degenerate weight
        # needed one), and reprojecting that residual through the same factor
        # takes it back down to rounding. The correction stays in the row space
        # of W A', so what comes out is still the weighted projection.
        for _ in range(2):
            violation = current @ matrix.T
            solved = np.linalg.solve(chol.T, np.linalg.solve(chol, violation.T))
            current = current - (weighted.T @ solved).T
        out[:, step, :] = current
    for index, node in enumerate(nodes):
        series[node] = np.ascontiguousarray(out[:, :, index])
    return series


def _factor(gram: Array) -> Array:
    """The Cholesky factor of ``A W A'``, jittered only if it has to be.

    With positive weights and a tree, the system is positive definite and needs
    no help, and a jitter added anyway would bias the projection in proportion to
    how far from coherent the forecast was. So the jitter is the fallback rather
    than the default: it exists for the node whose draws are constant, or whose
    residual variance came back at zero.
    """
    try:
        return np.asarray(np.linalg.cholesky(gram), dtype=np.float64)
    except np.linalg.LinAlgError:
        scale = max(float(np.mean(np.diag(gram))), 1.0)
        nudged = gram + np.eye(gram.shape[0]) * (_JITTER * scale)
        try:
            return np.asarray(np.linalg.cholesky(nudged), dtype=np.float64)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the MinT system A W A' is not positive definite even with a jitter, which "
                "means the weights collapsed: check for a node whose draws are constant or "
                "whose residual variance is zero"
            ) from exc


# -- the entry point -----------------------------------------------------------


def reconcile(
    result: ForecastResult,
    hierarchy: Hierarchy,
    *,
    method: str = "bottom_up",
    data: HierarchicalTSData | None = None,
    proportions: str | None = None,
    level: int | None = None,
    weights: str | None = None,
    trust: Mapping[str, float] | None = None,
    residual_var: Mapping[str, float] | None = None,
) -> ForecastResult:
    """Make the levels of ``result`` agree, and say how it was done.

    Works on any draws-shaped result whose nodes cover the tree, so a set of
    forecasts can be reconciled several ways without refitting anything, and the
    unreconciled version stays available through ``method="none"``.

    Parameters
    ----------
    result
        The unreconciled forecast: every node of the hierarchy, ``(S, H)`` draws
        each, all sharing one draw axis. Nodes outside the hierarchy (a driver,
        say) ride along untouched.
    hierarchy
        The tree. Sum aggregation only.
    method
        One of :data:`RECONCILE_METHODS`.
    data
        The training data, which ``top_down`` and ``middle_out`` read for the
        historical shares. Required for those two, refused for the others.
    proportions
        ``"historical"`` (the default) or ``"forecast"``, for the two strategies
        that split a parent downward.
    level
        Which depth ``middle_out`` believes, with the root at 0. Required there,
        refused elsewhere.
    weights
        ``"posterior"`` (the default) or ``"mint_shrink"``, for ``mint``.
    trust
        ``{node: weight}``, read by ``mint`` only. A node with weight 10 moves
        about a tenth as far as it otherwise would.
    residual_var
        In-sample one-step residual variance per node, which
        ``weights="mint_shrink"`` needs.

    Returns
    -------
    ForecastResult
        A new result carrying the reconciled draws, ``reconciliation`` naming the
        method and its options, and ``coherent`` set by measuring rather than by
        assuming.
    """
    if method not in RECONCILE_METHODS:
        raise ValueError(f"method must be one of {RECONCILE_METHODS}, got {method!r}")
    _require_sum(hierarchy)
    _require_coverage(result, hierarchy)
    _check_options(
        method,
        data=data,
        proportions=proportions,
        level=level,
        weights=weights,
        trust=trust,
        residual_var=residual_var,
    )

    unreconciled = {node: result.draws(node) for node in hierarchy.nodes}
    series: dict[str, Array] = {
        node: np.array(draws, copy=True) for node, draws in unreconciled.items()
    }
    rule = proportions or "historical"
    weight_rule = weights or "posterior"

    if method == "bottom_up":
        series = _bottom_up(series, hierarchy)
    elif method == "top_down":
        shares = historical_shares(_require_data(data, method), hierarchy)
        series = _allocate(series, unreconciled, hierarchy, _root_down(hierarchy), shares, rule)
    elif method == "middle_out":
        shares = historical_shares(_require_data(data, method), hierarchy)
        series = _middle_out(series, unreconciled, hierarchy, int(level or 0), shares, rule)
    elif method == "mint":
        nodes, diagonal = mint_weights(
            result,
            hierarchy,
            weights=weight_rule,
            residual_var=residual_var,
            trust=trust,
        )
        series = _mint(series, hierarchy, nodes, diagonal)

    merged = {node: series.get(node, result.draws(node)) for node in result.nodes}
    # Measured, not assumed. `none` changes nothing and still reports honestly,
    # which is how criterion 2 sees an incoherent set for what it is.
    error = coherence_gap(merged, hierarchy)
    return replace(
        result,
        series=merged,
        reconciliation=_provenance(method, rule, level, weight_rule, trust),
        coherent=bool(error <= COHERENCE_TOL),
    )


def _require_data(data: HierarchicalTSData | None, method: str) -> HierarchicalTSData:
    """``top_down`` and ``middle_out`` read the training window, so they need the data."""
    if data is None:
        raise ValueError(
            f"method={method!r} splits a parent between its children by their historical "
            "shares, which only the training window knows: pass data=the "
            "HierarchicalTSData the forecast was fitted on. (Forecast proportions need it "
            "too, for the sibling sets a draw leaves at zero.)"
        )
    return data


def _check_options(
    method: str,
    *,
    data: HierarchicalTSData | None,
    proportions: str | None,
    level: int | None,
    weights: str | None,
    trust: Mapping[str, float] | None,
    residual_var: Mapping[str, float] | None,
) -> None:
    """Refuse an option the chosen strategy does not read, rather than ignoring it.

    An ignored keyword is the quietest way to get a forecast that is not the one
    that was asked for: ``trust=`` on ``bottom_up`` looks like it did something.
    """
    allowed: dict[str, set[str]] = {
        "bottom_up": set(),
        "top_down": {"data", "proportions"},
        "middle_out": {"data", "proportions", "level"},
        "mint": {"weights", "trust", "residual_var"},
        "none": set(),
    }
    given = {
        "data": data is not None,
        "proportions": proportions is not None,
        "level": level is not None,
        "weights": weights is not None,
        "trust": trust is not None,
        "residual_var": residual_var is not None,
    }
    extra = sorted(
        name for name, present in given.items() if present and name not in allowed[method]
    )
    if extra:
        reads = sorted(allowed[method]) or "no options"
        raise ValueError(
            f"method={method!r} does not read {extra}; it reads {reads}. Leaving an option "
            "in place that the strategy ignores is how a forecast ends up quietly different "
            "from the one that was asked for."
        )
    if proportions is not None and proportions not in PROPORTION_RULES:
        raise ValueError(f"proportions must be one of {PROPORTION_RULES}, got {proportions!r}")
    if method == "middle_out" and level is None:
        raise ValueError(
            "method='middle_out' needs the level it should believe: level=0 is the root "
            "(which is top_down), level=1 the first split below it, and the deepest level "
            "is bottom_up"
        )
    if weights is not None and weights not in WEIGHT_RULES:
        raise ValueError(f"weights must be one of {WEIGHT_RULES}, got {weights!r}")
    if residual_var is not None and weights != "mint_shrink":
        raise ValueError(
            "residual_var is the input to weights='mint_shrink' and nothing else reads it; "
            "pass weights='mint_shrink' as well, or drop it"
        )


def _provenance(
    method: str,
    proportions: str,
    level: int | None,
    weights: str,
    trust: Mapping[str, float] | None,
) -> str:
    """The strategy and its options as one string, for the result to carry."""
    options: list[str] = []
    if method in ("top_down", "middle_out"):
        if method == "middle_out":
            options.append(f"level={level}")
        options.append(f"proportions={proportions}")
    if method == "mint":
        options.append(f"weights={weights}")
        if trust:
            pairs = ", ".join(f"{node}:{float(value):g}" for node, value in sorted(trust.items()))
            options.append(f"trust={{{pairs}}}")
    return f"{method}({', '.join(options)})" if options else method
