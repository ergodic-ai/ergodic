# SPDX-License-Identifier: Apache-2.0
"""Identification: turn a causal graph into an estimand.

Given a graph, a treatment, and an outcome, these functions answer two questions.
Can the interventional distribution P(Y | do(X)) be written in terms of the
observational distribution, and if so, how. The work is graph-only and
deterministic: no data, no statistics, no estimator choices yet.

Three strategies are covered, in the order ``identify_effect`` tries them:

1. Adjustment. The complete adjustment criterion (Perkovic et al., 2018), which
   generalizes Pearl's back-door criterion. If any valid adjustment set exists,
   the canonical set over ancestors is valid, so a single check settles it.
2. Front-door. A mediator set that carries all of the effect, with the back-door
   conditions on both legs.
3. Instruments. Conditional instrumental variables, for the cases adjustment and
   the front-door miss. Instrument identification is parametric: it pins the
   effect down under a linear model, where the other two strategies do not need
   one.

Before any of them, a trivial case is settled: when no directed path runs from
the treatment to the outcome, the intervention cannot move the outcome, so
P(y | do(x)) = P(y) and the strategy is ``"null"``. Latent nodes
(``graph.latent``) are unobserved, so they never appear in adjustment sets,
mediator sets, instruments, or conditioning sets.

The full do-calculus ID algorithm (which decides every identifiable case) is not
built yet. A graph with circle marks (a PAG) is rejected, since identification
there needs separate reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .graph import MixedGraph


class _PagNotSupportedError(NotImplementedError):
    """Identification was asked to run on a graph with circle marks."""


def _as_set(value: str | Iterable[str]) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


def _join(names: frozenset[str]) -> str:
    return ", ".join(sorted(names))


def _validate(
    graph: MixedGraph,
    x: frozenset[str],
    y: frozenset[str],
    other: frozenset[str] = frozenset(),
) -> None:
    """Check treatment and outcome are nonempty and disjoint, and every name is a node."""
    if not x:
        raise ValueError("treatment must name at least one node")
    if not y:
        raise ValueError("outcome must name at least one node")
    if x & y:
        raise ValueError(f"treatment and outcome must be disjoint; both contain {_join(x & y)}")
    unknown = (x | y | other) - graph.nodes
    if unknown:
        raise ValueError(f"unknown node names: {_join(unknown)}")


def _check_no_circle(graph: MixedGraph) -> None:
    if any(e.has_circle for e in graph.edges):
        raise _PagNotSupportedError(
            "identification needs definite reasoning for circle marks (PAGs); "
            "it is defined here for DAG and ADMG only"
        )


def _descendants(graph: MixedGraph, nodes: frozenset[str]) -> frozenset[str]:
    out: set[str] = set()
    for n in nodes:
        out |= graph.descendants(n)
    return frozenset(out)


def _ancestors(graph: MixedGraph, nodes: frozenset[str]) -> frozenset[str]:
    out: set[str] = set()
    for n in nodes:
        out |= graph.ancestors(n)
    return frozenset(out)


def _causal_nodes(graph: MixedGraph, x: frozenset[str], y: frozenset[str]) -> frozenset[str]:
    """Nodes on a proper causal path from ``x`` to ``y`` (the path's only x-node is its start).

    The leg toward ``y`` is traced with ``x`` removed from the graph, so a node
    whose only routes to ``y`` pass through a treatment node does not count.
    With a single treatment the two computations agree; the difference shows up
    for set treatments, where one treatment sits downstream of another.
    """
    reachable = _descendants(graph, x)
    without_x = graph.induced_subgraph((graph.nodes - x) | y)
    toward_y = _ancestors(without_x, y) | y
    return (reachable & toward_y) - x


def _forbidden(graph: MixedGraph, x: frozenset[str], y: frozenset[str]) -> frozenset[str]:
    """Nodes a valid adjustment set must avoid: x, the causal-path nodes, and their descendants."""
    cn = _causal_nodes(graph, x, y)
    return x | cn | _descendants(graph, cn)


def _remove_out(graph: MixedGraph, nodes: frozenset[str]) -> MixedGraph:
    """Drop every directed edge whose source is in ``nodes``."""
    out = graph
    for e in list(graph.edges):
        pair = e.directed_pair()
        if pair is not None and pair[0] in nodes:
            out = out.remove_edge(e.u, e.v)
    return out


def _proper_backdoor_graph(graph: MixedGraph, x: frozenset[str], cn: frozenset[str]) -> MixedGraph:
    """Remove the ``x -> m`` edges that start a proper causal path (``m`` in ``cn``)."""
    out = graph
    for e in list(graph.edges):
        pair = e.directed_pair()
        if pair is not None and pair[0] in x and pair[1] in cn:
            out = out.remove_edge(e.u, e.v)
    return out


# -- adjustment --------------------------------------------------------------


def is_valid_adjustment_set(
    graph: MixedGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    adjustment: str | Iterable[str],
) -> bool:
    """Return ``True`` if ``adjustment`` is a valid adjustment set for the effect.

    Uses the complete adjustment criterion: the set avoids the forbidden nodes
    (descendants of the causal-path nodes) and blocks every proper non-causal path
    from treatment to outcome. A set holding a latent node is never valid, since
    a latent node cannot be conditioned on.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    z = _as_set(adjustment)
    _validate(graph, x, y, z)
    _check_no_circle(graph)
    if (x & z) or (y & z) or (z & graph.latent):
        return False
    cn = _causal_nodes(graph, x, y)
    if z & _forbidden(graph, x, y):
        return False
    proper = _proper_backdoor_graph(graph, x, cn)
    return proper.m_separated(x, y, z)


def adjustment_set(
    graph: MixedGraph, treatment: str | Iterable[str], outcome: str | Iterable[str]
) -> frozenset[str] | None:
    """Return the canonical valid adjustment set, or ``None`` if none exists.

    The canonical set is the ancestors of treatment and outcome, minus the
    forbidden nodes and the latent nodes. If any valid adjustment set over the
    observed nodes exists, this one is valid.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    _validate(graph, x, y)
    candidate = (_ancestors(graph, x | y) | x | y) - _forbidden(graph, x, y) - x - y - graph.latent
    if is_valid_adjustment_set(graph, x, y, candidate):
        return frozenset(candidate)
    return None


def minimal_adjustment_set(
    graph: MixedGraph, treatment: str | Iterable[str], outcome: str | Iterable[str]
) -> frozenset[str] | None:
    """Return one minimal valid adjustment set, or ``None`` if none exists."""
    _validate(graph, _as_set(treatment), _as_set(outcome))
    start = adjustment_set(graph, treatment, outcome)
    if start is None:
        return None
    keep = set(start)
    changed = True
    while changed:
        changed = False
        for node in sorted(keep):
            if is_valid_adjustment_set(graph, treatment, outcome, keep - {node}):
                keep.discard(node)
                changed = True
    return frozenset(keep)


# -- front-door --------------------------------------------------------------


def is_frontdoor_set(
    graph: MixedGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    mediators: str | Iterable[str],
) -> bool:
    """Return ``True`` if ``mediators`` satisfies the front-door criterion.

    A mediator set holding a latent node is never valid, since the front-door
    formula conditions on the mediators.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    m = _as_set(mediators)
    _validate(graph, x, y, m)
    _check_no_circle(graph)
    if not m or (x & m) or (y & m) or (m & graph.latent):
        return False
    # (1) every directed path from x to y passes through m
    without_m = graph.induced_subgraph(graph.nodes - m)
    if _descendants(without_m, x) & y:
        return False
    # (2) no open back-door path from x to m
    if not _remove_out(graph, x).m_separated(x, m):
        return False
    # (3) treatment blocks every back-door path from m to y
    return _remove_out(graph, m).m_separated(m, y, x)


def frontdoor_set(
    graph: MixedGraph, treatment: str | Iterable[str], outcome: str | Iterable[str]
) -> frozenset[str] | None:
    """Return a front-door mediator set, or ``None`` if none is found.

    Tries the full set of observed nodes between treatment and outcome first,
    then single mediators. This is a search over likely candidates, not every
    subset.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    _validate(graph, x, y)
    mediators = _causal_nodes(graph, x, y) - y - x - graph.latent
    if mediators and is_frontdoor_set(graph, x, y, mediators):
        return frozenset(mediators)
    for single in sorted(mediators):
        if is_frontdoor_set(graph, x, y, {single}):
            return frozenset({single})
    return None


# -- instruments -------------------------------------------------------------


def is_instrument(
    graph: MixedGraph,
    instrument: str,
    treatment: str,
    outcome: str,
    given: str | Iterable[str] = frozenset(),
) -> bool:
    """Return ``True`` if ``instrument`` is a conditional instrument given ``given``.

    The instrument and the conditioning set are observed non-descendants of the
    treatment and the outcome, the instrument is associated with the treatment,
    and the instrument reaches the outcome only through the treatment (it
    separates from the outcome once the treatment's outgoing edges are cut).

    Descendants of the treatment are rejected outright: cutting the treatment's
    outgoing edges would sever the very edge that makes them endogenous, so the
    exclusion check alone would pass them.
    """
    i = frozenset({instrument})
    x = frozenset({treatment})
    y = frozenset({outcome})
    w = _as_set(given)
    _validate(graph, x, y, i | w)
    _check_no_circle(graph)
    if (i & x) or (i & y) or (w & (i | x | y)):
        return False
    if (i | w) & graph.latent:
        return False
    if (i | w) & _descendants(graph, x):
        return False
    if w & (_descendants(graph, y) | y):
        return False
    if graph.m_separated(i, x, w):  # relevance: must be associated
        return False
    return _remove_out(graph, x).m_separated(i, y, w)  # exclusion


def instruments(
    graph: MixedGraph, treatment: str, outcome: str
) -> list[tuple[str, frozenset[str]]]:
    """Find conditional instruments for ``treatment -> outcome``.

    Returns ``(instrument, conditioning set)`` pairs. Candidates and conditioning
    nodes are observed non-descendants of the treatment and the outcome; the
    search tries the empty set and single nodes as conditioning sets.
    """
    x = frozenset({treatment})
    y = frozenset({outcome})
    _validate(graph, x, y)
    usable = graph.nodes - graph.latent - _descendants(graph, x) - x - y
    non_descendant = sorted(usable - _descendants(graph, y))
    found: list[tuple[str, frozenset[str]]] = []
    for candidate in sorted(usable):
        condition_sets: list[frozenset[str]] = [frozenset()]
        condition_sets += [frozenset({w}) for w in non_descendant if w != candidate]
        for w in condition_sets:
            if is_instrument(graph, candidate, treatment, outcome, w):
                found.append((candidate, frozenset(w)))
                break
    return found


# -- orchestration -----------------------------------------------------------


@dataclass(frozen=True)
class Identification:
    """The result of trying to identify an effect from a graph."""

    treatment: frozenset[str]
    outcome: frozenset[str]
    identifiable: bool
    strategy: str  # "null", "adjustment", "frontdoor", "iv", or "none"
    adjustment_set: frozenset[str] | None = None
    mediators: frozenset[str] | None = None
    instruments: tuple[tuple[str, frozenset[str]], ...] = ()
    estimand: str = ""

    def __bool__(self) -> bool:
        return self.identifiable

    def __repr__(self) -> str:
        arrow = f"{sorted(self.treatment)} -> {sorted(self.outcome)}"
        if not self.identifiable:
            return f"Identification(not identifiable, {arrow})"
        return f"Identification({self.strategy}: {self.estimand})"


def _adjustment_formula(x: frozenset[str], y: frozenset[str], z: frozenset[str]) -> str:
    if not z:
        return f"P({_join(y)} | do({_join(x)})) = P({_join(y)} | {_join(x)})"
    return (
        f"P({_join(y)} | do({_join(x)})) = "
        f"sum_z P({_join(y)} | {_join(x)}, z) P(z), with z = {{{_join(z)}}}"
    )


def _frontdoor_formula(x: frozenset[str], y: frozenset[str], m: frozenset[str]) -> str:
    return (
        f"P({_join(y)} | do({_join(x)})) = "
        f"sum_m P(m | {_join(x)}) sum_x' P({_join(y)} | m, x') P(x'), "
        f"with m = {{{_join(m)}}}"
    )


def identify_effect(
    graph: MixedGraph, treatment: str | Iterable[str], outcome: str | Iterable[str]
) -> Identification:
    """Try adjustment, then the front-door, then instruments, and report the result.

    Before any strategy runs, the trivial case is settled: when no directed path
    runs from the treatment to the outcome, the intervention cannot move the
    outcome, so P(y | do(x)) = P(y) and the strategy is ``"null"``.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    _validate(graph, x, y)
    _check_no_circle(graph)

    if not (_descendants(graph, x) & y):
        return Identification(
            x,
            y,
            True,
            "null",
            estimand=(
                f"P({_join(y)} | do({_join(x)})) = P({_join(y)}); "
                "no directed path from treatment to outcome"
            ),
        )

    via_adjustment = adjustment_set(graph, x, y)
    if via_adjustment is not None:
        return Identification(
            x,
            y,
            True,
            "adjustment",
            adjustment_set=via_adjustment,
            estimand=_adjustment_formula(x, y, via_adjustment),
        )

    via_frontdoor = frontdoor_set(graph, x, y)
    if via_frontdoor is not None:
        return Identification(
            x,
            y,
            True,
            "frontdoor",
            mediators=via_frontdoor,
            estimand=_frontdoor_formula(x, y, via_frontdoor),
        )

    if len(x) == 1 and len(y) == 1:
        found = instruments(graph, next(iter(x)), next(iter(y)))
        if found:
            names = ", ".join(i for i, _ in found)
            return Identification(
                x,
                y,
                True,
                "iv",
                instruments=tuple(found),
                estimand=(
                    f"identified through instrument(s) {names} under a linear "
                    "(parametric) model; needs an IV estimator"
                ),
            )

    return Identification(
        x,
        y,
        False,
        "none",
        estimand="not identifiable by adjustment, front-door, or an instrument",
    )
