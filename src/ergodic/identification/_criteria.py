# SPDX-License-Identifier: Apache-2.0
"""The named identification criteria: adjustment, front-door, and instruments.

Given a graph, a treatment, and an outcome, these functions answer two questions.
Can the interventional distribution P(Y | do(X)) be written in terms of the
observational distribution, and if so, how. The work is graph-only and
deterministic: no data, no statistics, no estimator choices yet.

The graph's face decides what a query means. A DAG or ADMG is a single causal
world. A CPDAG, MAG, or PAG (what discovery returns) is an equivalence class,
and an answer must hold in *every* graph the class represents: one estimand
that covers the whole class, however the undecided marks resolve. A mixed
graph raises, since it certifies neither reading.

Three strategies are covered, in the order ``identify_effect`` tries them:

1. Adjustment. The complete adjustment criterion (Perkovic et al., 2018), which
   generalizes Pearl's back-door criterion. If any valid adjustment set exists,
   the canonical set over ancestors is valid, so a single check settles it. On
   a class the same criterion runs in its generalized form: possible ancestry
   replaces ancestry, the class must be amenable (every possibly causal path
   out of the treatment starts with a visible directed edge), and blocking is
   read through definite-status paths.
2. Front-door. A mediator set that carries all of the effect, with the back-door
   conditions on both legs. Single graphs only; on a class it raises, since the
   criterion is not class-invariant. Enumerate members instead
   (``dags_of_cpdag``, ``mags_of_pag``).
3. Instruments. Conditional instrumental variables, for the cases adjustment and
   the front-door miss. Instrument identification is parametric: it pins the
   effect down under a linear model, where the other two strategies do not need
   one. Single graphs only, like the front-door.

Before any of them, a trivial case is settled: when no directed path (on a
class: no possibly causal path, in any member) runs from the treatment to the
outcome, the intervention cannot move the outcome, so P(y | do(x)) = P(y) and
the strategy is ``"null"``. Latent nodes (``graph.latent``) are unobserved, so
they never appear in adjustment sets, mediator sets, instruments, or
conditioning sets; class faces carry no latent designations at all.

When the named ladder misses on a single graph, the general do-calculus ID
algorithm runs as the fallthrough (in :mod:`._distribution`): it decides every
remaining case, returning the typed functional (``strategy="id"``, or a
recognized named reduction) or the hedge that proves non-identifiability. The
raw ID/IDC engine is exposed directly through ``identify_distribution``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..graph import (
    GraphKind,
    Mark,
    consistent_extension,
    possible_ancestors,
    possible_descendants,
    visible_edges,
)
from ..graph._surgery import _cut_outgoing
from ._record import Identification

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..graph import MixedGraph


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
    *,
    allow_empty_treatment: bool = False,
) -> None:
    """Check treatment and outcome are nonempty and disjoint, and every name is a node.

    ``allow_empty_treatment`` lets the do-set be empty, for ``identify_distribution``:
    ``P(y | do(), z)`` is the legitimate observational query ``P(y | z)``, which the
    ID engine handles. The named-criteria callers leave it ``False``.
    """
    if not x and not allow_empty_treatment:
        raise ValueError("treatment must name at least one node")
    if not y:
        raise ValueError("outcome must name at least one node")
    if x & y:
        raise ValueError(f"treatment and outcome must be disjoint; both contain {_join(x & y)}")
    unknown = (x | y | other) - graph.nodes
    if unknown:
        raise ValueError(f"unknown node names: {_join(unknown)}")


_CLASS_KINDS = (GraphKind.CPDAG, GraphKind.MAG, GraphKind.PAG)


def _mixed_face_hint(graph: MixedGraph) -> str:
    """Say which face a ``mixed`` graph fits, if any, and how to re-face it.

    Only advice: the graph is still refused. The checks follow the marks so a
    partially directed graph is never pointed at the MAG or PAG face, where an
    undirected edge would mean selection bias rather than an undecided
    orientation. The face a graph should take depends on where it came from,
    which the graph does not record, so a graph that fits both a single-world
    face and the MAG face gets both readings, each with its condition: an FCI
    or RFCI result means ancestry, and its directed edges can hide
    confounding, so re-facing it as a DAG or ADMG would be unsound.
    """
    edges = graph.edges
    mag_reading = (
        "; if it came from FCI or RFCI (or otherwise means ancestry, where a "
        "directed edge can hide confounding), re-face it with graph.as_kind('mag') "
        "instead, which identification reads as an equivalence class"
    )
    if graph.is_dag():
        return (
            "every edge is directed and there is no directed cycle. If it came "
            "from a search that assumes no hidden confounding (PC or GES with "
            "domain knowledge orienting every edge, say) and you read it as one "
            "causal world, re-face it with graph.as_kind('dag')" + mag_reading
        )
    if all(e.is_directed or e.is_undirected for e in edges):
        if graph.is_cpdag():
            return (
                "it satisfies the cpdag rules, so if it stands for a Markov "
                "equivalence class, re-face it with graph.as_kind('cpdag')"
            )
        undirected = sorted(e.glyph() for e in edges if e.is_undirected)
        if consistent_extension(graph) is None:
            return (
                "no DAG has its directed edges and v-structures (a directed "
                "cycle, or orientations that clash), so the orientations and "
                "the data disagree; check the domain knowledge or rerun discovery"
            )
        count = "1 edge is" if len(undirected) == 1 else f"{len(undirected)} edges are"
        return (
            f"it is partially directed ({count} undirected, such as "
            f"{undirected[0]!r}) and is neither a dag nor a cpdag, which is what "
            "a search such as PC can return when domain knowledge orients edges "
            "the data leaves open. Orient the rest with more knowledge "
            "(DomainKnowledge.require) and re-face with graph.as_kind('dag'), "
            "take one member with ergodic.graph.consistent_extension(graph) (a "
            "dag; the answer then holds for that member only), or rediscover "
            "without the orienting knowledge, which usually keeps the cpdag face"
        )
    if graph.is_admg():
        single = (
            "its edges are directed or bidirected with no directed cycle. If you "
            "read it as one causal world whose only hidden confounding is the "
            "drawn <-> edges, re-face it with graph.as_kind('admg')"
        )
        return single + (mag_reading if graph.is_mag() else "")
    for kind in (GraphKind.PAG, GraphKind.MAG):
        try:
            graph.validate(kind)
        except ValueError:
            continue
        return (
            f"it satisfies the {kind.value} rules, so if it stands for an "
            f"equivalence class, re-face it with graph.as_kind('{kind.value}')"
        )
    return (
        "its marks satisfy no face's rules. If the search ran with domain "
        "knowledge, the marks the knowledge seeded are the likely cause (FCI and "
        "RFCI keep them even where the class disagrees, and report "
        "diagnostics['pag_valid'] at 0.0): rerun without that knowledge, which "
        "usually returns the class face. Otherwise the data and the orientation "
        "rules disagree: rerun with more data or another alpha, or write the "
        "graph you believe with dag(), admg(), or pag()"
    )


def _is_class_kind(graph: MixedGraph) -> bool:
    """Dispatch on the graph's face: single causal world or equivalence class.

    DAG and ADMG reason about one graph. CPDAG, MAG, and PAG reason about
    every graph the class represents, through the generalized criterion.
    MIXED raises: identification needs a verified face, and an FCI fallback
    output (``pag_valid`` 0.0) stays mixed on purpose because it is not a
    class object.
    """
    if graph.kind is GraphKind.MIXED:
        raise ValueError(
            "identification needs a verified graph face (dag, admg, mag, cpdag, "
            f"or pag), not 'mixed'; {_mixed_face_hint(graph)}"
        )
    if graph.kind not in _CLASS_KINDS:
        return False
    if graph.latent or graph.selection:
        raise ValueError(
            f"a {graph.kind.value} carries no latent or selection designations; "
            "the class semantics already account for what is unobserved"
        )
    if graph.kind in (GraphKind.MAG, GraphKind.PAG):
        for e in graph.edges:
            marks = {e.u_mark, e.v_mark}
            if e.is_undirected or marks == {Mark.TAIL, Mark.CIRCLE}:
                raise ValueError(
                    f"edge {e.glyph()!r} says the class carries selection bias, "
                    "and the generalized adjustment criterion assumes there is "
                    "none; causal effects are generally not identifiable from a "
                    "selection-biased margin"
                )
    return True


def _check_single_graph(graph: MixedGraph, what: str) -> None:
    if _is_class_kind(graph):
        raise NotImplementedError(
            f"{what} is not defined on an equivalence class ({graph.kind.value}); "
            "extract a member (consistent_extension, mag_of_pag) or enumerate "
            "the class (dags_of_cpdag, mags_of_pag) and identify per member"
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


def _proper_backdoor_graph(graph: MixedGraph, x: frozenset[str], cn: frozenset[str]) -> MixedGraph:
    """Remove the ``x -> m`` edges that start a proper causal path (``m`` in ``cn``)."""
    out = graph
    for e in list(graph.edges):
        pair = e.directed_pair()
        if pair is not None and pair[0] in x and pair[1] in cn:
            out = out.remove_edge(e.u, e.v)
    return out


# -- the class (equivalence-set) analogs ---------------------------------------


def _possible_causal_nodes(
    graph: MixedGraph, x: frozenset[str], y: frozenset[str]
) -> frozenset[str]:
    """Nodes on a proper possibly causal path from ``x`` to ``y``.

    The class analog of :func:`_causal_nodes`: possible reachability replaces
    reachability, so a node counts when *some* member of the class puts it on
    a proper causal path.
    """
    reachable = possible_descendants(graph, x)
    without_x = graph.induced_subgraph((graph.nodes - x) | y)
    toward_y = possible_ancestors(without_x, y) | y
    return (reachable & toward_y) - x


def _possible_forbidden(graph: MixedGraph, x: frozenset[str], y: frozenset[str]) -> frozenset[str]:
    """The class analog of :func:`_forbidden`, over possible descendants."""
    cn = _possible_causal_nodes(graph, x, y)
    return x | cn | possible_descendants(graph, cn)


def _amenability_violations(graph: MixedGraph, x: frozenset[str], cn: frozenset[str]) -> list[str]:
    """First edges of proper possibly causal paths that are not visible directed.

    Amenability (Perkovic et al., 2018) demands every proper possibly causal
    path from the treatment to the outcome start with a visible directed edge
    out of the treatment; otherwise no set adjusts uniformly over the class.
    On a CPDAG every directed edge is visible (no latents under causal
    sufficiency); on a MAG or PAG visibility is Zhang's certificate. Returns
    the offending edges as glyphs, empty when amenable.
    """
    visible = visible_edges(graph) if graph.kind in (GraphKind.MAG, GraphKind.PAG) else None
    bad: list[str] = []
    for e in sorted(graph.edges, key=lambda e: (e.u, e.v)):
        for x0 in (e.u, e.v):
            if x0 not in x:
                continue
            v = e.other(x0)
            if v in x or v not in cn or e.mark_at(x0) is Mark.ARROW:
                continue
            pair = e.directed_pair()
            if pair == (x0, v) and (visible is None or pair in visible):
                continue
            if e.glyph() not in bad:
                bad.append(e.glyph())
    return bad


# -- adjustment --------------------------------------------------------------


def is_valid_adjustment_set(
    graph: MixedGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    adjustment: str | Iterable[str],
) -> bool:
    """Return ``True`` if ``adjustment`` is a valid adjustment set for the effect.

    On a DAG or ADMG, the complete adjustment criterion: the set avoids the
    forbidden nodes (descendants of the causal-path nodes) and blocks every
    proper non-causal path from treatment to outcome. A set holding a latent
    node is never valid, since a latent node cannot be conditioned on.

    On a CPDAG, MAG, or PAG, the generalized criterion (Perkovic et al.,
    2018): the class must be amenable, the set avoids the possible
    descendants of the possibly-causal-path nodes, and it blocks every
    proper definite-status non-causal path. ``True`` then means the set is
    valid in *every* graph the class represents.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    z = _as_set(adjustment)
    _validate(graph, x, y, z)
    if _is_class_kind(graph):
        if (x & z) or (y & z):
            return False
        cn = _possible_causal_nodes(graph, x, y)
        if _amenability_violations(graph, x, cn):
            return False
        if z & _possible_forbidden(graph, x, y):
            return False
        return _proper_backdoor_graph(graph, x, cn).m_separated(x, y, z)
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

    The canonical set is the ancestors of treatment and outcome (possible
    ancestors, on a class face), minus the forbidden nodes and the latent
    nodes. If any valid adjustment set over the observed nodes exists, this
    one is valid; on a class, validity means valid in every member.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    _validate(graph, x, y)
    if _is_class_kind(graph):
        candidate = possible_ancestors(graph, x | y) - _possible_forbidden(graph, x, y) - x - y
    else:
        candidate = (
            (_ancestors(graph, x | y) | x | y) - _forbidden(graph, x, y) - x - y - graph.latent
        )
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
    _check_single_graph(graph, "the front-door criterion")
    if not m or (x & m) or (y & m) or (m & graph.latent):
        return False
    # (1) every directed path from x to y passes through m
    without_m = graph.induced_subgraph(graph.nodes - m)
    if _descendants(without_m, x) & y:
        return False
    # (2) no open back-door path from x to m
    if not _cut_outgoing(graph, x).m_separated(x, m):
        return False
    # (3) treatment blocks every back-door path from m to y
    return _cut_outgoing(graph, m).m_separated(m, y, x)


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
    _check_single_graph(graph, "the front-door criterion")
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
    _check_single_graph(graph, "instrument identification")
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
    return _cut_outgoing(graph, x).m_separated(i, y, w)  # exclusion


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
    _check_single_graph(graph, "instrument identification")
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

    Before any strategy runs, the trivial case is settled: when no directed
    path runs from the treatment to the outcome, the intervention cannot move
    the outcome, so P(y | do(x)) = P(y) and the strategy is ``"null"``.

    On a class face (CPDAG, MAG, PAG) only the null and adjustment strategies
    apply, and a success holds in every graph the class represents: the null
    when no member has a causal path, adjustment through the generalized
    criterion. The front-door and instruments need a single graph, so a class
    that adjustment cannot identify reports not-identifiable and says whether
    amenability failed or no uniform set exists.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    _validate(graph, x, y)
    is_class = _is_class_kind(graph)

    if is_class:
        no_path = not (possible_descendants(graph, x) & y)
        reason = "no possibly causal path from treatment to outcome in any member of the class"
    else:
        no_path = not (_descendants(graph, x) & y)
        reason = "no directed path from treatment to outcome"
    if no_path:
        return Identification(
            x,
            y,
            True,
            "null",
            estimand=f"P({_join(y)} | do({_join(x)})) = P({_join(y)}); {reason}",
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

    if is_class:
        violations = _amenability_violations(graph, x, _possible_causal_nodes(graph, x, y))
        if violations:
            detail = (
                f"not amenable: the possibly causal path leaving the treatment "
                f"through {violations[0]!r} does not start with a visible directed "
                "edge, so members of the class disagree about it"
            )
        else:
            detail = "no adjustment set is valid in every member of the class"
        return Identification(
            x,
            y,
            False,
            "none",
            estimand=(
                f"{detail}; the front-door and instruments need a single graph "
                "(enumerate members with dags_of_cpdag or mags_of_pag)"
            ),
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

    # The named ladder missed. On a single graph the general do-calculus ID
    # algorithm decides every remaining case: it identifies (returning the
    # functional, or a recognized named reduction) or exhibits the hedge that
    # proves non-identifiability. The import is function-local to avoid a load
    # cycle (the ID engine and recognition import this module's criteria).
    from ._distribution import id_fallthrough

    return id_fallthrough(graph, x, y)
