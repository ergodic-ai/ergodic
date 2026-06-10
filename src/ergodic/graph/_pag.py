# SPDX-License-Identifier: Apache-2.0
"""The PAG toolkit: latent projection, the FCI rules, and the MAG/PAG conversions.

A MAG over the observed nodes stands for a DAG with latent variables:
adjacency means no observed subset separates the pair (an inducing path), an
arrowhead at ``V`` on the edge to ``W`` says ``V`` is not an ancestor of
``W``, a tail says it is. A PAG carries the marks shared by every MAG in the
Markov equivalence class, circles where members disagree.

:func:`mag_of_dag` projects a DAG (or ADMG) onto its MAG over the observed
margin. :func:`fci_closure` applies Zhang's (2008) orientation rules R1 to R4
and R8 to R10 to a fixpoint, which is what FCI runs after its skeleton and
collider phases. :func:`pag_of_mag` builds the canonical PAG of a MAG (the
class's invariant marks), and :func:`mag_of_pag` extracts one member MAG back
out of a PAG, which together make ``is_pag`` a real validator by round trip,
exactly as the Meek machinery does for ``is_cpdag``.

Everything here assumes no selection bias: selection nodes raise, undirected
and circle-tail edges are rejected as selection artifacts, and rules R5 to R7
(which only fire under selection) are not implemented.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import TYPE_CHECKING

from ._graph import Edge, GraphKind, InvalidGraphError, MixedGraph
from ._marks import Mark
from ._meek import consistent_extension
from ._separation import m_separated

if TYPE_CHECKING:
    from collections.abc import Callable

# mark_at[a][b] is the mark at a on the edge joining a and b; adjacency never
# changes during closure, only mark values do.
_Marks = dict[str, dict[str, Mark]]


def mag_of_dag(graph: MixedGraph) -> MixedGraph:
    """Project a DAG or ADMG with latent nodes onto its MAG over the observed.

    Two observed nodes are adjacent exactly when no observed subset separates
    them, decided by the one witness set that suffices: the observed ancestors
    of either endpoint. Marks read ancestry in the full graph: a tail at ``X``
    on the edge to ``Y`` means ``X`` is an ancestor of ``Y`` there, an
    arrowhead means it is not. The result preserves every m-separation over
    the observed margin, which is the defining property of the projection.
    """
    if graph.selection:
        raise NotImplementedError(
            "mag_of_dag assumes no selection bias; graphs with selection nodes "
            "are not supported yet"
        )
    if not graph.is_admg():
        raise ValueError(
            "mag_of_dag needs a DAG or ADMG (directed and bidirected edges, no "
            "directed cycle); project the graph you have onto one first"
        )
    observed = sorted(graph.nodes - graph.latent)
    ancestors = {v: graph.ancestors(v) for v in graph.nodes}
    edges: list[Edge] = []
    for x, y in combinations(observed, 2):
        witness = ((ancestors[x] | ancestors[y]) & set(observed)) - {x, y}
        if m_separated(graph, x, y, witness):
            continue
        x_mark = Mark.TAIL if x in ancestors[y] else Mark.ARROW
        y_mark = Mark.TAIL if y in ancestors[x] else Mark.ARROW
        edges.append(Edge.make(x, x_mark, y, y_mark))
    out = MixedGraph(nodes=frozenset(observed), edges=frozenset(edges), kind=GraphKind.MAG)
    out.validate()
    return out


# -- the FCI orientation rules -------------------------------------------------


def _mark_maps(graph: MixedGraph) -> _Marks:
    marks: _Marks = {n: {} for n in graph.nodes}
    for e in graph.edges:
        marks[e.u][e.v] = e.u_mark
        marks[e.v][e.u] = e.v_mark
    return marks


def _rebuild(graph: MixedGraph, marks: _Marks) -> MixedGraph:
    edges = frozenset(Edge.make(e.u, marks[e.u][e.v], e.v, marks[e.v][e.u]) for e in graph.edges)
    return replace(graph, edges=edges, kind=GraphKind.MIXED)


def _set(marks: _Marks, a: str, b: str, mark: Mark) -> bool:
    """Set the mark at ``a`` on the edge to ``b``, but only over a circle."""
    if marks[a][b] is Mark.CIRCLE and marks[a][b] is not mark:
        marks[a][b] = mark
        return True
    return False


def _direct(marks: _Marks, b: str, c: str) -> bool:
    """Orient ``b -> c`` over circles: a tail at b, an arrowhead at c."""
    changed = _set(marks, b, c, Mark.TAIL)
    return _set(marks, c, b, Mark.ARROW) or changed


def _is_directed(marks: _Marks, a: str, b: str) -> bool:
    """``a -> b``: a tail at a and an arrowhead at b."""
    return marks[a][b] is Mark.TAIL and marks[b][a] is Mark.ARROW


def _possibly_directed(marks: _Marks, v: str, w: str) -> bool:
    """The edge could point from v to w: no arrowhead at v, no tail at w."""
    return marks[v][w] is not Mark.ARROW and marks[w][v] is not Mark.TAIL


def _r1(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a *-> b o-* c, a and c non-adjacent: orient b -> c."""
    changed = False
    for b in sorted(marks):
        for a in adj[b]:
            if marks[b][a] is not Mark.ARROW:
                continue
            for c in adj[b]:
                if c == a or c in marks[a]:
                    continue
                if marks[b][c] is Mark.CIRCLE:
                    changed |= _direct(marks, b, c)
    return changed


def _r2(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a -> b *-> c or a *-> b -> c, with a *-o c: an arrowhead at c."""
    changed = False
    for b in sorted(marks):
        for a in adj[b]:
            for c in adj[b]:
                if c == a or c not in marks[a] or marks[c][a] is not Mark.CIRCLE:
                    continue
                first = _is_directed(marks, a, b) and marks[c][b] is Mark.ARROW
                second = marks[b][a] is Mark.ARROW and _is_directed(marks, b, c)
                if first or second:
                    changed |= _set(marks, c, a, Mark.ARROW)
    return changed


def _r3(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a *-> b <-* c, a *-o d o-* c, a and c non-adjacent, d *-o b: arrowhead at b on (d, b)."""
    changed = False
    for b in sorted(marks):
        arrows = [a for a in adj[b] if marks[b][a] is Mark.ARROW]
        for a, c in combinations(arrows, 2):
            if c in marks[a]:
                continue
            for d in adj[b]:
                if d in (a, c) or marks[b][d] is not Mark.CIRCLE:
                    continue
                if d not in marks[a] or d not in marks[c]:
                    continue
                if marks[d][a] is Mark.CIRCLE and marks[d][c] is Mark.CIRCLE:
                    changed |= _set(marks, b, d, Mark.ARROW)
    return changed


def _discriminating_ends(
    marks: _Marks, adj: dict[str, list[str]], b: str, c: str
) -> list[tuple[str, str]]:
    """Every ``(d, a)`` closing a discriminating path ``<d, ..., a, b, c>`` for b.

    Walking backward from b: each vertex passed becomes interior, so it must
    be a collider on the path (arrowheads at it on both path edges) and a
    parent of c. A predecessor not adjacent to c closes a path as d. States
    carry the first interior vertex a (b's neighbour on the path), which R4
    orients when b turns out to be a collider.
    """
    found: list[tuple[str, str]] = []
    starts = [
        a
        for a in adj[b]
        if a != c and marks[a][b] is Mark.ARROW and a in marks[c] and _is_directed(marks, a, c)
    ]
    seen: set[tuple[str, str, str]] = set()
    queue: list[tuple[str, str, str]] = [(a, b, a) for a in sorted(starts)]
    while queue:
        head, succ, first = queue.pop(0)
        for p in adj[head]:
            if p in (succ, b, c) or marks[head][p] is not Mark.ARROW:
                continue
            if p not in marks[c]:
                found.append((p, first))
                continue
            if _is_directed(marks, p, c) and marks[p][head] is Mark.ARROW:
                state = (p, head, first)
                if state not in seen:
                    seen.add(state)
                    queue.append(state)
    return found


def _r4(
    marks: _Marks,
    adj: dict[str, list[str]],
    discriminating: Callable[[str, str, str, str], bool | None],
) -> bool:
    """The discriminating-path rule, with the collider question delegated.

    ``discriminating(d, a, b, c)`` decides whether b is a collider on the
    path: FCI answers it from its separating sets, pag_of_mag from the MAG's
    own marks. ``None`` means unknown, and that path is skipped.
    """
    changed = False
    for c in sorted(marks):
        for b in adj[c]:
            if marks[b][c] is not Mark.CIRCLE:
                continue
            for d, a in _discriminating_ends(marks, adj, b, c):
                verdict = discriminating(d, a, b, c)
                if verdict is None:
                    continue
                if verdict:
                    changed |= _set(marks, a, b, Mark.ARROW)
                    changed |= _set(marks, b, a, Mark.ARROW)
                    changed |= _set(marks, b, c, Mark.ARROW)
                    changed |= _set(marks, c, b, Mark.ARROW)
                else:
                    changed |= _direct(marks, b, c)
                break
    return changed


def _r8(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a -> b -> c or a -o b -> c, with a o-> c: a tail at a."""
    changed = False
    for c in sorted(marks):
        for a in adj[c]:
            if not (marks[a][c] is Mark.CIRCLE and marks[c][a] is Mark.ARROW):
                continue
            for b in adj[a]:
                if b == c or b not in marks[c] or not _is_directed(marks, b, c):
                    continue
                into_b = _is_directed(marks, a, b)
                circle_b = marks[a][b] is Mark.TAIL and marks[b][a] is Mark.CIRCLE
                if into_b or circle_b:
                    changed |= _set(marks, a, c, Mark.TAIL)
                    break
    return changed


def _uncovered_pd_reaches(
    marks: _Marks, adj: dict[str, list[str]], start: str, first: str, goal: str
) -> bool:
    """A simple uncovered possibly-directed path ``start - first - ... - goal``.

    Every edge points possibly toward the goal end and every consecutive
    triple is unshielded. Depth-first over simple paths: uncovered-ness is a
    property of the realized path, so plain state reachability would
    overclaim after loop erasure.
    """
    if not _possibly_directed(marks, start, first):
        return False
    if first == goal:
        return True

    def extend(prev: str, cur: str, used: frozenset[str]) -> bool:
        for nxt in adj[cur]:
            if nxt in used or nxt in marks[prev]:
                continue
            if not _possibly_directed(marks, cur, nxt):
                continue
            if nxt == goal or extend(cur, nxt, used | {nxt}):
                return True
        return False

    return extend(start, first, frozenset({start, first}))


def _r9(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a o-> c with an uncovered possibly-directed path <a, mu, ..., c>, mu not adjacent c."""
    changed = False
    for c in sorted(marks):
        for a in adj[c]:
            if not (marks[a][c] is Mark.CIRCLE and marks[c][a] is Mark.ARROW):
                continue
            for mu in adj[a]:
                if mu == c or mu in marks[c]:
                    continue
                if _uncovered_pd_reaches(marks, adj, a, mu, c):
                    changed |= _set(marks, a, c, Mark.TAIL)
                    break
    return changed


def _r10(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a o-> c fed by two uncovered possibly-directed paths with unshielded first steps."""
    changed = False
    for c in sorted(marks):
        parents = [b for b in adj[c] if _is_directed(marks, b, c)]
        if len(parents) < 2:
            continue
        for a in adj[c]:
            if not (marks[a][c] is Mark.CIRCLE and marks[c][a] is Mark.ARROW):
                continue
            if _r10_fires(marks, adj, a, parents):
                changed |= _set(marks, a, c, Mark.TAIL)
    return changed


def _r10_fires(marks: _Marks, adj: dict[str, list[str]], a: str, parents: list[str]) -> bool:
    for b, d in combinations(parents, 2):
        for mu in adj[a]:
            for omega in adj[a]:
                if mu == omega or mu in marks[omega] or omega in marks[mu]:
                    continue
                if _uncovered_pd_reaches(marks, adj, a, mu, b) and _uncovered_pd_reaches(
                    marks, adj, a, omega, d
                ):
                    return True
                if _uncovered_pd_reaches(marks, adj, a, mu, d) and _uncovered_pd_reaches(
                    marks, adj, a, omega, b
                ):
                    return True
    return False


def fci_closure(
    graph: MixedGraph,
    *,
    discriminating: Callable[[str, str, str, str], bool | None],
) -> MixedGraph:
    """Close a partially oriented graph under Zhang's rules R1 to R4 and R8 to R10.

    The input carries its v-structure arrowheads (and any knowledge-seeded
    marks); circles elsewhere. Rules only ever turn a circle into a tail or
    an arrowhead: a mark that is already definite never moves, so a seeded
    mark that disagrees with a rule simply stays and the conflict surfaces at
    the caller's validation. ``discriminating(d, a, b, c)`` answers R4's
    question, whether b is a collider on the discriminating path; ``None``
    skips that path. Sweeps run in sorted order until a full pass changes
    nothing, so the closure is deterministic. Rules R5 to R7 only fire under
    selection bias and are not implemented. Returns kind ``mixed``.
    """
    marks = _mark_maps(graph)
    adj = {n: sorted(marks[n]) for n in marks}
    changed = True
    while changed:
        changed = False
        changed |= _r1(marks, adj)
        changed |= _r2(marks, adj)
        changed |= _r3(marks, adj)
        changed |= _r4(marks, adj, discriminating)
        changed |= _r8(marks, adj)
        changed |= _r9(marks, adj)
        changed |= _r10(marks, adj)
    return _rebuild(graph, marks)


# -- the conversions -----------------------------------------------------------


def mag_of_pag(p: MixedGraph) -> MixedGraph:
    """Extract one member MAG from a PAG (Zhang 2008).

    Every ``o->`` edge becomes directed into its arrowhead, and the
    circle-circle component is oriented into a DAG that adds no unshielded
    collider (the Dor-Tarsi extension of its undirected skeleton; for a valid
    complete PAG that component is chordal, so the extension exists).
    Undirected and circle-tail edges are selection artifacts and raise, as
    does an inextricable circle component or a result that fails the MAG
    rules. Maximality of the member is not checked here because ``is_mag``
    does not check it; the ``is_pag`` round trip is what guards it.
    """
    fixed: list[Edge] = []
    circle_pairs: list[tuple[str, str]] = []
    for e in sorted(p.edges, key=lambda e: (e.u, e.v)):
        cu, cv = e.u_mark, e.v_mark
        if cu is Mark.CIRCLE and cv is Mark.CIRCLE:
            circle_pairs.append((e.u, e.v))
        elif cu is Mark.CIRCLE and cv is Mark.ARROW:
            fixed.append(Edge.make(e.u, Mark.TAIL, e.v, Mark.ARROW))
        elif cu is Mark.ARROW and cv is Mark.CIRCLE:
            fixed.append(Edge.make(e.u, Mark.ARROW, e.v, Mark.TAIL))
        elif e.is_undirected or Mark.CIRCLE in (cu, cv):
            raise InvalidGraphError(
                f"edge {e.glyph()!r} only arises under selection bias, which the "
                "PAG toolkit does not support yet"
            )
        else:
            fixed.append(e)
    if circle_pairs:
        component_nodes = frozenset(n for pair in circle_pairs for n in pair)
        undirected = MixedGraph(
            nodes=component_nodes,
            edges=frozenset(Edge.make(u, Mark.TAIL, v, Mark.TAIL) for u, v in circle_pairs),
            kind=GraphKind.MIXED,
        )
        extension = consistent_extension(undirected)
        if extension is None:
            raise InvalidGraphError(
                "the circle component of this graph is not extendable without new "
                "colliders, so it is not a valid PAG"
            )
        fixed.extend(extension.edges)
    mag = MixedGraph(nodes=p.nodes, edges=frozenset(fixed), kind=GraphKind.MAG)
    if not mag.is_mag():
        raise InvalidGraphError("the extracted member violates the MAG rules")
    return mag


def pag_of_mag(m: MixedGraph) -> MixedGraph:
    """The canonical PAG of a MAG: the marks every Markov-equivalent MAG shares.

    Same skeleton, every mark a circle; unshielded colliders copied from the
    MAG (they are class invariants); then the rule closure, with R4's
    discriminating-path question answered by the MAG's own marks. Zhang
    (2008) proved this rule set complete without selection bias, so the
    result carries exactly the invariant marks. The input should be maximal;
    ``is_mag`` cannot check that, and a non-maximal input gives a PAG for the
    wrong class.
    """
    if not m.is_mag():
        raise ValueError("pag_of_mag needs a MAG; project or validate the graph first")
    marks: _Marks = {n: {} for n in m.nodes}
    for e in m.edges:
        marks[e.u][e.v] = Mark.CIRCLE
        marks[e.v][e.u] = Mark.CIRCLE
    adjacency = {n: sorted(marks[n]) for n in marks}
    for z in sorted(m.nodes):
        for x, y in combinations(adjacency[z], 2):
            if y in marks[x]:
                continue  # shielded
            if m.mark_at(x, z, z) is Mark.ARROW and m.mark_at(y, z, z) is Mark.ARROW:
                marks[z][x] = Mark.ARROW
                marks[z][y] = Mark.ARROW

    def from_mag(d: str, a: str, b: str, c: str) -> bool | None:
        return m.mark_at(a, b, b) is Mark.ARROW and m.mark_at(b, c, b) is Mark.ARROW

    skeleton = MixedGraph(
        nodes=m.nodes,
        edges=frozenset(Edge.make(e.u, marks[e.u][e.v], e.v, marks[e.v][e.u]) for e in m.edges),
        kind=GraphKind.MIXED,
    )
    closed = fci_closure(skeleton, discriminating=from_mag)
    # replace, not as_kind: validating PAG kind would re-enter this function
    # through the is_pag round trip.
    return replace(closed, kind=GraphKind.PAG)
