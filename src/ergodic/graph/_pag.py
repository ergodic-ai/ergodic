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

Selection bias is in scope: :func:`mag_of_dag` projects selection nodes the
way it projects latents (an undirected edge says both endpoints are ancestors
of selection), the closure runs Zhang's full rule set R1 to R10 (R5 to R7 are
the selection rules), and :func:`mag_of_pag` resolves circle-tail edges per
his arrowhead augmentation. The completeness of the ten rules is held
executably by the same round trips that hold the no-selection case.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, pairwise, product
from typing import TYPE_CHECKING

from ._graph import Edge, GraphKind, InvalidGraphError, MixedGraph
from ._marks import Mark
from ._meek import consistent_extension
from ._separation import m_separated

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# mark_at[a][b] is the mark at a on the edge joining a and b; adjacency never
# changes during closure, only mark values do.
_Marks = dict[str, dict[str, Mark]]


def mag_of_dag(graph: MixedGraph) -> MixedGraph:
    """Project a DAG or ADMG with latent and selection nodes onto its MAG.

    Two observed nodes are adjacent exactly when no observed subset separates
    them given the selection nodes, decided by the one witness set that
    suffices (Richardson and Spirtes 2002, Theorem 4.2): the observed
    ancestors of either endpoint or of selection, plus the selection nodes
    themselves. Marks read ancestry in the full graph: a tail at ``X`` on the
    edge to ``Y`` means ``X`` is an ancestor of ``Y`` or of selection there,
    an arrowhead means it is neither; two tails make an undirected edge,
    which on a DAG means both endpoints are ancestors of selection. The
    result preserves every m-separation over the observed margin conditional
    on selection, which is the defining property of the projection.
    """
    if not graph.is_admg():
        raise ValueError(
            "mag_of_dag needs a DAG or ADMG (directed and bidirected edges, no "
            "directed cycle); project the graph you have onto one first"
        )
    stray = graph.selection - graph.nodes
    if stray:
        raise ValueError(f"selection names not in the graph: {', '.join(sorted(stray))}")
    overlap = graph.selection & graph.latent
    if overlap:
        raise ValueError(f"nodes cannot be both latent and selection: {', '.join(sorted(overlap))}")
    selection = set(graph.selection)
    observed = sorted(graph.nodes - graph.latent - graph.selection)
    ancestors = {v: graph.ancestors(v) for v in graph.nodes}
    sel_closure = selection | {a for s in selection for a in ancestors[s]}
    edges: list[Edge] = []
    for x, y in combinations(observed, 2):
        pool = (ancestors[x] | ancestors[y] | sel_closure) & (set(observed) | selection)
        if m_separated(graph, x, y, pool - {x, y}):
            continue
        x_mark = Mark.TAIL if (x in ancestors[y] or x in sel_closure) else Mark.ARROW
        y_mark = Mark.TAIL if (y in ancestors[x] or y in sel_closure) else Mark.ARROW
        edges.append(Edge.make(x, x_mark, y, y_mark))
    out = MixedGraph(nodes=frozenset(observed), edges=frozenset(edges), kind=GraphKind.MAG)
    out.validate()
    return out


# -- possible ancestry and visibility --------------------------------------------


def _seed_names(graph: MixedGraph, nodes: str | Iterable[str]) -> frozenset[str]:
    seeds = frozenset({nodes}) if isinstance(nodes, str) else frozenset(nodes)
    unknown = seeds - graph.nodes
    if unknown:
        raise ValueError(f"unknown node names: {', '.join(sorted(unknown))}")
    return seeds


def possible_descendants(graph: MixedGraph, nodes: str | Iterable[str]) -> frozenset[str]:
    """Nodes reachable from ``nodes`` along a possibly directed path.

    A path is possibly directed when no edge on it carries an arrowhead at
    the endpoint nearer the start, so some member of the graph's class can
    point every edge away from the start. On a DAG or ADMG this is plain
    ``descendants``. Each seed is excluded from its own walk, mirroring
    :meth:`MixedGraph.descendants`.
    """
    seeds = _seed_names(graph, nodes)
    marks = _mark_maps(graph)
    out: set[str] = set()
    for seed in sorted(seeds):
        seen = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nxt, mark in marks[cur].items():
                if nxt not in seen and mark is not Mark.ARROW:
                    seen.add(nxt)
                    stack.append(nxt)
        out |= seen - {seed}
    return frozenset(out)


def possible_ancestors(graph: MixedGraph, nodes: str | Iterable[str]) -> frozenset[str]:
    """Nodes with a possibly directed path into ``nodes``.

    The mirror of :func:`possible_descendants`: ``w`` counts when some
    member of the class can orient a path from ``w`` all the way into a
    seed. Each seed is excluded from its own walk.
    """
    seeds = _seed_names(graph, nodes)
    marks = _mark_maps(graph)
    out: set[str] = set()
    for seed in sorted(seeds):
        seen = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nxt in marks[cur]:
                if nxt not in seen and marks[nxt][cur] is not Mark.ARROW:
                    seen.add(nxt)
                    stack.append(nxt)
        out |= seen - {seed}
    return frozenset(out)


def visible_edges(graph: MixedGraph) -> frozenset[tuple[str, str]]:
    """The visible directed edges, as ``(tail, head)`` pairs (Zhang 2008).

    ``A -> B`` is visible when some ``C`` not adjacent to ``B`` has an edge
    into ``A``, or reaches ``A`` through a collider path into ``A`` whose
    every interior node is a parent of ``B``. Visibility certifies the edge
    carries no hidden confounding in any represented world, which is what
    adjustment amenability on MAGs and PAGs consumes. The marks are read as
    given (arrowheads must be definite); on a DAG or CPDAG every directed
    edge is visible by causal sufficiency, and callers there should not need
    this function.
    """
    marks = _mark_maps(graph)
    adj = {n: sorted(marks[n]) for n in marks}
    out: set[tuple[str, str]] = set()
    for a in sorted(marks):
        for b in adj[a]:
            if _is_directed(marks, a, b) and _is_visible(marks, adj, a, b):
                out.add((a, b))
    return frozenset(out)


def _is_visible(marks: _Marks, adj: dict[str, list[str]], a: str, b: str) -> bool:
    """Zhang's visibility for the directed edge ``a -> b``.

    The collider-path clause walks backward from ``a``: every interior node
    must be a parent of ``b`` with arrowheads at it on both path edges, so
    the condition is locally checkable and loop erasure is safe (the same
    argument as R4's discriminating walk).
    """
    parents_of_b = {p for p in adj[b] if _is_directed(marks, p, b)}
    for c in adj[a]:
        if c != b and b not in marks[c] and marks[a][c] is Mark.ARROW:
            return True
    queue = [
        q
        for q in adj[a]
        if q in parents_of_b and marks[a][q] is Mark.ARROW and marks[q][a] is Mark.ARROW
    ]
    seen = set(queue)
    while queue:
        q = queue.pop(0)
        for r in adj[q]:
            if r in (a, b) or marks[q][r] is not Mark.ARROW:
                continue
            if b not in marks[r]:
                return True
            if r in parents_of_b and r not in seen and marks[r][q] is Mark.ARROW:
                seen.add(r)
                queue.append(r)
    return False


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


def _undirect(marks: _Marks, a: str, b: str) -> bool:
    """Orient ``a --- b`` over circles: a tail at both ends."""
    changed = _set(marks, a, b, Mark.TAIL)
    return _set(marks, b, a, Mark.TAIL) or changed


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


def _uncovered_circle_path(
    marks: _Marks, adj: dict[str, list[str]], a: str, b: str
) -> list[str] | None:
    """The first uncovered all-circle path ``<a, c, ..., d, b>`` for R5.

    Every edge on the path is o-o, every consecutive triple is unshielded,
    the first interior node is not adjacent to ``b``, and the last is not
    adjacent to ``a``. Depth-first over simple paths in sorted order, the
    same argument as :func:`_uncovered_pd_reaches`: uncovered-ness belongs to
    the realized path, so plain reachability would overclaim. Returns the
    node list including both endpoints, or ``None``.
    """

    def circle(s: str, t: str) -> bool:
        return marks[s][t] is Mark.CIRCLE and marks[t][s] is Mark.CIRCLE

    def extend(prev: str, cur: str, used: frozenset[str]) -> list[str] | None:
        for nxt in adj[cur]:
            if nxt == b:
                if b not in marks[prev] and circle(cur, b) and a not in marks[cur]:
                    return [cur, b]
                continue
            if nxt in used or nxt in marks[prev] or not circle(cur, nxt):
                continue
            rest = extend(cur, nxt, used | {nxt})
            if rest is not None:
                return [cur, *rest]
        return None

    for c in adj[a]:
        if c == b or b in marks[c] or not circle(a, c):
            continue
        rest = extend(a, c, frozenset({a, c}))
        if rest is not None:
            return [a, *rest]
    return None


def _r5(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a o-o b with an uncovered circle path <a, c, ..., d, b>, c not adjacent
    to b and d not adjacent to a: the a-b edge and the whole path go undirected.

    The first of Zhang's selection rules; it can only fire when the class
    carries selection, so on a selection-free graph it is a no-op.
    """
    changed = False
    for a in sorted(marks):
        for b in adj[a]:
            if a >= b:
                continue
            if not (marks[a][b] is Mark.CIRCLE and marks[b][a] is Mark.CIRCLE):
                continue
            path = _uncovered_circle_path(marks, adj, a, b)
            if path is None:
                continue
            changed |= _undirect(marks, a, b)
            for s, t in pairwise(path):
                changed |= _undirect(marks, s, t)
    return changed


def _r6(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a --- b o-* c: a tail at b on (b, c); a and c may or may not be adjacent."""
    changed = False
    for b in sorted(marks):
        if not any(marks[b][a] is Mark.TAIL and marks[a][b] is Mark.TAIL for a in adj[b]):
            continue
        for c in adj[b]:
            if marks[b][c] is Mark.CIRCLE:
                changed |= _set(marks, b, c, Mark.TAIL)
    return changed


def _r7(marks: _Marks, adj: dict[str, list[str]]) -> bool:
    """a -o b o-* c, a and c non-adjacent: a tail at b on (b, c)."""
    changed = False
    for b in sorted(marks):
        for a in adj[b]:
            if not (marks[a][b] is Mark.TAIL and marks[b][a] is Mark.CIRCLE):
                continue
            for c in adj[b]:
                if c == a or c in marks[a]:
                    continue
                if marks[b][c] is Mark.CIRCLE:
                    changed |= _set(marks, b, c, Mark.TAIL)
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
    """Close a partially oriented graph under Zhang's rules R1 to R10.

    The input carries its v-structure arrowheads (and any knowledge-seeded
    marks); circles elsewhere. Rules only ever turn a circle into a tail or
    an arrowhead: a mark that is already definite never moves, so a seeded
    mark that disagrees with a rule simply stays and the conflict surfaces at
    the caller's validation. ``discriminating(d, a, b, c)`` answers R4's
    question, whether b is a collider on the discriminating path; ``None``
    skips that path. Sweeps run in sorted order until a full pass changes
    nothing, so the closure is deterministic. R5 to R7 are the selection
    rules and never fire on a selection-free class; Zhang stages the rules
    (R1 to R4, then R5 once, then R6 to R10) but every rule is individually
    sound and marks only move circle to definite, so the joint fixpoint is
    the same closure, which the round-trip tests hold executably. Returns
    kind ``mixed``.
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
        changed |= _r5(marks, adj)
        changed |= _r6(marks, adj)
        changed |= _r7(marks, adj)
        changed |= _r8(marks, adj)
        changed |= _r9(marks, adj)
        changed |= _r10(marks, adj)
    return _rebuild(graph, marks)


# -- the conversions -----------------------------------------------------------


def mag_of_pag(p: MixedGraph) -> MixedGraph:
    """Extract one member MAG from a PAG (Zhang 2008, arrowhead augmentation).

    Every ``o->`` edge becomes directed into its arrowhead, every ``-o``
    edge becomes directed out of its tail (the circle takes the arrowhead),
    undirected and bidirected and directed edges stay as they are, and the
    circle-circle component is oriented into a DAG that adds no unshielded
    collider (the Dor-Tarsi extension of its undirected skeleton; for a
    valid complete PAG that component is chordal, R5 guarantees it, so the
    extension exists). An inextricable circle component or a result that
    fails the MAG rules raises. Maximality of the member is not checked here
    because ``is_mag`` does not check it; the ``is_pag`` round trip is what
    guards it.
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
        elif cu is Mark.TAIL and cv is Mark.CIRCLE:
            fixed.append(Edge.make(e.u, Mark.TAIL, e.v, Mark.ARROW))
        elif cu is Mark.CIRCLE and cv is Mark.TAIL:
            fixed.append(Edge.make(e.u, Mark.ARROW, e.v, Mark.TAIL))
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
    (2008) proved the ten rules complete with latent confounders and
    selection bias both, so the result carries exactly the invariant marks.
    The input should be maximal; ``is_mag`` cannot check that, and a
    non-maximal input gives a PAG for the wrong class.
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


def mags_of_pag(
    p: MixedGraph, *, max_members: int | None = None, selection: bool = False
) -> tuple[MixedGraph, ...]:
    """Every MAG in the PAG's equivalence class, in a deterministic order.

    Each circle mark takes a tail or an arrowhead; an assignment survives
    when it yields a valid MAG whose canonical PAG (:func:`pag_of_mag`) is
    exactly ``p``. With ``selection=False`` (the default) only the members
    without undirected edges count: the reading every consumer that assumes
    no selection bias quantifies over, identification among them. Pass
    ``selection=True`` for the full Markov equivalence class, where a
    circle-circle edge may also resolve to an undirected selection edge. A
    PAG that itself carries selection marks (undirected or circle-tail
    edges) has no selection-free members and raises under the default.
    Exponential in the number of circle marks by design, which is fine at
    the scale the library targets; ``max_members`` raises once the count
    would pass it (no silent truncation). ``mag_of_pag(p)`` is always among
    the members.
    """
    if not p.is_pag():
        raise ValueError("mags_of_pag needs a complete PAG; check is_pag or build with pag()")
    if not selection:
        for e in p.edges:
            if e.is_undirected or {e.u_mark, e.v_mark} == {Mark.TAIL, Mark.CIRCLE}:
                raise ValueError(
                    f"edge {e.glyph()!r} is a selection mark, so the class has no "
                    "selection-free members; pass selection=True to enumerate the "
                    "full Markov equivalence class"
                )
    slots: list[tuple[str, str]] = []
    for e in sorted(p.edges, key=lambda e: (e.u, e.v)):
        if e.u_mark is Mark.CIRCLE:
            slots.append((e.u, e.v))
        if e.v_mark is Mark.CIRCLE:
            slots.append((e.v, e.u))
    found: list[MixedGraph] = []
    for assignment in product((Mark.TAIL, Mark.ARROW), repeat=len(slots)):
        marks = _mark_maps(p)
        for (node, other), mark in zip(slots, assignment, strict=True):
            marks[node][other] = mark
        candidate = replace(
            p,
            edges=frozenset(Edge.make(e.u, marks[e.u][e.v], e.v, marks[e.v][e.u]) for e in p.edges),
            kind=GraphKind.MAG,
        )
        if not selection and any(e.is_undirected for e in candidate.edges):
            continue
        if not candidate.is_mag() or pag_of_mag(candidate).edges != p.edges:
            continue
        if max_members is not None and len(found) >= max_members:
            raise ValueError(
                f"the class has more than {max_members} members; raise max_members "
                "or work on the PAG itself"
            )
        found.append(candidate)
    return tuple(found)
