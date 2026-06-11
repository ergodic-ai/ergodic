# SPDX-License-Identifier: Apache-2.0
"""The inductive miner: a block-structured process model from an event log.

:func:`discover_model` recurses over the log's traces, kept as a multiset
weighted by case counts. Base cases go first: all traces empty gives the
silent step, a mix of empty and non-empty gives a choice between the silent
step and the rest, and a single repeated activity gives a leaf or a self-loop.
Otherwise the miner builds the directly-follows graph of the current sublog
and tries four cuts in order: exclusive choice (connected components of the
graph read undirected), sequence (a strict total order over merged strongly
connected components), parallel (every cross-part pair connected in both
directions, every part holding a start and an end activity), and loop (a do
body holding all start and end activities, redo parts hanging off its
boundary). The first valid cut partitions the activities, the traces project
into one sublog per part, and the miner recurses. When no cut is valid the
subtree falls through to the flower model ``loop(TAU, xor(...))``, which
accepts its activities in any order. The construction follows Leemans,
Fahland, and van der Aalst, "Discovering block-structured process models from
event logs" (2013).

``noise`` turns on the infrequent variant (Leemans, Fahland, and van der
Aalst, "Discovering block-structured process models from event logs
containing infrequent behaviour", 2014): at every recursion, before cut
detection, a directly-follows edge is dropped when its count falls below
``noise`` times the heaviest edge leaving the same activity. Start and end
sets stay as observed.

Whatever the miner removes is reported, never hidden: the returned
:class:`ProcessModel` records where the flower fall-through fired, how many
edges the filter dropped, and how many events the choice projection
discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

from ._tree import TAU, ProcessTree, loop, par, seq, xor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..data import EventLog
    from ._net import WorkflowNet

__all__ = ["ProcessModel", "discover_model"]

_Trace = tuple[str, ...]
_SubLog = dict[_Trace, int]
_Edges = dict[tuple[str, str], int]


@dataclass(frozen=True)
class ProcessModel:
    """A mined process model and the provenance of everything removed.

    ``tree`` is the block-structured model and ``net`` is its workflow net,
    ``tree.to_net()``, sound by construction. The rest records what the miner
    filtered or dropped on the way (filtering is visible, never silent):

    - ``fallthroughs``: paths, in :meth:`ProcessTree.nodes` convention (child
      indices from the root), of the subtrees where no cut was valid and the
      miner fell through to the flower model, which accepts its activities in
      any order. Empty when every subtree came from a real cut.
    - ``n_filtered_arcs``: directly-follows edges removed by the infrequent
      filter, summed over all recursions. Zero when ``noise`` is 0.
    - ``n_dropped_events``: events discarded while projecting traces through
      an exclusive-choice cut, where each trace goes whole to one branch and
      its events outside that branch are dropped.
    - ``dropped_share``: ``n_dropped_events`` over the log's total events.
    - ``noise``: the filter level the model was mined with.
    """

    tree: ProcessTree
    net: WorkflowNet
    fallthroughs: tuple[tuple[int, ...], ...]
    n_filtered_arcs: int
    n_dropped_events: int
    dropped_share: float
    noise: float

    def __repr__(self) -> str:
        extras = []
        if self.fallthroughs:
            extras.append(f"flowers={list(self.fallthroughs)}")
        if self.n_filtered_arcs:
            extras.append(f"filtered_arcs={self.n_filtered_arcs}")
        if self.n_dropped_events:
            extras.append(f"dropped_events={self.n_dropped_events} ({self.dropped_share:.1%})")
        note = f", {', '.join(extras)}" if extras else ""
        return f"ProcessModel(tree={self.tree!r}, noise={self.noise}{note})"


def discover_model(log: EventLog, *, noise: float = 0.0) -> ProcessModel:
    """Mine a block-structured process model from ``log``.

    The miner works on the log's variants, weighting each distinct trace by
    how many cases follow it, so a large log mines as fast as its variant
    table. ``noise`` must lie in ``[0, 1)``: at 0 the log is taken literally,
    and a larger value drops, at every recursion and before cut detection,
    each directly-follows edge carrying less than ``noise`` times the
    heaviest edge leaving the same activity. The returned
    :class:`ProcessModel` carries the tree, its workflow net, and the
    provenance counters.
    """
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"noise must lie in [0.0, 1.0), got {noise}")
    variants = {trace: int(count) for trace, count in sorted(log.variants().items())}
    state = _MinerState()
    tree = _mine(variants, noise, state)
    fallthroughs = tuple(
        path for path, node in tree.nodes() if any(node is flower for flower in state.flowers)
    )
    share = 0.0 if log.n_rows == 0 else state.n_dropped_events / log.n_rows
    return ProcessModel(
        tree=tree,
        net=tree.to_net(),
        fallthroughs=fallthroughs,
        n_filtered_arcs=state.n_filtered_arcs,
        n_dropped_events=state.n_dropped_events,
        dropped_share=share,
        noise=noise,
    )


# -- the recursion ---------------------------------------------------------------


@dataclass
class _MinerState:
    """Provenance gathered across the recursion.

    ``flowers`` holds the flower subtrees by object identity; after the tree
    is assembled, :func:`discover_model` locates their paths with
    :meth:`ProcessTree.nodes`, so two structurally equal flowers at different
    paths each count once.
    """

    n_filtered_arcs: int = 0
    n_dropped_events: int = 0
    flowers: list[ProcessTree] = field(default_factory=list)


def _mine(sublog: _SubLog, noise: float, state: _MinerState) -> ProcessTree:
    """One recursion of the miner over a trace multiset."""
    if all(not trace for trace in sublog):
        return TAU  # all traces empty (or no traces at all): the silent step
    if () in sublog:
        nonempty = {trace: count for trace, count in sublog.items() if trace}
        return xor(TAU, _mine(nonempty, noise, state))
    alphabet = tuple(sorted({activity for trace in sublog for activity in trace}))
    if len(alphabet) == 1:
        leaf = ProcessTree(label=alphabet[0])
        if all(trace == (alphabet[0],) for trace in sublog):
            return leaf
        return loop(leaf, TAU)

    edges, starts, ends = _dfg(sublog)
    if noise > 0.0:
        edges, removed = _filter_edges(edges, noise)
        state.n_filtered_arcs += removed

    parts = _xor_cut(alphabet, edges)
    if parts is not None:
        sublogs, dropped = _project_choice(sublog, parts)
        state.n_dropped_events += dropped
        return xor(*(_mine(part_log, noise, state) for part_log in sublogs))
    parts = _sequence_cut(alphabet, edges)
    if parts is not None:
        return seq(*(_mine(part_log, noise, state) for part_log in _project_onto(sublog, parts)))
    parts = _parallel_cut(alphabet, edges, starts, ends)
    if parts is not None:
        return par(*(_mine(part_log, noise, state) for part_log in _project_onto(sublog, parts)))
    cut = _loop_cut(alphabet, edges, starts, ends)
    if cut is not None:
        do_part, redo_parts = cut
        sublogs = _project_loop(sublog, (do_part, *redo_parts))
        do_tree = _mine(sublogs[0], noise, state)
        redo_trees = [_mine(part_log, noise, state) for part_log in sublogs[1:]]
        redo = redo_trees[0] if len(redo_trees) == 1 else xor(*redo_trees)
        return loop(do_tree, redo)

    flower = loop(TAU, xor(*(ProcessTree(label=activity) for activity in alphabet)))
    state.flowers.append(flower)
    return flower


def _dfg(sublog: _SubLog) -> tuple[_Edges, set[str], set[str]]:
    """The weighted directly-follows graph with the start and end activities."""
    edges: _Edges = {}
    starts: set[str] = set()
    ends: set[str] = set()
    for trace, count in sublog.items():
        starts.add(trace[0])
        ends.add(trace[-1])
        for x, y in pairwise(trace):
            edges[(x, y)] = edges.get((x, y), 0) + count
    return edges, starts, ends


def _filter_edges(edges: _Edges, noise: float) -> tuple[_Edges, int]:
    """Drop each edge lighter than ``noise`` times its source's heaviest edge."""
    heaviest: dict[str, int] = {}
    for (x, _), count in edges.items():
        heaviest[x] = max(heaviest.get(x, 0), count)
    kept = {pair: count for pair, count in edges.items() if count >= noise * heaviest[pair[0]]}
    return kept, len(edges) - len(kept)


# -- the four cuts, in paper order -------------------------------------------------


def _components(
    nodes: Iterable[str], links: Iterable[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    """Connected components under ``links`` read undirected, sorted twice over."""
    parent = {node: node for node in nodes}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for x, y in links:
        parent[find(x)] = find(y)
    groups: dict[str, list[str]] = {}
    for node in sorted(parent):
        groups.setdefault(find(node), []).append(node)
    return tuple(sorted(tuple(group) for group in groups.values()))


def _xor_cut(alphabet: tuple[str, ...], edges: _Edges) -> tuple[tuple[str, ...], ...] | None:
    """Connected components of the graph read undirected; valid with two or more."""
    parts = _components(alphabet, edges)
    return parts if len(parts) >= 2 else None


def _sccs(nodes: tuple[str, ...], edges: _Edges) -> list[tuple[str, ...]]:
    """Strongly connected components (Kosaraju), deterministic over sorted input."""
    adjacent: dict[str, list[str]] = {node: [] for node in nodes}
    reverse: dict[str, list[str]] = {node: [] for node in nodes}
    for x, y in sorted(edges):
        if x != y:
            adjacent[x].append(y)
            reverse[y].append(x)
    order: list[str] = []
    seen: set[str] = set()
    for start in nodes:
        if start in seen:
            continue
        seen.add(start)
        stack: list[tuple[str, int]] = [(start, 0)]
        while stack:
            node, i = stack[-1]
            if i < len(adjacent[node]):
                stack[-1] = (node, i + 1)
                nxt = adjacent[node][i]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append((nxt, 0))
            else:
                stack.pop()
                order.append(node)
    components: list[tuple[str, ...]] = []
    assigned: set[str] = set()
    for start in reversed(order):
        if start in assigned:
            continue
        member_list = [start]
        assigned.add(start)
        frontier = [start]
        while frontier:
            node = frontier.pop()
            for nxt in reverse[node]:
                if nxt not in assigned:
                    assigned.add(nxt)
                    member_list.append(nxt)
                    frontier.append(nxt)
        components.append(tuple(sorted(member_list)))
    return components


def _sequence_cut(alphabet: tuple[str, ...], edges: _Edges) -> tuple[tuple[str, ...], ...] | None:
    """A strict total order over merged strongly connected components.

    Pairwise-unreachable components cannot be ordered, so they merge into one
    part; the remaining parts must form a chain under reachability, verified
    explicitly, earliest part first.
    """
    components = _sccs(alphabet, edges)
    n = len(components)
    if n < 2:
        return None
    index = {activity: i for i, component in enumerate(components) for activity in component}
    direct: list[set[int]] = [set() for _ in range(n)]
    for x, y in edges:
        i, j = index[x], index[y]
        if i != j:
            direct[i].add(j)
    reach: list[set[int]] = []
    for i in range(n):
        seen = set(direct[i])
        frontier = list(direct[i])
        while frontier:
            j = frontier.pop()
            for k in direct[j]:
                if k not in seen:
                    seen.add(k)
                    frontier.append(k)
        reach.append(seen)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if j not in reach[i] and i not in reach[j]:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 2:
        return None
    members = list(groups.values())

    def reaches(first: list[int], second: list[int]) -> bool:
        return any(j in reach[i] for i in first for j in second)

    ranked = sorted(
        members,
        key=lambda group: sum(reaches(group, other) for other in members if other is not group),
        reverse=True,
    )
    for a in range(len(ranked)):
        for b in range(a + 1, len(ranked)):
            if not reaches(ranked[a], ranked[b]) or reaches(ranked[b], ranked[a]):
                return None
    return tuple(
        tuple(sorted(activity for i in group for activity in components[i])) for group in ranked
    )


def _parallel_cut(
    alphabet: tuple[str, ...], edges: _Edges, starts: set[str], ends: set[str]
) -> tuple[tuple[str, ...], ...] | None:
    """Parts pairwise connected by double edges, each holding a start and an end.

    Components of the graph that joins two activities whenever their double
    edge is missing give the finest candidate parts; a part lacking a start
    or an end activity merges with another deficient part when one exists
    (deficiencies often complement), else with the first whole part. One
    part left means the cut is invalid.
    """
    links = [
        (x, y)
        for i, x in enumerate(alphabet)
        for y in alphabet[i + 1 :]
        if (x, y) not in edges or (y, x) not in edges
    ]
    parts = list(_components(alphabet, links))
    if len(parts) < 2:
        return None

    def lacking(part: tuple[str, ...]) -> bool:
        members = set(part)
        return not (members & starts) or not (members & ends)

    while len(parts) > 1:
        bad = [i for i, part in enumerate(parts) if lacking(part)]
        if not bad:
            break
        i = bad[0]
        j = bad[1] if len(bad) > 1 else (1 if i == 0 else 0)
        merged = tuple(sorted(parts[i] + parts[j]))
        parts = sorted([part for k, part in enumerate(parts) if k not in (i, j)] + [merged])
    if len(parts) < 2 or any(lacking(part) for part in parts):
        return None
    return tuple(parts)


def _loop_cut(
    alphabet: tuple[str, ...], edges: _Edges, starts: set[str], ends: set[str]
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    """A do body holding all start and end activities, redo parts off its boundary.

    Removing the start and end activities splits the rest into candidate
    components (which have no edges to each other by construction). A
    candidate is a valid redo part when every edge from the body into it
    leaves an end activity and every edge from it back into the body arrives
    at a start activity; a candidate failing that merges into the do part.
    """
    body = starts | ends
    rest = tuple(activity for activity in alphabet if activity not in body)
    if not rest:
        return None
    rest_set = set(rest)
    links = [(x, y) for x, y in edges if x in rest_set and y in rest_set]
    do = set(body)
    redos: list[tuple[str, ...]] = []
    for component in _components(rest, links):
        members = set(component)
        entering_ok = all(x in ends for x, y in edges if x in body and y in members)
        leaving_ok = all(y in starts for x, y in edges if x in members and y in body)
        if entering_ok and leaving_ok:
            redos.append(component)
        else:
            do |= members
    if not redos:
        return None
    return tuple(sorted(do)), tuple(redos)


# -- projections to sublogs --------------------------------------------------------


def _project_choice(
    sublog: _SubLog, parts: tuple[tuple[str, ...], ...]
) -> tuple[list[_SubLog], int]:
    """Each trace goes whole to the part holding most of its events.

    Ties go to the part of the trace's earliest event among the tied parts.
    Events outside the assigned part are dropped; the count of dropped
    events, weighted by trace multiplicity, is returned alongside.
    """
    index = {activity: i for i, part in enumerate(parts) for activity in part}
    sublogs: list[_SubLog] = [{} for _ in parts]
    dropped = 0
    for trace, count in sublog.items():
        tallies = [0] * len(parts)
        for activity in trace:
            tallies[index[activity]] += 1
        best = max(tallies)
        chosen = next(index[a] for a in trace if tallies[index[a]] == best)
        kept = tuple(activity for activity in trace if index[activity] == chosen)
        dropped += (len(trace) - len(kept)) * count
        sublogs[chosen][kept] = sublogs[chosen].get(kept, 0) + count
    return sublogs, dropped


def _project_onto(sublog: _SubLog, parts: tuple[tuple[str, ...], ...]) -> list[_SubLog]:
    """Each trace projects onto each part's alphabet as a subsequence.

    Nothing is dropped: every event lands in the part owning its activity,
    in original order, and a trace with no events in a part contributes the
    empty trace there. Serves the sequence and parallel cuts alike.
    """
    index = {activity: i for i, part in enumerate(parts) for activity in part}
    sublogs: list[_SubLog] = [{} for _ in parts]
    for trace, count in sublog.items():
        pieces: list[list[str]] = [[] for _ in parts]
        for activity in trace:
            pieces[index[activity]].append(activity)
        for i, piece in enumerate(pieces):
            key = tuple(piece)
            sublogs[i][key] = sublogs[i].get(key, 0) + count
    return sublogs


def _project_loop(sublog: _SubLog, parts: tuple[tuple[str, ...], ...]) -> list[_SubLog]:
    """Each trace cuts into maximal segments within one part's alphabet.

    Each segment appends to its part's sublog as a trace of its own, so the
    do part collects one sublog trace per visit and each redo part one per
    round. Nothing is dropped.
    """
    index = {activity: i for i, part in enumerate(parts) for activity in part}
    sublogs: list[_SubLog] = [{} for _ in parts]
    for trace, count in sublog.items():
        current = index[trace[0]]
        segment = [trace[0]]
        for activity in trace[1:]:
            owner = index[activity]
            if owner == current:
                segment.append(activity)
                continue
            key = tuple(segment)
            sublogs[current][key] = sublogs[current].get(key, 0) + count
            current = owner
            segment = [activity]
        key = tuple(segment)
        sublogs[current][key] = sublogs[current].get(key, 0) + count
    return sublogs
