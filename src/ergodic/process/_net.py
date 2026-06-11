# SPDX-License-Identifier: Apache-2.0
"""Workflow nets: Petri nets with one entry and one exit.

A :class:`WorkflowNet` is a Petri net whose places include a single ``source``
(no incoming arcs) and a single ``sink`` (no outgoing arcs), with every node on
some path between the two. Transitions carry an activity ``label`` or ``None``
for a silent step. The constructor validates all of that, so a value that
exists is structurally a workflow net; whether it is also sound (every run can
finish, finishes cleanly, and every transition can fire) is the behavioral
question :func:`~ergodic.process.is_sound` answers by reachability.

A marking is a canonical tuple of ``(place, count)`` pairs, hashable so state
search can set-ify it. :meth:`WorkflowNet.index` returns the pre/post maps the
firing rule reads; soundness checking and conformance replay share it.
:func:`net_of_tree` builds the net of a process tree by structural recursion,
and that construction is sound for every tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ._tree import ProcessTree

__all__ = ["Marking", "NetIndex", "WorkflowNet", "marking", "net_of_tree"]

Marking = tuple[tuple[str, int], ...]
"""A net marking: ``(place, count)`` pairs, sorted, counts positive."""


def marking(counts: Mapping[str, int] | Iterable[tuple[str, int]]) -> Marking:
    """The canonical marking for ``counts``: sorted pairs, zero counts dropped."""
    items = counts.items() if hasattr(counts, "items") else counts
    merged: dict[str, int] = {}
    for place, count in items:
        if count < 0:
            raise ValueError(f"negative token count {count} on place {place!r}")
        if count:
            merged[place] = merged.get(place, 0) + count
    return tuple(sorted(merged.items()))


@dataclass(frozen=True)
class NetIndex:
    """The firing-rule view of a net: per-transition pre and post places.

    Built once per algorithm run via :meth:`WorkflowNet.index`; ``enabled``
    and ``fire`` are the one firing rule that soundness checking and
    conformance replay share.
    """

    pre: Mapping[str, tuple[str, ...]]
    post: Mapping[str, tuple[str, ...]]
    labels: Mapping[str, str | None]

    def enabled(self, state: Marking) -> tuple[str, ...]:
        """The transitions enabled at ``state``, in id order."""
        tokens = dict(state)
        return tuple(
            tid for tid in self.labels if all(tokens.get(place, 0) >= 1 for place in self.pre[tid])
        )

    def fire(self, state: Marking, tid: str) -> Marking:
        """The marking after firing ``tid`` at ``state``; raises if not enabled."""
        tokens = dict(state)
        for place in self.pre[tid]:
            if tokens.get(place, 0) < 1:
                raise ValueError(f"transition {tid!r} is not enabled at {state!r}")
            tokens[place] -= 1
        for place in self.post[tid]:
            tokens[place] = tokens.get(place, 0) + 1
        return tuple(sorted((place, count) for place, count in tokens.items() if count))


@dataclass(frozen=True)
class WorkflowNet:
    """A workflow net: places, labeled or silent transitions, and arcs.

    ``transitions`` maps ids to labels as ``(id, label)`` pairs, with ``None``
    for a silent transition; ids and place names live in one namespace and
    must not collide. Arcs connect a place to a transition or a transition to
    a place, once each (these are weight-one nets). The constructor sorts the
    fields into canonical order and validates the workflow-net shape: a single
    ``source`` with no inputs, a single ``sink`` with no outputs, and every
    node on some directed path from source to sink.
    """

    places: tuple[str, ...]
    transitions: tuple[tuple[str, str | None], ...]
    arcs: tuple[tuple[str, str], ...]
    source: str
    sink: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "places", tuple(sorted(self.places)))
        object.__setattr__(self, "transitions", tuple(sorted(self.transitions)))
        object.__setattr__(self, "arcs", tuple(sorted(self.arcs)))
        place_set = set(self.places)
        tids = [tid for tid, _ in self.transitions]
        tid_set = set(tids)
        if len(place_set) != len(self.places):
            raise ValueError("duplicate place names")
        if len(tid_set) != len(tids):
            raise ValueError("duplicate transition ids")
        shared = sorted(place_set & tid_set)
        if shared:
            raise ValueError(f"names used as both place and transition: {shared}")
        for name, role in ((self.source, "source"), (self.sink, "sink")):
            if name not in place_set:
                raise ValueError(f"{role} {name!r} is not a place")
        if self.source == self.sink:
            raise ValueError("source and sink must be different places")
        if len(set(self.arcs)) != len(self.arcs):
            raise ValueError("duplicate arcs (these are weight-one nets)")
        for a, b in self.arcs:
            ok = (a in place_set and b in tid_set) or (a in tid_set and b in place_set)
            if not ok:
                raise ValueError(
                    f"arc ({a!r}, {b!r}) must connect a place and a transition, "
                    "and both ends must exist"
                )
            if b == self.source:
                raise ValueError(f"source {self.source!r} cannot have incoming arcs")
            if a == self.sink:
                raise ValueError(f"sink {self.sink!r} cannot have outgoing arcs")
        forward: dict[str, list[str]] = {}
        backward: dict[str, list[str]] = {}
        for a, b in self.arcs:
            forward.setdefault(a, []).append(b)
            backward.setdefault(b, []).append(a)
        from_source = _reach(self.source, forward)
        to_sink = _reach(self.sink, backward)
        stranded = sorted((place_set | tid_set) - (from_source & to_sink))
        if stranded:
            raise ValueError(
                f"every node must lie on a path from source to sink; these do not: {stranded}"
            )

    @property
    def initial_marking(self) -> Marking:
        """One token on the source place."""
        return ((self.source, 1),)

    @property
    def final_marking(self) -> Marking:
        """One token on the sink place."""
        return ((self.sink, 1),)

    def index(self) -> NetIndex:
        """The pre/post maps for the firing rule; build once per algorithm run."""
        pre: dict[str, tuple[str, ...]] = {}
        post: dict[str, tuple[str, ...]] = {}
        labels = dict(self.transitions)
        pre_lists: dict[str, list[str]] = {tid: [] for tid in labels}
        post_lists: dict[str, list[str]] = {tid: [] for tid in labels}
        for a, b in self.arcs:
            if b in labels:
                pre_lists[b].append(a)
            else:
                post_lists[a].append(b)
        for tid in labels:
            pre[tid] = tuple(sorted(pre_lists[tid]))
            post[tid] = tuple(sorted(post_lists[tid]))
        return NetIndex(pre=pre, post=post, labels=labels)

    def __repr__(self) -> str:
        n_silent = sum(1 for _, label in self.transitions if label is None)
        return (
            f"WorkflowNet(places={len(self.places)}, transitions={len(self.transitions)} "
            f"({n_silent} silent), arcs={len(self.arcs)})"
        )


def _reach(start: str, edges: Mapping[str, list[str]]) -> set[str]:
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for nxt in edges.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def net_of_tree(tree: ProcessTree) -> WorkflowNet:
    """The workflow net of a process tree, by structural recursion.

    Each subtree is built between an entry and an exit place: a leaf becomes
    one transition (silent for :data:`~ergodic.process.TAU`), ``seq`` chains
    its children through fresh places, ``xor`` lets every child share the
    entry and exit, ``par`` brackets per-child places with silent split and
    join transitions, and ``loop`` enters through a silent step into a do
    body whose exit either leaves through a silent step or runs the redo body
    back to the do entry. The construction is sound for every tree (and
    :func:`~ergodic.process.is_sound` will verify it for real).
    """
    places: list[str] = ["source", "sink"]
    transitions: list[tuple[str, str | None]] = []
    arcs: list[tuple[str, str]] = []

    def new_place() -> str:
        name = f"p{len(places) - 2}"
        places.append(name)
        return name

    def new_transition(label: str | None) -> str:
        tid = f"t{len(transitions)}"
        transitions.append((tid, label))
        return tid

    def build(node: ProcessTree, entry: str, exit_: str) -> None:
        if node.op is None:
            tid = new_transition(node.label)
            arcs.append((entry, tid))
            arcs.append((tid, exit_))
        elif node.op == "seq":
            current = entry
            for child in node.children[:-1]:
                following = new_place()
                build(child, current, following)
                current = following
            build(node.children[-1], current, exit_)
        elif node.op == "xor":
            for child in node.children:
                build(child, entry, exit_)
        elif node.op == "par":
            split = new_transition(None)
            join = new_transition(None)
            arcs.append((entry, split))
            arcs.append((join, exit_))
            for child in node.children:
                child_in = new_place()
                child_out = new_place()
                arcs.append((split, child_in))
                arcs.append((child_out, join))
                build(child, child_in, child_out)
        else:  # loop
            do_in = new_place()
            do_out = new_place()
            enter = new_transition(None)
            leave = new_transition(None)
            arcs.append((entry, enter))
            arcs.append((enter, do_in))
            arcs.append((do_out, leave))
            arcs.append((leave, exit_))
            build(node.children[0], do_in, do_out)
            build(node.children[1], do_out, do_in)

    build(tree, "source", "sink")
    return WorkflowNet(
        places=tuple(places),
        transitions=tuple(transitions),
        arcs=tuple(arcs),
        source="source",
        sink="sink",
    )
