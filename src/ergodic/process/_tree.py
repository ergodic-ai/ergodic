# SPDX-License-Identifier: Apache-2.0
"""Process trees: block-structured process models.

A process tree is either a leaf (one activity, or the silent step :data:`TAU`)
or an operator over at least two children: ``seq`` runs its children left to
right, ``xor`` runs exactly one, ``par`` interleaves them, and
``loop(do, redo)`` runs the do body, then any number of redo-then-do rounds.
Trees are immutable values with structural equality in a canonical form: a
nested sequence is spliced into its parent (``seq(a, seq(b, c))`` is
``seq(a, b, c)``), and ``xor``/``par`` children are kept sorted
by their activity alphabets, so a mined tree compares directly against one
written by hand. The inductive
miner (:func:`~ergodic.process.discover_model`) produces them, and
:meth:`ProcessTree.to_net` converts one to a workflow net that is sound by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._net import WorkflowNet

__all__ = ["TAU", "ProcessTree", "loop", "par", "seq", "xor"]

_OPERATORS = ("seq", "xor", "par", "loop")


@dataclass(frozen=True)
class ProcessTree:
    """One node of a process tree.

    A leaf has no children and carries an activity ``label`` (or ``None``,
    which is the silent step :data:`TAU`). An operator node has ``op`` set,
    no label, and at least two children; ``loop`` takes exactly two, the do
    body and the redo body. A ``seq`` child of a ``seq`` is spliced into its
    parent (sequence is associative), and the children of ``xor`` and ``par`` are stored sorted by
    their activity alphabets (a silent child first), so equal models are equal
    objects regardless of how they were assembled.
    """

    op: str | None = None
    label: str | None = None
    children: tuple[ProcessTree, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.op is None:
            if self.children:
                raise ValueError("a leaf has no children")
            return
        if self.op not in _OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}; expected one of {_OPERATORS}")
        if self.label is not None:
            raise ValueError("an operator node has no label")
        if self.op == "loop":
            if len(self.children) != 2:
                raise ValueError("loop takes exactly two children: the do body and the redo body")
        elif len(self.children) < 2:
            raise ValueError(f"{self.op} takes at least two children")
        if self.op == "seq":
            # seq is associative and a nested seq marks nothing: a seq child splices
            # into its parent, so seq(a, seq(b, c)) is seq(a, b, c), the one
            # canonical form a hand-written tree and a mined one share. A nested
            # xor is kept, because decision_points reads each xor as its own
            # decision, and a nested par is left as built.
            flat: list[ProcessTree] = []
            for child in self.children:
                flat.extend(child.children if child.op == "seq" else (child,))
            object.__setattr__(self, "children", tuple(flat))
        if self.op in ("xor", "par"):
            ordered = tuple(sorted(self.children, key=lambda child: child.activities()))
            object.__setattr__(self, "children", ordered)

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (an activity or the silent step)."""
        return self.op is None

    @property
    def is_tau(self) -> bool:
        """Whether this node is the silent step."""
        return self.op is None and self.label is None

    def activities(self) -> tuple[str, ...]:
        """The activity labels in this subtree, sorted and unique."""
        if self.op is None:
            return () if self.label is None else (self.label,)
        seen: set[str] = set()
        for child in self.children:
            seen.update(child.activities())
        return tuple(sorted(seen))

    def nodes(self) -> tuple[tuple[tuple[int, ...], ProcessTree], ...]:
        """Every node with its path from the root, depth first.

        A path is the tuple of child indices leading to the node; the root's
        path is ``()``.
        """
        out: list[tuple[tuple[int, ...], ProcessTree]] = []

        def walk(node: ProcessTree, path: tuple[int, ...]) -> None:
            out.append((path, node))
            for i, child in enumerate(node.children):
                walk(child, (*path, i))

        walk(self, ())
        return tuple(out)

    def subtree(self, path: tuple[int, ...]) -> ProcessTree:
        """The node at ``path``, where each entry picks a child by index."""
        node = self
        for depth, i in enumerate(path):
            if not 0 <= i < len(node.children):
                raise ValueError(f"no child {i} at path {path[:depth]} of {node!r}")
            node = node.children[i]
        return node

    def to_net(self) -> WorkflowNet:
        """The workflow net of this tree, sound by construction."""
        from ._net import net_of_tree

        return net_of_tree(self)

    def __repr__(self) -> str:
        if self.op is None:
            return "TAU" if self.label is None else repr(self.label)
        return f"{self.op}({', '.join(repr(child) for child in self.children)})"


TAU = ProcessTree()
"""The silent step: a leaf that takes no time and leaves no event."""


def _coerce(child: str | ProcessTree) -> ProcessTree:
    if isinstance(child, ProcessTree):
        return child
    return ProcessTree(label=child)


def seq(*children: str | ProcessTree) -> ProcessTree:
    """A sequence node: the children run left to right. Strings become leaves."""
    return ProcessTree(op="seq", children=tuple(_coerce(child) for child in children))


def xor(*children: str | ProcessTree) -> ProcessTree:
    """An exclusive choice node: exactly one child runs. Strings become leaves."""
    return ProcessTree(op="xor", children=tuple(_coerce(child) for child in children))


def par(*children: str | ProcessTree) -> ProcessTree:
    """A parallel node: the children interleave. Strings become leaves."""
    return ProcessTree(op="par", children=tuple(_coerce(child) for child in children))


def loop(do: str | ProcessTree, redo: str | ProcessTree) -> ProcessTree:
    """A loop node: ``do`` runs, then any number of ``redo`` then ``do`` rounds."""
    return ProcessTree(op="loop", children=(_coerce(do), _coerce(redo)))
