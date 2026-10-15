# SPDX-License-Identifier: Apache-2.0
"""Hierarchy: an aggregation tree over series, where a parent sums its children.

A hierarchy is hard structure. A parent equals the aggregate of its children by
construction, so the edges are known and deterministic, not something to learn.
It drives :class:`HierarchicalTSData` now and is reusable for grouped data later.

It covers strict trees with sum or mean aggregation. Crossed groupings (region by
category) are not supported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_AGGREGATIONS = ("sum", "mean")


@dataclass(frozen=True)
class Hierarchy:
    """A tree of named series, with edges running parent to child, and an aggregation rule."""

    edges: frozenset[tuple[str, str]]
    aggregation: str = "sum"

    def __post_init__(self) -> None:
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_AGGREGATIONS}, got {self.aggregation!r}"
            )
        dups = sorted(child for child, n in Counter(c for _, c in self.edges).items() if n > 1)
        if dups:
            raise ValueError(f"nodes with more than one parent: {dups}")
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        parent = {c: p for p, c in self.edges}
        for start in parent:
            node = start
            steps = 0
            while node in parent:
                node = parent[node]
                steps += 1
                if node == start or steps > len(parent):
                    raise ValueError("hierarchy has a cycle")

    @classmethod
    def tree(cls, adjacency: Mapping[str, Iterable[str]], aggregation: str = "sum") -> Hierarchy:
        """Build from a parent-to-children mapping, for example ``{"total": ["a", "b"]}``."""
        edges = frozenset((parent, child) for parent, kids in adjacency.items() for child in kids)
        return cls(edges=edges, aggregation=aggregation)

    @cached_property
    def _children(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for parent, child in self.edges:
            out.setdefault(parent, []).append(child)
        return {parent: tuple(sorted(kids)) for parent, kids in out.items()}

    @cached_property
    def _parent(self) -> dict[str, str]:
        return {child: parent for parent, child in self.edges}

    @cached_property
    def nodes(self) -> frozenset[str]:
        out: set[str] = set()
        for parent, child in self.edges:
            out.add(parent)
            out.add(child)
        return frozenset(out)

    def children_of(self, node: str) -> tuple[str, ...]:
        return self._children.get(node, ())

    def parent_of(self, node: str) -> str | None:
        return self._parent.get(node)

    def is_leaf(self, node: str) -> bool:
        return node not in self._children

    @property
    def roots(self) -> tuple[str, ...]:
        """Nodes with no parent."""
        return tuple(sorted(n for n in self.nodes if n not in self._parent))

    @property
    def leaves(self) -> tuple[str, ...]:
        """Nodes with no children: the base series."""
        return tuple(sorted(n for n in self.nodes if n not in self._children))

    @property
    def internal_nodes(self) -> tuple[str, ...]:
        """Nodes with children: the aggregates."""
        return tuple(sorted(self._children))

    def leaf_descendants(self, node: str) -> tuple[str, ...]:
        """The base series that aggregate up to ``node`` (``node`` itself when it is a leaf)."""
        if self.is_leaf(node):
            return (node,)
        out: list[str] = []
        for child in self.children_of(node):
            out.extend(self.leaf_descendants(child))
        return tuple(out)

    def levels(self) -> dict[str, int]:
        """Depth of each node, with roots at 0."""
        depth: dict[str, int] = {}
        frontier = [(root, 0) for root in self.roots]
        while frontier:
            node, current = frontier.pop()
            depth[node] = current
            for child in self.children_of(node):
                frontier.append((child, current + 1))
        return depth

    def __repr__(self) -> str:
        return (
            f"Hierarchy(nodes={len(self.nodes)}, leaves={len(self.leaves)}, "
            f"agg={self.aggregation!r})"
        )
