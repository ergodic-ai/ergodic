# SPDX-License-Identifier: Apache-2.0
"""The training schedule: which nodes get fitted, in what order, by what strategy.

Fitting a graph of series jointly is one big sampler over everything at once.
Staged fitting cuts it into waves: fit the nodes with no modeled parents, hand
their posteriors down, fit the next wave, and so on. That is what makes a
twelve-node graph affordable, and it is what lets independent nodes in the same
wave run side by side.

:class:`TrainingSchedule` is the plan for that, and only the plan. It holds no
data, runs no sampler, and can be printed and inspected before anything is
fitted. Three roles drive it. A ``SCENARIO`` node is given, not fitted, so it
resolves first. An ``AGGREGATE`` node is the sum of its hierarchy children, so it
resolves last. Everything else is a ``MODEL`` node, and its wave is one deeper
than the deepest modeled parent it reads.

Self-memory (``sales[t-1] -> sales[t]``) is not a staging dependency. It lives
inside the node's own mechanism, so it never pushes a node into a later wave.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ._structure import split_lag

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..data import Hierarchy
    from ..graph import MixedGraph

__all__ = ["NodeRole", "Stage", "StageStrategy", "TrainingSchedule"]


class NodeRole(Enum):
    """What a node is, which decides how the schedule treats it."""

    MODEL = "model"
    """Fitted from data by its own mechanism."""

    SCENARIO = "scenario"
    """Supplied, not fitted: a known future or an assumption under test."""

    AGGREGATE = "aggregate"
    """Summed from its hierarchy children, never fitted."""


class StageStrategy(Enum):
    """How one stage's nodes get resolved."""

    RESOLVED = "resolved"
    """Nothing to fit: the values are already known."""

    MCMC = "mcmc"
    """Fitted by the sampler, reading the posteriors of earlier stages."""

    AGGREGATE = "aggregate"
    """Added up from children that earlier stages produced."""


@dataclass(frozen=True)
class Stage:
    """One wave of the schedule.

    ``parent_edges`` maps a node to the ``(variable, lag)`` pairs it reads from
    *earlier* stages, which is exactly the set of posteriors that has to be
    handed in when the stage runs. Parents inside the same stage never appear,
    because same-wave nodes have no edges between them.
    """

    index: int
    nodes: tuple[str, ...]
    strategy: StageStrategy
    parent_edges: Mapping[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    hierarchy_children: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __repr__(self) -> str:
        body = ", ".join(self.nodes)
        suffix = ""
        if self.strategy is StageStrategy.AGGREGATE:
            for parent, children in self.hierarchy_children.items():
                suffix += f"  {parent} = sum({' + '.join(children)})"
        return f"Stage {self.index} [{self.strategy.value:9s}]: {body}{suffix}"


@dataclass(frozen=True)
class TrainingSchedule:
    """An ordered list of stages. Build it with :meth:`from_graph`."""

    stages: tuple[Stage, ...]

    # -- construction --------------------------------------------------------

    @classmethod
    def from_graph(
        cls,
        graph: MixedGraph | None,
        hierarchy: Hierarchy | None,
        roles: Mapping[str, NodeRole],
        *,
        max_batch_size: int = 1,
    ) -> TrainingSchedule:
        """Derive the schedule from the window graph, the hierarchy, and the roles.

        Parameters
        ----------
        graph
            The window graph over lagged node names, or ``None`` when no node
            reads another. Only the edges into a node's present-time copy count
            as dependencies.
        hierarchy
            The aggregation tree, or ``None`` when there is none.
        roles
            One :class:`NodeRole` per variable. The names are plain variable
            names, not lagged ones.
        max_batch_size
            How many independent nodes may share one MCMC stage. One (the
            default) gives each its own stage, which keeps each sampler small.
            A large number puts every node of a wave in one stage.

        Raises
        ------
        ValueError
            When a hierarchy parent carries a role other than ``AGGREGATE``
            (staged fitting reconciles bottom-up only), when an ``AGGREGATE``
            node has no children to add up, or when the modeled nodes depend on
            each other in a cycle.
        """
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        assignment = {str(name): NodeRole(role) for name, role in roles.items()}
        children = _hierarchy_children(hierarchy, assignment)
        _check_bottom_up(children, assignment)

        resolved = sorted(n for n, r in assignment.items() if r is NodeRole.SCENARIO)
        aggregates = sorted(n for n, r in assignment.items() if r is NodeRole.AGGREGATE)
        modeled = sorted(n for n, r in assignment.items() if r is NodeRole.MODEL)
        parents = _causal_parents(graph, modeled, set(assignment))
        depth = _depths(modeled, parents, set(modeled))

        stages: list[Stage] = []
        if resolved:
            stages.append(
                Stage(index=len(stages), nodes=tuple(resolved), strategy=StageStrategy.RESOLVED)
            )
        waves: dict[int, list[str]] = {}
        for node in modeled:
            waves.setdefault(depth[node], []).append(node)
        for level in sorted(waves):
            wave = sorted(waves[level])
            same_wave = set(wave)
            for start in range(0, len(wave), max_batch_size):
                batch = tuple(wave[start : start + max_batch_size])
                edges = {
                    node: tuple((base, lag) for base, lag in parents[node] if base not in same_wave)
                    for node in batch
                }
                stages.append(
                    Stage(
                        index=len(stages),
                        nodes=batch,
                        strategy=StageStrategy.MCMC,
                        parent_edges={n: e for n, e in edges.items() if e},
                    )
                )
        if aggregates:
            stages.append(
                Stage(
                    index=len(stages),
                    nodes=tuple(aggregates),
                    strategy=StageStrategy.AGGREGATE,
                    hierarchy_children={n: children[n] for n in aggregates},
                )
            )
        return cls(stages=tuple(stages))

    # -- queries -------------------------------------------------------------

    @property
    def mcmc_stages(self) -> tuple[Stage, ...]:
        """The stages that actually run a sampler."""
        return tuple(s for s in self.stages if s.strategy is StageStrategy.MCMC)

    @property
    def num_mcmc_stages(self) -> int:
        """How many sampler runs the schedule costs."""
        return len(self.mcmc_stages)

    @property
    def all_nodes(self) -> tuple[str, ...]:
        """Every node, in schedule order."""
        return tuple(node for stage in self.stages for node in stage.nodes)

    def stage_of(self, node: str) -> Stage | None:
        """The stage holding ``node``, or ``None`` when it is not scheduled."""
        for stage in self.stages:
            if node in stage.nodes:
                return stage
        return None

    def summary(self) -> str:
        """A printable plan, one line per stage."""
        lines = [f"TrainingSchedule: {len(self.stages)} stages, {self.num_mcmc_stages} by mcmc"]
        lines += [f"  {stage!r}" for stage in self.stages]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return self.summary()


# -- the pieces ---------------------------------------------------------------


def _hierarchy_children(
    hierarchy: Hierarchy | None, roles: Mapping[str, NodeRole]
) -> dict[str, tuple[str, ...]]:
    """Each parent's children that are actually part of the model."""
    if hierarchy is None:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for parent in hierarchy.internal_nodes:
        kids = tuple(c for c in hierarchy.children_of(parent) if c in roles)
        if kids:
            out[parent] = kids
    return out


def _check_bottom_up(
    children: Mapping[str, tuple[str, ...]], roles: Mapping[str, NodeRole]
) -> None:
    """Refuse a hierarchy parent that wants its own model, and an aggregate with no children."""
    for parent in children:
        if parent in roles and roles[parent] is not NodeRole.AGGREGATE:
            raise ValueError(
                f"{parent!r} is a hierarchy parent with role {roles[parent].value}; staged "
                "fitting aggregates bottom-up, so a parent of modeled children has to be "
                "NodeRole.AGGREGATE. Fit it jointly, or reconcile after the fact."
            )
    for node, role in roles.items():
        if role is NodeRole.AGGREGATE and node not in children:
            raise ValueError(f"{node!r} has role aggregate but no hierarchy children to add up")


def _causal_parents(
    graph: MixedGraph | None, modeled: list[str], known: set[str]
) -> dict[str, tuple[tuple[str, int], ...]]:
    """The ``(variable, lag)`` parents of each modeled node's present-time copy."""
    incoming: dict[str, set[tuple[str, int]]] = {node: set() for node in modeled}
    for edge in () if graph is None else graph.edges:
        pair = edge.directed_pair()
        if pair is None:
            continue
        source, target = pair
        base, lag = split_lag(target)
        if lag != 0 or base not in incoming:
            continue
        parent, parent_lag = split_lag(source)
        if parent in known and parent != base:
            incoming[base].add((parent, parent_lag))
    return {node: tuple(sorted(pairs)) for node, pairs in incoming.items()}


def _depths(
    modeled: list[str],
    parents: Mapping[str, tuple[tuple[str, int], ...]],
    model_set: set[str],
) -> dict[str, int]:
    """Wave index per modeled node: one deeper than its deepest modeled parent."""
    depth: dict[str, int] = {}
    visiting: list[str] = []

    def walk(node: str) -> int:
        if node in depth:
            return depth[node]
        if node in visiting:
            cycle = " -> ".join([*visiting[visiting.index(node) :], node])
            raise ValueError(f"the modeled nodes depend on each other in a cycle: {cycle}")
        visiting.append(node)
        level = 0
        for base, _lag in parents.get(node, ()):
            if base in model_set:
                level = max(level, walk(base) + 1)
        visiting.pop()
        depth[node] = level
        return level

    for node in modeled:
        walk(node)
    return depth
