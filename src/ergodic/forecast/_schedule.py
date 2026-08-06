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

Two rules decide which edges of the window graph become dependencies, and they
differ because time is doing the work in one case and not the other.

- A **cross-slice** edge counts, whatever marks it carries. ``pmi[t-1] o-> y[t]``
  is a dependency exactly like ``pmi[t-1] -> y[t]``: the circle says the causal
  direction is not established, but time already settles which node comes first,
  and forecasting reads the past to predict the present. It does not need the
  unconfoundedness claim that a causal reading would.
- A **same-slice** edge counts only when it is fully directed. ``x[t] --- y[t]``
  and ``x[t] o-> y[t]`` genuinely do not say which node is the parent, so
  neither direction enters the schedule. Those pairs are dropped, but never
  silently: they are counted in ``unoriented_dropped`` and warned about.

Two things the schedule refuses rather than getting quietly wrong. A model node
that reads an aggregate is refused, because aggregates resolve last and the
posterior would not exist when the stage runs. And nodes that drive each other
in a loop, including feedback across lags (``x[t-1] -> y[t]`` with
``y[t-1] -> x[t]``), are refused, because each staged node forecasts its whole
horizon in one pass and neither could go first.
"""

from __future__ import annotations

import warnings
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
    """An ordered list of stages. Build it with :meth:`from_graph`.

    ``unoriented_dropped`` holds the same-slice pairs the graph left unoriented,
    which is what the schedule could not use. It is a record, not an error: the
    fix is to orient them with domain knowledge and build the schedule again.
    """

    stages: tuple[Stage, ...]
    unoriented_dropped: tuple[tuple[str, str], ...] = ()

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
            as dependencies, and only under the two rules in the module
            docstring. Every variable the graph names needs a role.
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
            node has no children to add up, when the graph names a variable with
            no role, when a model node reads an aggregate, or when the modeled
            nodes drive each other in a loop.

        Warns
        -----
        UserWarning
            When the graph leaves a same-slice pair unoriented, so the schedule
            cannot tell which way the dependency runs.
        """
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        assignment = {str(name): NodeRole(role) for name, role in roles.items()}
        _check_roles_cover(graph, assignment)
        children = _hierarchy_children(hierarchy, assignment)
        _check_bottom_up(children, assignment)

        resolved = sorted(n for n, r in assignment.items() if r is NodeRole.SCENARIO)
        aggregates = sorted(n for n, r in assignment.items() if r is NodeRole.AGGREGATE)
        modeled = sorted(n for n, r in assignment.items() if r is NodeRole.MODEL)
        parents, dropped = _causal_parents(graph, modeled, set(assignment))
        if dropped:
            pairs = ", ".join(f"{a} --- {b}" for a, b in dropped)
            warnings.warn(
                f"the window graph leaves {len(dropped)} same-slice pair(s) unoriented "
                f"({pairs}); an unoriented pair does not say which node is the parent, so "
                "neither direction enters the schedule. Orient them with domain knowledge "
                "to use them.",
                UserWarning,
                stacklevel=2,
            )
        _check_no_aggregate_drivers(parents, assignment)
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
            ordered = _aggregate_order(set(aggregates), children)
            stages.append(
                Stage(
                    index=len(stages),
                    nodes=ordered,
                    strategy=StageStrategy.AGGREGATE,
                    hierarchy_children={n: children[n] for n in ordered},
                )
            )
        return cls(stages=tuple(stages), unoriented_dropped=dropped)

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
    def n_unoriented_dropped(self) -> int:
        """How many same-slice pairs the graph left unoriented, so the schedule skipped them."""
        return len(self.unoriented_dropped)

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


def _check_roles_cover(graph: MixedGraph | None, roles: Mapping[str, NodeRole]) -> None:
    """Refuse a graph that names a variable the roles say nothing about."""
    if graph is None:
        return
    missing = sorted({split_lag(node)[0] for node in graph.nodes} - set(roles))
    if missing:
        raise ValueError(
            f"the window graph names variables with no role: {missing}. Every variable in "
            "the graph needs a NodeRole, so the schedule knows whether to fit it, take it "
            "as given, or add it up."
        )


def _check_no_aggregate_drivers(
    parents: Mapping[str, tuple[tuple[str, int], ...]], roles: Mapping[str, NodeRole]
) -> None:
    """Refuse a model node that reads an aggregate, which resolves after it."""
    for node, edges in parents.items():
        for base, lag in edges:
            if roles.get(base) is NodeRole.AGGREGATE:
                arrow = f"{base}[t-{lag}]" if lag else f"{base}[t]"
                raise ValueError(
                    f"{node!r} reads {arrow}, and {base!r} is an aggregate of its hierarchy "
                    f"children. Staged fitting adds aggregates up last, after every model "
                    f"that might read them, so the posterior {node!r} needs would not exist "
                    f"when its stage runs. Give {base!r} its own mechanism and reconcile the "
                    f"hierarchy afterwards, or point {node!r} at the children instead; "
                    "aggregate-as-driver scheduling is future work."
                )


def _causal_parents(
    graph: MixedGraph | None, modeled: list[str], known: set[str]
) -> tuple[dict[str, tuple[tuple[str, int], ...]], tuple[tuple[str, str], ...]]:
    """The ``(variable, lag)`` parents of each modeled node, and the pairs left behind.

    Cross-slice edges count whatever their marks are, because time already says
    which end comes first. Same-slice edges count only when they are fully
    directed; the rest are returned as dropped pairs for the caller to report.
    """
    incoming: dict[str, set[tuple[str, int]]] = {node: set() for node in modeled}
    dropped: set[tuple[str, str]] = set()
    for edge in () if graph is None else graph.edges:
        u_base, u_lag = split_lag(edge.u)
        v_base, v_lag = split_lag(edge.v)
        if u_lag != v_lag:
            parent, lag = (u_base, u_lag) if u_lag > v_lag else (v_base, v_lag)
            child, child_lag = (v_base, v_lag) if u_lag > v_lag else (u_base, u_lag)
            if child_lag == 0 and child in incoming and parent in known and parent != child:
                incoming[child].add((parent, lag))
            continue
        if u_lag != 0 or u_base not in known or v_base not in known:
            continue
        pair = edge.directed_pair()
        if pair is None:
            if u_base in incoming or v_base in incoming:
                dropped.add((u_base, v_base) if u_base < v_base else (v_base, u_base))
            continue
        parent, child = split_lag(pair[0])[0], split_lag(pair[1])[0]
        if child in incoming and parent != child:
            incoming[child].add((parent, 0))
    return (
        {node: tuple(sorted(pairs)) for node, pairs in incoming.items()},
        tuple(sorted(dropped)),
    )


def _aggregate_order(
    aggregates: set[str], children: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Aggregates ordered so a nested one is added up before its parent needs it."""
    order: list[str] = []
    placed: set[str] = set()

    def visit(node: str) -> None:
        if node in placed or node not in aggregates:
            return
        placed.add(node)
        for child in sorted(children.get(node, ())):
            visit(child)
        order.append(node)

    for node in sorted(aggregates):
        visit(node)
    return tuple(order)


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
            loop = " -> ".join([*visiting[visiting.index(node) :], node])
            raise ValueError(
                f"the modeled nodes drive each other in a loop: {loop}. Staged fitting "
                "resolves one node at a time and each node forecasts its whole horizon in "
                "one pass, so every node in the loop would need another's horizon before "
                "that horizon exists. This includes feedback across lags, where the loop is "
                "not a cycle in time at all: x[t-1] -> y[t] together with y[t-1] -> x[t] is "
                "a perfectly ordinary process that staging still cannot serve. Feedback "
                "belongs in a joint multivariate mechanism, which is future work; until "
                "then, model one direction or fit those nodes jointly."
            )
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
