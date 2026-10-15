# SPDX-License-Identifier: Apache-2.0
"""Staged fitting: one node at a time, posteriors handed down the graph.

A window graph turns a set of series into a set of *stages*. Nodes with no
modeled parent go first; their forecasts become the drivers of the next wave;
and so on down. :class:`~ergodic.forecast.TrainingSchedule` is the plan for that
(S1 built it); this module runs it.

What travels between stages is a posterior, not a point. When a child is fitted
its parents enter the regression design as **history**, plain data columns in the
parent's own scaled space, which costs nothing because nothing about the parent
is sampled again. When the child is forecast, each frozen parent's stored draws
are written into a copy of the child's trace under the name ``{base}_fc`` and
replayed from there, draw for matching draw. So the child's band carries the
parent's whole forecast distribution, and no step of this anywhere collapses a
parent to its median.

Three node roles decide what happens. A ``MODEL`` node is fitted. A ``SCENARIO``
node is given: its future arrives through ``scenario=`` or through a
:class:`~ergodic.forecast.ScenarioMechanism`'s own pinned values, and it is never
sampled. An ``AGGREGATE`` node is the sum of its hierarchy children and resolves
after every fit. The roles are computed here, from the mechanisms plus the graph
plus the hierarchy, and handed to the schedule.

What staged fitting drops, said plainly: the joint posterior *between* nodes.
Each node is fitted against its parents' observed history, so a child's
likelihood says nothing back about its parent's states. What survives is the
handoff, which is the part that matters for a forecast band.
:func:`~ergodic.forecast.forecast` picks the joint fit instead on a small graph,
where paying for the correlation is affordable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._fit import (
    FrozenParent,
    JointNode,
    NodeFit,
    NodeForecast,
    ParentColumn,
    align_draws,
    fit_joint,
    fit_node,
    forecast_joint,
    forecast_node,
    lag_align,
)
from ._mechanism import ScenarioMechanism, auto_scaler
from ._scalers import Scaler
from ._schedule import NodeRole, StageStrategy, TrainingSchedule, _causal_parents
from ._structure import split_lag, validate_window_graph

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ..data import Hierarchy, TimeSeriesData
    from ..graph import MixedGraph
    from ._mechanism import TemporalMechanism
    from ._schedule import Stage

    Array = npt.NDArray[np.float64]

__all__ = [
    "AUTO_JOINT_MAX_NODES",
    "FIT_STRATEGIES",
    "NodeState",
    "PosteriorStore",
    "StagedRun",
    "build_schedule",
    "design_keys",
    "mcmc_node_count",
    "node_roles",
    "resolve_scenarios",
    "resolve_strategy",
    "run_joint",
    "run_staged",
]

#: Every value ``fit=`` takes.
FIT_STRATEGIES = ("auto", "independent", "staged", "joint")

#: ``fit="auto"`` fits a graph jointly at or below this many modeled nodes.
#:
#: The joint model keeps the posterior correlation between nodes, and it costs one
#: sampler over everything at once. Four is where that stops being the cheaper
#: trade on the models this pillar fits; past it, staging is what makes a graph
#: affordable, and what it drops is the between-node correlation rather than the
#: handoff.
AUTO_JOINT_MAX_NODES = 4

#: The suffix that separates the extra lags of a parent that drives at several.
_LAG_SUFFIX = "_lag"


# -- the ledger ---------------------------------------------------------------


@dataclass(frozen=True)
class NodeState:
    """Everything the stage loop knows about one node.

    ``history`` is the node's own training series in the data's own units, and
    ``scaler`` is the scale a *child* reads it on. ``forecast`` is the horizon as
    ``(S, H)`` draws, also in the data's own units. ``path`` is set only when the
    horizon is a single pinned path rather than a distribution, which is what a
    scenario is; a child reading a pinned parent needs no injected site at all,
    and that is exactly why a pinned driver gives a narrower band than a
    forecast one.
    """

    node: str
    role: NodeRole
    scaler: Scaler
    history: Array | None = None
    forecast: Array | None = None
    path: Array | None = None
    fit: NodeFit | None = None
    view: NodeForecast | None = None

    @property
    def resolved(self) -> bool:
        """True once this node has a forecast to hand down."""
        return self.forecast is not None


class PosteriorStore:
    """The stage loop's working memory: one :class:`NodeState` per node.

    Deliberately mutable, and deliberately not a value object. It is the ledger a
    schedule fills in as it runs, and the thing a child asks for its parents'
    posteriors. Every state inside it is frozen, so an entry is replaced rather
    than edited.
    """

    def __init__(self, states: Mapping[str, NodeState] | None = None) -> None:
        self._states: dict[str, NodeState] = dict(states or {})

    def __contains__(self, node: object) -> bool:
        return node in self._states

    def __getitem__(self, node: str) -> NodeState:
        if node not in self._states:
            raise KeyError(f"no state for {node!r}; the store holds {sorted(self._states)}")
        return self._states[node]

    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self) -> Any:
        return iter(self._states)

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node in the store, sorted."""
        return tuple(sorted(self._states))

    def put(self, state: NodeState) -> None:
        """Record (or replace) one node's state."""
        self._states[state.node] = state

    def update(self, node: str, **changes: Any) -> NodeState:
        """Replace one node's state with a changed copy and return it."""
        state = replace(self[node], **changes)
        self._states[node] = state
        return state

    def __repr__(self) -> str:
        body = ", ".join(
            f"{name}:{state.role.value}{'' if state.resolved else ' (pending)'}"
            for name, state in sorted(self._states.items())
        )
        return f"PosteriorStore({body})"


@dataclass(frozen=True)
class StagedRun:
    """What a run of the schedule produced, before reconciliation reads it."""

    store: PosteriorStore
    schedule: TrainingSchedule
    series: Mapping[str, Array]
    fits: Mapping[str, NodeFit]
    views: Mapping[str, NodeForecast]
    diagnostics: Mapping[str, Any]


# -- roles, scenarios, and the schedule ---------------------------------------


def node_roles(
    data: TimeSeriesData,
    graph: MixedGraph | None,
    mechanisms: Mapping[str, TemporalMechanism],
    *,
    given: Mapping[str, TemporalMechanism] | None = None,
    scenario: Mapping[str, Any] | None = None,
    hierarchy: Hierarchy | None = None,
) -> dict[str, NodeRole]:
    """One :class:`~ergodic.forecast.NodeRole` per node, from the mechanisms and the graph.

    Three rules, in this order:

    - A node the hierarchy holds but the data has no column for is an
      ``AGGREGATE``: nothing models it, it is the sum of its children.
    - A node carrying a :class:`~ergodic.forecast.ScenarioMechanism`, or named in
      ``scenario=`` without the caller having named a mechanism for it, is a
      ``SCENARIO``: its future is given, so it is never fitted.
    - Everything else with a column is a ``MODEL``.

    ``given`` is the caller's own ``mechanisms=`` mapping, which is what makes the
    second rule readable: naming a node in ``scenario=`` pins its future, and
    unless you also handed it a mechanism there is no reason to spend a sampler on
    it. Hand it both and it stays a ``MODEL`` whose forecast segment is pinned,
    which is the variant that still costs a fit.
    """
    columns = set(data.variables)
    named = set(given or ())
    pinned = set(scenario or ())
    tree_nodes = set(hierarchy.nodes) if hierarchy is not None else set()
    graph_bases = {split_lag(node)[0] for node in graph.nodes} if graph is not None else set()

    # A hierarchy node with no column is not missing, it is an aggregate, and the
    # schedule has a better sentence for what goes wrong when one drives a model.
    missing = sorted(graph_bases - columns - tree_nodes)
    if missing:
        raise ValueError(
            f"the window graph names {missing}, which the data has no column for. A driver "
            "has to be a series of the dataset: its history is what a child regresses on, "
            "and no scenario can supply a past that was never recorded."
        )

    roles: dict[str, NodeRole] = {}
    for node in sorted(columns | tree_nodes | graph_bases):
        if node not in columns:
            roles[node] = NodeRole.AGGREGATE
            continue
        mechanism = mechanisms.get(node)
        if isinstance(mechanism, ScenarioMechanism) or (node in pinned and node not in named):
            roles[node] = NodeRole.SCENARIO
            continue
        roles[node] = NodeRole.MODEL
    return roles


def resolve_scenarios(
    data: TimeSeriesData,
    horizon: int,
    *,
    graph: MixedGraph | None,
    mechanisms: Mapping[str, TemporalMechanism],
    roles: Mapping[str, NodeRole],
    scenario: Mapping[str, Any] | None,
) -> dict[str, Array]:
    """Turn ``scenario=`` into one ``(H,)`` path per pinned node, refusing before any sampling.

    A value may be a length-``H`` array, a scalar (held at that value), or the
    string ``"hold"``, which repeats the last observed value. ``"hold"`` is only
    ever spelled out: freezing a driver is an assumption about the world, and an
    assumption nobody typed is one nobody checked.

    Four refusals, all of them before the first NUTS run:

    - a key the data has no column for;
    - a key on a node with graph parents, which is an intervention rather than a
      scenario (see the message);
    - a scenario node with nothing to resolve from;
    - a path of the wrong length.
    """
    pins = dict(scenario or {})
    unknown = sorted(set(pins) - set(data.variables))
    if unknown:
        raise ValueError(
            f"scenario= names {unknown}, which the data has no column for; it holds "
            f"{list(data.variables)}"
        )

    endogenous = _graph_parents(graph)
    named = sorted(node for node in pins if endogenous.get(node))
    if named:
        node = named[0]
        drivers = ", ".join(
            f"{base}[t-{lag}]" if lag else f"{base}[t]" for base, lag in endogenous[node]
        )
        raise ValueError(
            f"scenario={{{node!r}: ...}} pins a node the graph says is driven by {drivers}. "
            "scenario= is for a node whose future is exogenous or already known, and it "
            "makes no causal claim: it pins a path and trusts the fitted regressions "
            "downstream. Pinning an endogenous node is an intervention, which needs the "
            "effect to be identified on the window graph; forecast() does not support "
            "interventions. Forecast this node from its parents, or drop the edges into "
            "it if it really is exogenous."
        )

    out: dict[str, Array] = {}
    for node, value in pins.items():
        out[node] = _one_path(node, value, data, horizon)
    for node, role in roles.items():
        if role is not NodeRole.SCENARIO or node in out:
            continue
        mechanism = mechanisms.get(node)
        values = getattr(mechanism, "values", None)
        if values is None:
            raise ValueError(
                f"driver {node!r} needs a mechanism or scenario=: it carries a "
                "ScenarioMechanism, which is resolved rather than sampled, and nothing "
                f"supplied its future. Either pass scenario={{{node!r}: ...}} (an array, a "
                "scalar, or 'hold' to repeat the last observed value), pin it at "
                f"construction with ScenarioMechanism(values=...), or give {node!r} a "
                "StructuralMechanism so it is forecast like any other series."
            )
        out[node] = _one_path(node, np.asarray(values, dtype=np.float64), data, horizon)
    return out


def _one_path(node: str, value: Any, data: TimeSeriesData, horizon: int) -> Array:
    """One scenario entry as a length-``H`` path in the node's own units."""
    if isinstance(value, str):
        if value != "hold":
            raise ValueError(
                f"scenario[{node!r}] is {value!r}; the only string a scenario takes is 'hold', "
                "which repeats the last observed value. Pass an array or a scalar for anything "
                "else."
            )
        series = np.asarray(data.column(node), dtype=np.float64)
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            raise ValueError(f"scenario[{node!r}] is 'hold' but {node!r} has no observed value")
        return np.full(int(horizon), float(finite[-1]), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(int(horizon), float(array), dtype=np.float64)
    if array.shape != (int(horizon),):
        raise ValueError(
            f"scenario[{node!r}] has shape {array.shape}, expected ({horizon},): a scenario "
            "covers exactly the horizon"
        )
    return array


def mcmc_node_count(roles: Mapping[str, NodeRole]) -> int:
    """How many nodes a sampler will have to fit, which is what ``fit="auto"`` reads.

    Read off the roles rather than off the schedule, because the schedule needs to
    know which strategy is running before it can be built (a joint plan lifts the
    hierarchy rule that a staged one enforces) and the count cannot wait for it.
    Every ``MODEL`` node lands in exactly one MCMC stage, so the two agree.
    """
    return sum(1 for role in roles.values() if role is NodeRole.MODEL)


def resolve_strategy(requested: str, graph: MixedGraph | None, n_mcmc: int) -> str:
    """Turn ``fit=`` into the strategy that will actually run.

    ``"auto"`` is independent without a graph (with nothing shared, a joint model
    is a bigger sampler for the same answer), and with one it is joint at or under
    :data:`AUTO_JOINT_MAX_NODES` modeled nodes and staged above it.
    """
    if requested not in FIT_STRATEGIES:
        raise ValueError(f"fit must be one of {FIT_STRATEGIES}, got {requested!r}")
    if requested != "auto":
        return requested
    if graph is None:
        # Staged over an empty graph is one independent fit per node, which is what
        # this reduces to; it is spelled "staged" so the scenario path has a home.
        return "staged"
    return "joint" if n_mcmc <= AUTO_JOINT_MAX_NODES else "staged"


def build_schedule(
    data: TimeSeriesData,
    graph: MixedGraph | None,
    mechanisms: Mapping[str, TemporalMechanism],
    *,
    given: Mapping[str, TemporalMechanism] | None = None,
    scenario: Mapping[str, Any] | None = None,
    hierarchy: Hierarchy | None = None,
    fit: str = "auto",
    max_batch_size: int = 1,
) -> tuple[TrainingSchedule, dict[str, NodeRole], str]:
    """Roles, then the strategy, then the schedule. The order of every run.

    All three come back, because the strategy is what decides one of the
    schedule's own rules (a joint plan may carry a hierarchy parent that models
    itself, a staged plan may not), and because everything downstream needs the
    roles. Computing any of them twice is how two of them drift apart.
    """
    roles = node_roles(
        data,
        graph,
        mechanisms,
        given=given,
        scenario=scenario,
        hierarchy=hierarchy,
    )
    strategy = resolve_strategy(fit, graph, mcmc_node_count(roles))
    checked = None if graph is None else validate_window_graph(graph)
    plan = TrainingSchedule.from_graph(
        checked,
        hierarchy,
        roles,
        max_batch_size=max_batch_size,
        staged=strategy != "joint",
    )
    return plan, roles, strategy


def _graph_parents(graph: MixedGraph | None) -> dict[str, tuple[tuple[str, int], ...]]:
    """Every node's graph parents, by the schedule's own two rules, for the guards to read."""
    if graph is None:
        return {}
    bases = sorted({split_lag(node)[0] for node in graph.nodes})
    parents, _dropped = _causal_parents(graph, bases, set(bases))
    return {node: edges for node, edges in parents.items() if edges}


# -- the design ---------------------------------------------------------------


def design_keys(edges: Sequence[tuple[str, int]]) -> dict[tuple[str, int], str]:
    """Name each ``(parent, lag)`` regressor, keeping the plain name for the common case.

    A parent that drives a node at one lag keeps its own variable name, so the
    coefficient reads ``y_beta_pmi`` and the S2 naming contract is untouched. A
    parent that drives at several lags keeps the plain name for the *shortest*
    (the one a reader means when they say "the pmi coefficient") and suffixes the
    rest as ``{base}_lag{k}``. The rule is deterministic and does not depend on
    what else is in the graph.

    A collision (a real column named ``pmi_lag2`` alongside ``pmi`` at lag two)
    raises rather than quietly merging two regressors.
    """
    by_base: dict[str, list[int]] = defaultdict(list)
    for base, lag in edges:
        by_base[base].append(int(lag))
    out: dict[tuple[str, int], str] = {}
    for base, lags in by_base.items():
        for position, lag in enumerate(sorted(set(lags))):
            out[(base, lag)] = base if position == 0 else f"{base}{_LAG_SUFFIX}{lag}"
    taken: dict[str, tuple[str, int]] = {}
    for edge, key in sorted(out.items()):
        if key in taken:
            raise ValueError(
                f"the regressors {taken[key]} and {edge} would both be called {key!r}; "
                f"rename the column {edge[0]!r} or {taken[key][0]!r} so the two coefficients "
                "stay apart"
            )
        taken[key] = edge
    return out


def _designs(
    edges: Sequence[tuple[str, int]],
    store: PosteriorStore,
    horizon: int,
    *,
    n_draws: int,
    seed: int,
) -> tuple[dict[str, Array], dict[str, Array], tuple[FrozenParent, ...]]:
    """Fit columns, forecast columns, and frozen handoffs for one child's parents.

    Every column is on the **parent's** scaled space, which is the space its
    coefficient is written against. The split is by what the parent's future is: a
    pinned path is data and rides in the forecast columns; a forecast is a
    posterior and rides in the handoffs.
    """
    keys = design_keys(edges)
    fit_columns: dict[str, Array] = {}
    forecast_columns: dict[str, Array] = {}
    frozen: list[FrozenParent] = []
    by_base: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (base, lag), key in sorted(keys.items()):
        by_base[base].append((key, lag))

    for base, columns in sorted(by_base.items()):
        state = store[base]
        if state.history is None:
            raise ValueError(f"{base!r} has no history, so nothing can regress on it")
        history = np.asarray(state.scaler.transform(state.history), dtype=np.float64)
        for key, lag in columns:
            fit_columns[key] = lag_align(history, None, lag, 0)
        if state.path is not None:
            future = np.asarray(state.scaler.transform(state.path), dtype=np.float64)
            for key, lag in columns:
                forecast_columns[key] = lag_align(history, future, lag, horizon)
            continue
        if state.forecast is None:
            raise ValueError(
                f"{base!r} has no forecast yet, so it cannot be handed down; the schedule "
                "should have resolved it in an earlier stage"
            )
        draws = align_draws(
            np.asarray(state.scaler.transform(state.forecast), dtype=np.float64),
            n_draws,
            seed=seed,
        )
        frozen.append(
            FrozenParent(
                base=base,
                draws=draws,
                history=history,
                columns=tuple(columns),
            )
        )
    return fit_columns, forecast_columns, tuple(frozen)


# -- the runs -----------------------------------------------------------------


def _initial_store(
    data: TimeSeriesData,
    roles: Mapping[str, NodeRole],
    paths: Mapping[str, Array],
    *,
    n_draws: int,
) -> PosteriorStore:
    """The store before any stage runs: histories, scenario futures, scenario scales."""
    store = PosteriorStore()
    for node, role in sorted(roles.items()):
        history = (
            np.asarray(data.column(node), dtype=np.float64) if node in data.variables else None
        )
        path = paths.get(node)
        if role is NodeRole.SCENARIO:
            if history is None:  # pragma: no cover - node_roles refuses this first
                raise ValueError(f"{node!r} is a scenario node with no column")
            if path is None:  # pragma: no cover - resolve_scenarios refuses this first
                raise ValueError(f"{node!r} is a scenario node with no resolved path")
            # A scenario node's scaler is fitted on its own observed history rather
            # than taken from its mechanism. The mechanism's identity scaler is
            # about the path arriving in the caller's units, which it does; this
            # scaler is a different job, the scale a child's coefficient is written
            # against, and a driver in the thousands needs it as much as any other
            # series does.
            store.put(
                NodeState(
                    node=node,
                    role=role,
                    scaler=auto_scaler(history[np.isfinite(history)]),
                    history=history,
                    forecast=np.tile(path, (n_draws, 1)),
                    path=path,
                )
            )
            continue
        # A MODEL node may carry a pinned path too (the pinned-path variant). It is
        # recorded from the start, because a child reads the pin rather than the
        # model's own horizon and a joint model needs to know that while it builds.
        store.put(NodeState(node=node, role=role, scaler=Scaler(), history=history, path=path))
    return store


def run_staged(
    data: TimeSeriesData,
    horizon: int,
    *,
    schedule: TrainingSchedule,
    mechanisms: Mapping[str, TemporalMechanism],
    roles: Mapping[str, NodeRole],
    paths: Mapping[str, Array],
    seeds: Mapping[str, int],
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 1,
    target_accept: float = 0.99,
) -> StagedRun:
    """Walk the schedule: fit each wave, hand its posteriors to the next.

    ``paths`` are the resolved scenario futures, ``seeds`` one seed per node. A
    node named in ``paths`` that is nevertheless a ``MODEL`` node is fitted and
    then has its forecast segment replaced by the pinned path: that is the
    pinned-path variant, which still costs a fit and still makes no causal claim.
    """
    n_draws = int(draws) * int(chains)
    store = _initial_store(data, roles, paths, n_draws=n_draws)
    fits: dict[str, NodeFit] = {}
    views: dict[str, NodeForecast] = {}
    diagnostics: dict[str, Any] = {}

    for stage in schedule.stages:
        if stage.strategy is StageStrategy.RESOLVED:
            continue
        if stage.strategy is StageStrategy.AGGREGATE:
            _aggregate(stage, store)
            continue
        for node in stage.nodes:
            edges = stage.parent_edges.get(node, ())
            seed = seeds[node]
            fit_columns, forecast_columns, frozen = _designs(
                edges, store, int(horizon), n_draws=n_draws, seed=seed + 5000
            )
            values = np.asarray(data.column(node), dtype=np.float64)
            fitted = fit_node(
                node,
                values,
                mechanisms[node],
                parents=fit_columns,
                draws=draws,
                warmup=warmup,
                chains=chains,
                seed=seed,
                target_accept=target_accept,
            )
            view = forecast_node(
                fitted,
                int(horizon),
                parents=forecast_columns,
                frozen=frozen,
                seed=seed + 1000,
            )
            fits[node] = fitted
            views[node] = view
            diagnostics[node] = fitted.diagnostics
            pinned = paths.get(node)
            store.update(
                node,
                scaler=fitted.scaler,
                fit=fitted,
                view=view,
                forecast=np.tile(pinned, (n_draws, 1)) if pinned is not None else view.draws,
                path=pinned,
            )

    series = {node: store[node].forecast for node in store.nodes if store[node].resolved}
    return StagedRun(
        store=store,
        schedule=schedule,
        series={node: np.asarray(values) for node, values in series.items()},
        fits=fits,
        views=views,
        diagnostics=diagnostics,
    )


def run_joint(
    data: TimeSeriesData,
    horizon: int,
    *,
    schedule: TrainingSchedule,
    mechanisms: Mapping[str, TemporalMechanism],
    roles: Mapping[str, NodeRole],
    paths: Mapping[str, Array],
    seed: int = 0,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 1,
    target_accept: float = 0.99,
) -> StagedRun:
    """Fit every modeled node in one model, reading upstream mean paths directly.

    The schedule still decides the order (a parent has to be built before the
    child that reads it) and still decides what is fitted at all; what changes is
    that the waves land in one sampler instead of several, so the nodes keep their
    joint posterior. Scenario nodes enter as data, exactly as they do staged.
    """
    n_draws = int(draws) * int(chains)
    store = _initial_store(data, roles, paths, n_draws=n_draws)
    slots: list[JointNode] = []
    for stage in schedule.mcmc_stages:
        for node in stage.nodes:
            values = np.asarray(data.column(node), dtype=np.float64)
            observed = np.isfinite(values)
            mechanism = mechanisms[node]
            scaler = mechanism.scaler_for(values[observed])
            store.update(node, scaler=scaler)
            slots.append(
                JointNode(
                    node=node,
                    mechanism=mechanism,
                    scaler=scaler,
                    obs_scaled=scaler.transform(np.where(observed, values, np.nan)),
                    mask=observed,
                    parents=_joint_columns(stage.parent_edges.get(node, ()), store, horizon),
                )
            )
    if not slots:
        raise ValueError("a joint fit needs at least one modeled node")

    fitted = fit_joint(
        slots,
        draws=draws,
        warmup=warmup,
        chains=chains,
        seed=seed,
        target_accept=target_accept,
    )
    forecasts = forecast_joint(fitted, int(horizon), seed=seed + 1000)

    fits: dict[str, NodeFit] = {}
    views: dict[str, NodeForecast] = {}
    diagnostics: dict[str, Any] = {}
    for slot in slots:
        node = slot.node
        # One NodeFit per node over the one shared trace: provenance in the shape
        # every other path uses, so a caller reads `result.fits[node].trace` the
        # same way whichever strategy ran. It is a view, not a handle to refit
        # from: the parents it read were tensors of the joint model.
        fits[node] = NodeFit(
            node=node,
            mechanism=slot.mechanism,
            scaler=slot.scaler,
            trace=fitted.trace,
            diagnostics=fitted.diagnostics[node],
            obs_scaled=slot.obs_scaled,
            mask=slot.mask,
            parents={},
        )
        views[node] = forecasts[node]
        diagnostics[node] = fitted.diagnostics[node]
        pinned = paths.get(node)
        store.update(
            node,
            fit=fits[node],
            view=forecasts[node],
            forecast=(
                np.tile(pinned, (n_draws, 1)) if pinned is not None else forecasts[node].draws
            ),
            path=pinned,
        )

    for stage in schedule.stages:
        if stage.strategy is StageStrategy.AGGREGATE:
            _aggregate(stage, store)

    series = {node: store[node].forecast for node in store.nodes if store[node].resolved}
    return StagedRun(
        store=store,
        schedule=schedule,
        series={node: np.asarray(values) for node, values in series.items()},
        fits=fits,
        views=views,
        diagnostics=diagnostics,
    )


def _joint_columns(
    edges: Sequence[tuple[str, int]], store: PosteriorStore, horizon: int
) -> tuple[ParentColumn, ...]:
    """One node's regressors inside a joint model: upstream means, or scenario data."""
    keys = design_keys(edges)
    columns: list[ParentColumn] = []
    for (base, lag), key in sorted(keys.items()):
        state = store[base]
        # A pinned parent is data whichever role it carries: the pin is what the
        # child reads, and reading the model's own horizon instead would make the
        # joint answer differ from the staged one for a reason nobody asked for.
        if state.path is None and state.role is NodeRole.MODEL:
            columns.append(ParentColumn(key=key, base=base, lag=lag, upstream=True))
            continue
        if state.history is None or state.path is None:  # pragma: no cover - guarded upstream
            raise ValueError(f"{base!r} has neither a model nor a resolved path")
        history = np.asarray(state.scaler.transform(state.history), dtype=np.float64)
        future = np.asarray(state.scaler.transform(state.path), dtype=np.float64)
        columns.append(ParentColumn(key=key, base=base, lag=lag, history=history, future=future))
    return tuple(columns)


def _aggregate(stage: Stage, store: PosteriorStore) -> None:
    """Fill the aggregate stage: every parent is the sum of the children below it."""
    for parent in stage.nodes:
        kids = stage.hierarchy_children[parent]
        missing = [kid for kid in kids if not store[kid].resolved]
        if missing:  # pragma: no cover - the schedule orders children first
            raise ValueError(f"{parent!r} cannot be aggregated: {missing} have no forecast")
        stacked = [np.asarray(store[kid].forecast, dtype=np.float64) for kid in kids]
        store.update(parent, forecast=np.asarray(np.sum(stacked, axis=0), dtype=np.float64))
