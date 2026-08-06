# SPDX-License-Identifier: Apache-2.0
"""The facade: data and a horizon in, a forecast out.

:func:`forecast` is the one call most users need. It reads the schema to pick a
mechanism per column, reads the time index to find a seasonal period, fits each
series, and continues it forward, handing back a :class:`ForecastResult` with the
draws and the provenance to explain where they came from.

Give it a ``graph`` and the series stop being independent. A window graph says
which series drive which, :func:`schedule` turns that into an order, and the fit
runs either one node at a time with each posterior handed down (staged) or every
node in one model (joint). A driver with its own mechanism is simply forecast and
its uncertainty propagates; a driver whose future is known arrives through
``scenario=``; a driver with neither is refused before anything samples.

The signature is the finished one. Everything it will eventually accept is
already a keyword, so no working call ever has to change shape later. The one
thing still missing is ``do=``, which is intervention rather than assumption and
waits for identification on window graphs; ``scenario=`` says by name when a
request is really that.

One name, two things, and the function wins. ``ergodic.forecast`` at the top
level is this function, so ``from ergodic import forecast`` gives something
callable. The subpackage is still reachable as ``from ergodic.forecast import
backtest``, because that goes through the import system rather than through the
attribute. What does not work the way it looks is ``import ergodic.forecast as
mod``: that binds the *function*, since Python prefers the attribute when one is
there. This is the only place in the package where a name is shadowed like this,
and it is deliberate: the facade is what people reach for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import HierarchicalTSData, TimeSeriesData
from ._fit import NodeFit, NodeForecast, fit_node, forecast_node, in_sample_median, sorted_seeds
from ._mechanism import assign_mechanisms
from ._reconcile import (
    RECONCILE_METHODS,
    _leaves_up,
    _require_sum,
    _series_of,
)
from ._reconcile import reconcile as reconcile_result
from ._result import FitStrategy, ForecastResult
from ._staged import (
    AUTO_JOINT_MAX_NODES,
    FIT_STRATEGIES,
    build_schedule,
    mcmc_node_count,
    resolve_scenarios,
    run_joint,
    run_staged,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from ..data import Hierarchy
    from ..graph import MixedGraph
    from ._mechanism import TemporalMechanism
    from ._schedule import TrainingSchedule
    from ._staged import StagedRun

    Array = npt.NDArray[np.float64]

__all__ = ["forecast", "schedule"]

#: Fitting strategies that reach the independent per-series path.
_INDEPENDENT_FITS = ("auto", "independent")
#: Strategies that read the graph.
_GRAPH_FITS = ("staged", "joint")
#: The strategies that read the training window for their shares.
_NEEDS_DATA = ("top_down", "middle_out")


def forecast(
    data: TimeSeriesData,
    horizon: int,
    *,
    graph: MixedGraph | None = None,
    mechanisms: Mapping[str, TemporalMechanism] | None = None,
    reconcile: str | None = None,
    proportions: str | None = None,
    level: int | None = None,
    weights: str | None = None,
    trust: Mapping[str, float] | None = None,
    scenario: Mapping[str, Any] | None = None,
    fit: str = "auto",
    robust: bool = False,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 1,
    seed: int = 0,
    target_accept: float = 0.99,
) -> ForecastResult:
    """Forecast every series in ``data`` ``horizon`` steps forward.

    Each column gets its own mechanism (a damped local linear trend with the
    season its time index implies) and its own NUTS fit, and each is continued by
    replaying its posterior and simulating only the horizon. The result carries
    the draws, the mechanisms that produced them, and the sampler diagnostics.

    Given a :class:`~ergodic.data.HierarchicalTSData`, every level is forecast
    the same way. A node with a column gets its own model, leaf or total alike,
    and a node that exists only as an aggregate is summed from its children. The
    levels are then made to agree by the strategy ``reconcile`` names, and the
    unreconciled version stays available through ``reconcile="none"``.

    Given a ``graph``, the series read each other. Each node's parents enter its
    regression as lag-shifted columns of their own history, and when the node is
    forecast, a parent that was itself forecast hands its whole posterior across
    rather than a path: draw for matching draw, so the child's band carries the
    driver's uncertainty. :func:`schedule` shows the order this will run in
    before any of it runs.

    Parameters
    ----------
    data:
        A :class:`~ergodic.data.TimeSeriesData`, or a
        :class:`~ergodic.data.HierarchicalTSData` for the hierarchical path.
    horizon:
        How many steps past the end of the data to forecast.
    graph:
        A window graph over lagged node names (``pmi[t-2] -> sales[t]``), saying
        which series drive which. Build one with
        :func:`~ergodic.forecast.window_graph`, or hand over the ``.graph`` of a
        temporal discovery result unchanged. Every variable it names has to be a
        column of the data.
    mechanisms:
        Per-column overrides. A column named here keeps the mechanism it is
        given; everything else is assigned from the schema. A
        :class:`~ergodic.forecast.ScenarioMechanism` marks a column as
        known-future, which means it is never fitted and ``scenario=`` has to
        supply its path.
    reconcile:
        The reconciliation strategy: ``"bottom_up"``, ``"top_down"``,
        ``"middle_out"``, ``"mint"``, or ``"none"``. Unset means bottom-up once
        a hierarchy exists and nothing at all on flat data, where naming a
        strategy raises rather than pretending there was something to reconcile.
    proportions:
        ``"historical"`` or ``"forecast"``, how ``top_down`` and ``middle_out``
        split a parent between its children.
    level:
        Which depth ``middle_out`` believes, the root being 0.
    weights:
        ``"posterior"`` or ``"mint_shrink"``, where ``mint`` gets its weights.
        The shrink rule reads each node's in-sample residuals, which this facade
        computes from the fits it just ran.
    trust:
        ``{node: weight}`` for ``mint``: a node trusted ten times as much moves
        about a tenth as far when the levels are squared up.
    scenario:
        Pinned future paths, ``{node: path}``, where a path is a length-``horizon``
        array, a scalar, or the string ``"hold"`` (repeat the last observed
        value). It makes no causal claim: it pins a path and trusts the fitted
        regressions downstream, which is honest for a node whose future is
        exogenous or already known and quietly wrong for one the graph says is
        driven, so the second case is refused by name.
    fit:
        ``"independent"`` fits each series alone. ``"staged"`` fits one node at a
        time and hands each posterior down the graph. ``"joint"`` fits every node
        in one model, keeping their posteriors correlated. ``"auto"`` is
        independent without a graph, and with one is joint at or under
        :data:`AUTO_JOINT_MAX_NODES` modeled nodes and staged above it. The
        strategy that ran, and the arithmetic behind an automatic choice, are on
        the result as ``strategy``.
    robust:
        Fit a Student-t likelihood rather than a Normal one, so a few outliers do
        not drag the level. Reaches the auto-assigned mechanisms only; a
        mechanism you built keeps its own setting.
    draws, warmup, chains, seed:
        The sampler budget and the seed. Each node's seed is ``seed`` plus its
        position in sorted node order, so a node's fit does not move when a
        sibling is added.
    target_accept:
        NUTS target acceptance. The default is high on purpose: a non-centered
        state-space model with a tight innovation scale funnels, and 0.99 is what
        keeps the divergence count near zero without slowing the fit much.

    Returns
    -------
    ForecastResult
        Draws of shape ``(draws * chains, horizon)`` per series, in the units the
        data arrived in. From a hierarchy it also carries ``reconciliation`` and
        ``coherent``, and its series run top of the tree downward.
    """
    if not isinstance(data, TimeSeriesData):
        raise TypeError(
            f"forecast() needs a TimeSeriesData, got {type(data).__name__}; build one with "
            "ergodic.timeseries(frame, time=...)"
        )
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if fit not in FIT_STRATEGIES:
        raise ValueError(f"fit must be one of {FIT_STRATEGIES}, got {fit!r}")
    if graph is None and fit in _GRAPH_FITS and scenario is None:
        raise ValueError(
            f"fit={fit!r} is about several series reading each other, and no graph says "
            "which reads which. Pass graph=window_graph([...]) (or a discovery result's "
            ".graph), or leave fit='auto' to forecast each series alone."
        )
    if graph is not None and fit == "independent":
        raise ValueError(
            "fit='independent' fits each series alone, which would ignore the graph you "
            "passed. Use fit='staged', fit='joint', or fit='auto' to read it, or drop "
            "graph= to forecast independently on purpose."
        )
    if scenario is not None and graph is None and fit == "independent":
        raise ValueError(
            "fit='independent' fits each series alone, so a scenario= path would pin a "
            "future nothing reads. Use fit='auto'."
        )

    options = {"proportions": proportions, "level": level, "weights": weights, "trust": trust}
    if not isinstance(data, HierarchicalTSData):
        named = sorted(name for name, value in options.items() if value is not None)
        if reconcile is not None:
            named = [f"reconcile={reconcile!r}", *named]
        if named:
            raise ValueError(
                f"{', '.join(named)} needs a hierarchy, and this dataset is flat: there is "
                "nothing to make coherent. Build a HierarchicalTSData with ergodic.hts(...) "
                "to reconcile across levels."
            )
    else:
        # All three checks happen before anything samples: neither a typo, nor a
        # tree this pillar cannot reconcile, nor a leaf nothing will forecast
        # should cost five NUTS runs first.
        _require_sum(data.hierarchy)
        if reconcile is not None and reconcile not in RECONCILE_METHODS:
            raise ValueError(f"reconcile must be one of {RECONCILE_METHODS}, got {reconcile!r}")
        modelless = [leaf for leaf in data.hierarchy.leaves if leaf not in data.variables]
        if modelless:
            raise ValueError(
                f"the hierarchy's leaves {modelless} are not modelled columns of this dataset, "
                "and a leaf has no children to be aggregated from, so nothing would forecast "
                "them. Include the columns (an excluded column is not modelled) or build the "
                "hierarchy without those leaves."
            )

    nodes = data.variables
    if not nodes:
        raise ValueError("the dataset models no columns, so there is nothing to forecast")
    assigned = assign_mechanisms(data, mechanisms, robust=robust)
    seeds = sorted_seeds(nodes, seed)
    tree = data.hierarchy if isinstance(data, HierarchicalTSData) else None

    if graph is not None or scenario is not None or fit in _GRAPH_FITS:
        return _through_the_graph(
            data,
            int(horizon),
            graph=graph,
            given=mechanisms,
            assigned=assigned,
            scenario=scenario,
            tree=tree,
            requested=fit,
            reconcile=reconcile,
            proportions=proportions,
            level=level,
            weights=weights,
            trust=trust,
            seeds=seeds,
            seed=seed,
            draws=draws,
            warmup=warmup,
            chains=chains,
            target_accept=target_accept,
        )

    fits: dict[str, NodeFit] = {}
    views: dict[str, NodeForecast] = {}
    diagnostics: dict[str, Any] = {}
    for node in nodes:
        values = np.asarray(data.column(node), dtype=np.float64)
        fitted = fit_node(
            node,
            values,
            assigned[node],
            draws=draws,
            warmup=warmup,
            chains=chains,
            seed=seeds[node],
            target_accept=target_accept,
        )
        # A separate seed for the predictive pass, offset far enough that it can
        # never collide with another node's fit seed.
        fits[node] = fitted
        views[node] = forecast_node(fitted, int(horizon), seed=seeds[node] + 1000)
        diagnostics[node] = fitted.diagnostics

    series: dict[str, Array] = {node: views[node].draws for node in nodes}
    return _assemble(
        data,
        int(horizon),
        series=series,
        assigned=assigned,
        diagnostics=diagnostics,
        fits=fits,
        views=views,
        reconcile=reconcile,
        proportions=proportions,
        level=level,
        weights=weights,
        trust=trust,
        strategy=FitStrategy(strategy="independent", requested=fit),
    )


def _through_the_graph(
    data: TimeSeriesData,
    horizon: int,
    *,
    graph: MixedGraph | None,
    given: Mapping[str, TemporalMechanism] | None,
    assigned: Mapping[str, TemporalMechanism],
    scenario: Mapping[str, Any] | None,
    tree: Hierarchy | None,
    requested: str,
    reconcile: str | None,
    proportions: str | None,
    level: int | None,
    weights: str | None,
    trust: Mapping[str, float] | None,
    seeds: Mapping[str, int],
    seed: int,
    draws: int,
    warmup: int,
    chains: int,
    target_accept: float,
) -> ForecastResult:
    """The graph path: roles, then a schedule, then staged or joint fitting.

    Everything that can be refused is refused here, before the first sampler
    starts: a graph naming a column the data does not have, a scenario key that
    is really an intervention, a driver whose future nobody supplied, a hierarchy
    parent that wants its own model under staged fitting.
    """
    plan, roles, chosen = build_schedule(
        data,
        graph,
        assigned,
        given=given,
        scenario=scenario,
        hierarchy=tree,
        fit=requested,
    )
    paths = resolve_scenarios(
        data,
        horizon,
        graph=graph,
        mechanisms=assigned,
        roles=roles,
        scenario=scenario,
    )
    n_mcmc = mcmc_node_count(roles)

    run: StagedRun
    if chosen == "joint":
        run = run_joint(
            data,
            horizon,
            schedule=plan,
            mechanisms=assigned,
            roles=roles,
            paths=paths,
            seed=seed,
            draws=draws,
            warmup=warmup,
            chains=chains,
            target_accept=target_accept,
        )
    else:
        run = run_staged(
            data,
            horizon,
            schedule=plan,
            mechanisms=assigned,
            roles=roles,
            paths=paths,
            seeds=seeds,
            draws=draws,
            warmup=warmup,
            chains=chains,
            target_accept=target_accept,
        )
    return _assemble(
        data,
        horizon,
        series=dict(run.series),
        assigned=assigned,
        diagnostics=dict(run.diagnostics),
        fits=run.fits,
        views=run.views,
        reconcile=reconcile,
        proportions=proportions,
        level=level,
        weights=weights,
        trust=trust,
        strategy=FitStrategy(
            strategy=chosen,
            requested=requested,
            n_mcmc_nodes=n_mcmc,
            auto_threshold=AUTO_JOINT_MAX_NODES,
        ),
        graph=graph,
        schedule=plan,
    )


def _assemble(
    data: TimeSeriesData,
    horizon: int,
    *,
    series: dict[str, Array],
    assigned: Mapping[str, TemporalMechanism],
    diagnostics: Mapping[str, Any],
    fits: Mapping[str, NodeFit],
    views: Mapping[str, NodeForecast],
    reconcile: str | None,
    proportions: str | None,
    level: int | None,
    weights: str | None,
    trust: Mapping[str, float] | None,
    strategy: FitStrategy,
    graph: MixedGraph | None = None,
    schedule: TrainingSchedule | None = None,
) -> ForecastResult:
    """Wrap the draws in a result, and reconcile the levels when there is a hierarchy."""
    if not isinstance(data, HierarchicalTSData):
        return ForecastResult(
            series=series,
            horizon=horizon,
            kind="draws",
            start=_next_label(data),
            mechanisms=assigned,
            graph=graph,
            schedule=schedule,
            strategy=strategy,
            fits=dict(fits),
            diagnostics=diagnostics,
        )

    tree = data.hierarchy
    _fill_aggregates(tree, series)
    unreconciled = ForecastResult(
        series={node: series[node] for node in _hierarchy_order(tree, tuple(series))},
        horizon=horizon,
        kind="draws",
        start=_next_label(data),
        mechanisms=assigned,
        graph=graph,
        schedule=schedule,
        strategy=strategy,
        fits=dict(fits),
        diagnostics=diagnostics,
        # Always, not only when MinT-shrink asks: it costs one median over draws
        # of a path the predictive pass already produced, and carrying it is what
        # lets a result be reconciled every way afterwards without refitting.
        residual_var=_residual_variance(data, tree, fits, views),
    )
    method = reconcile or "bottom_up"
    return reconcile_result(
        unreconciled,
        tree,
        method=method,
        data=data if method in _NEEDS_DATA else None,
        proportions=proportions,
        level=level,
        weights=weights,
        trust=trust,
    )


def schedule(
    data: TimeSeriesData,
    graph: MixedGraph | None = None,
    mechanisms: Mapping[str, TemporalMechanism] | None = None,
    *,
    scenario: Mapping[str, Any] | None = None,
    fit: str = "auto",
) -> TrainingSchedule:
    """The plan :func:`forecast` would follow, without fitting anything.

    Reads the same three things the forecast reads (the mechanisms, the graph,
    the hierarchy) and returns the
    :class:`~ergodic.forecast.TrainingSchedule`: which nodes are given, which are
    fitted and in what order, which are summed at the end, and what each fitted
    node reads from the waves before it. Print it before spending a sampler on
    it.

    It refuses everything the forecast would refuse about the *structure*: a
    graph naming a column that is not there, a hierarchy parent that wants its
    own model under staged fitting, a model node reading an aggregate, nodes that
    drive each other in a loop. What it cannot see is anything about the horizon,
    so a missing scenario path is still the forecast's to catch.

    ``fit`` matters here for one reason: a joint plan may carry a hierarchy
    parent that models itself and a staged plan may not, so passing the same
    ``fit`` the forecast will get is what makes this the plan that actually runs.
    """
    if not isinstance(data, TimeSeriesData):
        raise TypeError(
            f"schedule() needs a TimeSeriesData, got {type(data).__name__}; build one with "
            "ergodic.timeseries(frame, time=...)"
        )
    tree = data.hierarchy if isinstance(data, HierarchicalTSData) else None
    assigned = assign_mechanisms(data, mechanisms)
    plan, _roles, _strategy = build_schedule(
        data,
        graph,
        assigned,
        given=mechanisms,
        scenario=scenario,
        hierarchy=tree,
        fit=fit,
    )
    return plan


def _hierarchy_order(tree: Hierarchy, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Tree nodes from the root down, then any column that is not part of the tree.

    Reading order rather than column order: a hierarchical result is easiest to
    read total first, and a series that is not in the hierarchy (a driver, later)
    keeps its place at the end.
    """
    depth = tree.levels()
    inside = sorted(tree.nodes, key=lambda node: (depth[node], node))
    outside = [name for name in columns if name not in tree.nodes]
    return (*inside, *outside)


def _fill_aggregates(tree: Hierarchy, series: dict[str, Array]) -> None:
    """Give every modelless node the only forecast it can have: its children's sum.

    A node with no column exists only as an aggregate, so its unreconciled draws
    are the sum of its children's. Deepest parent first, so a chain of
    aggregate-only levels fills from the bottom.
    """
    for parent in _leaves_up(tree):
        if parent in series:
            continue
        kids = tree.children_of(parent)
        missing = [kid for kid in kids if kid not in series]
        if missing:  # pragma: no cover - a leaf without a column cannot be built
            raise ValueError(
                f"{parent!r} has no column and its children {missing} have no forecast either, "
                "so there is nothing to aggregate"
            )
        series[parent] = np.sum([series[kid] for kid in kids], axis=0)


def _residual_variance(
    data: HierarchicalTSData,
    tree: Hierarchy,
    fits: Mapping[str, NodeFit],
    views: Mapping[str, NodeForecast],
) -> dict[str, float]:
    """Each node's in-sample one-step residual second moment, the MinT-shrink diagonal.

    The mean of the squared residuals, uncentered on purpose: MinT's W estimates
    ``E[e e']``, so a fit that sits systematically above its series is charged
    for that, where a centered variance would forgive it.

    A modeled node compares its own series against its posterior-median fitted
    path. A node that exists only as an aggregate has no fit, so its fitted path
    is its children's fitted paths added up, which makes its residual exactly the
    sum of its children's residuals: no independence assumption anywhere, just
    the same arithmetic the hierarchy already promises.
    """
    fitted: dict[str, Array] = {
        node: in_sample_median(fits[node], views[node]) for node in fits if node in tree.nodes
    }
    for leaf in tree.leaves:
        # A pinned leaf has no fit, and its fitted path is the series itself: a
        # supplied path has no residual, which is MinT trusting it completely, and
        # that is the right reading of "we already know this one".
        if leaf not in fitted:
            fitted[leaf] = _series_of(data, leaf)
    for parent in _leaves_up(tree):
        if parent in fitted:
            continue
        fitted[parent] = np.sum([fitted[kid] for kid in tree.children_of(parent)], axis=0)
    out: dict[str, float] = {}
    for node in tree.nodes:
        residual = _series_of(data, node) - fitted[node]
        out[node] = float(np.mean(residual**2))
    return out


def _next_label(data: TimeSeriesData) -> Any:
    """The time label one step past the end of the data, or ``None`` when unclear.

    The step is the median spacing of the index, which is right for a regular
    series and honest about being a guess for anything else.
    """
    times = np.asarray(data.times)
    if times.size < 2:
        return None
    if times.dtype.kind == "M":
        seconds = times.astype("datetime64[s]").astype(np.int64)
        step = int(np.median(np.diff(seconds)))
        return times[-1].astype("datetime64[s]") + np.timedelta64(step, "s")
    if times.dtype.kind in "iuf":
        values = times.astype(np.float64)
        return values[-1] + float(np.median(np.diff(values)))
    return None
