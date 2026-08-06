# SPDX-License-Identifier: Apache-2.0
"""The attribution waterfall: what a forecast says is behind the forecast.

A forecast is a number with a story in it. The story the model actually told is
recoverable, because the default mechanism is *composed* rather than fused: a
damped trend, a season, and one term per driver, multiplied and added in a stated
order. :meth:`~ergodic.forecast.ForecastResult.explain` takes that composition
apart again and returns one bar per piece, with a posterior behind each bar.

Two engines, one contract.

The **structural** engine is the primary one and costs nothing. A forecasting
model already records its pieces as deterministics (``{node}_trend``,
``{node}_season_factor``, ``{node}_reg_{driver}``), one array per posterior draw,
so switching a factor off is arithmetic on arrays that already exist. The
identity elements are the pinned parameterization rather than an approximation:
a driver contributes zero when it is off, and the monthly season is written as
``12 * softmax(raw)``, which is exactly one when the raw parameters are zero.

The **replay** engine is the fallback for a mechanism that does not expose its
pieces. It zeroes the factor's sites in a copy of the trace and forecasts again
with the *same seed*, so the fresh innovations come back identical and the
difference between two replays is the factor and nothing else. It costs one
posterior-predictive pass per bar, which is why it is the fallback.

Both report ``closure_error``, always, and it is not decoration. On the
structural path the bars are built by a composition rule that could drift away
from the mechanism's own ``build``; the closure measures them against the model's
recorded mean path, so drift shows up as a number. On the replay path it measures
whether the re-run reproduced the forecast it is explaining.

What the bars explain is the **mean path**, the signal the model fitted. The
forecast draws also carry observation noise, which averages to nothing, and, on a
reconciled result, whatever the reconciliation moved. Both gaps are visible:
``total`` is the decomposition's own total and ``forecast_total`` is what the
result holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._fit import forecast_node, posterior_draws, zero_sites
from ._mechanism import SEASON

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import numpy.typing as npt
    import pandas as pd

    from ._fit import NodeFit, NodeForecast
    from ._result import ForecastResult
    from ._scalers import Scaler

    Array = npt.NDArray[np.float64]

__all__ = [
    "Attribution",
    "FactorContribution",
    "NodeAttribution",
    "Window",
    "attribute",
]

#: The reduction applied per draw over a window: a flow is added up, a rate is
#: averaged. Getting this wrong is the difference between "the driver added 4,200
#: units this quarter" and "the driver added 4,200 units a month".
REDUCERS = ("sum", "mean")
#: How the bars are cut. ``sequential`` switches factors on one at a time in
#: composition order and books each step; ``loo`` switches each one off from the
#: full model and books the remainder as interaction.
METHODS = ("sequential", "loo")
#: The label the interaction bar carries under ``method="loo"``.
INTERACTION = "Interaction"
#: How many posterior-predictive passes one :func:`attribute` call may spend
#: before it refuses. Only the replay engine spends any.
MAX_REPLAYS = 200

_KINDS = {SEASON: "season", INTERACTION: "interaction"}


# -- windows ------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A named, half-open slice of the horizon.

    ``start`` and ``end`` are either integer positions into the forecast steps
    (zero-based, ``end`` exclusive, ``None`` meaning "to the end of the horizon")
    or labels on the result's own time axis, which is what makes
    ``Window("Q1", "2026-01-01", "2026-04-01")`` mean what it looks like. A label
    resolves to the first step at or after it, so a window that starts between
    two steps starts at the later one.
    """

    label: str
    start: Any = 0
    end: Any | None = None

    def resolve(self, horizon: int, index: Sequence[Any] | None = None) -> tuple[int, int]:
        """The ``(start, end)`` positions this window covers, as integers."""
        first = self._position(self.start, horizon, index, default=0)
        last = horizon if self.end is None else self._position(self.end, horizon, index, default=0)
        if not 0 <= first < last <= horizon:
            raise ValueError(
                f"window {self.label!r} resolves to steps [{first}, {last}) of a "
                f"{horizon}-step horizon, which is empty or out of range"
            )
        return first, last

    @staticmethod
    def _position(value: Any, horizon: int, index: Sequence[Any] | None, *, default: int) -> int:
        """One endpoint as a position: an integer is one already, a label is looked up."""
        if value is None:
            return default
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        if index is None:
            raise ValueError(
                f"window {value!r} is a label, and this result carries no time axis to "
                "resolve it against (the data's index gave no spacing). Use integer "
                "positions into the horizon instead."
            )
        labels = np.asarray(index)
        try:
            probe = np.asarray(value, dtype=labels.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"window bound {value!r} does not read as a label on a time axis of "
                f"dtype {labels.dtype}"
            ) from exc
        return int(np.searchsorted(labels, probe, side="left"))


# -- the records --------------------------------------------------------------


@dataclass(frozen=True)
class FactorContribution:
    """One bar: how much of the window's total this piece of the model accounts for.

    ``draws`` is the per-draw contribution, so the bar carries a posterior rather
    than a point. :attr:`value` is its mean, which is what makes the bars close:
    means add up even when medians do not.
    """

    name: str
    kind: str
    draws: Array = field(repr=False)

    def __post_init__(self) -> None:
        values = np.array(self.draws, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError(f"a contribution's draws must be 1-D (S,), got {values.shape}")
        values.setflags(write=False)
        object.__setattr__(self, "draws", values)

    @property
    def value(self) -> float:
        """The posterior mean of this bar."""
        return float(np.mean(self.draws))

    def credible_interval(self, level: float = 0.9) -> tuple[float, float]:
        """The equal-tailed interval of this bar at ``level``."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        lo, hi = np.percentile(self.draws, [tail * 100.0, (1.0 - tail) * 100.0])
        return float(lo), float(hi)

    def __repr__(self) -> str:
        return f"FactorContribution({self.name}: {self.value:+,.4g})"


@dataclass(frozen=True)
class NodeAttribution:
    """One node's waterfall over one window.

    The bars are :attr:`baseline` (the node's own dynamics, whatever the
    mechanism calls them) followed by :attr:`factors`, in the order they were
    composed. They add up to :attr:`total`, the model's mean path aggregated over
    the window; :attr:`closure_error` is what is left over, and it should be
    rounding.

    :attr:`forecast_total` is the same window read off the result's own draws. It
    differs from :attr:`total` by the observation noise (mean zero, so the gap is
    Monte Carlo scale) and, on a reconciled result, by whatever reconciliation
    moved. A decomposition explains a model, and reconciliation happens after the
    model.
    """

    node: str
    window: Window
    steps: tuple[int, int]
    method: str
    reduce: str
    engine: str
    baseline: FactorContribution
    factors: tuple[FactorContribution, ...]
    total_draws: Array = field(repr=False)
    forecast_draws: Array = field(repr=False)
    n_replays: int = 0

    @property
    def bars(self) -> tuple[FactorContribution, ...]:
        """Every bar of the waterfall, the baseline first."""
        return (self.baseline, *self.factors)

    @property
    def total(self) -> float:
        """The window's total under the model's own mean path."""
        return float(np.mean(self.total_draws))

    @property
    def forecast_total(self) -> float:
        """The same window read off the result's forecast draws."""
        return float(np.mean(self.forecast_draws))

    @property
    def closure_error(self) -> float:
        """``total`` minus the bars, in the units of the series.

        Reported always. Read it against :attr:`total`, or use
        :attr:`relative_closure_error`, which does that for you: a residual left
        by floating-point addition scales with the numbers being added, so an
        absolute threshold means different things on a series in units and the
        same series in millions.
        """
        return float(self.total - sum(bar.value for bar in self.bars))

    @property
    def relative_closure_error(self) -> float:
        """:attr:`closure_error` against the size of the total, floored at one."""
        return abs(self.closure_error) / max(1.0, abs(self.total))

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """The window total's own credible interval."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        lo, hi = np.percentile(self.total_draws, [tail * 100.0, (1.0 - tail) * 100.0])
        return float(lo), float(hi)

    def to_frame(self, level: float = 0.9) -> pd.DataFrame:
        """One row per bar: name, kind, value, band, and share of the total."""
        import pandas as pd

        total = self.total
        rows: list[dict[str, Any]] = []
        for bar in self.bars:
            lo, hi = bar.credible_interval(level)
            rows.append(
                {
                    "node": self.node,
                    "window": self.window.label,
                    "factor": bar.name,
                    "kind": bar.kind,
                    "value": bar.value,
                    "lo": lo,
                    "hi": hi,
                    "share": bar.value / total if total else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        bars = ", ".join(f"{bar.name} {bar.value:+,.4g}" for bar in self.bars)
        return (
            f"NodeAttribution({self.node} over {self.window.label}: total {self.total:,.4g} "
            f"= {bars}; closure {self.closure_error:+.2e}, {self.engine})"
        )


@dataclass(frozen=True)
class Attribution:
    """Every node and window one :meth:`ForecastResult.explain` call asked for."""

    nodes: tuple[NodeAttribution, ...]
    method: str = "sequential"
    reduce: str = "sum"

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[NodeAttribution]:
        return iter(self.nodes)

    def __getitem__(self, key: str | tuple[str, str]) -> NodeAttribution:
        """One node's attribution by name, or one ``(node, window)`` pair."""
        if isinstance(key, tuple):
            node, window = key
            for record in self.nodes:
                if record.node == node and record.window.label == window:
                    return record
            raise KeyError(f"no attribution for node {node!r} over window {window!r}")
        matches = [record for record in self.nodes if record.node == key]
        if not matches:
            raise KeyError(
                f"no attribution for {key!r}; this one holds "
                f"{sorted({record.node for record in self.nodes})}"
            )
        if len(matches) > 1:
            raise KeyError(
                f"{key!r} was explained over {len(matches)} windows "
                f"{[record.window.label for record in matches]}; "
                "index with (node, window) to pick one"
            )
        return matches[0]

    @property
    def node_names(self) -> tuple[str, ...]:
        """Every node explained, in order, without repeats."""
        return tuple(dict.fromkeys(record.node for record in self.nodes))

    @property
    def windows(self) -> tuple[str, ...]:
        """Every window label, in order, without repeats."""
        return tuple(dict.fromkeys(record.window.label for record in self.nodes))

    def to_frame(self, level: float = 0.9) -> pd.DataFrame:
        """Every bar of every node and window, one row each."""
        import pandas as pd

        frames = [record.to_frame(level) for record in self.nodes]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __repr__(self) -> str:
        engines = sorted({record.engine for record in self.nodes})
        return (
            f"Attribution({len(self.nodes)} record(s) over {len(self.node_names)} node(s), "
            f"method={self.method!r}, reduce={self.reduce!r}, engine={'+'.join(engines)})"
        )


# -- the engines --------------------------------------------------------------


def attribute(
    result: ForecastResult,
    *,
    windows: Window | Sequence[Window] | None = None,
    nodes: str | Sequence[str] | None = None,
    method: str = "sequential",
    reduce: str = "sum",
    max_replays: int = MAX_REPLAYS,
) -> Attribution:
    """Decompose a fitted forecast into baseline, seasonality, and one bar per driver.

    The engine is chosen per node: structural when the node's mechanism can
    rebuild its own mean path from what it recorded, replay when it cannot. A
    result may therefore come back with both, and each record says which it got.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")

    # Arguments before state: a mistyped window should say so whatever the
    # result happens to carry.
    slices = _resolve_windows(result, windows)
    chosen = _resolve_nodes(result, nodes)
    fits: Mapping[str, Any] = result.fits or {}
    views: Mapping[str, Any] = result.views or {}

    records: list[NodeAttribution] = []
    budget = int(max_replays)
    for node in chosen:
        fit, view = fits[node], views[node]
        keys = fit.mechanism.factors(node, view.components)
        for window in slices:
            if keys is None:
                record, spent = _replay(result, fit, view, window, method, reduce, budget)
                budget -= spent
            else:
                record = _structural(result, fit, view, window, method, reduce, tuple(keys))
            records.append(record)
    return Attribution(nodes=tuple(records), method=method, reduce=reduce)


def _resolve_nodes(result: ForecastResult, nodes: str | Sequence[str] | None) -> tuple[str, ...]:
    """Which nodes to explain, refusing the ones that have no model to explain."""
    fits: Mapping[str, Any] = result.fits or {}
    views: Mapping[str, Any] = result.views or {}
    explainable = tuple(node for node in result.nodes if node in fits and node in views)
    if nodes is None:
        if not explainable:
            raise ValueError(
                "this result carries no fitted model to explain. A forecast made by a "
                "baseline has no decomposition, and a result rebuilt without its "
                "provenance has lost the pieces; explain the result forecast() returned."
            )
        return explainable
    wanted = (nodes,) if isinstance(nodes, str) else tuple(nodes)
    for node in wanted:
        if node in explainable:
            continue
        if node not in result.nodes:
            raise KeyError(f"no series {node!r}; the result holds {list(result.nodes)}")
        raise ValueError(
            f"{node!r} has no model of its own to take apart: it is either an aggregate "
            "of its children, a node whose path was supplied through scenario=, or a "
            f"series from a model that keeps no fit. Explainable here: {list(explainable)}."
        )
    return wanted


def _resolve_windows(
    result: ForecastResult, windows: Window | Sequence[Window] | None
) -> tuple[Window, ...]:
    """The requested windows, defaulting to one that spans the whole horizon."""
    if windows is None:
        return (Window("horizon", 0, None),)
    given = (windows,) if isinstance(windows, Window) else tuple(windows)
    if not given:
        raise ValueError("windows= is empty; leave it out for the whole horizon")
    for window in given:
        if not isinstance(window, Window):
            raise TypeError(
                f"windows= takes Window objects, got {type(window).__name__}; build one "
                "with Window('Q1', 0, 3)"
            )
        window.resolve(result.horizon, result.index)
    return given


def _reduce(values: Array, how: str) -> Array:
    """Per-draw aggregation over the window's steps."""
    return np.asarray(values.sum(axis=1) if how == "sum" else values.mean(axis=1))


def _affine_terms(scaler: Scaler) -> tuple[float, float]:
    """The scaler's inverse as ``a * x + b``, refusing anything that is not a line.

    An additive decomposition needs an additive way back to the data's own units:
    every scaler the pillar picks automatically is a line (a ratio scaler is a
    multiplication, a z-score is a multiplication and a shift), so the terms come
    out of two probes. A log scale is not a line, so a waterfall on it would not
    add up, and refusing is better than handing back bars that do not.
    """
    probe = np.array([0.0, 1.0, -2.5, 0.5, 3.0, 10.0], dtype=np.float64)
    mapped = np.asarray(scaler.inverse(probe), dtype=np.float64)
    slope, intercept = float(mapped[1] - mapped[0]), float(mapped[0])
    if not np.allclose(mapped, slope * probe + intercept, rtol=1e-9, atol=1e-9):
        raise NotImplementedError(
            f"{type(scaler).__name__} does not map back to the data's units by a "
            "straight line, so an additive waterfall cannot be written in them. Fit "
            "the node with a ratio or z-score scaler if you want it explained."
        )
    return slope, intercept


def _structural(
    result: ForecastResult,
    fit: NodeFit,
    view: NodeForecast,
    window: Window,
    method: str,
    reduce: str,
    keys: tuple[str, ...],
) -> NodeAttribution:
    """Recompose the mean path with factors switched off, per draw. No sampling."""
    node = fit.node
    horizon = result.horizon
    first, last = window.resolve(horizon, result.index)
    length = last - first
    slope, intercept = _affine_terms(fit.scaler)
    offset = intercept * length if reduce == "sum" else intercept
    posterior = posterior_draws(fit.trace)
    components = view.components

    def composed(active: tuple[str, ...]) -> Array:
        """One window aggregate, in the data's own units, before the offset."""
        path = np.asarray(
            fit.mechanism.compose(node, components, posterior, active=active), dtype=np.float64
        )
        return slope * _reduce(path[:, -horizon:][:, first:last], reduce)

    mean_path = np.asarray(components[fit.mechanism.mean_site(node)], dtype=np.float64)
    total = slope * _reduce(mean_path[:, -horizon:][:, first:last], reduce) + offset

    baseline_draws = composed(()) + offset
    bars: list[FactorContribution] = []
    if method == "sequential":
        running = baseline_draws - offset
        for index, key in enumerate(keys):
            nxt = composed(keys[: index + 1])
            bars.append(FactorContribution(key, _kind(key), nxt - running))
            running = nxt
    else:
        full = composed(keys)
        for key in keys:
            without = composed(tuple(other for other in keys if other != key))
            bars.append(FactorContribution(key, _kind(key), full - without))
        bars.append(_interaction(total, baseline_draws, bars))

    return NodeAttribution(
        node=node,
        window=window,
        steps=(first, last),
        method=method,
        reduce=reduce,
        engine="structural",
        baseline=FactorContribution(fit.mechanism.baseline_label(), "baseline", baseline_draws),
        factors=tuple(bars),
        total_draws=total,
        forecast_draws=_reduce(result.draws(node)[:, first:last], reduce),
    )


def _replay(
    result: ForecastResult,
    fit: NodeFit,
    view: NodeForecast,
    window: Window,
    method: str,
    reduce: str,
    budget: int,
) -> tuple[NodeAttribution, int]:
    """Switch a factor off by zeroing its sites and forecasting again on the same seed."""
    node = fit.node
    recipe = view.replay
    if recipe is None:
        raise NotImplementedError(
            f"{node!r} cannot be explained: its mechanism does not expose the pieces it "
            "composed, and the forecast cannot be run again on its own either (a joint "
            "fit produces every node in one pass). Fit it staged, or give the mechanism "
            "factors() and compose()."
        )
    sites = _replay_sites(node, fit.trace)
    keys = tuple(sites)
    states = _states(keys, method)
    if len(states) > budget:
        raise ValueError(
            f"explaining {node!r} by replay needs {len(states)} posterior-predictive "
            f"passes and only {budget} are left of the max_replays budget. Raise "
            "max_replays, ask for fewer nodes, or use a mechanism that decomposes "
            "structurally."
        )

    first, last = window.resolve(result.horizon, result.index)
    drawn: dict[tuple[str, ...], Array] = {}
    for active in states:
        off = [site for key in keys if key not in active for site in sites[key]]
        source = fit if not off else replace(fit, trace=zero_sites(fit.trace, off))
        again = forecast_node(
            source,
            recipe.horizon,
            parents=recipe.parents,
            frozen=recipe.frozen,
            seed=recipe.seed,
        )
        drawn[active] = _reduce(again.draws[:, first:last], reduce)

    total = _reduce(view.draws[:, first:last], reduce)
    baseline_draws = drawn[()]
    bars: list[FactorContribution] = []
    if method == "sequential":
        running = baseline_draws
        for index, key in enumerate(keys):
            nxt = drawn[keys[: index + 1]]
            bars.append(FactorContribution(key, _kind(key), nxt - running))
            running = nxt
    else:
        full = drawn[keys]
        for key in keys:
            without = tuple(other for other in keys if other != key)
            bars.append(FactorContribution(key, _kind(key), full - drawn[without]))
        bars.append(_interaction(total, baseline_draws, bars))

    record = NodeAttribution(
        node=node,
        window=window,
        steps=(first, last),
        method=method,
        reduce=reduce,
        engine="replay",
        baseline=FactorContribution(fit.mechanism.baseline_label(), "baseline", baseline_draws),
        factors=tuple(bars),
        total_draws=total,
        forecast_draws=_reduce(result.draws(node)[:, first:last], reduce),
        n_replays=len(states),
    )
    return record, len(states)


def _replay_sites(node: str, trace: Any) -> dict[str, tuple[str, ...]]:
    """Which sites to zero for each factor, read off the trace by the naming contract.

    A custom mechanism cannot be asked what it composed, but it still had to
    register its sites under the contract, so the trace itself says what the
    factors are: one ``{node}_beta_{key}`` per driver, and a season under
    ``{node}_season_raw`` or ``{node}_fourier``.
    """
    posterior = posterior_draws(trace)
    prefix = f"{node}_beta_"
    seasonal = tuple(
        name for name in (f"{node}_season_raw", f"{node}_fourier") if name in posterior
    )
    sites: dict[str, tuple[str, ...]] = {SEASON: seasonal} if seasonal else {}
    for name in sorted(posterior):
        if name.startswith(prefix):
            key = name[len(prefix) :]
            if key == SEASON:
                raise ValueError(
                    f"{node!r} has a driver whose design key is {SEASON!r}, which is the "
                    "key seasonality uses in an attribution. Rename the column."
                )
            sites[key] = (name,)
    return sites


def _states(keys: tuple[str, ...], method: str) -> tuple[tuple[str, ...], ...]:
    """Which factor subsets a replay has to forecast, given the method."""
    if method == "sequential":
        return tuple(keys[:index] for index in range(len(keys) + 1))
    without = tuple(tuple(other for other in keys if other != key) for key in keys)
    return ((), *without, keys) if keys else ((),)


def _interaction(
    total: Array, baseline: Array, bars: Sequence[FactorContribution]
) -> FactorContribution:
    """What leave-one-out leaves over: the part no single factor owns.

    A season that multiplies rather than adds is the ordinary source of it. Take
    a driver out and the season no longer scales that driver's contribution
    either, so the pieces removed one at a time do not add back up to the whole,
    and the difference is a genuine interaction rather than an error. It is
    exactly zero when every piece is additive, which is what makes it worth
    showing.
    """
    residual = np.asarray(total) - np.asarray(baseline)
    for bar in bars:
        residual = residual - bar.draws
    return FactorContribution(INTERACTION, "interaction", residual)


def _kind(key: str) -> str:
    """The bar's kind: a driver unless it is one of the named pieces."""
    return _KINDS.get(key, "driver")
