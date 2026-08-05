# SPDX-License-Identifier: Apache-2.0
"""The facade: data and a horizon in, a forecast out.

:func:`forecast` is the one call most users need. It reads the schema to pick a
mechanism per column, reads the time index to find a seasonal period, fits each
series, and continues it forward, handing back a :class:`ForecastResult` with the
draws and the provenance to explain where they came from.

The signature is the finished one. Everything it will eventually accept is
already a keyword, and the parts that are not built yet refuse by name instead of
being absent, so no working call ever has to change shape later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import HierarchicalTSData, TimeSeriesData
from ._fit import fit_node, forecast_node, sorted_seeds
from ._mechanism import assign_mechanisms
from ._result import ForecastResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from ..graph import MixedGraph
    from ._mechanism import TemporalMechanism

    Array = npt.NDArray[np.float64]

__all__ = ["forecast"]

#: Fitting strategies that reach the independent per-series path.
_INDEPENDENT_FITS = ("auto", "independent")
#: Strategies that exist in the vocabulary but are not built yet.
_STAGED_FITS = ("staged", "joint")


def forecast(
    data: TimeSeriesData,
    horizon: int,
    *,
    graph: MixedGraph | None = None,
    mechanisms: Mapping[str, TemporalMechanism] | None = None,
    reconcile: str | None = None,
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

    Parameters
    ----------
    data:
        A :class:`~ergodic.data.TimeSeriesData`. A hierarchical dataset needs the
        reconciliation layer and refuses here for now.
    horizon:
        How many steps past the end of the data to forecast.
    graph:
        A window graph naming which series drive which. Not read yet.
    mechanisms:
        Per-column overrides. A column named here keeps the mechanism it is
        given; everything else is assigned from the schema.
    reconcile:
        The reconciliation strategy, which only means something for a hierarchy.
        Leave it unset for flat data. The default once a hierarchy exists is
        bottom-up.
    scenario:
        Pinned future paths for known-future nodes. Not read yet.
    fit:
        ``"auto"`` or ``"independent"``: one fit per series, no information
        crossing between them. ``"staged"`` and ``"joint"`` are named but not
        built.
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
        data arrived in.
    """
    if isinstance(data, HierarchicalTSData):
        raise NotImplementedError(
            "forecast() does not read a HierarchicalTSData yet: forecasting every level "
            "and making the levels agree is the reconciliation sprint (S3). Pass a flat "
            "TimeSeriesData, or forecast the levels separately for now."
        )
    if not isinstance(data, TimeSeriesData):
        raise TypeError(
            f"forecast() needs a TimeSeriesData, got {type(data).__name__}; build one with "
            "ergodic.timeseries(frame, time=...)"
        )
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if graph is not None:
        raise NotImplementedError(
            "graph= is the driver sprint (S4): reading a window graph means resolving each "
            "driver's future and handing posteriors down the stages. Forecast the series "
            "independently for now by leaving graph unset."
        )
    if scenario is not None:
        raise NotImplementedError(
            "scenario= is the driver sprint (S4): a pinned future path only means something "
            "once a node can read another node's path. Leave it unset for now."
        )
    if reconcile is not None:
        raise ValueError(
            f"reconcile={reconcile!r} needs a hierarchy, and this dataset is flat: there is "
            "nothing to make coherent. Build a HierarchicalTSData with ergodic.hts(...) to "
            "reconcile across levels."
        )
    if fit in _STAGED_FITS:
        raise NotImplementedError(
            f"fit={fit!r} is the driver sprint (S4): staged and joint fitting are about "
            "several nodes sharing structure, and this sprint fits each series alone. "
            "Use fit='auto'."
        )
    if fit not in _INDEPENDENT_FITS:
        raise ValueError(f"fit must be one of {(*_INDEPENDENT_FITS, *_STAGED_FITS)}, got {fit!r}")

    nodes = data.variables
    if not nodes:
        raise ValueError("the dataset models no columns, so there is nothing to forecast")
    assigned = assign_mechanisms(data, mechanisms, robust=robust)
    seeds = sorted_seeds(nodes, seed)

    series: dict[str, Array] = {}
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
        continued = forecast_node(fitted, int(horizon), seed=seeds[node] + 1000)
        series[node] = continued.draws
        diagnostics[node] = fitted.diagnostics

    return ForecastResult(
        series=series,
        horizon=int(horizon),
        kind="draws",
        start=_next_label(data),
        mechanisms=assigned,
        diagnostics=diagnostics,
    )


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
