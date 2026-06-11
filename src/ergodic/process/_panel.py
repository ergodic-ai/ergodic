# SPDX-License-Identifier: Apache-2.0
"""The log-to-panel bridge: aggregate an event log into KPI panels.

The quasi-experimental estimators (``did``, ``event_study``,
``synthetic_control``, ``interrupted_time_series``) read a panel or a single
series, while process data arrives as an event log. :func:`kpi_panel` closes
that gap: bucket the cases by when they start, split them by a case-constant
unit column when one is given, and aggregate per-cell KPIs (cycle time, case
counts, events per case, or the mean of any case-constant numeric column).
The result is a :class:`~ergodic.data.PanelData`, or with no unit a
:class:`~ergodic.data.TimeSeriesData`, so a routing change logged per case
goes straight into a difference-in-differences design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import panel, timeseries
from ._cases import case_attribute, case_bounds, case_slices

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from ..data import EventLog, PanelData, TimeSeriesData

__all__ = ["kpi_panel"]

_DERIVED = ("cycle_time", "n_cases", "n_events")


def _bucket_labels(
    first: npt.NDArray[np.float64],
    period: str | float,
    timestamp: str,
    stamps_are_datetime: bool,
) -> npt.NDArray[Any]:
    """Each case's bucket label from its first event time in epoch seconds."""
    if isinstance(period, str):
        if not stamps_are_datetime:
            raise TypeError(
                f"period {period!r} is a calendar unit, but the timestamp column "
                f"{timestamp!r} is numeric; pass a positive numeric width instead"
            )
        days = np.floor(first / 86400.0).astype(np.int64)
        if period == "day":
            return days.astype("datetime64[D]")
        if period == "week":
            monday = days - ((days + 3) % 7)  # day 0, the epoch, was a Thursday
            return monday.astype("datetime64[D]")
        if period == "month":
            return days.astype("datetime64[D]").astype("datetime64[M]")
        raise ValueError(
            f"period must be 'day', 'week', or 'month' on datetime stamps, got {period!r}"
        )
    if stamps_are_datetime:
        raise TypeError(
            f"period {period!r} is a numeric width, but the timestamp column "
            f"{timestamp!r} is datetime; pass 'day', 'week', or 'month' instead"
        )
    width = float(period)
    if not width > 0.0:
        raise ValueError(f"period must be a positive width on numeric stamps, got {period!r}")
    return np.asarray(np.floor(first / width) * width, dtype=np.float64)


def kpi_panel(
    log: EventLog,
    period: str | float,
    *,
    unit: str | None = None,
    kpis: Iterable[str] = ("cycle_time",),
) -> PanelData | TimeSeriesData:
    """Aggregate an event log into per-period KPI rows for the panel estimators.

    Each case belongs to the bucket of its first event and, when ``unit`` is
    given, to its unit, read from a case-constant column of the log (a column
    that varies within a case raises). Every aggregate in a row is over the
    cases that start in that unit-period cell. A cell where no case starts is
    simply absent; the panel designs handle the unbalanced grid.

    ``period`` picks the bucketing. With datetime stamps pass ``"day"``,
    ``"week"``, or ``"month"``; the bucket labels are ``np.datetime64``
    values: the stamp's calendar day (``datetime64[D]``), the Monday of the
    stamp's week (``datetime64[D]``), or the stamp's calendar month
    (``datetime64[M]``). With numeric stamps pass a positive width ``w``; a
    case starting at ``t`` lands in the bucket labeled ``floor(t / w) * w``,
    in the stamps' own units. A calendar name on numeric stamps raises
    ``TypeError``, as does a numeric width on datetime stamps, and a width at
    or below zero raises ``ValueError``.

    Each name in ``kpis`` becomes one float column, named after itself, in
    the order given. Three names are derived from the log: ``"cycle_time"``
    is the mean case duration among the cell's cases, in seconds for datetime
    stamps and in the stamps' own units otherwise; ``"n_cases"`` counts the
    cases that start in the cell; ``"n_events"`` is the mean number of events
    per case. Any other name reads a case-constant numeric or boolean column
    of the log and aggregates it by its mean in the cell, so a 0/1 case flag
    that switches on a period boundary comes out as an exact 0/1 column,
    ready to be a ``did`` treatment; a non-numeric column raises
    ``ValueError``. The three derived names are reserved: when the log also
    carries a column by a requested derived name, the request is ambiguous
    and raises ``ValueError``.

    The output columns are the unit column (named after the source column),
    a ``"period"`` column holding the bucket labels, then the kpi columns. A
    kpi named like the unit column or ``"period"`` collides with those index
    columns and raises, as does a unit column itself named ``"period"``.
    With ``unit`` given the result is a :class:`~ergodic.data.PanelData`
    keyed by ``(unit, "period")``, the shape ``did``, ``event_study``, and
    ``synthetic_control`` consume. With ``unit=None`` the whole log is one
    series and the result is a :class:`~ergodic.data.TimeSeriesData` with
    ``time="period"``, so ``interrupted_time_series`` consumes it directly.

    Two caveats. A case is attributed to the bucket where it starts, so a
    case that runs long spills its effect into earlier buckets: its whole
    duration counts where it began, even when it finished much later. And
    the aggregation assumes the log is complete, with no still-open cases; a
    censored log drops exactly the long cases near its end, which biases
    late-bucket cycle times downward.
    """
    names = [str(name) for name in kpis]
    if len(set(names)) != len(names):
        raise ValueError(f"kpis repeat a name: {names}")
    if unit == "period":
        raise ValueError(
            "unit column 'period' collides with the 'period' column the output adds; "
            "rename the log column"
        )
    for name in names:
        if name == "period":
            raise ValueError("kpi 'period' collides with the output period column; rename it")
        if unit is not None and name == unit:
            raise ValueError(f"kpi {name!r} collides with the output unit column {unit!r}")
        if name in _DERIVED and name in log.schema:
            raise ValueError(
                f"kpi {name!r} is derived from the log, but the log also carries a column "
                f"named {name!r}; the request is ambiguous, so rename the column"
            )

    _, first, last = case_bounds(log)
    labels = _bucket_labels(first, period, log.timestamp, log.schema[log.timestamp].is_datetime)

    label_keys, label_inv = np.unique(labels, return_inverse=True)
    store: dict[str, npt.NDArray[Any]] = {}
    if unit is None:
        cell_of_case = np.asarray(label_inv, dtype=np.int64)
        n_cells = int(label_keys.size)
        store["period"] = label_keys
    else:
        unit_values = case_attribute(log, unit)
        unit_keys, unit_inv = np.unique(unit_values, return_inverse=True)
        stride = max(int(label_keys.size), 1)
        pair = np.asarray(unit_inv, dtype=np.int64) * stride + np.asarray(label_inv, dtype=np.int64)
        cell_ids, inverse = np.unique(pair, return_inverse=True)
        cell_of_case = np.asarray(inverse, dtype=np.int64)
        n_cells = int(cell_ids.size)
        store[unit] = unit_keys[cell_ids // stride]
        store["period"] = label_keys[cell_ids % stride]

    counts = np.bincount(cell_of_case, minlength=n_cells).astype(np.float64)

    def cell_mean(per_case: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        sums = np.bincount(cell_of_case, weights=per_case, minlength=n_cells)
        return np.asarray(sums / counts, dtype=np.float64)

    for name in names:
        if name == "cycle_time":
            store[name] = cell_mean(np.asarray(last - first, dtype=np.float64))
        elif name == "n_cases":
            store[name] = counts.copy()
        elif name == "n_events":
            _, starts = case_slices(log)
            store[name] = cell_mean(np.diff(starts).astype(np.float64))
        else:
            values = case_attribute(log, name)
            if values.dtype.kind not in "biuf":
                raise ValueError(
                    f"kpi column {name!r} has dtype {values.dtype}; a kpi aggregates by its "
                    "cell mean, so the column must be numeric or boolean"
                )
            store[name] = cell_mean(np.asarray(values, dtype=np.float64))

    if unit is None:
        return timeseries(store, time="period")
    return panel(store, entity=unit, time="period")
