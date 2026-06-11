# SPDX-License-Identifier: Apache-2.0
"""Shared case arithmetic over an :class:`~ergodic.data.EventLog`.

An event log's rows are grouped by case and ordered by timestamp within each
case, so per-case work is slice arithmetic. These helpers own the slicing and
the timestamp conversion; the process modules build on them instead of
re-deriving either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..data import EventLog

__all__ = [
    "case_attribute",
    "case_bounds",
    "case_slices",
    "open_cases",
    "time_seconds",
]


def time_seconds(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Timestamps as float seconds: datetimes since the epoch, numbers as they are."""
    if values.dtype.kind == "M":
        return values.astype("datetime64[ns]").astype(np.int64).astype(np.float64) / 1e9
    return np.asarray(values, dtype=np.float64)


def case_slices(log: EventLog) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Each case's row block, in row order.

    Returns ``(cases, starts)`` where ``cases[i]`` owns rows
    ``starts[i]:starts[i + 1]`` (with ``log.n_rows`` as the final sentinel).
    Cases appear in the log's row order, which is their sorted order.
    """
    case_col = log.column(log.case)
    if case_col.size == 0:
        return case_col, np.array([0], dtype=np.int64)
    changed = np.r_[True, case_col[1:] != case_col[:-1]]
    starts = np.flatnonzero(changed).astype(np.int64)
    return case_col[starts], np.r_[starts, np.int64(log.n_rows)]


def case_bounds(
    log: EventLog,
) -> tuple[npt.NDArray[Any], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per case, in row order: the case ids, first and last event times in seconds."""
    cases, starts = case_slices(log)
    stamps = time_seconds(log.column(log.timestamp))
    first = stamps[starts[:-1]]
    last = stamps[starts[1:] - 1]
    return cases, first, last


def case_attribute(log: EventLog, name: str) -> npt.NDArray[Any]:
    """One value per case for a case-constant column, in case row order.

    Raises ``ValueError`` naming the column and an offending case when the
    column varies within a case (it is then an event attribute, not a case
    attribute).
    """
    if name not in log.schema:
        raise KeyError(f"no column named {name!r}")
    values = log.column(name)
    cases, starts = case_slices(log)
    for i, case in enumerate(cases):
        block = values[starts[i] : starts[i + 1]]
        head = block[0]
        same = (block == head) | ((block != block) & (head != head))  # NaN equals NaN here
        if not bool(np.all(same)):
            raise ValueError(
                f"column {name!r} varies within case {case!r}; "
                "case_table covariates must be constant per case"
            )
    return values[starts[:-1]]


def open_cases(
    first: npt.NDArray[np.float64],
    last: npt.NDArray[np.float64],
    at: npt.NDArray[np.float64],
    *,
    exclude_self: bool = True,
) -> npt.NDArray[np.float64]:
    """How many cases are open at each time in ``at``.

    Open at time t means started at or before t and not yet finished; a case
    whose last event is exactly t counts as closed. With ``exclude_self``
    (the default) ``at[i]`` is read as case i's own reference time and the
    case does not count itself.
    """
    order_first = np.sort(first)
    order_last = np.sort(last)
    started = np.searchsorted(order_first, at, side="right")
    finished = np.searchsorted(order_last, at, side="right")
    count = (started - finished).astype(np.float64)
    if exclude_self:
        own_open = (first <= at) & (last > at)
        count = count - own_open.astype(np.float64)
    return count
