# SPDX-License-Identifier: Apache-2.0
"""Filters over an :class:`~ergodic.data.EventLog`.

Each filter returns a new log built from the kept row positions, passed to
``take`` in ascending order so the (case, timestamp) row invariant survives.
Case-level filters keep or drop whole traces; event-level filters drop single
events, and a case whose every event drops vanishes with them. A filter that
would leave nothing raises instead of building an empty log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._cases import case_bounds, case_slices, time_seconds
from ._variants import variant_table

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from ..data import EventLog

__all__ = [
    "filter_activities",
    "filter_endpoints",
    "filter_follows",
    "filter_performance",
    "filter_time",
    "filter_variants",
]


# -- shared plumbing ----------------------------------------------------------


def _finish(log: EventLog, rows: npt.NDArray[np.bool_]) -> EventLog:
    """The kept rows as a new log; raise rather than build an empty one."""
    if not bool(rows.any()):
        raise ValueError("the filter removed every case; loosen it")
    return log.take(np.flatnonzero(rows).astype(np.int64))


def _expand(log: EventLog, keep_case: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """A per-case keep mask expanded to one entry per row."""
    _, starts = case_slices(log)
    return np.asarray(np.repeat(keep_case, np.diff(starts)), dtype=bool)


def _bound_seconds(value: Any) -> float:
    """A window bound as float seconds, through the same conversion as the stamps."""
    arr = np.asarray([value])
    if arr.dtype.kind not in "Miuf":
        arr = np.asarray([value], dtype="datetime64[ns]")
    return float(time_seconds(arr)[0])


def _check_known(log: EventLog, names: Iterable[str]) -> None:
    """Raise on activity names the log has never seen, naming them."""
    known = set(log.activities)
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown activities {unknown}; the log's activities are {sorted(known)}")


# -- filters -------------------------------------------------------------------


def filter_time(
    log: EventLog,
    *,
    start: Any | None = None,
    end: Any | None = None,
    mode: str = "intersecting",
) -> EventLog:
    """Drops cases: keep cases meeting the window on event times; mode "trim" drops events.

    ``start`` and ``end`` accept whatever compares to the log's stamps, a
    datetime-like or a number, and a ``None`` bound is open. Three modes:
    "contained" keeps cases whose every event falls inside ``[start, end]``,
    "intersecting" (the default) keeps cases with at least one event inside,
    and "trim" keeps only the events inside, so a case losing all its events
    vanishes.
    """
    if mode not in ("contained", "intersecting", "trim"):
        raise ValueError(f"unknown mode {mode!r}; use 'contained', 'intersecting', or 'trim'")
    stamps = time_seconds(log.column(log.timestamp))
    inside = np.ones(log.n_rows, dtype=bool)
    if start is not None:
        inside &= stamps >= _bound_seconds(start)
    if end is not None:
        inside &= stamps <= _bound_seconds(end)
    if mode == "trim":
        return _finish(log, inside)
    _, starts = case_slices(log)
    reduce = np.logical_and.reduceat if mode == "contained" else np.logical_or.reduceat
    keep_case: npt.NDArray[np.bool_] = reduce(inside, starts[:-1])
    return _finish(log, _expand(log, keep_case))


def filter_variants(
    log: EventLog, *, top: int | None = None, min_cases: int | None = None
) -> EventLog:
    """Drops cases: keep cases whose full activity sequence is a frequent variant.

    Give exactly one selector. ``top`` keeps the ``top`` most frequent
    variants (count descending, ties by sequence, the :func:`variant_table`
    ranking); ``min_cases`` keeps every variant with at least that many cases.
    """
    if top is not None and min_cases is not None:
        raise ValueError("give exactly one of top or min_cases, not both")
    if top is not None:
        kept = {variant.sequence for variant in variant_table(log, top=top)}
    elif min_cases is not None:
        kept = {seq for seq, count in log.variants().items() if count >= min_cases}
    else:
        raise ValueError("give exactly one of top or min_cases")
    traces = log.traces()
    cases, _ = case_slices(log)
    keep_case = np.asarray([traces[case_id] in kept for case_id in cases], dtype=bool)
    return _finish(log, _expand(log, keep_case))


def filter_endpoints(
    log: EventLog,
    *,
    start_in: Iterable[str] | None = None,
    end_in: Iterable[str] | None = None,
) -> EventLog:
    """Drops cases: keep cases that start and end in the given activity sets.

    ``start_in`` constrains a trace's first activity and ``end_in`` its last;
    a ``None`` set leaves that end unconstrained. The standard way to drop
    incomplete cases.
    """
    acts = log.column(log.activity)
    _, starts = case_slices(log)
    keep_case = np.ones(len(starts) - 1, dtype=bool)
    if start_in is not None:
        keep_case &= np.isin(acts[starts[:-1]], sorted(set(start_in)))
    if end_in is not None:
        keep_case &= np.isin(acts[starts[1:] - 1], sorted(set(end_in)))
    return _finish(log, _expand(log, keep_case))


def filter_activities(
    log: EventLog,
    *,
    keep: Iterable[str] | None = None,
    mandatory: Iterable[str] | None = None,
    forbidden: Iterable[str] | None = None,
) -> EventLog:
    """Drops cases: by mandatory and forbidden activities; ``keep`` then drops events.

    A case is dropped when it misses any ``mandatory`` activity or contains
    any ``forbidden`` one, judged on its full trace. When ``keep`` is given,
    the surviving cases then lose every event whose activity is not in it,
    and a case left with no events vanishes. At least one argument is
    required, and an activity name the log has never seen raises.
    """
    keep_set = None if keep is None else set(keep)
    need = None if mandatory is None else set(mandatory)
    ban = None if forbidden is None else set(forbidden)
    if keep_set is None and need is None and ban is None:
        raise ValueError("give at least one of keep, mandatory, or forbidden")
    _check_known(log, (keep_set or set()) | (need or set()) | (ban or set()))
    acts = log.column(log.activity)
    cases, starts = case_slices(log)
    keep_case = np.ones(len(cases), dtype=bool)
    if need is not None or ban is not None:
        for i in range(len(cases)):
            block = set(acts[starts[i] : starts[i + 1]].tolist())
            keep_case[i] = (need is None or need <= block) and (ban is None or not ban & block)
    rows = _expand(log, keep_case)
    if keep_set is not None:
        rows &= np.isin(acts, sorted(keep_set))
    return _finish(log, rows)


def filter_follows(
    log: EventLog,
    a: str,
    b: str,
    *,
    directly: bool = True,
    within: float | None = None,
) -> EventLog:
    """Drops cases: keep cases where activity ``b`` follows activity ``a``.

    With ``directly`` (the default) ``b`` must be the very next event after
    an ``a``; otherwise any later event counts. ``within`` bounds the gap in
    seconds from the ``a`` event to the ``b`` event, so a pair further apart
    does not count. Both activities must be known to the log.
    """
    _check_known(log, (a, b))
    acts = log.column(log.activity)
    stamps = time_seconds(log.column(log.timestamp))
    cases, starts = case_slices(log)
    keep_case = np.zeros(len(cases), dtype=bool)
    for i in range(len(cases)):
        lo, hi = int(starts[i]), int(starts[i + 1])
        keep_case[i] = _follows_in(acts[lo:hi], stamps[lo:hi], a, b, directly, within)
    return _finish(log, _expand(log, keep_case))


def _follows_in(
    acts: npt.NDArray[Any],
    stamps: npt.NDArray[np.float64],
    a: str,
    b: str,
    directly: bool,
    within: float | None,
) -> bool:
    """Whether some ``a`` event in this trace is followed by a qualifying ``b``."""
    for pos in np.flatnonzero(acts == a):
        i = int(pos)
        stop = min(i + 2, acts.size) if directly else int(acts.size)
        for j in range(i + 1, stop):
            if acts[j] == b and (within is None or stamps[j] - stamps[i] <= within):
                return True
    return False


def filter_performance(
    log: EventLog,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
    min_events: int | None = None,
    max_events: int | None = None,
) -> EventLog:
    """Drops cases: by case duration in seconds (first to last event) and event count.

    Bounds are inclusive and a ``None`` bound is open; give at least one.
    """
    if all(bound is None for bound in (min_duration, max_duration, min_events, max_events)):
        raise ValueError("give at least one bound")
    _, first, last = case_bounds(log)
    _, starts = case_slices(log)
    duration = last - first
    counts = np.diff(starts)
    keep_case = np.ones(duration.size, dtype=bool)
    if min_duration is not None:
        keep_case &= duration >= min_duration
    if max_duration is not None:
        keep_case &= duration <= max_duration
    if min_events is not None:
        keep_case &= counts >= min_events
    if max_events is not None:
        keep_case &= counts <= max_events
    return _finish(log, _expand(log, keep_case))
