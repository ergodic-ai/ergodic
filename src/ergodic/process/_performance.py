# SPDX-License-Identifier: Apache-2.0
"""Cycle-time and waiting summaries over an event log.

:func:`performance` reads case durations (first event to last) and a
per-activity timing table. With one timestamp per activity you cannot split
waiting from working, so the complete-only summary reports the gap from the
previous event in the case as the wait before each activity and leaves
service time as ``None``. A lifecycle column with ``"start"`` and
``"complete"`` rows makes the split real: service is complete minus start,
wait is start minus the previous complete in the same case, and rows that
never find a partner are counted in ``n_unmatched``, not hidden.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._cases import case_bounds, case_slices, time_seconds

if TYPE_CHECKING:
    from ..data import EventLog

__all__ = ["ActivityStats", "ProcessPerformance", "performance"]


@dataclass(frozen=True)
class ActivityStats:
    """Per-activity timing: event count, mean wait before it, mean service inside it.

    ``count`` is how many events carry the activity. ``mean_wait`` is None
    when no event of the activity has anything earlier in its case to wait
    on; ``mean_service`` is None on complete-only logs, where one timestamp
    per activity cannot split waiting from working, and for activities with
    no matched start/complete pair.
    """

    activity: str
    count: int
    mean_wait: float | None
    mean_service: float | None


@dataclass(frozen=True)
class ProcessPerformance:
    """Case cycle times plus the per-activity timing table.

    Durations are each case's last event minus its first, in seconds.
    ``activities`` is sorted by activity name. ``lifecycle`` says whether a
    waiting/service split was possible, and ``n_unmatched`` counts lifecycle
    rows that found no partner.
    """

    n_cases: int
    n_events: int
    mean_duration: float
    median_duration: float
    p90_duration: float
    activities: tuple[ActivityStats, ...]
    lifecycle: bool
    n_unmatched: int = 0


def _mean(sums: dict[str, float], counts: dict[str, int], name: str) -> float | None:
    """The mean for ``name`` from running sums, None when nothing was observed."""
    n = counts.get(name, 0)
    return sums[name] / n if n else None


def performance(log: EventLog, *, lifecycle: str | None = None) -> ProcessPerformance:
    """Summarize case cycle times and per-activity waiting and service.

    Case durations run first event to last, in seconds, summarized as the
    mean, median, and 90th percentile. Per activity, ``count`` is how many
    events carry it.

    With ``lifecycle=None`` the log has one timestamp per activity, which
    cannot split waiting from working: ``mean_wait`` is the mean gap from the
    previous event in the same case (a case's first event contributes no
    wait), ``mean_service`` is None, and the result's ``lifecycle`` is False.

    With ``lifecycle="column"`` the named column must exist (``KeyError``
    otherwise). Rows valued ``"start"`` and ``"complete"`` are paired per
    case and activity in time order, each start with the next complete of the
    same case and activity: service is complete minus start, wait is start
    minus the previous complete event in the case (a start with no earlier
    complete contributes no wait), and rows that never find a partner are
    counted in ``n_unmatched``. Other lifecycle values are ignored. Durations
    still run first event to last over all events.
    """
    _, first, last = case_bounds(log)
    durations = last - first
    if durations.size:
        mean_duration = float(np.mean(durations))
        median_duration = float(np.median(durations))
        p90_duration = float(np.percentile(durations, 90))
    else:
        mean_duration = median_duration = p90_duration = math.nan

    activities = log.column(log.activity)
    stamps = time_seconds(log.column(log.timestamp))
    cases, starts = case_slices(log)
    counts = Counter(str(name) for name in activities.tolist())

    wait_sums: defaultdict[str, float] = defaultdict(float)
    wait_counts: defaultdict[str, int] = defaultdict(int)
    service_sums: defaultdict[str, float] = defaultdict(float)
    service_counts: defaultdict[str, int] = defaultdict(int)
    n_unmatched = 0

    if lifecycle is None:
        for i in range(cases.size):
            lo, hi = int(starts[i]), int(starts[i + 1])
            for j in range(lo + 1, hi):
                name = str(activities[j])
                wait_sums[name] += float(stamps[j] - stamps[j - 1])
                wait_counts[name] += 1
    else:
        if lifecycle not in log.schema:
            raise KeyError(
                f"lifecycle column {lifecycle!r} not found; have {sorted(log.schema.names)}"
            )
        marks = log.column(lifecycle)
        for i in range(cases.size):
            lo, hi = int(starts[i]), int(starts[i + 1])
            open_starts: defaultdict[str, deque[float]] = defaultdict(deque)
            last_complete: float | None = None
            for j in range(lo, hi):
                mark = str(marks[j])
                name = str(activities[j])
                when = float(stamps[j])
                if mark == "start":
                    if last_complete is not None:
                        wait_sums[name] += when - last_complete
                        wait_counts[name] += 1
                    open_starts[name].append(when)
                elif mark == "complete":
                    if open_starts[name]:
                        service_sums[name] += when - open_starts[name].popleft()
                        service_counts[name] += 1
                    else:
                        n_unmatched += 1
                    last_complete = when
                # other lifecycle values are ignored
            n_unmatched += sum(len(queue) for queue in open_starts.values())

    stats = tuple(
        ActivityStats(
            activity=str(name),
            count=counts[str(name)],
            mean_wait=_mean(wait_sums, wait_counts, str(name)),
            mean_service=_mean(service_sums, service_counts, str(name)),
        )
        for name in log.activities
    )
    return ProcessPerformance(
        n_cases=log.n_cases,
        n_events=log.n_rows,
        mean_duration=mean_duration,
        median_duration=median_duration,
        p90_duration=p90_duration,
        activities=stats,
        lifecycle=lifecycle is not None,
        n_unmatched=n_unmatched,
    )
