# SPDX-License-Identifier: Apache-2.0
"""The process map: a directly-follows graph with counts and timing on every arc.

:func:`process_map` walks each case of an :class:`~ergodic.data.EventLog` and
counts how often one activity directly follows another, with the mean and
median gap in seconds on every arc. Synthetic :data:`START` and :data:`END`
arcs record where cases enter and leave. :meth:`ProcessMap.simplify` thins the
map to the busiest arcs and writes what it dropped on the result, so a
filtered view never hides that it is filtered.

A process map is deliberately not a :class:`~ergodic.graph.MixedGraph`. The
graph family holds one edge per node pair, which is the right law for causal
graphs, but in a real process ``A -> B`` and ``B -> A`` both happen (rework, a
handoff that bounces back), each direction with its own count and timing.
Rather than bend the one-edge-per-pair model, the map is its own small value
object.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._cases import case_slices, time_seconds

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..data import EventLog

__all__ = ["END", "START", "ArcStats", "ProcessMap", "process_map"]

START = "__start__"
"""Source of the synthetic arc into each case's first activity."""

END = "__end__"
"""Target of the synthetic arc out of each case's last activity."""


def _is_boundary(pair: tuple[str, str]) -> bool:
    """Whether the arc is a synthetic START or END arc rather than a real one."""
    return pair[0] == START or pair[1] == END


def _same_gap(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


@dataclass(frozen=True, eq=False)
class ArcStats:
    """Traversal count and gap timing for one arc of the map.

    ``mean_gap`` and ``median_gap`` are seconds between the two events. A
    START or END arc has no second event, so both are NaN there. Equality treats
    two NaN gaps as equal, so identically built maps compare equal on every
    Python (dataclass equality compares fields one by one from 3.13 on, where
    ``nan == nan`` is false).
    """

    count: int
    mean_gap: float
    median_gap: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArcStats):
            return NotImplemented
        return (
            self.count == other.count
            and _same_gap(self.mean_gap, other.mean_gap)
            and _same_gap(self.median_gap, other.median_gap)
        )

    def __hash__(self) -> int:
        def key(gap: float) -> float | None:
            return None if math.isnan(gap) else gap

        return hash((self.count, key(self.mean_gap), key(self.median_gap)))


@dataclass(frozen=True)
class ProcessMap:
    """A directly-follows map: activities, timed arcs, and filter provenance.

    ``activities`` lists the real activities, sorted. ``arcs`` maps each
    ``(source, target)`` pair to its :class:`ArcStats` and iterates in sorted
    key order; it holds one ``(START, a)`` arc per first activity and one
    ``(b, END)`` arc per last, with per-case counts aggregated. A freshly
    built map has ``dropped_arcs`` and ``dropped_share`` at zero; they record
    what :meth:`simplify` has removed and accumulate over repeated calls.
    """

    activities: tuple[str, ...]
    arcs: Mapping[tuple[str, str], ArcStats]
    n_cases: int
    n_events: int
    dropped_arcs: int = 0
    dropped_share: float = 0.0

    def frequency(self, a: str, b: str) -> int:
        """How many times the arc ``a -> b`` was traversed, 0 when it is absent."""
        stats = self.arcs.get((a, b))
        return 0 if stats is None else stats.count

    def simplify(self, keep: float = 0.8) -> ProcessMap:
        """A thinner map keeping the busiest arcs that cover ``keep`` of the traversals.

        ``keep`` must lie in (0, 1]. START and END arcs always stay, so every
        case still enters and leaves the map. Real arcs are ranked by count,
        heaviest first (ties broken by the sorted pair), and the smallest
        prefix whose counts reach ``keep`` times the total real traversals
        survives; the rest are dropped. Every activity stays, isolated ones
        included.

        Filtering is visible, never silent: ``dropped_arcs`` and
        ``dropped_share`` on the returned map say how many arcs are gone and
        what share of the original log's real traversals they carried, and
        both accumulate across repeated calls.
        """
        if not 0.0 < keep <= 1.0:
            raise ValueError(f"keep must be in (0, 1], got {keep}")
        real = [(pair, stats) for pair, stats in self.arcs.items() if not _is_boundary(pair)]
        total = sum(stats.count for _, stats in real)
        if total == 0:
            return self
        ranked = sorted(real, key=lambda item: (-item[1].count, item[0]))
        threshold = keep * total
        covered = 0
        n_keep = 0
        for _, stats in ranked:
            n_keep += 1
            covered += stats.count
            if covered >= threshold:
                break
        kept = {pair for pair, _ in ranked[:n_keep]}
        arcs = {
            pair: stats for pair, stats in self.arcs.items() if _is_boundary(pair) or pair in kept
        }
        # The current real traversals are (1 - dropped_share) of the original
        # log's, so the newly dropped share is measured against that original.
        original_total = total / (1.0 - self.dropped_share)
        return ProcessMap(
            activities=self.activities,
            arcs=arcs,
            n_cases=self.n_cases,
            n_events=self.n_events,
            dropped_arcs=self.dropped_arcs + (len(ranked) - n_keep),
            dropped_share=self.dropped_share + (total - covered) / original_total,
        )

    def to_dict(self) -> dict[str, Any]:
        """The map as plain data: the activities and arcs as lists, plus the counts."""
        return {
            "activities": list(self.activities),
            "arcs": [
                {
                    "source": pair[0],
                    "target": pair[1],
                    "count": stats.count,
                    "mean_gap": stats.mean_gap,
                    "median_gap": stats.median_gap,
                }
                for pair, stats in self.arcs.items()
            ],
            "n_cases": self.n_cases,
            "n_events": self.n_events,
            "dropped_arcs": self.dropped_arcs,
            "dropped_share": self.dropped_share,
        }

    def __repr__(self) -> str:
        note = ""
        if self.dropped_arcs:
            note = f", dropped {self.dropped_arcs} arcs ({self.dropped_share:.1%} of traversals)"
        return (
            f"ProcessMap(activities={len(self.activities)}, arcs={len(self.arcs)}, "
            f"cases={self.n_cases}{note})"
        )


def process_map(log: EventLog) -> ProcessMap:
    """Build the directly-follows map of ``log`` with counts and gap timing.

    Every consecutive event pair within a case traverses the arc between its
    two activities; each real arc reports its count and the mean and median
    gap in seconds (datetime and numeric timestamps alike). Each case also
    traverses ``(START, first activity)`` and ``(last activity, END)``, whose
    gaps are NaN. An activity named like a boundary sentinel raises a
    ``ValueError``, since the boundary arcs would then be ambiguous.
    """
    clash = sorted(set(log.activities) & {START, END})
    if clash:
        raise ValueError(
            f"activities {clash} collide with the reserved boundary names "
            f"{START!r} and {END!r}; rename them before mapping"
        )
    activities = log.column(log.activity)
    stamps = time_seconds(log.column(log.timestamp))
    cases, starts = case_slices(log)

    gap_lists: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    first_counts: defaultdict[str, int] = defaultdict(int)
    last_counts: defaultdict[str, int] = defaultdict(int)
    for i in range(cases.size):
        lo, hi = int(starts[i]), int(starts[i + 1])
        first_counts[str(activities[lo])] += 1
        last_counts[str(activities[hi - 1])] += 1
        for j in range(lo, hi - 1):
            pair = (str(activities[j]), str(activities[j + 1]))
            gap_lists[pair].append(float(stamps[j + 1] - stamps[j]))

    entries: dict[tuple[str, str], ArcStats] = {}
    for pair, gaps in gap_lists.items():
        arr = np.asarray(gaps, dtype=np.float64)
        entries[pair] = ArcStats(
            count=len(gaps),
            mean_gap=float(np.mean(arr)),
            median_gap=float(np.median(arr)),
        )
    for name, count in first_counts.items():
        entries[(START, name)] = ArcStats(count=count, mean_gap=math.nan, median_gap=math.nan)
    for name, count in last_counts.items():
        entries[(name, END)] = ArcStats(count=count, mean_gap=math.nan, median_gap=math.nan)

    return ProcessMap(
        activities=tuple(str(name) for name in log.activities),
        arcs=dict(sorted(entries.items())),
        n_cases=log.n_cases,
        n_events=log.n_rows,
    )
