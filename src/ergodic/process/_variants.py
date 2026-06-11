# SPDX-License-Identifier: Apache-2.0
"""The variant table of an :class:`~ergodic.data.EventLog`.

A variant is a distinct activity sequence, and the table ranks the variants by
how many cases follow each one, with the share and duration summary a process
analyst reads first. The variant filter reuses this ranking, so "the top three
variants" means the same thing everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._cases import case_bounds

if TYPE_CHECKING:
    from ..data import EventLog

__all__ = ["Variant", "variant_table"]


@dataclass(frozen=True)
class Variant:
    """One distinct activity sequence and its case statistics."""

    sequence: tuple[str, ...]
    n_cases: int
    share: float
    mean_duration: float
    median_duration: float


def variant_table(log: EventLog, *, top: int | None = None) -> tuple[Variant, ...]:
    """The distinct variants, most frequent first.

    Ranked by case count descending, ties by sequence. ``share`` is the
    fraction of all cases on the variant, and the durations are first-to-last
    event times in seconds over the variant's cases. ``top`` truncates the
    table; ``None`` keeps every variant.
    """
    if top is not None and top < 1:
        raise ValueError(f"top must be at least 1, got {top}")
    traces = log.traces()
    cases, first, last = case_bounds(log)
    durations = last - first
    by_sequence: dict[tuple[str, ...], list[float]] = {}
    for i, case_id in enumerate(cases):
        by_sequence.setdefault(traces[case_id], []).append(float(durations[i]))
    n_total = len(cases)
    order = sorted(by_sequence, key=lambda seq: (-len(by_sequence[seq]), seq))
    table = tuple(
        Variant(
            sequence=seq,
            n_cases=len(by_sequence[seq]),
            share=len(by_sequence[seq]) / n_total,
            mean_duration=float(np.mean(by_sequence[seq])),
            median_duration=float(np.median(by_sequence[seq])),
        )
        for seq in order
    )
    return table if top is None else table[:top]
