# SPDX-License-Identifier: Apache-2.0
"""Event log: variable-length traces of discrete events, keyed by a case.

Build it with :func:`event_log`, naming the case, activity, and timestamp
columns. Rows are events, grouped by case and ordered in time. The log reads as
the process-mining primitives the discovery algorithms need: the trace per case,
the distinct variants, and the directly-follows counts.

The projections to the grid shapes (case features to tabular, event rates to a
time-series) land with the process pillar. This is the container and its core
relations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from ._dataset import Dataset
from ._frame import to_columns
from ._schema import infer_schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ._schema import Ordinal, VariableType


@dataclass(frozen=True, eq=False)
class EventLog(Dataset):
    """Events keyed by case, activity, and timestamp; one row per event."""

    case: str
    activity: str
    timestamp: str

    @property
    def index_columns(self) -> tuple[str, ...]:
        return (self.case, self.activity, self.timestamp)

    @property
    def cases(self) -> tuple[Any, ...]:
        """The distinct case identifiers, sorted."""
        return tuple(sorted(set(self._store[self.case].tolist())))

    @property
    def n_cases(self) -> int:
        """The number of distinct cases."""
        return len(set(self._store[self.case].tolist()))

    @property
    def activities(self) -> tuple[str, ...]:
        """The distinct activity names, sorted."""
        return tuple(sorted(set(self._store[self.activity].tolist())))

    @property
    def n_activities(self) -> int:
        """The number of distinct activities."""
        return len(set(self._store[self.activity].tolist()))

    def traces(self) -> dict[Any, tuple[str, ...]]:
        """The ordered activity sequence per case."""
        cases = self._store[self.case].tolist()
        acts = self._store[self.activity].tolist()
        out: dict[Any, list[str]] = {}
        for case_id, act in zip(cases, acts, strict=True):
            out.setdefault(case_id, []).append(act)
        return {case_id: tuple(seq) for case_id, seq in out.items()}

    def variants(self) -> Counter[tuple[str, ...]]:
        """How many cases follow each distinct activity sequence."""
        return Counter(self.traces().values())

    def directly_follows(self) -> Counter[tuple[str, str]]:
        """Counts of each ``(a, b)`` where ``b`` immediately follows ``a`` in a case."""
        out: Counter[tuple[str, str]] = Counter()
        for trace in self.traces().values():
            for first, second in pairwise(trace):
                out[(first, second)] += 1
        return out

    def __repr__(self) -> str:
        return (
            f"EventLog(cases={self.n_cases}, events={self.n_rows}, activities={self.n_activities})"
        )


def event_log(
    data: Any,
    *,
    case: str,
    activity: str,
    timestamp: str,
    columns: Sequence[str] | None = None,
    types: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
) -> EventLog:
    """Build an :class:`EventLog`, grouped by case and ordered by timestamp."""
    store, n_rows, declared = to_columns(data, columns)
    for role, name in (("case", case), ("activity", activity), ("timestamp", timestamp)):
        if name not in store:
            raise KeyError(f"{role} column {name!r} not found in {sorted(store)}")
    overrides: dict[str, VariableType | str | Ordinal] = dict(types or {})
    overrides.setdefault(case, "categorical")
    overrides.setdefault(activity, "categorical")
    if timestamp not in overrides:
        overrides[timestamp] = "datetime" if store[timestamp].dtype.kind == "M" else "continuous"
    schema = infer_schema(
        store,
        n_rows=n_rows,
        declared=declared,
        overrides=overrides,
        exclude=exclude,
        categorical_threshold=categorical_threshold,
    )
    order = np.lexsort((store[timestamp], store[case]))
    ordered = {name: arr[order] for name, arr in store.items()}
    return EventLog(
        schema=schema,
        n_rows=n_rows,
        _store=ordered,
        case=case,
        activity=activity,
        timestamp=timestamp,
    )
