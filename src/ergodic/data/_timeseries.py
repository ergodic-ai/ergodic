# SPDX-License-Identifier: Apache-2.0
"""Time-series data: one or more series ordered by a time index.

Build it with :func:`timeseries`, naming the time column. Rows come back sorted
by that column, since the temporal methods that follow read them in order. The
time column is an index, so it is not one of the modeled variables.

The temporal algorithms (PCMCI, Granger, interrupted time series, forecasting)
live in their pillars. This is the container they read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._dataset import Dataset
from ._frame import to_columns
from ._schema import infer_schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

    from ._schema import Ordinal, VariableType


@dataclass(frozen=True, eq=False)
class TimeSeriesData(Dataset):
    """Series ordered by a time column, with an optional sampling frequency."""

    time: str
    freq: str | None

    @property
    def index_columns(self) -> tuple[str, ...]:
        return (self.time,)

    @property
    def times(self) -> npt.NDArray[Any]:
        """The time index, in ascending order."""
        return self.column(self.time)


def timeseries(
    data: Any,
    *,
    time: str,
    freq: str | None = None,
    columns: Sequence[str] | None = None,
    types: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
) -> TimeSeriesData:
    """Build a :class:`TimeSeriesData`, sorting rows by the ``time`` column.

    The time column is typed from its data (datetime, else continuous) unless
    you override it, so an integer step index is not mistaken for a category.
    """
    store, n_rows, declared = to_columns(data, columns)
    if time not in store:
        raise KeyError(f"time column {time!r} not found in {sorted(store)}")
    overrides: dict[str, VariableType | str | Ordinal] = dict(types or {})
    if time not in overrides:
        overrides[time] = "datetime" if store[time].dtype.kind == "M" else "continuous"
    schema = infer_schema(
        store,
        n_rows=n_rows,
        declared=declared,
        overrides=overrides,
        exclude=exclude,
        categorical_threshold=categorical_threshold,
    )
    order = np.argsort(store[time], kind="stable")
    ordered = {name: arr[order] for name, arr in store.items()}
    return TimeSeriesData(schema=schema, n_rows=n_rows, _store=ordered, time=time, freq=freq)
