# SPDX-License-Identifier: Apache-2.0
"""The lagged design matrix: time-series rows become windows over lags.

Temporal discovery reads a stationary process through windows. Row ``t`` of
the lagged frame holds every variable at lags ``0..max_lag``, and the column
for variable ``v`` at lag ``k`` is named ``v[t-k]`` (``v[t]`` at lag zero).
The result is a plain :class:`~ergodic.data.TabularData`, so the existing
conditional-independence tests run on it unchanged. Panel data builds its
windows within each entity and stacks them, which pools the entities under
the usual assumption that they all follow the same process.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import Ordinal, PanelData, TabularData, TimeSeriesData, VariableType, tabular

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["lagged_node", "lagged_tabular", "parse_lagged"]


def lagged_node(name: str, lag: int) -> str:
    """The window-column name for ``name`` at ``lag`` steps back.

    Lag zero reads ``"X[t]"``, lag ``k`` reads ``"X[t-k]"``. The name must
    carry no whitespace, since the graph layer parses edges on whitespace.
    """
    if any(ch.isspace() for ch in name):
        raise ValueError(f"variable name {name!r} contains whitespace; lagged names cannot")
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if lag == 0:
        return f"{name}[t]"
    return f"{name}[t-{lag}]"


def parse_lagged(node: str) -> tuple[str, int]:
    """Invert :func:`lagged_node`: ``"X[t-2]"`` becomes ``("X", 2)``."""
    cut = node.rfind("[t")
    if cut > 0 and node.endswith("]"):
        base, suffix = node[:cut], node[cut:]
        if suffix == "[t]":
            return base, 0
        digits = suffix[3:-1]
        canonical = digits.isdigit() and str(int(digits)) == digits and int(digits) >= 1
        if suffix.startswith("[t-") and canonical:
            return base, int(digits)
    raise ValueError(f"{node!r} is not a lagged node name; expected 'name[t]' or 'name[t-k]'")


def lagged_tabular(data: TimeSeriesData | PanelData, max_lag: int) -> TabularData:
    """Stack the windows of ``data`` into a tabular frame of lagged columns.

    Row ``t`` holds every variable at lags ``0..max_lag``; the column for
    variable ``v`` at lag ``k`` is named ``lagged_node(v, k)`` and holds ``v``
    at time ``t - k``. Column types carry over: continuous stays continuous,
    categorical keeps its categories, ordinal keeps its level order.

    A time series needs at least ``max_lag + 8`` rows. Panel windows are
    built within each entity (sorted entity order) and stacked; entities with
    fewer than ``max_lag + 2`` rows are skipped with a warning, and it is an
    error when no rows survive. Unequal time steps warn too: lags count rows,
    not calendar time.
    """
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    if isinstance(data, PanelData):
        columns, irregular, skipped = _panel_columns(data, max_lag)
        if skipped:
            warnings.warn(
                f"skipped {skipped} of {data.n_entities} entities with fewer than "
                f"max_lag + 2 = {max_lag + 2} rows",
                UserWarning,
                stacklevel=2,
            )
    elif isinstance(data, TimeSeriesData):
        columns, irregular = _series_columns(data, max_lag)
    else:
        raise TypeError(
            f"lagged_tabular needs TimeSeriesData or PanelData, got {type(data).__name__}"
        )
    if irregular:
        warnings.warn(
            "time steps are not equally spaced; lags are positional (rows back), not calendar",
            UserWarning,
            stacklevel=2,
        )
    return _typed_frame(data, columns, max_lag)


def _equally_spaced(times: npt.NDArray[Any]) -> bool:
    """True when consecutive time steps are all equal, or when we cannot tell."""
    if times.dtype.kind not in "Mmiuf" or times.size < 3:
        return True
    steps = np.diff(times)
    return bool(np.all(steps == steps[0]))


def _series_columns(data: TimeSeriesData, max_lag: int) -> tuple[dict[str, npt.NDArray[Any]], bool]:
    """The lagged columns of one series, plus an irregular-spacing flag."""
    t = data.n_rows
    if t < max_lag + 8:
        raise ValueError(
            f"need at least max_lag + 8 rows to build lagged windows; "
            f"got {t} rows with max_lag={max_lag}"
        )
    columns: dict[str, npt.NDArray[Any]] = {}
    for name in data.variables:
        values = data.column(name)
        for lag in range(max_lag + 1):
            columns[lagged_node(name, lag)] = values[max_lag - lag : t - lag]
    return columns, not _equally_spaced(data.times)


def _panel_columns(data: PanelData, max_lag: int) -> tuple[dict[str, npt.NDArray[Any]], bool, int]:
    """Stacked per-entity lagged columns, the irregular flag, and the skip count."""
    entity_column = data.column(data.entity)
    time_column = data.column(data.time)
    values = {name: data.column(name) for name in data.variables}
    parts: dict[str, list[npt.NDArray[Any]]] = {
        lagged_node(name, lag): [] for name in data.variables for lag in range(max_lag + 1)
    }
    skipped = 0
    irregular = False
    for entity in data.entities:
        rows = np.flatnonzero(entity_column == entity)
        length = int(rows.size)
        if length < max_lag + 2:
            skipped += 1
            continue
        irregular = irregular or not _equally_spaced(time_column[rows])
        for name in data.variables:
            series = values[name][rows]
            for lag in range(max_lag + 1):
                parts[lagged_node(name, lag)].append(series[max_lag - lag : length - lag])
    if skipped == data.n_entities:
        raise ValueError(
            f"no rows survive: every entity has fewer than max_lag + 2 = {max_lag + 2} rows"
        )
    columns = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    return columns, irregular, skipped


def _typed_frame(
    data: TimeSeriesData | PanelData,
    columns: dict[str, npt.NDArray[Any]],
    max_lag: int,
) -> TabularData:
    """Build the tabular frame, carrying each source column's type to its lags."""
    overrides: dict[str, VariableType | str | Ordinal] = {}
    for name in data.variables:
        column_type = data.schema[name]
        spec: VariableType | str | Ordinal = (
            Ordinal(column_type.categories or ()) if column_type.is_ordinal else column_type.kind
        )
        for lag in range(max_lag + 1):
            overrides[lagged_node(name, lag)] = spec
    frame = tabular(columns, types=overrides)
    for name in data.variables:
        column_type = data.schema[name]
        if column_type.is_categorical and column_type.categories is not None:
            for lag in range(max_lag + 1):
                frame = frame.with_type(
                    lagged_node(name, lag), VariableType.CATEGORICAL, column_type.categories
                )
    return frame
