# SPDX-License-Identifier: Apache-2.0
"""The dataframe boundary: turn user input into a plain column store.

Input can be a pandas dataframe, a polars or arrow object (handled through its
``to_pandas``), a mapping of name to column, or a 2-D numpy array. Output is a
dict of name to 1-D numpy array plus the row count, the one internal form the
dataset and its algorithms compute on. Pandas categorical columns carry their
category order out, so an ordered categorical lands as an ordinal column.

This is the only place a dataframe library is touched. Swapping in narwhals for
true polars and arrow support later is a change to this file alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._schema import ColumnType, VariableType

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

Columns = tuple[dict[str, "npt.NDArray[Any]"], int, dict[str, ColumnType]]


def to_columns(data: Any, columns: Sequence[str] | None = None) -> Columns:
    """Normalize any supported input into ``(store, n_rows, declared)``."""
    if isinstance(data, Mapping):
        return _from_mapping(data)
    if isinstance(data, np.ndarray):
        return _from_numpy(data, columns)
    if type(data).__module__.split(".")[0] == "pandas":
        return _from_pandas(data)
    to_pandas = getattr(data, "to_pandas", None)
    if callable(to_pandas):
        return _from_pandas(to_pandas())
    raise TypeError(
        f"cannot build a dataset from {type(data)!r}; pass a pandas or polars dataframe, "
        "a dict of columns, or a numpy array"
    )


def _from_mapping(data: Mapping[Any, Any]) -> Columns:
    store: dict[str, npt.NDArray[Any]] = {}
    n_rows: int | None = None
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise ValueError(f"column {key!r} is not one-dimensional")
        if n_rows is None:
            n_rows = arr.shape[0]
        elif arr.shape[0] != n_rows:
            raise ValueError("columns have unequal length")
        store[str(key)] = arr
    return store, int(n_rows or 0), {}


def _from_numpy(data: npt.NDArray[Any], columns: Sequence[str] | None) -> Columns:
    if data.dtype.names:
        fields = list(data.dtype.names)
        store = {str(f): np.asarray(data[f]) for f in fields}
        return store, int(data.shape[0]), {}
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim != 2:
        raise ValueError("a numpy array must be 1-D or 2-D")
    n_cols = data.shape[1]
    if columns is None:
        names = [f"x{i}" for i in range(n_cols)]
    else:
        names = [str(c) for c in columns]
        if len(names) != n_cols:
            raise ValueError(f"got {len(names)} column names for {n_cols} columns")
    store = {names[i]: data[:, i] for i in range(n_cols)}
    return store, int(data.shape[0]), {}


def _from_pandas(df: Any) -> Columns:
    import pandas as pd

    store: dict[str, npt.NDArray[Any]] = {}
    declared: dict[str, ColumnType] = {}
    for original in list(df.columns):
        name = str(original)
        series = df[original]
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            kind = VariableType.ORDINAL if dtype.ordered else VariableType.CATEGORICAL
            declared[name] = ColumnType(name, kind, tuple(dtype.categories.tolist()))
            store[name] = np.asarray(series.astype(object).to_numpy())
        else:
            store[name] = np.asarray(series.to_numpy())
    return store, len(df), declared
