# SPDX-License-Identifier: Apache-2.0
"""Tabular data: cross-sectional rows, each an independent unit.

This is the base shape and the one discovery and inference use first. Build it
with :func:`tabular`, which accepts a pandas or polars dataframe, a dict of
columns, or a numpy array, infers the column schema, and warns on the ambiguous
integer columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._dataset import Dataset
from ._frame import to_columns
from ._schema import infer_schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ._schema import Ordinal, VariableType


@dataclass(frozen=True, eq=False)
class TabularData(Dataset):
    """Cross-sectional data: rows are independent units, with no index column."""


def tabular(
    data: Any,
    *,
    columns: Sequence[str] | None = None,
    types: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
) -> TabularData:
    """Build a :class:`TabularData` from a dataframe, a dict of columns, or an array.

    Args:
        data: A pandas or polars dataframe, a mapping of name to column, or a
            2-D numpy array (name the columns with ``columns``).
        columns: Column names, used when ``data`` is a bare numpy array.
        types: Per-column type overrides, for example
            ``{"age": "continuous", "stage": Ordinal(["I", "II", "III"])}``.
        exclude: Columns to mark excluded, so they are kept but not treated as
            variables (identifiers, free text).
        categorical_threshold: Integer columns with at most this many distinct
            values are inferred categorical, with a warning.
    """
    store, n_rows, declared = to_columns(data, columns)
    schema = infer_schema(
        store,
        n_rows=n_rows,
        declared=declared,
        overrides=types,
        exclude=exclude,
        categorical_threshold=categorical_threshold,
    )
    return TabularData(schema=schema, n_rows=n_rows, _store=store)
