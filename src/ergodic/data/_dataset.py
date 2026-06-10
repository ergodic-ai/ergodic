# SPDX-License-Identifier: Apache-2.0
"""The dataset base: a typed, immutable column store the algorithms read.

``Dataset`` holds the normalized columns, the :class:`Schema`, and the accessors
that discovery and inference need: a float matrix, integer codes for categorical
columns, missing-value reporting, and a name check against a graph. It is
immutable, so retyping a column, selecting columns, or dropping rows returns a
new dataset. The concrete shapes (tabular, time-series, panel, and the rest)
extend it with their index columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ._schema import (
    Ordinal,
    Schema,
    VariableType,
    _encode,
    _missing_mask,
    _resolve_type,
    _unique_nonnull,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from typing import Self

    import numpy.typing as npt
    import pandas as pd

    from ..graph import MixedGraph


@dataclass(frozen=True, eq=False)
class Dataset:
    """A typed column store. Built through the shape factories, not directly."""

    schema: Schema
    n_rows: int
    _store: Mapping[str, npt.NDArray[Any]] = field(repr=False)

    # -- shape ---------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return self.schema.names

    @property
    def n_columns(self) -> int:
        return len(self.schema)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_columns)

    def __len__(self) -> int:
        return self.n_rows

    @property
    def index_columns(self) -> tuple[str, ...]:
        """Columns that index the rows (time, entity, case), not modeled variables."""
        return ()

    @property
    def variables(self) -> tuple[str, ...]:
        """Column names the algorithms see: not the excluded ones, not the index."""
        index = set(self.index_columns)
        return tuple(name for name in self.schema.variables if name not in index)

    # -- column access -------------------------------------------------------

    def column(self, name: str) -> npt.NDArray[Any]:
        """A read-only copy of one column's raw values."""
        if name not in self.schema:
            raise KeyError(name)
        arr = np.array(self._store[name], copy=True)
        arr.setflags(write=False)
        return arr

    def codes(self, name: str) -> npt.NDArray[np.int64]:
        """Integer category codes for a categorical or ordinal column, -1 for missing."""
        col = self.schema[name]
        if not col.is_discrete:
            raise ValueError(f"column {name!r} is {col.kind.value}, not categorical or ordinal")
        return _encode(self._store[name], col.categories, col.ordered)

    def categories(self, name: str) -> tuple[Any, ...] | None:
        return self.schema[name].categories

    def matrix(self, names: Sequence[str] | None = None) -> npt.NDArray[np.float64]:
        """A float matrix of the named columns. Discrete columns become their codes."""
        chosen = list(names) if names is not None else list(self.variables)
        if not chosen:
            return np.empty((self.n_rows, 0), dtype=np.float64)
        cols: list[npt.NDArray[Any]] = []
        for name in chosen:
            col = self.schema[name]
            if col.is_continuous:
                cols.append(np.asarray(self._store[name], dtype=np.float64))
            elif col.is_discrete:
                cols.append(self.codes(name).astype(np.float64))
            elif col.is_datetime:
                stamps = np.asarray(self._store[name], dtype="datetime64[ns]")
                cols.append(stamps.view(np.int64).astype(np.float64))
            else:
                raise ValueError(f"column {name!r} is excluded; it has no numeric form")
        return np.asarray(np.column_stack(cols), dtype=np.float64)

    def to_numpy(self) -> npt.NDArray[np.float64]:
        return self.matrix()

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame({name: self._store[name] for name in self.schema.names})

    # -- missing data --------------------------------------------------------

    def missing(self) -> dict[str, int]:
        """Missing-value count per column."""
        return {name: int(_missing_mask(self._store[name]).sum()) for name in self.schema.names}

    @property
    def has_missing(self) -> bool:
        return any(count > 0 for count in self.missing().values())

    def dropna(self) -> Self:
        """Drop every row with a missing value in any column."""
        keep = np.ones(self.n_rows, dtype=bool)
        for arr in self._store.values():
            keep &= ~_missing_mask(arr)
        store = {name: arr[keep] for name, arr in self._store.items()}
        return replace(self, _store=store, n_rows=int(keep.sum()))

    # -- immutable edits -----------------------------------------------------

    def select(self, names: Iterable[str]) -> Self:
        """A new dataset with the named columns, plus any index columns, in order."""
        chosen = [str(n) for n in names]
        absent = [n for n in chosen if n not in self.schema]
        if absent:
            raise KeyError(f"no such columns: {absent}")
        for index_name in self.index_columns:
            if index_name not in chosen:
                chosen.append(index_name)
        schema = Schema(tuple(self.schema[n] for n in chosen))
        store = {n: self._store[n] for n in chosen}
        return replace(self, schema=schema, _store=store)

    def with_type(
        self,
        name: str,
        kind: VariableType | str | Ordinal,
        categories: Sequence[Any] | None = None,
    ) -> Self:
        """A new dataset with ``name`` retyped, categories read from data when needed."""
        if name not in self.schema:
            raise KeyError(name)
        new = _resolve_type(name, _unique_nonnull(self._store[name]), kind, categories)
        cols = tuple(new if c.name == name else c for c in self.schema.columns)
        return replace(self, schema=Schema(cols))

    def exclude(self, *names: str) -> Self:
        """A new dataset with the named columns marked excluded."""
        out = self
        for name in names:
            out = out.with_type(name, VariableType.EXCLUDED)
        return out

    # -- graph alignment -----------------------------------------------------

    def alignment(self, graph: MixedGraph) -> list[str]:
        """Where the variables and the graph's nodes disagree, empty when they match."""
        nodes = set(graph.nodes)
        cols = set(self.variables)
        out = [f"graph node {n!r} has no column" for n in sorted(nodes - cols)]
        out += [f"column {c!r} is not a graph node" for c in sorted(cols - nodes)]
        return out

    def aligns_with(self, graph: MixedGraph) -> bool:
        """True when the variables are exactly the graph's nodes."""
        return not self.alignment(graph)

    def __repr__(self) -> str:
        parts = [
            f"{len(names)} {kind.value}"
            for kind in VariableType
            if (names := self.schema.of_kind(kind))
        ]
        body = ", ".join(parts)
        return f"{type(self).__name__}(rows={self.n_rows}, cols={self.n_columns}: {body})"
