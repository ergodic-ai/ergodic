# SPDX-License-Identifier: Apache-2.0
"""Hierarchical time-series: series ordered by time, organized in an aggregation tree.

Build it with :func:`hts`, giving the time column and a :class:`Hierarchy` (or a
parent-to-children dict). The columns are the series, and the hierarchy says how
they sum up. Coherence (a parent equals the aggregate of its children) is
checkable, and a parent that has no column can be computed from its leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._frame import to_columns
from ._hierarchy import Hierarchy
from ._schema import infer_schema
from ._timeseries import TimeSeriesData

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

    from ._schema import Ordinal, VariableType


@dataclass(frozen=True, eq=False)
class HierarchicalTSData(TimeSeriesData):
    """A time-series whose columns form an aggregation hierarchy."""

    hierarchy: Hierarchy

    def aggregate(self, node: str) -> npt.NDArray[np.float64]:
        """Build a node's series by aggregating its leaf descendants' columns."""
        leaves = self.hierarchy.leaf_descendants(node)
        missing = [leaf for leaf in leaves if leaf not in self._store]
        if missing:
            raise KeyError(f"missing leaf columns for {node!r}: {missing}")
        stacked = np.vstack([np.asarray(self._store[leaf], dtype=np.float64) for leaf in leaves])
        if self.hierarchy.aggregation == "mean":
            reduced = stacked.mean(axis=0)
        else:
            reduced = stacked.sum(axis=0)
        return np.asarray(reduced, dtype=np.float64)

    def coherence(self, rtol: float = 1e-05, atol: float = 1e-08) -> list[str]:
        """Where a present parent column does not match the aggregate of its children."""
        out: list[str] = []
        for parent in self.hierarchy.internal_nodes:
            kids = self.hierarchy.children_of(parent)
            if parent not in self._store or any(kid not in self._store for kid in kids):
                continue
            total = np.zeros(self.n_rows, dtype=np.float64)
            for kid in kids:
                total = total + np.asarray(self._store[kid], dtype=np.float64)
            if self.hierarchy.aggregation == "mean":
                total = total / len(kids)
            present = np.asarray(self._store[parent], dtype=np.float64)
            if not np.allclose(present, total, rtol=rtol, atol=atol):
                out.append(f"{parent} != {self.hierarchy.aggregation}({', '.join(kids)})")
        return out

    def is_coherent(self, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """True when every present parent equals the aggregate of its children."""
        return not self.coherence(rtol=rtol, atol=atol)


def hts(
    data: Any,
    *,
    time: str,
    hierarchy: Hierarchy | Mapping[str, Iterable[str]],
    freq: str | None = None,
    columns: Sequence[str] | None = None,
    types: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
) -> HierarchicalTSData:
    """Build a :class:`HierarchicalTSData` from series columns and a hierarchy."""
    tree = hierarchy if isinstance(hierarchy, Hierarchy) else Hierarchy.tree(hierarchy)
    store, n_rows, declared = to_columns(data, columns)
    if time not in store:
        raise KeyError(f"time column {time!r} not found in {sorted(store)}")
    missing_leaves = [leaf for leaf in tree.leaves if leaf not in store]
    if missing_leaves:
        raise ValueError(f"hierarchy leaves have no column: {sorted(missing_leaves)}")
    overrides: dict[str, VariableType | str | Ordinal] = dict(types or {})
    if time not in overrides:
        overrides[time] = "datetime" if store[time].dtype.kind == "M" else "continuous"
    for node in tree.nodes:
        if node in store and node not in overrides:
            overrides[node] = "continuous"
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
    return HierarchicalTSData(
        schema=schema, n_rows=n_rows, _store=ordered, time=time, freq=freq, hierarchy=tree
    )
