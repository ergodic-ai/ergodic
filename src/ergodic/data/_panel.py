# SPDX-License-Identifier: Apache-2.0
"""Panel data: a grid of entities observed over time.

Build it with :func:`panel`, naming the entity and time columns. The pair keys
the rows, so it has to be unique, and rows come back grouped by entity and
ordered in time. Both index columns stay out of the modeled variables.

The panel designs (difference-in-differences, event studies, synthetic control)
live in :mod:`ergodic.inference`. This is the container, with the panel-specific queries
those methods need: the entity and period counts, whether the grid is balanced,
and which variables are constant within an entity.
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

    from ._schema import Ordinal, VariableType


@dataclass(frozen=True, eq=False)
class PanelData(Dataset):
    """Entities observed over time, keyed by a unique (entity, time) pair."""

    entity: str
    time: str
    freq: str | None = None

    @property
    def index_columns(self) -> tuple[str, ...]:
        return (self.entity, self.time)

    @property
    def entities(self) -> tuple[Any, ...]:
        return tuple(sorted(set(self._store[self.entity].tolist())))

    @property
    def periods(self) -> tuple[Any, ...]:
        return tuple(sorted(set(self._store[self.time].tolist())))

    @property
    def n_entities(self) -> int:
        return len(set(self._store[self.entity].tolist()))

    @property
    def n_periods(self) -> int:
        return len(set(self._store[self.time].tolist()))

    @property
    def is_balanced(self) -> bool:
        """True when every entity is seen in every period (a full grid)."""
        return self.n_rows == self.n_entities * self.n_periods

    def time_invariant(self) -> tuple[str, ...]:
        """Variables whose value never changes within an entity."""
        frame = self.to_pandas()
        out: list[str] = []
        for name in self.variables:
            counts = frame.groupby(self.entity)[name].nunique(dropna=False)
            if bool((counts <= 1).all()):
                out.append(name)
        return tuple(out)


def panel(
    data: Any,
    *,
    entity: str,
    time: str,
    freq: str | None = None,
    columns: Sequence[str] | None = None,
    types: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
) -> PanelData:
    """Build a :class:`PanelData`, keyed and sorted by ``(entity, time)``."""
    store, n_rows, declared = to_columns(data, columns)
    for role, name in (("entity", entity), ("time", time)):
        if name not in store:
            raise KeyError(f"{role} column {name!r} not found in {sorted(store)}")
    overrides: dict[str, VariableType | str | Ordinal] = dict(types or {})
    if time not in overrides:
        overrides[time] = "datetime" if store[time].dtype.kind == "M" else "continuous"
    if entity not in overrides:
        overrides[entity] = "categorical"
    schema = infer_schema(
        store,
        n_rows=n_rows,
        declared=declared,
        overrides=overrides,
        exclude=exclude,
        categorical_threshold=categorical_threshold,
    )
    pairs = set(zip(store[entity].tolist(), store[time].tolist(), strict=True))
    if len(pairs) != n_rows:
        raise ValueError(
            "(entity, time) pairs are not unique; a panel needs one row per entity-period"
        )
    order = np.lexsort((store[time], store[entity]))
    ordered = {name: arr[order] for name, arr in store.items()}
    return PanelData(
        schema=schema, n_rows=n_rows, _store=ordered, entity=entity, time=time, freq=freq
    )
