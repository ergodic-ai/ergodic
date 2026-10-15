# SPDX-License-Identifier: Apache-2.0
"""Column schema: the typed view of a dataset's columns.

A :class:`Schema` is an ordered set of :class:`ColumnType` records, one per
column, that says how each column should be read: continuous, categorical,
ordinal, datetime, or excluded. Algorithms switch on these types to pick a test,
a score, or an estimator, so the schema is fixed once and shared, never
re-inferred per algorithm.

Types are inferred from the data with sensible defaults, and the user overrides
them. The one genuinely ambiguous case is integer columns, which can be a count
or an encoded category. Those are inferred and warned about, not guessed in
silence.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import numpy.typing as npt


class SchemaInferenceWarning(UserWarning):
    """A column type was guessed from ambiguous data (for example integer levels)."""


class VariableType(Enum):
    """How a column is measured, which decides the methods that apply to it."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    EXCLUDED = "excluded"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Ordinal:
    """Mark a column ordinal, listing its categories in increasing order.

    Use it in a type override: ``types={"stage": Ordinal(["I", "II", "III"])}``.
    """

    levels: Sequence[Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))


@dataclass(frozen=True)
class ColumnType:
    """The type of one column: its kind, and the categories when it is discrete."""

    name: str
    kind: VariableType
    categories: tuple[Any, ...] | None = None
    inferred: bool = True

    @property
    def ordered(self) -> bool:
        """Whether the categories have an order (the column is ordinal)."""
        return self.kind is VariableType.ORDINAL

    @property
    def is_continuous(self) -> bool:
        """Whether the column is continuous."""
        return self.kind is VariableType.CONTINUOUS

    @property
    def is_categorical(self) -> bool:
        """Whether the column is categorical (unordered)."""
        return self.kind is VariableType.CATEGORICAL

    @property
    def is_ordinal(self) -> bool:
        """Whether the column is ordinal."""
        return self.kind is VariableType.ORDINAL

    @property
    def is_datetime(self) -> bool:
        """Whether the column holds timestamps."""
        return self.kind is VariableType.DATETIME

    @property
    def is_excluded(self) -> bool:
        """Whether the column is excluded from modeling."""
        return self.kind is VariableType.EXCLUDED

    @property
    def is_discrete(self) -> bool:
        """Categorical or ordinal: backed by a finite set of categories."""
        return self.kind in (VariableType.CATEGORICAL, VariableType.ORDINAL)

    @property
    def cardinality(self) -> int | None:
        """The number of categories, or ``None`` when the column has none."""
        return None if self.categories is None else len(self.categories)

    @property
    def is_binary(self) -> bool:
        """Whether the column is discrete with exactly two categories."""
        return self.is_discrete and self.cardinality == 2


@dataclass(frozen=True)
class Schema:
    """An ordered set of column types, queryable by name and by kind."""

    columns: tuple[ColumnType, ...]

    @cached_property
    def _by_name(self) -> dict[str, ColumnType]:
        return {c.name: c for c in self.columns}

    def __getitem__(self, name: str) -> ColumnType:
        return self._by_name[name]

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[ColumnType]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def get(self, name: str) -> ColumnType | None:
        """Return the type of column ``name``, or ``None`` when there is no such column."""
        return self._by_name.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        """The column names, in order."""
        return tuple(c.name for c in self.columns)

    def of_kind(self, kind: VariableType) -> tuple[str, ...]:
        """Return the names of the columns of one kind, in schema order."""
        return tuple(c.name for c in self.columns if c.kind is kind)

    @property
    def continuous(self) -> tuple[str, ...]:
        """The names of the continuous columns."""
        return self.of_kind(VariableType.CONTINUOUS)

    @property
    def categorical(self) -> tuple[str, ...]:
        """The names of the categorical columns (ordinal columns are not included)."""
        return self.of_kind(VariableType.CATEGORICAL)

    @property
    def ordinal(self) -> tuple[str, ...]:
        """The names of the ordinal columns."""
        return self.of_kind(VariableType.ORDINAL)

    @property
    def datetime(self) -> tuple[str, ...]:
        """The names of the datetime columns."""
        return self.of_kind(VariableType.DATETIME)

    @property
    def excluded(self) -> tuple[str, ...]:
        """The names of the excluded columns."""
        return self.of_kind(VariableType.EXCLUDED)

    @property
    def discrete(self) -> tuple[str, ...]:
        """Categorical and ordinal columns together."""
        return tuple(c.name for c in self.columns if c.is_discrete)

    @property
    def variables(self) -> tuple[str, ...]:
        """Every column the algorithms see: all but the excluded ones."""
        return tuple(c.name for c in self.columns if not c.is_excluded)

    def with_type(
        self, name: str, kind: VariableType | str | Ordinal, categories: Sequence[Any] | None = None
    ) -> Schema:
        """Return a new schema with ``name`` retyped. Marks the column user-set."""
        if name not in self._by_name:
            raise KeyError(name)
        new = _resolve_type(name, self[name].categories, kind, categories)
        cols = tuple(new if c.name == name else c for c in self.columns)
        return Schema(cols)

    def __repr__(self) -> str:
        parts = [
            f"{len(names)} {kind.value}" for kind in VariableType if (names := self.of_kind(kind))
        ]
        return f"Schema({len(self)} columns: {', '.join(parts)})"


# -- inference ---------------------------------------------------------------


def _missing_mask(values: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    import pandas as pd

    return np.asarray(pd.isna(values), dtype=bool)


def _unique_nonnull(values: npt.NDArray[Any]) -> tuple[Any, ...]:
    present = values[~_missing_mask(values)]
    seen = list(dict.fromkeys(present.tolist()))
    with contextlib.suppress(TypeError):
        seen = sorted(seen)
    return tuple(seen)


def _encode(
    values: npt.NDArray[Any], categories: tuple[Any, ...] | None, ordered: bool, name: str
) -> npt.NDArray[np.int64]:
    """Category codes for one column: -1 for missing, an error for unknown values."""
    if categories is None:
        raise ValueError("cannot encode a column with no categories")
    import pandas as pd

    allowed = set(categories)
    present = values[~_missing_mask(values)]
    offending = [v for v in dict.fromkeys(present.tolist()) if v not in allowed]
    if offending:
        raise ValueError(f"column {name!r} has values outside its declared categories: {offending}")
    codes = pd.Categorical(values, categories=list(categories), ordered=ordered).codes
    return np.asarray(codes, dtype=np.int64)


def _coerce_kind(value: VariableType | str) -> VariableType:
    if isinstance(value, VariableType):
        return value
    if isinstance(value, str):
        try:
            return VariableType(value.lower())
        except ValueError:
            allowed = ", ".join(k.value for k in VariableType)
            raise ValueError(f"unknown variable type {value!r}; use one of {allowed}") from None
    raise TypeError(f"variable type must be a str or VariableType, got {type(value)!r}")


def _resolve_type(
    name: str,
    current: tuple[Any, ...] | None,
    spec: VariableType | str | Ordinal,
    categories: Sequence[Any] | None,
) -> ColumnType:
    if isinstance(spec, Ordinal):
        return ColumnType(name, VariableType.ORDINAL, tuple(spec.levels), inferred=False)
    kind = _coerce_kind(spec)
    if categories is not None:
        return ColumnType(name, kind, tuple(categories), inferred=False)
    if kind in (VariableType.CONTINUOUS, VariableType.DATETIME, VariableType.EXCLUDED):
        return ColumnType(name, kind, inferred=False)
    return ColumnType(name, kind, current, inferred=False)


def _infer_one(name: str, values: npt.NDArray[Any], n_rows: int, threshold: int) -> ColumnType:
    char = values.dtype.kind
    if char == "M":
        return ColumnType(name, VariableType.DATETIME)
    if char == "m":
        return ColumnType(name, VariableType.CONTINUOUS)
    if char == "b":
        return ColumnType(name, VariableType.CATEGORICAL, categories=(False, True))
    if char == "f":
        return ColumnType(name, VariableType.CONTINUOUS)
    if char in ("i", "u"):
        cats = _unique_nonnull(values)
        if 0 < len(cats) <= threshold:
            warnings.warn(
                f"column {name!r}: inferred categorical from {len(cats)} integer levels. "
                f"Pass types={{{name!r}: 'continuous'}} to keep it numeric.",
                SchemaInferenceWarning,
                stacklevel=2,
            )
            return ColumnType(name, VariableType.CATEGORICAL, categories=cats)
        return ColumnType(name, VariableType.CONTINUOUS)
    cats = _unique_nonnull(values)
    if n_rows > 0 and len(cats) == n_rows:
        warnings.warn(
            f"column {name!r}: every value is distinct, so it looks like an identifier. "
            f"Pass exclude=[{name!r}] to drop it from the variables.",
            SchemaInferenceWarning,
            stacklevel=2,
        )
    return ColumnType(name, VariableType.CATEGORICAL, categories=cats)


def infer_schema(
    store: Mapping[str, npt.NDArray[Any]],
    *,
    n_rows: int,
    declared: Mapping[str, ColumnType] | None = None,
    overrides: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
) -> Schema:
    """Build a schema from columns: user overrides beat source-declared beat inferred."""
    declared = declared or {}
    overrides = overrides or {}
    excluded = set(exclude or ())
    out: list[ColumnType] = []
    for name, values in store.items():
        if name in excluded:
            out.append(ColumnType(name, VariableType.EXCLUDED, inferred=False))
        elif name in overrides:
            spec = overrides[name]
            resolved = _resolve_type(name, _unique_nonnull(values), spec, None)
            if isinstance(spec, Ordinal):
                # encode once now, so a level list that misses a value fails at build
                _encode(values, resolved.categories, resolved.ordered, name)
            out.append(resolved)
        elif name in declared:
            out.append(declared[name])
        else:
            out.append(_infer_one(name, values, n_rows, categorical_threshold))
    return Schema(tuple(out))
