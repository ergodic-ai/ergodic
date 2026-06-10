# SPDX-License-Identifier: Apache-2.0
"""Encoding for the discovery ingredients: the test or score owns it, not the data.

A test or score declares the input form it needs and the oracle prepares it
once at bind. ``numeric`` is the block form: continuous and datetime columns
as ``(n, 1)`` floats, discrete columns one-hot with the first level dropped,
``(n, cardinality - 1)``. ``codes`` is the native categorical form: integer
codes plus cardinalities, refused for non-discrete columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..data import TabularData


def _numeric_blocks(data: TabularData) -> dict[str, npt.NDArray[np.float64]]:
    """Per-variable 2-D float blocks, in the declared numeric form."""
    blocks: dict[str, npt.NDArray[np.float64]] = {}
    for name in data.variables:
        col = data.schema[name]
        if col.is_continuous or col.is_datetime:
            blocks[name] = np.asarray(data.matrix([name]), dtype=np.float64)
        elif col.is_discrete:
            codes = data.codes(name)
            width = col.cardinality or 0
            onehot = (codes[:, None] == np.arange(width)[None, :]).astype(np.float64)
            blocks[name] = onehot[:, 1:] if onehot.shape[1] > 1 else onehot[:, :0]
        else:
            raise ValueError(f"variable {name!r} is excluded; it has no numeric form")
    return blocks


def _codes_columns(
    data: TabularData,
) -> tuple[dict[str, npt.NDArray[np.int64]], dict[str, int]]:
    """Per-variable integer codes and cardinalities; every column must be discrete."""
    codes: dict[str, npt.NDArray[np.int64]] = {}
    cards: dict[str, int] = {}
    for name in data.variables:
        col = data.schema[name]
        if not col.is_discrete:
            raise TypeError(
                f"a codes-based test needs discrete columns; {name!r} is {col.kind}; "
                "use Linear or Mixed for continuous data"
            )
        codes[name] = np.asarray(data.codes(name), dtype=np.int64)
        cards[name] = int(col.cardinality or 0)
    return codes, cards
