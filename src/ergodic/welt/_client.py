# SPDX-License-Identifier: Apache-2.0
"""One rule for every Welt call: use the caller's client, or a short-lived one."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt
    import pandas as pd


@contextmanager
def using(client: Any) -> Iterator[Any]:
    """Yield ``client`` unchanged, or open a ``welt.Client()`` and close it afterwards.

    A caller's client is never closed here; it may serve other work.
    """
    if client is not None:
        yield client
        return
    import welt

    with welt.Client() as fresh:
        yield fresh


def _cell(value: Any) -> Any:
    """A JSON-ready table cell: numpy scalars unwrapped, missing values as None."""
    import math

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        raise ValueError("Welt tables cannot hold infinite values")
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"Welt tables hold numbers and strings; got a {type(value).__name__} cell")
    return value


def frame_rows(frame: pd.DataFrame) -> tuple[list[str], list[list[Any]]]:
    """Column names and JSON-ready rows of a dataframe, in column order."""
    columns = [str(name) for name in frame.columns]
    rows = [[_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return columns, rows


def matrix_rows(X: npt.NDArray[np.float64]) -> list[list[float | None]]:
    """JSON-ready rows of a float matrix; NaN becomes a missing cell."""
    return [[_cell(value) for value in row] for row in X.tolist()]
