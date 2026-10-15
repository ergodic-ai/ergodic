# SPDX-License-Identifier: Apache-2.0
"""The data bridge: ergodic tabular datasets to Welt, Welt research tables to ergodic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..data import TabularData, tabular
from ._client import frame_rows, using

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ..data import Ordinal, VariableType


def _require_tabular(data: Any) -> TabularData:
    if not isinstance(data, TabularData):
        raise TypeError(
            f"Welt takes tabular rows; got {type(data).__name__}. Build the data with "
            "ergodic.tabular(...)"
        )
    return data


def upload_dataset(
    data: TabularData,
    *,
    target: str | None = None,
    name: str = "ergodic dataset",
    client: Any = None,
) -> dict[str, Any]:
    """Upload a tabular dataset to your Welt workspace.

    The upload holds the dataset's variables in their column order; excluded
    columns stay local. Values go up as they are stored (categories as their
    labels, missing values as empty cells). Uploading is free, and the dataset
    stays in the workspace until you delete it with ``client.delete_dataset(id)``.

    Args:
        data: The rows to upload.
        target: A variable to mark as the prediction target, for a later fit or
            evaluation. ``None`` uploads a target-free table, which is what causal
            discovery needs.
        name: The dataset name shown in Welt.
        client: A ``welt.Client`` to use as is; ``None`` opens a short-lived one.

    Returns:
        The Welt dataset record, with its ``"id"``.

    Raises:
        TypeError: ``data`` is not a :class:`~ergodic.data.TabularData`.
        ValueError: ``target`` is not one of the variables, or a cell is not a
            number or a string.
    """
    data = _require_tabular(data)
    variables = list(data.variables)
    if not variables:
        raise ValueError("the dataset has no variables to upload")
    if target is not None and target not in variables:
        raise ValueError(f"target {target!r} is not one of the variables {variables}")
    columns, rows = frame_rows(data.to_pandas()[variables])
    with using(client) as welt_client:
        record: dict[str, Any] = welt_client.upload(
            columns=columns, rows=rows, name=name, target=target
        )
    return record


def load_research_dataset(
    dataset_id: str,
    *,
    version: str | None = None,
    types: Mapping[str, VariableType | str | Ordinal] | None = None,
    exclude: Iterable[str] | None = None,
    categorical_threshold: int = 10,
    client: Any = None,
) -> TabularData:
    """Load a dataset from Welt's research catalogue as tabular data.

    Welt verifies the download's size and checksum and reads it into a dataframe;
    ergodic then infers the schema as :func:`~ergodic.data.tabular` does. Each
    download is billed per dataset (see Welt's pricing). Reading Parquet content
    needs ``pyarrow``: ``pip install "welt-client[research]"``.

    Args:
        dataset_id: The research dataset's id, from ``client.research_datasets()``.
        version: A content version to pin; ``None`` takes the current one.
        types: Per-column type overrides, as in :func:`~ergodic.data.tabular`.
        exclude: Columns to keep but not treat as variables.
        categorical_threshold: Integer columns with at most this many distinct
            values are read as categorical, with a warning.
        client: A ``welt.Client`` to use as is; ``None`` opens a short-lived one.

    Returns:
        The table as :class:`~ergodic.data.TabularData`.
    """
    with using(client) as welt_client:
        frame = welt_client.load_research_dataset(dataset_id, version=version)
    return tabular(frame, types=types, exclude=exclude, categorical_threshold=categorical_threshold)
