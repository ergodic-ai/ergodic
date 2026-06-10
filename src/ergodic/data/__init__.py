# SPDX-License-Identifier: Apache-2.0
"""Typed datasets: the data objects discovery and inference read.

Each shape embeds a column :class:`Schema` (continuous, categorical, ordinal,
datetime, or excluded) and shares the accessors on :class:`Dataset`. The shapes
are tabular, time-series, panel, and hierarchical time-series; the event log
follows.

Build a dataset with :func:`tabular`, :func:`timeseries`, :func:`panel`, or
:func:`hts`, each of which takes a pandas or polars dataframe, a dict of columns,
or a numpy array.
"""

from __future__ import annotations

from ._dataset import Dataset
from ._eventlog import EventLog, event_log
from ._hierarchy import Hierarchy
from ._hts import HierarchicalTSData, hts
from ._panel import PanelData, panel
from ._schema import (
    ColumnType,
    Ordinal,
    Schema,
    SchemaInferenceWarning,
    VariableType,
    infer_schema,
)
from ._tabular import TabularData, tabular
from ._timeseries import TimeSeriesData, timeseries

__all__ = [
    "ColumnType",
    "Dataset",
    "EventLog",
    "HierarchicalTSData",
    "Hierarchy",
    "Ordinal",
    "PanelData",
    "Schema",
    "SchemaInferenceWarning",
    "TabularData",
    "TimeSeriesData",
    "VariableType",
    "event_log",
    "hts",
    "infer_schema",
    "panel",
    "tabular",
    "timeseries",
]
