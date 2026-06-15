# SPDX-License-Identifier: Apache-2.0
"""Ergodic: Causal AI.

Tools for causal discovery, causal inference, and process intelligence. The
shared objects live at the top level: the mixed-graph family (:class:`MixedGraph`
with the :func:`dag`, :func:`admg`, :func:`cpdag`, :func:`pag` builders) and
:class:`DomainKnowledge` for prior constraints. Identification turns a graph into
an estimand with :func:`identify_effect`.

Homepage: https://ergodic.ai
"""

from __future__ import annotations

from . import datasets
from .data import (
    Dataset,
    EventLog,
    HierarchicalTSData,
    Hierarchy,
    Ordinal,
    PanelData,
    Schema,
    SchemaInferenceWarning,
    TabularData,
    TimeSeriesData,
    VariableType,
    event_log,
    hts,
    panel,
    tabular,
    timeseries,
)
from .discovery import discover, granger
from .graph import (
    Edge,
    GraphKind,
    InvalidGraphError,
    Mark,
    MixedGraph,
    admg,
    cpdag,
    d_separated,
    dag,
    m_separated,
    mixed,
    pag,
)
from .identification import (
    Identification,
    adjustment_set,
    frontdoor_set,
    identify_effect,
    instruments,
    is_frontdoor_set,
    is_instrument,
    is_valid_adjustment_set,
    minimal_adjustment_set,
)
from .inference import (
    EffectEstimate,
    EffectTarget,
    Refutation,
    did,
    estimate,
    estimate_effect,
    event_study,
    interrupted_time_series,
    refute_effect,
    synthetic_control,
)
from .knowledge import DomainKnowledge, InconsistentKnowledgeError
from .process import (
    case_table,
    conformance,
    decision_points,
    decision_table,
    discover_model,
    kpi_panel,
    process_map,
)

__version__ = "0.0.1"

__all__ = [
    "Dataset",
    "DomainKnowledge",
    "Edge",
    "EffectEstimate",
    "EffectTarget",
    "EventLog",
    "GraphKind",
    "HierarchicalTSData",
    "Hierarchy",
    "Identification",
    "InconsistentKnowledgeError",
    "InvalidGraphError",
    "Mark",
    "MixedGraph",
    "Ordinal",
    "PanelData",
    "Refutation",
    "Schema",
    "SchemaInferenceWarning",
    "TabularData",
    "TimeSeriesData",
    "VariableType",
    "__version__",
    "adjustment_set",
    "admg",
    "case_table",
    "conformance",
    "cpdag",
    "d_separated",
    "dag",
    "datasets",
    "decision_points",
    "decision_table",
    "did",
    "discover",
    "discover_model",
    "estimate",
    "estimate_effect",
    "event_log",
    "event_study",
    "frontdoor_set",
    "granger",
    "hts",
    "identify_effect",
    "instruments",
    "interrupted_time_series",
    "is_frontdoor_set",
    "is_instrument",
    "is_valid_adjustment_set",
    "kpi_panel",
    "m_separated",
    "minimal_adjustment_set",
    "mixed",
    "pag",
    "panel",
    "process_map",
    "refute_effect",
    "synthetic_control",
    "tabular",
    "timeseries",
]
