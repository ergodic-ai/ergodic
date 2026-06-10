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
from .discovery import discover
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
from .inference import EffectEstimate, EffectTarget, estimate, estimate_effect
from .knowledge import DomainKnowledge, InconsistentKnowledgeError

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
    "Schema",
    "SchemaInferenceWarning",
    "TabularData",
    "TimeSeriesData",
    "VariableType",
    "__version__",
    "adjustment_set",
    "admg",
    "cpdag",
    "d_separated",
    "dag",
    "discover",
    "estimate",
    "estimate_effect",
    "event_log",
    "frontdoor_set",
    "hts",
    "identify_effect",
    "instruments",
    "is_frontdoor_set",
    "is_instrument",
    "is_valid_adjustment_set",
    "m_separated",
    "minimal_adjustment_set",
    "mixed",
    "pag",
    "panel",
    "tabular",
    "timeseries",
]
