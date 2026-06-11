# SPDX-License-Identifier: Apache-2.0
"""Process mining and intelligence: from event logs to causal questions.

The descriptive layer recovers the daily workflow over an
:class:`~ergodic.data.EventLog`: :func:`process_map` builds the
directly-follows map as a value object (frequencies, gap statistics, and a
``simplify`` that records what it dropped), :func:`performance` reports cycle
times and, when a lifecycle column exists, the waiting and service split,
:func:`variant_table` ranks the trace variants, and the filter vocabulary
(:func:`filter_time`, :func:`filter_variants`, :func:`filter_endpoints`,
:func:`filter_activities`, :func:`filter_follows`, :func:`filter_performance`)
composes, with each filter saying whether it drops events or cases.

The causal layer is the point. :func:`case_table` encodes the log into one row
per case with timestamp-gated features, the named process confounders
(workload, calendar), and an auto-built :class:`~ergodic.knowledge.DomainKnowledge`
whose tiers forbid post-treatment adjustment mechanically; its output feeds
:func:`~ergodic.discovery.discover`, :func:`~ergodic.identification.identify_effect`,
and :func:`~ergodic.inference.estimate_effect` unchanged. :func:`kpi_panel`
aggregates the log into a :class:`~ergodic.data.PanelData` (or a single
series), so a process change becomes a natural experiment for
:func:`~ergodic.inference.did`, :func:`~ergodic.inference.event_study`,
:func:`~ergodic.inference.synthetic_control`, and
:func:`~ergodic.inference.interrupted_time_series`.
"""

from __future__ import annotations

from ._case_table import CaseTable, case_table
from ._filters import (
    filter_activities,
    filter_endpoints,
    filter_follows,
    filter_performance,
    filter_time,
    filter_variants,
)
from ._map import END, START, ArcStats, ProcessMap, process_map
from ._panel import kpi_panel
from ._performance import ActivityStats, ProcessPerformance, performance
from ._variants import Variant, variant_table

__all__ = [
    "END",
    "START",
    "ActivityStats",
    "ArcStats",
    "CaseTable",
    "ProcessMap",
    "ProcessPerformance",
    "Variant",
    "case_table",
    "filter_activities",
    "filter_endpoints",
    "filter_follows",
    "filter_performance",
    "filter_time",
    "filter_variants",
    "kpi_panel",
    "performance",
    "process_map",
    "variant_table",
]
