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

The model layer goes past the map. :func:`discover_model` is the inductive
miner: it cuts the log into a block-structured :class:`ProcessTree` (build one
by hand with :func:`seq`, :func:`xor`, :func:`par`, :func:`loop`, and
:data:`TAU`) whose :meth:`~ProcessTree.to_net` gives a :class:`WorkflowNet`
that is sound by construction, with :func:`is_sound` checking any net for
real. :func:`conformance` replays the log against a model, token replay for
the fast fitness and optimal alignments for the per-variant deviations. And
:func:`decision_points` with :func:`decision_table` make the model causal: an
exclusive split is a decision, the branch taken is a treatment, the features
known at the decision moment are the honest adjusters, and positivity ships
as a first-class report instead of a footnote.
"""

from __future__ import annotations

from ._case_table import CaseTable, case_table
from ._conformance import ConformanceResult, VariantConformance, conformance
from ._decisions import (
    DecisionPoint,
    DecisionTable,
    PositivityReport,
    decision_points,
    decision_table,
)
from ._filters import (
    filter_activities,
    filter_endpoints,
    filter_follows,
    filter_performance,
    filter_time,
    filter_variants,
)
from ._map import END, START, ArcStats, ProcessMap, process_map
from ._mine import ProcessModel, discover_model
from ._net import WorkflowNet, net_of_tree
from ._panel import kpi_panel
from ._performance import ActivityStats, ProcessPerformance, performance
from ._soundness import is_sound
from ._tree import TAU, ProcessTree, loop, par, seq, xor
from ._variants import Variant, variant_table

__all__ = [
    "END",
    "START",
    "TAU",
    "ActivityStats",
    "ArcStats",
    "CaseTable",
    "ConformanceResult",
    "DecisionPoint",
    "DecisionTable",
    "PositivityReport",
    "ProcessMap",
    "ProcessModel",
    "ProcessPerformance",
    "ProcessTree",
    "Variant",
    "VariantConformance",
    "WorkflowNet",
    "case_table",
    "conformance",
    "decision_points",
    "decision_table",
    "discover_model",
    "filter_activities",
    "filter_endpoints",
    "filter_follows",
    "filter_performance",
    "filter_time",
    "filter_variants",
    "is_sound",
    "kpi_panel",
    "loop",
    "net_of_tree",
    "par",
    "performance",
    "process_map",
    "seq",
    "variant_table",
    "xor",
]
