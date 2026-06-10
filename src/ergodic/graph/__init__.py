# SPDX-License-Identifier: Apache-2.0
"""Causal graphs: one mixed-graph core with typed faces (DAG, ADMG, CPDAG, PAG).

Build graphs with the :func:`dag`, :func:`admg`, :func:`cpdag`, :func:`pag`, or
:func:`mixed` helpers. Edges read as glyph strings, so ``dag(["X -> Y", "Y -> Z"])``
and ``pag(["X o-> Y", "Y <-> Z"])`` do what they look like.
"""

from __future__ import annotations

from ._graph import (
    Edge,
    EdgeSpec,
    GraphKind,
    InvalidGraphError,
    MixedGraph,
    admg,
    cpdag,
    dag,
    mixed,
    pag,
)
from ._marks import Mark
from ._meek import consistent_extension, cpdag_of_dag, extendable, meek_closure
from ._separation import d_separated, m_connected, m_separated

__all__ = [
    "Edge",
    "EdgeSpec",
    "GraphKind",
    "InvalidGraphError",
    "Mark",
    "MixedGraph",
    "admg",
    "consistent_extension",
    "cpdag",
    "cpdag_of_dag",
    "d_separated",
    "dag",
    "extendable",
    "m_connected",
    "m_separated",
    "meek_closure",
    "mixed",
    "pag",
]
