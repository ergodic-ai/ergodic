# SPDX-License-Identifier: Apache-2.0
"""Causal graphs: one mixed-graph core with typed faces (DAG, ADMG, CPDAG, PAG).

Build graphs with the :func:`dag`, :func:`admg`, :func:`cpdag`, :func:`pag`, or
:func:`mixed` helpers. Edges read as glyph strings, so ``dag(["X -> Y", "Y -> Z"])``
and ``pag(["X o-> Y", "Z o-> Y"])`` do what they look like. Separation
(:func:`m_separated`), the orientation machinery (:func:`meek_closure`,
:func:`fci_closure`), the projections between the faces (:func:`cpdag_of_dag`,
:func:`mag_of_dag`, :func:`pag_of_mag`, :func:`mag_of_pag`), the class
toolkit (:func:`possible_descendants`, :func:`visible_edges`,
:func:`dags_of_cpdag`, :func:`mags_of_pag`), and the primitives the
identification algorithms read (:func:`districts`, :func:`topological_order`,
:func:`latent_projection`) live here too.
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
from ._meek import consistent_extension, cpdag_of_dag, dags_of_cpdag, extendable, meek_closure
from ._pag import (
    fci_closure,
    mag_of_dag,
    mag_of_pag,
    mags_of_pag,
    pag_of_mag,
    possible_ancestors,
    possible_descendants,
    visible_edges,
)
from ._primitives import district_of, districts, latent_projection, topological_order
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
    "dags_of_cpdag",
    "district_of",
    "districts",
    "extendable",
    "fci_closure",
    "latent_projection",
    "m_connected",
    "m_separated",
    "mag_of_dag",
    "mag_of_pag",
    "mags_of_pag",
    "meek_closure",
    "mixed",
    "pag",
    "pag_of_mag",
    "possible_ancestors",
    "possible_descendants",
    "topological_order",
    "visible_edges",
]
