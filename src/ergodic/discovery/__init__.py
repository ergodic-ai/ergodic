# SPDX-License-Identifier: Apache-2.0
"""Causal discovery: learn causal structure from data.

Two ingredients drive the searches. A conditional-independence test
(:class:`CITest`) is a class constructed with data, which makes it the
stateful oracle the constraint algorithms query: :class:`Linear` (partial
correlation), :class:`Discrete` (G squared, natively categorical),
:class:`Mixed` (degenerate Gaussian for mixed columns), :class:`GCM` (any
regression :class:`~ergodic.inference.Learner` becomes a test), and
:class:`DSeparationOracle` as ground truth. A decomposable :class:`Score`
answers ``local(node, parents)`` for the score-based searches: :class:`BIC`
(Gaussian), :class:`DiscreteBIC`, :class:`BDeu`, and :class:`MixedBIC`
(degenerate Gaussian). Algorithms are thin consumers: :class:`PC` wraps a
test and returns a CPDAG, :class:`FCI` wraps a test and returns a PAG (latent
confounders allowed), :class:`MMPC` wraps a test and returns an undirected
skeleton screen, :class:`GES` wraps a score and returns a CPDAG,
:class:`AStar` wraps a score and returns the exact optimum, and
:class:`HillClimb` is the fast greedy baseline. ``discover`` is the one-line
facade, and :func:`bootstrap` reruns any method on resamples and returns edge
frequencies (:class:`BootstrapResult`), the first soft output. Background
knowledge (including a skeleton restriction through ``restrict_to``) enters
through :class:`ergodic.DomainKnowledge`.

Time series and panels get their own methods: :class:`PCMCI` reads a
:class:`~ergodic.data.TimeSeriesData` or :class:`~ergodic.data.PanelData`
through a lagged design matrix (:func:`lagged_tabular`) and returns a window
graph over nodes like ``"X[t-1]"`` (:class:`TimeSeriesDiscoveryResult`), and
:func:`granger` is the predictive baseline, a pair-level table on purpose
rather than a graph.
"""

from __future__ import annotations

from ._bootstrap import BootstrapResult, bootstrap
from ._ci import CIResult, CITest, DSeparationOracle
from ._discover import discover
from ._granger import GrangerResult, granger
from ._lagged import lagged_node, lagged_tabular, parse_lagged
from ._result import DiscoveryResult
from ._score import Score
from ._tsresult import TimeSeriesDiscoveryResult
from .algorithms import (
    BOSS,
    FCI,
    GES,
    MMPC,
    PC,
    PCMCI,
    RFCI,
    AStar,
    DirectLiNGAM,
    Glasso,
    GRaSP,
    HillClimb,
)
from .ci import GCM, Discrete, Linear, Mixed
from .scores import BIC, BDeu, DiscreteBIC, MixedBIC

__all__ = [
    "BIC",
    "BOSS",
    "FCI",
    "GCM",
    "GES",
    "MMPC",
    "PC",
    "PCMCI",
    "RFCI",
    "AStar",
    "BDeu",
    "BootstrapResult",
    "CIResult",
    "CITest",
    "DSeparationOracle",
    "DirectLiNGAM",
    "DiscoveryResult",
    "Discrete",
    "DiscreteBIC",
    "GRaSP",
    "Glasso",
    "GrangerResult",
    "HillClimb",
    "Linear",
    "Mixed",
    "MixedBIC",
    "Score",
    "TimeSeriesDiscoveryResult",
    "bootstrap",
    "discover",
    "granger",
    "lagged_node",
    "lagged_tabular",
    "parse_lagged",
]
