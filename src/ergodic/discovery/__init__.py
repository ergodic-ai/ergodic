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
"""

from __future__ import annotations

from ._bootstrap import BootstrapResult, bootstrap
from ._ci import CIResult, CITest, DSeparationOracle
from ._discover import discover
from ._result import DiscoveryResult
from ._score import Score
from .algorithms import FCI, GES, MMPC, PC, AStar, HillClimb
from .ci import GCM, Discrete, Linear, Mixed
from .scores import BIC, BDeu, DiscreteBIC, MixedBIC

__all__ = [
    "BIC",
    "FCI",
    "GCM",
    "GES",
    "MMPC",
    "PC",
    "AStar",
    "BDeu",
    "BootstrapResult",
    "CIResult",
    "CITest",
    "DSeparationOracle",
    "DiscoveryResult",
    "Discrete",
    "DiscreteBIC",
    "HillClimb",
    "Linear",
    "Mixed",
    "MixedBIC",
    "Score",
    "bootstrap",
    "discover",
]
