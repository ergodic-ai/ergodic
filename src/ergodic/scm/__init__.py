# SPDX-License-Identifier: Apache-2.0
"""The SCM pillar: a Markovian Bayesian structural causal model, PyMC under the hood.

A :class:`SCM` is a :class:`~ergodic.graph.MixedGraph` plus a fitted mechanism per
node. Every node is a structural equation over an explicit exogenous-noise
variable, so the model answers the three rungs of the causal ladder: observational
sampling, ``pm.do`` interventions, and the explicit-noise twin-network
counterfactual (abduct the unit's noise on the factual graph, predict on an action
branch that reuses it).

The honesty thesis is built in. Before answering an effect, the SCM asks the
identification layer. An identified effect returns a posterior
:class:`SCMResult`; a non-identified one returns a :class:`Sensitivity` (the
``b(c)`` curve over the assumed confounding), never a naked point, so a caller
cannot accidentally read a confident number off an effect the data does not
determine.

PyMC is an optional dependency. Importing this package never imports pymc; the
``import pymc`` lives inside the methods that fit or query, so the bare install
keeps working (the :class:`~ergodic.inference.PyMCLearner` precedent). Install the
extra with ``pip install ergodic[scm]``.

Mechanisms are auto-assigned by the column schema (continuous to a Bayesian-linear
Gaussian-additive, binary and categorical to a Gumbel-max), and overridable.
"""

from __future__ import annotations

from ._bounds import Sensitivity
from ._mechanism import (
    BinaryGumbelMax,
    Capability,
    CategoricalGumbelMax,
    GaussianAdditive,
    Mechanism,
)
from ._result import CategoricalResult, SCMResult
from ._scm import SCM, InterventionView

__all__ = [
    "SCM",
    "BinaryGumbelMax",
    "Capability",
    "CategoricalGumbelMax",
    "CategoricalResult",
    "GaussianAdditive",
    "InterventionView",
    "Mechanism",
    "SCMResult",
    "Sensitivity",
]
