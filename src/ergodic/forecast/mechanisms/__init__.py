# SPDX-License-Identifier: Apache-2.0
"""The mechanism zoo: one structural equation per kind of series.

:class:`StructuralMechanism` is the workhorse and the only inhabitant so far: a
damped local linear trend, a seasonal term, and a regression on whatever parents
the graph gives the node. It is composed from those pieces rather than written
as one formula, so a fitted node can be taken apart again into trend, season,
and one bar per driver.

The dynamics that read a driver as a force rather than a covariate (causal
Ornstein-Uhlenbeck, Bass diffusion) land with the driver sprint.
"""

from __future__ import annotations

from ._structural import StructuralMechanism

__all__ = ["StructuralMechanism"]
