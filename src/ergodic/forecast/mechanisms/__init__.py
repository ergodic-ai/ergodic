# SPDX-License-Identifier: Apache-2.0
"""The mechanism zoo: one structural equation per kind of series.

:class:`StructuralMechanism` is the workhorse: a damped local linear trend, a
seasonal term, and a regression on whatever parents the graph gives the node. It
is composed from those pieces rather than written as one formula, so a fitted
node can be taken apart again into trend, season, and one bar per driver.

Two more read a driver as a force rather than as a covariate.
:class:`CausalOU` puts the drivers in the long-run attractor, so a driver decides
where the series is heading and ``kappa`` decides how fast it gets there.
:class:`Bass` is the S-curve for something with no history at all: a launch, an
NPI, a new store format, where two numbers and a market size are the whole model.

Choosing between them is a question about the series, not about fit. Does a
driver move the value or the equilibrium? Is there a history to extrapolate, or
only a shape to fill in?
"""

from __future__ import annotations

from ._bass import Bass
from ._ou import CausalOU
from ._structural import StructuralMechanism

__all__ = ["Bass", "CausalOU", "StructuralMechanism"]
