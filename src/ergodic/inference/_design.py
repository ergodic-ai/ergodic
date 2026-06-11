# SPDX-License-Identifier: Apache-2.0
"""Quasi-experimental designs: the identification record for the panel estimators.

The graph path identifies an effect by what it adjusts for; a design identifies it
by what comparison stands in for the counterfactual. ``Design`` plays the role
:class:`~ergodic.identification.Identification` plays for the graph path: a frozen
record of the strategy, the assumptions it leans on, and the estimand in words.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Design"]


@dataclass(frozen=True)
class Design:
    """How a quasi-experimental estimator claims identification.

    ``strategy`` is one of ``"did"``, ``"synthetic_control"``, or
    ``"interrupted_time_series"``. ``assumptions`` are the untestable claims the
    design rests on, stated in words (parallel trends, convex donor span, no
    co-timed shock); they are what a reader should challenge, so they ride with
    every result.
    """

    strategy: str
    outcome: str
    treatment: str | None
    assumptions: tuple[str, ...]
    estimand: str = ""
