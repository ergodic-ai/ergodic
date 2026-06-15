# SPDX-License-Identifier: Apache-2.0
"""The :class:`Identification` record: the result of trying to identify an effect."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Identification:
    """The result of trying to identify an effect from a graph."""

    treatment: frozenset[str]
    outcome: frozenset[str]
    identifiable: bool
    strategy: str  # "null", "adjustment", "frontdoor", "iv", or "none"
    adjustment_set: frozenset[str] | None = None
    mediators: frozenset[str] | None = None
    instruments: tuple[tuple[str, frozenset[str]], ...] = ()
    estimand: str = ""

    def __bool__(self) -> bool:
        return self.identifiable

    def __repr__(self) -> str:
        arrow = f"{sorted(self.treatment)} -> {sorted(self.outcome)}"
        if not self.identifiable:
            return f"Identification(not identifiable, {arrow})"
        return f"Identification({self.strategy}: {self.estimand})"
