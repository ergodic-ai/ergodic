# SPDX-License-Identifier: Apache-2.0
"""The :class:`Identification` record: the result of trying to identify an effect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._functional import Functional
    from ._id import Hedge


@dataclass(frozen=True)
class Identification:
    """The result of trying to identify an effect from a graph.

    The named strategies (``"null"``, ``"adjustment"``, ``"frontdoor"``,
    ``"iv"``) carry the set or instruments their estimator reads and a string
    ``estimand``. When only the general do-calculus ID algorithm identifies an
    effect (outside the three named criteria, the napkin and Verma graphs and
    their kin), ``strategy`` is ``"id"`` and ``functional`` carries the typed
    estimand, with ``estimand`` set to ``functional.render()``. A
    not-identifiable result built by the ID fallthrough carries the ``hedge``
    witness. Existing strategies set neither, and ``estimand`` stays a string,
    so the field is a pure extension.
    """

    treatment: frozenset[str]
    outcome: frozenset[str]
    identifiable: bool
    strategy: str  # "null", "adjustment", "frontdoor", "iv", "id", or "none"
    adjustment_set: frozenset[str] | None = None
    mediators: frozenset[str] | None = None
    instruments: tuple[tuple[str, frozenset[str]], ...] = ()
    estimand: str = ""
    functional: Functional | None = None
    hedge: Hedge | None = None

    def __bool__(self) -> bool:
        return self.identifiable

    def __repr__(self) -> str:
        arrow = f"{sorted(self.treatment)} -> {sorted(self.outcome)}"
        if not self.identifiable:
            return f"Identification(not identifiable, {arrow})"
        return f"Identification({self.strategy}: {self.estimand})"
