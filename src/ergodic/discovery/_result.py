# SPDX-License-Identifier: Apache-2.0
"""The discovery result: a graph with its provenance.

A discovered graph without the story of how it was found is hard to defend and
hard to debug. :class:`DiscoveryResult` carries the graph (typed by face: a
CPDAG from PC, a DAG from exact search), the algorithm and ingredient names,
the significance level where one applied, and the run diagnostics. Constraint
methods also hand back their separating sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..graph import MixedGraph


@dataclass(frozen=True)
class DiscoveryResult:
    """What a discovery run found, and how.

    ``graph`` is the headline. ``test`` or ``score`` names the ingredient the
    algorithm consumed (one of them is ``None``). ``diagnostics`` carries the
    run counters (tests by conditioning size, cache hits, score evaluations).
    ``sepsets`` maps an unordered removed pair to the set that separated it,
    when a constraint method produced one.
    """

    graph: MixedGraph
    method: str
    test: str | None = None
    score: str | None = None
    alpha: float | None = None
    n: int = 0
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    sepsets: Mapping[frozenset[str], frozenset[str]] | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        ingredient = self.test if self.test is not None else self.score
        body = f"{self.method}/{ingredient}: {self.graph.kind}"
        body += f", {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"
        if self.alpha is not None:
            body += f", alpha={self.alpha:g}"
        return f"DiscoveryResult({body}, n={self.n})"
