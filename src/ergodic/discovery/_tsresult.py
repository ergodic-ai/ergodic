# SPDX-License-Identifier: Apache-2.0
"""The time-series discovery result: a window graph plus the per-link p-values.

There is no summary *graph* on purpose. A stationary process happily carries
feedback across lags (x drives y at lag 1 while y drives x at lag 2), which a
variable-level graph would have to write as two edges between one pair, and
the one-edge-per-pair graph model should not be bent around that. The window
graph over the ``v[t-k]`` nodes is the primary object; :meth:`summary_edges`
gives a flat variable-level reading when that is all you need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._lagged import lagged_node
from ._result import DiscoveryResult

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["TimeSeriesDiscoveryResult"]


@dataclass(frozen=True)
class TimeSeriesDiscoveryResult(DiscoveryResult):
    """A :class:`DiscoveryResult` over window nodes, with the per-link p-values.

    ``links`` maps ``(source, target, lag)`` to the p-value of the test that
    kept the link. A lag of one or more is the lagged edge
    ``source[t-lag] -> target[t]``. A lag of zero is a contemporaneous edge:
    an oriented edge is keyed ``(source, target)``, an undirected edge is
    keyed once with the pair in sorted order.
    """

    max_lag: int = 0
    links: Mapping[tuple[str, str, int], float] = field(default_factory=dict)

    def __repr__(self) -> str:
        n_lagged = sum(1 for _, _, lag in self.links if lag >= 1)
        n_contemporaneous = sum(1 for _, _, lag in self.links if lag == 0)
        ingredient = self.test if self.test is not None else self.score
        body = (
            f"{self.method}/{ingredient}: {self.graph.kind.value} on {len(self.graph.nodes)} "
            f"window nodes, max_lag={self.max_lag}, {n_lagged} lagged links, "
            f"{n_contemporaneous} contemporaneous"
        )
        if self.alpha is not None:
            body += f", alpha={self.alpha:g}"
        return f"TimeSeriesDiscoveryResult({body}, n={self.n})"

    def summary_edges(self) -> frozenset[tuple[str, str]]:
        """Variable-level pairs implied by the links.

        ``(x, y)`` is present when any lagged link ``x -> y`` exists or a
        contemporaneous edge ``x -> y`` is oriented in the window graph. An
        undirected contemporaneous pair contributes both orders, and a lagged
        self-link (autocorrelation) contributes ``(x, x)``.
        """
        out: set[tuple[str, str]] = set()
        for source, target, lag in self.links:
            if lag == 0:
                edge = self.graph.get_edge(lagged_node(source, 0), lagged_node(target, 0))
                if edge is not None and edge.is_undirected:
                    out.add((target, source))
            out.add((source, target))
        return frozenset(out)
