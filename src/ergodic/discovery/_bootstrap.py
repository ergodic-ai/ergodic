# SPDX-License-Identifier: Apache-2.0
"""Bootstrap stability: how much of a discovered graph survives resampling.

One run of a discovery algorithm returns one graph and no sense of how fragile
it is. The bootstrap wrapper reruns any method (constraint or score based,
because it goes through :func:`discover`) on row resamples of the data and
counts what comes back: an adjacency frequency per node pair and a frequency
per oriented edge glyph. The result is the library's first soft output, edge
frequencies instead of a single graph, with :meth:`BootstrapResult.summary`
turning them back into a stability graph at a chosen threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..graph import mixed
from ._discover import discover
from ._registry import TEMPORAL_METHODS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..data import TabularData
    from ..graph import MixedGraph
    from ..knowledge import DomainKnowledge


@dataclass(frozen=True)
class BootstrapResult:
    """Edge frequencies from rerunning one discovery method on bootstrap resamples.

    ``edges`` maps an unordered node pair to the fraction of successful runs
    in which the pair was adjacent, whatever the marks. ``marks`` maps a
    canonical edge glyph (:meth:`~ergodic.graph.Edge.glyph`, for example
    ``"X --> Y"`` or ``"X <-> Y"``) to its frequency, so one pair can spread
    its mass over several glyphs; the glyph frequencies for a pair sum to its
    adjacency frequency. Both mappings hold only pairs and glyphs that
    appeared at least once, with keys in sorted order. Frequencies are over
    the ``n_runs`` successful replicates, not over what was attempted.
    """

    method: str
    n_runs: int
    n_failed: int
    edges: Mapping[frozenset[str], float]
    marks: Mapping[str, float]
    variables: tuple[str, ...] = field(repr=False)

    def frequency(self, a: str, b: str) -> float:
        """The adjacency frequency of the pair, ``0.0`` when it never appeared."""
        return self.edges.get(frozenset((a, b)), 0.0)

    def summary(self, threshold: float = 0.5) -> MixedGraph:
        """The stability graph: pairs at or above ``threshold``, one edge each.

        A qualifying pair gets its modal glyph, the orientation it wore most
        often (ties broken by the sorted glyph string, so the result is
        deterministic). The graph keeps the ``mixed`` kind: a vote over graphs
        carries no equivalence-class guarantee, even when every replicate was
        a CPDAG. Nodes are all the data's variables, so a variable that never
        earned a stable edge shows up isolated. ``threshold`` must be in
        ``(0, 1]``.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        votes: dict[frozenset[str], list[tuple[float, str]]] = {}
        for glyph, freq in self.marks.items():
            parts = glyph.split()
            votes.setdefault(frozenset((parts[0], parts[-1])), []).append((freq, glyph))
        winners: list[str] = []
        for pair, adjacency in self.edges.items():
            if adjacency >= threshold:
                # highest frequency wins; on a tie, the smaller glyph string
                best = min(votes[pair], key=lambda vote: (-vote[0], vote[1]))
                winners.append(best[1])
        return mixed(winners, nodes=self.variables)

    def __repr__(self) -> str:
        stable = sum(1 for freq in self.edges.values() if freq >= 0.5)
        return (
            f"BootstrapResult({self.method}: {self.n_runs} runs, {self.n_failed} failed, "
            f"{stable}/{len(self.edges)} edges stable at 0.5)"
        )


def bootstrap(
    data: TabularData,
    *,
    method: str = "pc",
    n_bootstrap: int = 100,
    seed: int = 0,
    knowledge: DomainKnowledge | None = None,
    **kwargs: Any,
) -> BootstrapResult:
    """Rerun a discovery method on bootstrap resamples and count what survives.

    Answers the question a single run cannot: how stable is this edge under
    resampling? Each replicate draws ``data.n_rows`` rows with replacement,
    runs ``discover(sample, method=method, knowledge=knowledge, **kwargs)``,
    and the result aggregates adjacency and glyph frequencies over the runs
    that finished. Extra keyword arguments flow to the algorithm unchanged
    (``alpha``, ``max_cond_size``, ``score=...``, ``tabu=...``), so any
    registered method, constraint or score based, sits behind the same call.

    The frequencies measure resampling variability, not Bayesian uncertainty;
    they are not posterior probabilities over graphs (graph posteriors belong
    to a later phase). The run is deterministic: the same data, seed, and
    settings give the same result.

    A replicate that raises ``ValueError`` or ``RuntimeError`` (degenerate
    resamples happen: a constant column, a knowledge conflict on noisy data)
    counts in ``n_failed`` and is skipped; frequencies are over the successful
    runs only. When every replicate fails, the wrapper raises ``RuntimeError``
    chaining the last error.

    Args:
        data: The tabular dataset to resample.
        method: A registered discovery method name, as in :func:`discover`.
        n_bootstrap: How many resamples to draw, at least 1.
        seed: Seed for the resampling generator.
        knowledge: Hard constraints, handed to every replicate.
        **kwargs: Passed through to ``discover`` and on to the algorithm.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if method.lower() in TEMPORAL_METHODS:
        raise ValueError(
            f"bootstrap resamples rows, which breaks the dependence {method!r} reads; "
            "a moving-block bootstrap for temporal methods is future work"
        )
    rng = np.random.default_rng(seed)
    n = data.n_rows
    edge_counts: dict[frozenset[str], int] = {}
    mark_counts: dict[str, int] = {}
    n_runs = 0
    n_failed = 0
    last_error: Exception | None = None
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = data.take(idx)
        try:
            result = discover(sample, method=method, knowledge=knowledge, **kwargs)
        except (ValueError, RuntimeError) as error:
            n_failed += 1
            last_error = error
            continue
        n_runs += 1
        for edge in result.graph.edges:
            pair = frozenset((edge.u, edge.v))
            edge_counts[pair] = edge_counts.get(pair, 0) + 1
            glyph = edge.glyph()
            mark_counts[glyph] = mark_counts.get(glyph, 0) + 1
    if n_runs == 0:
        raise RuntimeError(
            f"every bootstrap replicate failed ({n_failed} of {n_bootstrap}); "
            "the chained error is the last failure"
        ) from last_error
    edges = {
        pair: count / n_runs
        for pair, count in sorted(edge_counts.items(), key=lambda item: tuple(sorted(item[0])))
    }
    marks = {glyph: count / n_runs for glyph, count in sorted(mark_counts.items())}
    return BootstrapResult(
        method=method,
        n_runs=n_runs,
        n_failed=n_failed,
        edges=edges,
        marks=marks,
        variables=data.variables,
    )
