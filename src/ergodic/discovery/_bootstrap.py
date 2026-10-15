# SPDX-License-Identifier: Apache-2.0
"""Bootstrap stability: how much of a discovered graph survives resampling.

One run of a discovery algorithm returns one graph and no sense of how fragile
it is. The bootstrap wrapper reruns any method (constraint or score based,
because it goes through :func:`discover`) on resamples of the data and counts
what comes back: an adjacency frequency per node pair and a frequency per
oriented edge glyph. The result is the library's first soft output, edge
frequencies instead of a single graph, with :meth:`BootstrapResult.summary`
turning them back into a stability graph at a chosen threshold.

Tabular rows are exchangeable, so a replicate is a plain row resample. A
temporal method gets the temporal designs instead: a time series is resampled
in moving blocks that enter the replicate as separate panel entities (so no
window is ever built across a block seam), and a panel is resampled at the
cluster level, whole entities with replacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import Ordinal, PanelData, TimeSeriesData, VariableType, panel
from ..graph import mixed
from ._discover import discover
from ._registry import TEMPORAL_METHODS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy.typing as npt

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
    the ``n_runs`` successful replicates, not over what was attempted. A
    temporal method keys them on the window nodes, ``"X[t-1] --> Y[t]"``.
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


def _ceil_sqrt(n: int) -> int:
    """The integer ceiling of ``sqrt(n)``, exact for any non-negative int."""
    return math.isqrt(n - 1) + 1 if n > 0 else 0


def _fresh_name(base: str, taken: Iterable[str]) -> str:
    """A column name not already in ``taken``, prefixing underscores until free."""
    names = set(taken)
    name = base
    while name in names:
        name = "_" + name
    return name


def _typed_panel(
    source: TimeSeriesData | PanelData,
    columns: dict[str, npt.NDArray[Any]],
    *,
    entity: str,
    time: str,
    freq: str | None,
) -> PanelData:
    """Build the resampled panel, carrying each variable's type from ``source``."""
    types: dict[str, VariableType | str | Ordinal] = {}
    for name in source.variables:
        column_type = source.schema[name]
        types[name] = (
            Ordinal(column_type.categories or ()) if column_type.is_ordinal else column_type.kind
        )
    sample = panel(columns, entity=entity, time=time, freq=freq, types=types)
    for name in source.variables:
        column_type = source.schema[name]
        if column_type.is_categorical and column_type.categories is not None:
            sample = sample.with_type(name, VariableType.CATEGORICAL, column_type.categories)
    return sample


def _cluster_sample(data: PanelData, rng: np.random.Generator) -> PanelData:
    """One cluster resample: whole entities with replacement, relabeled in draw order."""
    entities = data.entities
    entity_column = data.column(data.entity)
    draw = rng.integers(0, len(entities), size=len(entities))
    blocks = [np.flatnonzero(entity_column == entities[j]) for j in draw]
    rows = np.concatenate(blocks)
    labels = np.repeat(
        np.array([f"entity_{i}" for i in range(len(blocks))]),
        [block.size for block in blocks],
    )
    columns: dict[str, npt.NDArray[Any]] = {
        data.entity: labels,
        data.time: data.column(data.time)[rows],
    }
    for name in data.variables:
        columns[name] = data.column(name)[rows]
    return _typed_panel(data, columns, entity=data.entity, time=data.time, freq=data.freq)


def _temporal_sample(
    data: TimeSeriesData | PanelData,
    rng: np.random.Generator,
    *,
    block_length: int | None = None,
    max_lag: int = 1,
) -> PanelData:
    """One temporal resample, returned as a panel the temporal methods can pool.

    A time series of ``T`` rows (kept in time order) draws ``ceil(T / L)``
    block starts uniformly with replacement from ``0 .. T - L``, and each
    block of ``L`` consecutive rows becomes one entity of the sample, labeled
    ``"block_0"``, ``"block_1"``, ... in draw order with fresh integer stamps
    ``0 .. L - 1``: once blocks are drawn the original time values carry no
    usable order, so they are dropped. ``L`` is ``block_length`` when given,
    else ``max(ceil(sqrt(T)), 2 * (max_lag + 1))``. Raises ``ValueError``
    when a block cannot out-span the lag depth (``L <= max_lag``) or the
    series is too short to resample (``T < 2 * L``).

    A panel is resampled at the cluster level: ``n_entities`` entities drawn
    whole with replacement, rows kept in time order, relabeled
    ``"entity_0"``, ... in draw order so a twice-drawn entity stays two
    distinct clusters under per-entity pooling. ``block_length`` does not
    apply to a panel and raises.
    """
    if isinstance(data, PanelData):
        if block_length is not None:
            raise ValueError(
                "panels resample entities whole, so block_length does not apply; leave it None"
            )
        return _cluster_sample(data, rng)
    t = data.n_rows
    length = block_length if block_length is not None else max(_ceil_sqrt(t), 2 * (max_lag + 1))
    if length <= max_lag:
        raise ValueError(
            f"a block must out-span the lag depth: block_length={length} with max_lag={max_lag}"
        )
    if t < 2 * length:
        raise ValueError(
            f"the series is too short to resample: {t} rows against blocks of {length} "
            f"(need at least {2 * length}); pass a smaller block_length"
        )
    n_blocks = -(-t // length)
    starts = rng.integers(0, t - length + 1, size=n_blocks)
    rows = np.concatenate([np.arange(start, start + length) for start in starts])
    entity = _fresh_name("entity", data.variables)
    columns: dict[str, npt.NDArray[Any]] = {
        entity: np.repeat(np.array([f"block_{i}" for i in range(n_blocks)]), length),
        data.time: np.tile(np.arange(length), n_blocks),
    }
    for name in data.variables:
        columns[name] = data.column(name)[rows]
    return _typed_panel(data, columns, entity=entity, time=data.time, freq=None)


def bootstrap(
    data: TabularData | TimeSeriesData | PanelData,
    *,
    method: str = "pc",
    n_bootstrap: int = 100,
    seed: int = 0,
    knowledge: DomainKnowledge | None = None,
    block_length: int | None = None,
    **kwargs: Any,
) -> BootstrapResult:
    """Rerun a discovery method on bootstrap resamples and count what survives.

    Answers the question a single run cannot: how stable is this edge under
    resampling? Each replicate draws a resample, runs
    ``discover(sample, method=method, knowledge=knowledge, **kwargs)``, and
    the result aggregates adjacency and glyph frequencies over the runs that
    finished. Extra keyword arguments flow to the algorithm unchanged
    (``alpha``, ``max_cond_size``, ``score=...``, ``max_lag=...``), so any
    registered method, iid or temporal, sits behind the same call.

    An iid method on a :class:`~ergodic.data.TabularData` draws
    ``data.n_rows`` rows with replacement, the classic bootstrap;
    ``block_length`` does not apply there and raises. A temporal method
    (``"pcmci"``) never resamples rows, which would shred the serial
    dependence it reads. On a :class:`~ergodic.data.TimeSeriesData` it runs a
    moving-block bootstrap: ``ceil(T / L)`` blocks of ``L`` consecutive rows
    (``block_length``, default ``max(ceil(sqrt(T)), 2 * (max_lag + 1))``) are
    drawn with replacement and enter the replicate as separate entities of a
    panel, ``"block_0"``, ``"block_1"``, ... with fresh time stamps.
    Blocks-as-entities is the point: gluing blocks back into one series would
    build lagged windows that straddle two unrelated fragments and let fake
    adjacencies in, while per-entity pooling never reads across a block seam.
    On a :class:`~ergodic.data.PanelData` it runs the cluster bootstrap
    instead, drawing ``n_entities`` entities whole with replacement
    (relabeled in draw order so repeated draws stay distinct);
    ``block_length`` does not apply to a panel and raises. One honesty note:
    block resampling treats the series as stationary at the block scale, so a
    trend or a regime change inside the sampling window leaks into the
    frequencies. An iid method on temporal data still refuses, inside the
    replicate, exactly as :func:`discover` does.

    The frequencies measure resampling variability, not Bayesian uncertainty;
    they are not posterior probabilities over graphs (ergodic has no graph
    posteriors). The run is deterministic: the same data, seed, and
    settings give the same result. A temporal method keys the frequencies on
    window-graph glyphs like ``"X[t-1] --> Y[t]"``.

    A replicate that raises ``ValueError`` or ``RuntimeError`` (degenerate
    resamples happen: a constant column, a knowledge conflict on noisy data)
    counts in ``n_failed`` and is skipped; frequencies are over the successful
    runs only. When every replicate fails, the wrapper raises ``RuntimeError``
    chaining the last error.

    Args:
        data: The dataset to resample: tabular for the iid methods, a time
            series or panel for the temporal ones.
        method: A registered discovery method name, as in :func:`discover`.
            ``welt:<model>`` methods are refused: each resample would be a paid job.
        n_bootstrap: How many resamples to draw, at least 1.
        seed: Seed for the resampling generator.
        knowledge: Hard constraints, handed to every replicate.
        block_length: Moving-block length for a time series under a temporal
            method; ``None`` picks ``max(ceil(sqrt(T)), 2 * (max_lag + 1))``.
        **kwargs: Passed through to ``discover`` and on to the algorithm.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if method.lower().startswith("welt:"):
        raise ValueError(
            f"bootstrap would submit {n_bootstrap} paid Welt jobs for {method!r}, one per "
            "resample, so it refuses Welt methods; run ergodic.discover once instead"
        )
    temporal: TimeSeriesData | PanelData | None = None
    if method.lower() in TEMPORAL_METHODS:
        if not isinstance(data, TimeSeriesData | PanelData):
            raise ValueError(
                f"{method!r} reads temporal data, so its moving-block bootstrap needs "
                f"TimeSeriesData or PanelData; got {type(data).__name__}"
            )
        temporal = data
    elif block_length is not None:
        raise ValueError(
            f"block_length is for temporal data under a temporal method; {method!r} resamples rows"
        )
    max_lag = int(kwargs.get("max_lag", 1))  # PCMCI's own default
    rng = np.random.default_rng(seed)
    n = data.n_rows
    edge_counts: dict[frozenset[str], int] = {}
    mark_counts: dict[str, int] = {}
    n_runs = 0
    n_failed = 0
    last_error: Exception | None = None
    for _ in range(n_bootstrap):
        if temporal is not None:
            sample: TabularData | TimeSeriesData | PanelData = _temporal_sample(
                temporal, rng, block_length=block_length, max_lag=max_lag
            )
        else:
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
