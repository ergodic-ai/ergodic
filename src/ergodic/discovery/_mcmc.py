# SPDX-License-Identifier: Apache-2.0
"""Structure MCMC over DAGs, and its convergence diagnostics.

**The move.** Each step picks one unordered pair uniformly among the pairs with
at least two possible states (a pair the hard knowledge settles never moves),
then proposes one of the pair's other possible states uniformly: from "no edge"
an addition in either direction, from ``i -> j`` a deletion or the reversal
``j -> i``. The proposal is symmetric on the space of pair-state configurations,
``q(G -> G') = q(G' -> G) = 1 / (m (k - 1))`` with ``m`` movable pairs and ``k``
possible states for the chosen pair, so the Hastings ratio is 1 and the
acceptance probability is ``min(1, pi(G') / pi(G))``. A proposal that would close
a directed cycle or break an ancestral order constraint has ``pi(G') = 0`` and is
rejected; that is the Metropolis rule for a target that is zero off the DAGs, so
detailed balance holds with respect to the target restricted to valid DAGs. The
reversal is a single move, so the chain can flip an edge without passing
through the state without it. The target ratio is local: one pair's state
weights, plus, for a score, the local scores of the one or two nodes whose
parents change.

**The chains.** Chain 0 starts from the best graph found before sampling (the
greedy decode or the exact MAP); the others start from random DAGs drawn
uniformly over each pair's possible states with cycle rejection. Every chain
runs ``warmup`` sweeps, then keeps one draw every ``thin`` sweeps, a sweep being
one proposal per movable pair. Randomness comes only from the generator passed in.

**Diagnostics.** The rank-normalized split R-hat and the bulk effective sample
size (Vehtari, Gelman, Simpson, Carpenter and Buerkner 2021) on the log target and
on every directed-edge indicator, the acceptance rate per chain, and the starting
graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._target import GraphTarget

    FloatArray = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]
    StateArray = npt.NDArray[np.uint8]

RHAT_LIMIT: Final = 1.01
"""Warn when the log target's rank-normalized split R-hat exceeds this."""
EDGE_RHAT_LIMIT: Final = 1.05
"""Warn when an edge indicator's split R-hat exceeds this (posterior mean in [0.01, 0.99])."""
ESS_PER_CHAIN: Final = 100
"""Warn when the log target's bulk effective sample size is below this many per chain."""


class PosteriorConvergenceWarning(UserWarning):
    """The structure MCMC chains disagree or mix too slowly for the reported numbers."""


@dataclass(frozen=True)
class ChainRun:
    """Kept draws of every chain: pair states and log targets, plus counters."""

    states: StateArray  # (chains, draws, pairs): 0 none, 1 low -> high, 2 high -> low
    log_targets: FloatArray  # (chains, draws)
    acceptance: tuple[float, ...]
    proposals_per_sweep: int


def pair_index(size: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The unordered pairs ``(low, high)`` in row-major upper-triangle order."""
    low, high = np.triu_indices(size, k=1)
    return low.astype(np.int64), high.astype(np.int64)


def states_of(graph: BoolArray) -> StateArray:
    """A DAG's pair states (0 none, 1 low -> high, 2 high -> low)."""
    low, high = pair_index(graph.shape[0])
    result: StateArray = graph[low, high].astype(np.uint8) + 2 * graph[high, low].astype(np.uint8)
    return result


def graphs_of(states: StateArray, size: int) -> BoolArray:
    """Adjacency matrices from pair states, for a ``(..., pairs)`` stack."""
    low, high = pair_index(size)
    shape = states.shape[:-1]
    graphs = np.zeros((*shape, size, size), dtype=bool)
    graphs[..., low, high] = states == 1
    graphs[..., high, low] = states == 2
    return graphs


class _Chain:
    """One chain's mutable state: parent and child bitmasks, pair states, log target."""

    def __init__(self, target: GraphTarget, start: BoolArray) -> None:
        size = target.size
        self.target = target
        self.size = size
        self.children = [0] * size
        self.parents = [0] * size
        for source, child in zip(*np.nonzero(start), strict=True):
            self.children[int(source)] |= 1 << int(child)
            self.parents[int(child)] |= 1 << int(source)
        self.state = bytearray(states_of(start).tobytes())
        self.current = target.log_target(start)
        if not math.isfinite(self.current):  # pragma: no cover - starts are built valid
            raise ValueError("a chain start is outside the target's support")

    def _reach(self, start: int, links: list[int]) -> int:
        """Bitmask of ``start`` and every node reachable from it along ``links``."""
        seen = 1 << start
        frontier = links[start]
        while frontier:
            seen |= frontier
            nxt = 0
            while frontier:
                bit = frontier & -frontier
                nxt |= links[bit.bit_length() - 1]
                frontier ^= bit
            frontier = nxt & ~seen
        return seen

    def can_add(self, source: int, child: int) -> bool:
        """Whether ``source -> child`` keeps the graph acyclic and the ancestral order."""
        below = self._reach(child, self.children)
        if below >> source & 1:
            return False
        if self.target.ancestral:
            above = self._reach(source, self.parents)
            for first, second in self.target.ancestral:
                if above >> second & 1 and below >> first & 1:
                    return False
        return True

    def set_edge(self, source: int, child: int, present: bool) -> None:
        if present:
            self.children[source] |= 1 << child
            self.parents[child] |= 1 << source
        else:
            self.children[source] &= ~(1 << child)
            self.parents[child] &= ~(1 << source)


def run_chains(
    target: GraphTarget,
    starts: list[BoolArray],
    *,
    draws: int,
    warmup: int,
    thin: int,
    rng: np.random.Generator,
) -> ChainRun:
    """Run one chain per start; ``draws`` kept per chain, ``warmup``/``thin`` in sweeps."""
    size = target.size
    low, high = pair_index(size)
    pairs = low.size
    weights: list[tuple[float, float, float]] = []
    movable: list[int] = []
    others: list[list[list[int]]] = []
    for p in range(pairs):
        i, j = int(low[p]), int(high[p])
        row = (
            float(target.log_none[i, j]),
            float(target.log_edge[i, j]),
            float(target.log_edge[j, i]),
        )
        weights.append(row)
        possible = [s for s in range(3) if row[s] > -math.inf]
        others.append([[s for s in possible if s != old] for old in range(3)])
        if len(possible) >= 2:
            movable.append(p)
    sweep = max(1, len(movable))
    states = np.zeros((len(starts), draws, pairs), dtype=np.uint8)
    log_targets = np.zeros((len(starts), draws))
    acceptance: list[float] = []
    lows = low.tolist()
    highs = high.tolist()
    scored = target.score is not None
    for index, start in enumerate(starts):
        chain = _Chain(target, start)
        total = (warmup + draws * thin) * sweep
        accepted = 0
        kept = 0
        if not movable:
            states[index] = np.frombuffer(bytes(chain.state), dtype=np.uint8)
            log_targets[index] = chain.current
            acceptance.append(0.0)
            continue
        block = 1 << 16
        for offset in range(0, total, block):
            count = min(block, total - offset)
            picks = rng.integers(0, len(movable), size=count).tolist()
            choices = rng.random(count).tolist()
            logs = np.log(rng.random(count)).tolist()
            for step in range(count):
                p = movable[picks[step]]
                old = chain.state[p]
                options = others[p][old]
                new = options[int(choices[step] * len(options))]
                row = weights[p]
                delta = row[new] - row[old]
                i, j = lows[p], highs[p]
                if scored:
                    delta += _local_delta(chain, target, i, j, old, new)
                if logs[step] < delta:
                    # Remove the old edge, then check the new one.
                    if old == 1:
                        chain.set_edge(i, j, False)
                    elif old == 2:
                        chain.set_edge(j, i, False)
                    valid = True
                    if new == 1:
                        valid = chain.can_add(i, j)
                    elif new == 2:
                        valid = chain.can_add(j, i)
                    if valid:
                        if new == 1:
                            chain.set_edge(i, j, True)
                        elif new == 2:
                            chain.set_edge(j, i, True)
                        chain.state[p] = new
                        chain.current += delta
                        accepted += 1
                    elif old == 1:
                        chain.set_edge(i, j, True)
                    elif old == 2:
                        chain.set_edge(j, i, True)
                done = offset + step + 1
                if done > warmup * sweep and (done - warmup * sweep) % (thin * sweep) == 0:
                    states[index, kept] = np.frombuffer(bytes(chain.state), dtype=np.uint8)
                    log_targets[index, kept] = chain.current
                    kept += 1
        acceptance.append(accepted / total)
    return ChainRun(states, log_targets, tuple(acceptance), sweep)


def _local_delta(chain: _Chain, target: GraphTarget, i: int, j: int, old: int, new: int) -> float:
    """The change in local scores when pair ``(i, j)`` moves from ``old`` to ``new``."""
    parents = chain.parents
    before_i, before_j = parents[i], parents[j]
    after_i = before_i & ~(1 << j)
    after_j = before_j & ~(1 << i)
    if new == 1:
        after_j |= 1 << i
    elif new == 2:
        after_i |= 1 << j
    delta = 0.0
    if after_i != before_i:
        delta += target.local(i, after_i) - target.local(i, before_i)
    if after_j != before_j:
        delta += target.local(j, after_j) - target.local(j, before_j)
    return delta


def random_start(target: GraphTarget, rng: np.random.Generator) -> BoolArray:
    """A random valid DAG: required edges, then each pair a random possible state, if valid."""
    size = target.size
    low, high = pair_index(size)
    graph = np.zeros((size, size), dtype=bool)
    chain_free: list[int] = []
    for p in range(low.size):
        i, j = int(low[p]), int(high[p])
        row = (target.log_none[i, j], target.log_edge[i, j], target.log_edge[j, i])
        possible = [s for s in range(3) if row[s] > -math.inf]
        if len(possible) == 1:
            if possible[0] == 1:
                graph[i, j] = True
            elif possible[0] == 2:
                graph[j, i] = True
        else:
            chain_free.append(p)
    builder = _Chain.__new__(_Chain)
    builder.target = target
    builder.size = size
    builder.children = [0] * size
    builder.parents = [0] * size
    for source, child in zip(*np.nonzero(graph), strict=True):
        builder.set_edge(int(source), int(child), True)
    for position in rng.permutation(len(chain_free)):
        p = chain_free[int(position)]
        i, j = int(low[p]), int(high[p])
        row = (target.log_none[i, j], target.log_edge[i, j], target.log_edge[j, i])
        possible = [s for s in range(3) if row[s] > -math.inf]
        state = possible[int(rng.integers(0, len(possible)))]
        source, child = (i, j) if state == 1 else (j, i)
        if state and builder.can_add(source, child):
            builder.set_edge(source, child, True)
            graph[source, child] = True
    return graph


# -- convergence diagnostics -------------------------------------------------------------


def _inverse_normal(p: FloatArray) -> FloatArray:
    """The standard normal quantile function (Acklam's rational approximation, ~1e-9)."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)  # fmt: skip
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)  # fmt: skip
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)  # fmt: skip
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    out = np.empty_like(p)
    lower = p < 0.02425
    upper = p > 1 - 0.02425
    middle = ~(lower | upper)
    q = np.sqrt(-2 * np.log(p[lower]))
    out[lower] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    q = np.sqrt(-2 * np.log(1 - p[upper]))
    out[upper] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    q = p[middle] - 0.5
    r = q * q
    out[middle] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
    return out


def _split(series: FloatArray) -> FloatArray:
    """``(S, chains, draws)`` to ``(S, 2 chains, draws // 2)``: each chain halved."""
    half = series.shape[2] // 2
    first = series[:, :, :half]
    second = series[:, :, series.shape[2] - half :]
    return np.concatenate([first, second], axis=1)


def _rank_normalize(series: FloatArray) -> FloatArray:
    """Average-tie ranks over all draws of each series, mapped to normal scores."""
    flat = series.reshape(series.shape[0], -1)
    total = flat.shape[1]
    out = np.empty_like(flat)
    for row in range(flat.shape[0]):
        _, inverse, counts = np.unique(flat[row], return_inverse=True, return_counts=True)
        upper = np.cumsum(counts)
        ranks = upper - (counts - 1) / 2.0
        out[row] = ranks[inverse]
    scores = _inverse_normal((out - 0.375) / (total + 0.25))
    return scores.reshape(series.shape)


def split_rhat(series: FloatArray) -> FloatArray:
    """Rank-normalized split R-hat of each ``(chains, draws)`` series; NaN when constant."""
    chains = _split(_rank_normalize(series)) if series.shape[2] >= 4 else series
    n = chains.shape[2]
    means = chains.mean(axis=2)
    within = chains.var(axis=2, ddof=1).mean(axis=1)
    between = n * means.var(axis=1, ddof=1)
    pooled = (n - 1) / n * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt(pooled / within)
    constant = series.reshape(series.shape[0], -1).std(axis=1) == 0.0
    result[constant] = np.nan
    out: FloatArray = result
    return out


def effective_size(series: FloatArray, *, rank: bool = True) -> FloatArray:
    """Bulk effective sample size of each ``(chains, draws)`` series; NaN when constant.

    Split chains, Geyer's initial monotone sequence on the multi-chain
    autocorrelation (the Stan estimator). ``rank=False`` skips the rank
    normalization, for the Monte Carlo error of a mean.
    """
    values = _rank_normalize(series) if rank else series.astype(np.float64)
    chains = _split(values) if values.shape[2] >= 4 else values
    count, n = chains.shape[1], chains.shape[2]
    centered = chains - chains.mean(axis=2, keepdims=True)
    size = 1 << math.ceil(math.log2(2 * n))
    spectrum = np.fft.rfft(centered, n=size, axis=2)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=2)[:, :, :n] / n
    chain_var = acov[:, :, 0] * n / (n - 1)
    mean_var = chain_var.mean(axis=1)
    between = chains.mean(axis=2).var(axis=1, ddof=1) if count > 1 else np.zeros(chains.shape[0])
    var_plus = mean_var * (n - 1) / n + between
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1.0 - (mean_var[:, None] - acov.mean(axis=1)) / var_plus[:, None]
    rho[:, 0] = 1.0
    pairs = n // 2
    sums = rho[:, : 2 * pairs : 2] + rho[:, 1 : 2 * pairs : 2]
    positive = np.cumprod(sums > 0.0, axis=1).astype(bool)
    monotone = np.minimum.accumulate(np.where(positive, sums, 0.0), axis=1)
    tau = -1.0 + 2.0 * np.where(positive, monotone, 0.0).sum(axis=1)
    total = count * n
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = total / np.maximum(tau, 1.0 / math.log10(max(total, 10)))
    constant = series.reshape(series.shape[0], -1).std(axis=1) == 0.0
    ess[constant] = np.nan
    out: FloatArray = ess
    return out


def mean_error(indicator: FloatArray) -> tuple[float, float]:
    """The mean of a ``(chains, draws)`` series and its Monte Carlo standard error."""
    mean = float(indicator.mean())
    if indicator.std() == 0.0:
        return mean, 0.0
    ess = float(effective_size(indicator[None], rank=False)[0])
    variance = float(indicator.var())
    return mean, math.sqrt(variance / ess) if ess > 0 else math.inf
