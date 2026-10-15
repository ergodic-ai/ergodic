# SPDX-License-Identifier: Apache-2.0
"""Exact computation over DAGs for targets that factorize by pair.

A pair-factorized target gives each unordered pair ``{i, j}`` three state weights:
``a_ij`` for no edge, ``w_ij`` for ``i -> j`` and ``w_ji`` for ``j -> i``. A DAG's
weight is the product of its pairs' state weights. Everything here is exact in
exact arithmetic and runs over subsets of the variables.

**The normalizer.** Write ``f(V)`` for the summed weight of every DAG on the node
set ``V`` (pairs inside ``V`` only) and ``e_ij = a_ij + w_ij``. Every DAG has at least
one sink, and when every node of a set ``S`` is a sink no edge leaves ``S``: pairs
inside ``S`` carry ``a``, a pair from ``i`` outside ``S`` to ``j`` in ``S`` carries
``a_ij`` or ``w_ij``, and the rest is any DAG on ``V - S``. Inclusion-exclusion over
the sink sets gives Robinson's recursion with pair weights,

``f(V) = sum over nonempty S in V of (-1)^(|S|+1) f(V - S)
prod_{j in S} prod_{i in V - S} e_ij prod_{i < j in S} a_ij``,

with ``f`` of the empty set 1. It visits each pair of disjoint sets once, about
``3^n`` terms, and runs here one popcount level at a time, vectorized, in log
space with signs.

**Required edges.** A pair whose ``i -> j`` is required has ``a_ij = 0`` and
``w_ji = 0``. The recursion needs nothing else: with both ends in ``S`` the term
carries ``a_ij = 0``; with ``i`` outside and ``j`` in ``S`` it carries
``e_ij = w_ij``; with ``j`` outside and ``i`` in ``S`` it carries ``e_ji = 0``.
Those are exactly the configurations in which the edge is or is not possible.
A banned edge is ``w = 0``. Zero factors are tracked as counts, so no
``-inf - -inf`` arises.

**Edge marginals.** ``P(i -> j) = d log Z / d log w_ij``, and ``w_ij`` enters only
through ``e_ij`` on terms with ``i`` outside and ``j`` inside ``S``. Reverse-mode
differentiation of the recursion gives all of them in one more pass of the same
size: with ``r(V, S)`` a term's signed share of ``f(V)`` and
``g(U) = d log Z / d log f(U)`` (``g`` of the full set 1,
``g(U) = sum_V g(V) r(V, V - U)``), ``P(i -> j) = (w_ij / e_ij)
sum_{V, S: i in V - S, j in S} g(V) r(V, S)``.

**Exact sampling.** The sink layering of a DAG (its sinks, then the sinks of
what is left, and so on) is unique. The weight of DAGs on ``R`` whose sinks are
exactly ``S`` is the Moebius transform of the inclusion-exclusion terms,
``h(R, S) = sum_{T >= S} (-1)^(|T - S|) A(R, T)``. Given that the layer before
was ``P``, every node of the next layer needs an edge into ``P`` (it was not a
sink before), and a node that is not in the next layer is free, so the next layer
``S`` has probability proportional to ``h(R, S) prod_{j in S} (1 - prod_{t in P}
a_jt / e_jt)``. Edges from a layer into ``P`` are drawn conditioned on at least
one, edges into earlier layers freely.

**MAP.** The best DAG consistent with an order takes, for each pair, its better
state among "none" and the forward edge, so the MAP is a maximum over orders: a
dynamic program over subsets of ``n 2^n`` steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]
    BoolArray = npt.NDArray[np.bool_]

_CHUNK = 1 << 20  # elements per vectorized block
_CACHE_LIMIT = 1 << 22  # cached exact-sink weights kept by the sampler
LOST_PRECISION = "the exact computation lost precision (cancellation in the signed recursion)"


class ExactPrecisionError(ArithmeticError):
    """The signed recursion cancelled below rounding; use MCMC instead."""


# -- small subset helpers ------------------------------------------------------------


def _subset_sums(values: FloatArray) -> FloatArray:
    """``out[mask] = sum of values[i] for bit i in mask``, for every mask."""
    out = np.zeros(1, dtype=np.float64)
    for value in values:
        out = np.concatenate([out, out + value])
    return out


def _subset_counts(values: IntArray) -> IntArray:
    out = np.zeros(1, dtype=np.int64)
    for value in values:
        out = np.concatenate([out, out + value])
    return out


@lru_cache(maxsize=32)
def _binary(width: int) -> IntArray:
    """``(width, 2^width)`` 0/1 matrix: entry ``[l, t]`` is bit ``l`` of ``t``."""
    columns = np.arange(1 << width, dtype=np.int64)
    result: IntArray = (columns[None, :] >> np.arange(width, dtype=np.int64)[:, None]) & 1
    result.setflags(write=False)
    return result


@lru_cache(maxsize=32)
def _popcount(width: int) -> IntArray:
    result: IntArray = _binary(width).sum(axis=0)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=32)
def _levels(size: int) -> tuple[IntArray, ...]:
    """Masks over ``size`` bits grouped by popcount, each in increasing order."""
    masks = np.arange(1 << size, dtype=np.int64)
    counts = _popcount(size)
    return tuple(masks[counts == k] for k in range(size + 1))


def _bits_of(masks: IntArray, size: int, width: int) -> IntArray:
    """For masks of popcount ``width``, the node indices of their bits, increasing."""
    if width == 0:
        return np.zeros((masks.size, 0), dtype=np.int64)
    has = ((masks[:, None] >> np.arange(size, dtype=np.int64)[None, :]) & 1).astype(bool)
    nodes = np.broadcast_to(np.arange(size, dtype=np.int64), has.shape)
    result: IntArray = nodes[has].reshape(masks.size, width)
    return result


def _signed_logsumexp(terms: FloatArray, signs: FloatArray) -> FloatArray:
    """Row-wise ``log sum sign * exp(term)``; ``-inf`` rows when every term is ``-inf``."""
    top = np.max(terms, axis=1)
    finite = np.isfinite(top)
    safe = np.where(finite, top, 0.0)
    with np.errstate(invalid="ignore", over="ignore"):
        total = np.sum(signs * np.exp(terms - safe[:, None]), axis=1)
    out = np.full(terms.shape[0], -np.inf)
    good = finite & (total > 0.0)
    out[good] = safe[good] + np.log(total[good])
    if np.any(finite & (total <= 0.0)):
        raise ExactPrecisionError(LOST_PRECISION)
    return out


# -- the pair weights -----------------------------------------------------------------


@dataclass(frozen=True)
class PairWeights:
    """State weights per pair, each pair scaled so its largest state weighs 1.

    ``log_none`` is symmetric, ``log_edge[i, j]`` is the log weight of ``i -> j``;
    ``-inf`` marks an impossible state. ``offset`` is the summed log scale, so a
    DAG's log target is ``offset`` plus its pairs' scaled log weights.
    """

    log_none: FloatArray
    log_edge: FloatArray
    offset: float

    @classmethod
    def from_logs(cls, log_none: FloatArray, log_edge: FloatArray) -> PairWeights:
        size = log_none.shape[0]
        scale = np.maximum(np.maximum(log_none, log_edge), log_edge.T)
        np.fill_diagonal(scale, 0.0)
        upper = np.triu_indices(size, k=1)
        if not np.all(np.isfinite(scale[upper])):
            raise ValueError("a pair has no possible state")
        none = log_none - scale
        edge = log_edge - scale
        np.fill_diagonal(none, -np.inf)
        np.fill_diagonal(edge, -np.inf)
        return cls(none, edge, float(scale[upper].sum()))

    @property
    def size(self) -> int:
        return int(self.log_none.shape[0])

    @property
    def none(self) -> FloatArray:
        out: FloatArray = np.exp(self.log_none)
        return out

    @property
    def edge(self) -> FloatArray:
        out: FloatArray = np.exp(self.log_edge)
        return out

    def log_weight(self, graph: BoolArray) -> float:
        """The DAG's log target: ``offset`` plus the scaled state weights."""
        size = self.size
        upper = np.triu_indices(size, k=1)
        forward = graph[upper]
        reverse = graph.T[upper]
        states = np.where(
            forward,
            self.log_edge[upper],
            np.where(reverse, self.log_edge.T[upper], self.log_none[upper]),
        )
        return float(self.offset + states.sum())


# -- the exact engine ------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetDP:
    """The subset recursion for one set of pair weights: normalizer, marginals, samples, MAP."""

    weights: PairWeights
    log_f: FloatArray = field(repr=False)  # log f(U) for every mask U
    _lf: FloatArray = field(repr=False)  # finite part of log e_ij
    _li: IntArray = field(repr=False)  # 1 where e_ij == 0
    _c: FloatArray = field(repr=False)  # per-mask constant: log a inside S minus log e inside S
    _ci: IntArray = field(repr=False)

    @classmethod
    def build(cls, weights: PairWeights) -> SubsetDP:
        size = weights.size
        with np.errstate(divide="ignore"):
            log_e = np.logaddexp(weights.log_none, weights.log_edge)
        np.fill_diagonal(log_e, 0.0)
        impossible = ~np.isfinite(log_e)
        lf = np.where(impossible, 0.0, log_e)
        li = impossible.astype(np.int64)
        np.fill_diagonal(lf, 0.0)
        np.fill_diagonal(li, 0)
        none_impossible = ~np.isfinite(weights.log_none)
        log_a = np.where(none_impossible, 0.0, weights.log_none)
        np.fill_diagonal(log_a, 0.0)
        a_zero = none_impossible.astype(np.int64)
        np.fill_diagonal(a_zero, 0)
        # c(S) = sum_{i<j in S} log a_ij - sum_{i != j in S} log e_ij (finite parts), built
        # by adding the highest node b to every mask over the nodes below it.
        c = np.zeros(1, dtype=np.float64)
        ci = np.zeros(1, dtype=np.int64)
        for b in range(size):
            step = log_a[:b, b] - lf[:b, b] - lf[b, :b]
            count = a_zero[:b, b] - li[:b, b] - li[b, :b]
            c = np.concatenate([c, c + _subset_sums(step)])
            ci = np.concatenate([ci, ci + _subset_counts(count)])
        log_f = np.full(1 << size, -np.inf)
        log_f[0] = 0.0
        dp = cls(weights, log_f, lf, li, c, ci)
        for width in range(1, size + 1):
            masks = _levels(size)[width]
            for block in _blocks(masks, width):
                terms, signs, _, _ = dp._terms(block, width)
                log_f[block] = _signed_logsumexp(terms, signs)
        log_f.setflags(write=False)
        return dp

    @property
    def size(self) -> int:
        return self.weights.size

    @property
    def log_z(self) -> float:
        """The log normalizer: ``offset`` plus ``log f`` of the full set."""
        return float(self.weights.offset + self.log_f[-1])

    def _terms(
        self, masks: IntArray, width: int
    ) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
        """The signed log terms of ``f(V)`` for each ``V`` in ``masks`` (popcount ``width``).

        Returns ``(terms, signs, sinks, bits)``: ``terms[v, t]`` for the sink set
        ``sinks[v, t]`` (submask ``t`` of ``V``'s bits, ``t = 0`` the empty set and
        ``-inf``), and ``bits[v]`` the nodes of ``V``.
        """
        size = self.size
        binary = _binary(width).astype(np.float64)
        bits = _bits_of(masks, size, width)
        sinks = np.rint(np.left_shift(1, bits).astype(np.float64) @ binary).astype(np.int64)
        rest = masks[:, None] - sinks
        member = ((masks[:, None] >> np.arange(size, dtype=np.int64)[None, :]) & 1).astype(
            np.float64
        )
        incoming = np.take_along_axis(member @ self._lf, bits, axis=1)  # sum_{i in V} lf[i, j]
        blocked = np.take_along_axis(member @ self._li.astype(np.float64), bits, axis=1)
        terms = self.log_f[rest] + incoming @ binary + self._c[sinks]
        zeros = np.rint(blocked @ binary).astype(np.int64) + self._ci[sinks]
        terms[zeros > 0] = -np.inf
        terms[:, 0] = -np.inf
        signs = np.where(_popcount(width) % 2 == 1, 1.0, -1.0)
        return terms, np.broadcast_to(signs, terms.shape), sinks, bits

    # -- edge marginals ------------------------------------------------------------------

    def edge_marginals(self) -> FloatArray:
        """``P(i -> j)`` for every ordered pair, exactly, by one reverse pass."""
        size = self.size
        if size == 1:
            return np.zeros((1, 1))
        adjoint = np.zeros(1 << size, dtype=np.float64)
        adjoint[-1] = 1.0
        flow = np.zeros(size * size, dtype=np.float64)
        for width in range(size, 0, -1):
            masks = _levels(size)[width]
            binary = _binary(width).astype(np.float64)
            for block in _blocks(masks, width):
                terms, signs, sinks, bits = self._terms(block, width)
                base = self.log_f[block]
                share = np.where(
                    np.isfinite(base)[:, None],
                    signs * np.exp(terms - np.where(np.isfinite(base), base, 0.0)[:, None]),
                    0.0,
                )
                contribution = adjoint[block][:, None] * share
                rest = block[:, None] - sinks
                adjoint += np.bincount(
                    rest.ravel(), weights=contribution.ravel(), minlength=1 << size
                )
                # sum_S c [b in S], and sum_S c [a in S][b in S], per V.
                into = contribution @ binary.T
                both = np.einsum("vs,as,bs->vab", contribution, binary, binary, optimize=True)
                cross = into[:, None, :] - both  # [v, a, b]: a in V - S, b in S
                index = bits[:, :, None] * size + bits[:, None, :]
                flow += np.bincount(index.ravel(), weights=cross.ravel(), minlength=size * size)
        matrix = flow.reshape(size, size)
        np.fill_diagonal(matrix, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_e = np.logaddexp(self.weights.log_none, self.weights.log_edge)
            ratio = np.where(np.isfinite(log_e), np.exp(self.weights.log_edge - log_e), 0.0)
        result: FloatArray = np.clip(ratio * matrix, 0.0, 1.0)
        np.fill_diagonal(result, 0.0)
        return result

    # -- exact sampling --------------------------------------------------------------------

    def sampler(self) -> ExactSampler:
        """An exact sampler (sink layer by sink layer) over this target's DAGs."""
        return ExactSampler(self)

    # -- MAP -----------------------------------------------------------------------------------

    def map_graph(self) -> BoolArray:
        """The DAG with the largest weight; ties go to "no edge" and to lower node indices."""
        return map_graph(self.weights)


def _blocks(masks: IntArray, width: int) -> list[IntArray]:
    rows = max(1, _CHUNK // max(1, (1 << width) * max(1, width)))
    return [masks[start : start + rows] for start in range(0, masks.size, rows)]


class ExactSampler:
    """Exact DAG draws from a :class:`SubsetDP`, one sink layer at a time."""

    def __init__(self, dp: SubsetDP) -> None:
        self._dp = dp
        self._cache: dict[int, tuple[IntArray, IntArray, FloatArray]] = {}
        self._cached = 0
        weights = dp.weights
        with np.errstate(divide="ignore", invalid="ignore"):
            log_e = np.logaddexp(weights.log_none, weights.log_edge)
            self._p_edge = np.where(np.isfinite(log_e), np.exp(weights.log_edge - log_e), 0.0)
            self._log_rho = np.where(np.isfinite(log_e), weights.log_none - log_e, 0.0)

    def _exact_sinks(self, remaining: int) -> tuple[IntArray, IntArray, FloatArray]:
        """``(bits, sink masks, h)``: the exact-sink-set weights of DAGs on ``remaining``."""
        hit = self._cache.get(remaining)
        if hit is not None:
            return hit
        dp = self._dp
        width = bin(remaining).count("1")
        masks = np.asarray([remaining], dtype=np.int64)
        terms, _, sinks, bits = dp._terms(masks, width)
        log_terms = terms[0].copy()
        log_terms[0] = dp.log_f[remaining]  # A(empty set) = every DAG on R
        top = float(np.max(log_terms))
        values = np.exp(log_terms - top)
        # Moebius over supersets: h(S) = sum_{T >= S} (-1)^{|T - S|} A(T).
        cube = values.reshape((2,) * width) if width else values
        for axis in range(width):
            low: list[int | slice] = [slice(None)] * width
            high: list[int | slice] = [slice(None)] * width
            low[axis] = 0
            high[axis] = 1
            cube[tuple(low)] -= cube[tuple(high)]
        # Every bit gets the same transform, so the flat index stays t.
        values = np.clip(cube.reshape(-1), 0.0, None)
        values[0] = 0.0
        result = (bits[0], sinks[0], values)
        self._cached += values.size
        if self._cached > _CACHE_LIMIT:
            self._cache.clear()
            self._cached = values.size
        self._cache[remaining] = result
        return result

    def draw(self, rng: np.random.Generator) -> BoolArray:
        """One exact draw, as a boolean adjacency matrix."""
        size = self._dp.size
        graph = np.zeros((size, size), dtype=bool)
        remaining = (1 << size) - 1
        earlier_mask = np.zeros(size, dtype=bool)  # removed before the previous layer
        previous = np.zeros(0, dtype=np.int64)
        while remaining:
            bits, sinks, weights = self._exact_sinks(remaining)
            if previous.size:
                log_keep = self._log_rho[np.ix_(bits, previous)].sum(axis=1)
                with np.errstate(divide="ignore"):
                    tilt = np.log1p(-np.exp(np.minimum(log_keep, 0.0)))
                tilt = np.where(log_keep >= 0.0, -np.inf, tilt)
                with np.errstate(over="ignore"):
                    weights = weights * np.exp(_subset_sums(tilt))
            cumulative = np.cumsum(weights)
            total = float(cumulative[-1])
            if not total > 0.0:  # pragma: no cover - h is positive on some layer
                raise ExactPrecisionError(LOST_PRECISION)
            pick = min(
                int(np.searchsorted(cumulative, rng.random() * total, side="right")),
                weights.size - 1,
            )
            while weights[pick] <= 0.0:  # pragma: no cover - a zero-width bin at the boundary
                pick -= 1
            layer = bits[((pick >> np.arange(bits.size)) & 1) == 1]
            earlier = np.flatnonzero(earlier_mask)
            if earlier.size:
                chances = self._p_edge[np.ix_(layer, earlier)]
                graph[np.ix_(layer, earlier)] = rng.random(chances.shape) < chances
            if previous.size:
                self._at_least_one(graph, layer, previous, rng)
            earlier_mask[previous] = True
            previous = layer
            remaining &= ~int(sinks[pick])
        return graph

    def _at_least_one(
        self,
        graph: BoolArray,
        layer: IntArray,
        targets: IntArray,
        rng: np.random.Generator,
    ) -> None:
        """Edges from each ``layer`` node into ``targets``, independently, at least one each.

        Target by target: while no edge is drawn yet, the next one has chance
        ``p / (1 - P(no edge from here on))``; after one, the rest are free.
        """
        chances = self._p_edge[np.ix_(layer, targets)]
        # none_from[:, k] = prod_{l >= k} (1 - p_l)
        none_from = np.cumprod((1.0 - chances)[:, ::-1], axis=1)[:, ::-1]
        needed = np.ones(layer.size, dtype=bool)
        draws = rng.random(chances.shape)
        for position in range(targets.size):
            p = chances[:, position]
            some = 1.0 - none_from[:, position]
            with np.errstate(divide="ignore", invalid="ignore"):
                conditional = np.where(some > 0.0, np.minimum(1.0, p / some), 1.0)
            chance = np.where(needed, conditional, p)
            hit = draws[:, position] < chance
            graph[layer, targets[position]] = hit
            needed &= ~hit


def map_graph(weights: PairWeights) -> BoolArray:
    """The maximum-weight DAG of a pair-factorized target, exactly, by DP over orders."""
    size = weights.size
    best_pair = np.maximum(weights.log_none, weights.log_edge)  # i before j: none or i -> j
    np.fill_diagonal(best_pair, 0.0)
    gain = np.full(1 << size, -np.inf)
    gain[0] = 0.0
    last = np.zeros(1 << size, dtype=np.int64)
    # into[j][U] = sum_{i in U} best_pair[i, j]
    into = np.stack([_subset_sums(best_pair[:, j]) for j in range(size)])
    for width in range(1, size + 1):
        masks = _levels(size)[width]
        bits = _bits_of(masks, size, width)
        before = masks[:, None] - np.left_shift(1, bits)
        candidates = gain[before] + into[bits, before]
        choice = np.argmax(candidates, axis=1)
        gain[masks] = candidates[np.arange(masks.size), choice]
        last[masks] = bits[np.arange(masks.size), choice]
    order: list[int] = []
    mask = (1 << size) - 1
    while mask:
        node = int(last[mask])
        order.append(node)
        mask &= ~(1 << node)
    order.reverse()
    graph = np.zeros((size, size), dtype=bool)
    for position, target in enumerate(order):
        for source in order[:position]:
            if weights.log_edge[source, target] > weights.log_none[source, target]:
                graph[source, target] = True
    return graph


# -- enumeration ------------------------------------------------------------------------


@lru_cache(maxsize=8)
def all_dags(size: int) -> BoolArray:
    """Every DAG on ``size`` labelled nodes, as a ``(count, size, size)`` boolean array."""
    if size > 5:
        raise ValueError("enumeration is limited to 5 variables (29,281 DAGs)")
    upper = list(zip(*np.triu_indices(size, k=1), strict=True))
    count = len(upper)
    states = np.indices((3,) * count).reshape(count, -1).T if count else np.zeros((1, 0), int)
    graphs = np.zeros((states.shape[0], size, size), dtype=bool)
    for column, (i, j) in enumerate(upper):
        graphs[:, i, j] = states[:, column] == 1
        graphs[:, j, i] = states[:, column] == 2
    keep = acyclic(graphs)
    result: BoolArray = graphs[keep]
    result.setflags(write=False)
    return result


def acyclic(graphs: BoolArray) -> BoolArray:
    """Which of a stack of adjacency matrices are acyclic (repeatedly strip the sources)."""
    alive = np.ones(graphs.shape[:2], dtype=bool)
    for _ in range(graphs.shape[1]):
        indegree = np.einsum("kij,ki->kj", graphs.astype(np.int64), alive.astype(np.int64))
        sources = alive & (indegree == 0)
        if not sources.any():
            break
        alive &= ~sources
    result: BoolArray = np.logical_not(alive.any(axis=1))
    return result


def reachability(graphs: BoolArray) -> BoolArray:
    """Transitive closure of a stack of adjacency matrices: ``[k, i, j]``, ``i`` reaches ``j``."""
    reach = graphs.copy()
    for middle in range(graphs.shape[1]):
        reach |= reach[:, :, middle, None] & reach[:, None, middle, :]
    return reach
