# SPDX-License-Identifier: Apache-2.0
"""Explicit, versioned decoders from edge probabilities to a DAG.

An :class:`~ergodic.discovery.EdgeProbabilities` matrix is not a graph: both
directions of a pair can look likely, and the likely edges can close a cycle.
A decoder is the modelling choice that turns the matrix into one DAG (or a set
of candidate DAGs). Every decoder here reads the pair-state view (three
exclusive states per unordered pair: no edge, ``i -> j``, ``j -> i``) and
scores a directed edge by ``w(i -> j) = log P(i -> j) - log P(no edge)``, both
floored at ``1e-12``. A DAG's factorized pair score is the sum of the log
probabilities of the states it picks.

The decoders are ported from Ergodic's research library (``ergodic-ai/welt``,
WELT-292 / EXP-0386 and EXP-0626, where greedy pair-MAP matched exact MAP on 238
of 240 graphs). Each id carries a version, and a decoded graph always says it
was decoded, by which decoder, with which parameters and seed. It is never a
graph posterior, and its orientations are not certified by anything.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..graph import dag
from ._probabilities import EdgeProbabilities
from ._result import DiscoveryResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ..graph import MixedGraph
    from ..knowledge import DomainKnowledge

    FloatArray = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

PROBABILITY_FLOOR: Final = 1e-12
DECODED_NOTE: Final = "not a graph posterior; orientation not certified"
DEFAULT_MAX_NODES: Final = 20


@dataclass(frozen=True)
class _Spec:
    port_of: str
    stochastic: bool
    order_search: bool
    summary: str


DECODERS: Final[Mapping[str, _Spec]] = {
    "greedy_v1": _Spec(
        "greedy_pair_map_v1",
        False,
        False,
        "greedy pair-MAP: best-scoring edges first, skip any that closes a cycle",
    ),
    "exact_map_v1": _Spec(
        "exact_factorized_map_astar_v1",
        False,
        True,
        "exact MAP of the factorized pair score, A* over topological orders",
    ),
    "expected_count_v1": _Spec(
        "expected_count_cardinality_v1",
        False,
        True,
        "the MAP order's best K edges, K the rounded expected edge count",
    ),
    "soft_f1_v1": _Spec(
        "soft_f1_cardinality_v1",
        False,
        True,
        "the MAP order's best K edges, K from a label-free soft-F1 proxy",
    ),
    "pair_sampler_v1": _Spec(
        "independent_pair_cycle_rejection_v1",
        True,
        False,
        "sample each pair's state, add edges in random order, drop cycle-closing ones",
    ),
    "perturb_and_map_v1": _Spec(
        "gumbel_perturb_and_map_v1",
        True,
        True,
        "Gumbel-perturb the pair states, then solve exact MAP per sample",
    ),
}
"""The registered decoders by versioned id."""

_ALIASES: Final = {name.rsplit("_v", 1)[0]: name for name in DECODERS}


def resolve_decoder(method: str) -> str:
    """The versioned decoder id for ``method`` (``"greedy"`` reads as ``"greedy_v1"``)."""
    if not isinstance(method, str):
        raise TypeError(f"method must be a decoder id string, got {type(method).__name__}")
    key = method.lower()
    if key in DECODERS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    names = ", ".join(sorted(DECODERS))
    raise ValueError(f"unknown decoder {method!r}; use one of: {names} (or drop the _v1)")


@dataclass(frozen=True)
class DecodeResult(DiscoveryResult):
    """A DAG decoded from edge probabilities, with how it was decoded.

    A :class:`DiscoveryResult` whose ``graph`` is the decoded DAG (``kind``
    ``dag``): the single decoded graph for a deterministic decoder, the consensus
    (medoid) of the candidates for a sampler. ``method`` is
    ``"decode:<decoder id>"`` (a Welt run decoded locally keeps ``"welt:<model>"``).

    ``diagnostics`` always labels the graph: ``decoded`` is ``True``, ``native``
    is ``False``, and it records ``decoder``, ``parameters``, ``seed``,
    ``samples``, ``rejected_cycles`` and ``search_expansions`` (one per
    candidate), ``edge_budget``, ``objective`` (the factorized pair log score of
    ``graph``), ``n_edges``, ``source_kind``, ``source_provenance``,
    ``knowledge_applied`` and ``note`` ("not a graph posterior; orientation not
    certified").

    Attributes:
        decoder: The versioned decoder id, for example ``"greedy_v1"``.
        candidates: Every decoded DAG; one for a deterministic decoder, one per
            sample (in sample order) for a sampler. Candidates, not posterior draws.
        probabilities: The edge probabilities that were decoded, as given.
    """

    decoder: str = ""
    candidates: tuple[MixedGraph, ...] = field(default=(), repr=False)
    probabilities: EdgeProbabilities | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        body = f"{self.method}: decoded {self.graph.kind.value}"
        body += f", {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"
        if len(self.candidates) > 1:
            body += f", consensus of {len(self.candidates)} candidates"
        return f"DecodeResult({body})"


# -- the decoder configuration -----------------------------------------------------


@dataclass(frozen=True)
class Decoder:
    """A decoder and its settings, checked up front and reusable.

    ``Decoder("greedy", threshold=0.3).decode(probabilities)`` is
    ``decode(probabilities, "greedy", threshold=0.3)``. Pass one to
    :func:`ergodic.welt.discover` as ``decoder=`` to fix every setting before a
    paid run.

    Attributes:
        method: A decoder id (see :func:`decode`).
        threshold: Greedy only: nominate an edge only when its input value is
            above this (``None`` keeps edges with positive weight).
        samples: Candidate count for a sampler; must be 1 otherwise.
        seed: Sampler seed (required for a sampler, refused otherwise).
        max_nodes: The variable cap for decoders that search orders exactly.
    """

    method: str = "greedy"
    threshold: float | None = None
    samples: int = 1
    seed: int | None = None
    max_nodes: int = DEFAULT_MAX_NODES

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", resolve_decoder(self.method))
        _check_settings(self.method, self.threshold, self.samples, self.seed, self.max_nodes)

    @property
    def stochastic(self) -> bool:
        """Whether this decoder samples candidates (and so needs a seed)."""
        return DECODERS[self.method].stochastic

    def decode(
        self, probabilities: EdgeProbabilities, *, knowledge: DomainKnowledge | None = None
    ) -> DecodeResult:
        """Decode ``probabilities`` with these settings; see :func:`decode`."""
        return decode(
            probabilities,
            self.method,
            threshold=self.threshold,
            knowledge=knowledge,
            samples=self.samples,
            seed=self.seed,
            max_nodes=self.max_nodes,
        )


def _check_settings(
    decoder: str, threshold: float | None, samples: int, seed: int | None, max_nodes: int
) -> None:
    spec = DECODERS[decoder]
    if threshold is not None:
        if decoder != "greedy_v1":
            raise ValueError(f"threshold applies to the greedy decoder only, not {decoder!r}")
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise TypeError("threshold must be a number or None")
        if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {threshold!r}")
    if type(samples) is not int or samples < 1:
        raise ValueError(f"samples must be a positive int, got {samples!r}")
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError(f"max_nodes must be a positive int, got {max_nodes!r}")
    if spec.stochastic:
        if seed is None:
            raise ValueError(
                f"{decoder!r} samples candidate DAGs, so it needs an explicit integer seed"
            )
        if type(seed) is not int or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    else:
        if samples != 1:
            raise ValueError(f"{decoder!r} is deterministic, so samples must be 1")
        if seed is not None:
            raise ValueError(f"{decoder!r} is deterministic and takes no seed")


# -- the facade ----------------------------------------------------------------------


def decode(
    probabilities: EdgeProbabilities,
    method: str = "greedy",
    *,
    threshold: float | None = None,
    knowledge: DomainKnowledge | None = None,
    samples: int = 1,
    seed: int | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> DecodeResult:
    """Decode edge probabilities into a DAG with a named, versioned decoder.

    The input is read as pair states (:meth:`EdgeProbabilities.to_pair_states`)
    and every decoder works on the edge weight
    ``w(i -> j) = log P(i -> j) - log P(no edge)``. The decoders:

    - ``"greedy"`` (``greedy_v1``, the default): for each pair nominate its
      better direction when ``w > 0``, or, with ``threshold=t``, when the input
      value of that direction is above ``t``; add nominations by decreasing
      weight and skip any that would close a cycle. On independent marginals,
      ``w > 0`` is ``p > 0.5``, AVICI's native rule; ``threshold`` lets a model's
      own cut (CDFM's adaptive one) choose the edges instead.
    - ``"exact_map"`` (``exact_map_v1``): the DAG maximizing the factorized pair
      score, solved exactly by A* over topological orders.
    - ``"expected_count"`` / ``"soft_f1"`` (``expected_count_v1``,
      ``soft_f1_v1``): the best ``K`` edges compatible with the exact MAP
      order, ``K`` the rounded expected edge count or the size that maximizes a
      label-free soft-F1 proxy. They may keep edges a MAP decoder drops.
    - ``"pair_sampler"`` / ``"perturb_and_map"`` (``pair_sampler_v1``,
      ``perturb_and_map_v1``): ``samples`` candidate DAGs, from independent
      pair-state draws with random-order cycle rejection, or from exact MAP under
      Gumbel-perturbed pair states. ``graph`` is their medoid (the candidate with
      the smallest total pair-state distance to the others, earliest on a tie).
      Candidates are not posterior samples.

    Ties are broken by variable order, so a decoder is deterministic given its
    inputs (and seed). The input is never changed and the global numpy random
    state is never touched.

    Knowledge constrains the decode. Forbidden edges, tiers, roots, sinks and a
    skeleton restriction remove directed edges (their weight becomes minus
    infinity). Precedence (``order``, tiers) is enforced on ancestry, not just
    on edges. Required edges go in first, whatever their weight, and must not
    form a cycle. ``confound`` is refused: a DAG cannot hold a latent
    confounder. The decoded graph is checked against the knowledge afterwards.

    Args:
        probabilities: The edge probabilities to decode.
        method: A decoder id, with or without its ``_v1`` version.
        threshold: Greedy only; see above. ``None`` keeps the ``w > 0`` rule.
        knowledge: Hard constraints, all naming variables of ``probabilities``.
        samples: How many candidates a sampler draws; 1 for the others.
        seed: The sampler's seed (required); deterministic decoders refuse one.
        max_nodes: The variable cap for the decoders that search orders exactly
            (all but greedy and pair_sampler). The search can grow exponentially
            with the variable count, so above the cap they raise.

    Returns:
        A :class:`DecodeResult`: ``graph`` a DAG over every variable, with the
        decoder, its parameters and the diagnostics recorded.

    Raises:
        TypeError: ``probabilities`` is not an :class:`EdgeProbabilities`.
        ValueError: An unknown decoder, a setting that does not apply, too many
            variables for an exact search, knowledge naming unknown variables,
            confounded pairs, inconsistent knowledge or a cyclic required set.
    """
    if not isinstance(probabilities, EdgeProbabilities):
        raise TypeError(
            "decode reads an EdgeProbabilities; build one with "
            "EdgeProbabilities(variables, values, kind=...) or EdgeProbabilities.from_frame"
        )
    decoder = resolve_decoder(method)
    _check_settings(decoder, threshold, samples, seed, max_nodes)
    spec = DECODERS[decoder]
    names = probabilities.variables
    size = len(names)
    if spec.order_search and size > max_nodes:
        raise ValueError(
            f"{decoder!r} searches topological orders exactly and refuses {size} variables "
            f"above max_nodes={max_nodes}; use 'greedy' or 'pair_sampler', or raise max_nodes"
        )
    constraints = _Constraints.build(names, knowledge)
    pair = probabilities.to_pair_states().values
    weights = _relative_log_weights(pair, PROBABILITY_FLOOR)
    weights[~constraints.permitted] = -np.inf
    parameters: dict[str, Any] = {"probability_floor": PROBABILITY_FLOOR}
    edge_budget: int | None = None
    if decoder == "greedy_v1":
        parameters["threshold"] = threshold
        graph, rejected = _greedy(weights, probabilities.values, threshold, constraints)
        graphs, rejections, expansions = [graph], [rejected], [0]
    elif decoder == "pair_sampler_v1":
        graphs, rejections = _pair_sampler(pair, samples, seed or 0, constraints)
        expansions = [0] * samples
    elif decoder == "perturb_and_map_v1":
        parameters["max_nodes"] = max_nodes
        graphs, expansions = _perturb_and_map(pair, samples, seed or 0, constraints)
        rejections = [0] * samples
    else:
        parameters["max_nodes"] = max_nodes
        order, expanded = _maximum_weight_order(_order_weights(weights, constraints), constraints)
        if decoder == "exact_map_v1":
            graph = _graph_for_order(weights, order, constraints)
        else:
            rule = "expected_count" if decoder == "expected_count_v1" else "soft_f1"
            edge_budget = (
                _expected_edge_count(pair) if rule == "expected_count" else _soft_f1_budget(pair)
            )
            parameters["edge_budget_rule"] = rule
            graph = _cardinality_graph(weights, order, edge_budget, constraints)
        graphs, rejections, expansions = [graph], [0], [expanded]
    chosen = graphs[_medoid(graphs)] if len(graphs) > 1 else graphs[0]
    candidates = tuple(_to_dag(names, g) for g in graphs)
    result_graph = _to_dag(names, chosen)
    if knowledge is not None:
        broken = knowledge.violations(result_graph)
        if broken:  # pragma: no cover - the decoders enforce every constraint they accept
            raise RuntimeError(f"decoded graph breaks the knowledge: {broken}")
    diagnostics: dict[str, Any] = {
        "decoded": True,
        "native": False,
        "decoder": decoder,
        "decoder_port_of": spec.port_of,
        "parameters": parameters,
        "seed": seed,
        "samples": samples,
        "rejected_cycles": tuple(rejections),
        "search_expansions": tuple(expansions),
        "edge_budget": edge_budget,
        "objective": _pair_log_score(chosen, pair, PROBABILITY_FLOOR),
        "n_edges": int(chosen.sum()),
        "source_kind": probabilities.kind,
        "source_calibrated": probabilities.calibrated,
        "source_provenance": probabilities.provenance,
        "knowledge_applied": knowledge is not None,
        "note": DECODED_NOTE,
    }
    return DecodeResult(
        graph=result_graph,
        method=f"decode:{decoder}",
        diagnostics=diagnostics,
        decoder=decoder,
        candidates=candidates,
        probabilities=probabilities,
    )


# -- knowledge -------------------------------------------------------------------------


def check_knowledge(names: Sequence[str], knowledge: DomainKnowledge) -> None:
    """Raise as :func:`decode` would on knowledge it cannot apply to ``names``.

    Lets a caller (the Welt bridge) refuse unusable knowledge before a paid run.
    """
    _Constraints.build(tuple(names), knowledge)


@dataclass(frozen=True)
class _Constraints:
    """Knowledge as index arrays: permitted cells, required edges, precedence."""

    permitted: BoolArray
    required: tuple[tuple[int, int], ...]
    precedence: tuple[tuple[int, int], ...]
    must_precede: tuple[int, ...]  # bitmask of nodes that must come earlier in the order

    @property
    def required_pairs(self) -> frozenset[frozenset[int]]:
        return frozenset(frozenset(edge) for edge in self.required)

    @classmethod
    def build(cls, names: Sequence[str], knowledge: DomainKnowledge | None) -> _Constraints:
        size = len(names)
        permitted = ~np.eye(size, dtype=bool)
        if knowledge is None:
            return cls(permitted, (), (), (0,) * size)
        unknown = sorted(knowledge.variables - set(names))
        if unknown:
            raise ValueError(f"knowledge names variables the probabilities lack: {unknown}")
        if knowledge.confounded:
            raise ValueError(
                "decoding returns a DAG, which cannot hold a latent confounder; "
                "drop the confound() constraints to decode"
            )
        knowledge.validate()
        mask, _ = knowledge.edge_mask(list(names))
        index = {name: i for i, name in enumerate(names)}
        required = tuple(sorted((index[u], index[v]) for u, v in knowledge.required))
        precedence = tuple(
            sorted((index[a], index[b]) for a, b in knowledge.precedence() if a != b)
        )
        must = [0] * size
        for source, target in (*required, *precedence):
            must[target] |= 1 << source
        constraints = cls(np.asarray(mask, dtype=bool), required, precedence, tuple(must))
        base = constraints.base_graph(size)
        if not _is_acyclic(base):
            raise ValueError(
                "the required edges (with the precedence order) form a cycle, so no DAG "
                "satisfies the knowledge"
            )
        return constraints

    def base_graph(self, size: int) -> BoolArray:
        """Required edges plus precedence as virtual edges, for cycle checks."""
        graph = np.zeros((size, size), dtype=bool)
        for source, target in (*self.required, *self.precedence):
            graph[source, target] = True
        return graph


# -- shared arithmetic ------------------------------------------------------------------


def _relative_log_weights(pair: FloatArray, floor: float) -> FloatArray:
    none = np.maximum(0.0, 1.0 - pair - pair.T)
    result: FloatArray = np.log(np.maximum(pair, floor)) - np.log(np.maximum(none, floor))
    np.fill_diagonal(result, 0.0)
    return result


def _pair_log_score(graph: BoolArray, pair: FloatArray, floor: float) -> float:
    """The factorized pair score: the summed log probability of each pair's state."""
    total = 0.0
    size = graph.shape[0]
    for i in range(size):
        for j in range(i + 1, size):
            if graph[i, j]:
                state = pair[i, j]
            elif graph[j, i]:
                state = pair[j, i]
            else:
                state = 1.0 - pair[i, j] - pair[j, i]
            total += math.log(max(floor, float(state)))
    return total


def _reaches(graph: BoolArray, start: int, goal: int) -> bool:
    """Whether a directed path leads from ``start`` to ``goal``."""
    if start == goal:
        return True
    seen = np.zeros(graph.shape[0], dtype=bool)
    seen[start] = True
    stack = [start]
    while stack:
        node = stack.pop()
        for child in np.flatnonzero(graph[node] & ~seen):
            if child == goal:
                return True
            seen[child] = True
            stack.append(int(child))
    return False


def _is_acyclic(graph: BoolArray) -> bool:
    indegree = graph.sum(axis=0)
    ready = [int(i) for i in np.flatnonzero(indegree == 0)]
    seen = 0
    indegree = indegree.copy()
    while ready:
        node = ready.pop()
        seen += 1
        for child in np.flatnonzero(graph[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return bool(seen == graph.shape[0])


def _to_dag(names: Sequence[str], graph: BoolArray) -> MixedGraph:
    sources, targets = np.nonzero(graph)
    edges = [(names[int(s)], names[int(t)]) for s, t in zip(sources, targets, strict=True)]
    return dag(edges, nodes=names)


def _pair_states(graph: BoolArray) -> npt.NDArray[np.uint8]:
    low, high = np.triu_indices(graph.shape[0], k=1)
    forward = graph[low, high].astype(np.uint8)
    reverse = graph[high, low].astype(np.uint8)
    result: npt.NDArray[np.uint8] = forward + 2 * reverse
    return result


def _medoid(graphs: Sequence[BoolArray]) -> int:
    """The candidate with the least summed pair-state distance; earliest on a tie."""
    states = np.stack([_pair_states(g) for g in graphs])
    columns = np.arange(states.shape[1])
    counts = np.stack([(states == state).sum(axis=0) for state in range(3)])
    # distance of candidate i to all others = sum over pairs of the candidates disagreeing
    distances = (len(graphs) - counts[states, columns]).sum(axis=1)
    return int(np.argmin(distances))


# -- greedy -------------------------------------------------------------------------------


def _greedy(
    weights: FloatArray,
    values: FloatArray,
    threshold: float | None,
    constraints: _Constraints,
) -> tuple[BoolArray, int]:
    size = weights.shape[0]
    fixed = constraints.required_pairs
    nominations: list[tuple[float, int, int]] = []
    for first in range(size):
        for second in range(first + 1, size):
            if frozenset((first, second)) in fixed:
                continue
            if weights[first, second] >= weights[second, first]:
                source, target = first, second
            else:
                source, target = second, first
            weight = float(weights[source, target])
            if weight == -np.inf:
                continue
            keep = weight > 0.0 if threshold is None else values[source, target] > threshold
            if keep:
                nominations.append((weight, source, target))
    nominations.sort(key=lambda item: (-item[0], item[1], item[2]))
    graph = np.zeros((size, size), dtype=bool)
    for source, target in constraints.required:
        graph[source, target] = True
    reach = constraints.base_graph(size)
    rejected = 0
    for _, source, target in nominations:
        if _reaches(reach, target, source):
            rejected += 1
            continue
        graph[source, target] = True
        reach[source, target] = True
    return graph, rejected


# -- exact order search -------------------------------------------------------------------


def _order_weights(weights: FloatArray, constraints: _Constraints) -> FloatArray:
    """Weights for the order search: required pairs contribute a constant, so zero."""
    result = weights.copy()
    for source, target in constraints.required:
        result[source, target] = 0.0
        result[target, source] = 0.0
    return result


def _maximum_weight_order(
    weights: FloatArray, constraints: _Constraints
) -> tuple[tuple[int, ...], int]:
    """A* over topological orders for the positive compatible-edge objective.

    Ported from the research ``_maximum_weight_order``: a state is the set of
    placed nodes, the realized score sums the positive weights from placed nodes
    to each newly placed one, and the bound adds every positive weight still
    available. Ties go to the lexically smaller order. Knowledge adds one rule:
    a node is placed only after every node that must precede it.
    """
    size = weights.shape[0]
    full = (1 << size) - 1
    positive = np.maximum(weights, 0.0)
    pair_best = np.maximum(positive, positive.T)
    must = constraints.must_precede
    uppers: dict[int, tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]] = {}

    def upper_of(count: int) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        if count not in uppers:
            uppers[count] = np.triu_indices(count, k=1)
        return uppers[count]

    def bound(mask: int) -> float:
        placed = np.fromiter(((mask >> node) & 1 for node in range(size)), dtype=bool)
        remaining = ~placed
        cross = float(positive[np.ix_(placed, remaining)].sum())
        # Summed one pair at a time in the research order: the queue breaks ties on
        # these floats, so a different summation order could pick another optimum.
        internal = 0.0
        for value in pair_best[np.ix_(remaining, remaining)][upper_of(int(remaining.sum()))]:
            internal += float(value)
        return cross + internal

    queue: list[tuple[float, float, tuple[int, ...], int]] = [(-bound(0), -0.0, (), 0)]
    best_score: dict[int, float] = {0: 0.0}
    best_order: dict[int, tuple[int, ...]] = {0: ()}
    expanded = 0
    while queue:
        _, negative, order, mask = heapq.heappop(queue)
        score = -negative
        incumbent = best_score.get(mask)
        if incumbent is None or score < incumbent - 1e-12 or order != best_order[mask]:
            continue
        expanded += 1
        if mask == full:
            return order, expanded
        for target in range(size):
            bit = 1 << target
            if mask & bit or (must[target] & mask) != must[target]:
                continue
            increment = 0.0
            for source in order:  # sequential, as in the research code
                increment += float(positive[source, target])
            next_mask = mask | bit
            next_score = score + increment
            next_order = (*order, target)
            previous = best_score.get(next_mask, -math.inf)
            previous_order = best_order.get(next_mask)
            improves = next_score > previous + 1e-12
            ties_better = abs(next_score - previous) <= 1e-12 and (
                previous_order is None or next_order < previous_order
            )
            if not improves and not ties_better:
                continue
            best_score[next_mask] = next_score
            best_order[next_mask] = next_order
            upper = next_score + bound(next_mask)
            heapq.heappush(queue, (-upper, -next_score, next_order, next_mask))
    raise RuntimeError("the order search ran out of states")  # pragma: no cover


def _graph_for_order(
    weights: FloatArray, order: tuple[int, ...], constraints: _Constraints
) -> BoolArray:
    size = weights.shape[0]
    graph = np.zeros((size, size), dtype=bool)
    for position, source in enumerate(order):
        for target in order[position + 1 :]:
            if weights[source, target] > 0.0:
                graph[source, target] = True
    _apply_required(graph, constraints)
    return graph


def _apply_required(graph: BoolArray, constraints: _Constraints) -> None:
    for source, target in constraints.required:
        graph[target, source] = False
        graph[source, target] = True


def _expected_edge_count(pair: FloatArray) -> int:
    upper = np.triu_indices(pair.shape[0], k=1)
    return math.floor(float((pair + pair.T)[upper].sum()) + 0.5)


def _soft_f1_budget(pair: FloatArray) -> int:
    upper = np.triu_indices(pair.shape[0], k=1)
    existence = np.sort((pair + pair.T)[upper])[::-1]
    expected = float(existence.sum())
    if expected == 0.0:
        return 0
    cumulative = np.cumsum(existence)
    counts = np.arange(1, existence.size + 1, dtype=np.float64)
    return int(np.argmax(2.0 * cumulative / (counts + expected))) + 1


def _cardinality_graph(
    weights: FloatArray, order: tuple[int, ...], budget: int, constraints: _Constraints
) -> BoolArray:
    """The ``budget`` best order-compatible edges; required edges count toward it."""
    size = weights.shape[0]
    required = set(constraints.required)
    candidates: list[tuple[float, int, int]] = []
    for position, source in enumerate(order):
        for target in order[position + 1 :]:
            if (source, target) in required or (target, source) in required:
                continue
            if weights[source, target] > -np.inf:
                candidates.append((float(weights[source, target]), source, target))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    graph = np.zeros((size, size), dtype=bool)
    for _, source, target in candidates[: max(0, budget - len(required))]:
        graph[source, target] = True
    _apply_required(graph, constraints)
    return graph


# -- samplers -------------------------------------------------------------------------------


def _pair_sampler(
    pair: FloatArray, samples: int, seed: int, constraints: _Constraints
) -> tuple[list[BoolArray], list[int]]:
    rng = np.random.default_rng(seed)
    size = pair.shape[0]
    fixed = constraints.required_pairs
    permitted = constraints.permitted
    upper = tuple(zip(*np.triu_indices(size, k=1), strict=True))
    graphs: list[BoolArray] = []
    rejections: list[int] = []
    for _ in range(samples):
        proposed: list[tuple[int, int]] = []
        for low_value, high_value in upper:
            low, high = int(low_value), int(high_value)
            forward = float(pair[low, high]) if permitted[low, high] else 0.0
            reverse = float(pair[high, low]) if permitted[high, low] else 0.0
            none = max(0.0, 1.0 - float(pair[low, high]) - float(pair[high, low]))
            masses = np.asarray((none, forward, reverse), dtype=np.float64)
            total = float(masses.sum())
            masses = masses / total if total > 0.0 else np.asarray((1.0, 0.0, 0.0))
            state = int(rng.choice(3, p=masses))
            if frozenset((low, high)) in fixed:
                continue
            if state == 1:
                proposed.append((low, high))
            elif state == 2:
                proposed.append((high, low))
        graph = np.zeros((size, size), dtype=bool)
        for source, target in constraints.required:
            graph[source, target] = True
        reach = constraints.base_graph(size)
        rejected = 0
        for index in rng.permutation(len(proposed)):
            source, target = proposed[int(index)]
            if _reaches(reach, target, source):
                rejected += 1
                continue
            graph[source, target] = True
            reach[source, target] = True
        graphs.append(graph)
        rejections.append(rejected)
    return graphs, rejections


def _perturb_and_map(
    pair: FloatArray, samples: int, seed: int, constraints: _Constraints
) -> tuple[list[BoolArray], list[int]]:
    rng = np.random.default_rng(seed)
    size = pair.shape[0]
    permitted = constraints.permitted
    upper = tuple(zip(*np.triu_indices(size, k=1), strict=True))
    graphs: list[BoolArray] = []
    expansions: list[int] = []
    for _ in range(samples):
        weights = np.zeros((size, size), dtype=np.float64)
        for low_value, high_value in upper:
            low, high = int(low_value), int(high_value)
            state = np.asarray(
                (
                    max(0.0, 1.0 - pair[low, high] - pair[high, low]),
                    pair[low, high],
                    pair[high, low],
                ),
                dtype=np.float64,
            )
            perturbed = np.log(np.maximum(state, PROBABILITY_FLOOR)) + rng.gumbel(size=3)
            weights[low, high] = perturbed[1] - perturbed[0] if permitted[low, high] else -np.inf
            weights[high, low] = perturbed[2] - perturbed[0] if permitted[high, low] else -np.inf
        order, expanded = _maximum_weight_order(_order_weights(weights, constraints), constraints)
        graphs.append(_graph_for_order(weights, order, constraints))
        expansions.append(expanded)
    return graphs, expansions
