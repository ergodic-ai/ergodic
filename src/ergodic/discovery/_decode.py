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

The decoders are ported from Ergodic's research library, where greedy pair-MAP
matched exact MAP on 238 of 240 graphs. Each id carries a version, and a decoded
graph always says it was decoded, by which decoder, with which parameters and
seed. It is never a graph posterior, and its orientations are not certified by
anything.

Domain knowledge enters in three ways. Hard constraints remove or force edges.
Soft statements reweigh the pair states (:meth:`~ergodic.DomainKnowledge.log_odds`)
before any decoder reads them, so every decoder runs unchanged on the weighted
states. A confounder (``confound``) adds a bidirected edge after the directed part
is decoded, which makes the result an ADMG. A decode with knowledge also audits
each statement against the model.
"""

from __future__ import annotations

import heapq
import math
import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..graph import Mark, admg, dag
from ..knowledge import _pair_rules
from ._probabilities import EdgeProbabilities
from ._result import DiscoveryResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ..graph import MixedGraph
    from ..knowledge import DomainKnowledge, KnowledgeStatement

    FloatArray = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

PROBABILITY_FLOOR: Final = 1e-12
DECODED_NOTE: Final = "not a graph posterior; orientation not certified"
DEFAULT_MAX_NODES: Final = 20
CONFLICT_PROBABILITY: Final = 0.1
"""A statement the model gives less than this probability is a strong conflict."""


@dataclass(frozen=True)
class _Spec:
    port_of: str
    stochastic: bool
    order_search: bool
    summary: str


_SPECS: Final[dict[str, _Spec]] = {
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

DECODERS: Final[Mapping[str, str]] = MappingProxyType(
    {name: spec.summary for name, spec in _SPECS.items()}
)
"""The registered decoders: versioned id to a one-line summary (read-only)."""

_ALIASES: Final = {name.rsplit("_v", 1)[0]: name for name in _SPECS}


def resolve_decoder(method: str) -> str:
    """The versioned decoder id for ``method`` (``"greedy"`` reads as ``"greedy_v1"``)."""
    if not isinstance(method, str):
        raise TypeError(f"method must be a decoder id string, got {type(method).__name__}")
    key = method.lower()
    if key in _SPECS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    names = ", ".join(sorted(DECODERS))
    raise ValueError(f"unknown decoder {method!r}; use one of: {names} (or drop the _v1)")


@dataclass(frozen=True)
class DecodeResult(DiscoveryResult):
    """A DAG (or ADMG) decoded from edge probabilities, with how it was decoded.

    A :class:`DiscoveryResult` whose ``graph`` is the decoded graph: the single
    decoded graph for a deterministic decoder, the consensus (medoid) of the
    candidates for a sampler. ``method`` is ``"decode:<decoder id>"`` (a Welt run
    decoded locally keeps ``"welt:<model>"``). Its ``kind`` is ``dag``, or
    ``admg`` when the knowledge holds ``confound`` pairs: each such pair gets
    ``a <-> b`` on top of the decoded directed edges. When the decoder also chose
    a directed edge on that pair (a bow), the confounder is written as an explicit
    latent node ``L__a__b`` (in ``graph.latent``, with an edge into each end), since
    a graph holds one edge per pair; identification reads both forms alike.

    ``diagnostics`` always labels the graph: ``decoded`` is ``True``, ``native`` is
    ``False``, and it records ``decoder``, ``parameters``, ``seed``,
    ``samples``, ``rejected_cycles`` (nominated edges skipped because they would
    close a directed cycle), ``order_skipped`` (skipped only because they would
    break the order knowledge) and ``search_expansions`` (each one per
    candidate), ``threshold_source``, ``edge_budget``, ``objective`` (the
    factorized pair log score of ``graph``'s directed edges on the input's pair
    states), ``knowledge_objective`` (the same score on the knowledge-weighted
    pair states the decoder read; equal to ``objective`` without soft
    statements), ``n_edges`` (directed edges), ``graph_kind``,
    ``confounded_pairs``, ``latent_confounders`` (latent node to pair, bows
    only), ``source_kind``, ``source_provenance``, ``knowledge_applied``,
    ``n_hard_statements``, ``n_soft_statements`` and ``note`` ("not a graph
    posterior; orientation not certified").

    With knowledge it also holds the audit:

    - ``knowledge_audit``: one entry per statement as stated
      (:meth:`~ergodic.DomainKnowledge.statements`), a dict with ``statement``
      (readable text), ``kind``, ``confidence`` (``None`` for hard),
      ``probability`` (the model's chance the statement holds, from the input's
      pair states with pairs taken as independent; ``None`` for a confounder,
      which no edge state expresses, with the reason in ``note``), ``holds`` (in
      ``graph``) and ``holds_without_knowledge`` (in a decode with the same
      settings and seed but no knowledge). Statements are read pair by pair, as
      the decoders weigh them: an order or tier holds when no direct edge runs
      against it.
    - ``knowledge_cost``: the knowledge-free decode's ``objective`` minus this
      one's. Non-negative for ``exact_map``; the other decoders are not exact on
      the score, so it can be negative.
    - ``knowledge_conflicts``: the audit entries the model gives a probability
      below 0.1. Any such entry also raises one ``UserWarning``.

    Without knowledge these are ``()``, ``None`` and ``()``.

    Attributes:
        decoder: The versioned decoder id, for example ``"greedy_v1"``.
        candidates: Every decoded graph; one for a deterministic decoder, one per
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
        return _SPECS[self.method].stochastic

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
    spec = _SPECS[decoder]
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


def check_size(decoder: str, size: int, max_nodes: int) -> None:
    """Raise when an exact order search would exceed ``max_nodes`` variables."""
    if _SPECS[resolve_decoder(decoder)].order_search and size > max_nodes:
        raise ValueError(
            f"{decoder!r} searches topological orders exactly and refuses {size} variables "
            f"above max_nodes={max_nodes}; use 'greedy' or 'pair_sampler', or raise max_nodes"
        )


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
      the smallest total pair-state distance to the others, earliest on an exact
      tie; the research ``posterior_medoid`` can pick another tied candidate).
      Candidates are not posterior samples.

    Ties are broken by variable order, so a decoder is deterministic given its
    inputs (and seed). The input is never changed and the global numpy random
    state is never touched.

    Knowledge constrains the decode. Hard forbidden edges, tiers, roots, sinks
    and a skeleton restriction remove directed edges (their weight becomes minus
    infinity). Hard order (``order``, tiers) is enforced as one time order: the
    decoded graph has a topological order consistent with every order statement
    and every edge at once. That is exactly the tier reading, and stricter than
    :meth:`~ergodic.DomainKnowledge.violations` for separate ``order`` pairs (it
    only asks that no later variable be an ancestor of an earlier one), so a DAG
    that satisfies each pair alone but no single order is ruled out. Required
    edges go in first, whatever their weight, and must not form a cycle. The
    decoded graph is checked against the hard knowledge afterwards.

    Soft statements (any builder's ``confidence=`` below 1) are weighed, not
    enforced. After the input is read as pair states, each pair's directed
    states are multiplied by ``exp(knowledge.log_odds(variables))`` and the pair
    is renormalized; every decoder then runs unchanged on those states, so
    ``exact_map`` is exact on the combined score (pair score plus
    :meth:`~ergodic.DomainKnowledge.log_prior`). Greedy with ``threshold=t``
    multiplies the input value's odds by the same ratio before comparing it with
    ``t``. A soft statement on a pair a hard one already settles changes nothing.

    ``confound(a, b)`` adds ``a <-> b`` after the directed part is decoded, so
    the graph and every candidate is an ADMG (see :class:`DecodeResult` for a
    pair the decoder also joined with a directed edge).

    With knowledge, the decode runs a second time without it (same settings and
    seed) and ``diagnostics`` records the audit: per statement, the model's
    probability that it holds and whether it holds with and without the
    knowledge, the score the knowledge cost, and the strong conflicts (model
    probability below 0.1), which also raise one ``UserWarning``.

    Args:
        probabilities: The edge probabilities to decode.
        method: A decoder id, with or without its ``_v1`` version.
        threshold: Greedy only; see above. ``None`` keeps the ``w > 0`` rule.
        knowledge: Hard and soft statements, all naming variables of
            ``probabilities``.
        samples: How many candidates a sampler draws; 1 for the others.
        seed: The sampler's seed (required); deterministic decoders refuse one.
        max_nodes: The variable cap for the decoders that search orders exactly
            (all but greedy and pair_sampler). The search can grow exponentially
            with the variable count, so above the cap they raise.

    Returns:
        A :class:`DecodeResult`: ``graph`` a DAG over every variable (an ADMG
        with confounders), with the decoder, its parameters and the diagnostics
        recorded.

    Raises:
        TypeError: ``probabilities`` is not an :class:`EdgeProbabilities`.
        ValueError: An unknown decoder, a setting that does not apply, too many
            variables for an exact search, knowledge naming unknown variables,
            inconsistent knowledge or a cyclic required set.
    """
    if not isinstance(probabilities, EdgeProbabilities):
        raise TypeError(
            "decode reads an EdgeProbabilities; build one with "
            "EdgeProbabilities(variables, values, kind=...) or EdgeProbabilities.from_frame"
        )
    decoder = resolve_decoder(method)
    _check_settings(decoder, threshold, samples, seed, max_nodes)
    spec = _SPECS[decoder]
    names = probabilities.variables
    check_size(decoder, len(names), max_nodes)
    constraints = _Constraints.build(names, knowledge)
    source = probabilities.to_pair_states().values
    run = _run(probabilities, source, decoder, threshold, constraints, samples, seed, max_nodes)
    confounded = tuple((names[i], names[j]) for i, j in constraints.confounded)
    candidates = tuple(_to_graph(names, g, confounded) for g in run.graphs)
    result_graph = candidates[run.chosen]
    objective = _pair_log_score(run.graphs[run.chosen], source, PROBABILITY_FLOOR)
    audit: tuple[dict[str, Any], ...] = ()
    conflicts: tuple[dict[str, Any], ...] = ()
    cost: float | None = None
    statements = knowledge.statements() if knowledge is not None else ()
    if knowledge is not None:
        broken = knowledge.violations(result_graph)
        if broken:  # pragma: no cover - the decoders enforce every constraint they accept
            raise RuntimeError(f"decoded graph breaks the knowledge: {broken}")
        free = _run(
            probabilities,
            source,
            decoder,
            threshold,
            _Constraints.build(names, None),
            samples,
            seed,
            max_nodes,
        )
        free_graph = free.graphs[free.chosen]
        cost = _pair_log_score(free_graph, source, PROBABILITY_FLOOR) - objective
        audit = _audit(statements, names, source, run.graphs[run.chosen], free_graph)
        conflicts = tuple(
            entry
            for entry in audit
            if entry["probability"] is not None and entry["probability"] < CONFLICT_PROBABILITY
        )
        if conflicts:
            listed = "; ".join(
                f"{entry['statement']} ({entry['probability']:.3g})" for entry in conflicts
            )
            warnings.warn(
                f"{len(conflicts)} knowledge statement(s) get a model probability below "
                f"{CONFLICT_PROBABILITY:g}: {listed}. See diagnostics['knowledge_audit']",
                UserWarning,
                stacklevel=2,
            )
    latent = {
        node: pair
        for node in sorted(result_graph.latent)
        for pair in confounded
        if result_graph.children(node) == frozenset(pair)
    }
    diagnostics: dict[str, Any] = {
        "decoded": True,
        "native": False,
        "decoder": decoder,
        "decoder_port_of": spec.port_of,
        "parameters": run.parameters,
        "threshold_source": "decoder" if threshold is not None else None,
        "seed": seed,
        "samples": samples,
        "rejected_cycles": run.rejections,
        "order_skipped": run.order_skips,
        "search_expansions": run.expansions,
        "edge_budget": run.edge_budget,
        "objective": objective,
        "knowledge_objective": _pair_log_score(run.graphs[run.chosen], run.pair, PROBABILITY_FLOOR),
        "n_edges": int(run.graphs[run.chosen].sum()),
        "graph_kind": result_graph.kind.value,
        "confounded_pairs": confounded,
        "latent_confounders": latent,
        "source_kind": probabilities.kind,
        "source_calibrated": probabilities.calibrated,
        "source_provenance": probabilities.provenance,
        "knowledge_applied": knowledge is not None,
        "n_hard_statements": sum(not statement.is_soft for statement in statements),
        "n_soft_statements": sum(statement.is_soft for statement in statements),
        "knowledge_audit": audit,
        "knowledge_cost": cost,
        "knowledge_conflicts": conflicts,
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


@dataclass(frozen=True)
class _Run:
    """One decode's raw output: candidate adjacency matrices and their counters."""

    graphs: tuple[BoolArray, ...]
    chosen: int
    rejections: tuple[int, ...]
    order_skips: tuple[int, ...]
    expansions: tuple[int, ...]
    edge_budget: int | None
    parameters: dict[str, Any]
    pair: FloatArray  # the pair states the decoder read, knowledge-weighted


def _run(
    probabilities: EdgeProbabilities,
    source: FloatArray,
    decoder: str,
    threshold: float | None,
    constraints: _Constraints,
    samples: int,
    seed: int | None,
    max_nodes: int,
) -> _Run:
    odds = constraints.log_odds
    pair = source if odds is None else _weighted_pair_states(source, odds)
    values = probabilities.values
    if odds is not None and threshold is not None:
        values = _weighted_values(values, odds)
    weights = _relative_log_weights(pair, PROBABILITY_FLOOR)
    weights[~constraints.permitted] = -np.inf
    parameters: dict[str, Any] = {"probability_floor": PROBABILITY_FLOOR}
    edge_budget: int | None = None
    if decoder == "greedy_v1":
        parameters["threshold"] = threshold
        graph, rejected, skipped = _greedy(weights, values, threshold, constraints)
        graphs, rejections, order_skips, expansions = [graph], [rejected], [skipped], [0]
    elif decoder == "pair_sampler_v1":
        graphs, rejections, order_skips = _pair_sampler(pair, samples, seed or 0, constraints)
        expansions = [0] * samples
    elif decoder == "perturb_and_map_v1":
        parameters["max_nodes"] = max_nodes
        graphs, expansions = _perturb_and_map(pair, samples, seed or 0, constraints)
        rejections = [0] * samples
        order_skips = [0] * samples
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
        graphs, rejections, order_skips, expansions = [graph], [0], [0], [expanded]
    return _Run(
        graphs=tuple(graphs),
        chosen=_medoid(graphs) if len(graphs) > 1 else 0,
        rejections=tuple(rejections),
        order_skips=tuple(order_skips),
        expansions=tuple(expansions),
        edge_budget=edge_budget,
        parameters=parameters,
        pair=pair,
    )


# -- knowledge -------------------------------------------------------------------------


def check_knowledge(names: Sequence[str], knowledge: DomainKnowledge) -> None:
    """Raise as :func:`decode` would on knowledge it cannot apply to ``names``.

    Lets a caller (the Welt bridge) refuse unusable knowledge before a paid run.
    """
    _Constraints.build(tuple(names), knowledge)


@dataclass(frozen=True)
class _Constraints:
    """Knowledge as index arrays: permitted cells, required edges, precedence.

    Also the confounded pairs (added after decoding) and the soft statements'
    log likelihood ratios (``None`` when they change nothing).
    """

    permitted: BoolArray
    required: tuple[tuple[int, int], ...]
    precedence: tuple[tuple[int, int], ...]
    must_precede: tuple[int, ...]  # bitmask of nodes that must come earlier in the order
    confounded: tuple[tuple[int, int], ...] = ()
    log_odds: FloatArray | None = None

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
        confounded = tuple(
            sorted(
                (min(index[a], index[b]), max(index[a], index[b]))
                for a, b in (sorted(pair) for pair in knowledge.confounded)
            )
        )
        odds = knowledge.log_odds(list(names)) if knowledge.soft else None
        constraints = cls(
            np.asarray(mask, dtype=bool),
            required,
            precedence,
            tuple(must),
            confounded,
            odds if odds is not None and odds.any() else None,
        )
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


# -- soft knowledge and confounders --------------------------------------------------------


def _weighted_pair_states(pair: FloatArray, odds: FloatArray) -> FloatArray:
    """Multiply each pair's directed states by ``exp(odds)`` and renormalize the pair.

    Pairs with no soft weight keep their numbers exactly. Computed in log space,
    so large confidences neither overflow nor lose the smaller states.
    """
    result = pair.copy()
    size = pair.shape[0]
    for low in range(size):
        for high in range(low + 1, size):
            if odds[low, high] == 0.0 and odds[high, low] == 0.0:
                continue
            forward, reverse = float(pair[low, high]), float(pair[high, low])
            terms = (
                _log(max(0.0, 1.0 - forward - reverse)),
                _log(forward) + float(odds[low, high]),
                _log(reverse) + float(odds[high, low]),
            )
            top = max(terms)
            if top == -math.inf:  # pragma: no cover - the three states sum to one
                continue
            masses = [math.exp(term - top) for term in terms]
            total = sum(masses)
            result[low, high] = masses[1] / total
            result[high, low] = masses[2] / total
    return result


def _weighted_values(values: FloatArray, odds: FloatArray) -> FloatArray:
    """Input values with their odds multiplied by ``exp(odds)``: the threshold path."""
    result = values.copy()
    for source, target in zip(*np.nonzero(odds), strict=True):
        value = float(values[source, target])
        if 0.0 < value < 1.0:
            logit = math.log(value) - math.log1p(-value) + float(odds[source, target])
            result[source, target] = 1.0 / (1.0 + math.exp(-logit)) if logit > -700 else 0.0
    return result


def _log(value: float) -> float:
    return math.log(value) if value > 0.0 else -math.inf


def _latent_name(a: str, b: str, taken: set[str]) -> str:
    first, second = sorted((a, b))
    name = f"L__{first}__{second}"
    while name in taken:
        name = "_" + name
    return name


def _to_graph(
    names: Sequence[str], graph: BoolArray, confounded: Sequence[tuple[str, str]]
) -> MixedGraph:
    """The decoded DAG, or the ADMG it becomes with the confounded pairs added."""
    sources, targets = np.nonzero(graph)
    edges: list[Any] = [
        (names[int(s)], names[int(t)]) for s, t in zip(sources, targets, strict=True)
    ]
    if not confounded:
        return dag(edges, nodes=names)
    index = {name: i for i, name in enumerate(names)}
    taken = set(names)
    latent: list[str] = []
    for a, b in confounded:
        if graph[index[a], index[b]] or graph[index[b], index[a]]:
            # A bow: the pair already holds a directed edge, and a graph holds one
            # edge per pair, so the confounder becomes an explicit latent node.
            node = _latent_name(a, b, taken)
            taken.add(node)
            latent.append(node)
            edges += [(node, a), (node, b)]
        else:
            edges.append((a, Mark.ARROW, b, Mark.ARROW))
    return admg(edges, nodes=(*names, *latent), latent=latent)


def _pair_state(graph: BoolArray, i: int, j: int) -> int:
    return 1 if graph[i, j] else 2 if graph[j, i] else 0


def _audit(
    statements: Sequence[KnowledgeStatement],
    names: Sequence[str],
    source: FloatArray,
    decoded: BoolArray,
    free: BoolArray,
) -> tuple[dict[str, Any], ...]:
    """One entry per statement: the model's probability and where it holds."""
    index = {name: i for i, name in enumerate(names)}
    entries: list[dict[str, Any]] = []
    for statement in statements:
        entry: dict[str, Any] = {
            "statement": str(statement),
            "kind": statement.kind,
            "confidence": statement.confidence,
        }
        if statement.kind == "confound":
            entry.update(
                probability=None,
                holds=True,
                holds_without_knowledge=False,
                note="a hidden common cause is not an edge state, so the edge "
                "probabilities cannot price it",
            )
            entries.append(entry)
            continue
        satisfied: dict[tuple[int, int], list[bool]] = {}
        holds = holds_free = True
        for a, b, rule in _pair_rules(statement, names):
            i, j = index[a], index[b]
            holds = holds and rule[_pair_state(decoded, i, j)]
            holds_free = holds_free and rule[_pair_state(free, i, j)]
            low, high = min(i, j), max(i, j)
            # The rule's states, re-read from (low, high): state 1 is low -> high.
            states = rule if (low, high) == (i, j) else (rule[0], rule[2], rule[1])
            current = satisfied.setdefault((low, high), [True, True, True])
            for state in range(3):
                current[state] = current[state] and states[state]
        probability = 1.0
        for (low, high), mask in sorted(satisfied.items()):
            forward, reverse = float(source[low, high]), float(source[high, low])
            masses = (max(0.0, 1.0 - forward - reverse), forward, reverse)
            probability *= min(1.0, sum(m for m, keep in zip(masses, mask, strict=True) if keep))
        entry.update(
            probability=probability,
            holds=bool(holds),
            holds_without_knowledge=bool(holds_free),
            note=None,
        )
        entries.append(entry)
    return tuple(entries)


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
) -> tuple[BoolArray, int, int]:
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
    rejected = skipped = 0
    for _, source, target in nominations:
        if _reaches(reach, target, source):
            if _reaches(graph, target, source):
                rejected += 1
            else:
                skipped += 1
            continue
        graph[source, target] = True
        reach[source, target] = True
    return graph, rejected, skipped


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
) -> tuple[list[BoolArray], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    size = pair.shape[0]
    fixed = constraints.required_pairs
    permitted = constraints.permitted
    upper = tuple(zip(*np.triu_indices(size, k=1), strict=True))
    graphs: list[BoolArray] = []
    rejections: list[int] = []
    order_skips: list[int] = []
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
        rejected = skipped = 0
        for index in rng.permutation(len(proposed)):
            source, target = proposed[int(index)]
            if _reaches(reach, target, source):
                if _reaches(graph, target, source):
                    rejected += 1
                else:
                    skipped += 1
                continue
            graph[source, target] = True
            reach[source, target] = True
        graphs.append(graph)
        rejections.append(rejected)
        order_skips.append(skipped)
    return graphs, rejections, order_skips


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
