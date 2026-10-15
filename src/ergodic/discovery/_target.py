# SPDX-License-Identifier: Apache-2.0
"""The unnormalized log target over DAGs that the graph posterior samples.

One object serves both kinds of method:

- **Edge probabilities.** A DAG's log target is its factorized pair log score
  (each unordered pair contributes the log probability of the state the DAG gives
  it, read as pair states exactly as the decoders read them and floored at
  ``1e-12``) plus the knowledge's :meth:`~ergodic.DomainKnowledge.log_prior`.
- **A score.** A DAG's log target is the sum of its local scores plus the same
  log prior. The structure prior is uniform over DAGs times the knowledge.

Both are stored as pair state log weights (``log_none`` per pair, ``log_edge`` per
directed edge) plus, for a score, the node-modular local scores. Hard knowledge
puts ``-inf`` on the states it rules out: a banned edge, the reverse of a required
edge and "no edge" on a required pair. Soft knowledge adds
:meth:`~ergodic.DomainKnowledge.log_odds` to the directed states, and the
constant ``soft_constant`` (each soft rule's term at "no edge") makes the total
equal ``pair score + log_prior`` exactly, not just up to a constant.

Order statements (``order``, tiers) are ancestral: ``a`` before ``b`` means ``b`` is
not an ancestor of ``a``, as :meth:`~ergodic.DomainKnowledge.violations` reads them.
Edges against the (transitively closed) order are banned pair by pair. When no
directed path of permitted edges can run from ``b`` back to ``a`` that ban is the
whole constraint (tiers that cover every variable are the common case). Otherwise
the pair stays an ancestral constraint, which no pair factorization can express,
and the exact computation does not apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..knowledge import _pair_rules
from ._decode import PROBABILITY_FLOOR
from ._exact import reachability

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ..graph import MixedGraph
    from ..knowledge import DomainKnowledge
    from ._probabilities import EdgeProbabilities
    from ._score import Score

    FloatArray = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class _Knowledge:
    """Knowledge as arrays for one variable order."""

    permitted: BoolArray
    required: tuple[tuple[int, int], ...]
    ancestral: tuple[tuple[int, int], ...]  # (a, b): b must not be an ancestor of a
    log_odds: FloatArray
    soft_constant: float
    confounded: tuple[tuple[str, str], ...]

    @classmethod
    def build(cls, names: Sequence[str], knowledge: DomainKnowledge | None) -> _Knowledge:
        size = len(names)
        permitted = ~np.eye(size, dtype=bool)
        if knowledge is None:
            return cls(permitted, (), (), np.zeros((size, size)), 0.0, ())
        unknown = sorted(knowledge.variables - set(names))
        if unknown:
            raise ValueError(f"knowledge names variables the posterior lacks: {unknown}")
        knowledge.validate()
        mask, _ = knowledge.edge_mask(list(names))
        permitted = np.asarray(mask, dtype=bool)
        index = {name: i for i, name in enumerate(names)}
        required = tuple(sorted((index[u], index[v]) for u, v in knowledge.required))
        for source, target in required:
            if not permitted[source, target]:
                raise ValueError(
                    f"the knowledge both requires and rules out {names[source]} -> {names[target]}"
                )
        required_graph = np.zeros((size, size), dtype=bool)
        for source, target in required:
            required_graph[source, target] = True
        required_reach = reachability(required_graph[None])[0]
        if np.any(np.diag(required_reach)):
            raise ValueError("the required edges form a cycle, so no DAG satisfies the knowledge")
        # A permitted path b ~> a is the only way to break "a before b" once the direct
        # edges against the order are banned.
        allowed = permitted.copy()
        for source, target in required:
            allowed[target, source] = False
        open_reach = reachability(allowed[None])[0]
        ancestral: list[tuple[int, int]] = []
        for a, b in sorted(knowledge.precedence()):
            first, second = index[a], index[b]
            if required_reach[second, first]:
                raise ValueError(
                    f"the required edges make {b} an ancestor of {a}, against the order "
                    f"{a} before {b}"
                )
            if open_reach[second, first]:
                ancestral.append((first, second))
        odds = knowledge.log_odds(list(names)) if knowledge.soft else np.zeros((size, size))
        constant = 0.0
        for statement in knowledge.soft:
            confidence = float(statement.confidence or 0.5)
            for _, _, holds in _pair_rules(statement, list(names)):
                constant += math.log(confidence) if holds[0] else math.log1p(-confidence)
        confounded = tuple(sorted((min(pair), max(pair)) for pair in knowledge.confounded))
        return cls(permitted, required, tuple(ancestral), odds, constant, confounded)


@dataclass(frozen=True, eq=False)
class GraphTarget:
    """A DAG's unnormalized log target, as pair state weights plus optional local scores.

    ``log_target(G) = constant + sum over pairs of the state's log weight
    + sum over nodes of local(node, parents)`` (no local term for a pair target).
    """

    names: tuple[str, ...]
    kind: str  # "pair_state" or "score:<name>"
    log_none: FloatArray = field(repr=False)
    log_edge: FloatArray = field(repr=False)
    constant: float
    ancestral: tuple[tuple[int, int], ...]
    knowledge: DomainKnowledge | None = field(repr=False)
    confounded: tuple[tuple[str, str], ...] = ()
    score: Score | None = field(default=None, repr=False)
    approximate: bool = False
    _cache: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)

    # -- builders ----------------------------------------------------------------------

    @classmethod
    def from_probabilities(
        cls, probabilities: EdgeProbabilities, knowledge: DomainKnowledge | None
    ) -> GraphTarget:
        names = probabilities.variables
        pair = probabilities.to_pair_states().values
        none = np.maximum(0.0, 1.0 - pair - pair.T)
        log_none = np.log(np.maximum(none, PROBABILITY_FLOOR))
        log_edge = np.log(np.maximum(pair, PROBABILITY_FLOOR))
        rules = _Knowledge.build(names, knowledge)
        log_none, log_edge = _apply(rules, log_none, log_edge)
        return cls(
            names,
            "pair_state",
            log_none,
            log_edge,
            rules.soft_constant,
            rules.ancestral,
            knowledge,
            rules.confounded,
        )

    @classmethod
    def from_score(
        cls, score: Score, names: Sequence[str], knowledge: DomainKnowledge | None
    ) -> GraphTarget:
        names = tuple(names)
        size = len(names)
        rules = _Knowledge.build(names, knowledge)
        log_none, log_edge = _apply(rules, np.zeros((size, size)), np.zeros((size, size)))
        return cls(
            names,
            f"score:{score.name}",
            log_none,
            log_edge,
            rules.soft_constant,
            rules.ancestral,
            knowledge,
            rules.confounded,
            score,
            not getattr(score, "marginal_likelihood", False),
        )

    # -- reading ------------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def pair_factorized(self) -> bool:
        """Whether the target factorizes by pair (no local scores, no ancestral constraint)."""
        return self.score is None and not self.ancestral

    def local(self, node: int, parents: int) -> float:
        """The cached local score of ``node`` given the parent bitmask ``parents``, or 0."""
        if self.score is None:
            return 0.0
        key = (node, parents)
        hit = self._cache.get(key)
        if hit is None:
            chosen = [self.names[i] for i in range(self.size) if parents >> i & 1]
            hit = float(self.score.local(self.names[node], chosen))
            self._cache[key] = hit
        return hit

    def pair_log_weights(self, graph: BoolArray) -> float:
        """The pair part of the log target (``-inf`` when a state is ruled out)."""
        upper = np.triu_indices(self.size, k=1)
        forward = graph[upper]
        reverse = graph.T[upper]
        states = np.where(
            forward,
            self.log_edge[upper],
            np.where(reverse, self.log_edge.T[upper], self.log_none[upper]),
        )
        return float(states.sum())

    def log_target(self, graph: BoolArray) -> float:
        """The unnormalized log target of an adjacency matrix; ``-inf`` off the support."""
        if np.any(graph & graph.T) or not _acyclic(graph):
            return -math.inf
        total = self.constant + self.pair_log_weights(graph)
        if total == -math.inf:
            return total
        if self.ancestral:
            reach = reachability(graph[None])[0]
            if any(reach[b, a] for a, b in self.ancestral):
                return -math.inf
        if self.score is not None:
            for node in range(self.size):
                total += self.local(node, _mask(graph[:, node]))
        return total

    def stack_log_targets(self, graphs: BoolArray) -> FloatArray:
        """``log_target`` of each of a stack of DAGs already known to be on the support."""
        upper = np.triu_indices(self.size, k=1)
        forward = graphs[:, upper[0], upper[1]]
        reverse = graphs[:, upper[1], upper[0]]
        states = np.where(
            forward,
            self.log_edge[upper],
            np.where(reverse, self.log_edge.T[upper], self.log_none[upper]),
        )
        totals: FloatArray = self.constant + states.sum(axis=1)
        if self.score is not None:
            powers = 1 << np.arange(self.size, dtype=np.int64)
            masks = np.einsum("kij,i->kj", graphs.astype(np.int64), powers)
            for row in range(graphs.shape[0]):
                totals[row] += sum(
                    self.local(node, int(masks[row, node])) for node in range(self.size)
                )
        return totals

    def matrix(self, graph: MixedGraph) -> BoolArray:
        """The DAG's adjacency matrix in this target's variable order."""
        matrix, _ = graph.to_adjacency(order=list(self.names))
        result: BoolArray = np.asarray(matrix, dtype=bool)
        return result

    def describe(self) -> dict[str, Any]:
        """A JSON-ready description for diagnostics."""
        return {
            "kind": self.kind,
            "variables": list(self.names),
            "pair_factorized": self.pair_factorized,
            "ancestral_constraints": [(self.names[a], self.names[b]) for a, b in self.ancestral],
            "approximate": self.approximate,
            "probability_floor": PROBABILITY_FLOOR if self.score is None else None,
        }


def _apply(
    rules: _Knowledge, log_none: FloatArray, log_edge: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Hard knowledge as ``-inf`` states, soft knowledge as added log odds."""
    edge = log_edge + rules.log_odds
    edge = np.where(rules.permitted, edge, -np.inf)
    none = log_none.copy()
    for source, target in rules.required:
        none[source, target] = none[target, source] = -np.inf
        edge[target, source] = -np.inf
    np.fill_diagonal(edge, 0.0)
    np.fill_diagonal(none, 0.0)
    return none, edge


def _mask(column: BoolArray) -> int:
    return int(sum(1 << int(i) for i in np.flatnonzero(column)))


def _acyclic(graph: BoolArray) -> bool:
    from ._exact import acyclic

    return bool(acyclic(graph[None])[0])
