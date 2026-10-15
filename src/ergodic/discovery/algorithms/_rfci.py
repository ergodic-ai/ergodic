# SPDX-License-Identifier: Apache-2.0
"""RFCI: FCI's mark soundness with latent confounders, without the possible-d-sep phase.

FCI's cost lives in one place. After the PC skeleton it re-tests every
surviving pair against subsets of Possible-D-Sep sets, which can span most of
the graph. RFCI (Colombo, Maathuis, Kalisch, and Richardson 2012, Annals of
Statistics 40(1)) skips that phase entirely and never conditions on anything
the skeleton phase did not already produce: stage two re-tests the two edges
of each unshielded triple given the recorded separating set of its ends
(Lemma 3.1), and stage three guards the discriminating-path rule by
re-testing every consecutive pair on the path against subsets of the recorded
separating set of the path's ends (Lemma 3.2). Every conditioning set is a
subset of an adjacency set or of a recorded separating set, which is the
whole speed story. A removal found late feeds back: the edge comes off, a
minimal separating set is recorded, and the triple and orientation stages
rerun on the smaller graph.

The output is an RFCI-PAG (the paper's Definition 3.2), weaker than FCI's PAG
but sound (Theorem 3.2): an arrowhead at ``V`` on an edge to ``W`` says ``V``
is not an ancestor of ``W`` or of selection in the true MAG, and a tail says
it is. The skeleton is in general a supergraph of the FCI PAG's skeleton: an
extra edge survives when it sits on no unshielded triple or discriminating
path that would expose it, so one RFCI graph can cover more than one Markov
equivalence class. An output that fails the complete-PAG round trip is
therefore an expected honest outcome, not an error: it keeps the ``mixed``
face with ``diagnostics["pag_valid"] = 0.0``.

Selection bias is in scope, as in the paper: the closure runs Zhang's full
rule set including the selection rules R5 to R7, and data sampled under
selection simply answers every test conditional on it (the oracle conditions
on ``graph.selection`` internally), so selection worlds run unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import TYPE_CHECKING, ClassVar

from ...graph import GraphKind, InvalidGraphError, Mark
from ...graph._pag import _is_directed, _mark_maps, fci_closure
from ...knowledge import _refuse_soft
from .._registry import resolve_test
from .._result import DiscoveryResult
from ._fci import _circle_marks, _seed_knowledge, _to_graph
from ._skeleton import _protected_pairs, stable_skeleton

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...data import TabularData
    from ...graph import MixedGraph
    from ...knowledge import DomainKnowledge
    from .._ci import CITest

_Marks = dict[str, dict[str, Mark]]
_Adjacency = dict[str, set[str]]
_Sepsets = dict[frozenset[str], frozenset[str]]
_Triple = tuple[str, str, str]  # (x, m, y) with m the middle and x < y
_Pending = dict[frozenset[str], frozenset[str]]


@dataclass
class _Counts:
    """The per-run counters the two RFCI stages accumulate for the diagnostics."""

    stage2_removed: int = 0
    stage3_removed: int = 0
    triple_tests: int = 0
    path_tests: int = 0


class RFCI:
    """RFCI over a conditional-independence test: FCI's marks at PC's price.

    Args:
        test: A :class:`CITest` instance (bound or unbound, including
            :class:`DSeparationOracle`), or a name: ``"auto"`` (pick by the
            data schema), ``"linear"``/``"fisher_z"``, ``"discrete"``/
            ``"g_sq"``, ``"mixed"``/``"dg"``, ``"gcm"``.
        alpha: Significance level for the independence decisions.
        max_cond_size: Cap on the conditioning-set size in the skeleton
            phase; ``None`` for no cap. The later stages only ever condition
            on subsets of recorded separating sets, which the cap already
            bounds, so unlike FCI there is no second phase the cap must reach.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    carries the PAG face when the output happens to be a valid complete PAG
    and the ``mixed`` face with ``diagnostics["pag_valid"] = 0.0`` otherwise.
    Unlike FCI, the mixed face is an expected honest outcome here even under
    a perfect oracle: the RFCI skeleton can keep edges only the
    possible-d-sep phase would remove, and the result then stands for more
    than one Markov equivalence class. Domain knowledge enters exactly as in
    FCI: prune (a pair forbidden in both directions is never tested), protect
    (required and confounded pairs are never tested or removed), orient
    through ancestry (the same seeding helpers), validate at the end.
    """

    ingredient: ClassVar[str] = "test"

    def __init__(
        self,
        test: CITest | str = "auto",
        *,
        alpha: float = 0.05,
        max_cond_size: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_cond_size is not None and max_cond_size < 0:
            raise ValueError(f"max_cond_size must be None or >= 0, got {max_cond_size}")
        self.test = test
        self.alpha = alpha
        self.max_cond_size = max_cond_size

    # -- the run ---------------------------------------------------------

    def fit(
        self, data: TabularData | None = None, knowledge: DomainKnowledge | None = None
    ) -> DiscoveryResult:
        """Run RFCI and return the discovered graph with its provenance."""
        _refuse_soft(knowledge, "RFCI")
        oracle = resolve_test(self.test, data, algorithm=type(self).__name__)
        nodes = sorted(oracle.variables)
        adj, sepsets = stable_skeleton(
            oracle, nodes, knowledge, alpha=self.alpha, max_cond_size=self.max_cond_size
        )
        protected = _protected_pairs(knowledge)
        counts = _Counts()
        triples = self._triple_stage(oracle, adj, sepsets, protected, counts)
        closed, n_v_structures = self._orientation_stage(
            oracle, nodes, adj, sepsets, protected, triples, knowledge, counts
        )
        try:
            final = closed.as_kind(GraphKind.PAG)
            pag_valid = 1.0
        except InvalidGraphError:
            final = closed.as_kind(GraphKind.MIXED)
            pag_valid = 0.0
        if knowledge is not None and not knowledge.is_consistent(final):
            raise ValueError(
                "RFCI produced a graph that violates the domain knowledge; the data and "
                "the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics: dict[str, float] = dict(oracle.diagnostics())
        diagnostics["n_edges"] = float(len(final.edges))
        diagnostics["n_v_structures"] = float(n_v_structures)
        diagnostics["n_stage2_removed"] = float(counts.stage2_removed)
        diagnostics["n_stage3_removed"] = float(counts.stage3_removed)
        diagnostics["n_triple_tests"] = float(counts.triple_tests)
        diagnostics["n_path_tests"] = float(counts.path_tests)
        for label, mark in (
            ("n_arrowheads", Mark.ARROW),
            ("n_tails", Mark.TAIL),
            ("n_circles", Mark.CIRCLE),
        ):
            diagnostics[label] = float(
                sum((e.u_mark is mark) + (e.v_mark is mark) for e in final.edges)
            )
        diagnostics["pag_valid"] = pag_valid
        return DiscoveryResult(
            graph=final,
            method="rfci",
            test=oracle.name,
            alpha=self.alpha,
            n=oracle.n,
            diagnostics=diagnostics,
            sepsets=sepsets,
        )

    # -- stage two: the unshielded-triple rule -----------------------------

    def _triple_stage(
        self,
        oracle: CITest,
        adj: _Adjacency,
        sepsets: _Sepsets,
        protected: frozenset[frozenset[str]],
        counts: _Counts,
    ) -> list[_Triple]:
        """The unshielded-triple rule (Lemma 3.1): trust no triple untested.

        Each unshielded triple ``(x, m, y)`` re-tests its two edges given the
        recorded separating set of ``(x, y)`` minus ``m``. Both dependent:
        the triple is kept for orientation once the worklist empties
        (deferring the arrowheads is what makes the stage order independent,
        the paper's two-list design). Any independence: the edge comes off, a
        minimal separating set is recorded by re-testing subsets of the same
        set in ascending size (the set that exposed the independence need not
        be minimal, and Lemma 3.1's condition needs a minimal one), triples
        the removal destroyed leave both lists, and triples it unshielded
        join the worklist. A triple whose ends were never tested (a
        knowledge-pruned pair records no separating set) is skipped and stays
        unoriented, the same convention FCI's collider phase uses.
        """
        queue: deque[_Triple] = deque(_unshielded_triples(adj))
        queued: set[_Triple] = set(queue)
        kept: list[_Triple] = []
        while queue:
            triple = queue.popleft()
            queued.discard(triple)
            x, m, y = triple
            separator = sepsets.get(frozenset((x, y)))
            if separator is None:
                continue
            conditioning = sorted(separator - {m})
            independent: list[tuple[str, str]] = []
            for u, v in ((x, m), (m, y)):
                if frozenset((u, v)) in protected:
                    continue  # protect: required and confounded pairs are never tested
                counts.triple_tests += 1
                if oracle.independent(u, v, conditioning, alpha=self.alpha):
                    independent.append((u, v))
            if not independent:
                kept.append(triple)
                continue
            for u, v in independent:
                minimal = self._minimal_separator(oracle, u, v, conditioning, counts)
                _delete_edge(adj, sepsets, u, v, minimal, queue, queued, kept)
                counts.stage2_removed += 1
        return kept

    def _minimal_separator(
        self, oracle: CITest, u: str, v: str, base: list[str], counts: _Counts
    ) -> frozenset[str]:
        """The first separating subset of ``base``, swept in ascending size.

        Ascending order makes the winner minimum-cardinality, so minimal,
        which is what the soundness lemmas consume. The full set separated
        the pair a moment ago, so the sweep always lands.
        """
        for size in range(len(base) + 1):
            for subset in combinations(base, size):
                counts.triple_tests += 1
                if oracle.independent(u, v, subset, alpha=self.alpha):
                    return frozenset(subset)
        return frozenset(base)  # unreachable: the full set is the sweep's last subset

    # -- stage three: closure with the guarded discriminating-path rule -----

    def _orientation_stage(
        self,
        oracle: CITest,
        nodes: list[str],
        adj: _Adjacency,
        sepsets: _Sepsets,
        protected: frozenset[frozenset[str]],
        triples: list[_Triple],
        knowledge: DomainKnowledge | None,
        counts: _Counts,
    ) -> tuple[MixedGraph, int]:
        """Stage three: Zhang's rules R1 to R10 with R4 guarded by Lemma 3.2.

        Marks start as circles plus the stage-two v-structures and any
        knowledge seeds, then close under ``fci_closure``. The discriminating
        callback reads a snapshot of the marks the closure started from, so a
        path that only exists through marks set later in the same pass comes
        back unfound; a pass that changed marks but deleted nothing therefore
        reruns from the closed graph with a fresh snapshot, until a pass
        changes nothing, at which point the callback saw exactly the marks
        the closure saw. A pass that found independencies instead fills the
        pending box: those edges come off with their minimal separating sets
        recorded, stage two reruns in full on the smaller adjacency (the
        simplest correct restart; the oracle's cache makes the re-walk
        cheap), marks reset, and the closure starts over. The loop ends when
        a closure pass ends with no pending deletions. The paper accepts that
        this stage is order dependent; every walk here is sorted, so the run
        is deterministic.
        """
        while True:
            marks = _circle_marks(adj)
            n_v_structures = _orient_triples(marks, sepsets, triples)
            _seed_knowledge(marks, knowledge)
            graph = _to_graph(nodes, marks)
            closed, pending = self._close_once(oracle, graph, sepsets, protected, counts)
            while not pending and closed.edges != graph.edges:
                graph = closed
                closed, pending = self._close_once(oracle, graph, sepsets, protected, counts)
            if not pending:
                return closed, n_v_structures
            for pair in sorted(pending, key=sorted):
                u, v = sorted(pair)
                adj[u].discard(v)
                adj[v].discard(u)
                sepsets[pair] = pending[pair]
                counts.stage3_removed += 1
            triples = self._triple_stage(oracle, adj, sepsets, protected, counts)

    def _close_once(
        self,
        oracle: CITest,
        graph: MixedGraph,
        sepsets: _Sepsets,
        protected: frozenset[frozenset[str]],
        counts: _Counts,
    ) -> tuple[MixedGraph, _Pending]:
        """One closure pass; the pending box collects Lemma 3.2 removals."""
        pending: _Pending = {}
        closed = fci_closure(
            graph,
            discriminating=self._lemma_3_2(oracle, graph, sepsets, protected, pending, counts),
        )
        return closed, pending

    def _lemma_3_2(
        self,
        oracle: CITest,
        graph: MixedGraph,
        sepsets: _Sepsets,
        protected: frozenset[frozenset[str]],
        pending: _Pending,
        counts: _Counts,
    ) -> Callable[[str, str, str, str], bool | None]:
        """R4's collider question, guarded by Lemma 3.2's extra tests.

        The closure hands over the ends ``(d, a, b, c)`` it found; the path
        itself is recovered here over the snapshot marks. Every consecutive
        pair on the recovered path must stay dependent given every subset of
        the recorded separating set of ``(d, c)`` minus the pair (the paper
        is explicit that the single full set is not sufficient). The first
        independence found goes into the pending box with its minimal
        separating set, and the path abstains; a path through an edge already
        condemned this pass abstains too, since the pass restarts anyway.
        With every pair dependent the answer is FCI's: ``b`` is a collider
        exactly when it is outside the recorded separating set. Verdicts are
        memoized per pass; the snapshot, the oracle's cache, and the recorded
        sets make the callback a pure function of its arguments within one.
        """
        marks = _mark_maps(graph)
        adj = {n: sorted(marks[n]) for n in marks}
        memo: dict[tuple[str, str, str, str], bool | None] = {}

        def judge(d: str, a: str, b: str, c: str) -> bool | None:
            separator = sepsets.get(frozenset((d, c)))
            if separator is None:
                return None  # the pair was never tested (knowledge-pruned)
            path = _minimal_discriminating_path(marks, adj, d, a, b, c)
            if path is None:
                return None  # not in this snapshot; the refresh loop will re-ask
            pairs = list(pairwise(path))
            if any(frozenset(pair) in pending for pair in pairs):
                return None  # the path runs through an edge already condemned
            for r, q in pairs:
                pair = frozenset((r, q))
                if pair in protected:
                    continue  # protect: never tested or removed
                base = sorted(separator - {r, q})
                for size in range(len(base) + 1):
                    for subset in combinations(base, size):
                        counts.path_tests += 1
                        if oracle.independent(r, q, subset, alpha=self.alpha):
                            pending[pair] = frozenset(subset)
                            return None
            return b not in separator

        def callback(d: str, a: str, b: str, c: str) -> bool | None:
            key = (d, a, b, c)
            if key not in memo:
                memo[key] = judge(d, a, b, c)
            return memo[key]

        return callback


# -- the triple bookkeeping ------------------------------------------------


def _unshielded_triples(adj: _Adjacency) -> list[_Triple]:
    """Every unshielded triple ``(x, m, y)``, canonically ordered, sorted."""
    out: list[_Triple] = []
    for m in sorted(adj):
        for x, y in combinations(sorted(adj[m]), 2):
            if y not in adj[x]:
                out.append((x, m, y))
    return out


def _orient_triples(marks: _Marks, sepsets: _Sepsets, triples: list[_Triple]) -> int:
    """Arrowheads at the middle of every kept triple whose separator excludes it."""
    count = 0
    for x, m, y in sorted(triples):
        if m in sepsets[frozenset((x, y))]:
            continue
        marks[m][x] = Mark.ARROW
        marks[m][y] = Mark.ARROW
        count += 1
    return count


def _delete_edge(
    adj: _Adjacency,
    sepsets: _Sepsets,
    u: str,
    v: str,
    minimal: frozenset[str],
    queue: deque[_Triple],
    queued: set[_Triple],
    kept: list[_Triple],
) -> None:
    """Remove one edge mid-stage-two and keep the triple lists coherent.

    The pair's minimal separating set is recorded, every triple that used the
    edge leaves both lists, and every triple the removal unshielded (the pair
    as ends, a shared neighbour as middle) joins the worklist.
    """
    adj[u].discard(v)
    adj[v].discard(u)
    gone = frozenset((u, v))
    sepsets[gone] = minimal

    def survives(triple: _Triple) -> bool:
        x, m, y = triple
        return frozenset((x, m)) != gone and frozenset((m, y)) != gone

    if any(not survives(t) for t in queue):
        remaining = [t for t in queue if survives(t)]
        queued.intersection_update(remaining)
        queue.clear()
        queue.extend(remaining)
    kept[:] = [t for t in kept if survives(t)]
    lo, hi = sorted((u, v))
    for w in sorted(adj[lo] & adj[hi]):
        triple = (lo, w, hi)
        if triple not in queued:
            queue.append(triple)
            queued.add(triple)


# -- the path search ---------------------------------------------------------


def _minimal_discriminating_path(
    marks: _Marks, adj: dict[str, list[str]], d: str, a: str, b: str, c: str
) -> tuple[str, ...] | None:
    """A minimal-length discriminating path ``<d, ..., a, b, c>`` for ``b``.

    Mirrors the closure's own walk (every vertex between ``d`` and ``b`` must
    be a collider on the path and a parent of ``c``; ``d`` is not adjacent to
    ``c``), but pins both ends to the pair the closure asked about and tracks
    the actual vertices: breadth first over simple paths in sorted order, so
    the first hit is a shortest path (Lemma 3.2 is stated for minimal ones)
    and the search is deterministic. Returns ``None`` when the snapshot has
    no such path.
    """
    if c in marks[d] or marks[a][b] is not Mark.ARROW or not _is_directed(marks, a, c):
        return None
    queue: deque[tuple[str, ...]] = deque([(a,)])
    while queue:
        path = queue.popleft()
        head = path[-1]
        for p in adj[head]:
            if p in (b, c) or p in path:
                continue
            if marks[head][p] is not Mark.ARROW:
                continue  # the vertex before p must collide on the path
            if p == d:
                return (d, *reversed(path), b, c)
            if c not in marks[p]:
                continue  # would close a different discriminating path, not this one
            if _is_directed(marks, p, c) and marks[p][head] is Mark.ARROW:
                queue.append((*path, p))
    return None
