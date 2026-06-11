# SPDX-License-Identifier: Apache-2.0
"""FCI: the constraint-based search that tolerates latent confounders.

PC assumes every common cause is measured; FCI (Spirtes, Glymour, and
Scheines) drops that assumption and returns a partial ancestral graph, whose
marks survive hidden confounding: an arrowhead at ``V`` says ``V`` does not
cause the other endpoint in any member of the class, a tail says it does in
every member, a circle says the data cannot tell. The run is PC's stable
skeleton, then the possible-d-sep phase that PC has no analogue for (with
latents, two adjacent-looking nodes can be separated only by conditioning
outside their neighbourhoods, so candidate sets widen to Possible-D-Sep),
then re-orientation from the final separating sets and Zhang's (2008) rule
closure, all ten rules, which is complete with latent confounders and
selection bias both. Selection needs nothing from the algorithm: data sampled
under selection simply answers every test conditional on it, and the
selection rules R5 to R7 turn what that implies into undirected and
circle-tail marks. An undirected edge in the output says both endpoints
cause the selection in every member of the class.

Domain knowledge translates through ancestry, not edge direction. Prune: a
pair forbidden in both directions (an adjacency ban) is never tested.
Protect: required and confounded pairs are never tested or removed. Orient:
``require(u, v)`` seeds ``u -> v``; ``confound(a, b)`` seeds ``a <-> b`` (the
data model reads a confounded pair as a pure hidden common cause); a tier or
``order(a, b)`` seeds only an arrowhead at ``b``, because a later variable
cannot be an ancestor of an earlier one while confounding stays possible.
Directional forbids, roots, and sinks fix no single mark in a latent world
(forbidding ``x -> y`` does not forbid ``x <-> y``), so they only validate.
Seeds never overwrite a mark the data compelled; a conflict surfaces when the
final graph is checked against the knowledge. With orientation knowledge the
output usually carries marks no unconstrained FCI run would produce, so it is
not a complete PAG: the result then keeps the ``mixed`` face and
``diagnostics["pag_valid"]`` is ``0.0``, the same convention PC uses for
``cpdag_valid``.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

from ...graph import Edge, GraphKind, InvalidGraphError, Mark, MixedGraph
from ...graph._pag import fci_closure
from .._registry import resolve_test
from .._result import DiscoveryResult
from ._skeleton import _protected_pairs, stable_skeleton

if TYPE_CHECKING:
    from ...data import TabularData
    from ...knowledge import DomainKnowledge
    from .._ci import CITest

_Marks = dict[str, dict[str, Mark]]
_Adjacency = dict[str, set[str]]
_Sepsets = dict[frozenset[str], frozenset[str]]


class FCI:
    """FCI over a conditional-independence test, returning a PAG.

    Args:
        test: A :class:`CITest` instance (bound or unbound, including
            :class:`DSeparationOracle`), or a name: ``"auto"`` (pick by the
            data schema), ``"linear"``/``"fisher_z"``, ``"discrete"``/
            ``"g_sq"``, ``"mixed"``/``"dg"``, ``"gcm"``.
        alpha: Significance level for the independence decisions.
        max_cond_size: Cap on the conditioning-set size, applied to the
            skeleton phase and to the possible-d-sep subsets alike; ``None``
            for no cap. Possible-d-sep sets can span most of the graph, so
            the cap is the honest scaling knob.

    ``fit(data, knowledge)`` returns a :class:`DiscoveryResult` whose graph
    carries the PAG face when the result is a valid complete PAG and the
    ``mixed`` face otherwise (noisy tests or knowledge-seeded marks). With a
    perfect oracle and no knowledge the output is exactly the canonical PAG:
    ``pag_of_mag(mag_of_dag(truth))``.
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
        """Run FCI and return the discovered graph with its provenance."""
        oracle = resolve_test(self.test, data)
        nodes = sorted(oracle.variables)
        adj, sepsets = stable_skeleton(
            oracle, nodes, knowledge, alpha=self.alpha, max_cond_size=self.max_cond_size
        )
        protected = _protected_pairs(knowledge)

        marks = _circle_marks(adj)
        _orient_colliders(marks, adj, sepsets)
        n_pds_removed = self._pds_prune(oracle, nodes, adj, marks, sepsets, protected)

        marks = _circle_marks(adj)
        n_v_structures = _orient_colliders(marks, adj, sepsets)
        _seed_knowledge(marks, knowledge)

        def from_sepsets(d: str, a: str, b: str, c: str) -> bool | None:
            separator = sepsets.get(frozenset((d, c)))
            if separator is None:
                return None  # the pair was never tested (knowledge-pruned)
            return b not in separator

        closed = fci_closure(_to_graph(nodes, marks), discriminating=from_sepsets)
        try:
            final = closed.as_kind(GraphKind.PAG)
            pag_valid = 1.0
        except InvalidGraphError:
            final = closed.as_kind(GraphKind.MIXED)
            pag_valid = 0.0
        if knowledge is not None and not knowledge.is_consistent(final):
            raise ValueError(
                "FCI produced a graph that violates the domain knowledge; the data and "
                "the constraints disagree (knowledge.violations(graph) lists how)"
            )
        diagnostics: dict[str, float] = dict(oracle.diagnostics())
        diagnostics["n_edges"] = float(len(final.edges))
        diagnostics["n_v_structures"] = float(n_v_structures)
        diagnostics["n_pds_removed"] = float(n_pds_removed)
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
            method="fci",
            test=oracle.name,
            alpha=self.alpha,
            n=oracle.n,
            diagnostics=diagnostics,
            sepsets=sepsets,
        )

    def _pds_prune(
        self,
        oracle: CITest,
        nodes: list[str],
        adj: _Adjacency,
        marks: _Marks,
        sepsets: _Sepsets,
        protected: frozenset[frozenset[str]],
    ) -> int:
        """The FCI extension of the skeleton: re-test against Possible-D-Sep sets.

        The sets are computed once, on the adjacencies and v-structure marks
        as they stand after the first orientation, then every surviving pair
        is re-tested against subsets of pds(x) from each side in turn.
        Removals record their separating sets like any skeleton removal.
        """
        pds = {x: _possible_d_sep(x, adj, marks) for x in nodes}
        removed = 0
        for x in nodes:
            for y in sorted(adj[x]):
                if y not in adj[x]:
                    continue  # removed from the other side earlier
                if frozenset((x, y)) in protected:
                    continue
                candidates = sorted(pds[x] - {x, y})
                cap = len(candidates) if self.max_cond_size is None else self.max_cond_size
                found = False
                for size in range(min(cap, len(candidates)) + 1):
                    for subset in combinations(candidates, size):
                        if oracle.independent(x, y, subset, alpha=self.alpha):
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sepsets[frozenset((x, y))] = frozenset(subset)
                            removed += 1
                            found = True
                            break
                    if found:
                        break
        return removed


def _circle_marks(adj: _Adjacency) -> _Marks:
    return {x: dict.fromkeys(adj[x], Mark.CIRCLE) for x in adj}


def _orient_colliders(marks: _Marks, adj: _Adjacency, sepsets: _Sepsets) -> int:
    """Arrowheads at every unshielded collider the separating sets compel."""
    count = 0
    for z in sorted(adj):
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue  # shielded
            separator = sepsets.get(frozenset((x, y)))
            if separator is None or z in separator:
                continue
            marks[z][x] = Mark.ARROW
            marks[z][y] = Mark.ARROW
            count += 1
    return count


def _possible_d_sep(x: str, adj: _Adjacency, marks: _Marks) -> frozenset[str]:
    """Possible-D-Sep(x): reachable along paths of colliders or triangles.

    A vertex is reachable when every interior vertex of some path to it is a
    collider on that path (arrowheads at it on both path edges) or sits in a
    triangle with its path neighbours. Reachability runs over (previous,
    current) states, which is exactly the granularity the condition needs.
    """
    out: set[str] = set()
    seen: set[tuple[str, str]] = set()
    queue: list[tuple[str, str]] = []
    for nbr in sorted(adj[x]):
        out.add(nbr)
        seen.add((x, nbr))
        queue.append((x, nbr))
    while queue:
        prev, cur = queue.pop(0)
        for nxt in sorted(adj[cur]):
            if nxt in (prev, x):
                continue
            collider = marks[cur][prev] is Mark.ARROW and marks[cur][nxt] is Mark.ARROW
            triangle = nxt in adj[prev]
            if not (collider or triangle):
                continue
            state = (cur, nxt)
            if state not in seen:
                seen.add(state)
                out.add(nxt)
                queue.append(state)
    return frozenset(out)


def _seed_knowledge(marks: _Marks, knowledge: DomainKnowledge | None) -> None:
    """Write the knowledge-compelled marks over circles, never over data marks."""
    if knowledge is None:
        return

    def seed(at: str, other: str, mark: Mark) -> None:
        if at in marks and other in marks[at] and marks[at][other] is Mark.CIRCLE:
            marks[at][other] = mark

    for u, v in sorted(knowledge.required):
        seed(u, v, Mark.TAIL)
        seed(v, u, Mark.ARROW)
    for pair in sorted(knowledge.confounded, key=sorted):
        a, b = sorted(pair)
        seed(a, b, Mark.ARROW)
        seed(b, a, Mark.ARROW)
    for before, after in sorted(knowledge.precedence()):
        seed(after, before, Mark.ARROW)


def _to_graph(nodes: list[str], marks: _Marks) -> MixedGraph:
    edges = frozenset(
        Edge.make(u, marks[u][v], v, marks[v][u]) for u in marks for v in marks[u] if u < v
    )
    return MixedGraph(nodes=frozenset(nodes), edges=edges, kind=GraphKind.MIXED)
