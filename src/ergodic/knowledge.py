# SPDX-License-Identifier: Apache-2.0
"""Domain knowledge: prior constraints on causal structure.

``DomainKnowledge`` records what a user knows before looking at the data, the
edges that must or must not exist, the temporal order of variables, which
variables are roots or sinks, and which pairs share a hidden confounder. A
skeleton restriction (``restrict_to``) caps which pairs may be adjacent at all.
It is an immutable value object built up with small methods, and two knowledge
objects combine with ``&`` (more knowledge means fewer admissible graphs).

Hard constraints are the focus for now. The ``energy`` and ``log_prior`` hooks
are where soft, probabilistic knowledge will attach once the Bayesian work
starts; they raise for the moment so nothing depends on a half-built scoring rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import numpy.typing as npt

    from .graph import MixedGraph


class InconsistentKnowledgeError(ValueError):
    """Domain knowledge contradicts itself (for example a required edge cycle)."""


def _pairs(value: Iterable[str]) -> frozenset[str]:
    return frozenset(value)


@dataclass(frozen=True)
class DomainKnowledge:
    """Prior constraints on a causal graph over named variables."""

    variables: frozenset[str] = field(default_factory=frozenset)
    required: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    forbidden: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    precedes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    roots: frozenset[str] = field(default_factory=frozenset)
    sinks: frozenset[str] = field(default_factory=frozenset)
    confounded: frozenset[frozenset[str]] = field(default_factory=frozenset)
    allowed_adjacencies: frozenset[frozenset[str]] | None = None

    # -- builders ------------------------------------------------------------

    def with_variables(self, *names: str) -> DomainKnowledge:
        return replace(self, variables=self.variables | frozenset(names))

    def require(self, cause: str, effect: str) -> DomainKnowledge:
        """Require a directed edge ``cause -> effect``."""
        return replace(
            self,
            required=self.required | {(cause, effect)},
            variables=self.variables | {cause, effect},
        )

    def forbid(self, cause: str, effect: str) -> DomainKnowledge:
        """Forbid the directed edge ``cause -> effect`` (the reverse stays allowed)."""
        return replace(
            self,
            forbidden=self.forbidden | {(cause, effect)},
            variables=self.variables | {cause, effect},
        )

    def forbid_adjacency(self, a: str, b: str) -> DomainKnowledge:
        """Forbid any edge between two variables, in either direction."""
        return replace(
            self,
            forbidden=self.forbidden | {(a, b), (b, a)},
            variables=self.variables | {a, b},
        )

    def order(self, before: str, after: str) -> DomainKnowledge:
        """Record that ``before`` precedes ``after`` in time, so ``after`` cannot cause it."""
        return replace(
            self,
            precedes=self.precedes | {(before, after)},
            variables=self.variables | {before, after},
        )

    def with_tiers(self, tiers: Sequence[Iterable[str]]) -> DomainKnowledge:
        """Add a tiered order: no variable in a later tier causes one in an earlier tier."""
        groups = [_pairs(t) for t in tiers]
        precedes = set(self.precedes)
        variables = set(self.variables)
        for i, earlier in enumerate(groups):
            variables |= earlier
            for later in groups[i + 1 :]:
                for a in earlier:
                    for b in later:
                        precedes.add((a, b))
        return replace(self, precedes=frozenset(precedes), variables=frozenset(variables))

    def root(self, *names: str) -> DomainKnowledge:
        """Mark variables as roots: no incoming directed edges."""
        return replace(
            self, roots=self.roots | frozenset(names), variables=self.variables | frozenset(names)
        )

    def sink(self, *names: str) -> DomainKnowledge:
        """Mark variables as sinks: no outgoing directed edges."""
        return replace(
            self, sinks=self.sinks | frozenset(names), variables=self.variables | frozenset(names)
        )

    def confound(self, a: str, b: str) -> DomainKnowledge:
        """Record that ``a`` and ``b`` share a hidden common cause."""
        if a == b:
            raise ValueError(f"a confounded pair needs two distinct variables, got {a!r} twice")
        return replace(
            self,
            confounded=self.confounded | {frozenset({a, b})},
            variables=self.variables | {a, b},
        )

    def forbid_between(self, group_a: Iterable[str], group_b: Iterable[str]) -> DomainKnowledge:
        """Forbid every directed edge from ``group_a`` into ``group_b``."""
        a = list(group_a)
        b = list(group_b)
        bans = {(x, y) for x in a for y in b}
        return replace(
            self, forbidden=self.forbidden | bans, variables=self.variables | set(a) | set(b)
        )

    def restrict_to(self, skeleton: MixedGraph | Iterable[tuple[str, str]]) -> DomainKnowledge:
        """Allow adjacencies only on the pairs of ``skeleton``.

        ``skeleton`` is a :class:`~ergodic.graph.MixedGraph` (its edges name
        the pairs, whatever the marks) or an iterable of ``(u, v)`` pairs. A
        pair outside the allow-list is banned in every form: directed either
        way, bidirected, or undirected. Restricting an already restricted
        object keeps the intersection, so a second call only narrows the
        support. An empty iterable allows no adjacencies at all.
        """
        from .graph import MixedGraph

        if isinstance(skeleton, MixedGraph):
            raw: Iterable[tuple[str, str]] = ((e.u, e.v) for e in skeleton.edges)
        else:
            raw = skeleton
        pairs: set[frozenset[str]] = set()
        names: set[str] = set()
        for u, v in raw:
            if u == v:
                raise ValueError(
                    f"a skeleton adjacency needs two distinct variables, got {u!r} twice"
                )
            pairs.add(frozenset({u, v}))
            names.update((u, v))
        allowed = frozenset(pairs)
        if self.allowed_adjacencies is not None:
            allowed &= self.allowed_adjacencies
        return replace(
            self,
            allowed_adjacencies=allowed,
            variables=self.variables | names,
        )

    def __and__(self, other: DomainKnowledge) -> DomainKnowledge:
        """Combine two knowledge objects by taking the union of their constraints.

        Skeleton allow-lists intersect: an adjacency survives only when both
        sides allow it, and ``None`` (unrestricted) defers to the other side.
        """
        if self.allowed_adjacencies is None:
            allowed = other.allowed_adjacencies
        elif other.allowed_adjacencies is None:
            allowed = self.allowed_adjacencies
        else:
            allowed = self.allowed_adjacencies & other.allowed_adjacencies
        return DomainKnowledge(
            variables=self.variables | other.variables,
            required=self.required | other.required,
            forbidden=self.forbidden | other.forbidden,
            precedes=self.precedes | other.precedes,
            roots=self.roots | other.roots,
            sinks=self.sinks | other.sinks,
            confounded=self.confounded | other.confounded,
            allowed_adjacencies=allowed,
        )

    combine = __and__

    # -- validation ----------------------------------------------------------

    def conflicts(self) -> list[str]:
        """Return a list of internal contradictions, empty when consistent."""
        out: list[str] = []
        for u, v in sorted(self.required & self.forbidden):
            out.append(f"edge {u}->{v} is both required and forbidden")
        for u, v in sorted(self.required):
            if (v, u) in self.precedes:
                out.append(f"required edge {u}->{v} contradicts order {v} before {u}")
            if v in self.roots:
                out.append(f"required edge {u}->{v} contradicts root {v}")
            if u in self.sinks:
                out.append(f"required edge {u}->{v} contradicts sink {u}")
        if self.allowed_adjacencies is not None:
            for u, v in sorted(self.required):
                if frozenset({u, v}) not in self.allowed_adjacencies:
                    out.append(f"required edge {u}->{v} lies outside the skeleton restriction")
            for pair in sorted(self.confounded, key=lambda p: sorted(p)):
                if pair not in self.allowed_adjacencies:
                    a, b = sorted(pair)
                    out.append(f"confounded pair {a},{b} lies outside the skeleton restriction")
        required_cycles = self._cycle_conflicts(self.required, "required edges")
        precedence_cycles = self._cycle_conflicts(self.precedes, "precedence")
        out.extend(required_cycles)
        out.extend(precedence_cycles)
        if not required_cycles and not precedence_cycles:
            # A required edge can close a loop through the precedence order even
            # when each relation is acyclic on its own, so check their union too.
            out.extend(
                self._cycle_conflicts(
                    self.required | self.precedes, "required edges and precedence together"
                )
            )
        return out

    @staticmethod
    def _cycle_conflicts(pairs: frozenset[tuple[str, str]], label: str) -> list[str]:
        # Kahn's algorithm, iterative so long chains do not hit the recursion limit.
        forward: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}
        for a, b in pairs:
            forward.setdefault(a, set()).add(b)
            indegree.setdefault(a, 0)
            indegree[b] = indegree.get(b, 0) + 1
        ready = [n for n, degree in indegree.items() if degree == 0]
        removed = 0
        while ready:
            node = ready.pop()
            removed += 1
            for nxt in forward.get(node, ()):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
        return [] if removed == len(indegree) else [f"{label} contain a cycle"]

    def validate(self) -> None:
        """Raise :class:`InconsistentKnowledgeError` if the knowledge contradicts itself."""
        found = self.conflicts()
        if found:
            raise InconsistentKnowledgeError("; ".join(found))

    # -- projections used by algorithms --------------------------------------

    def _precedence_closure(self) -> frozenset[tuple[str, str]]:
        """The transitive closure of ``precedes``, walked with an explicit stack."""
        forward: dict[str, set[str]] = {}
        for a, b in self.precedes:
            forward.setdefault(a, set()).add(b)
        closed: set[tuple[str, str]] = set()
        for start in forward:
            seen: set[str] = set()
            stack = list(forward[start])
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(forward.get(node, ()))
            seen.discard(start)
            closed.update((start, later) for later in seen)
        return frozenset(closed)

    def _permits(self, cause: str, effect: str, precedes: frozenset[tuple[str, str]]) -> bool:
        if (
            self.allowed_adjacencies is not None
            and frozenset({cause, effect}) not in self.allowed_adjacencies
        ):
            return False
        if (cause, effect) in self.forbidden:
            return False
        if (effect, cause) in precedes:
            return False
        return not (effect in self.roots or cause in self.sinks)

    def permits_edge(self, cause: str, effect: str) -> bool:
        """Return ``True`` if a directed edge ``cause -> effect`` is allowed.

        Precedence is read transitively: with ``A`` before ``B`` and ``B``
        before ``C``, an edge ``C -> A`` is not permitted. Under a skeleton
        restriction, a pair outside the allow-list fails in both directions.
        """
        return self._permits(cause, effect, self._precedence_closure())

    def requires_edge(self, cause: str, effect: str) -> bool:
        return (cause, effect) in self.required

    def precedence(self) -> frozenset[tuple[str, str]]:
        """Every ``(before, after)`` pair, with the order read transitively.

        The projection discovery reads to orient: ``(a, b)`` here means ``b``
        cannot be an ancestor of ``a``, which in a latent-variable search
        (FCI) pins an arrowhead at ``b`` on any surviving adjacency.
        """
        return self._precedence_closure()

    def violations(self, graph: MixedGraph) -> list[str]:
        """Return the ways ``graph`` breaks this knowledge, empty when consistent."""
        out: list[str] = []
        for u, v in sorted(self.required):
            edge = graph.get_edge(u, v)
            if edge is None or edge.directed_pair() != (u, v):
                out.append(f"missing required edge {u}->{v}")
        # A pair forbidden in both directions is an adjacency ban, so any edge
        # between the two (bidirected and undirected included) violates it.
        reported: set[frozenset[str]] = set()
        for u, v in sorted(self.forbidden):
            edge = graph.get_edge(u, v)
            if edge is None:
                continue
            if (v, u) in self.forbidden:
                pair = frozenset((u, v))
                if pair not in reported:
                    reported.add(pair)
                    out.append(f"present edge between {u} and {v}, forbidden in both directions")
            elif edge.directed_pair() == (u, v):
                out.append(f"present forbidden edge {u}->{v}")
        if self.allowed_adjacencies is not None:
            for edge in graph.edge_list():
                if frozenset({edge.u, edge.v}) not in self.allowed_adjacencies:
                    a, b = sorted((edge.u, edge.v))
                    out.append(
                        f"present edge between {a} and {b}, outside the skeleton restriction"
                    )
        for before, after in sorted(self._precedence_closure()):
            if (
                before in graph.nodes
                and after in graph.nodes
                and before in graph.descendants(after)
            ):
                out.append(f"order violated: {after} is an ancestor of {before}")
        for r in sorted(self.roots):
            if r in graph.nodes and graph.parents(r):
                out.append(f"root {r} has parents")
        for s in sorted(self.sinks):
            if s in graph.nodes and graph.children(s):
                out.append(f"sink {s} has children")
        for pair in self.confounded:
            a, b = sorted(pair)
            edge = graph.get_edge(a, b)
            if edge is None or not edge.is_bidirected:
                out.append(f"confounded pair {a},{b} lacks a bidirected edge")
        return out

    def is_consistent(self, graph: MixedGraph) -> bool:
        """Return ``True`` if ``graph`` satisfies every constraint."""
        return not self.violations(graph)

    def apply_to(self, graph: MixedGraph) -> MixedGraph:
        """Enforce the hard constraints on ``graph``: drop forbidden, add required edges.

        A pair forbidden in both directions loses its edge whatever the marks; a
        single-direction ban only drops the matching directed edge. Under a
        skeleton restriction, every edge on a pair outside the allow-list is
        dropped too. Edges left undirected or with a circle are oriented when
        only one direction stays permitted. The result is returned as a new
        graph.
        """
        out = graph
        if self.allowed_adjacencies is not None:
            for present in list(out.edges):
                if frozenset({present.u, present.v}) not in self.allowed_adjacencies:
                    out = out.remove_edge(present.u, present.v)
        for u, v in self.forbidden:
            edge = out.get_edge(u, v)
            if edge is None:
                continue
            if (v, u) in self.forbidden or edge.directed_pair() == (u, v):
                out = out.remove_edge(u, v)
        for u, v in self.required:
            out = out.orient(u, v)
        precedes = self._precedence_closure()
        for edge in list(out.edges):
            if edge.is_directed:
                continue
            forward_ok = self._permits(edge.u, edge.v, precedes)
            backward_ok = self._permits(edge.v, edge.u, precedes)
            if forward_ok and not backward_ok:
                out = out.orient(edge.u, edge.v)
            elif backward_ok and not forward_ok:
                out = out.orient(edge.v, edge.u)
        return out

    def edge_mask(
        self, variables: Sequence[str] | None = None
    ) -> tuple[npt.NDArray[np.bool_], list[str]]:
        """Return a boolean matrix of permitted directed edges and its variable order.

        ``mask[i, j]`` is ``True`` when an edge from variable ``i`` to variable
        ``j`` is allowed. The diagonal is ``False``, and a skeleton restriction
        zeroes both directions of every pair outside its allow-list. Useful as
        a search mask for score-based and continuous-optimization discovery.
        """
        import numpy as np

        labels = list(variables) if variables is not None else sorted(self.variables)
        size = len(labels)
        precedes = self._precedence_closure()
        mask = np.zeros((size, size), dtype=bool)
        for i, cause in enumerate(labels):
            for j, effect in enumerate(labels):
                if i != j and self._permits(cause, effect, precedes):
                    mask[i, j] = True
        return mask, labels

    # -- soft knowledge, not built yet ---------------------------------------

    def energy(self, graph: MixedGraph) -> float:
        """Penalty for ``graph`` under soft knowledge. Not implemented; raises."""
        raise NotImplementedError(
            "soft/energy-based knowledge is not implemented; use the hard-constraint "
            "methods (require, forbid, tiers, and the rest)."
        )

    def log_prior(self, graph: MixedGraph) -> float:
        """Log prior of ``graph`` under soft knowledge. Not implemented; raises."""
        raise NotImplementedError(
            "soft/probabilistic knowledge is not implemented; use the hard-constraint "
            "methods (require, forbid, tiers, and the rest)."
        )

    def __repr__(self) -> str:
        restricted = (
            ""
            if self.allowed_adjacencies is None
            else f", allowed_adjacencies={len(self.allowed_adjacencies)}"
        )
        return (
            "DomainKnowledge("
            f"vars={len(self.variables)}, required={len(self.required)}, "
            f"forbidden={len(self.forbidden)}, precedes={len(self.precedes)}, "
            f"roots={len(self.roots)}, sinks={len(self.sinks)}, "
            f"confounded={len(self.confounded)}{restricted})"
        )
