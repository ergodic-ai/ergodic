# SPDX-License-Identifier: Apache-2.0
"""Domain knowledge: prior constraints on causal structure.

``DomainKnowledge`` records what a user knows before looking at the data, the
edges that must or must not exist, the temporal order of variables, which
variables are roots or sinks, and which pairs share a hidden confounder. It is an
immutable value object built up with small methods, and two knowledge objects
combine with ``&`` (more knowledge means fewer admissible graphs).

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

    def __and__(self, other: DomainKnowledge) -> DomainKnowledge:
        """Combine two knowledge objects by taking the union of their constraints."""
        return DomainKnowledge(
            variables=self.variables | other.variables,
            required=self.required | other.required,
            forbidden=self.forbidden | other.forbidden,
            precedes=self.precedes | other.precedes,
            roots=self.roots | other.roots,
            sinks=self.sinks | other.sinks,
            confounded=self.confounded | other.confounded,
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
        out.extend(self._cycle_conflicts(self.required, "required edges"))
        out.extend(self._cycle_conflicts(self.precedes, "precedence"))
        return out

    @staticmethod
    def _cycle_conflicts(pairs: frozenset[tuple[str, str]], label: str) -> list[str]:
        forward: dict[str, set[str]] = {}
        for a, b in pairs:
            forward.setdefault(a, set()).add(b)
        color: dict[str, int] = {}
        cyclic = False

        def visit(node: str) -> bool:
            color[node] = 1
            for nxt in forward.get(node, ()):
                state = color.get(nxt, 0)
                if state == 1 or (state == 0 and not visit(nxt)):
                    return False
            color[node] = 2
            return True

        for start in list(forward):
            if color.get(start, 0) == 0 and not visit(start):
                cyclic = True
                break
        return [f"{label} contain a cycle"] if cyclic else []

    def validate(self) -> None:
        """Raise :class:`InconsistentKnowledgeError` if the knowledge contradicts itself."""
        found = self.conflicts()
        if found:
            raise InconsistentKnowledgeError("; ".join(found))

    # -- projections used by algorithms --------------------------------------

    def permits_edge(self, cause: str, effect: str) -> bool:
        """Return ``True`` if a directed edge ``cause -> effect`` is allowed."""
        if (cause, effect) in self.forbidden:
            return False
        if (effect, cause) in self.precedes:
            return False
        return not (effect in self.roots or cause in self.sinks)

    def requires_edge(self, cause: str, effect: str) -> bool:
        return (cause, effect) in self.required

    def violations(self, graph: MixedGraph) -> list[str]:
        """Return the ways ``graph`` breaks this knowledge, empty when consistent."""
        out: list[str] = []
        for u, v in sorted(self.required):
            edge = graph.get_edge(u, v)
            if edge is None or edge.directed_pair() != (u, v):
                out.append(f"missing required edge {u}->{v}")
        for u, v in sorted(self.forbidden):
            edge = graph.get_edge(u, v)
            if edge is not None and edge.directed_pair() == (u, v):
                out.append(f"present forbidden edge {u}->{v}")
        for before, after in sorted(self.precedes):
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

        Edges left undirected or with a circle are oriented when only one
        direction stays permitted. The result is returned as a new graph.
        """
        out = graph
        for u, v in self.forbidden:
            edge = out.get_edge(u, v)
            if edge is not None and edge.directed_pair() == (u, v):
                out = out.remove_edge(u, v)
        for u, v in self.required:
            out = out.orient(u, v)
        for edge in list(out.edges):
            if edge.is_directed:
                continue
            forward_ok = self.permits_edge(edge.u, edge.v)
            backward_ok = self.permits_edge(edge.v, edge.u)
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
        ``j`` is allowed. The diagonal is ``False``. Useful as a search mask for
        score-based and continuous-optimization discovery.
        """
        import numpy as np

        labels = list(variables) if variables is not None else sorted(self.variables)
        size = len(labels)
        mask = np.zeros((size, size), dtype=bool)
        for i, cause in enumerate(labels):
            for j, effect in enumerate(labels):
                if i != j and self.permits_edge(cause, effect):
                    mask[i, j] = True
        return mask, labels

    # -- soft knowledge, not built yet ---------------------------------------

    def energy(self, graph: MixedGraph) -> float:
        """Penalty for ``graph`` under soft knowledge. Planned for the Bayesian phase."""
        raise NotImplementedError(
            "soft/energy-based knowledge is not implemented yet; it is planned for "
            "the Bayesian discovery work. Use the hard-constraint methods for now."
        )

    def log_prior(self, graph: MixedGraph) -> float:
        """Log prior of ``graph`` under soft knowledge. Planned for the Bayesian phase."""
        raise NotImplementedError(
            "soft/probabilistic knowledge is not implemented yet; it is planned for "
            "the Bayesian discovery work. Use the hard-constraint methods for now."
        )

    def __repr__(self) -> str:
        return (
            "DomainKnowledge("
            f"vars={len(self.variables)}, required={len(self.required)}, "
            f"forbidden={len(self.forbidden)}, precedes={len(self.precedes)}, "
            f"roots={len(self.roots)}, sinks={len(self.sinks)}, "
            f"confounded={len(self.confounded)})"
        )
