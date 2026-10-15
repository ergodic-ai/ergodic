# SPDX-License-Identifier: Apache-2.0
"""Domain knowledge: prior constraints on causal structure.

``DomainKnowledge`` records what a user knows before looking at the data, the
edges that must or must not exist, the temporal order of variables, which
variables are roots or sinks, and which pairs share a hidden confounder. A
skeleton restriction (``restrict_to``) caps which pairs may be adjacent at all.
It is an immutable value object built up with small methods, and two knowledge
objects combine with ``&`` (more knowledge means fewer admissible graphs).

Every builder but ``confound`` also takes ``confidence=``. Left out (or 1), the
statement is a hard constraint, stored exactly as before. A value in ``[0.5, 1)``
makes it soft: a statement you believe with that probability, which
:func:`ergodic.decode` weighs against a model's edge probabilities instead of
enforcing. Soft statements live in :attr:`DomainKnowledge.soft`; every
constraint projection (``conflicts``, ``violations``, ``permits_edge``,
``edge_mask``, ``apply_to``) reads the hard ones only. :meth:`DomainKnowledge.log_odds`,
:meth:`DomainKnowledge.log_prior` and :meth:`DomainKnowledge.energy` read the soft
ones.

Each statement is about pairs of variables. On one pair there are three
exclusive states (no edge, ``a -> b``, ``b -> a``), and a soft statement with
confidence ``c`` multiplies the chance of each state by ``c`` when the state makes
it hold and by ``1 - c`` when it makes it fail. Against "no edge" that is a
likelihood ratio on each directed state: ``c / (1 - c)`` for a state that makes
the statement hold where "no edge" fails it, ``(1 - c) / c`` for the reverse, and
1 otherwise. Statements multiply. A soft order (``order``, tiers) is read pair by
pair and is not closed transitively, as a hard one is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import numpy.typing as npt

    from .graph import MixedGraph


class InconsistentKnowledgeError(ValueError):
    """Domain knowledge contradicts itself (for example a required edge cycle)."""


def _pairs(value: Iterable[str]) -> frozenset[str]:
    return frozenset(value)


# The three states of one pair (a, b): no edge, a -> b, b -> a. A rule says in which
# of them a statement holds on that pair.
PairRule = tuple[str, str, tuple[bool, bool, bool]]
_REQUIRE = (False, True, False)
_FORBID = (True, False, True)
_NO_EDGE = (True, False, False)
_BEFORE = (True, True, False)

_KIND_ORDER = (
    "require",
    "forbid",
    "forbid_adjacency",
    "forbid_between",
    "order",
    "tiers",
    "root",
    "sink",
    "restrict_to",
    "confound",
)


def _confidence(value: Any) -> float | None:
    """``None`` for a hard statement, otherwise the checked soft confidence."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("confidence must be a number in [0.5, 1], not a bool")
    if not isinstance(value, Real):
        raise TypeError(f"confidence must be a number in [0.5, 1], got {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or not 0.5 <= number <= 1.0:
        raise ValueError(
            f"confidence must be in [0.5, 1], got {value!r}; for a statement you believe "
            "less than even odds, state the opposite instead"
        )
    return None if number == 1.0 else number


def _names(group: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(group)))


@dataclass(frozen=True)
class KnowledgeStatement:
    """One knowledge statement, as a builder recorded it.

    ``kind`` names the builder: ``"require"``, ``"forbid"``,
    ``"forbid_adjacency"``, ``"order"`` and ``"confound"`` take two names in
    ``args``, as given; ``"forbid_between"`` two sorted groups; ``"tiers"`` the
    tiers in order, each sorted; ``"root"`` and ``"sink"`` one name (a call naming
    several makes one statement per name); ``"restrict_to"`` the sorted skeleton
    pairs. ``confidence`` is ``None`` for a hard statement and in ``[0.5, 1)`` for
    a soft one.

    Attributes:
        kind: The builder that made the statement.
        args: Its arguments, as above.
        confidence: ``None`` (hard) or the soft statement's confidence.
    """

    kind: str
    args: tuple[Any, ...]
    confidence: float | None = None

    @property
    def is_soft(self) -> bool:
        """Whether the statement is soft (it has a confidence below 1)."""
        return self.confidence is not None

    @property
    def variables(self) -> frozenset[str]:
        """Every variable the statement names."""
        if self.kind in ("forbid_between", "tiers"):
            return frozenset(name for group in self.args for name in group)
        if self.kind == "restrict_to":
            return frozenset(name for pair in self.args for name in pair)
        return frozenset(self.args)

    def __str__(self) -> str:
        args = self.args
        if self.kind == "require":
            return f"{args[0]} -> {args[1]}"
        if self.kind == "forbid":
            return f"no {args[0]} -> {args[1]}"
        if self.kind == "forbid_adjacency":
            return f"no edge between {args[0]} and {args[1]}"
        if self.kind == "forbid_between":
            return f"no edge from [{', '.join(args[0])}] into [{', '.join(args[1])}]"
        if self.kind == "order":
            return f"{args[0]} before {args[1]}"
        if self.kind == "tiers":
            return "tiers " + " < ".join(f"[{', '.join(tier)}]" for tier in args)
        if self.kind == "root":
            return f"{args[0]} is a root"
        if self.kind == "sink":
            return f"{args[0]} is a sink"
        if self.kind == "restrict_to":
            return f"edges only on {len(args)} skeleton pair{'' if len(args) == 1 else 's'}"
        return f"{args[0]} <-> {args[1]}"


def _statement_key(statement: KnowledgeStatement) -> tuple[int, str, float]:
    return (
        _KIND_ORDER.index(statement.kind),
        repr(statement.args),
        -1.0 if statement.confidence is None else statement.confidence,
    )


def _pair_rules(statement: KnowledgeStatement, names: Sequence[str]) -> list[PairRule]:
    """The pairs ``statement`` speaks about, each with the states that satisfy it.

    ``names`` is the universe for statements about every other variable (root,
    sink) or every pair (restrict_to). A confounder speaks about no edge state.
    """
    kind, args = statement.kind, statement.args
    rules: list[PairRule] = []
    if kind == "require":
        rules = [(args[0], args[1], _REQUIRE)]
    elif kind == "forbid":
        rules = [(args[0], args[1], _FORBID)]
    elif kind == "forbid_adjacency":
        rules = [(args[0], args[1], _NO_EDGE)]
    elif kind == "order":
        rules = [(args[0], args[1], _BEFORE)]
    elif kind == "forbid_between":
        rules = [(x, y, _FORBID) for x in args[0] for y in args[1]]
    elif kind == "tiers":
        rules = [
            (x, y, _BEFORE)
            for i, earlier in enumerate(args)
            for later in args[i + 1 :]
            for x in earlier
            for y in later
        ]
    elif kind == "root":
        rules = [(v, args[0], _FORBID) for v in names if v != args[0]]
    elif kind == "sink":
        rules = [(args[0], v, _FORBID) for v in names if v != args[0]]
    elif kind == "restrict_to":
        allowed = {frozenset(pair) for pair in args}
        ordered = list(names)
        rules = [
            (u, v, _NO_EDGE)
            for i, u in enumerate(ordered)
            for v in ordered[i + 1 :]
            if frozenset((u, v)) not in allowed
        ]
    unique: list[PairRule] = []
    seen: set[tuple[str, str]] = set()
    for a, b, holds in rules:
        if a != b and (a, b) not in seen:
            seen.add((a, b))
            unique.append((a, b, holds))
    return unique


@dataclass(frozen=True)
class DomainKnowledge:
    """Prior constraints on a causal graph over named variables.

    The hard constraints live in the set fields (``required``, ``forbidden``,
    ``precedes``, ``roots``, ``sinks``, ``confounded``, ``allowed_adjacencies``),
    the soft statements in ``soft``. ``stated`` keeps the hard statements as the
    builders recorded them, for :meth:`statements`; it takes no part in equality,
    hashing or the repr.
    """

    variables: frozenset[str] = field(default_factory=frozenset)
    required: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    forbidden: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    precedes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    roots: frozenset[str] = field(default_factory=frozenset)
    sinks: frozenset[str] = field(default_factory=frozenset)
    confounded: frozenset[frozenset[str]] = field(default_factory=frozenset)
    allowed_adjacencies: frozenset[frozenset[str]] | None = None
    soft: frozenset[KnowledgeStatement] = field(default_factory=frozenset)
    stated: tuple[KnowledgeStatement, ...] = field(default=(), compare=False, repr=False)

    # -- builders ------------------------------------------------------------

    def _hard(self, statement: KnowledgeStatement, **changes: Any) -> DomainKnowledge:
        return replace(
            self,
            variables=self.variables | statement.variables,
            stated=(*self.stated, statement),
            **changes,
        )

    def _soft(self, statement: KnowledgeStatement) -> DomainKnowledge:
        if statement.kind in ("require", "forbid", "forbid_adjacency", "order"):
            a, b = statement.args
            if a == b:
                raise ValueError(f"a soft {statement.kind} needs two distinct variables")
        return replace(
            self,
            soft=self.soft | {statement},
            variables=self.variables | statement.variables,
        )

    def with_variables(self, *names: str) -> DomainKnowledge:
        """Return knowledge that also lists ``names`` in :attr:`variables`.

        Every builder records the names it mentions there; this adds names without
        adding a constraint.
        """
        return replace(self, variables=self.variables | frozenset(names))

    def require(
        self, cause: str, effect: str, *, confidence: float | None = None
    ) -> DomainKnowledge:
        """Require a directed edge ``cause -> effect``.

        Args:
            cause: The edge's source.
            effect: The edge's target.
            confidence: ``None`` or 1 for a hard constraint; a value in
                ``[0.5, 1)`` makes it soft (0.5 changes nothing).
        """
        statement = KnowledgeStatement("require", (cause, effect), _confidence(confidence))
        if statement.is_soft:
            return self._soft(statement)
        return self._hard(statement, required=self.required | {(cause, effect)})

    def forbid(
        self, cause: str, effect: str, *, confidence: float | None = None
    ) -> DomainKnowledge:
        """Forbid the directed edge ``cause -> effect`` (the reverse stays allowed).

        ``confidence`` makes it soft, as for :meth:`require`.
        """
        statement = KnowledgeStatement("forbid", (cause, effect), _confidence(confidence))
        if statement.is_soft:
            return self._soft(statement)
        return self._hard(statement, forbidden=self.forbidden | {(cause, effect)})

    def forbid_adjacency(
        self, a: str, b: str, *, confidence: float | None = None
    ) -> DomainKnowledge:
        """Forbid any edge between two variables, in either direction.

        ``confidence`` makes it soft, as for :meth:`require`; a soft ban weighs the
        two directed edges (decoders see no other edge type).
        """
        statement = KnowledgeStatement("forbid_adjacency", (a, b), _confidence(confidence))
        if statement.is_soft:
            return self._soft(statement)
        return self._hard(statement, forbidden=self.forbidden | {(a, b), (b, a)})

    def order(self, before: str, after: str, *, confidence: float | None = None) -> DomainKnowledge:
        """Record that ``before`` precedes ``after`` in time, so ``after`` cannot cause it.

        ``confidence`` makes it soft, as for :meth:`require`. A soft order weighs
        only the direct edge ``after -> before`` and is not closed transitively.
        """
        statement = KnowledgeStatement("order", (before, after), _confidence(confidence))
        if statement.is_soft:
            return self._soft(statement)
        return self._hard(statement, precedes=self.precedes | {(before, after)})

    def with_tiers(
        self, tiers: Sequence[Iterable[str]], *, confidence: float | None = None
    ) -> DomainKnowledge:
        """Add a tiered order: no variable in a later tier causes one in an earlier tier.

        ``confidence`` makes it soft, as for :meth:`require`: one statement that
        weighs every edge from a later tier into an earlier one. It says nothing
        about pairs within a tier, as the hard form does not.
        """
        groups = [_pairs(t) for t in tiers]
        statement = KnowledgeStatement(
            "tiers", tuple(_names(group) for group in groups), _confidence(confidence)
        )
        if statement.is_soft:
            return self._soft(statement)
        precedes = set(self.precedes)
        for i, earlier in enumerate(groups):
            for later in groups[i + 1 :]:
                for a in earlier:
                    for b in later:
                        precedes.add((a, b))
        return self._hard(statement, precedes=frozenset(precedes))

    def root(self, *names: str, confidence: float | None = None) -> DomainKnowledge:
        """Mark variables as roots: no incoming directed edges.

        ``confidence`` makes it soft, as for :meth:`require`, with one statement
        per name.
        """
        level = _confidence(confidence)
        out = self
        if level is None:
            for name in names:
                out = out._hard(KnowledgeStatement("root", (name,)))
            return replace(out, roots=self.roots | frozenset(names))
        for name in names:
            out = out._soft(KnowledgeStatement("root", (name,), level))
        return out

    def sink(self, *names: str, confidence: float | None = None) -> DomainKnowledge:
        """Mark variables as sinks: no outgoing directed edges.

        ``confidence`` makes it soft, as for :meth:`require`, with one statement
        per name.
        """
        level = _confidence(confidence)
        out = self
        if level is None:
            for name in names:
                out = out._hard(KnowledgeStatement("sink", (name,)))
            return replace(out, sinks=self.sinks | frozenset(names))
        for name in names:
            out = out._soft(KnowledgeStatement("sink", (name,), level))
        return out

    def confound(self, a: str, b: str, *, confidence: float | None = None) -> DomainKnowledge:
        """Record that ``a`` and ``b`` share a hidden common cause.

        A confounder is hard only: ``confidence`` must be ``None`` or 1. A hidden
        common cause is not an edge state, so edge probabilities cannot weigh it.
        """
        if _confidence(confidence) is not None:
            raise ValueError(
                "confound is hard only: a hidden common cause is not an edge state that "
                "edge probabilities can weigh; drop confidence= or leave the pair out"
            )
        if a == b:
            raise ValueError(f"a confounded pair needs two distinct variables, got {a!r} twice")
        return self._hard(
            KnowledgeStatement("confound", (a, b)),
            confounded=self.confounded | {frozenset({a, b})},
        )

    def forbid_between(
        self,
        group_a: Iterable[str],
        group_b: Iterable[str],
        *,
        confidence: float | None = None,
    ) -> DomainKnowledge:
        """Forbid every directed edge from ``group_a`` into ``group_b``.

        ``confidence`` makes it soft, as for :meth:`require`: one statement that
        weighs every such edge.
        """
        a = list(group_a)
        b = list(group_b)
        statement = KnowledgeStatement(
            "forbid_between", (_names(a), _names(b)), _confidence(confidence)
        )
        if statement.is_soft:
            return self._soft(statement)
        bans = {(x, y) for x in a for y in b}
        return self._hard(statement, forbidden=self.forbidden | bans)

    def restrict_to(
        self,
        skeleton: MixedGraph | Iterable[tuple[str, str]],
        *,
        confidence: float | None = None,
    ) -> DomainKnowledge:
        """Allow adjacencies only on the pairs of ``skeleton``.

        ``skeleton`` is a :class:`~ergodic.graph.MixedGraph` (its edges name
        the pairs, whatever the marks) or an iterable of ``(u, v)`` pairs. A
        pair outside the allow-list is banned in every form: directed either
        way, bidirected, or undirected. Restricting an already restricted
        object keeps the intersection, so a second call only narrows the
        support. An empty iterable allows no adjacencies at all.

        ``confidence`` makes it soft, as for :meth:`require`: one statement that
        weighs both directed edges of every pair outside the skeleton (among the
        variables being decoded). Soft restrictions do not intersect; each is its
        own statement.
        """
        from .graph import MixedGraph

        if isinstance(skeleton, MixedGraph):
            raw: Iterable[tuple[str, str]] = ((e.u, e.v) for e in skeleton.edges)
        else:
            raw = skeleton
        level = _confidence(confidence)
        pairs: set[frozenset[str]] = set()
        for u, v in raw:
            if u == v:
                raise ValueError(
                    f"a skeleton adjacency needs two distinct variables, got {u!r} twice"
                )
            pairs.add(frozenset({u, v}))
        listed = tuple(sorted(_names(pair) for pair in pairs))
        statement = KnowledgeStatement("restrict_to", listed, level)
        if statement.is_soft:
            return self._soft(statement)
        allowed = frozenset(pairs)
        if self.allowed_adjacencies is not None:
            allowed &= self.allowed_adjacencies
        return self._hard(statement, allowed_adjacencies=allowed)

    def __and__(self, other: DomainKnowledge) -> DomainKnowledge:
        """Combine two knowledge objects by taking the union of their constraints.

        Skeleton allow-lists intersect: an adjacency survives only when both
        sides allow it, and ``None`` (unrestricted) defers to the other side.
        Soft statements are pooled.
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
            soft=self.soft | other.soft,
            stated=(*self.stated, *other.stated),
        )

    combine = __and__

    def statements(self) -> tuple[KnowledgeStatement, ...]:
        """Every statement as it was made: the hard ones in order, then the soft ones.

        Hard constraints set without a builder (through the constructor or
        ``dataclasses.replace``) are listed one per stored item: a required or
        forbidden pair, a precedence pair, a root, a sink, a confounded pair, and
        one ``restrict_to`` for the allow-list. Soft statements come sorted by
        kind, then arguments.
        """
        covered = DomainKnowledge()
        hard: list[KnowledgeStatement] = []
        for statement in dict.fromkeys(self.stated):  # stated twice, listed once
            covered = _apply(covered, statement)
            hard.append(statement)
        hard += [
            KnowledgeStatement("require", pair) for pair in sorted(self.required - covered.required)
        ]
        rest = self.forbidden - covered.forbidden
        for u, v in sorted(rest):
            if (v, u) not in rest:
                hard.append(KnowledgeStatement("forbid", (u, v)))
            elif u < v:
                hard.append(KnowledgeStatement("forbid_adjacency", (u, v)))
        hard += [
            KnowledgeStatement("order", pair) for pair in sorted(self.precedes - covered.precedes)
        ]
        hard += [KnowledgeStatement("root", (name,)) for name in sorted(self.roots - covered.roots)]
        hard += [KnowledgeStatement("sink", (name,)) for name in sorted(self.sinks - covered.sinks)]
        hard += [
            KnowledgeStatement("confound", _names(pair))
            for pair in sorted(self.confounded - covered.confounded, key=sorted)
        ]
        if (
            self.allowed_adjacencies is not None
            and covered.allowed_adjacencies != self.allowed_adjacencies
        ):
            listed = tuple(sorted(_names(pair) for pair in self.allowed_adjacencies))
            hard.append(KnowledgeStatement("restrict_to", listed))
        return (*hard, *sorted(self.soft, key=_statement_key))

    # -- validation ----------------------------------------------------------

    def conflicts(self) -> list[str]:
        """Return a list of internal contradictions among the hard constraints.

        Empty when consistent. Soft statements never conflict: they are weighed,
        not enforced. A confounded pair whose adjacency is banned (both
        directions forbidden) is a contradiction, as is one outside a skeleton
        restriction.
        """
        out: list[str] = []
        for u, v in sorted(self.required & self.forbidden):
            out.append(f"edge {u}->{v} is both required and forbidden")
        for pair in sorted(self.confounded, key=sorted):
            a, b = sorted(pair)
            if (a, b) in self.forbidden and (b, a) in self.forbidden:
                out.append(f"confounded pair {a},{b} is banned from being adjacent")
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
        """Raise :class:`InconsistentKnowledgeError` if the hard constraints contradict.

        Soft statements are never checked: they are weighed, not enforced.
        """
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
        Only hard constraints count; soft statements permit everything.
        """
        return self._permits(cause, effect, self._precedence_closure())

    def requires_edge(self, cause: str, effect: str) -> bool:
        """Return ``True`` if ``cause -> effect`` was required with :meth:`require`.

        Only the stated pair counts; nothing is inferred from tiers or other constraints.
        A soft ``require`` does not count.
        """
        return (cause, effect) in self.required

    def precedence(self) -> frozenset[tuple[str, str]]:
        """Every ``(before, after)`` pair, with the order read transitively.

        The projection discovery reads to orient: ``(a, b)`` here means ``b``
        cannot be an ancestor of ``a``, which in a latent-variable search
        (FCI) pins an arrowhead at ``b`` on any surviving adjacency.
        """
        return self._precedence_closure()

    def violations(self, graph: MixedGraph) -> list[str]:
        """Return the ways ``graph`` breaks the hard constraints, empty when consistent.

        Soft statements are not checked. A node in ``graph.latent`` is read as a
        hidden cause, not as one of the variables: a latent parent shared by both
        ends of a confounded pair stands for its bidirected edge (the way a bow,
        ``a -> b`` plus ``a <-> b``, is written), and edges at a latent node count
        neither against a root, a sink nor a skeleton restriction.
        """
        out: list[str] = []
        latent = graph.latent
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
                if edge.u in latent or edge.v in latent:
                    continue
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
            if r in graph.nodes and graph.parents(r) - latent:
                out.append(f"root {r} has parents")
        for s in sorted(self.sinks):
            if s in graph.nodes and graph.children(s) - latent:
                out.append(f"sink {s} has children")
        for pair in sorted(self.confounded, key=sorted):
            a, b = sorted(pair)
            edge = graph.get_edge(a, b)
            if edge is not None and edge.is_bidirected:
                continue
            if (
                a in graph.nodes
                and b in graph.nodes
                and graph.parents(a) & graph.parents(b) & latent
            ):
                continue
            out.append(f"confounded pair {a},{b} lacks a bidirected edge")
        return out

    def is_consistent(self, graph: MixedGraph) -> bool:
        """Return ``True`` if ``graph`` satisfies every hard constraint."""
        return not self.violations(graph)

    def apply_to(self, graph: MixedGraph) -> MixedGraph:
        """Enforce the hard constraints on ``graph``: drop forbidden, add required edges.

        A pair forbidden in both directions loses its edge whatever the marks; a
        single-direction ban only drops the matching directed edge. Under a
        skeleton restriction, every edge on a pair outside the allow-list is
        dropped too. Edges left undirected or with a circle are oriented when
        only one direction stays permitted. The result is returned as a new
        graph. Soft statements are not applied.
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
        Only hard constraints shape the mask; :meth:`log_odds` reads the soft ones.
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

    # -- soft knowledge ------------------------------------------------------

    def _soft_names(self, names: Iterable[str]) -> list[str]:
        known = set(names)
        unknown = sorted({v for s in self.soft for v in s.variables} - known)
        if unknown:
            raise ValueError(f"soft statements name variables outside the given ones: {unknown}")
        return list(names)

    def log_odds(self, variables: Sequence[str]) -> npt.NDArray[np.float64]:
        """The soft statements' summed log likelihood ratios on each directed edge.

        ``out[i, j]`` is ``log`` of the product, over every soft statement, of the
        ratio it puts on the state ``variables[i] -> variables[j]`` against "no
        edge" on that pair: ``log(c / (1 - c))`` when the edge makes the statement
        hold and no edge fails it, ``log((1 - c) / c)`` the other way round, and 0
        otherwise. For example ``require(a, b)`` adds ``log(c / (1 - c))`` at
        ``[a, b]`` and nothing at ``[b, a]``; ``root(x)`` adds ``log((1 - c) / c)``
        at every ``[v, x]``. The diagonal is zero and hard constraints are left
        out (they are enforced, not weighed). :func:`ergodic.decode` multiplies
        the pair states by ``exp(out)`` and renormalizes each pair.

        Args:
            variables: The variable order of the rows and columns. Every
                variable a soft statement names must be among them; ``root``,
                ``sink`` and ``restrict_to`` range over all of them.

        Returns:
            A ``(d, d)`` float matrix.

        Raises:
            ValueError: A soft statement names a variable outside ``variables``,
                or a name repeats.
        """
        import numpy as np

        names = self._soft_names(variables)
        if len(set(names)) != len(names):
            raise ValueError("variables must be unique")
        index = {name: i for i, name in enumerate(names)}
        out = np.zeros((len(names), len(names)), dtype=np.float64)
        for statement in sorted(self.soft, key=_statement_key):
            confidence = float(statement.confidence or 0.5)
            ratio = math.log(confidence) - math.log1p(-confidence)
            for a, b, holds in _pair_rules(statement, names):
                for state, (source, target) in ((1, (a, b)), (2, (b, a))):
                    if holds[state] and not holds[0]:
                        out[index[source], index[target]] += ratio
                    elif holds[0] and not holds[state]:
                        out[index[source], index[target]] -= ratio
        return out

    def log_prior(self, graph: MixedGraph) -> float:
        """The log prior of ``graph`` under this knowledge, up to a constant.

        Minus infinity when ``graph`` breaks a hard constraint
        (:meth:`violations`). Otherwise the sum, over every soft statement and
        every pair it speaks about, of ``log c`` when the pair's state in
        ``graph`` satisfies the statement and ``log(1 - c)`` when it does not. That
        is the reading :meth:`log_odds` and the decoders use, so a multi-pair
        statement (tiers, a root, a sink, ``restrict_to``, ``forbid_between``)
        counts once per pair, and an order is read as "no direct edge against it".
        Hard knowledge alone gives 0 for every consistent graph.

        Args:
            graph: A graph with directed (and, ignored here, bidirected) edges, such
                as a DAG or an ADMG. Latent nodes are not variables: ``root``,
                ``sink`` and ``restrict_to`` range over the other nodes.

        Raises:
            ValueError: ``graph`` has an undirected or circle-marked edge, or a soft
                statement names a variable that is not an observed node.
        """
        loose = [e.glyph() for e in graph.edge_list() if not (e.is_directed or e.is_bidirected)]
        if loose:
            raise ValueError(
                "log_prior reads directed and bidirected edges only; orient these first: "
                + ", ".join(loose)
            )
        if self.violations(graph):
            return -math.inf
        names = self._soft_names(sorted(graph.nodes - graph.latent))
        directed = {pair for e in graph.edges if (pair := e.directed_pair()) is not None}
        total = 0.0
        for statement in sorted(self.soft, key=_statement_key):
            confidence = float(statement.confidence or 0.5)
            for a, b, holds in _pair_rules(statement, names):
                state = 1 if (a, b) in directed else 2 if (b, a) in directed else 0
                total += math.log(confidence) if holds[state] else math.log1p(-confidence)
        return total

    def energy(self, graph: MixedGraph) -> float:
        """Minus :meth:`log_prior`: plus infinity when a hard constraint is broken."""
        return -self.log_prior(graph)

    def __repr__(self) -> str:
        restricted = (
            ""
            if self.allowed_adjacencies is None
            else f", allowed_adjacencies={len(self.allowed_adjacencies)}"
        )
        soft = f", soft={len(self.soft)}" if self.soft else ""
        return (
            "DomainKnowledge("
            f"vars={len(self.variables)}, required={len(self.required)}, "
            f"forbidden={len(self.forbidden)}, precedes={len(self.precedes)}, "
            f"roots={len(self.roots)}, sinks={len(self.sinks)}, "
            f"confounded={len(self.confounded)}{restricted}{soft})"
        )


def _apply(knowledge: DomainKnowledge, statement: KnowledgeStatement) -> DomainKnowledge:
    """``knowledge`` with ``statement`` added through its builder."""
    kind, args = statement.kind, statement.args
    level = statement.confidence
    if kind == "tiers":
        return knowledge.with_tiers(args, confidence=level)
    if kind == "forbid_between":
        return knowledge.forbid_between(args[0], args[1], confidence=level)
    if kind == "restrict_to":
        return knowledge.restrict_to(args, confidence=level)
    if kind == "confound":
        return knowledge.confound(args[0], args[1])
    builder = getattr(knowledge, kind)
    result: DomainKnowledge = builder(*args, confidence=level)
    return result
