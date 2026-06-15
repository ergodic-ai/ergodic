# SPDX-License-Identifier: Apache-2.0
"""``ida``: identification across the members of an equivalence class.

The IDA pattern (Maathuis-Kalisch-Buhlmann) made first class. ID is defined on a
single graph; a CPDAG, MAG, or PAG is an equivalence class, so the effect of a
treatment on an outcome may identify differently (or not at all) across members.
:func:`ida` enumerates the members and identifies per member, returning an
:class:`IDA` record: the per-member results plus a summary (how many identify,
the distinct strategies and estimands, whether the class agrees).

Scope, stated honestly (the plan's A7):

- CPDAG. Each member is a DAG, so the full ID algorithm runs per member through
  :func:`~ergodic.identification.identify_effect`. The record is the per-member
  multiset.
- PAG or MAG. Each member is a MAG, not a DAG; general per-member ID needs a DAG
  realization of every MAG member, which is not built. The class-level
  generalized adjustment criterion (which :func:`identify_effect` already runs on
  a class face) gives a single answer that holds for *every* member where it
  applies (null or an adjustment set); that is returned as a one-element class
  record. Where it does not apply (the class is not adjustment-identifiable),
  :func:`ida` raises, rather than silently dropping the cases general ID would
  decide.
- A single graph (DAG or ADMG) raises with a pointer to :func:`identify_effect`.

Enumeration uses the capped-with-raise discipline of the class enumerators:
``max_members`` raises once the count would pass it, never truncates silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..graph import GraphKind, dags_of_cpdag
from ._criteria import _as_set, _validate, identify_effect

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..graph import MixedGraph
    from ._record import Identification

_DEFAULT_MAX_MEMBERS = 1000


@dataclass(frozen=True)
class IDA:
    """Per-member identification across an equivalence class.

    ``members`` is the tuple of per-member :class:`Identification` results, in
    the enumerator's deterministic order. For a CPDAG that is one entry per DAG
    member; for a PAG or MAG handled through the class-level adjustment criterion
    it is a single entry standing for the whole class (the criterion's answer
    holds in every member). ``class_kind`` records the face that was enumerated.

    The summary properties read the multiset: how many members identify, the
    distinct strategies and estimands seen, and whether the class agrees.
    """

    treatment: frozenset[str]
    outcome: frozenset[str]
    class_kind: GraphKind
    members: tuple[Identification, ...]

    @property
    def n_members(self) -> int:
        """The number of members examined."""
        return len(self.members)

    @property
    def n_identifiable(self) -> int:
        """How many members have an identifiable effect."""
        return sum(1 for m in self.members if m.identifiable)

    @property
    def all_identifiable(self) -> bool:
        """``True`` when every member identifies."""
        return all(m.identifiable for m in self.members)

    @property
    def strategies(self) -> frozenset[str]:
        """The distinct strategies seen across members (e.g. ``{"adjustment", "id"}``)."""
        return frozenset(m.strategy for m in self.members)

    @property
    def estimands(self) -> frozenset[str]:
        """The distinct estimand strings across the identifiable members."""
        return frozenset(m.estimand for m in self.members if m.identifiable)

    @property
    def agree(self) -> bool:
        """``True`` when every member identifies and shares one estimand.

        The classic IDA reading: a single, member-invariant answer. ``False``
        when some member fails or the members disagree about the estimand.
        """
        return self.all_identifiable and len(self.estimands) == 1

    def __bool__(self) -> bool:
        return self.all_identifiable

    def __repr__(self) -> str:
        arrow = f"{sorted(self.treatment)} -> {sorted(self.outcome)}"
        return (
            f"IDA({arrow}, {self.class_kind.value}: "
            f"{self.n_identifiable}/{self.n_members} identify, "
            f"strategies={sorted(self.strategies)}, agree={self.agree})"
        )


def ida(
    graph: MixedGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    *,
    max_members: int = _DEFAULT_MAX_MEMBERS,
) -> IDA:
    """Identify ``treatment -> outcome`` across every member of an equivalence class.

    On a CPDAG, enumerate the DAG members (:func:`dags_of_cpdag`) and run
    :func:`identify_effect` per member, returning the multiset as an :class:`IDA`.
    On a PAG or MAG, return the class-level generalized-adjustment answer where it
    applies (a single member-invariant result), and raise when the case needs
    general per-member ID (deferred: it needs a DAG realization of each MAG
    member). A single graph (DAG or ADMG) raises with a pointer to
    :func:`identify_effect`.
    """
    x = _as_set(treatment)
    y = _as_set(outcome)
    _validate(graph, x, y)
    kind = graph.kind

    if kind is GraphKind.MIXED:
        raise ValueError(
            "ida needs a verified class face (cpdag, pag, or mag); re-face the "
            "graph with as_kind once it satisfies a face's rules"
        )
    if kind in (GraphKind.DAG, GraphKind.ADMG):
        raise ValueError(
            "a single graph has one identify_effect; ida is for an equivalence "
            "class (cpdag, pag, or mag). Call identify_effect(graph, treatment, "
            "outcome) instead"
        )
    if kind is GraphKind.CPDAG:
        members = tuple(
            identify_effect(member, x, y)
            for member in dags_of_cpdag(graph, max_members=max_members)
        )
        return IDA(x, y, kind, members)
    # PAG or MAG: the class-level adjustment criterion gives a member-invariant
    # answer where it applies; otherwise general per-member ID is needed, which
    # needs a DAG realization of each MAG member (not built).
    class_level = identify_effect(graph, x, y)
    if class_level.identifiable:
        return IDA(x, y, kind, (class_level,))
    raise NotImplementedError(
        f"general ID over {kind.value} members needs a DAG realization of each "
        "MAG member (not built); the class-level adjustment criterion did not "
        f"identify {sorted(x)} -> {sorted(y)} here. Use the per-member adjustment "
        "criterion (identify_effect on the class) or enumerate the class "
        "(mags_of_pag) and reason per member"
    )
