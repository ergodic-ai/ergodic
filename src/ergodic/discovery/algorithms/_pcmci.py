# SPDX-License-Identifier: Apache-2.0
"""PCMCI+ over a lagged window frame: temporal discovery with one CI oracle.

The shape follows Runge's PCMCI+ (2020), phrased over the lagged frame from
:func:`ergodic.discovery._lagged.lagged_tabular`. Five steps:

1. PC1 lagged screen. Per target ``Y`` at lag zero, every ``(X, tau)`` with
   ``tau`` in ``1..max_lag`` (including ``X = Y``) starts as a candidate
   parent. Unconditional tests drop the clearly independent ones, then rounds
   of conditioning on the ``q`` strongest other survivors (q = 1, 2, ...)
   shrink the set, in deterministic order: ascending p from the previous
   round, ties broken by name.
2. MCI for lagged links. Each surviving ``(X, tau)`` is re-tested against
   ``Y[t]`` given the other screen parents of ``Y`` and the screen parents of
   ``X`` shifted by ``tau`` (a parent ``(W, k)`` of ``X`` becomes
   ``(W, k + tau)`` in ``Y``'s frame). The shifted parents reach back to lag
   ``2 * max_lag``, which is why the oracle binds to a frame twice as deep;
   parents shifted past the deep frame's depth are dropped.
3. Contemporaneous skeleton. PC-stable over the lag-zero pairs: the adjacency
   is frozen per level, subsets of size q come from either endpoint's
   contemporaneous neighbours, and the lagged MCI parents of both endpoints
   ride along in every conditioning set.
4. Orientation. The collider rule on unshielded contemporaneous triples reads
   the recorded sepsets; an edge whose two endpoints are both demanded as
   heads by different triples stays undirected and is counted as a conflict.
   Then exactly the two within-slice Meek rules run to a fixpoint: R1
   (``X -> Z - Y`` with X, Y non-adjacent becomes ``Z -> Y``) and R2
   (``X -> Z -> Y`` with ``X - Y`` undirected becomes ``X -> Y``). No rule
   reads lagged arrows: orienting a slice edge from a lagged parent is not
   sound without more care, so we leave it out.
5. The window graph. Nodes are every ``v[t-k]`` for ``k`` in ``0..max_lag``,
   isolated ones included. Lagged links and oriented contemporaneous edges
   are directed; the rest are undirected. An all-directed result takes the
   DAG face (falling back to mixed if a directed slice cycle slips through).

Background knowledge reads each variable as a process, so a constraint on a
pair applies at every lag, the contemporaneous slice included; the class
docstring lists the exact mapping for each kind of constraint.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

from ...data import PanelData, TimeSeriesData
from ...graph import InvalidGraphError, dag, mixed
from .._lagged import lagged_node, lagged_tabular
from .._registry import resolve_test
from .._tsresult import TimeSeriesDiscoveryResult

if TYPE_CHECKING:
    from ...graph import MixedGraph
    from ...knowledge import DomainKnowledge
    from .._ci import CITest

__all__ = ["PCMCI"]

_Lagged = tuple[str, int]


class PCMCI:
    """PCMCI+ over a conditional-independence test, for time-series and panel data.

    Args:
        test: A :class:`CITest` instance or a name; ``"auto"`` reads the
            lagged frame's schema (all-continuous data gets the
            partial-correlation test).
        alpha: Significance level for every independence decision.
        max_lag: Largest lag a reported link can have, at least one.
        max_cond_size: Cap on the screen round and on the contemporaneous
            subset size, ``None`` for no cap.

    ``fit(data)`` takes :class:`~ergodic.data.TimeSeriesData` or
    :class:`~ergodic.data.PanelData` (panel windows stack per entity) and
    returns a :class:`TimeSeriesDiscoveryResult` whose graph lives on the
    window nodes ``v[t-k]``.

    ``fit(data, knowledge)`` reads a
    :class:`~ergodic.knowledge.DomainKnowledge` over the process names, not
    the window nodes, so a constraint on ``(x, y)`` applies at every lag,
    the contemporaneous slice included:

    - A directional forbid ``x -> y`` means the link ``(x, y, lag)`` is
      never screened or MCI-tested at any lag and the lag-zero orientation
      ``x -> y`` is banned; the contemporaneous edge itself may survive when
      ``y -> x`` stays permitted (only the direction dies). A pair forbidden
      in both directions is an adjacency ban: the contemporaneous pair is
      removed without testing (and records no sepset) and both lagged
      directions are pruned. An explicit ``forbid(x, x)`` removes the
      self-memory link ``x[t-1] -> x[t]``.
    - Tiers and ``order``: a later variable never causes an earlier one at
      any lag, pruned exactly as a forbid; additionally a surviving
      contemporaneous edge between ordered processes is seeded
      earlier ``->`` later before the collider phase, and a data-driven
      arrowhead that disagrees follows the orientation-conflict convention
      already in this file (the edge stays undirected and is counted in
      ``diagnostics["orientation_conflicts"]``).
    - Roots take no incoming cross-variable link at any lag and sinks no
      outgoing one; self-memory ``x[t-1] -> x[t]`` survives both, since a
      process can carry its own past without being caused by anything else.
      A skeleton restriction (``restrict_to``) bans pairs outside its
      support the same way, self-memory exempt.
    - ``required (x, y)`` names no lag, so it cannot be forced; it is
      validated after the run instead, and ``fit`` raises ``ValueError``
      when no compatible link (a lagged ``x -> y``, an oriented lag-zero
      ``x -> y``, or a still-undirected lag-zero edge) survived.
    - ``confound`` has no analogue in a window graph (it carries no
      bidirected edges) and raises ``NotImplementedError``.

    ``diagnostics["n_knowledge_pruned"]`` counts what the knowledge took
    off the table: lagged links never screened, contemporaneous pairs
    removed outright, and lag-zero direction bans, one count each.
    """

    ingredient: ClassVar[str] = "test"

    def __init__(
        self,
        test: CITest | str = "auto",
        *,
        alpha: float = 0.05,
        max_lag: int = 1,
        max_cond_size: int | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if max_cond_size is not None and max_cond_size < 0:
            raise ValueError(f"max_cond_size must be None or >= 0, got {max_cond_size}")
        self.test = test
        self.alpha = alpha
        self.max_lag = max_lag
        self.max_cond_size = max_cond_size

    # -- the run ---------------------------------------------------------

    def fit(
        self,
        data: TimeSeriesData | PanelData | None = None,
        knowledge: DomainKnowledge | None = None,
    ) -> TimeSeriesDiscoveryResult:
        """Run PCMCI+ and return the window graph with its provenance.

        ``knowledge`` constrains processes, so every constraint applies at
        every lag; the class docstring lists the mappings. ``confound``
        raises here (a window graph has no bidirected edges), and a
        ``required`` edge that survived at no lag raises after the run.
        """
        if not isinstance(data, TimeSeriesData | PanelData):
            raise TypeError(
                f"PCMCI needs TimeSeriesData or PanelData, got {type(data).__name__}; "
                "for cross-sectional rows use a tabular method (PC, GES, ...) via discover()"
            )
        if knowledge is not None and knowledge.confounded:
            raise NotImplementedError(
                "confounded pairs have no analogue in a window graph (it carries no "
                "bidirected edges); drop knowledge.confound for PCMCI"
            )
        flags = {"mci_truncated": 0.0, "acyclicity_fallback": 0.0}
        shallow = lagged_tabular(data, self.max_lag)  # validates the rows, voices the warnings
        deep_lag = 2 * self.max_lag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the shallow build already warned once
            try:
                frame = lagged_tabular(data, deep_lag)
            except ValueError:  # too few rows for the deep frame: condition at max_lag
                frame = shallow
                deep_lag = self.max_lag
                flags["mci_truncated"] = 1.0
        oracle = resolve_test(self.test, frame)
        names = sorted(data.variables)
        banned, removed, seeds, n_pruned = _knowledge_maps(knowledge, names, self.max_lag)

        screen = {target: self._pc1(oracle, names, target, banned) for target in names}
        mci = {target: self._mci(oracle, screen, target, deep_lag, banned) for target in names}
        links: dict[tuple[str, str, int], float] = {}
        for target in names:
            for (source, tau), p in sorted(mci[target].items()):
                links[(source, target, tau)] = p
        n_lagged = len(links)

        mci_names = {
            target: frozenset(lagged_node(source, tau) for source, tau in mci[target])
            for target in names
        }
        adj, sepsets, last_p = self._contemporaneous(oracle, names, mci_names, removed)
        present_seeds: dict[frozenset[str], tuple[str, str]] = {}
        for pair, orientation in seeds.items():
            a, b = sorted(pair)
            if b in adj[a]:  # seed only edges that survived the skeleton
                present_seeds[pair] = orientation
        directed, conflicts = _orient_slice(names, adj, sepsets, seeds=present_seeds, banned=banned)

        graph, n_contemporaneous, n_oriented = self._assemble(
            names, adj, directed, last_p, links, flags
        )
        if knowledge is not None:
            _check_required(knowledge, names, links, graph, self.max_lag)
        diagnostics: dict[str, float] = dict(oracle.diagnostics())
        diagnostics.update(flags)
        diagnostics["n_lagged_links"] = float(n_lagged)
        diagnostics["n_contemporaneous"] = float(n_contemporaneous)
        diagnostics["n_oriented"] = float(n_oriented)
        diagnostics["orientation_conflicts"] = float(conflicts)
        diagnostics["n_knowledge_pruned"] = float(n_pruned)
        out_sepsets = {
            frozenset(lagged_node(v, 0) for v in pair): frozenset(lagged_node(s, 0) for s in sep)
            for pair, sep in sepsets.items()
        }
        return TimeSeriesDiscoveryResult(
            graph=graph,
            method="pcmci",
            test=oracle.name,
            alpha=self.alpha,
            n=oracle.n,
            diagnostics=diagnostics,
            sepsets=out_sepsets,
            max_lag=self.max_lag,
            links=links,
        )

    # -- step 1: the PC1 lagged screen ---------------------------------------

    def _pc1(
        self,
        oracle: CITest,
        names: list[str],
        target: str,
        banned: frozenset[tuple[str, str]],
    ) -> dict[_Lagged, float]:
        """Surviving lagged parent candidates of one target, with their last p.

        A ``banned`` pair (knowledge says the source never causes the target)
        is skipped at every lag, never tested.
        """
        y = lagged_node(target, 0)
        pvals: dict[_Lagged, float] = {}
        for tau in range(1, self.max_lag + 1):
            for source in names:
                if (source, target) in banned:
                    continue  # knowledge: source never causes target, at any lag
                p = oracle.test(lagged_node(source, tau), y).p_value
                if p <= self.alpha:
                    pvals[(source, tau)] = p
        q = 1
        while q <= len(pvals) - 1 and (self.max_cond_size is None or q <= self.max_cond_size):
            ranked = [c for c, _ in sorted(pvals.items(), key=lambda item: (item[1], item[0]))]
            for candidate in ranked:
                if candidate not in pvals:
                    continue  # removed earlier in this round
                strongest = [c for c in ranked if c != candidate and c in pvals][:q]
                if len(strongest) < q:
                    continue
                conditioning = [lagged_node(source, tau) for source, tau in strongest]
                x = lagged_node(candidate[0], candidate[1])
                p = oracle.test(x, y, conditioning).p_value
                if p > self.alpha:
                    del pvals[candidate]
                else:
                    pvals[candidate] = p
            q += 1
        return pvals

    # -- step 2: MCI for the lagged links -------------------------------------

    def _mci(
        self,
        oracle: CITest,
        screen: dict[str, dict[_Lagged, float]],
        target: str,
        deep_lag: int,
        banned: frozenset[tuple[str, str]],
    ) -> dict[_Lagged, float]:
        """Re-test each screened lagged parent under the MCI conditioning set."""
        y = lagged_node(target, 0)
        parents = screen[target]
        kept: dict[_Lagged, float] = {}
        for source, tau in sorted(parents):
            if (source, target) in banned:  # pragma: no cover - the screen already skipped it
                continue
            conditioning = {
                lagged_node(other, k) for other, k in parents if (other, k) != (source, tau)
            }
            for other, k in screen[source]:
                if k + tau <= deep_lag:  # parents shifted past the deep frame are dropped
                    conditioning.add(lagged_node(other, k + tau))
            p = oracle.test(lagged_node(source, tau), y, sorted(conditioning)).p_value
            if p <= self.alpha:
                kept[(source, tau)] = p
        return kept

    # -- step 3: the contemporaneous skeleton ---------------------------------

    def _contemporaneous(
        self,
        oracle: CITest,
        names: list[str],
        mci_names: dict[str, frozenset[str]],
        removed: frozenset[frozenset[str]],
    ) -> tuple[
        dict[str, set[str]],
        dict[frozenset[str], frozenset[str]],
        dict[frozenset[str], float],
    ]:
        """PC-stable over the lag-zero pairs, always conditioning on MCI parents.

        ``removed`` pairs (knowledge bans in both directions) never enter the
        adjacency, are never tested, and record no sepset, exactly like a
        knowledge-pruned pair in the iid skeleton.
        """
        adj: dict[str, set[str]] = {v: set(names) - {v} for v in names}
        for pair in removed:
            a, b = sorted(pair)
            adj[a].discard(b)
            adj[b].discard(a)
        sepsets: dict[frozenset[str], frozenset[str]] = {}
        last_p: dict[frozenset[str], float] = {}
        q = 0
        while self.max_cond_size is None or q <= self.max_cond_size:
            snapshot = {v: frozenset(adj[v]) for v in names}
            if all(len(snapshot[v]) <= q for v in names):
                break
            for x in names:
                for y in sorted(snapshot[x]):
                    if y not in adj[x]:
                        continue  # removed earlier at this level
                    candidates = sorted(snapshot[x] - {y})
                    if len(candidates) < q:
                        continue
                    base = mci_names[x] | mci_names[y]
                    for subset in combinations(candidates, q):
                        conditioning = base | {lagged_node(s, 0) for s in subset}
                        result = oracle.test(lagged_node(x, 0), lagged_node(y, 0), conditioning)
                        if result.p_value > self.alpha:
                            adj[x].discard(y)
                            adj[y].discard(x)
                            sepsets[frozenset((x, y))] = frozenset(subset)
                            break
                        last_p[frozenset((x, y))] = result.p_value
            q += 1
        return adj, sepsets, last_p

    # -- step 5: the window graph ----------------------------------------------

    def _assemble(
        self,
        names: list[str],
        adj: dict[str, set[str]],
        directed: dict[frozenset[str], tuple[str, str]],
        last_p: dict[frozenset[str], float],
        links: dict[tuple[str, str, int], float],
        flags: dict[str, float],
    ) -> tuple[MixedGraph, int, int]:
        """Build the window graph and fold the contemporaneous edges into ``links``."""
        edges = [
            f"{lagged_node(source, tau)} -> {lagged_node(target, 0)}"
            for source, target, tau in sorted(links)
        ]
        n_contemporaneous = 0
        n_oriented = 0
        has_undirected = False
        for x, y in combinations(names, 2):
            if y not in adj[x]:
                continue
            pair = frozenset((x, y))
            n_contemporaneous += 1
            orientation = directed.get(pair)
            if orientation is None:
                has_undirected = True
                links[(x, y, 0)] = last_p[pair]
                edges.append(f"{lagged_node(x, 0)} --- {lagged_node(y, 0)}")
            else:
                tail, head = orientation
                n_oriented += 1
                links[(tail, head, 0)] = last_p[pair]
                edges.append(f"{lagged_node(tail, 0)} -> {lagged_node(head, 0)}")
        nodes = [lagged_node(v, k) for v in names for k in range(self.max_lag + 1)]
        if has_undirected:
            graph = mixed(edges, nodes=nodes)
        else:
            try:
                graph = dag(edges, nodes=nodes)
            except InvalidGraphError:
                graph = mixed(edges, nodes=nodes)
                flags["acyclicity_fallback"] = 1.0
        return graph, n_contemporaneous, n_oriented


# -- knowledge, read as constraints on processes ---------------------------------


def _knowledge_maps(
    knowledge: DomainKnowledge | None,
    names: list[str],
    max_lag: int,
) -> tuple[
    frozenset[tuple[str, str]],
    frozenset[frozenset[str]],
    dict[frozenset[str], tuple[str, str]],
    int,
]:
    """Project the knowledge onto processes: bans, removed pairs, seeds, a count.

    ``banned`` holds every ordered pair ``(source, target)`` where the
    knowledge forbids the source causing the target, read through
    :meth:`~ergodic.knowledge.DomainKnowledge.permits_edge` so directional
    forbids, tiers and ``order``, roots, sinks, and a skeleton restriction
    all land; a ban applies at every lag. Self pairs ``(x, x)`` enter only
    on an explicit ``forbid(x, x)``: roots, sinks, and a skeleton
    restriction speak about distinct variables, so self-memory stays on the
    table. ``removed`` holds the contemporaneous pairs banned in both
    directions, ``seeds`` the lag-zero orientations the precedence order
    compels (earlier ``->`` later), and the count is what the diagnostics
    report: ``max_lag`` lagged candidates per ban, one per removed pair, one
    per surviving lag-zero direction ban.
    """
    if knowledge is None:
        return frozenset(), frozenset(), {}, 0
    banned: set[tuple[str, str]] = set()
    for source in names:
        for target in names:
            if source == target:
                if (source, source) in knowledge.forbidden:
                    banned.add((source, source))
            elif not knowledge.permits_edge(source, target):
                banned.add((source, target))
    precedes = knowledge.precedence()
    removed: set[frozenset[str]] = set()
    seeds: dict[frozenset[str], tuple[str, str]] = {}
    for x, y in combinations(names, 2):
        if (x, y) in banned and (y, x) in banned:
            removed.add(frozenset((x, y)))
        elif (x, y) in precedes:
            seeds[frozenset((x, y))] = (x, y)
        elif (y, x) in precedes:
            seeds[frozenset((x, y))] = (y, x)
    n_pruned = max_lag * len(banned) + len(removed)
    n_pruned += sum(1 for x, y in banned if x != y and frozenset((x, y)) not in removed)
    return frozenset(banned), frozenset(removed), seeds, n_pruned


def _check_required(
    knowledge: DomainKnowledge,
    names: list[str],
    links: dict[tuple[str, str, int], float],
    graph: MixedGraph,
    max_lag: int,
) -> None:
    """Raise when a required edge survived at no lag.

    ``required (x, y)`` names no lag, so PCMCI cannot force it; it validates
    instead. The requirement is met by a lagged link ``(x, y, tau)``, an
    oriented contemporaneous ``x -> y``, or a still-undirected
    contemporaneous pair, which stays compatible with ``x -> y``.
    """
    for cause, effect in sorted(knowledge.required):
        if any(src == cause and dst == effect and lag >= 1 for src, dst, lag in links):
            continue
        if cause in names and effect in names:
            edge = graph.get_edge(lagged_node(cause, 0), lagged_node(effect, 0))
            against = (lagged_node(effect, 0), lagged_node(cause, 0))
            if edge is not None and edge.directed_pair() != against:
                continue
        raise ValueError(
            f"required edge {cause} -> {effect} found no lag: no compatible link "
            f"survived at lags 0..{max_lag}"
        )


# -- step 4: orientation within the slice ---------------------------------------


def _orient_slice(
    names: list[str],
    adj: dict[str, set[str]],
    sepsets: dict[frozenset[str], frozenset[str]],
    *,
    seeds: dict[frozenset[str], tuple[str, str]] | None = None,
    banned: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[dict[frozenset[str], tuple[str, str]], int]:
    """Collider rule from the sepsets, then Meek R1 and R2 to a fixpoint.

    ``seeds`` are knowledge-compelled orientations (tiers and ``order``),
    entered as head demands before the colliders, so a data-driven arrowhead
    that disagrees follows the usual conflict convention: the edge stays
    undirected and is counted. A collider with either arrowhead on a
    ``banned`` direction is skipped whole (the convention PC uses for a
    forbidden arrowhead), and the Meek rules never orient into a banned
    direction.

    Returns the oriented pairs as ``{pair: (tail, head)}`` plus the count of
    edges left undirected because two demands wanted opposite heads.
    """
    demanded: dict[frozenset[str], set[str]] = {}
    if seeds:
        for pair in sorted(seeds, key=sorted):
            demanded.setdefault(pair, set()).add(seeds[pair][1])
    for z in names:
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue  # shielded
            separator = sepsets.get(frozenset((x, y)))
            if separator is None or z in separator:
                continue
            if (x, z) in banned or (y, z) in banned:
                continue  # a forbidden arrowhead skips the whole collider
            demanded.setdefault(frozenset((x, z)), set()).add(z)
            demanded.setdefault(frozenset((y, z)), set()).add(z)
    directed: dict[frozenset[str], tuple[str, str]] = {}
    conflicts = 0
    for pair in sorted(demanded, key=sorted):
        heads = demanded[pair]
        if len(heads) > 1:
            conflicts += 1  # both ends demanded into; leave the edge undirected
            continue
        (head,) = heads
        (tail,) = pair - {head}
        directed[pair] = (tail, head)
    changed = True
    while changed:
        changed = False
        for x in names:
            for z in sorted(adj[x]):
                if directed.get(frozenset((x, z))) != (x, z):
                    continue  # need x -> z
                for y in sorted(adj[z]):
                    if y == x or frozenset((z, y)) in directed or (z, y) in banned:
                        continue
                    if y not in adj[x]:  # R1: x -> z - y, x and y non-adjacent
                        directed[frozenset((z, y))] = (z, y)
                        changed = True
            for y in sorted(adj[x]):
                if frozenset((x, y)) in directed or (x, y) in banned:
                    continue
                for z in sorted(adj[x] & adj[y]):
                    chain = directed.get(frozenset((x, z))) == (x, z) and directed.get(
                        frozenset((z, y))
                    ) == (z, y)
                    if chain:  # R2: x -> z -> y with x - y undirected
                        directed[frozenset((x, y))] = (x, y)
                        changed = True
                        break
    return directed, conflicts
