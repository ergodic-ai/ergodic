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

Background knowledge is not wired in yet; passing it raises.
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
        """Run PCMCI+ and return the window graph with its provenance."""
        if knowledge is not None:
            raise NotImplementedError(
                "background knowledge enters temporal discovery in a later phase"
            )
        if not isinstance(data, TimeSeriesData | PanelData):
            raise TypeError(
                f"PCMCI needs TimeSeriesData or PanelData, got {type(data).__name__}; "
                "for cross-sectional rows use a tabular method (PC, GES, ...) via discover()"
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

        screen = {target: self._pc1(oracle, names, target) for target in names}
        mci = {target: self._mci(oracle, screen, target, deep_lag) for target in names}
        links: dict[tuple[str, str, int], float] = {}
        for target in names:
            for (source, tau), p in sorted(mci[target].items()):
                links[(source, target, tau)] = p
        n_lagged = len(links)

        mci_names = {
            target: frozenset(lagged_node(source, tau) for source, tau in mci[target])
            for target in names
        }
        adj, sepsets, last_p = self._contemporaneous(oracle, names, mci_names)
        directed, conflicts = _orient_slice(names, adj, sepsets)

        graph, n_contemporaneous, n_oriented = self._assemble(
            names, adj, directed, last_p, links, flags
        )
        diagnostics: dict[str, float] = dict(oracle.diagnostics())
        diagnostics.update(flags)
        diagnostics["n_lagged_links"] = float(n_lagged)
        diagnostics["n_contemporaneous"] = float(n_contemporaneous)
        diagnostics["n_oriented"] = float(n_oriented)
        diagnostics["orientation_conflicts"] = float(conflicts)
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

    def _pc1(self, oracle: CITest, names: list[str], target: str) -> dict[_Lagged, float]:
        """Surviving lagged parent candidates of one target, with their last p."""
        y = lagged_node(target, 0)
        pvals: dict[_Lagged, float] = {}
        for tau in range(1, self.max_lag + 1):
            for source in names:
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
    ) -> dict[_Lagged, float]:
        """Re-test each screened lagged parent under the MCI conditioning set."""
        y = lagged_node(target, 0)
        parents = screen[target]
        kept: dict[_Lagged, float] = {}
        for source, tau in sorted(parents):
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
    ) -> tuple[
        dict[str, set[str]],
        dict[frozenset[str], frozenset[str]],
        dict[frozenset[str], float],
    ]:
        """PC-stable over the lag-zero pairs, always conditioning on MCI parents."""
        adj: dict[str, set[str]] = {v: set(names) - {v} for v in names}
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


# -- step 4: orientation within the slice ---------------------------------------


def _orient_slice(
    names: list[str],
    adj: dict[str, set[str]],
    sepsets: dict[frozenset[str], frozenset[str]],
) -> tuple[dict[frozenset[str], tuple[str, str]], int]:
    """Collider rule from the sepsets, then Meek R1 and R2 to a fixpoint.

    Returns the oriented pairs as ``{pair: (tail, head)}`` plus the count of
    edges left undirected because two triples demanded opposite heads.
    """
    demanded: dict[frozenset[str], set[str]] = {}
    for z in names:
        for x, y in combinations(sorted(adj[z]), 2):
            if y in adj[x]:
                continue  # shielded
            separator = sepsets.get(frozenset((x, y)))
            if separator is None or z in separator:
                continue
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
                    if y == x or frozenset((z, y)) in directed:
                        continue
                    if y not in adj[x]:  # R1: x -> z - y, x and y non-adjacent
                        directed[frozenset((z, y))] = (z, y)
                        changed = True
            for y in sorted(adj[x]):
                if frozenset((x, y)) in directed:
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
