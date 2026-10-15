# SPDX-License-Identifier: Apache-2.0
"""Test a candidate graph against data: its implied independences, combined.

A causal graph implies conditional independences: a missing edge between
``u`` and ``v`` says some set of other variables separates them. The test
lists one such claim per non-adjacent pair (a *basis set*), checks each with a
conditional-independence test on the data, and combines the p-values with
Fisher's C. A small p-value says the data contradict at least one claim. A
large one says only that the data do not contradict them: Markov-equivalent
graphs imply the same independences, so the test cannot tell them apart, and
with few rows it has little power against any graph.

The basis set depends on what the graph says:

- **DAG.** Shipley's (2000) basis set: for each non-adjacent pair, ``u`` and
  ``v`` are independent given the parents of both. When ``u`` is not a
  descendant of ``v``, the local Markov property gives ``v`` independent of
  its non-descendants given its parents; ``u`` and every parent of ``u`` are
  non-descendants of ``v``, so weak union gives the claim.
- **CPDAG.** The basis set of its consistent extension (Dor and Tarsi 1992),
  a member DAG. Every member is Markov equivalent, so each claim holds in
  every member, and the class's own separation reading confirms it.
- **ADMG, MAG, a DAG with latent or selection nodes, PAG** (a PAG through one
  member MAG, :func:`~ergodic.graph.mag_of_pag`; every member has the same
  m-separations). m-separation over the observed variables: the parents of
  both when they separate the pair, otherwise the observed ancestors of the
  pair (and of selection), which separate any separable pair (Richardson and
  Spirtes 2002, Theorem 4.2), pruned to a minimal separating set. A pair no
  observed set separates (an inducing path) implies nothing and is reported
  as untestable.

Every conditioning set is checked against the graph's separation routine
before any data is touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..data import Dataset, TabularData
from ..graph import GraphKind, MixedGraph, consistent_extension, mag_of_pag, topological_order
from ._ci import CITest
from ._registry import _require_exchangeable, resolve_test
from ._stats import _chi2_sf

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

__all__ = ["GraphTest", "ImpliedIndependence", "test_graph"]

# A p-value of exactly zero (a tail below floating-point resolution) enters
# Fisher's C as this, so C stays finite; the combined p-value is then reported as
# zero, its limit, rather than a tail read off the floor.
_P_FLOOR = 2.2250738585072014e-308


@dataclass(frozen=True)
class ImpliedIndependence:
    """One claim of the basis set and its test on the data.

    Attributes:
        x: The earlier variable of the pair in the graph's topological order
            (names break ties).
        y: The later variable of the pair.
        given: The conditioning set, sorted.
        rule: How the set was chosen: ``"parents"`` (the parents of both) or
            ``"ancestral"`` (a minimal subset of the pair's observed ancestors).
        statistic: The test statistic.
        p_value: The test's p-value for "``x`` and ``y`` are independent given
            ``given``".
        p_holm: The Holm-adjusted p-value across every claim of the test.
        rejected: Whether ``p_holm <= alpha``, the claim the data contradict
            with the family-wise error rate held at ``alpha``.
        dof: The reference distribution's degrees of freedom, when the test
            reports them.
    """

    x: str
    y: str
    given: tuple[str, ...]
    rule: str
    statistic: float
    p_value: float
    p_holm: float
    rejected: bool
    dof: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """The claim as a JSON-ready dict."""
        return {
            "x": self.x,
            "y": self.y,
            "given": list(self.given),
            "rule": self.rule,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_holm": self.p_holm,
            "rejected": self.rejected,
            "dof": self.dof,
        }


@dataclass(frozen=True)
class GraphTest:
    """The result of :func:`test_graph`: Fisher's C over the basis set.

    ``p_value`` is a frequentist p-value for "the data are compatible with the
    candidate's implied independences". A small value is evidence against the
    graph. A large one is not evidence for it: every Markov-equivalent graph
    gets the same answer, and with few rows any graph passes.

    Attributes:
        statistic: Fisher's C, ``-2 * sum(log p_i)`` over the ``k`` claims.
        p_value: The chi-square tail of ``statistic`` on ``2k`` degrees of
            freedom; 1.0 when there is nothing to test, and 0.0 when a claim's
            own p-value is 0.0 (below floating-point resolution; that claim
            enters ``statistic`` as the smallest positive float).
        dof: ``2k``.
        k: How many claims were tested.
        alpha: The level for ``rejected`` and the Holm decisions.
        test: The conditional-independence test's name (``"linear"``,
            ``"discrete"``, ``"mixed"``, ...).
        kind: How the candidate was read: ``"dag"``, ``"cpdag"``, ``"admg"``,
            ``"mag"`` or ``"pag"``.
        n: Rows the tests used (complete rows over the observed variables).
        implications: The tested claims, smallest p-value first.
        untestable: Non-adjacent pairs that no observed set separates, so the
            candidate implies no independence for them.
        notes: Plain-language caveats about this run.
    """

    statistic: float
    p_value: float
    dof: int
    k: int
    alpha: float
    test: str
    kind: str
    n: int
    implications: tuple[ImpliedIndependence, ...] = field(repr=False)
    untestable: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    notes: tuple[str, ...] = field(default=(), repr=False)

    @property
    def rejected(self) -> bool:
        """Whether Fisher's combined p-value is at or below ``alpha``."""
        return self.k > 0 and self.p_value <= self.alpha

    @property
    def n_rejected(self) -> int:
        """How many claims the Holm procedure rejects at ``alpha``."""
        return sum(item.rejected for item in self.implications)

    def to_dict(self) -> dict[str, Any]:
        """The result as a JSON-ready dict."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "dof": self.dof,
            "k": self.k,
            "alpha": self.alpha,
            "rejected": self.rejected,
            "n_rejected": self.n_rejected,
            "test": self.test,
            "kind": self.kind,
            "n": self.n,
            "implications": [item.to_dict() for item in self.implications],
            "untestable": [list(pair) for pair in self.untestable],
            "notes": list(self.notes),
        }

    def to_frame(self) -> pd.DataFrame:
        """The claims as a dataframe, one row each, smallest p-value first."""
        import pandas as pd

        rows = [item.to_dict() for item in self.implications]
        columns = ["x", "y", "given", "rule", "statistic", "p_value", "p_holm", "rejected", "dof"]
        frame = pd.DataFrame(rows, columns=columns)
        frame["given"] = [", ".join(item.given) for item in self.implications]
        return frame

    def __repr__(self) -> str:
        verdict = "rejected" if self.rejected else "not rejected"
        return (
            f"GraphTest({self.kind}, {self.test} test, k={self.k}, C={self.statistic:.3f}, "
            f"p={self.p_value:.4g}, {verdict} at alpha={self.alpha}, "
            f"{self.n_rejected} claim(s) rejected after Holm)"
        )


# -- the basis set ----------------------------------------------------------------


@dataclass(frozen=True)
class _Claim:
    x: str
    y: str
    given: frozenset[str]
    rule: str


def _reading(candidate: MixedGraph) -> str:
    """Which separation the candidate's face asks for."""
    kind = candidate.kind
    if kind is GraphKind.DAG or (kind is GraphKind.MIXED and candidate.is_dag()):
        return "dag"
    if kind is GraphKind.ADMG or (kind is GraphKind.MIXED and candidate.is_admg()):
        return "admg"
    if kind in (GraphKind.CPDAG, GraphKind.MAG, GraphKind.PAG):
        return str(kind.value)
    raise ValueError(
        "test_graph reads a DAG, CPDAG, ADMG, MAG or PAG; this mixed-kind graph is none "
        "of the first two by structure, so tag what it is, e.g. graph.as_kind('pag')"
    )


def _pairs(order: tuple[str, ...], observed: frozenset[str]) -> list[tuple[str, str]]:
    """Observed pairs ``(x, y)`` with ``x`` before ``y`` in ``order``."""
    seen = [v for v in order if v in observed]
    return [(seen[i], seen[j]) for i in range(len(seen)) for j in range(i + 1, len(seen))]


def _dag_basis(
    graph: MixedGraph, check: Callable[[str, str, frozenset[str]], bool]
) -> list[_Claim]:
    """Shipley's basis set: each non-adjacent pair given the parents of both."""
    claims: list[_Claim] = []
    for x, y in _pairs(topological_order(graph), graph.nodes):
        if graph.has_edge(x, y):
            continue
        given = graph.parents(x) | graph.parents(y)
        if not check(x, y, given):  # pragma: no cover - a theorem, held by the tests
            raise RuntimeError(f"internal: the parents of {x!r} and {y!r} do not separate them")
        claims.append(_Claim(x, y, given, "parents"))
    return claims


def _anterior(graph: MixedGraph, nodes: frozenset[str]) -> frozenset[str]:
    """``nodes`` and their anterior: reachable backwards along ``->`` and along ``---``.

    On a DAG or ADMG this is the nodes and their ancestors; on a MAG it adds
    the undirected (selection) edges, as Richardson and Spirtes (2002) define it.
    """
    back: dict[str, set[str]] = {v: set() for v in graph.nodes}
    for edge in graph.edges:
        pair = edge.directed_pair()
        if pair is not None:
            back[pair[1]].add(pair[0])
        elif edge.is_undirected:
            back[edge.u].add(edge.v)
            back[edge.v].add(edge.u)
    out = set(nodes)
    stack = list(nodes)
    while stack:
        for prev in back[stack.pop()]:
            if prev not in out:
                out.add(prev)
                stack.append(prev)
    return frozenset(out)


def _msep_basis(
    graph: MixedGraph,
    observed: frozenset[str],
    check: Callable[[str, str, frozenset[str]], bool],
) -> tuple[list[_Claim], list[tuple[str, str]]]:
    """A separating set per non-adjacent observed pair, by m-separation.

    The parents of both when they separate; otherwise the observed anterior
    of the pair and of selection, pruned to a minimal set by dropping the
    latest variables first. Pairs that set does not separate are not
    separable by any observed set.
    """
    order = topological_order(graph)
    position = {v: i for i, v in enumerate(order)}
    base = _anterior(graph, graph.selection)
    claims: list[_Claim] = []
    untestable: list[tuple[str, str]] = []
    for x, y in _pairs(order, observed):
        if graph.has_edge(x, y):
            continue
        parents = (graph.parents(x) | graph.parents(y)) & observed
        if check(x, y, parents):
            claims.append(_Claim(x, y, parents, "parents"))
            continue
        pool = ((_anterior(graph, frozenset({x, y})) | base) & observed) - {x, y}
        if not check(x, y, pool):
            untestable.append((x, y))
            continue
        given = set(pool)
        for node in sorted(pool, key=lambda v: (-position[v], v)):
            trial = frozenset(given - {node})
            if check(x, y, trial):
                given.discard(node)
        claims.append(_Claim(x, y, frozenset(given), "ancestral"))
    return claims, untestable


def _basis(
    candidate: MixedGraph, reading: str, observed: frozenset[str]
) -> tuple[list[_Claim], list[tuple[str, str]]]:
    """The basis set for the candidate's reading, every set checked to separate."""
    selection = candidate.selection
    if reading == "dag" and not candidate.latent and not selection:

        def plain(x: str, y: str, given: frozenset[str]) -> bool:
            return candidate.m_separated(x, y, given)

        return _dag_basis(candidate, plain), []
    if reading == "cpdag":
        member = consistent_extension(candidate)
        if member is None:  # pragma: no cover - validate() already holds this
            raise ValueError("the CPDAG has no consistent extension, so it is not a valid CPDAG")

        def both(x: str, y: str, given: frozenset[str]) -> bool:
            return member.m_separated(x, y, given) and candidate.m_separated(x, y, given)

        return _dag_basis(member, both), []
    if reading == "pag":
        mag = mag_of_pag(candidate)

        def class_wide(x: str, y: str, given: frozenset[str]) -> bool:
            return mag.m_separated(x, y, given) and candidate.m_separated(x, y, given)

        return _msep_basis(mag, observed, class_wide)

    def given_selection(x: str, y: str, given: frozenset[str]) -> bool:
        return candidate.m_separated(x, y, given | selection)

    return _msep_basis(candidate, observed, given_selection)


# -- the combination ------------------------------------------------------------------


def _holm(p_values: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the input order."""
    k = len(p_values)
    order = sorted(range(k), key=lambda i: p_values[i])
    adjusted = [0.0] * k
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _fisher(p_values: list[float]) -> tuple[float, float]:
    """Fisher's C and its chi-square p-value on ``2k`` degrees of freedom.

    A claim with a p-value of exactly zero makes the combined p-value zero.
    """
    if not p_values:
        return 0.0, 1.0
    c = max(0.0, -2.0 * sum(math.log(max(p, _P_FLOOR)) for p in p_values))
    if min(p_values) <= 0.0:
        return c, 0.0
    return c, _chi2_sf(c, 2.0 * len(p_values))


# -- the entry point ------------------------------------------------------------------


def _bind(spec: CITest | str | None, data: TabularData, observed: tuple[str, ...]) -> CITest:
    """Resolve the test like ``discover`` does, on the candidate's observed columns.

    A test bound to data is refused, so the rows come from one place. The one
    bound test allowed is a graph oracle (:class:`DSeparationOracle`), which
    answers from its own graph and reads no rows.
    """
    if isinstance(spec, CITest) and spec.is_bound:
        if spec.data is not None:
            raise ValueError(
                f"{type(spec).__name__} is already bound to data; pass it unbound, e.g. "
                f"test={type(spec).__name__}(...) without data, and test_graph binds it "
                "to the candidate's columns of data"
            )
        absent = sorted(set(observed) - set(spec.variables))
        if absent:
            raise ValueError(f"the oracle does not answer about {absent}")
        return spec
    return resolve_test("auto" if spec is None else spec, data.select(observed))


def test_graph(
    candidate: MixedGraph,
    data: TabularData,
    *,
    test: CITest | str | None = None,
    alpha: float = 0.05,
) -> GraphTest:
    """Test the conditional independences a candidate graph implies against data.

    Lists one implied independence per non-adjacent pair of observed variables
    (Shipley's basis set for a DAG; m-separation for an ADMG, MAG or PAG; see
    the module docstring), tests each, ranks them, adjusts their p-values by
    Holm's method, and combines them with Fisher's C,
    ``C = -2 * sum(log p_i)``, referred to a chi-square on ``2k`` degrees of
    freedom. The chi-square reference assumes the ``k`` tests are independent,
    which Shipley shows for a DAG's basis set when the model holds; for ADMGs,
    MAGs and PAGs it is an approximation. The Holm decisions hold the
    family-wise error rate at ``alpha`` whatever the dependence.

    What the p-value means: how surprising the data would be if every implied
    independence held. It cannot separate Markov-equivalent graphs (they imply
    the same independences), it never tests an edge that is present (only
    absent ones), failing to reject does not confirm the graph, and its power
    grows with the number of rows.

    Args:
        candidate: The graph to test. A DAG, CPDAG, ADMG, MAG or PAG; a
            mixed-kind graph is read as a DAG or ADMG when its structure is one.
            Every node must be a data column, except nodes declared ``latent``
            or ``selection``.
        data: The rows, one column per observed node; extra columns are ignored.
        test: The conditional-independence test: ``None`` (the default) picks
            one from the observed columns' schema as ``discover`` does (partial
            correlation when all are continuous, G squared when all are
            discrete, the degenerate-Gaussian test otherwise), a name such as
            ``"linear"`` or ``"gcm"``, or a :class:`CITest`.
        alpha: The level for ``GraphTest.rejected`` and the Holm decisions.

    Returns:
        A :class:`GraphTest` with Fisher's C, its p-value, and the ranked claims.

    Raises:
        TypeError: ``candidate`` is not a graph or ``data`` is not tabular.
        ValueError: ``alpha`` is outside (0, 1), the candidate has a directed
            cycle or breaks the rules of its kind, or an observed node is not
            a column of ``data``.
    """
    if not isinstance(candidate, MixedGraph):
        raise TypeError(f"candidate must be a MixedGraph, got {type(candidate).__name__}")
    if not isinstance(data, TabularData):
        if isinstance(data, Dataset):
            _require_exchangeable(data)
        raise TypeError(
            f"data must be TabularData, got {type(data).__name__}; build it with "
            "ergodic.tabular(frame)"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not candidate.is_acyclic():
        raise ValueError("the candidate has a directed cycle; test_graph needs an acyclic graph")
    candidate.validate()
    reading = _reading(candidate)
    hidden = candidate.latent | candidate.selection
    if hidden and reading in ("cpdag", "mag", "pag"):
        raise ValueError(
            f"a {reading.upper()} has no latent or selection nodes: its nodes are the "
            "observed variables; declare latents on a DAG or ADMG instead"
        )
    observed = candidate.nodes - hidden
    columns = set(data.variables)
    missing = sorted(observed - columns)
    if missing:
        raise ValueError(
            f"candidate node(s) {missing} are not columns of the data; the data has "
            f"{sorted(columns)}. Declare an unmeasured node with latent=[...]"
        )

    notes: list[str] = []
    extra = sorted(columns - candidate.nodes)
    if extra:
        notes.append(f"Columns not in the candidate are ignored: {extra}.")
    seen_hidden = sorted(hidden & columns)
    if seen_hidden:
        notes.append(
            f"Latent or selection nodes are not tested, though in the data: {seen_hidden}."
        )

    claims, untestable = _basis(candidate, reading, observed)
    ordered = tuple(v for v in data.variables if v in observed)
    oracle = _bind(test, data, ordered)

    results = [oracle.test(c.x, c.y, c.given) for c in claims]
    p_values = [r.p_value for r in results]
    adjusted = _holm(p_values)
    statistic, p_value = _fisher(p_values)
    implications = tuple(
        sorted(
            (
                ImpliedIndependence(
                    x=c.x,
                    y=c.y,
                    given=tuple(sorted(c.given)),
                    rule=c.rule,
                    statistic=r.statistic,
                    p_value=r.p_value,
                    p_holm=adj,
                    rejected=adj <= alpha,
                    dof=r.dof,
                )
                for c, r, adj in zip(claims, results, adjusted, strict=True)
            ),
            key=lambda item: (item.p_value, item.x, item.y),
        )
    )

    n = oracle.n
    if not claims:
        notes.append(
            "Nothing to test: every pair of observed variables is adjacent or inseparable, "
            "so the candidate implies no independence and fits any data."
        )
    if untestable:
        shown = ", ".join(f"{x}-{y}" for x, y in untestable)
        notes.append(
            f"No observed set separates {shown} (an inducing path), so the candidate "
            "implies no independence for them."
        )
    largest = max((len(c.given) for c in claims), default=0)
    if claims and oracle.data is not None and n < max(50, 10 * (largest + 3)):
        notes.append(
            f"Few rows ({n}) for conditioning sets of up to {largest} variables: the "
            "reference distributions are asymptotic and power is low."
        )
    sparse = [r.dof for r in results if r.dof is not None and r.dof > 0 and n < 5 * r.dof]
    if sparse:
        notes.append(
            f"{len(sparse)} test(s) have fewer than 5 rows per degree of freedom, so their "
            "chi-square references are rough."
        )
    if reading in ("admg", "mag", "pag") or candidate.latent or candidate.selection:
        notes.append(
            "Fisher's C assumes the claims' tests are independent, which is shown for a "
            "DAG's basis set; with latent structure it is approximate, and the Holm "
            "decisions are the conservative reading."
        )

    return GraphTest(
        statistic=statistic,
        p_value=p_value,
        dof=2 * len(claims),
        k=len(claims),
        alpha=alpha,
        test=oracle.name,
        kind=reading,
        n=n,
        implications=implications,
        untestable=tuple(untestable),
        notes=tuple(notes),
    )


# pytest collects module-level functions named test_*; this one is a library
# function, so a test module that imports it must not collect it.
test_graph.__test__ = False  # type: ignore[attr-defined]
