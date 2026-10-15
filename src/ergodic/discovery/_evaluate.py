# SPDX-License-Identifier: Apache-2.0
"""Evaluate a candidate graph against a posterior over DAGs (DEC-082 decisions 2 and 3).

The measures, all relative to the method's posterior (:func:`graph_posterior`):

- **Log-ratio to the best graph**: ``log pi(candidate) - log pi(best)``. The
  normalizer cancels, so this needs no ``Z``. "Best" is the exact MAP when the
  computation is exact, else the better of the greedy decode (or a hill climb)
  and the best draw, polished by a hill climb.
- **Credible level**: the posterior mass of DAGs whose target is strictly above
  the candidate's, plus half the mass of DAGs that tie with it (the candidate
  included). It sits between the mass above the candidate and the mass of the
  smallest highest-posterior set that contains it, the mid-point of the two. A
  Bayesian tail probability: 0.03 means the candidate is among the method's most
  probable graphs, 0.99 that almost all the mass sits on graphs it prefers. It is
  not a p-value and tests nothing. Under the exact recursion the candidate's own
  half is exact and only the other DAGs' mass comes from the draws.
- **Claims**: each structural statement the candidate makes, with its posterior
  probability under the method and a Monte Carlo error.
- **Expected SHD** from the candidate to a posterior draw, against the expected
  SHD between two independent draws (the posterior's own spread). Both are sums
  over pairs of pair-state disagreement probabilities, so they come from the
  pair-state marginals.
- **P(acyclic)** under the raw matrix, for independent marginals only: the chance
  that independent coin flips with the model's probabilities form a DAG at all.

What a candidate claims depends on its kind:

- **DAG**: the state of every pair (``a --> b`` or no edge), plus the credible
  level, the log-ratio and the expected SHD.
- **CPDAG**: an adjacency for each undirected edge, the direction of each
  directed (compelled) edge, no edge for each absent pair, and the probability of
  the equivalence class (same skeleton and same v-structures).
- **PAG** (and MAG): adjacency for each edge and no edge for each absent pair. The
  marks are read ancestrally, as in a MAG without selection: a tail at ``a`` on
  an edge to ``b`` claims "``a`` is an ancestor of ``b``", an arrowhead at ``b``
  claims "``b`` is not an ancestor of ``a``", and a circle claims nothing.
- **ADMG**: each directed edge between observed variables. A bidirected edge, or a
  latent node with its edges, is a hidden common cause, which a posterior over
  DAGs of the observed variables cannot assess; it is listed with no probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..graph import GraphKind, Mark
from ._exact import PairWeights, SubsetDP, acyclic, reachability
from ._posterior import (
    DEFAULT_MAX_EXACT,
    POSTERIOR_NOTE,
    TIE_TOLERANCE,
    FeatureProbability,
    GraphSamples,
    graph_posterior,
)
from ._probabilities import EdgeProbabilities

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from ..graph import MixedGraph
    from ..knowledge import DomainKnowledge

    FloatArray = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

TAIL_NOTE: Final = (
    "a Bayesian tail probability: the posterior mass of DAGs the method ranks above the "
    "candidate (ties count half); not a p-value"
)
NOT_ASSESSABLE: Final = (
    "a hidden common cause is not a DAG state, so a posterior over DAGs cannot assess it"
)
P_ACYCLIC_DRAWS: Final = 20_000


@dataclass(frozen=True)
class GraphClaim:
    """One statement a candidate graph makes, and its posterior probability under the method.

    Attributes:
        claim: The statement, for example ``"a --> b"`` or ``"a is an ancestor of b"``.
        kind: ``"edge"``, ``"no_edge"``, ``"adjacent"``, ``"ancestor"``,
            ``"not_ancestor"``, ``"equivalence_class"`` or ``"confounded"``.
        variables: The two variables it is about (empty for the class claim).
        probability: Its posterior probability under the method; ``None`` when
            not assessable.
        mc_error: The Monte Carlo standard error (0 when exact; ``None`` when not
            assessable).
        exact: Whether ``probability`` was computed exactly.
        note: Why a claim is not assessable, else ``None``.
    """

    claim: str
    kind: str
    variables: tuple[str, ...]
    probability: float | None
    mc_error: float | None
    exact: bool
    note: str | None = None


@dataclass(frozen=True, eq=False)
class GraphEvaluation:
    """A candidate graph read against a posterior over DAGs; see :func:`evaluate_graph`.

    Attributes:
        candidate: The graph evaluated, as given.
        candidate_kind: ``"dag"``, ``"cpdag"``, ``"pag"``, ``"mag"`` or ``"admg"``.
        method: The posterior's method, ``"exact"`` or ``"mcmc"``.
        log_target: The candidate's unnormalized log target (DAGs only).
        best_graph: The best DAG found.
        best_log_target: Its log target.
        best_source: How it was found (``"exact_map"``, ``"enumeration"``,
            ``"greedy_decode"``, ``"best_draw"``, with ``"+hill_climb"`` when a hill
            climb improved it).
        log_ratio_to_best: ``log_target - best_log_target`` (DAGs only); 0 for the
            best graph, ``-inf`` for a DAG the hard knowledge rules out.
        credible_level: The posterior mass of DAGs ranked strictly above the
            candidate plus half the mass of ties (DAGs only).
        credible_level_error: Its Monte Carlo standard error (0 when exact).
        mass_above: The mass strictly above, without the tie half.
        claims: One :class:`GraphClaim` per statement.
        class_probability: The equivalence class's probability (CPDAGs only).
        expected_shd: Expected structural Hamming distance from the candidate to a
            posterior draw, counting pairs whose state differs (DAGs only).
        expected_shd_error: Its Monte Carlo standard error (0 when exact).
        expected_shd_between_draws: The same between two independent draws: a
            plug-in from the pair-state marginals, with no Monte Carlo error reported.
        p_acyclic: The probability that the raw independent marginals form a DAG
            (independent kinds only).
        p_acyclic_error: Its Monte Carlo standard error (0 when exact).
        not_assessable: The statements a DAG posterior cannot price.
        samples: The posterior the candidate was read against.
        diagnostics: How each number was computed.
    """

    candidate: MixedGraph = field(repr=False)
    candidate_kind: str
    method: str
    log_target: float | None
    best_graph: MixedGraph = field(repr=False)
    best_log_target: float
    best_source: str
    log_ratio_to_best: float | None
    credible_level: float | None
    credible_level_error: float | None
    mass_above: float | None
    claims: tuple[GraphClaim, ...] = field(repr=False)
    class_probability: FeatureProbability | None
    expected_shd: float | None
    expected_shd_error: float | None
    expected_shd_between_draws: float
    p_acyclic: float | None
    p_acyclic_error: float | None
    not_assessable: tuple[str, ...]
    samples: GraphSamples = field(repr=False)
    diagnostics: dict[str, Any] = field(repr=False)

    def claims_frame(self) -> pd.DataFrame:
        """The claims as a dataframe, one row each, weakest assessable claim first."""
        import pandas as pd

        rows = [
            {
                "claim": c.claim,
                "kind": c.kind,
                "probability": c.probability,
                "mc_error": c.mc_error,
                "exact": c.exact,
                "note": c.note,
            }
            for c in self.claims
        ]
        frame = pd.DataFrame(
            rows, columns=["claim", "kind", "probability", "mc_error", "exact", "note"]
        )
        return frame.sort_values("probability", kind="stable", na_position="last").reset_index(
            drop=True
        )

    def __repr__(self) -> str:
        parts = [f"{self.candidate_kind} vs {self.method} posterior"]
        if self.log_ratio_to_best is not None:
            parts.append(f"log-ratio to best {self.log_ratio_to_best:.3g}")
        if self.credible_level is not None:
            parts.append(f"credible level {self.credible_level:.3g}")
        assessed = [c for c in self.claims if c.probability is not None]
        parts.append(f"{len(assessed)} claims")
        if self.not_assessable:
            parts.append(f"{len(self.not_assessable)} not assessable")
        return f"GraphEvaluation({', '.join(parts)})"


def evaluate_graph(
    candidate: MixedGraph,
    source: EdgeProbabilities | GraphSamples,
    knowledge: DomainKnowledge | None = None,
    *,
    method: str = "auto",
    max_exact: int = DEFAULT_MAX_EXACT,
    samples: int = 10_000,
    chains: int = 4,
    warmup: int = 500,
    thin: int = 4,
    seed: int = 0,
) -> GraphEvaluation:
    """Read a candidate graph against a method's posterior over DAGs.

    With edge probabilities this first builds the posterior with
    :func:`graph_posterior` (``knowledge`` and the sampler settings go there);
    with :class:`GraphSamples`, for example a score posterior, it reads that one,
    and the settings are not used (pass ``knowledge`` to
    :func:`graph_posterior`, not here).

    The measures are those of DEC-082. For a DAG: the log-ratio to the best graph
    (the normalizer cancels), the credible level (the posterior mass of DAGs the
    method ranks strictly above the candidate, ties counted half; a Bayesian tail
    probability, not a p-value), every pair state as a claim, and the expected
    structural Hamming distance to a draw against that between two draws. For a
    CPDAG, PAG or ADMG: the claims that kind makes (see the module docstring for
    the reading of each mark) and, for a CPDAG, the equivalence class's
    probability. For independent marginals, P(acyclic) under the raw matrix:
    exact by the subset recursion up to ``max_exact`` variables, else from
    20,000 seeded draws.

    Pair-state claims are exact when the posterior is (enumeration or the subset
    recursion). Other claims, and the credible level, come from the draws with a
    Monte Carlo error unless every DAG was enumerated. All of them are posterior
    probabilities under the method: as calibrated as the method is.

    Args:
        candidate: A DAG, CPDAG, PAG, MAG or ADMG over the posterior's variables
            (an ADMG may add latent nodes). A ``mixed``-kind graph is read as a
            DAG when every edge is directed and it is acyclic.
        source: :class:`EdgeProbabilities`, or a :class:`GraphSamples`.
        knowledge: With edge probabilities, the knowledge for the posterior.
        method: ``"auto"``, ``"exact"`` or ``"mcmc"``; see :func:`graph_posterior`.
        max_exact: See :func:`graph_posterior`.
        samples: See :func:`graph_posterior`.
        chains: See :func:`graph_posterior`.
        warmup: See :func:`graph_posterior`.
        thin: See :func:`graph_posterior`.
        seed: See :func:`graph_posterior`; also seeds P(acyclic)'s draws.

    Returns:
        A :class:`GraphEvaluation`.

    Raises:
        TypeError: ``candidate`` is not a graph, or ``source`` is neither type.
        ValueError: The candidate's variables differ from the posterior's, a kind
            that cannot be read, or knowledge passed with samples.
    """
    from ..graph import MixedGraph

    if not isinstance(candidate, MixedGraph):
        raise TypeError(f"candidate must be a MixedGraph, got {type(candidate).__name__}")
    if isinstance(source, GraphSamples):
        if knowledge is not None:
            raise ValueError(
                "these samples already carry their knowledge; pass knowledge= to "
                "graph_posterior when you build them, not to evaluate_graph"
            )
        posterior = source
    elif isinstance(source, EdgeProbabilities):
        posterior = graph_posterior(
            source,
            knowledge=knowledge,
            method=method,
            max_exact=max_exact,
            samples=samples,
            chains=chains,
            warmup=warmup,
            thin=thin,
            seed=seed,
        )
    else:
        raise TypeError(
            "evaluate_graph reads EdgeProbabilities or GraphSamples; for data, build "
            f"samples with graph_posterior(data, score=...). Got {type(source).__name__}"
        )
    kind = _candidate_kind(candidate)
    names = posterior.variables
    _check_variables(candidate, names, kind)
    reader = _Reader(posterior)
    claims: list[GraphClaim] = []
    not_assessable: list[str] = []
    log_target: float | None = None
    log_ratio: float | None = None
    level: FeatureProbability | None = None
    above: FeatureProbability | None = None
    class_probability: FeatureProbability | None = None
    expected: float | None = None
    expected_error: float | None = None
    diagnostics: dict[str, Any] = {
        "candidate_kind": kind,
        "posterior_method": posterior.method,
        "posterior_engine": posterior.diagnostics["engine"],
        "converged": posterior.diagnostics.get("converged", True),
        "best_source": posterior.diagnostics["best_source"],
        "note": POSTERIOR_NOTE,
    }
    if kind == "dag":
        matrix = _adjacency(candidate, names)
        log_target = posterior._target.log_target(matrix)
        log_ratio = log_target - posterior.best_log_target
        claims = reader.dag_claims(matrix)
        level, above = reader.credible_level(log_target, matrix)
        expected, expected_error = reader.expected_shd(matrix)
        diagnostics["expected_shd_error"] = expected_error
        diagnostics["expected_shd_between_draws_note"] = (
            "a plug-in from the pair-state marginals (quadratic in them), without a Monte "
            "Carlo error"
        )
        diagnostics["credible_level_error_is_bound"] = bool(level is not None and level.bound)
        diagnostics["credible_level_method"] = reader.how
        diagnostics["credible_level_note"] = TAIL_NOTE
        diagnostics["tie_tolerance"] = TIE_TOLERANCE
        if posterior.log_normalizer is not None and math.isfinite(log_target):
            diagnostics["candidate_probability"] = math.exp(log_target - posterior.log_normalizer)
        if not math.isfinite(log_target):
            diagnostics["outside_support"] = (
                "the candidate breaks the hard knowledge (or is not acyclic), so its "
                "posterior probability is 0"
            )
        elif log_ratio > 0:
            diagnostics["above_best"] = (
                "the candidate scores above the best graph the search found; the search "
                "was not exact"
            )
    elif kind == "cpdag":
        claims, class_probability = reader.cpdag_claims(candidate)
    elif kind in ("pag", "mag"):
        claims = reader.ancestral_claims(candidate)
    else:
        claims, not_assessable = reader.admg_claims(candidate)
    p_acyclic, p_error, p_how = _p_acyclic(posterior, seed)
    diagnostics["p_acyclic_method"] = p_how
    return GraphEvaluation(
        candidate=candidate,
        candidate_kind=kind,
        method=posterior.method,
        log_target=log_target,
        best_graph=posterior.best_graph,
        best_log_target=posterior.best_log_target,
        best_source=posterior.diagnostics["best_source"],
        log_ratio_to_best=log_ratio,
        credible_level=None if level is None else level.probability,
        credible_level_error=None if level is None else level.mc_error,
        mass_above=None if above is None else above.probability,
        claims=tuple(claims),
        class_probability=class_probability,
        expected_shd=expected,
        expected_shd_error=expected_error,
        expected_shd_between_draws=reader.expected_shd_between_draws(),
        p_acyclic=p_acyclic,
        p_acyclic_error=p_error,
        not_assessable=tuple(not_assessable),
        samples=posterior,
        diagnostics=diagnostics,
    )


# -- reading candidates ----------------------------------------------------------------


def _candidate_kind(graph: MixedGraph) -> str:
    kind = graph.kind
    if kind is GraphKind.MIXED:
        if graph.is_dag() and not graph.latent:
            return "dag"
        raise ValueError(
            "a mixed-kind graph is read only when it is a DAG; tag it with "
            "as_kind('cpdag'), as_kind('pag') or as_kind('admg') to say how to read it"
        )
    return str(kind.value)


def _check_variables(graph: MixedGraph, names: tuple[str, ...], kind: str) -> None:
    if graph.latent and kind != "admg":
        raise ValueError(
            f"a {kind} candidate cannot have latent nodes here; write hidden causes as an ADMG"
        )
    if graph.selection:
        raise ValueError("selection variables cannot be read against a posterior over DAGs")
    observed = graph.nodes - graph.latent
    if observed != set(names):
        missing = sorted(set(names) - observed)
        extra = sorted(observed - set(names))
        raise ValueError(
            "the candidate's variables must match the posterior's: "
            f"missing {missing}, extra {extra}"
        )


def _adjacency(graph: MixedGraph, names: tuple[str, ...]) -> BoolArray:
    matrix, _ = graph.to_adjacency(order=list(names))
    result: BoolArray = np.asarray(matrix, dtype=bool)
    return result


class _Reader:
    """Posterior estimates for the claims, exact where the posterior is."""

    def __init__(self, posterior: GraphSamples) -> None:
        self.posterior = posterior
        self.names = posterior.variables
        self.index = {name: i for i, name in enumerate(self.names)}
        self.edges = posterior._edges
        self.errors = posterior._edge_errors
        self.pair_exact = posterior.diagnostics["edge_estimator"] == "exact"
        if posterior.exhaustive:
            self.how = "enumeration"
        elif posterior.method == "exact":
            self.how = "exact_draws"
        else:
            self.how = "mcmc"
        self._reach: BoolArray | None = None

    # -- pair states ------------------------------------------------------------------------

    def pair(self, a: int, b: int, states: tuple[int, ...]) -> FeatureProbability:
        """``P(the pair (a, b) is in one of states)``: 0 none, 1 a -> b, 2 b -> a."""
        if self.pair_exact:
            masses = (1.0 - self.edges[a, b] - self.edges[b, a], self.edges[a, b], self.edges[b, a])
            value = min(1.0, max(0.0, sum(masses[s] for s in states)))
            return FeatureProbability(value, 0.0, True)
        conditional = self.posterior._conditional
        assert conditional is not None
        parts = (
            1.0 - conditional[:, a, b] - conditional[:, b, a],
            conditional[:, a, b],
            conditional[:, b, a],
        )
        values = np.sum([parts[s] for s in states], axis=0)
        return self.posterior._estimate(np.asarray(values, dtype=np.float64), indicator=False)

    def pair_masses(self) -> FloatArray:
        """``(d, d, 3)``: each unordered pair's state masses, from the marginals."""
        size = len(self.names)
        out = np.zeros((size, size, 3))
        out[:, :, 1] = self.edges
        out[:, :, 2] = self.edges.T
        out[:, :, 0] = 1.0 - self.edges - self.edges.T
        return np.clip(out, 0.0, 1.0)

    def _claim(
        self, text: str, kind: str, pair: tuple[str, ...], estimate: FeatureProbability
    ) -> GraphClaim:
        return GraphClaim(text, kind, pair, estimate.probability, estimate.mc_error, estimate.exact)

    # -- DAG ---------------------------------------------------------------------------------

    def dag_claims(self, matrix: BoolArray) -> list[GraphClaim]:
        claims: list[GraphClaim] = []
        names = self.names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if matrix[i, j]:
                    claims.append(
                        self._claim(f"{a} --> {b}", "edge", (a, b), self.pair(i, j, (1,)))
                    )
                elif matrix[j, i]:
                    claims.append(
                        self._claim(f"{b} --> {a}", "edge", (b, a), self.pair(i, j, (2,)))
                    )
                else:
                    claims.append(
                        self._claim(
                            f"no edge between {a} and {b}", "no_edge", (a, b), self.pair(i, j, (0,))
                        )
                    )
        return claims

    def credible_level(
        self, value: float, matrix: BoolArray
    ) -> tuple[FeatureProbability, FeatureProbability]:
        posterior = self.posterior
        logs = posterior.log_targets
        if not math.isfinite(value):  # off the support: every DAG with mass is above it
            everything = FeatureProbability(1.0, 0.0, True)
            return everything, everything
        tolerance = TIE_TOLERANCE * max(1.0, abs(value))
        above = (logs > value + tolerance).astype(np.float64)
        tie = (np.abs(logs - value) <= tolerance).astype(np.float64)
        mass_above = posterior._estimate(above)
        log_z = posterior.log_normalizer
        if posterior.exhaustive or log_z is None:
            return posterior._estimate(above + 0.5 * tie), mass_above
        # Exact draws: the candidate's own probability is known exactly, so only the
        # other DAGs' share comes from the draws.
        own = math.exp(value - log_z)
        itself = np.all(posterior.adjacency == matrix[None], axis=(1, 2))
        others = posterior._estimate(above + 0.5 * np.where(itself, 0.0, tie))
        level = FeatureProbability(
            min(1.0, others.probability + own / 2.0), others.mc_error, False, others.bound
        )
        return level, mass_above

    def expected_shd(self, matrix: BoolArray) -> tuple[float, float]:
        """``E[SHD(candidate, G)]`` and its Monte Carlo error (0 when exact).

        It is linear in the pair-state marginals, so under MCMC it is the mean of a
        per-draw value (the expected disagreement given each draw's conditionals or
        indicators), whose error comes from its effective sample size.
        """
        size = len(self.names)
        low, high = np.triu_indices(size, k=1)
        states = np.where(matrix[low, high], 1, np.where(matrix[high, low], 2, 0))
        if self.pair_exact:
            masses = self.pair_masses()[low, high]
            return float((1.0 - masses[np.arange(low.size), states]).sum()), 0.0
        conditional = self.posterior._conditional
        assert conditional is not None
        forward = conditional[:, low, high]
        backward = conditional[:, high, low]
        agree = np.where(
            states == 1, forward, np.where(states == 2, backward, 1 - forward - backward)
        )
        estimate = self.posterior._estimate((1.0 - agree).sum(axis=1), indicator=False)
        return estimate.probability, estimate.mc_error

    def expected_shd_between_draws(self) -> float:
        masses = self.pair_masses()
        upper = np.triu_indices(len(self.names), k=1)
        same = (masses[upper] ** 2).sum(axis=1)
        return float((1.0 - same).sum())

    # -- CPDAG --------------------------------------------------------------------------------

    def cpdag_claims(self, graph: MixedGraph) -> tuple[list[GraphClaim], FeatureProbability]:
        claims: list[GraphClaim] = []
        names = self.names
        size = len(names)
        skeleton = np.zeros((size, size), dtype=bool)
        directed = np.zeros((size, size), dtype=bool)
        for edge in graph.edge_list():
            i, j = self.index[edge.u], self.index[edge.v]
            skeleton[i, j] = skeleton[j, i] = True
            pair = edge.directed_pair()
            if pair is not None:
                s, t = self.index[pair[0]], self.index[pair[1]]
                directed[s, t] = True
        for i in range(size):
            for j in range(i + 1, size):
                a, b = names[i], names[j]
                if not skeleton[i, j]:
                    claims.append(
                        self._claim(
                            f"no edge between {a} and {b}", "no_edge", (a, b), self.pair(i, j, (0,))
                        )
                    )
                elif directed[i, j]:
                    claims.append(
                        self._claim(f"{a} --> {b}", "edge", (a, b), self.pair(i, j, (1,)))
                    )
                elif directed[j, i]:
                    claims.append(
                        self._claim(f"{b} --> {a}", "edge", (b, a), self.pair(i, j, (2,)))
                    )
                else:
                    claims.append(
                        self._claim(
                            f"{a} and {b} adjacent", "adjacent", (a, b), self.pair(i, j, (1, 2))
                        )
                    )
        graphs = self.posterior.adjacency
        same = _same_class(graphs, skeleton, _v_structures(directed[None], skeleton[None])[0])
        estimate = self.posterior._estimate(same.astype(np.float64))
        claims.append(
            GraphClaim(
                "the graph's Markov equivalence class",
                "equivalence_class",
                (),
                estimate.probability,
                estimate.mc_error,
                estimate.exact,
            )
        )
        return claims, estimate

    # -- PAG / MAG ------------------------------------------------------------------------------

    def ancestry(self) -> BoolArray:
        if self._reach is None:
            self._reach = reachability(self.posterior.adjacency)
        return self._reach

    def ancestral_claims(self, graph: MixedGraph) -> list[GraphClaim]:
        claims: list[GraphClaim] = []
        names = self.names
        size = len(names)
        adjacent = np.zeros((size, size), dtype=bool)
        for edge in graph.edge_list():
            i, j = self.index[edge.u], self.index[edge.v]
            adjacent[i, j] = adjacent[j, i] = True
            a, b = sorted((edge.u, edge.v), key=self.index.__getitem__)
            low, high = self.index[a], self.index[b]
            claims.append(
                self._claim(
                    f"{a} and {b} adjacent", "adjacent", (a, b), self.pair(low, high, (1, 2))
                )
            )
            for here, there in ((edge.u, edge.v), (edge.v, edge.u)):
                mark = edge.mark_at(here)
                h, t = self.index[here], self.index[there]
                if mark is Mark.TAIL:
                    values = self.ancestry()[:, h, t].astype(np.float64)
                    claims.append(
                        self._claim(
                            f"{here} is an ancestor of {there}",
                            "ancestor",
                            (here, there),
                            self.posterior._estimate(values),
                        )
                    )
                elif mark is Mark.ARROW:
                    values = (~self.ancestry()[:, h, t]).astype(np.float64)
                    claims.append(
                        self._claim(
                            f"{here} is not an ancestor of {there}",
                            "not_ancestor",
                            (here, there),
                            self.posterior._estimate(values),
                        )
                    )
        for i in range(size):
            for j in range(i + 1, size):
                if not adjacent[i, j]:
                    a, b = names[i], names[j]
                    claims.append(
                        self._claim(
                            f"no edge between {a} and {b}", "no_edge", (a, b), self.pair(i, j, (0,))
                        )
                    )
        return claims

    # -- ADMG --------------------------------------------------------------------------------------

    def admg_claims(self, graph: MixedGraph) -> tuple[list[GraphClaim], list[str]]:
        claims: list[GraphClaim] = []
        hidden: list[str] = []
        latent = graph.latent
        for edge in graph.edge_list():
            if edge.u in latent or edge.v in latent:
                continue
            if edge.is_bidirected:
                text = f"{edge.u} <-> {edge.v}"
                hidden.append(text)
                claims.append(
                    GraphClaim(
                        text, "confounded", (edge.u, edge.v), None, None, False, NOT_ASSESSABLE
                    )
                )
                continue
            pair = edge.directed_pair()
            if pair is None:  # pragma: no cover - an ADMG holds directed and bidirected edges
                continue
            s, t = self.index[pair[0]], self.index[pair[1]]
            low, high = min(s, t), max(s, t)
            state = 1 if s == low else 2
            claims.append(
                self._claim(
                    f"{pair[0]} --> {pair[1]}", "edge", pair, self.pair(low, high, (state,))
                )
            )
        for node in sorted(latent):
            children = sorted(graph.children(node))
            text = f"hidden common cause {node} of {', '.join(children)}"
            hidden.append(text)
            claims.append(
                GraphClaim(text, "confounded", tuple(children), None, None, False, NOT_ASSESSABLE)
            )
        return claims, hidden


def _v_structures(graphs: BoolArray, skeletons: BoolArray) -> BoolArray:
    """``[k, a, b, c]``: ``a -> c <- b`` with ``a < b`` not adjacent."""
    size = graphs.shape[1]
    colliders = graphs[:, :, None, :] & graphs[:, None, :, :]
    open_pair = ~skeletons & ~np.eye(size, dtype=bool)[None]
    upper = np.triu(np.ones((size, size), dtype=bool), k=1)
    result: BoolArray = colliders & (open_pair & upper[None])[:, :, :, None]
    return result


def _same_class(graphs: BoolArray, skeleton: BoolArray, vstructures: BoolArray) -> BoolArray:
    """Which DAGs share the skeleton and the v-structures (Verma and Pearl)."""
    result = np.zeros(graphs.shape[0], dtype=bool)
    step = max(1, 4_000_000 // max(1, graphs.shape[1] ** 3))
    for start in range(0, graphs.shape[0], step):
        block = graphs[start : start + step]
        skeletons = block | np.swapaxes(block, 1, 2)
        same = np.all(skeletons == skeleton[None], axis=(1, 2))
        if same.any():
            v = _v_structures(block, skeletons)
            same &= np.all(v == vstructures[None], axis=(1, 2, 3))
        result[start : start + step] = same
    return result


# -- P(acyclic) under the raw matrix ------------------------------------------------------------


def _p_acyclic(posterior: GraphSamples, seed: int) -> tuple[float | None, float | None, str | None]:
    probabilities = posterior.probabilities
    if probabilities is None or probabilities.kind != "independent":
        return None, None, None
    values = probabilities.values
    size = values.shape[0]
    if size <= int(posterior.diagnostics.get("max_exact", DEFAULT_MAX_EXACT)):
        return p_acyclic_exact(values), 0.0, "exact"
    rng = np.random.default_rng([seed, 152])
    hits = 0
    total = P_ACYCLIC_DRAWS
    for start in range(0, total, 2_000):
        count = min(2_000, total - start)
        draws = rng.random((count, size, size)) < values[None]
        draws[:, np.arange(size), np.arange(size)] = False
        hits += int(acyclic(draws).sum())
    p = hits / total
    return p, math.sqrt(p * (1.0 - p) / total), "monte_carlo"


def p_acyclic_exact(values: FloatArray) -> float:
    """``P(independent coin flips with these probabilities form a DAG)``, exactly."""
    size = values.shape[0]
    with np.errstate(divide="ignore"):
        log_p = np.log(values)
        log_q = np.log1p(-values)
    log_none = log_q + log_q.T
    log_edge = log_p + log_q.T
    upper = np.triu_indices(size, k=1)
    possible = (
        np.isfinite(log_none[upper]) | np.isfinite(log_edge[upper]) | np.isfinite(log_edge.T[upper])
    )
    if not possible.all():
        return 0.0
    weights = PairWeights.from_logs(log_none, log_edge)
    dp = SubsetDP.build(weights)
    total = dp.log_z
    return float(min(1.0, math.exp(total))) if math.isfinite(total) else 0.0
