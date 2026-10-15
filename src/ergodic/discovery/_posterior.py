# SPDX-License-Identifier: Apache-2.0
"""A posterior over DAGs under a method: from edge probabilities, or from a score.

:func:`graph_posterior` turns a method's output into a distribution over DAGs and
returns it as :class:`GraphSamples`, the distinct DAGs with their weights. Two
kinds of method feed it:

- **Edge probabilities** (DEC-082 decision 1). The target is
  ``pi(G) ~ exp(pair log score(G) + log_prior(G))`` over the DAGs that satisfy
  the hard knowledge: the pair log score reads the matrix as pair states,
  exactly as the decoders do, and ``log_prior`` is
  :meth:`~ergodic.DomainKnowledge.log_prior`. For independent marginals this is
  the model's own product of Bernoulli distributions restricted to DAGs; pair
  states are the same distribution conditioned on no reciprocal pair.
- **A score** (decision 5). The target is ``exp(score(G)) * prior(G)``, the
  prior uniform over DAGs times the knowledge. BDeu and BGe are log marginal
  likelihoods, so this is a Bayesian posterior under their priors; the BIC
  family only approximates one, and the result says so.

Comparisons never need the normalizer ``Z``: a ratio of two graphs' targets is a
ratio of their posterior probabilities.

The computation is exact when it can be. A target that factorizes by pair (edge
probabilities with no ancestral order constraint) runs the subset recursion of
:mod:`._exact` up to ``max_exact`` variables: exact normalizer, exact edge
marginals and exact draws. Up to four variables (five for a score) every DAG is
enumerated. Otherwise the structure MCMC of :mod:`._mcmc` runs, and its
diagnostics are reported and checked.

The numbers are posterior probabilities under the method: they inherit the
method's calibration, which the docs report, and they are not frequencies of
anything in the world.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..graph import dag
from ._exact import PairWeights, SubsetDP, all_dags, reachability
from ._mcmc import (
    EDGE_RHAT_LIMIT,
    ESS_PER_CHAIN,
    RHAT_LIMIT,
    PosteriorConvergenceWarning,
    effective_size,
    graphs_of,
    mean_error,
    random_start,
    run_chains,
    split_rhat,
)
from ._probabilities import EdgeProbabilities
from ._target import GraphTarget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ..data import TabularData
    from ..graph import MixedGraph
    from ..knowledge import DomainKnowledge
    from ._score import Score

    FloatArray = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]
    IntArray = npt.NDArray[np.int64]

PROBABILITY_TERM: Final = "posterior probability under the method"
"""How the docs and the outputs name these numbers; the calibration gate may change it."""
POSTERIOR_NOTE: Final = (
    f"{PROBABILITY_TERM}: relative to the method's edge probabilities or score and to "
    "the knowledge, and only as calibrated as the method; see the docs for its calibration"
)
APPROXIMATE_NOTE: Final = (
    "the score is a penalized likelihood, not a marginal likelihood, so this posterior "
    "is an approximation"
)
DEFAULT_MAX_EXACT: Final = 15
MAX_EXACT_LIMIT: Final = 18
ENUMERATION_LIMIT: Final = 4
SCORE_ENUMERATION_LIMIT: Final = 5
TIE_TOLERANCE: Final = 1e-9
"""Two graphs tie when their log targets differ by at most this, relative to the larger."""


@dataclass(frozen=True)
class FeatureProbability:
    """A posterior probability with its Monte Carlo standard error.

    Attributes:
        probability: The posterior probability under the method.
        mc_error: The Monte Carlo standard error; 0 when computed exactly.
        exact: Whether it was computed exactly (by enumeration or the exact recursion).
    """

    probability: float
    mc_error: float
    exact: bool

    def __float__(self) -> float:
        return self.probability


@dataclass(frozen=True, eq=False)
class GraphSamples:
    """Distinct DAGs with their posterior weights under one method.

    ``adjacency[k]`` is a DAG over ``variables`` (rows the source), and
    ``weights[k]`` its weight: its exact posterior probability when every DAG was
    enumerated, otherwise its share of the draws (exact draws from the subset
    recursion, or MCMC draws after warmup). ``counts`` holds the draws per graph
    and ``draws`` the index of each kept draw, one row per chain (empty when
    enumerated). ``log_targets[k]`` is the unnormalized log target.

    Build one with :func:`graph_posterior`. Every probability it returns is a
    posterior probability under the method, with a Monte Carlo error unless it
    is exact; see :func:`graph_posterior` for what "the method" is.

    Attributes:
        variables: Variable names, the row and column order of every matrix.
        method: ``"exact"`` or ``"mcmc"``.
        target: ``"pair_state"`` (edge probabilities) or ``"score:<name>"``.
        adjacency: ``(k, d, d)`` boolean, read-only: the distinct DAGs.
        weights: ``(k,)``, read-only, summing to 1.
        log_targets: ``(k,)``, read-only.
        counts: ``(k,)`` draws per DAG, read-only; zeros when enumerated.
        draws: ``(chains, draws)`` indices into ``adjacency``, read-only.
        seed: The seed the draws came from.
        knowledge: The knowledge the target includes, or ``None``.
        probabilities: The edge probabilities, for a probability target.
        score: The bound score, for a score target.
        diagnostics: How it was computed; see :func:`graph_posterior`.
    """

    variables: tuple[str, ...]
    method: str
    target: str
    adjacency: BoolArray = field(repr=False)
    weights: FloatArray = field(repr=False)
    log_targets: FloatArray = field(repr=False)
    counts: IntArray = field(repr=False)
    draws: IntArray = field(repr=False)
    seed: int | None
    knowledge: DomainKnowledge | None = field(repr=False)
    probabilities: EdgeProbabilities | None = field(repr=False)
    score: Score | None = field(repr=False)
    diagnostics: dict[str, Any] = field(repr=False)
    _target: GraphTarget = field(repr=False)
    _edges: FloatArray = field(repr=False)
    _edge_errors: FloatArray = field(repr=False)
    _best: tuple[BoolArray, float, str] = field(repr=False)
    _log_z: float | None = field(repr=False)
    _conditional: FloatArray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "adjacency",
            "weights",
            "log_targets",
            "counts",
            "draws",
            "_edges",
            "_edge_errors",
        ):
            getattr(self, name).setflags(write=False)

    # -- shape --------------------------------------------------------------------------

    @property
    def exhaustive(self) -> bool:
        """Whether every DAG is listed with its exact probability (enumeration)."""
        return self.draws.size == 0

    @property
    def n_draws(self) -> int:
        """Kept draws over all chains (0 when enumerated)."""
        return int(self.draws.size)

    @property
    def log_normalizer(self) -> float | None:
        """``log Z`` of the target when computed exactly, else ``None``."""
        return self._log_z

    @property
    def exact_probabilities(self) -> FloatArray | None:
        """Each listed DAG's exact posterior probability, ``exp(log target - log Z)``.

        ``None`` for MCMC, where ``Z`` is unknown.
        """
        if self._log_z is None:
            return None
        result: FloatArray = np.exp(self.log_targets - self._log_z)
        result.setflags(write=False)
        return result

    def graph(self, index: int) -> MixedGraph:
        """The ``index``-th distinct DAG as a graph."""
        return _to_dag(self.variables, self.adjacency[index])

    def top(self, k: int = 5) -> list[tuple[MixedGraph, float]]:
        """The ``k`` heaviest DAGs and their weights, heaviest first (ties by log target)."""
        order = np.lexsort((-self.log_targets, -self.weights))[:k]
        return [(self.graph(int(i)), float(self.weights[i])) for i in order]

    @property
    def best_graph(self) -> MixedGraph:
        """The best DAG found: the exact MAP when exact, else the best of the searches."""
        return _to_dag(self.variables, self._best[0])

    @property
    def best_log_target(self) -> float:
        """The best DAG's log target."""
        return self._best[1]

    # -- the target -------------------------------------------------------------------------

    def log_target(self, graph: MixedGraph) -> float:
        """The unnormalized log target of a DAG over ``variables``; ``-inf`` off the support.

        For edge probabilities it is the pair log score plus the knowledge's log
        prior; for a score, the score plus the log prior. Only differences matter:
        ``exp(log_target(G1) - log_target(G2))`` is the posterior ratio.
        """
        return self._target.log_target(_matrix_of(graph, self.variables, "log_target"))

    # -- probabilities ------------------------------------------------------------------------

    def edge_probabilities(self) -> EdgeProbabilities:
        """Posterior edge marginals, as pair states.

        ``values[i, j]`` is the posterior probability under the method of the edge
        ``variables[i] -> variables[j]``. A DAG holds at most one direction per
        pair, so the two directions and "no edge" are exclusive and the result has
        kind ``"pair_state"`` (``"frequency"`` means bootstrap stability, which
        this is not). The marginals are exact for the exact computation; for MCMC
        on edge probabilities they are Rao-Blackwellized (each draw contributes the
        pair's conditional state probabilities given the rest of the graph), else
        draw frequencies. ``provenance`` records the method, the estimator and the
        largest Monte Carlo error; :meth:`edge_errors` has them all.
        """
        provenance = {
            "source": "graph_posterior",
            "method": self.method,
            "target": self.target,
            "estimator": self.diagnostics["edge_estimator"],
            "max_mc_error": float(self._edge_errors.max()) if self._edge_errors.size else 0.0,
            "note": POSTERIOR_NOTE,
        }
        return EdgeProbabilities(
            self.variables, self._edges.copy(), "pair_state", False, provenance
        )

    def edge_errors(self) -> FloatArray:
        """The Monte Carlo standard error of each edge marginal; zeros when exact."""
        return self._edge_errors

    def probability(self, feature: Callable[[MixedGraph], bool]) -> FeatureProbability:
        """The posterior probability that ``feature(G)`` holds, with its Monte Carlo error.

        ``feature`` is any predicate on a DAG (a :class:`~ergodic.graph.MixedGraph`
        over ``variables``), called once per distinct DAG. Exact (``mc_error`` 0)
        when every DAG was enumerated. For exact draws the error is the binomial
        one, ``sqrt(p (1 - p) / draws)``; for MCMC it uses the feature's effective
        sample size across chains.
        """
        values = np.asarray(
            [bool(feature(self.graph(i))) for i in range(self.adjacency.shape[0])],
            dtype=np.float64,
        )
        return self._estimate(values)

    def _estimate(self, values: FloatArray) -> FeatureProbability:
        """The posterior mean of a per-graph value, with its Monte Carlo error."""
        if self.exhaustive:
            return FeatureProbability(float(np.dot(self.weights, values)), 0.0, True)
        traces = values[self.draws]
        if self.method == "exact":
            mean = float(traces.mean())
            spread = float(traces.var())
            return FeatureProbability(mean, math.sqrt(spread / traces.size), False)
        mean, error = mean_error(traces)
        return FeatureProbability(mean, error, False)

    def __repr__(self) -> str:
        how = "enumerated" if self.exhaustive else f"{self.n_draws} draws"
        return (
            f"GraphSamples({self.method}, {self.target}, {len(self.variables)} variables, "
            f"{how}, {self.adjacency.shape[0]} distinct DAGs)"
        )


# -- the facade ---------------------------------------------------------------------------


def graph_posterior(
    source: EdgeProbabilities | TabularData,
    *,
    score: Score | str | None = None,
    knowledge: DomainKnowledge | None = None,
    method: str = "auto",
    max_exact: int = DEFAULT_MAX_EXACT,
    samples: int = 10_000,
    chains: int = 4,
    warmup: int = 500,
    thin: int = 4,
    seed: int = 0,
) -> GraphSamples:
    """A posterior over DAGs under a method, exact when feasible, else by MCMC.

    From edge probabilities the target is ``exp(pair log score(G) + log_prior(G))``:
    the pair states the decoders read (independent marginals conditioned on no
    reciprocal pair, frequencies as they are, floored at ``1e-12``) and the
    knowledge's :meth:`~ergodic.DomainKnowledge.log_prior`. From data it is
    ``exp(score(G) + log_prior(G))``, uniform over DAGs times the knowledge. BDeu
    and BGe are marginal likelihoods; a BIC-family score only approximates one and
    ``diagnostics["approximate"]`` says so.

    Hard knowledge restricts the DAGs: banned edges, required edges, and the
    order statements read ancestrally (``a`` before ``b``: ``b`` is not an ancestor
    of ``a``, as :meth:`~ergodic.DomainKnowledge.violations` reads them). Soft
    statements weigh the pair states. ``confound`` statements speak about no
    directed edge, so a DAG posterior ignores them and lists them in
    ``diagnostics["ignored_confounders"]``.

    ``method="auto"`` picks, in order: every DAG enumerated (up to 4 variables, 5
    for a score); the exact subset recursion (a pair-factorized target, edge
    probabilities whose order knowledge reduces to banned edges, up to
    ``max_exact`` variables), which gives the exact normalizer and edge marginals
    and ``samples`` exact draws; otherwise ``chains`` structure MCMC chains. Chain
    0 starts from the greedy decode (edge probabilities) or a hill climb (a score),
    the others from random DAGs; each runs ``warmup`` sweeps and keeps one draw
    every ``thin`` sweeps, ``samples // chains`` per chain (a sweep proposes one
    move per movable pair). ``"exact"`` and ``"mcmc"`` force one; ``"exact"``
    raises when it does not apply.

    MCMC reports the rank-normalized split R-hat and bulk effective sample size of
    the log target and of every directed-edge indicator, and warns with
    :class:`PosteriorConvergenceWarning` when the log target's R-hat is above
    1.01, its effective sample size below 100 per chain, or an edge indicator
    with a posterior mean in [0.01, 0.99] has an R-hat above 1.05.

    Every probability derived from the result is a posterior probability under
    the method: as good as the method's own calibration, which the docs report.

    Args:
        source: An :class:`EdgeProbabilities`, or a ``TabularData`` with ``score``.
        score: For data: a :class:`Score` (bound or not) or a score name. ``None``
            picks ``"bge"`` for all-continuous and ``"bdeu"`` for all-discrete data.
        knowledge: Hard and soft statements about the variables.
        method: ``"auto"``, ``"exact"`` or ``"mcmc"``.
        max_exact: The largest variable count for the exact recursion (at most 18;
            it takes about ``3^n`` steps and ``2^n`` memory).
        samples: Total kept draws (exact draws, or MCMC draws over all chains).
        chains: MCMC chains (at least 2 for R-hat).
        warmup: MCMC warmup sweeps per chain.
        thin: MCMC sweeps between kept draws.
        seed: Seeds a ``numpy.random.Generator``; the global random state is untouched.

    Returns:
        :class:`GraphSamples`. ``diagnostics`` holds ``method``, ``engine``
        (``"enumeration"``, ``"subset_recursion"`` or ``"structure_mcmc"``),
        ``target`` (its kind, variables, whether it factorizes by pair, the
        ancestral constraints, ``approximate``), ``seed``, ``samples``,
        ``log_normalizer`` (exact only), ``n_distinct``, ``edge_estimator``
        (``"exact"``, ``"rao_blackwell"`` or ``"frequency"``), ``best_source``,
        ``ignored_confounders``, ``approximate`` and ``note``; for MCMC also
        ``chains``, ``draws_per_chain``, ``warmup``, ``thin``,
        ``proposals_per_sweep``, ``acceptance``, ``rhat_log_target``,
        ``ess_log_target``, ``max_edge_rhat``, ``min_edge_ess``, ``converged`` and
        ``convergence_problems``.

    Raises:
        TypeError: ``source`` is neither, or ``score`` is given with probabilities.
        ValueError: A bad setting, knowledge naming unknown variables or
            contradicting itself, or ``method="exact"`` where it does not apply.
    """
    _check_int("samples", samples, 1)
    _check_int("chains", chains, 1)
    _check_int("warmup", warmup, 0)
    _check_int("thin", thin, 1)
    _check_int("seed", seed, 0)
    _check_int("max_exact", max_exact, 1)
    if max_exact > MAX_EXACT_LIMIT:
        raise ValueError(
            f"max_exact must be at most {MAX_EXACT_LIMIT}: the exact recursion takes about "
            "3^n steps"
        )
    if method not in ("auto", "exact", "mcmc"):
        raise ValueError(f"method must be 'auto', 'exact' or 'mcmc', got {method!r}")
    if isinstance(source, EdgeProbabilities):
        if score is not None:
            raise TypeError("score= applies to data; edge probabilities carry their own target")
        target = GraphTarget.from_probabilities(source, knowledge)
        probabilities: EdgeProbabilities | None = source
        bound: Score | None = None
    else:
        from ..data import TabularData

        if not isinstance(source, TabularData):
            raise TypeError(
                "graph_posterior reads EdgeProbabilities or a TabularData with score=; "
                f"got {type(source).__name__}"
            )
        bound = _bind_score(source, score)
        names = tuple(source.dropna().variables)
        target = GraphTarget.from_score(bound, names, knowledge)
        probabilities = None
    size = target.size
    engine = _engine(target, method, max_exact)
    rng = np.random.default_rng(seed)
    diagnostics: dict[str, Any] = {
        "method": "mcmc" if engine == "structure_mcmc" else "exact",
        "engine": engine,
        "target": target.describe(),
        "seed": seed,
        "samples": samples,
        "max_exact": max_exact,
        "ignored_confounders": target.confounded,
        "approximate": target.approximate,
        "note": APPROXIMATE_NOTE if target.approximate else POSTERIOR_NOTE,
    }
    log_z: float | None = None
    conditional: FloatArray | None = None
    if engine == "enumeration":
        graphs, logs = _enumerate(target)
        log_z = _logsumexp(logs)
        weights = np.exp(logs - log_z)
        counts = np.zeros(graphs.shape[0], dtype=np.int64)
        draws = np.zeros((0, 0), dtype=np.int64)
        edges = np.einsum("k,kij->ij", weights, graphs.astype(np.float64))
        errors = np.zeros((size, size))
        best_index = int(np.argmax(logs))
        best = (graphs[best_index].copy(), float(logs[best_index]), "enumeration")
        diagnostics["edge_estimator"] = "exact"
    elif engine == "subset_recursion":
        weights_by_pair = PairWeights.from_logs(target.log_none, target.log_edge)
        dp = SubsetDP.build(weights_by_pair)
        log_z = dp.log_z + target.constant
        sampler = dp.sampler()
        stack = np.stack([sampler.draw(rng) for _ in range(samples)])
        graphs, index, counts = _distinct(stack)
        draws = index.reshape(1, samples)
        logs = target.stack_log_targets(graphs)
        weights = counts / samples
        edges = dp.edge_marginals()
        errors = np.zeros((size, size))
        best_graph = dp.map_graph()
        best = (best_graph, target.log_target(best_graph), "exact_map")
        diagnostics["edge_estimator"] = "exact"
    else:
        start, start_source = _start(target, probabilities, knowledge)
        starts = [start] + [random_start(target, rng) for _ in range(chains - 1)]
        per_chain = max(1, samples // chains)
        run = run_chains(target, starts, draws=per_chain, warmup=warmup, thin=thin, rng=rng)
        stack = graphs_of(run.states.reshape(-1, run.states.shape[-1]), size)
        graphs, index, counts = _distinct(stack)
        draws = index.reshape(chains, per_chain)
        logs = target.stack_log_targets(graphs)
        weights = counts / draws.size
        if target.pair_factorized:
            conditional = rao_blackwell(target, graphs)
            diagnostics["edge_estimator"] = "rao_blackwell"
        else:
            conditional = graphs.astype(np.float64)
            diagnostics["edge_estimator"] = "frequency"
        edges, errors = _traced(conditional, draws)
        best = _best_of(target, graphs, logs, start, start_source)
        # Each distinct graph's own log target, rounded: the running sums drift by
        # rounding, and ranks would read that drift as autocorrelation among ties.
        traced = np.round(logs[draws], 8)
        diagnostics.update(_convergence(traced, graphs, draws, run.acceptance))
        diagnostics.update(
            chains=chains,
            draws_per_chain=per_chain,
            warmup=warmup,
            thin=thin,
            proposals_per_sweep=run.proposals_per_sweep,
        )
    diagnostics["log_normalizer"] = log_z
    diagnostics["n_distinct"] = int(graphs.shape[0])
    diagnostics["best_source"] = best[2]
    if engine == "structure_mcmc" and not diagnostics["converged"]:
        warnings.warn(
            "structure MCMC may not have converged: "
            + "; ".join(diagnostics["convergence_problems"])
            + ". Raise samples, warmup or thin, or read diagnostics",
            PosteriorConvergenceWarning,
            stacklevel=2,
        )
    return GraphSamples(
        variables=target.names,
        method=diagnostics["method"],
        target=target.kind,
        adjacency=graphs,
        weights=np.asarray(weights, dtype=np.float64),
        log_targets=np.asarray(logs, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
        draws=np.asarray(draws, dtype=np.int64),
        seed=seed,
        knowledge=knowledge,
        probabilities=probabilities,
        score=bound,
        diagnostics=diagnostics,
        _target=target,
        _edges=np.asarray(edges, dtype=np.float64),
        _edge_errors=np.asarray(errors, dtype=np.float64),
        _best=best,
        _log_z=log_z,
        _conditional=conditional,
    )


# -- pieces ----------------------------------------------------------------------------------


def _check_int(name: str, value: Any, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int of at least {minimum}, got {value!r}")


def _bind_score(data: TabularData, score: Score | str | None) -> Score:
    from ._registry import resolve_score

    if score is None:
        kinds = [data.schema[name] for name in data.variables]
        if all(kind.is_continuous for kind in kinds):
            score = "bge"
        elif all(kind.is_discrete for kind in kinds):
            score = "bdeu"
        else:
            raise ValueError(
                "mixed continuous and discrete columns have no marginal-likelihood score "
                "here; pass score='mixed_bic' for an approximate posterior"
            )
    bound = resolve_score(score, data, algorithm="graph_posterior")
    if not bound.decomposable:
        raise ValueError(f"{type(bound).__name__} is not decomposable, so it cannot score DAGs")
    return bound


def _engine(target: GraphTarget, method: str, max_exact: int) -> str:
    size = target.size
    limit = SCORE_ENUMERATION_LIMIT if not target.pair_factorized else ENUMERATION_LIMIT
    if method == "mcmc":
        return "structure_mcmc"
    if size <= limit:
        return "enumeration"
    if target.pair_factorized and size <= max_exact:
        return "subset_recursion"
    if method == "exact":
        if target.pair_factorized:
            reason = f"{size} variables is above max_exact={max_exact}"
        elif target.score is not None:
            reason = (
                f"a score target is exact by enumeration only, up to "
                f"{SCORE_ENUMERATION_LIMIT} variables"
            )
        else:
            reason = (
                "the order knowledge is ancestral (a path, not an edge, could break it), so "
                f"the target does not factorize by pair; enumeration covers up to "
                f"{SCORE_ENUMERATION_LIMIT} variables"
            )
        raise ValueError(f"method='exact' does not apply: {reason}. Use method='mcmc'")
    return "structure_mcmc"


def _enumerate(target: GraphTarget) -> tuple[BoolArray, FloatArray]:
    """Every DAG on the support, with its log target."""
    size = target.size
    graphs = all_dags(size)
    low, high = np.triu_indices(size, k=1)
    forward = graphs[:, low, high]
    reverse = graphs[:, high, low]
    logs = target.constant + np.where(
        forward,
        target.log_edge[low, high],
        np.where(reverse, target.log_edge[high, low], target.log_none[low, high]),
    ).sum(axis=1)
    if target.ancestral:
        reach = reachability(graphs)
        for first, second in target.ancestral:
            logs = np.where(reach[:, second, first], -np.inf, logs)
    if target.score is not None:
        powers = 1 << np.arange(size)
        masks = np.einsum("kij,i->kj", graphs.astype(np.int64), powers)
        table = np.zeros((size, 1 << size))
        for node in range(size):
            for mask in range(1 << size):
                if not mask >> node & 1:
                    table[node, mask] = target.local(node, mask)
        logs = logs + table[np.arange(size)[None, :], masks].sum(axis=1)
    keep = np.isfinite(logs)
    if not keep.any():  # pragma: no cover - the knowledge builder refuses empty supports
        raise ValueError("no DAG satisfies the knowledge")
    return np.ascontiguousarray(graphs[keep]), logs[keep]


def _logsumexp(values: FloatArray) -> float:
    top = float(np.max(values))
    return top + math.log(float(np.exp(values - top).sum()))


def _distinct(stack: BoolArray) -> tuple[BoolArray, IntArray, IntArray]:
    """Distinct graphs in first-seen order, each draw's index, and the counts."""
    flat = np.packbits(stack.reshape(stack.shape[0], -1), axis=1)
    keys = [row.tobytes() for row in flat]
    seen: dict[bytes, int] = {}
    index = np.empty(len(keys), dtype=np.int64)
    for position, key in enumerate(keys):
        index[position] = seen.setdefault(key, len(seen))
    first = np.zeros(len(seen), dtype=np.int64)
    for position in range(len(keys) - 1, -1, -1):
        first[index[position]] = position
    counts = np.bincount(index, minlength=len(seen)).astype(np.int64)
    return np.ascontiguousarray(stack[first]), index, counts


def rao_blackwell(target: GraphTarget, graphs: BoolArray) -> FloatArray:
    """Each graph's conditional edge probabilities, pair by pair given the rest.

    For a pair-factorized target the state of one pair given every other edge has
    probability proportional to its state weight when the state keeps the graph
    acyclic, and 0 otherwise: "no edge" is always valid; ``i -> j`` is valid unless
    ``j`` reaches ``i`` without the pair's own edge. Averaging these conditionals
    over draws estimates the edge marginals with less variance than counting.
    """
    size = target.size
    reach = reachability(graphs)
    ints = graphs.astype(np.int64)
    longer = np.einsum("kij,kjl->kil", ints, reach.astype(np.int64)) > 0  # paths of length >= 2
    present = graphs
    reverse_present = np.swapaxes(graphs, 1, 2)
    absent = ~present & ~reverse_present
    # Valid i -> j given the rest of the graph.
    valid = np.where(
        absent,
        ~np.swapaxes(reach, 1, 2),
        np.where(present, True, ~np.swapaxes(longer, 1, 2)),
    )
    none = np.exp(
        target.log_none
        - np.maximum(np.maximum(target.log_none, target.log_edge), target.log_edge.T)
    )
    edge = np.exp(
        target.log_edge
        - np.maximum(np.maximum(target.log_none, target.log_edge), target.log_edge.T)
    )
    np.fill_diagonal(none, 0.0)
    np.fill_diagonal(edge, 0.0)
    forward = edge[None] * valid
    backward = np.swapaxes(forward, 1, 2)
    total = none[None] + forward + backward
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(total > 0, forward / total, 0.0)
    result[:, np.arange(size), np.arange(size)] = 0.0
    out: FloatArray = result
    return out


def _traced(values: FloatArray, draws: IntArray) -> tuple[FloatArray, FloatArray]:
    """Per-draw matrices' mean over draws and its Monte Carlo error (effective sample size)."""
    size = values.shape[1]
    traces = values[draws].reshape(draws.shape[0], draws.shape[1], size * size)
    series = np.moveaxis(traces, 2, 0)
    mean = series.mean(axis=(1, 2))
    variance = series.var(axis=(1, 2))
    ess = effective_size(series, rank=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.where(variance > 0.0, np.sqrt(variance / ess), 0.0)
    error = np.nan_to_num(error, nan=0.0, posinf=0.0)
    return mean.reshape(size, size), error.reshape(size, size)


def _start(
    target: GraphTarget,
    probabilities: EdgeProbabilities | None,
    knowledge: DomainKnowledge | None,
) -> tuple[BoolArray, str]:
    """Chain 0's start: the greedy decode for edge probabilities, else a hill climb."""
    if probabilities is not None:
        from ._decode import decode

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decoded = decode(probabilities, "greedy", knowledge=knowledge)
            graph = target.matrix(decoded.graph)
            if math.isfinite(target.log_target(graph)):
                return graph, "greedy_decode"
        except ValueError:
            pass
    graph = np.zeros((target.size, target.size), dtype=bool)
    for source, child in _required(target):
        graph[source, child] = True
    return hill_climb(target, graph), "hill_climb"


def _required(target: GraphTarget) -> list[tuple[int, int]]:
    out = []
    for i in range(target.size):
        for j in range(target.size):
            if i != j and target.log_none[i, j] == -np.inf and target.log_edge[j, i] == -np.inf:
                out.append((i, j))
    return out


def hill_climb(target: GraphTarget, graph: BoolArray, *, max_steps: int = 10_000) -> BoolArray:
    """Best-improvement ascent over single-pair moves from a valid DAG (deterministic)."""
    current = graph.copy()
    value = target.log_target(current)
    size = target.size
    for _ in range(max_steps):
        best_value, best_graph = value, None
        for i in range(size):
            for j in range(i + 1, size):
                for state in range(3):
                    candidate = current.copy()
                    candidate[i, j] = state == 1
                    candidate[j, i] = state == 2
                    if np.array_equal(candidate, current):
                        continue
                    trial = target.log_target(candidate)
                    if trial > best_value + 1e-12:
                        best_value, best_graph = trial, candidate
        if best_graph is None:
            return current
        current, value = best_graph, best_value
    return current  # pragma: no cover - ascent stops long before


def _best_of(
    target: GraphTarget, graphs: BoolArray, logs: FloatArray, start: BoolArray, source: str
) -> tuple[BoolArray, float, str]:
    """The better of the chain start and the best draw, then polished by a hill climb."""
    sampled = int(np.argmax(logs))
    start_value = target.log_target(start)
    if logs[sampled] > start_value:
        graph, origin = graphs[sampled], "best_draw"
    else:
        graph, origin = start, source
    polished = hill_climb(target, graph)
    value = target.log_target(polished)
    if value > target.log_target(graph) + 1e-12:
        origin += "+hill_climb"
    return polished, value, origin


def _convergence(
    log_targets: FloatArray, graphs: BoolArray, draws: IntArray, acceptance: tuple[float, ...]
) -> dict[str, Any]:
    chains = draws.shape[0]
    size = graphs.shape[1]
    out: dict[str, Any] = {"acceptance": acceptance}
    if chains < 2 or draws.shape[1] < 4:
        out.update(
            rhat_log_target=None,
            ess_log_target=None,
            max_edge_rhat=None,
            min_edge_ess=None,
            converged=False,
            convergence_problems=["R-hat needs at least 2 chains of at least 4 draws"],
        )
        return out
    rhat = float(split_rhat(log_targets[None])[0])
    ess = float(effective_size(log_targets[None])[0])
    indicators = graphs.reshape(graphs.shape[0], -1).astype(np.float64)[draws]
    series = np.moveaxis(indicators, 2, 0)
    means = series.mean(axis=(1, 2))
    watched = (means >= 0.01) & (means <= 0.99)
    edge_rhat = split_rhat(series[watched]) if watched.any() else np.zeros(0)
    edge_ess = effective_size(series[watched]) if watched.any() else np.zeros(0)
    max_rhat = float(np.nanmax(edge_rhat)) if np.isfinite(edge_rhat).any() else None
    min_ess = float(np.nanmin(edge_ess)) if np.isfinite(edge_ess).any() else None
    problems: list[str] = []
    if math.isfinite(rhat) and rhat > RHAT_LIMIT:
        problems.append(f"log-target R-hat {rhat:.3f} > {RHAT_LIMIT}")
    if math.isfinite(ess) and ess < ESS_PER_CHAIN * chains:
        problems.append(f"log-target effective sample size {ess:.0f} < {ESS_PER_CHAIN * chains}")
    if max_rhat is not None and max_rhat > EDGE_RHAT_LIMIT:
        flat = int(np.flatnonzero(watched)[int(np.nanargmax(edge_rhat))])
        problems.append(
            f"edge-indicator R-hat {max_rhat:.3f} > {EDGE_RHAT_LIMIT} (cell {divmod(flat, size)})"
        )
    out.update(
        rhat_log_target=rhat if math.isfinite(rhat) else None,
        ess_log_target=ess if math.isfinite(ess) else None,
        max_edge_rhat=max_rhat,
        min_edge_ess=min_ess,
        converged=not problems,
        convergence_problems=problems,
    )
    return out


def _to_dag(names: tuple[str, ...], matrix: BoolArray) -> MixedGraph:
    sources, targets = np.nonzero(matrix)
    return dag(
        [(names[int(s)], names[int(t)]) for s, t in zip(sources, targets, strict=True)],
        nodes=names,
    )


def _matrix_of(graph: MixedGraph, names: tuple[str, ...], caller: str) -> BoolArray:
    from ..graph import MixedGraph as _Graph

    if not isinstance(graph, _Graph):
        raise TypeError(f"{caller} reads a MixedGraph, got {type(graph).__name__}")
    observed = graph.nodes - graph.latent
    if observed != set(names):
        missing = sorted(set(names) - observed)
        extra = sorted(observed - set(names))
        raise ValueError(
            f"the graph's variables must match the posterior's; missing {missing}, extra {extra}"
        )
    loose = [e.glyph() for e in graph.edge_list() if not e.is_directed]
    if loose or graph.latent:
        raise ValueError(f"{caller} reads a DAG; the graph has {loose or sorted(graph.latent)}")
    matrix, _ = graph.to_adjacency(order=list(names))
    result: BoolArray = np.asarray(matrix, dtype=bool)
    return result
