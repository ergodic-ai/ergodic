# SPDX-License-Identifier: Apache-2.0
"""The :class:`SCM`: a fitted Bayesian structural causal model.

An ``SCM`` is a :class:`~ergodic.graph.MixedGraph` plus a fitted mechanism per
node. It is a stateful object: build it over a graph and a dataset, ``fit`` it,
then query it. The queries are methods, because the fitted object holds the state
they read.

Fitting factorizes. For a Markovian DAG with complete data the posterior factors
over nodes (the pressure-test measured max cross-node parameter correlation 0.054,
shrinking as 1/sqrt(samples)), so each node is fitted on its own likelihood given
its observed parents. The default is sequential, which is robust and CI-safe;
``fit(parallel=True)`` spreads the nodes over a process pool with the BLAS/OMP
threads pinned in the workers (multi-threaded BLAS under multiprocess NUTS
oversubscribes the cores and the OS kills the workers).

Queries:

- :meth:`sample` -- observational draws.
- :meth:`do` -- a lightweight intervened view for chaining
  (``scm.do({"X": 1}).sample(...)``).
- :meth:`interventional` -- ``E[Y | do(X = x)]`` through ``pm.do``, carrying
  parameter uncertainty.
- :meth:`counterfactual` -- the explicit-noise twin-network counterfactual,
  abduct-then-predict, batched over units.
- :meth:`ate` -- routed through :func:`~ergodic.identification.identify_effect`
  first. Identified returns a posterior :class:`~ergodic.scm.SCMResult`; not
  identified returns a :class:`~ergodic.scm.Sensitivity` (bounds), never a naked
  point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..graph import GraphKind
from ..graph._primitives import topological_order
from ..identification import identify_effect
from . import _compose
from ._bounds import Sensitivity, linear_sensitivity_curve
from ._mechanism import CategoricalGumbelMax, GaussianAdditive, resolve_mechanism
from ._result import CategoricalResult, SCMResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from ..data import Dataset
    from ..graph import MixedGraph
    from ._mechanism import FitResult, Mechanism

    Array = npt.NDArray[np.float64]


class SCM:
    """A Markovian Bayesian structural causal model over a DAG.

    Build with ``SCM(graph, data)`` (mechanisms auto-assigned by schema) or pass
    ``mechanisms={node: spec}`` to override. ``spec`` is a built-in name
    (``"linear"``, ``"categorical"``, ``"binary"``) or a
    :class:`~ergodic.scm.Mechanism` instance. Call :meth:`fit`, then query.

    The graph must be a DAG: the MVP is Markovian (no latent confounding in the
    fit). A latent-confounded effect is still answered honestly through
    :meth:`ate`, which reads identification and returns bounds when the effect is
    not a function of the data alone.
    """

    def __init__(
        self,
        graph: MixedGraph,
        data: Dataset,
        *,
        mechanisms: Mapping[str, Any] | None = None,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        random_state: int | None = None,
    ) -> None:
        if graph.kind is not GraphKind.DAG:
            raise ValueError(
                f"SCM fits a DAG; got kind={graph.kind.value}. The MVP is Markovian "
                "(latent-district fitting is post-MVP); for a latent-confounded effect "
                "pass the DAG with the latent confounder marked latent= and read "
                "ate(), which returns bounds when the effect is not identified"
            )
        # Latent nodes (graph.latent) carry no data and are not fitted: the MVP
        # fits the observed margin. They still shape identification, so a
        # latent-confounded effect (a bow) is detected and answered with bounds.
        self.observed: tuple[str, ...] = tuple(sorted(graph.nodes - graph.latent))
        missing = set(self.observed) - set(data.variables)
        if missing:
            raise ValueError(
                f"every observed graph node needs a data column; missing: "
                f"{', '.join(sorted(missing))}"
            )
        self.graph = graph
        self.data = data
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.random_state = random_state
        full_order = topological_order(graph)
        # Fit the observed nodes only, in topological order; a latent node has no
        # column. A latent parent is dropped from each mechanism (observed_parents),
        # so its influence falls into the noise -- which is exactly why a
        # latent-confounded effect is non-identified, surfaced by ate().
        self._order: tuple[str, ...] = tuple(n for n in full_order if n not in graph.latent)
        specs = dict(mechanisms or {})
        self.mechanisms: dict[str, Mechanism] = {
            node: resolve_mechanism(specs.get(node), graph, data, node) for node in self._order
        }
        self._fits: dict[str, FitResult] = {}

    # -- fit -----------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return len(self._fits) == len(self.mechanisms)

    def fit(self, *, parallel: bool = False) -> SCM:
        """Fit every node's mechanism on its observed parents.

        Sequential by default (robust, CI-safe). ``parallel=True`` spreads nodes
        over a process pool with the BLAS/OMP threads pinned in the workers.
        Returns ``self``.
        """
        seed = 0 if self.random_state is None else self.random_state
        if parallel:
            self._fits = self._fit_parallel(seed)
        else:
            self._fits = {
                node: mech.fit(
                    self.data, draws=self.draws, tune=self.tune, chains=self.chains, seed=seed
                )
                for node, mech in self.mechanisms.items()
            }
        return self

    def _fit_parallel(self, seed: int) -> dict[str, FitResult]:
        import concurrent.futures
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        out: dict[str, FitResult] = {}
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=ctx, initializer=_pin_threads
        ) as pool:
            futures = {
                pool.submit(
                    _fit_one,
                    mech,
                    self.data,
                    self.draws,
                    self.tune,
                    self.chains,
                    seed,
                ): node
                for node, mech in self.mechanisms.items()
            }
            for fut in concurrent.futures.as_completed(futures):
                out[futures[fut]] = fut.result()
        return out

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("SCM is not fitted; call .fit() first")

    def _seed(self, offset: int) -> int:
        return (0 if self.random_state is None else self.random_state) + offset

    # -- observational + interventional --------------------------------------

    def sample(self, n: int = 1000) -> SCMResult:
        """Observational draws of every node (read the target with :meth:`marginal`).

        Returns the posterior predictive of the generative model over ``n`` units;
        the result carries the draws of the first node in topological order as a
        summary. Use :meth:`interventional` or :meth:`counterfactual` for a query
        on a specific node.
        """
        self._require_fit()
        self._require_continuous("sample")
        target = self._order[-1]
        draws = _compose.observational_draws(
            self._order, self.mechanisms, self._fits, target, n=n, seed=self._seed(1)
        )
        return SCMResult(
            query="sample", target=target, kind="posterior", identified=True, _draws=draws
        )

    def marginal(self, target: str, n: int = 1000) -> SCMResult:
        """Observational posterior-predictive draws of one node."""
        self._require_fit()
        self._check_node(target)
        self._require_continuous("marginal")
        draws = _compose.observational_draws(
            self._order, self.mechanisms, self._fits, target, n=n, seed=self._seed(2)
        )
        return SCMResult(
            query="sample", target=target, kind="posterior", identified=True, _draws=draws
        )

    def do(self, intervention: Mapping[str, float]) -> InterventionView:
        """A lightweight intervened view of the SCM for chaining.

        ``scm.do({"X": 1.0}).interventional("Y")`` reads the post-intervention
        distribution. The view shares the fit; it does not refit.
        """
        for node in intervention:
            self._check_node(node)
        return InterventionView(self, dict(intervention))

    def interventional(
        self, target: str, do: Mapping[str, float] | None = None, *, n: int = 2000
    ) -> SCMResult:
        """``E[Y | do(X = x)]`` and its posterior, through ``pm.do``.

        ``do`` maps each intervened node to its value (broadcast to a length-n
        vector internally, the verified shape gotcha). ``target`` must not be an
        intervened node. Carries parameter-posterior uncertainty via the merged
        ``idata``.
        """
        self._require_fit()
        self._check_node(target)
        intervention = dict(do or {})
        for node in intervention:
            self._check_node(node)
        self._require_continuous("interventional")
        draws = _compose.interventional_draws(
            self._order,
            self.mechanisms,
            self._fits,
            intervention,
            target,
            n=n,
            seed=self._seed(3),
        )
        return SCMResult(
            query="interventional", target=target, kind="posterior", identified=True, _draws=draws
        )

    # -- counterfactual ------------------------------------------------------

    def counterfactual(
        self,
        units: Dataset | Mapping[str, Any],
        do: Mapping[str, float],
        target: str,
        *,
        use_draws: bool = True,
    ) -> SCMResult | CategoricalResult:
        """The twin-network counterfactual for the given units under ``do``.

        Abduct each unit's structural noise on the factual graph, then predict
        ``target`` on the action branch reusing it. ``units`` is a dataset (or a
        mapping of node name to a column of factual values) holding every node's
        factual value per unit. For invertible mechanisms the counterfactual is
        exact (``use_draws=False`` collapses to the deterministic point; the
        default carries parameter uncertainty). A categorical ``target`` returns a
        :class:`~ergodic.scm.CategoricalResult` (the answer is a distribution).
        """
        self._require_fit()
        self._check_node(target)
        for node in do:
            self._check_node(node)
        factual = self._factual_values(units)

        mech = self.mechanisms[target]
        if isinstance(mech, CategoricalGumbelMax):
            codes, marginal = _compose.counterfactual_categorical(
                self.mechanisms,
                self._fits,
                factual,
                do,
                target,
                parents_of_target=mech.parents,
                seed=self._seed(5),
            )
            cats = self.data.categories(target) or tuple(range(mech.n_categories))
            invertible = True
            return CategoricalResult(
                query="counterfactual",
                target=target,
                categories=tuple(cats),
                probabilities=marginal,
                identified=invertible,
                _per_unit=codes,
            )

        if not isinstance(mech, GaussianAdditive):
            raise TypeError(f"no counterfactual for mechanism {type(mech).__name__} on {target!r}")
        # any non-invertible (probabilistic-only) node on the path makes the
        # counterfactual a wider posterior; flag it.
        identified = all(
            getattr(m, "capability", None) is not None and _is_invertible(m)
            for m in self.mechanisms.values()
        )
        draws = _compose.counterfactual_continuous(
            self._order,
            self.mechanisms,
            self._fits,
            factual,
            do,
            target,
            use_draws=use_draws,
        )
        return SCMResult(
            query="counterfactual",
            target=target,
            kind="posterior",
            identified=identified,
            _draws=draws,
        )

    # -- effects, identification-gated ---------------------------------------

    def ate(
        self,
        treatment: str,
        outcome: str,
        *,
        levels: tuple[float, float] = (0.0, 1.0),
        engine: str = "posterior",
        n: int = 2000,
    ) -> SCMResult | Sensitivity:
        """The average treatment effect, routed through identification first.

        ``identify_effect`` decides the return shape. Identified: the SCM posterior
        plug-in contrast ``E[Y | do(X = high)] - E[Y | do(X = low)]`` (a posterior
        :class:`~ergodic.scm.SCMResult`). Not identified: a
        :class:`~ergodic.scm.Sensitivity` (the ``b(c)`` curve over the assumed
        confounding), so a naked point is impossible. ``engine="dml"`` is the
        cross-fit reuse path (post-MVP; raises with a pointer for now).
        """
        self._require_fit()
        self._check_node(treatment)
        self._check_node(outcome)
        ident = identify_effect(self.graph, treatment, outcome)
        if not ident.identifiable:
            return self._bounds(treatment, outcome, ident)
        if engine == "dml":
            raise NotImplementedError(
                "engine='dml' (cross-fit reuse of the existing DML estimator) is "
                "post-MVP; the default engine='posterior' returns the SCM plug-in "
                "contrast. Use ergodic.estimate_effect(graph, data, X, Y, "
                "strategy='dml') for a DML estimate"
            )
        if engine != "posterior":
            raise ValueError(f"unknown engine {engine!r}; use 'posterior' or 'dml'")
        self._require_continuous("ate")
        low, high = levels
        draws = _compose.ate_contrast(
            self._order,
            self.mechanisms,
            self._fits,
            treatment,
            outcome,
            low=low,
            high=high,
            n=n,
            seed=self._seed(7),
        )
        return SCMResult(
            query="ate",
            target=outcome,
            kind="posterior",
            identified=True,
            diagnostics={"strategy": _strategy_code(ident.strategy)},
            _draws=draws,
        )

    def _bounds(self, treatment: str, outcome: str, ident: Any) -> Sensitivity:
        """The honest-bounds return for a non-identified effect.

        Reads the observed ``cov(X, Y)`` and ``var(X)`` and builds the analytic
        ``b(c) = (cov - c) / var(X)`` sensitivity curve. The obstruction string
        carries the hedge witness when the ID engine produced one.
        """
        x = np.asarray(self.data.matrix([treatment]).reshape(-1), dtype=np.float64)
        y = np.asarray(self.data.matrix([outcome]).reshape(-1), dtype=np.float64)
        cov_xy = float(np.cov(x, y)[0, 1])
        var_x = float(np.var(x))
        obstruction = (
            f"no adjustment, front-door, or instrument identifies the effect "
            f"({ident.estimand or 'not a function of the observed data alone'})"
        )
        return linear_sensitivity_curve(
            treatment=treatment,
            outcome=outcome,
            cov_xy=cov_xy,
            var_x=var_x,
            obstruction=obstruction,
            diagnostics={"observational_slope": cov_xy / var_x},
        )

    # -- helpers -------------------------------------------------------------

    def _check_node(self, node: str) -> None:
        if node not in self.graph.nodes:
            nodes = sorted(self.graph.nodes)
            raise ValueError(f"unknown node {node!r}; the SCM's nodes are {nodes}")

    def _require_continuous(self, what: str) -> None:
        """The PyMC generative/contrast paths build only Gaussian-additive nodes."""
        bad = [n for n, m in self.mechanisms.items() if not isinstance(m, GaussianAdditive)]
        if bad:
            raise NotImplementedError(
                f"{what} over a generative model with non-Gaussian nodes "
                f"({', '.join(sorted(bad))}) is not in the MVP; the categorical "
                "counterfactual path is supported through counterfactual()"
            )

    def _factual_values(self, units: Dataset | Mapping[str, Any]) -> dict[str, npt.NDArray[Any]]:
        """Per-node factual columns for the queried units, as numpy arrays.

        A categorical node is read as integer codes (the Gumbel-max abduction
        needs codes); a continuous node as floats.
        """
        from ..data import Dataset as _Dataset

        out: dict[str, npt.NDArray[Any]] = {}
        for node in self.graph.nodes:
            mech = self.mechanisms[node]
            if isinstance(units, _Dataset):
                col = self._column_from_dataset(units, node)
            else:
                col = np.asarray(units[node])
            if isinstance(mech, CategoricalGumbelMax):
                out[node] = np.asarray(col, dtype=np.int64)
            else:
                out[node] = np.asarray(col, dtype=np.float64)
        return out

    def _column_from_dataset(self, units: Dataset, node: str) -> npt.NDArray[Any]:
        mech = self.mechanisms[node]
        if isinstance(mech, CategoricalGumbelMax):
            return np.asarray(units.codes(node), dtype=np.int64)
        return np.asarray(units.matrix([node]).reshape(-1), dtype=np.float64)

    def __repr__(self) -> str:
        state = "fitted" if self.fitted else "unfitted"
        return f"SCM({len(self.graph.nodes)} nodes, {state}, n={self.data.n_rows})"


class InterventionView:
    """A lightweight intervened view of a fitted :class:`SCM`, for chaining.

    Returned by :meth:`SCM.do`. It carries the intervention and forwards
    :meth:`interventional`/:meth:`sample` with the do-map applied, sharing the
    parent's fit. Immutable; ``do`` returns a new view.
    """

    def __init__(self, scm: SCM, intervention: Mapping[str, float]) -> None:
        self._scm = scm
        self.intervention = dict(intervention)

    def do(self, more: Mapping[str, float]) -> InterventionView:
        merged = {**self.intervention, **more}
        return InterventionView(self._scm, merged)

    def interventional(self, target: str, *, n: int = 2000) -> SCMResult:
        return self._scm.interventional(target, self.intervention, n=n)

    def sample(self, target: str, *, n: int = 2000) -> SCMResult:
        """Alias for the interventional distribution of ``target`` under the view."""
        return self._scm.interventional(target, self.intervention, n=n)

    def __repr__(self) -> str:
        return f"InterventionView(do={self.intervention})"


def _is_invertible(mech: Mechanism) -> bool:
    from ._mechanism import Capability

    return mech.capability is Capability.INVERTIBLE


def _strategy_code(strategy: str) -> float:
    """A numeric tag for the identification strategy, for the diagnostics map."""
    order = {"null": 0.0, "adjustment": 1.0, "frontdoor": 2.0, "iv": 3.0, "id": 4.0}
    return order.get(strategy, -1.0)


# -- process-pool worker helpers (module level so they pickle) ----------------


def _pin_threads() -> None:
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def _fit_one(
    mech: Mechanism,
    data: Dataset,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> FitResult:
    return mech.fit(data, draws=draws, tune=tune, chains=chains, seed=seed)
