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

# A do-target's value: a float for a continuous node, an integer category code for a
# categorical/binary node. Kept loose at the boundary so a caller can write do={"T": 1}.
DoValue = float | int


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

    def sample(self, n: int = 1000) -> SCMResult | CategoricalResult:
        """Observational draws of every node (read the target with :meth:`marginal`).

        Returns the posterior predictive of the generative model over ``n`` units;
        the result carries the draws of the last node in topological order (a sink)
        as a summary. A categorical sink returns a :class:`CategoricalResult` (the
        per-category marginal). Use :meth:`marginal` to read a specific node, or
        :meth:`interventional` / :meth:`counterfactual` for a query on one.
        """
        self._require_fit()
        target = self._order[-1]
        return self._observational(target, n=n, seed=self._seed(1))

    def marginal(self, target: str, n: int = 1000) -> SCMResult | CategoricalResult:
        """Observational posterior-predictive draws of one node.

        A continuous node returns an :class:`SCMResult`; a categorical/binary node a
        :class:`CategoricalResult` (the per-category marginal), exactly as
        :meth:`counterfactual` does for a categorical target.
        """
        self._require_fit()
        self._check_observable(target, "query target")
        return self._observational(target, n=n, seed=self._seed(2))

    def _observational(self, target: str, *, n: int, seed: int) -> SCMResult | CategoricalResult:
        """The shared observational readout: codes-to-marginal for a categorical target."""
        draws = _compose.observational_draws(
            self._order, self.mechanisms, self._fits, target, n=n, seed=seed
        )
        if isinstance(self.mechanisms[target], CategoricalGumbelMax):
            return self._categorical_result("sample", target, draws)
        return SCMResult(
            query="sample", target=target, kind="posterior", identified=True, _draws=draws
        )

    def do(self, intervention: Mapping[str, DoValue]) -> InterventionView:
        """A lightweight intervened view of the SCM for chaining.

        ``scm.do({"X": 1.0}).interventional("Y")`` reads the post-intervention
        distribution. The view shares the fit; it does not refit. A continuous
        do-target takes a float; a categorical/binary do-target takes its integer
        category code.
        """
        for node in intervention:
            self._check_observable(node, "do-target")
            # validate a categorical code early when the SCM is fitted (the resolved
            # class count lives on the fit); an unfitted chainable view still builds,
            # and interventional() re-validates before it samples regardless.
            if self.fitted:
                self._check_do_value(node, intervention[node])
        return InterventionView(self, dict(intervention))

    def interventional(
        self, target: str, do: Mapping[str, DoValue] | None = None, *, n: int = 2000
    ) -> SCMResult | CategoricalResult:
        """``E[Y | do(X = x)]`` and its posterior, through ``pm.do``.

        ``do`` maps each intervened node to its value (broadcast to a length-n
        vector internally, the verified shape gotcha; an integer code for a
        categorical/binary do-target, a float for a continuous one). ``target`` must
        not be an intervened node. A categorical target returns a
        :class:`CategoricalResult` (the post-intervention per-category marginal); a
        continuous target an :class:`SCMResult`. Carries parameter-posterior
        uncertainty via the merged ``idata``.
        """
        self._require_fit()
        self._check_observable(target, "query target")
        intervention = dict(do or {})
        for node in intervention:
            self._check_observable(node, "do-target")
            self._check_do_value(node, intervention[node])
        draws = _compose.interventional_draws(
            self._order,
            self.mechanisms,
            self._fits,
            intervention,
            target,
            n=n,
            seed=self._seed(3),
        )
        if isinstance(self.mechanisms[target], CategoricalGumbelMax):
            return self._categorical_result("interventional", target, draws)
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
        self._check_observable(target, "counterfactual target")
        for node in do:
            self._check_observable(node, "do-target")
            self._check_do_value(node, do[node])
        factual = self._factual_values(units)

        mech = self.mechanisms[target]
        if isinstance(mech, CategoricalGumbelMax):
            codes, _ = _compose.counterfactual_categorical(
                self.mechanisms,
                self._fits,
                factual,
                do,
                target,
                parents_of_target=mech.parents,
                seed=self._seed(5),
            )
            # the abducted-noise counterfactual is exact (invertible), so identified.
            return self._categorical_result("counterfactual", target, codes, per_unit=codes)

        if not isinstance(mech, GaussianAdditive):
            raise TypeError(f"no counterfactual for mechanism {type(mech).__name__} on {target!r}")
        # The twin-network counterfactual abducts the exact structural noise, which
        # is only valid for invertible mechanisms. A non-invertible (probabilistic-
        # only) node would need the genuinely wider abduction posterior, which is not
        # in the MVP; refuse rather than reuse the exact residual and mislabel the
        # result. Every shipped mechanism is invertible, so this guards a future
        # mechanism, not today's worlds.
        non_invertible = sorted(n for n, m in self.mechanisms.items() if not _is_invertible(m))
        if non_invertible:
            raise NotImplementedError(
                f"node(s) {non_invertible} use a non-invertible (probabilistic-only) "
                "mechanism; non-invertible-mechanism counterfactuals (the genuinely "
                "wider abduction posterior) are not in the MVP. Every built-in "
                "mechanism is invertible, so this only arises for a custom mechanism"
            )
        draws = _compose.counterfactual_continuous(
            self._order,
            self.mechanisms,
            self._fits,
            factual,
            do,
            target,
            use_draws=use_draws,
            seed=self._seed(6),
        )
        return SCMResult(
            query="counterfactual",
            target=target,
            kind="posterior",
            identified=True,
            _draws=draws,
        )

    # -- effects, identification-gated ---------------------------------------

    def ate(
        self,
        treatment: str,
        outcome: str,
        *,
        levels: tuple[float, float] | None = None,
        engine: str = "posterior",
        n: int = 2000,
    ) -> SCMResult | Sensitivity:
        """The average treatment effect, routed through identification first.

        ``identify_effect`` decides the return shape. Identified *by back-door
        adjustment* (or the no-path ``"null"`` case): the SCM posterior plug-in
        contrast ``E[Y | do(X = high)] - E[Y | do(X = low)]`` (a posterior
        :class:`~ergodic.scm.SCMResult`). Not identified: a
        :class:`~ergodic.scm.Sensitivity` (the ``b(c)`` curve over the assumed
        confounding), so a naked point is impossible.

        The treatment may be continuous, binary, or categorical. ``levels`` is the
        ``(low, high)`` contrast; it defaults to ``(0.0, 1.0)`` (the two codes of a
        binary treatment, or a unit step for a continuous one). A multi-category
        treatment has no canonical pair, so it *requires* explicit ``levels`` (the
        two category codes to contrast), mirroring the inference pillar's ``treated=``
        pattern. The outcome must be numeric (a categorical outcome has no scalar
        contrast and raises); confounders and mediators may be any type -- a
        categorical one is sampled in the generative model, so the back-door
        g-formula marginalizes over it correctly.

        Identified by a *non*-back-door criterion (front-door, instruments, the
        general ID functional) raises :class:`NotImplementedError`: the posterior
        engine computes back-door adjustment only, so it would otherwise return a
        confident wrong number for an effect it does not actually compute. Use
        :func:`~ergodic.inference.estimate_effect`, which handles those strategies.
        ``engine="dml"`` is the cross-fit reuse path (post-MVP; raises with a
        pointer for now).
        """
        self._require_fit()
        self._check_node(treatment)
        self._check_node(outcome)
        # the outcome must be numeric: a categorical contrast is not a scalar.
        if isinstance(self.mechanisms.get(outcome), CategoricalGumbelMax):
            raise ValueError(
                f"ate needs a numeric outcome; {outcome!r} is categorical/binary, which "
                f"has no scalar contrast. Read its post-intervention distribution with "
                f"interventional({outcome!r}, do={{{treatment!r}: ...}}) instead"
            )
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
        # The posterior plug-in computes back-door g-computation only (it pm.do's on
        # a generative model fit on each node's observed parents). For a front-door /
        # IV / general-ID effect that fit is confounded, so it would return a
        # confident WRONG point stamped identified=True. Gate on the strategy and
        # raise rather than mislead; estimate_effect handles those criteria.
        if ident.strategy not in ("adjustment", "null"):
            raise NotImplementedError(
                f"the effect of {treatment!r} on {outcome!r} is identified by the "
                f"{ident.strategy!r} criterion, but the SCM posterior engine computes "
                f"back-door adjustment only. Use estimate_effect(graph, data, "
                f"{treatment!r}, {outcome!r}), which handles {ident.strategy}, or "
                f"scm.ate(..., engine='dml')"
            )
        low, high = self._ate_levels(treatment, levels)
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

    def _ate_levels(
        self, treatment: str, levels: tuple[float, float] | None
    ) -> tuple[float, float]:
        """Resolve the ``(low, high)`` contrast for the treatment.

        A continuous or binary treatment defaults to ``(0.0, 1.0)`` (a unit step, or
        the two binary codes). A multi-category treatment has no canonical pair, so
        explicit ``levels`` are required; the two codes are validated against the
        treatment's category count.
        """
        mech = self.mechanisms[treatment]
        is_categorical = isinstance(mech, CategoricalGumbelMax)
        n_cat = self._fits[treatment].n_categories or getattr(mech, "n_categories", 0)
        multi = is_categorical and n_cat > 2
        if levels is None:
            if multi:
                raise ValueError(
                    f"treatment {treatment!r} has {n_cat} categories, which has no default "
                    f"contrast; pass levels=(low_code, high_code) (the two category codes "
                    f"to contrast), as the inference pillar's treated= picks one level"
                )
            return (0.0, 1.0)
        low, high = levels
        if is_categorical:
            for lvl in (low, high):
                v = float(lvl)
                if v != round(v):
                    raise ValueError(
                        f"level {lvl!r} for categorical/binary treatment {treatment!r} is "
                        f"not an integer category code (a fractional level is not snapped); "
                        f"pass integer codes (its codes are 0..{n_cat - 1})"
                    )
                code = round(v)
                if code < 0 or code >= n_cat:
                    raise ValueError(
                        f"level {lvl!r} is not a category code of {treatment!r}; its codes "
                        f"are 0..{n_cat - 1}"
                    )
        return (float(low), float(high))

    def _categorical_result(
        self,
        query: str,
        target: str,
        codes: npt.NDArray[Any],
        *,
        per_unit: npt.NDArray[Any] | None = None,
    ) -> CategoricalResult:
        """Build a :class:`CategoricalResult` from a categorical target's code draws.

        ``codes`` are the integer category draws (pooled over units and posterior
        draws for an interventional/observational marginal, or per-unit for a
        counterfactual). The marginal is the normalized bincount over the node's
        resolved class count, and the category labels come from the data column when
        named. ``per_unit`` carries genuine per-unit codes (a counterfactual) for the
        result's ``per_unit()`` accessor; an interventional marginal leaves it ``None``.
        Every shipped categorical mechanism is invertible, so ``identified=True``.
        """
        mech = self.mechanisms[target]
        n_cat = self._fits[target].n_categories or getattr(mech, "n_categories", 0)
        marginal = _compose.marginal_from_codes(codes, n_cat)
        cats = self.data.categories(target) or tuple(range(n_cat))
        pu = None if per_unit is None else np.asarray(per_unit, dtype=np.int64)
        return CategoricalResult(
            query=query,
            target=target,
            categories=tuple(cats),
            probabilities=marginal,
            identified=True,
            _per_unit=pu,
        )

    def _bounds(self, treatment: str, outcome: str, ident: Any) -> Sensitivity:
        """The honest-bounds return for a non-identified effect.

        Reads the observed ``cov(X, Y)`` and ``var(X)`` and builds the analytic
        ``b(c) = (cov - c) / var(X)`` sensitivity curve. The obstruction string
        carries the hedge witness when the ID engine produced one.
        """
        x = np.asarray(self.data.matrix([treatment]).reshape(-1), dtype=np.float64)
        y = np.asarray(self.data.matrix([outcome]).reshape(-1), dtype=np.float64)
        # both at ddof=1 so cov_xy / var_x is the consistent OLS slope (np.cov is
        # ddof=1 by default; np.var defaults to ddof=0, the bug this fixes).
        cov_xy = float(np.cov(x, y)[0, 1])
        var_x = float(np.var(x, ddof=1))
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

    def _check_observable(self, node: str, role: str) -> None:
        """Reject a latent node where a fitted mechanism is required.

        A latent node has no data column and no fitted mechanism (the MVP fits the
        observed margin), so intervening on it or reading its query would fail with
        an opaque ``KeyError`` from deep in the compose/PyMC path. Reject up front.
        """
        self._check_node(node)
        if node in self.graph.latent:
            raise ValueError(
                f"node {node!r} is latent (no data column, no fitted mechanism), so it "
                f"cannot be a {role}; the SCM fits the observed margin. The SCM's "
                f"observed nodes are {list(self.observed)}"
            )

    def _check_do_value(self, node: str, value: DoValue) -> None:
        """Validate a do-target's value against the node's kind.

        A categorical/binary do-target is pinned by its *integer category code*, so a
        fractional value (``do(C=0.5)``) or an out-of-range code (``do(C=99)`` on a
        4-category node) is a user error, not something to silently snap or treat as
        the reference category. This mirrors :meth:`_ate_levels`' code-range check on
        the do-path: ``interventional``/``do``/``counterfactual`` all route through it,
        so neither silently returns a confident wrong answer. A continuous do-target
        takes any float, so nothing is checked. Runs post-fit (the resolved class
        count lives on the fit).
        """
        mech = self.mechanisms[node]
        if not isinstance(mech, CategoricalGumbelMax):
            return
        v = float(value)
        if v != round(v):
            raise ValueError(
                f"do-value {value!r} for categorical/binary node {node!r} is not an "
                f"integer category code; pass an integer code (its codes are 0.."
                f"{(self._fits[node].n_categories or mech.n_categories) - 1}), not a "
                "fractional value (a fractional do-value is not snapped)"
            )
        code = round(v)
        n_cat = self._fits[node].n_categories or mech.n_categories
        if code < 0 or code >= n_cat:
            raise ValueError(
                f"do-value {value!r} is not a category code of {node!r}; its codes are "
                f"0..{n_cat - 1}"
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

    def __init__(self, scm: SCM, intervention: Mapping[str, DoValue]) -> None:
        self._scm = scm
        self.intervention = dict(intervention)

    def do(self, more: Mapping[str, DoValue]) -> InterventionView:
        merged = {**self.intervention, **more}
        return InterventionView(self._scm, merged)

    def interventional(self, target: str, *, n: int = 2000) -> SCMResult | CategoricalResult:
        return self._scm.interventional(target, self.intervention, n=n)

    def sample(self, target: str, *, n: int = 2000) -> SCMResult | CategoricalResult:
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
