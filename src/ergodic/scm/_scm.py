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
- :meth:`interventional` -- the distribution of ``Y`` under ``do(X = x)``
  through ``pm.do``: posterior-predictive draws of the outcome, so the mean
  estimates ``E[Y | do(X = x)]`` and the spread is that of ``Y`` itself.
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
from ..graph._primitives import latent_projection, topological_order
from ..identification import identify_effect
from . import _compose
from ._bounds import Sensitivity, latent_prior_sweep, linear_sensitivity_curve
from ._district import latent_structure
from ._mechanism import CategoricalGumbelMax, GaussianAdditive, resolve_mechanism
from ._result import CategoricalResult, SCMResult
from ._sampling import using_cores

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from ..data import Dataset
    from ..graph import MixedGraph
    from ..inference import ITEResult
    from ._mechanism import FitResult, Mechanism

    Array = npt.NDArray[np.float64]

# A do-target's value: a float for a continuous node, an integer category code for a
# categorical/binary node. Kept loose at the boundary so a caller can write do={"T": 1}.
DoValue = float | int


DISTRICT_SIZE_CAP = 8


class SCM:
    """A Bayesian structural causal model over a DAG or a latent-confounded ADMG.

    Build with ``SCM(graph, data)`` (mechanisms auto-assigned by schema) or pass
    ``mechanisms={node: spec}`` to override. ``spec`` is a built-in name
    (``"linear"``, ``"categorical"``, ``"binary"``) or a
    :class:`~ergodic.scm.Mechanism` instance. Call :meth:`fit`, then query.

    The graph may be a Markovian DAG, or an ADMG (``<->`` edges) / a DAG with
    ``latent=`` confounders. A bidirected edge is realized as one explicit per-unit
    latent common cause: the confounded district fits jointly (the latent absorbs the
    confounding) so an effect identified despite the confounding (front-door, the
    general criterion with the latent modeled) computes correctly. A genuinely
    non-identified effect still returns a :class:`~ergodic.scm.Sensitivity` through
    :meth:`ate`, never a naked point -- identification, not the fit, decides.
    Selection bias stays out of scope (identification raises on selection marks).
    A CPDAG, MAG, PAG, or mixed graph is refused: pick one world first.

    ``draws``, ``tune``, and ``chains`` set every NUTS run. ``cores`` caps the
    worker processes PyMC may use for the chains (default 1, so chains run one
    after another and a fit never fans out over the machine); raise it to run
    chains in parallel. The draws do not depend on it.
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
        cores: int = 1,
        random_state: int | None = None,
        district_size_cap: int = DISTRICT_SIZE_CAP,
    ) -> None:
        if graph.kind not in (GraphKind.DAG, GraphKind.ADMG):
            raise ValueError(
                f"SCM fits a DAG or an ADMG (one causal world); got kind="
                f"{graph.kind.value}. A MAG's or PAG's directed edges mean ancestry, not "
                "direct causation, and a CPDAG/MAG/PAG stands for a class of graphs, so "
                "fitting one as an ADMG would read the edges as mechanisms they are not. "
                "Pick one member (consistent_extension or dags_of_cpdag for a CPDAG; for "
                "a MAG/PAG, a DAG with explicit latents you can defend), or re-face a "
                "graph you do read as one world with graph.as_kind('dag') or "
                "graph.as_kind('admg')"
            )
        if cores < 1:
            raise ValueError(f"cores must be at least 1, got {cores}")
        if not graph.is_admg():
            raise ValueError(
                f"SCM fits a DAG or an ADMG (latent-confounded); got kind="
                f"{graph.kind.value}, which is not acyclic-directed-plus-bidirected. "
                "A CPDAG/MAG/PAG is an equivalence class, not a single SCM; pick a "
                "member (dags_of_cpdag / mags_of_pag) or pass a DAG/ADMG"
            )
        # The observed margin: nodes carrying data. A ``latent=`` node has no column
        # and is realized as a ``<->`` over the projection; an ADMG's nodes are all
        # observed. The latent projection turns either into an ADMG over the observed
        # margin, which is what the fit and identification read.
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
        self.cores = cores
        self.random_state = random_state
        self.district_size_cap = district_size_cap

        # The projected ADMG over the observed margin: ``latent=`` nodes become
        # ``<->`` edges; an ADMG that already has them passes through. Parents,
        # districts, and latents are all read off this single object, so a DAG with a
        # latent confounder and the equivalent ADMG fit identically.
        self._projection = latent_projection(graph)
        self._latent = latent_structure(self._projection)
        full_order = topological_order(self._projection)
        self._order: tuple[str, ...] = tuple(n for n in full_order if n in set(self.observed))
        specs = dict(mechanisms or {})
        self.mechanisms: dict[str, Mechanism] = {
            node: self._build_mechanism(specs.get(node), node) for node in self._order
        }
        self._warn_large_districts()
        self._fits: dict[str, FitResult] = {}

    def _build_mechanism(self, spec: Any, node: str) -> Mechanism:
        """Resolve ``node``'s mechanism over the projection, carrying its latents.

        The parents come from the projected ADMG (so a projection-introduced edge is
        honored), and the mechanism is given its incident latents and the subset it
        anchors the sign gauge for. A Markovian node lists no latents and reduces to
        the plain mechanism exactly.
        """
        from dataclasses import replace

        mech = resolve_mechanism(spec, self._projection, self.data, node)
        lats = self._latent.latents_of(node)
        if lats:
            # only the two built-in mechanisms carry the latent-loading fields; a custom
            # mechanism in a confounded district has no place to put the loadings.
            if not isinstance(mech, GaussianAdditive | CategoricalGumbelMax):
                raise TypeError(
                    f"node {node!r} sits in a confounded district (incident latents "
                    f"{list(lats)}) but its mechanism {type(mech).__name__} is not a "
                    "built-in Gaussian/categorical, so it cannot carry latent loadings; "
                    "custom mechanisms in a bidirected district are not supported"
                )
            mech = replace(
                mech,
                latents=lats,
                gauged_latents=self._latent.gauged_of(node),
            )
        return mech

    def _warn_large_districts(self) -> None:
        """Warn (not raise) when a confounded district exceeds the sampling-cost cap.

        Decision 5: a big district samples slower, not wrong, so the cap is advisory.
        """
        import warnings

        for nodes in self._latent.confounded_districts:
            if len(nodes) > self.district_size_cap:
                warnings.warn(
                    f"confounded district {list(nodes)} has {len(nodes)} nodes, over "
                    f"the cap {self.district_size_cap}; the joint fit will sample but "
                    "may be slow. Raise district_size_cap to silence, or screen the "
                    "graph (e.g. a discovery restrict_to) to shrink the district",
                    stacklevel=3,
                )

    # -- fit -----------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return len(self._fits) == len(self.mechanisms)

    def fit(self, *, parallel: bool = False) -> SCM:
        """Fit every node's mechanism, per node for singletons, jointly per district.

        A Markovian singleton district fits per node (cheap and exact); a
        confounded district (nodes sharing a ``<->``) does not factorize -- the latent
        couples it -- so it fits as one joint NUTS block with the explicit latents,
        the gauge, a non-centered parameterization, and a high ``target_accept``. The
        composed posterior merges the two the same way it merges per-node fits.

        Sequential by default (robust, CI-safe). ``parallel=True`` spreads the
        *singleton* nodes over a process pool with the BLAS/OMP threads pinned in the
        workers; a confounded district's joint fit stays in the main process (its
        chains already parallelize and its model is one block). Returns ``self``.
        """
        seed = 0 if self.random_state is None else self.random_state
        singletons = set(self._latent.singletons)
        single_mechs = {n: m for n, m in self.mechanisms.items() if n in singletons}
        with using_cores(self.cores):
            if parallel:
                fits = self._fit_parallel(single_mechs, seed)
            else:
                fits = {
                    node: mech.fit(
                        self.data, draws=self.draws, tune=self.tune, chains=self.chains, seed=seed
                    )
                    for node, mech in single_mechs.items()
                }
            for offset, nodes in enumerate(self._latent.confounded_districts):
                fits.update(self._fit_district(nodes, seed + offset))
        self._fits = fits
        return self

    def _fit_district(self, nodes: tuple[str, ...], seed: int) -> dict[str, FitResult]:
        """Joint fit of one confounded district (its nodes + their incident latents).

        Assembles the observed columns for every district node and every external
        parent (a parent outside the district, e.g. the front-door mediator that is
        its own singleton), names the district's latents, and runs the joint NUTS fit.
        """
        order = tuple(n for n in self._order if n in set(nodes))
        latents = self._district_latents(nodes)
        observed = self._district_observed(order)
        return _compose.fit_district(
            order,
            {n: self.mechanisms[n] for n in order},
            latents,
            observed,
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            seed=seed,
        )

    def _district_latents(self, nodes: tuple[str, ...]) -> tuple[str, ...]:
        """The latents incident to a district's nodes, deduplicated and sorted."""
        seen: set[str] = set()
        for node in nodes:
            seen.update(self._latent.latents_of(node))
        return tuple(sorted(seen))

    def _district_observed(self, order: tuple[str, ...]) -> dict[str, Any]:
        """Observed columns for the district fit: each node plus every external parent.

        A node's parents may sit outside the district (the canonical case is the
        front-door mediator, a singleton parent of an in-district node). The joint
        fit conditions on those observed parent columns, so the structural
        coefficient on a clean path is recovered while the latent soaks up the
        confounding on the bidirected one.
        """
        names: set[str] = set(order)
        for node in order:
            names.update(self.mechanisms[node].parents)
        out: dict[str, Any] = {}
        for name in names:
            mech = self.mechanisms.get(name)
            if mech is not None and isinstance(mech, CategoricalGumbelMax):
                # a categorical district node is observed as integer codes (its
                # likelihood is Categorical); the raw float code column also serves
                # as a discrete parent's design (the encoder one-hots it).
                out[name] = np.asarray(self.data.codes(name), dtype=np.int64)
            else:
                out[name] = np.asarray(self.data.matrix([name]).reshape(-1), dtype=np.float64)
        return out

    def _fit_parallel(self, mechanisms: Mapping[str, Mechanism], seed: int) -> dict[str, FitResult]:
        import concurrent.futures
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        out: dict[str, FitResult] = {}
        if not mechanisms:
            return out
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
                    self.cores,
                ): node
                for node, mech in mechanisms.items()
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
        """Observational posterior-predictive draws of one node (``kind="predictive"``).

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
            self._order,
            self.mechanisms,
            self._fits,
            target,
            n=n,
            seed=seed,
            latents=self._latent.latents,
        )
        if isinstance(self.mechanisms[target], CategoricalGumbelMax):
            return self._categorical_result("sample", target, draws)
        return SCMResult(
            query="sample", target=target, kind="predictive", identified=True, _draws=draws
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
        """The distribution of ``target`` under ``do``, through ``pm.do``.

        Returns posterior-predictive draws of the outcome for ``n`` units under the
        intervention: each draw is one unit's ``Y`` (parameter uncertainty plus the
        mechanism's noise). So ``value`` (the mean) estimates ``E[Y | do(X = x)]``,
        while ``std_error`` and ``interval()`` describe the spread of ``Y`` itself,
        not the uncertainty in that mean; the result's ``kind`` is
        ``"predictive"`` and its repr says "predictive interval". For an effect
        with a credible interval on the expectation, use :meth:`ate`.

        ``do`` maps each intervened node to its value (broadcast to a length-n
        vector internally, the verified shape gotcha; an integer code for a
        categorical/binary do-target, a float for a continuous one). ``target`` must
        not be an intervened node. A categorical target returns a
        :class:`CategoricalResult` (the post-intervention per-category marginal); a
        continuous target an :class:`SCMResult`.
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
            latents=self._latent.latents,
        )
        if isinstance(self.mechanisms[target], CategoricalGumbelMax):
            return self._categorical_result("interventional", target, draws)
        return SCMResult(
            query="interventional", target=target, kind="predictive", identified=True, _draws=draws
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
        # implemented; refuse rather than reuse the exact residual and mislabel the
        # result. Every shipped mechanism is invertible, so this guards a future
        # mechanism, not today's worlds.
        non_invertible = sorted(n for n, m in self.mechanisms.items() if not _is_invertible(m))
        if non_invertible:
            raise NotImplementedError(
                f"node(s) {non_invertible} use a non-invertible (probabilistic-only) "
                "mechanism; non-invertible-mechanism counterfactuals (the genuinely "
                "wider abduction posterior) are not supported. Every built-in "
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

    # -- individual treatment effects ----------------------------------------

    def ite(
        self,
        units: Dataset | Mapping[str, Any],
        treatment: str,
        outcome: str,
        *,
        levels: tuple[float, float] | None = None,
        shift: float | None = None,
    ) -> ITEResult:
        """Per-unit individual treatment effects, through the twin network.

        For each unit, the exact counterfactual contrast of ``outcome`` under two
        treatment worlds that share *that unit's* abducted noise (the same-person,
        two-worlds twin network). Returns one effect per unit, ordered as ``units`` --
        the structural-model sibling of a CATE estimator's
        :meth:`~ergodic.inference.OutcomeRegression.cate`, which returns a *conditional
        average* over the noise rather than the individual's own counterfactual.

        Two framings of the contrast, and you pick one:

        * an absolute **level** contrast (the default): ``Y | do(X=high) -
          Y | do(X=low)``, both worlds pinning the treatment to a fixed level for every
          unit, the very contrast :meth:`ate` averages. Pass ``levels=(low, high)``; it
          defaults to ``(0.0, 1.0)`` (a unit step, or the two codes of a binary
          treatment). Works for a continuous, binary, or (with explicit codes)
          categorical treatment, and for a continuous or binary outcome -- a binary
          outcome's per-unit effect is its counterfactual flip, in ``{-1, 0, 1}``.
        * a **shift** (a "shock"): ``Y | do(X = x_i + shift) - Y_factual``, perturbing
          each unit's treatment from its *own* factual value, against the world where
          nothing was done. Pass ``shift=delta`` (a continuous treatment and outcome).

        The two coincide for a linear-additive mechanism: the shift contrast is the
        constant ``coef * delta`` for every unit, equal to the matching level step. They
        part only once a mechanism is nonlinear, where the shift is the unit's local
        response at its own baseline. So with the built-in linear-Gaussian mechanisms
        the continuous ITE is constant across units (the coefficient), and genuine
        per-unit spread comes from a **binary or categorical outcome**, whose logistic
        response varies with each unit's baseline.

        Each effect is the exact point (invertible mechanisms). For an individual's
        posterior *interval*, call :meth:`counterfactual` on that one unit per arm with
        ``use_draws=True`` and difference the draws.
        """
        from ..inference import ITEResult

        self._require_fit()
        self._check_observable(treatment, "treatment")
        self._check_observable(outcome, "outcome")
        if levels is not None and shift is not None:
            raise ValueError(
                "pass either levels= or shift=, not both (they are two framings of the "
                "same per-unit contrast)"
            )
        out_mech = self.mechanisms[outcome]
        if isinstance(out_mech, CategoricalGumbelMax):
            n_cat = self._fits[outcome].n_categories or out_mech.n_categories
            if n_cat > 2:
                raise ValueError(
                    f"ite needs a numeric outcome; {outcome!r} has {n_cat} categories, "
                    f"which has no scalar individual effect. Read per-unit counterfactual "
                    f"distributions with counterfactual(units, do, {outcome!r}) instead"
                )
        if shift is not None:
            values = self._ite_shift(units, treatment, outcome, shift)
            framing = f"shift={shift}"
            note = (
                "structural individual treatment effects: per-unit "
                f"Y|do({treatment}=x_i+{shift}) - Y_factual through the twin network "
                "(abduct the unit's noise, predict on the shifted branch)"
            )
        else:
            low, high = self._ate_levels(treatment, levels)
            values = self._ite_levels(units, treatment, outcome, low, high)
            framing = f"levels=({low}, {high})"
            note = (
                "structural individual treatment effects: per-unit "
                f"Y|do({treatment}={high}) - Y|do({treatment}={low}) on shared abducted "
                "noise (the same-person, two-worlds twin network)"
            )
        factual = self._factual_values(units)
        out_col = np.asarray(factual.get(outcome), dtype=np.float64)
        n = int(values.shape[0])
        return ITEResult(
            treatment=treatment,
            outcome=outcome,
            covariates=(),
            n=n,
            kind="point",
            source="scm",
            framing=framing,
            note=note,
            _values=np.asarray(values, dtype=np.float64),
            _index=np.arange(n, dtype=np.int64),
            _factual=out_col,
        )

    def _ite_levels(
        self,
        units: Dataset | Mapping[str, Any],
        treatment: str,
        outcome: str,
        low: float,
        high: float,
    ) -> npt.NDArray[np.float64]:
        """The per-unit ``Y|do(X=high) - Y|do(X=low)`` contrast, on shared abducted noise.

        Both arms route through :meth:`counterfactual` (same units, same fixed seed), so
        each unit's noise is identical across the two worlds and the contrast is a
        coherent individual effect, not two independent draws. A continuous outcome
        differences the per-unit draws; a binary outcome differences the per-unit
        counterfactual codes (the flip, in ``{-1, 0, 1}``).
        """
        high_cf = self.counterfactual(units, {treatment: high}, outcome, use_draws=False)
        low_cf = self.counterfactual(units, {treatment: low}, outcome, use_draws=False)
        if isinstance(high_cf, CategoricalResult):
            assert isinstance(low_cf, CategoricalResult)
            high_u, low_u = high_cf.per_unit(), low_cf.per_unit()
            assert high_u is not None and low_u is not None
            return high_u.astype(np.float64) - low_u.astype(np.float64)
        assert isinstance(low_cf, SCMResult)
        return high_cf.draws() - low_cf.draws()

    def _ite_shift(
        self,
        units: Dataset | Mapping[str, Any],
        treatment: str,
        outcome: str,
        shift: float,
    ) -> npt.NDArray[np.float64]:
        """The per-unit shock ``Y|do(X = x_i + shift) - Y_factual``, from each unit's baseline.

        A continuous treatment and outcome only. The baseline world ``do(X = x_i)`` pins
        the treatment to its own factual value, which (on invertible mechanisms)
        reproduces the factual outcome exactly, so the subtrahend is just the factual
        column. The shifted world hands the engine a per-unit do-vector ``x_i + shift``
        -- the one query that reads a non-scalar do.
        """
        if isinstance(self.mechanisms[treatment], CategoricalGumbelMax):
            raise ValueError(
                f"shift= is for a continuous treatment; {treatment!r} is categorical/"
                f"binary, which has no additive shift. Use levels=(low_code, high_code)"
            )
        if isinstance(self.mechanisms[outcome], CategoricalGumbelMax):
            raise ValueError(
                f"shift= needs a continuous outcome; {outcome!r} is categorical/binary. "
                f"Use a levels= contrast (do(X=high) - do(X=low)) instead"
            )
        factual = self._factual_values(units)
        x_factual = np.asarray(factual[treatment], dtype=np.float64)
        shifted = _compose.counterfactual_continuous(
            self._order,
            self.mechanisms,
            self._fits,
            factual,
            {treatment: x_factual + shift},
            outcome,
            use_draws=False,
            seed=self._seed(6),
        )
        baseline = np.asarray(factual[outcome], dtype=np.float64)
        return shifted - baseline

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

        Identified by back-door adjustment, the no-path ``"null"`` case, or
        **front-door** computes the latent-aware do-contrast: a confounded district
        fits its explicit latent, so the structural mechanism on the causal path fits
        clean and ``do`` severs the treatment from the district latent, leaving the
        identified value (the front-door mediated effect, the C1 world's 1.2). The
        **instrument** strategy still raises with a pointer to
        :func:`~ergodic.inference.estimate_effect`: IV identification is linear and
        rests on the instrument's exclusion restriction, which the latent-district fit
        does not impose, so its treatment->outcome coefficient stays on the same
        non-identified ridge as the bow. The **general ID functional** likewise
        raises (its latent-fit do-contrast was not verified against
        ``estimate_effect`` to tolerance). ``engine="dml"`` is reserved for a
        cross-fit reuse path that is not available; it raises with a pointer.
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
                "engine='dml' (cross-fit reuse of the existing DML estimator) is not "
                "available; the default engine='posterior' returns the SCM plug-in "
                "contrast. Use ergodic.estimate_effect(graph, data, X, Y, "
                "strategy='dml') for a DML estimate"
            )
        if engine != "posterior":
            raise ValueError(f"unknown engine {engine!r}; use 'posterior' or 'dml'")
        # The latent-aware do-contrast computes back-door adjustment, the null case,
        # and front-door (the latent absorbs the X-Y confounding, so do() on the
        # treatment severs it from the district latent and the identified value
        # survives). Instruments and the general ID functional are NOT computed by
        # this contrast (decision 3, verify-then-widen): IV rests on an exclusion
        # restriction the latent fit does not impose, and the general functional was
        # not validated to tolerance here, so both raise with a pointer rather than
        # return a confident wrong number.
        if ident.strategy not in ("adjustment", "null", "frontdoor"):
            raise NotImplementedError(
                f"the effect of {treatment!r} on {outcome!r} is identified by the "
                f"{ident.strategy!r} criterion, which the SCM latent-fit do-contrast "
                f"does not yet compute (front-door, adjustment, and the null case do). "
                f"Use estimate_effect(graph, data, {treatment!r}, {outcome!r}), which "
                f"handles {ident.strategy}, or scm.ate(..., engine='dml')"
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
            latents=self._latent.latents,
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

        A linear-Gaussian treatment-outcome pair gets the exact analytic
        ``b(c) = (cov - c) / var(X)`` curve. A non-linear / non-Gaussian district (a
        categorical/binary treatment or outcome) gets the *latent-prior sweep*: the
        confounded district refits under a grid of confounding-strength priors and the
        range of the resulting do-contrasts is the bound (decision 2). Either way the
        result is a :class:`Sensitivity`, so a naked point read is impossible.
        """
        obstruction = (
            f"no adjustment, front-door, or instrument identifies the effect "
            f"({ident.estimand or 'not a function of the observed data alone'})"
        )
        if self._is_linear_gaussian_pair(treatment, outcome):
            x = np.asarray(self.data.matrix([treatment]).reshape(-1), dtype=np.float64)
            y = np.asarray(self.data.matrix([outcome]).reshape(-1), dtype=np.float64)
            # both at ddof=1 so cov_xy / var_x is the consistent OLS slope (np.cov is
            # ddof=1 by default; np.var defaults to ddof=0, the bug this fixes).
            cov_xy = float(np.cov(x, y)[0, 1])
            var_x = float(np.var(x, ddof=1))
            return linear_sensitivity_curve(
                treatment=treatment,
                outcome=outcome,
                cov_xy=cov_xy,
                var_x=var_x,
                obstruction=obstruction,
                diagnostics={"observational_slope": cov_xy / var_x},
            )
        return self._latent_prior_sweep_bounds(treatment, outcome, obstruction)

    def _is_linear_gaussian_pair(self, treatment: str, outcome: str) -> bool:
        """Whether the analytic ``b(c)`` curve applies: both endpoints Gaussian-additive.

        The closed-form Gaussian-moment derivation of ``b(c)`` needs a continuous
        treatment and outcome. A categorical/binary endpoint breaks it, so that case
        routes to the latent-prior sweep instead.
        """
        return isinstance(self.mechanisms.get(treatment), GaussianAdditive) and isinstance(
            self.mechanisms.get(outcome), GaussianAdditive
        )

    def _latent_prior_sweep_bounds(
        self, treatment: str, outcome: str, obstruction: str
    ) -> Sensitivity:
        """The non-linear non-identified bound: refit the district over a prior grid.

        Refits the confounded district holding everything fixed but the latent loading
        prior sd, and reads the do-contrast at each. The across-prior spread is the
        partial-identification interval (the pressure-test's prior-dependence: the
        answer is whatever the confounding prior says). The treatment must be a
        district endpoint for this to bite; a level pair defaults the same way
        :meth:`ate` does.
        """
        low, high = self._ate_levels(treatment, None)
        nodes = self._latent.node_district.get(treatment, (treatment,))
        order = tuple(n for n in self._order if n in set(nodes))
        latents = self._district_latents(tuple(nodes))
        observed = self._district_observed(order)
        with using_cores(self.cores):
            return latent_prior_sweep(
                order,
                {n: self.mechanisms[n] for n in order},
                latents,
                observed,
                treatment=treatment,
                outcome=outcome,
                low=low,
                high=high,
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                seed=self._seed(8),
                obstruction=obstruction,
            )

    # -- helpers -------------------------------------------------------------

    def _check_node(self, node: str) -> None:
        if node not in self.graph.nodes:
            nodes = sorted(self.graph.nodes)
            raise ValueError(f"unknown node {node!r}; the SCM's nodes are {nodes}")

    def _check_observable(self, node: str, role: str) -> None:
        """Reject a latent node where a fitted mechanism is required.

        A latent node has no data column and no fitted mechanism (the SCM fits the
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
        needs codes); a continuous node as floats. Only the *observed* nodes (those
        carrying a fitted mechanism) are read: a latent node has no column and is
        abducted implicitly into its district children's residuals (the district
        counterfactual pins the latent at its factual value),
        so iterating ``self.graph.nodes`` would ``KeyError`` on the latent.
        """
        from ..data import Dataset as _Dataset

        out: dict[str, npt.NDArray[Any]] = {}
        for node in self._order:
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
    cores: int,
) -> FitResult:
    with using_cores(cores):
        return mech.fit(data, draws=draws, tune=tune, chains=chains, seed=seed)
