# SPDX-License-Identifier: Apache-2.0
"""The bridge from the per-node fitting view to the generative view.

The pressure-test settled the fitting architecture: fit each node alone (cheap,
exact for a Markovian DAG with complete data), then *compose* a generative model
for the queries. This module is the only one that touches a raw PyMC transform
(``pm.do``, ``observe``, ``sample_posterior_predictive``); the rest of the SCM
never does. It encapsulates every verified gotcha so they live in one place.

What it owns:

- **Interventional sampling.** Build one generative PyMC model whose nodes depend
  on parent RVs through the structural equation, with the fitted parameters as
  named RVs (matching the merged posterior's names exactly, so the by-name match
  cannot silently sample garbage). ``pm.do`` takes a length-n vector, not a
  scalar; after ``do(X)`` node X leaves the predictive set, so a downstream node
  is read. Posterior uncertainty rides through the merged ``idata``.
- **Counterfactuals, the explicit-noise twin network.** Abduct the units' noise on
  the factual graph, then predict on an action branch that reuses it. For
  invertible mechanisms the abduction is the exact residual and the prediction is
  deterministic numpy over the parameter posterior (machine precision, the
  ``Y_cf = y + b(x' - x)`` anchor). For the categorical Gumbel-max node the
  abduction is the truncated-Gumbel numpy sampler. A general (non-invertible)
  node falls back to the batched ``observe`` + ``pm.sample`` abduction over a
  units axis (the ~50-70x batched route).

The gotchas, all handled here: ``do`` is a length-n vector; the intervened node
leaves the predictive set; ``sample_posterior_predictive`` matches by name and is
silent on a mismatch (one naming scheme everywhere); you cannot ``observe`` a
``Deterministic`` (the structural form puts noise on explicit RVs); ``do`` and
``observe`` return new models; arviz is v2 (``from_dict``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._mechanism import (
    CategoricalGumbelMax,
    GaussianAdditive,
    Mechanism,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import numpy.typing as npt

    from ._mechanism import FitResult

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]
    AnyArray = npt.NDArray[Any]


def _merged_idata(fits: Mapping[str, FitResult]) -> Any:
    """Stack the per-node posteriors into one ``idata`` keyed by parameter name.

    Each node's draws are ``(S, ...)``; arviz wants ``(chain, draw, ...)``, so we
    present them as a single chain of S draws. No joint sampling ever happened;
    this is the compose step the pressure-test validated (max |composed - joint|
    interventional mean 0.004).
    """
    import arviz as az

    posterior: dict[str, Any] = {}
    for fit in fits.values():
        for pname, draws in fit.params.items():
            arr = np.asarray(draws, dtype=np.float64)
            posterior[pname] = arr.reshape(1, arr.shape[0], *arr.shape[1:])
    # arviz v2 keys the dict by group (DataTree); the posterior group is what
    # sample_posterior_predictive reads its parameter draws from.
    return az.from_dict({"posterior": posterior})


def _build_generative(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    n: int,
) -> Any:
    """One PyMC generative model: each node ``f(parent RVs) + noise``, vector-shaped.

    Parameters enter as named RVs whose names match the merged posterior exactly,
    so ``sample_posterior_predictive`` reads the fitted draws (not the prior). Only
    the Gaussian-additive nodes contribute to this model; a categorical node's
    interventional query goes through the numpy Gumbel-max path instead, so this
    model is built for the continuous interventional readout.
    """
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        nodes: dict[str, Any] = {}
        for name in order:
            mech = mechanisms[name]
            if not isinstance(mech, GaussianAdditive):
                raise TypeError(
                    f"the generative interventional model handles Gaussian-additive "
                    f"nodes; node {name!r} is {type(mech).__name__}. Use the "
                    "counterfactual or categorical query path"
                )
            intercept = pm.Normal(mech.intercept_name, 0.0, mech.coef_prior_sd)
            mean: Any = intercept * pt.ones((n,))
            if mech.parents:
                coef = pm.Normal(mech.coef_name, 0.0, mech.coef_prior_sd, shape=len(mech.parents))
                for j, parent in enumerate(mech.parents):
                    mean = mean + coef[j] * nodes[parent]
            sigma = pm.HalfNormal(mech.sigma_name, mech.sigma_prior_sd)
            nodes[name] = pm.Normal(name, mu=mean, sigma=sigma, shape=(n,))
    return model


def interventional_draws(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    do: Mapping[str, float],
    target: str,
    *,
    n: int,
    seed: int,
) -> Array:
    """Draws of ``target`` under ``do``, through ``pm.do`` on the generative model.

    Builds the generative model, applies ``pm.do`` (each value broadcast to a
    length-n vector, the verified shape gotcha), and reads ``target`` via
    ``sample_posterior_predictive`` over the merged posterior, so the draws carry
    parameter uncertainty. ``target`` must not be an intervened node (it leaves the
    predictive set). Returns the flattened ``(n * S,)`` draws.
    """
    import pymc as pm
    from pymc.model.transform.conditioning import do as pm_do

    if target in do:
        raise ValueError(
            f"target {target!r} is intervened on; after do() it is a constant and "
            "leaves the predictive set. Read a downstream node instead"
        )
    model = _build_generative(order, mechanisms, n)
    idata = _merged_idata(fits)
    intervention = {k: np.full(n, float(v), dtype=np.float64) for k, v in do.items()}
    model_do = pm_do(model, intervention)
    with model_do:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[target],
            predictions=True,
            progressbar=False,
            random_seed=seed,
        )
    arr = np.asarray(ppc.predictions[target].to_numpy(), dtype=np.float64)
    return arr.reshape(-1)


def ate_contrast(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    treatment: str,
    outcome: str,
    *,
    low: float,
    high: float,
    n: int,
    seed: int,
) -> Array:
    """A posterior over ``E[Y | do(X = high)] - E[Y | do(X = low)]``.

    Builds the generative model once, runs ``pm.do`` at each treatment level over
    the *same* merged posterior, reduces over the unit axis per draw, and
    differences, so the contrast is a posterior whose spread is the parameter
    uncertainty (the draws line up because both interventions read the same
    ``idata`` with the same seed). This is the SCM plug-in ATE, internally
    consistent with the simulator (one fitted object).
    """
    import pymc as pm
    from pymc.model.transform.conditioning import do as pm_do

    idata = _merged_idata(fits)

    def per_draw_mean(level: float) -> Array:
        model = _build_generative(order, mechanisms, n)
        model_do = pm_do(model, {treatment: np.full(n, float(level), dtype=np.float64)})
        with model_do:
            ppc = pm.sample_posterior_predictive(
                idata,
                var_names=[outcome],
                predictions=True,
                progressbar=False,
                random_seed=seed,
            )
        arr = np.asarray(ppc.predictions[outcome].to_numpy(), dtype=np.float64)
        # (chain, draw, n) -> mean over the unit axis, flattened over chain*draw.
        return np.asarray(arr.mean(axis=-1).reshape(-1), dtype=np.float64)

    return per_draw_mean(high) - per_draw_mean(low)


def observational_draws(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    target: str,
    *,
    n: int,
    seed: int,
) -> Array:
    """Posterior-predictive draws of ``target`` from the generative model (no do)."""
    import pymc as pm

    model = _build_generative(order, mechanisms, n)
    idata = _merged_idata(fits)
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[target],
            predictions=True,
            progressbar=False,
            random_seed=seed,
        )
    arr = np.asarray(ppc.predictions[target].to_numpy(), dtype=np.float64)
    return arr.reshape(-1)


# -- counterfactuals: the explicit-noise twin network ------------------------


def counterfactual_continuous(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    factual: Mapping[str, AnyArray],
    do: Mapping[str, float],
    target: str,
    *,
    use_draws: bool,
) -> Array:
    """Exact twin-network counterfactual for invertible Gaussian-additive nodes.

    Abduct each node's residual ``U = X - f(PA)`` on the factual values, then
    predict on the action branch (``do`` pins the intervened nodes, every other
    node recomputes ``f(PA_cf) + U`` reusing the abducted ``U``). This is the
    machine-precision route (the pressure-test's WAY C): no eps, no MCMC floor.

    With ``use_draws`` the prediction runs per parameter-posterior draw, so the
    counterfactual is a distribution whose spread is ``|x' - x| * sd(b)``; the
    factual residual is abducted at the posterior mean (the unit's exact residual
    given the fitted mechanism). Returns the flattened draws of ``target``.
    """
    factual_arr = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in factual.items()}
    n_units = len(next(iter(factual_arr.values())))

    # abduction on the factual graph: the exact residual at the posterior mean.
    noise: dict[str, Array] = {}
    for name in order:
        mech = mechanisms[name]
        if not isinstance(mech, GaussianAdditive):
            raise TypeError(
                f"counterfactual_continuous handles Gaussian-additive nodes; node "
                f"{name!r} is {type(mech).__name__}"
            )
        pv = _factual_parents(mech, factual_arr)
        noise[name] = mech.abduct(fits[name], factual_arr[name], pv)

    # prediction on the action branch, reusing the abducted noise. With draws,
    # every node is (S, n) and propagation is honestly per parameter-posterior
    # draw, so the spread is the full structural uncertainty, not just sd(b)
    # on the last hop. Without draws every node is (n,) at the posterior mean.
    cf: dict[str, Array] = {}
    for name in order:
        mech = mechanisms[name]
        assert isinstance(mech, GaussianAdditive)
        if name in do:
            value = float(do[name])
            cf[name] = np.full((1, n_units), value) if use_draws else np.full(n_units, value)
            continue
        u = noise[name]
        if use_draws:
            pv_draws = _cf_parents_per_draw(mech, cf, n_units)
            mean = mech.mean_per_draw(fits[name], pv_draws)
            cf[name] = mean + u[None, :]
        else:
            pv = _cf_parents(mech, cf, n_units)
            cf[name] = mech.structural_mean(fits[name], pv) + u
    return np.asarray(cf[target], dtype=np.float64).reshape(-1)


def _factual_parents(mech: Mechanism, factual: Mapping[str, Array]) -> Array:
    if not mech.parents:
        n = len(next(iter(factual.values())))
        return np.empty((n, 0), dtype=np.float64)
    return np.column_stack([factual[p] for p in mech.parents])


def _cf_parents(mech: Mechanism, cf: Mapping[str, Array], n_units: int) -> Array:
    """The ``(n, p)`` counterfactual parent design at the posterior mean."""
    if not mech.parents:
        return np.empty((n_units, 0), dtype=np.float64)
    cols = [np.asarray(cf[p], dtype=np.float64).reshape(-1) for p in mech.parents]
    return np.column_stack(cols)


def _cf_parents_per_draw(mech: Mechanism, cf: Mapping[str, Array], n_units: int) -> Array:
    """The ``(S, n, p)`` per-draw counterfactual parent design.

    A node already computed as ``(S, n)`` contributes per draw; an intervened
    node stored as ``(1, n)`` broadcasts over the draw axis.
    """
    if not mech.parents:
        # the draw count is set by the node's own parameter posterior; a single
        # column of zeros with the right unit axis is enough for the no-parent case.
        return np.zeros((1, n_units, 0), dtype=np.float64)
    cols = []
    for p in mech.parents:
        val = np.asarray(cf[p], dtype=np.float64)
        if val.ndim == 1:
            val = val[None, :]
        cols.append(val)
    s = max(c.shape[0] for c in cols)
    stacked = [np.broadcast_to(c, (s, n_units)) for c in cols]
    return np.stack(stacked, axis=-1)  # (S, n, p)


def counterfactual_categorical(
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    factual: Mapping[str, AnyArray],
    do: Mapping[str, float],
    target: str,
    *,
    parents_of_target: Sequence[str],
    seed: int,
) -> tuple[IntArray, Array]:
    """Gumbel-max counterfactual for a categorical target node.

    Abduct the target's per-unit Gumbel noise from its observed category and its
    factual parent logits, shift the parents per ``do``, then re-decide
    ``argmax(new_logits + same noise)``. Returns ``(per_unit_codes, marginal)``:
    the per-unit counterfactual category codes and the marginal distribution over
    the target's categories. Reusing the abducted noise (not a fresh draw) is what
    makes it a counterfactual rather than an interventional resample.
    """
    mech = mechanisms[target]
    if not isinstance(mech, CategoricalGumbelMax):
        raise TypeError(f"counterfactual_categorical needs a Gumbel-max target; got {type(mech)}")
    factual_arr = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in factual.items()}
    obs_codes = np.asarray(factual[target], dtype=np.int64).reshape(-1)

    pv_factual = (
        np.column_stack([factual_arr[p] for p in parents_of_target])
        if parents_of_target
        else np.empty((len(obs_codes), 0), dtype=np.float64)
    )
    gumbel = mech.abduct(fits[target], obs_codes, pv_factual, seed=seed)

    # counterfactual parent values: do pins an intervened parent, others stay factual.
    cf_parent_cols = []
    for p in parents_of_target:
        col = np.full(len(obs_codes), float(do[p])) if p in do else factual_arr[p]
        cf_parent_cols.append(col)
    pv_cf = (
        np.column_stack(cf_parent_cols)
        if parents_of_target
        else np.empty((len(obs_codes), 0), dtype=np.float64)
    )
    logits_cf = mech.logits(fits[target], pv_cf)
    cf_codes = mech.decide(logits_cf, gumbel)

    n_cat = mech.n_categories
    marginal = np.bincount(cf_codes, minlength=n_cat).astype(np.float64) / len(cf_codes)
    return cf_codes, marginal
