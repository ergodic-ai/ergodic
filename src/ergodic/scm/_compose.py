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
  the factual graph, then predict on an action branch that reuses it. The SCM
  handles invertible mechanisms only: the additive-Gaussian abduction is the exact
  residual and the prediction is deterministic numpy over the parameter posterior
  (machine precision, the ``Y_cf = y + b(x' - x)`` anchor), and the categorical
  Gumbel-max abduction is the truncated-Gumbel numpy sampler. A general
  (non-invertible) node's counterfactual would need the genuinely wider abduction
  posterior, which is not implemented, so the SCM refuses it upstream rather than reuse the
  exact-residual route.

The gotchas, all handled here: ``do`` is a length-n vector; the intervened node
leaves the predictive set; ``sample_posterior_predictive`` matches by name and is
silent on a mismatch (one naming scheme everywhere); ``do`` returns a new model;
arviz is v2 (``from_dict``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._mechanism import (
    CategoricalGumbelMax,
    FitResult,
    GaussianAdditive,
    Mechanism,
    encode_parent_design,
)
from ._sampling import sampler_cores

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]
    AnyArray = npt.NDArray[Any]


def _merged_idata(fits: Mapping[str, FitResult]) -> Any:
    """Stack the per-node posteriors into one ``idata`` keyed by parameter name.

    Each node's draws are ``(S, ...)``; arviz wants ``(chain, draw, ...)``, so we
    present them as a single chain of S draws. No joint sampling ever happened;
    this is the compose step the pressure-test validated (max |composed - joint|
    interventional mean 0.004).

    The result is a plain posterior ``xarray.Dataset``, which
    ``sample_posterior_predictive`` reads the same way on PyMC 5 (arviz 0.23,
    Python 3.11) and PyMC 6 (arviz 1.x). ``az.from_dict`` is not portable here: arviz
    0.23 would read a group-keyed dict as one variable named "posterior", and the
    parameters would then be silently drawn from the prior.
    """
    import xarray as xr

    posterior: dict[str, Any] = {}
    for fit in fits.values():
        for pname, draws in fit.params.items():
            arr = np.asarray(draws, dtype=np.float64)
            dims = ("chain", "draw", *(f"{pname}_dim_{i}" for i in range(arr.ndim - 1)))
            posterior[pname] = (dims, arr.reshape(1, arr.shape[0], *arr.shape[1:]))
    return xr.Dataset(posterior)


def _resolved_mechanisms(
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
) -> dict[str, Mechanism]:
    """A copy of the mechanisms with each categorical node's class count filled in.

    A categorical mechanism's ``n_categories`` may be unresolved on the spec (0); the
    fit resolves it. The generative model needs it on the mechanism to size the
    logits, so we pour the fit-resolved count back onto a copy. Single-sourcing it
    here keeps the generative-build helper a pure function of (mechanisms, fits).
    """
    from dataclasses import replace

    out: dict[str, Mechanism] = {}
    for name, mech in mechanisms.items():
        if isinstance(mech, CategoricalGumbelMax):
            resolved = fits[name].n_categories or mech.n_categories
            out[name] = replace(mech, n_categories=resolved)
        else:
            out[name] = mech
    return out


def _build_generative(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    n: int,
    *,
    latents: Sequence[str] = (),
) -> Any:
    """One PyMC generative model: each node built by its own mechanism, vector-shaped.

    Every node delegates to ``mechanism.generative_rv``: a Gaussian-additive node is
    ``Normal(b0 + encode(PA) . w, sigma)``, a categorical/binary node is the
    Gumbel-max ``Categorical(logit=bias + encode(PA) . w)``. Parameters enter as
    named RVs whose names match the merged posterior exactly (the mechanism's own
    name properties single-source them), so ``sample_posterior_predictive`` reads the
    fitted draws, not the prior. ``pm.do`` can pin any node -- continuous to a float,
    categorical to an integer code -- and downstream propagates through the mixed
    graph. Categorical mechanisms must arrive with their class count resolved (see
    :func:`_resolved_mechanisms`).

    ``latents`` are the district confounders: each is built first as a per-unit
    ``Normal(0, 1)`` RV (the explicit common cause of its district endpoints) and
    placed in ``nodes`` under its own name, so a mechanism that lists it in
    ``latents`` reads it like any parent. A latent is *not* a fitted parameter -- it
    is re-drawn fresh here, exactly as ``latent_projection`` semantics intend (a
    ``<->`` is a marginalized common cause) -- so it never appears in the merged
    posterior; only the loadings on it do. ``pm.do`` on an endpoint severs that
    endpoint from the latent while its district sibling still reads it, which is what
    makes the confounding cancel in an interventional contrast.
    """
    import pymc as pm

    with pm.Model() as model:
        nodes: dict[str, Any] = {}
        for lat in latents:
            nodes[lat] = pm.Normal(lat, 0.0, 1.0, shape=(n,))
        for name in order:
            nodes[name] = mechanisms[name].generative_rv(nodes, n)
    return model


def fit_district(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    latents: Sequence[str],
    observed: Mapping[str, AnyArray],
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    target_accept: float = 0.95,
) -> dict[str, FitResult]:
    """Joint NUTS fit of a confounded district.

    A district whose nodes share a ``<->`` does not factorize -- the latent couples
    them -- so the district fits as one model rather than per node. Each latent is a
    non-centered per-unit ``Normal(0, 1)`` (the explicit common cause); each district
    node carries its structural equation observed at its data, conditioning on its
    observed parents (in-district siblings by their data, external parents likewise)
    and adding ``loading_L * L`` for each incident latent. The gauge (one loading per
    latent pinned positive, via the mechanism's ``gauged_latents``) kills the
    reflection symmetry, and a high ``target_accept`` tames the non-identified ridge.

    ``order`` is the district's nodes in topological order; ``observed`` maps every
    referenced name (district nodes and their external parents) to its data column
    (codes for a discrete column, floats otherwise). Returns one
    :class:`FitResult` per district node, so the merged-posterior machinery (which is
    node-keyed) is unchanged -- the loadings land on the node whose ``loading_name``
    matches. The latent RVs are *not* returned (the generative model re-draws them);
    only the loadings are.

    The non-identified directions sample badly here (low ESS on a coupled latent's
    loadings, the ``corr(effect, confounding) ~ -1`` ridge); that pathology is the
    non-identification signal, not a bug, and the honesty layer bounds exactly those
    effects. A clean district edge (no open bidirected back-door) still recovers its
    coefficient with healthy sampling.
    """
    import pymc as pm

    n = len(next(iter(observed.values())))
    with pm.Model():
        lat_rvs: dict[str, Any] = {}
        for lat in latents:
            raw = pm.Normal(f"{lat}__raw", 0.0, 1.0, shape=(n,))  # non-centered
            lat_rvs[lat] = pm.Deterministic(lat, raw)
        for name in order:
            mech = mechanisms[name]
            _district_likelihood(mech, observed, lat_rvs)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=sampler_cores(chains),
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    return {name: _split_fit(mechanisms[name], post) for name in order}


def _district_likelihood(
    mech: Mechanism,
    observed: Mapping[str, AnyArray],
    lat_rvs: Mapping[str, Any],
) -> Any:
    """Add one district node's observed likelihood to the active model.

    A Gaussian node is ``Normal(b0 + encode(PA) . w + sum loading_L L, sigma)``
    observed at its data; a categorical node the equivalent ``Categorical`` over
    per-class logits. The parameter RVs share the mechanism's names and the
    loadings share its gauge, so the generative view reads the fitted draws by name.
    """
    import pymc as pm

    n = len(next(iter(observed.values())))
    raw = _raw_parent_design(mech, observed)
    if isinstance(mech, GaussianAdditive):
        intercept = pm.Normal(mech.intercept_name, 0.0, mech.coef_prior_sd)
        mean: Any = intercept * pm.math.ones((n,))
        if mech.parents:
            design = encode_parent_design(raw, mech.parent_cards)
            coef = pm.Normal(mech.coef_name, 0.0, mech.coef_prior_sd, shape=mech._design_width)
            mean = mean + pm.math.dot(design, coef)
        for lat in mech.latents:
            mean = mean + mech._loading_rv(lat) * lat_rvs[lat]
        sigma = pm.HalfNormal(mech.sigma_name, mech.sigma_prior_sd)
        y = np.asarray(observed[mech.name], dtype=np.float64)
        return pm.Normal(mech.name, mu=mean, sigma=sigma, observed=y)
    if isinstance(mech, CategoricalGumbelMax):
        k = mech.n_categories
        if k <= 0:
            raise ValueError(
                f"categorical district node {mech.name!r} has an unresolved class count"
            )
        bias = pm.Normal(mech.bias_name, 0.0, mech.coef_prior_sd, shape=k)
        logits: Any = bias[None, :] * pm.math.ones((n, 1))
        if mech.parents:
            design = encode_parent_design(raw, mech.parent_cards)
            weight = pm.Normal(
                mech.weight_name, 0.0, mech.coef_prior_sd, shape=(mech._design_width, k)
            )
            logits = logits + pm.math.dot(design, weight)
        for lat in mech.latents:
            logits = logits + lat_rvs[lat][:, None] * mech._loading_rv(lat)[None, :]
        codes = np.asarray(observed[mech.name], dtype=np.int64)
        return pm.Categorical(mech.name, logit_p=logits, observed=codes)
    raise TypeError(  # pragma: no cover - both built-ins are handled above.
        f"district fit over node {mech.name!r} of unsupported mechanism {type(mech).__name__}"
    )


def _raw_parent_design(mech: Mechanism, observed: Mapping[str, AnyArray]) -> Array:
    """The ``(n, p)`` raw parent matrix for a district node, from observed columns."""
    if not mech.parents:
        n = len(next(iter(observed.values())))
        return np.empty((n, 0), dtype=np.float64)
    return np.column_stack([np.asarray(observed[p], dtype=np.float64) for p in mech.parents])


def _split_fit(mech: Mechanism, post: Any) -> FitResult:
    """Pull one node's parameters (coef/intercept/sigma/loadings) out of the joint posterior.

    The joint district posterior carries every node's parameters under the
    mechanism's disciplined names; this slices out the ones belonging to ``mech`` and
    rebuilds the per-node :class:`FitResult` (with ``n_categories`` for a categorical
    node), so the merged-posterior store stays node-keyed exactly as the per-node
    path produces.
    """
    params: dict[str, Array] = {}
    for pname in _param_names(mech):
        params[pname] = _flatten_post(post[pname])
    means = {k: np.asarray(v.mean(axis=0), dtype=np.float64) for k, v in params.items()}
    n_cat = mech.n_categories if isinstance(mech, CategoricalGumbelMax) else 0
    return FitResult(params=params, param_means=means, n_categories=n_cat)


def _param_names(mech: Mechanism) -> list[str]:
    """Every parameter name a district fit produces for ``mech`` (incl. loadings)."""
    names: list[str] = []
    if isinstance(mech, GaussianAdditive):
        names.append(mech.intercept_name)
        names.append(mech.sigma_name)
        if mech.parents:
            names.append(mech.coef_name)
    elif isinstance(mech, CategoricalGumbelMax):
        names.append(mech.bias_name)
        if mech.parents:
            names.append(mech.weight_name)
    names.extend(_loading_param_names(mech))
    return names


def _loading_param_names(mech: Mechanism) -> list[str]:
    """The loading parameter names a fit emits for ``mech`` (the gauge splits some).

    A gauged Gaussian loading is one ``HalfNormal`` under ``loading_name``. A gauged
    categorical loading is split into ``..__0`` (the pinned class) and ``..__rest``;
    an ungauged one is a single vector under ``loading_name``. These must match the
    RVs :meth:`_loading_rv` declares, since the generative view re-declares them and
    ``sample_posterior_predictive`` matches by name.
    """
    out: list[str] = []
    # only the two built-in mechanisms carry latents (and so a loading_name); a
    # latent-free Markovian mechanism returns no loading names. Narrowing to the
    # union keeps loading_name resolvable without widening the Mechanism protocol.
    if not isinstance(mech, GaussianAdditive | CategoricalGumbelMax):
        return out
    gauged = mech.gauged_latents
    for lat in mech.latents:
        base = mech.loading_name(lat)
        if isinstance(mech, CategoricalGumbelMax) and lat in gauged and mech.n_categories > 1:
            out.append(f"{base}__0")
            out.append(f"{base}__rest")
        else:
            out.append(base)
    return out


def _flatten_post(arr: Any) -> Array:
    """Collapse an arviz ``(chain, draw, ...)`` posterior array to ``(S, ...)``."""
    a = np.asarray(arr.values, dtype=np.float64)
    return a.reshape(-1, *a.shape[2:])


def _is_categorical(mech: Mechanism) -> bool:
    return isinstance(mech, CategoricalGumbelMax)


def _intervention_vector(mech: Mechanism, value: float, n: int) -> AnyArray:
    """The length-n ``pm.do`` value for one node, dtyped to the node's kind.

    A categorical/binary node is pinned with an integer code (the Categorical RV is
    integer-valued, so a float pin corrupts it); a continuous node with a float. The
    verified shape gotcha: ``pm.do`` wants a length-n vector, not a scalar.
    """
    if _is_categorical(mech):
        v = float(value)
        if v != round(v):
            raise ValueError(
                f"categorical do-value {value!r} for node {mech.name!r} is not an "
                "integer category code (a fractional do-value is not snapped); pass an "
                "integer code"
            )
        return np.full(n, round(v), dtype=np.int64)
    return np.full(n, float(value), dtype=np.float64)


def _read_target(ppc_predictions: Any, mech: Mechanism) -> AnyArray:
    """Flatten a target node's PPC draws, preserving integer codes for a categorical.

    A categorical target reads as integer category codes (the caller bincounts them
    into a marginal); a continuous target as floats. The dtype is the node's, so the
    return is an ``AnyArray`` -- the caller dispatches on the mechanism kind.
    """
    raw = ppc_predictions.to_numpy()
    if _is_categorical(mech):
        return np.asarray(raw, dtype=np.int64).reshape(-1)
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _ppc_target(
    model: Any,
    idata: Any,
    target: str,
    *,
    seed: int,
) -> Any:
    """Run ``sample_posterior_predictive`` for ``target`` over the merged posterior."""
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=[target],
            predictions=True,
            progressbar=False,
            random_seed=seed,
        )
    return ppc.predictions[target]


def interventional_draws(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    do: Mapping[str, float],
    target: str,
    *,
    n: int,
    seed: int,
    latents: Sequence[str] = (),
) -> AnyArray:
    """Draws of ``target`` under ``do``, through ``pm.do`` on the generative model.

    Builds the generative model, applies ``pm.do`` (each value broadcast to a
    length-n vector, dtyped to the node's kind: an integer code for a categorical
    do-target, a float otherwise), and reads ``target`` via
    ``sample_posterior_predictive`` over the merged posterior, so the draws carry
    parameter uncertainty. ``target`` must not be an intervened node (it leaves the
    predictive set). A categorical target's draws are returned as integer codes (the
    caller forms the marginal); a continuous target's as floats. ``latents`` are the
    district confounders woven into the generative model.
    """
    from pymc.model.transform.conditioning import do as pm_do

    if target in do:
        raise ValueError(
            f"target {target!r} is intervened on; after do() it is a constant and "
            "leaves the predictive set. Read a downstream node instead"
        )
    mechs = _resolved_mechanisms(mechanisms, fits)
    model = _build_generative(order, mechs, n, latents=latents)
    idata = _merged_idata(fits)
    intervention = {k: _intervention_vector(mechs[k], v, n) for k, v in do.items()}
    model_do = pm_do(model, intervention)
    preds = _ppc_target(model_do, idata, target, seed=seed)
    return _read_target(preds, mechs[target])


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
    latents: Sequence[str] = (),
) -> Array:
    """A posterior over ``E[Y | do(X = high)] - E[Y | do(X = low)]``.

    Builds the generative model once, runs ``pm.do`` at each treatment level over
    the *same* merged posterior, reduces over the unit axis per draw, and
    differences, so the contrast is a posterior whose spread is the parameter
    uncertainty (the draws line up because both interventions read the same
    ``idata`` with the same seed). This is the SCM plug-in ATE, internally
    consistent with the simulator (one fitted object). The treatment may be any node
    type (a binary/categorical level is pinned as its integer code); the outcome
    must be a numeric (Gaussian-additive) node -- a categorical contrast has no
    scalar difference, and the caller gates that.

    ``latents`` are the district confounders. When the treatment is a district
    endpoint, ``pm.do`` severs it from the latent while the outcome (its district
    sibling, or a downstream node) still reads the latent's marginal, which has the
    same mean at both levels and so cancels in the contrast -- this is exactly what
    lets an identified-despite-confounding effect (front-door and its kin) compute.
    """
    from pymc.model.transform.conditioning import do as pm_do

    mechs = _resolved_mechanisms(mechanisms, fits)
    idata = _merged_idata(fits)

    def per_draw_mean(level: float) -> Array:
        model = _build_generative(order, mechs, n, latents=latents)
        model_do = pm_do(model, {treatment: _intervention_vector(mechs[treatment], level, n)})
        preds = _ppc_target(model_do, idata, outcome, seed=seed)
        arr = np.asarray(preds.to_numpy(), dtype=np.float64)
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
    latents: Sequence[str] = (),
) -> AnyArray:
    """Posterior-predictive draws of ``target`` from the generative model (no do).

    A categorical target returns integer codes; a continuous target floats.
    ``latents`` are the district confounders woven into the generative model.
    """
    mechs = _resolved_mechanisms(mechanisms, fits)
    model = _build_generative(order, mechs, n, latents=latents)
    idata = _merged_idata(fits)
    preds = _ppc_target(model, idata, target, seed=seed)
    return _read_target(preds, mechs[target])


# -- counterfactuals: the explicit-noise twin network ------------------------


def _cf_do_column(value: float | AnyArray, n_units: int, name: str) -> Array:
    """A counterfactual do-value as an ``(n,)`` column: a scalar fills, an array passes.

    A scalar pins every unit to the same value (the absolute-level intervention,
    ``do(X = c)``). A length-``n`` array pins each unit to its own value (a per-unit
    intervention, e.g. the shift ``do(X = x_i + delta)`` from each unit's factual
    baseline). Any other length is a user error. This is the only place the
    counterfactual engine reads a non-scalar do.
    """
    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if raw.size == 1:
        return np.full(n_units, float(raw[0]))
    if raw.size == n_units:
        return raw
    raise ValueError(
        f"do-value for {name!r} has length {raw.size}; expected a scalar or one value "
        f"per unit ({n_units})"
    )


def counterfactual_continuous(
    order: Sequence[str],
    mechanisms: Mapping[str, Mechanism],
    fits: Mapping[str, FitResult],
    factual: Mapping[str, AnyArray],
    do: Mapping[str, float | AnyArray],
    target: str,
    *,
    use_draws: bool,
    seed: int = 0,
) -> Array:
    """Twin-network counterfactual for a Gaussian-additive ``target`` over a mixed graph.

    Abduct each node's noise on the factual values, then predict on the action
    branch (``do`` pins the intervened nodes, every other node recomputes its
    structural equation reusing its noise). A Gaussian node reuses the exact residual
    ``U = X - f(PA)`` -- the machine-precision route (the pressure-test's WAY C): no
    eps, no MCMC floor. A categorical *ancestor* reuses its abducted Gumbel and
    re-decides ``argmax`` under shifted parent logits (so it responds to an upstream
    intervention); the ``target`` itself must be Gaussian (a categorical target uses
    :func:`counterfactual_categorical`).

    With ``use_draws`` the Gaussian prediction runs per parameter-posterior draw, so
    the counterfactual is a distribution whose spread is the structural uncertainty
    of the mechanisms; the residual is abducted at the posterior mean (the unit's
    exact residual given the fitted mechanism). A categorical ancestor's decision is
    *also* spread per draw -- re-decided under each draw's logit parameters with its
    abducted Gumbel held fixed -- so its logit-parameter posterior enters the
    counterfactual interval too, not only the continuous hops'. Returns the flattened
    draws of ``target``.
    """
    factual_arr = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in factual.items()}
    n_units = len(next(iter(factual_arr.values())))

    # The twin network over a MIXED graph: every node is abducted on the factual
    # world, then recomputed on the action branch reusing its noise. A Gaussian node
    # is exact (the residual reused per parameter draw). A categorical ANCESTOR (a
    # confounder/mediator upstream of a continuous target) carries its abducted
    # Gumbel and re-decides argmax under shifted parent logits; under use_draws it is
    # re-decided PER posterior draw (each draw's logit params, the Gumbel held fixed),
    # so the categorical's own parameter uncertainty enters the counterfactual
    # interval -- not just the continuous hop's. A categorical TARGET takes the
    # dedicated counterfactual_categorical path, not this one.
    target_mech = mechanisms[target]
    if not isinstance(target_mech, GaussianAdditive):
        raise TypeError(
            f"counterfactual_continuous needs a Gaussian-additive target; node "
            f"{target!r} is {type(target_mech).__name__}. A categorical target uses "
            "counterfactual_categorical"
        )

    # abduction on the factual graph, by mechanism kind (all at the posterior mean).
    # A categorical ancestor's Gumbel uses a deterministic per-node seed (base seed +
    # its topological index), so the counterfactual is reproducible.
    gauss_noise: dict[str, Array] = {}
    gumbel_noise: dict[str, Array] = {}
    for idx, name in enumerate(order):
        mech = mechanisms[name]
        pv = _factual_parents(mech, factual_arr)
        if isinstance(mech, CategoricalGumbelMax):
            obs_codes = np.asarray(factual_arr[name], dtype=np.int64)
            gumbel_noise[name] = mech.abduct(fits[name], obs_codes, pv, seed=seed + idx)
        elif isinstance(mech, GaussianAdditive):
            gauss_noise[name] = mech.abduct(fits[name], factual_arr[name], pv)
        else:  # pragma: no cover - every built-in mechanism is one of the two.
            raise TypeError(f"counterfactual over node {name!r} of {type(mech).__name__}")

    # prediction on the action branch. A categorical node is stored as (n,) codes
    # (broadcast over the draw axis when use_draws); a Gaussian node as (n,) or (S, n).
    cf: dict[str, Array] = {}
    for name in order:
        mech = mechanisms[name]
        if name in do:
            col = _cf_do_column(do[name], n_units, name)
            cf[name] = col[None, :].copy() if use_draws else col
            continue
        if isinstance(mech, CategoricalGumbelMax):
            g = gumbel_noise[name]  # the unit's abducted structural Gumbel, (n, K), FIXED.
            if use_draws:
                # spread the decision PER posterior draw: re-decide under each draw's
                # logit parameters while holding the abducted Gumbel fixed (the Gumbel
                # is the unit's identity; the logit params are the posterior we
                # integrate over). This carries the categorical ancestor's parameter
                # uncertainty forward, exactly as the Gaussian hops carry theirs --
                # so the (draws, n) shape feeds the per-draw downstream machine.
                pv_draws = _cf_parents_per_draw(mech, cf, n_units)
                logits_draws = mech.logits_per_draw(fits[name], pv_draws)  # (S, n, K)
                codes = _decide_per_draw(logits_draws, g)  # (S, n)
                cf[name] = codes.astype(np.float64)
            else:
                # the deterministic point: re-decide at the posterior-mean logits.
                pv = _cf_parents(mech, cf, n_units)
                cf[name] = mech.decide(mech.logits(fits[name], pv), g).astype(np.float64)
            continue
        assert isinstance(mech, GaussianAdditive)
        u = gauss_noise[name]
        if use_draws:
            pv_draws = _cf_parents_per_draw(mech, cf, n_units)
            mean = mech.mean_per_draw(fits[name], pv_draws)
            cf[name] = mean + u[None, :]
        else:
            pv = _cf_parents(mech, cf, n_units)
            cf[name] = mech.structural_mean(fits[name], pv) + u
    return np.asarray(cf[target], dtype=np.float64).reshape(-1)


def _decide_per_draw(logits_draws: Array, gumbel: Array) -> IntArray:
    """``argmax(logits + g)`` per draw, with the unit's Gumbel held fixed across draws.

    ``logits_draws`` is ``(S, n, K)`` (per parameter-posterior draw); ``gumbel`` the
    ``(n, K)`` abducted structural noise, broadcast over the draw axis. Returns the
    ``(S, n)`` per-draw counterfactual codes. The Gumbel is the unit's identity
    (abducted once); spreading only the logits is what restores the categorical
    ancestor's parameter uncertainty without re-drawing its noise.
    """
    total = np.asarray(logits_draws, dtype=np.float64) + np.asarray(gumbel, dtype=np.float64)[None]
    return np.asarray(np.argmax(total, axis=-1), dtype=np.int64)


def _factual_parents(mech: Mechanism, factual: Mapping[str, Array]) -> Array:
    if not mech.parents:
        n = len(next(iter(factual.values())))
        return np.empty((n, 0), dtype=np.float64)
    return np.column_stack([factual[p] for p in mech.parents])


def _cf_parents(mech: Mechanism, cf: Mapping[str, Array], n_units: int) -> Array:
    """The ``(n, p)`` counterfactual parent design at the posterior mean.

    A parent stored per-draw as ``(S, n)`` (e.g. a Gaussian ancestor under
    ``use_draws``) collapses to its per-unit mean over the draw axis -- the right
    value to feed a downstream node evaluated at the posterior mean (a categorical
    node's logits, or the non-draws Gaussian path). A ``(1, n)`` pinned/categorical
    parent collapses identically to ``(n,)``.
    """
    if not mech.parents:
        return np.empty((n_units, 0), dtype=np.float64)
    cols = []
    for p in mech.parents:
        val = np.asarray(cf[p], dtype=np.float64)
        col = val.mean(axis=0) if val.ndim == 2 else val.reshape(-1)
        cols.append(col)
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

    n_cat = fits[target].n_categories or mech.n_categories
    marginal = np.bincount(cf_codes, minlength=n_cat).astype(np.float64) / len(cf_codes)
    return cf_codes, marginal


def marginal_from_codes(codes: AnyArray, n_categories: int) -> Array:
    """The per-category marginal from integer code draws (``bincount`` normalized).

    Used to turn a categorical node's interventional/observational code draws into a
    distribution over its ``n_categories`` classes.
    """
    c = np.asarray(codes, dtype=np.int64).reshape(-1)
    if c.size == 0:
        return np.zeros(n_categories, dtype=np.float64)
    return np.bincount(c, minlength=n_categories).astype(np.float64) / c.size
