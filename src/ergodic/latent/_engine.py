# SPDX-License-Identifier: Apache-2.0
"""The engine: chain priors, emissions, forward potential, sampler, decode.

The flow is fixed and every likelihood rides it unchanged. Prepare the series,
open a ``pm.Model``, register the chain priors (a Beta stay probability per
state and a Dirichlet start), let the likelihood register its emission sites,
sum the hidden path out with the log-space forward, and sample with NUTS
through the JAX backend when it is installed. Nothing here knows whether the
emission is a periodogram or an autoregression; the engine's whole knowledge
of the data is one log-density tensor.

Decoding then runs outside the model: an evenly spaced subsample of the
posterior (150 draws by default) goes draw by draw through the numpy
forward-backward, the trajectories are label-aligned, and each declared delta
block's draws are sign-corrected by the alignment's flip record before the
result freezes them. That ordering matters: a flipped draw's deltas describe
the mirror labeling, and reporting them unflipped would smear every interval
across the two modes.

Internally the model builder takes a *list* of emission tensors and adds their
log-densities before the forward pass. With one series the list has one entry;
the seam exists because several observed series sharing one hidden switch is
the same model with a longer sum, and the engine should not need rebuilding
when that arrives.

The certify statistic rides the same builder. :func:`improvement_statistic`
builds the identical model, takes a cheap point estimate of its parameters (a
short single-chain sample's posterior mean; the measured case against
``pm.find_MAP`` is in the certify docstring), and evaluates the two-state
marginal log-likelihood at that point with the numpy forward, minus the
likelihood's own one-state null. One computation, applied identically to the
observed series and to every simulated control world.

Diagnostics warn and never raise. A fit with a few divergences is information
the caller should see, and a pipeline that reruns dozens of control worlds
(calibration) has to be able to finish.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ._identify import align_gammas
from ._likelihood import MarkovChain, RegimeLikelihood, require_pymc, resolve_nuts_sampler
from ._marginalize import forward_backward, log_forward
from ._result import GateOutcome, LatentResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ._likelihood import EmissionSpec, PreparedData

    Array = npt.NDArray[np.float64]

__all__ = ["build_model", "fit_recover", "improvement_statistic", "marginal_loglik_np"]

#: A divergence share above this fraction of the draws earns a warning.
DIVERGENCE_WARN = 0.005
#: A split R-hat above this on any delta or chain site earns a warning.
RHAT_WARN = 1.05


def _combined_loglik(logliks: Sequence[Any]) -> Any:
    """One emission tensor for the forward pass: the elementwise sum of the list.

    Several series observing one switch multiply their emission densities,
    which is a sum in log space. All entries must share a shape; the built-ins always
    passes one.
    """
    if not logliks:
        raise ValueError("no emission tensors to combine")
    ndims = {int(t.ndim) for t in logliks}
    if len(ndims) != 1:
        raise NotImplementedError(
            "combining (T, K) and (T, K, K) emissions in one model is not "
            "supported; every emission tensor must share a shape"
        )
    total = logliks[0]
    for extra in logliks[1:]:
        total = total + extra
    return total


def build_model(
    prepared: PreparedData,
    target: str,
    likelihood: RegimeLikelihood,
    chain: MarkovChain,
) -> tuple[Any, EmissionSpec]:
    """The switching model as a ``pm.Model``: chain priors, emissions, forward Potential.

    One builder for every consumer: ``fit_recover`` samples it, and the
    certify statistic refits it cheaply on control worlds. Returns the model
    and the likelihood's emission spec (the declared deltas and gates).
    """
    pm = require_pymc()
    import pytensor.tensor as pt

    with pm.Model() as model:
        stay = pm.Beta(f"{target}_stay", chain.stay_alpha, chain.stay_beta, shape=2)
        pi0 = pm.Dirichlet(f"{target}_pi0", np.ones(2))
        spec = likelihood.emissions(target, prepared)
        log_trans = pt.log(
            pt.stack(
                [
                    pt.stack([stay[0], 1.0 - stay[0]]),
                    pt.stack([1.0 - stay[1], stay[1]]),
                ]
            )
        )
        total = log_forward(
            _combined_loglik([spec.loglik]),
            log_trans,
            pt.log(pi0),
            prepared.T_eff,
        )
        pm.Potential(f"{target}_marginal_loglik", total)
    return model, spec


def marginal_loglik_np(
    likelihood: RegimeLikelihood, prepared: PreparedData, draw: Mapping[str, Array]
) -> float:
    """The two-state marginal log-likelihood at one parameter point, in numpy.

    The same forward sum the model's Potential computes, evaluated at a
    single draw (bare parameter names, ``stay`` and ``pi0`` included), which
    is what the certify statistic plugs fitted parameters into.
    """
    loglik = likelihood.emissions_np(prepared, draw)
    stay = np.asarray(draw["stay"], dtype=np.float64)
    trans = np.array(
        [
            [stay[0], 1.0 - stay[0]],
            [1.0 - stay[1], stay[1]],
        ]
    )
    pi0 = np.asarray(draw["pi0"], dtype=np.float64)
    _, total = forward_backward(loglik, trans, pi0)
    return total


def improvement_statistic(
    likelihood: RegimeLikelihood,
    target: str,
    chain: MarkovChain,
    prepared: PreparedData,
    *,
    method: str,
    draws: int,
    warmup: int,
    seed: int,
) -> float:
    """Twice the log-likelihood gain of two states over the one-state null.

    The two-state side is the marginal log-likelihood (path summed out) at a
    cheap point estimate of the parameters: the MAP for ``method="map"``, or
    the posterior mean of a short single-chain sample for
    ``method="sample"``. The one-state side is the likelihood's own
    ``null_loglik``. ``certify()`` computes this identically on the observed
    series and on every simulated control world, which is what makes the
    resulting p-value exchangeable under the null.
    """
    pm = require_pymc()
    model, _ = build_model(prepared, target, likelihood, chain)
    prefix = f"{target}_"
    if method == "map":
        with model:
            point = pm.find_MAP(progressbar=False)
        draw = {
            name[len(prefix) :]: np.asarray(value, dtype=np.float64)
            for name, value in point.items()
            if name.startswith(prefix)
        }
    elif method == "sample":
        backend = resolve_nuts_sampler()
        with model:
            idata = pm.sample(
                draws=draws,
                tune=warmup,
                chains=1,
                cores=1,
                random_seed=seed,
                progressbar=False,
                nuts_sampler=backend,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": False},
            )
        flat = _flat_posterior(idata, prefix=prefix)
        draw = {name: np.asarray(value.mean(axis=0)) for name, value in flat.items()}
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unknown refit method {method!r}; use 'map' or 'sample'")
    ll_two = marginal_loglik_np(likelihood, prepared, draw)
    return 2.0 * (ll_two - likelihood.null_loglik(prepared))


def fit_recover(
    y: Array,
    target: str,
    likelihood: RegimeLikelihood,
    chain: MarkovChain,
    *,
    draws: int,
    warmup: int,
    chains: int,
    seed: int,
    decode_draws: int,
) -> LatentResult:
    """Fit the switching model to ``y`` and decode the hidden path.

    The facade has already validated everything user-facing; this function
    assumes clean inputs and owns the sampler and the decode.
    """
    pm = require_pymc()

    prepared = likelihood.prepare(y)
    model, spec = build_model(prepared, target, likelihood, chain)

    backend = resolve_nuts_sampler()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            # Named, never defaulted: PyMC picks nutpie when this is omitted,
            # which the forecast engine spike measured at 3.8x the JAX path.
            nuts_sampler=backend,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": False},
        )

    flat = _flat_posterior(idata, prefix=f"{target}_")
    total_draws = int(next(iter(flat.values())).shape[0])
    idx = np.linspace(0, total_draws - 1, min(decode_draws, total_draws)).astype(int)
    sub = {name: value[idx] for name, value in flat.items()}

    gammas = _decode(likelihood, prepared, sub)
    aligned, flips = align_gammas(gammas)
    sign = np.where(flips, -1.0, 1.0)

    delta_draws: dict[str, Array] = {}
    for block in spec.deltas:
        bare = _bare(block.site, f"{target}_")
        raw = sub[bare]
        shaped = sign.reshape((sign.shape[0],) + (1,) * (raw.ndim - 1))
        signed = shaped * raw
        reduced = block.transform(signed) if block.transform is not None else signed
        if reduced.shape != (len(idx),):
            raise ValueError(
                f"delta block {block.name!r} reduced to shape {reduced.shape}, "
                f"expected ({len(idx)},); a vector-valued site needs a transform "
                "that returns one scalar per draw"
            )
        delta_draws[block.name] = reduced

    gates = tuple(
        GateOutcome(gate.name, gate.blocks, *_run_gate(gate.check, sub)) for gate in spec.gates
    )

    rhat_aligned = aligned_rhat(delta_draws, idx, total_draws // max(chains, 1))
    divergences, rhat_unaligned = _diagnostics(
        idata, target, spec, total_draws, rhat_aligned=rhat_aligned
    )

    return LatentResult(
        target=target,
        gammas=aligned,
        flips=flips,
        delta_draws=delta_draws,
        gates=gates,
        tick_multiplier=prepared.tick_multiplier,
        offset=prepared.offset,
        chain=chain,
        likelihood=likelihood,
        n_draws=total_draws,
        divergences=divergences,
        rhat_max=rhat_aligned,
        rhat_max_unaligned=rhat_unaligned,
    )


def _bare(site: str, prefix: str) -> str:
    if not site.startswith(prefix):
        raise ValueError(
            f"site {site!r} does not carry the target prefix {prefix!r}; every "
            "emission site must be registered as f'{target}_{param}'"
        )
    return site[len(prefix) :]


def _flat_posterior(idata: Any, prefix: str) -> dict[str, Array]:
    """Every prefixed site's draws, flattened to ``(chains * draws, ...)``, bare-named."""
    posterior = idata.posterior
    flat: dict[str, Array] = {}
    for name in posterior.data_vars:
        text = str(name)
        if not text.startswith(prefix):
            continue
        values = posterior[text].values
        flat[text[len(prefix) :]] = values.reshape(-1, *values.shape[2:])
    return flat


def _decode(
    likelihood: RegimeLikelihood,
    prepared: PreparedData,
    sub: Mapping[str, Array],
) -> Array:
    """Per-draw forward-backward over the decode subsample: ``(S, T_eff)`` of P(state 1)."""
    stay = sub["stay"]
    pi0 = sub["pi0"]
    n_sub = stay.shape[0]
    gammas = np.zeros((n_sub, prepared.T_eff))
    for i in range(n_sub):
        draw = {name: value[i] for name, value in sub.items()}
        loglik = likelihood.emissions_np(prepared, draw)
        trans = np.array(
            [
                [stay[i, 0], 1.0 - stay[i, 0]],
                [1.0 - stay[i, 1], stay[i, 1]],
            ]
        )
        gamma, _ = forward_backward(loglik, trans, pi0[i])
        gammas[i] = gamma[:, 1]
    return gammas


def _run_gate(check: Any, sub: Mapping[str, Array]) -> tuple[float, bool]:
    value, passed = check(sub)
    return float(value), bool(passed)


def aligned_rhat(
    delta_draws: Mapping[str, Array], idx: npt.NDArray[np.integer[Any]], draws_per_chain: int
) -> float | None:
    """Worst split R-hat over the label-aligned delta draws, grouped by chain.

    ``delta_draws`` are the sign-corrected per-draw deltas over the decode
    subsample, ``idx`` their positions in the flattened ``(chains * draws)``
    posterior, so ``idx // draws_per_chain`` is each draw's chain. Every chain
    keeps the same number of draws (the shortest), as R-hat needs. This is the
    R-hat the verdicts should be judged by: two chains sitting in mirror modes
    disagree on every raw signed site, but agree once aligned. ``None`` when
    fewer than four draws per chain survive or arviz cannot compute it.
    """
    if not delta_draws or draws_per_chain <= 0:
        return None
    chain_of = np.asarray(idx, dtype=np.int64) // draws_per_chain
    chains = np.unique(chain_of)
    per_chain = min(int(np.sum(chain_of == c)) for c in chains)
    if per_chain < 4:
        return None
    try:
        import arviz as az

        worst = 0.0
        for draws in delta_draws.values():
            values = np.asarray(draws, dtype=np.float64)
            grid = np.stack([values[chain_of == c][:per_chain] for c in chains])
            worst = max(worst, float(az.rhat(grid)))
    except Exception:  # diagnostics never cost the result
        return None
    return worst if np.isfinite(worst) else None


def _diagnostics(
    idata: Any,
    target: str,
    spec: EmissionSpec,
    total_draws: int,
    *,
    rhat_aligned: float | None,
) -> tuple[int | None, float | None]:
    """Divergence count and worst raw split R-hat over the verdict-bearing sites.

    Wrapped defensively like the reference implementations: a diagnostic that
    cannot be computed is reported as ``None`` rather than costing the fit.

    The raw R-hat reads the sampler's draws before label alignment, so two
    chains that each mix well but sit in mirror modes disagree wildly on every
    signed site; that number is kept on the result as ``rhat_max_unaligned``.
    The warning reads ``rhat_aligned``, the R-hat after alignment, which only a
    genuine disagreement between chains raises.
    """
    divergences: int | None
    try:
        divergences = int(idata.sample_stats["diverging"].values.sum())
    except Exception:  # diagnostics never cost the result
        divergences = None

    rhat_max: float | None
    try:
        import arviz as az

        names = [block.site for block in spec.deltas] + [f"{target}_stay"]
        rhat = az.rhat(idata, var_names=names)
        rhat_max = float(max(float(value.max()) for value in rhat.values()))
    except Exception:  # diagnostics never cost the result
        rhat_max = None

    if divergences is not None and total_draws > 0:
        share = divergences / total_draws
        if share > DIVERGENCE_WARN:
            warnings.warn(
                f"{target!r}: {divergences} divergent transitions "
                f"({share:.1%} of draws); the posterior geometry is rough and the "
                "deltas may be underexplored",
                UserWarning,
                stacklevel=3,
            )
    if rhat_aligned is not None and rhat_aligned > RHAT_WARN:
        warnings.warn(
            f"{target!r}: split R-hat reaches {rhat_aligned:.3f} on a verdict-bearing "
            "delta after label alignment; chains genuinely disagree, so consider "
            "more warmup or more chains",
            UserWarning,
            stacklevel=3,
        )
    return divergences, rhat_max
