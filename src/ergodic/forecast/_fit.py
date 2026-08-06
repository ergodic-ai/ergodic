# SPDX-License-Identifier: Apache-2.0
"""Fitting one node, then forecasting it: two models, one posterior between them.

The fit is an ordinary NUTS run on a model built at ``horizon=0``. The forecast
is a *second* model, built at the same ``T`` plus ``H`` steps, into which the
fitted posterior is replayed by name: every training innovation comes back
exactly as it was sampled, and only the horizon's innovations are drawn fresh.
That split is the whole reason the band widens with the horizon instead of
inheriting the fitted history's certainty.

Replay by name is powerful and unforgiving. ``pm.sample_posterior_predictive``
matches variables to the trace by name, and a name it cannot find is drawn from
its prior without comment, which would turn a fitted forecast into a prior
sample that still looks plausible. :func:`_audit_sites` runs before every
posterior-predictive pass and checks the match in both directions, so a
mismatched name is an exception rather than a quiet lie.

The sampler binding: PyMC through its JAX backend. The engine spike measured
that path at parity with raw NumPyro and roughly four times the C backend on
exactly this kind of scan model, and it found that PyMC picks nutpie when
``nuts_sampler`` is left off. So the backend is always named at the call site,
never defaulted.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._mechanism import require_pymc

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from ._mechanism import TemporalMechanism
    from ._scalers import Scaler

    Array = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

__all__ = [
    "NodeDiagnostics",
    "NodeFit",
    "NodeForecast",
    "fit_node",
    "forecast_node",
    "in_sample_median",
]

#: A divergence share above this fraction of the draws earns a warning.
DIVERGENCE_WARN = 0.005
#: A minimum bulk effective sample size below this earns a warning.
MIN_ESS_WARN = 100.0

_FC_SUFFIX = "_fc"
_fallback_warned = False


# -- the engine binding -------------------------------------------------------


def resolve_nuts_sampler() -> str:
    """The NUTS backend to name at the call site: the JAX path when it is installed.

    Never left to PyMC's own default. PyMC picks nutpie when it is present and
    ``nuts_sampler`` is omitted, and the spike measured nutpie at 3.8 times the
    wall clock of the JAX path on the model this pillar fits, so the choice is
    made here and stated.

    Falling back to the C backend is correct but slow, so it says so once.
    """
    global _fallback_warned
    try:
        import jax
        import numpyro  # noqa: F401
    except ImportError:
        if not _fallback_warned:
            _fallback_warned = True
            warnings.warn(
                "sampling through PyMC's C backend because jax and numpyro are not "
                "installed; the JAX path measured about four times faster on this "
                "model. `pip install jax numpyro` to use it.",
                UserWarning,
                stacklevel=2,
            )
        return "pymc"
    # PyTensor works in float64 and JAX defaults to float32, so the backend has
    # to be told, or the sampler quietly loses half the precision of the states.
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    return "numpyro"


# -- records ------------------------------------------------------------------


@dataclass(frozen=True)
class NodeDiagnostics:
    """What the sampler reported about one node's fit.

    Carried on every result, warned about on breach, never raised on: a backtest
    that refits over dozens of folds has to be able to finish, and a diagnostic
    is information about a fit rather than a verdict on it.
    """

    node: str
    divergences: int
    n_draws: int
    min_ess: float
    max_rhat: float | None = None
    ess_by_site: Mapping[str, float] = field(default_factory=dict)

    @property
    def divergence_share(self) -> float:
        """Divergences as a fraction of the draws."""
        return self.divergences / self.n_draws if self.n_draws else float("nan")

    @property
    def healthy(self) -> bool:
        """True when nothing crossed the warning thresholds."""
        return self.divergence_share <= DIVERGENCE_WARN and self.min_ess >= MIN_ESS_WARN

    def __repr__(self) -> str:
        rhat = "" if self.max_rhat is None else f", max_rhat={self.max_rhat:.3f}"
        return (
            f"NodeDiagnostics({self.node}: {self.divergences}/{self.n_draws} divergent "
            f"({self.divergence_share:.2%}), min_ess={self.min_ess:.0f}{rhat})"
        )


@dataclass(frozen=True)
class NodeFit:
    """One fitted node: its mechanism, its scaler, its posterior, its diagnostics."""

    node: str
    mechanism: TemporalMechanism
    scaler: Scaler
    trace: Any
    diagnostics: NodeDiagnostics
    obs_scaled: Array
    mask: BoolArray
    parents: Mapping[str, Array]

    @property
    def n_obs(self) -> int:
        """The number of time steps the model was built over, masked ones included."""
        return int(self.obs_scaled.shape[0])


@dataclass(frozen=True)
class NodeForecast:
    """One node's horizon: predictive draws in the data's own units, plus the pieces.

    ``draws`` is ``(S, H)`` on the original scale. ``components`` holds the
    model's deterministics (the trend, the season, the regression, one array per
    driver) as they came out of the posterior-predictive pass, on the scaled
    space the mechanism works in. The attribution sprint reads them; nothing else
    does yet.
    """

    node: str
    draws: Array
    components: Mapping[str, Array]


# -- fitting ------------------------------------------------------------------


def fit_node(
    node: str,
    values: Array,
    mechanism: TemporalMechanism,
    *,
    mask: BoolArray | None = None,
    parents: Mapping[str, Array] | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 1,
    seed: int = 0,
    target_accept: float = 0.99,
    nuts_sampler: str | None = None,
) -> NodeFit:
    """Fit one node's mechanism to ``values`` and return the posterior with diagnostics.

    ``mask`` defaults to the finite entries of ``values``, which is how a series
    with holes in it declares them. Pass one explicitly to mask a step whose
    recorded value is present but not to be believed. The scaler is fitted on the
    observed steps alone, so a masked sentinel cannot move the scale.

    ``parents`` are scaled, lag-aligned driver paths of length ``T``; at fit time
    there is no horizon to cover. Seeds are explicit and reach both the sampler
    and, later, the forecast.
    """
    pm = require_pymc()

    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"{node!r}: values must be 1-D, got shape {y.shape}")
    observed_mask = np.isfinite(y) if mask is None else np.asarray(mask, dtype=bool)
    if observed_mask.shape != y.shape:
        raise ValueError(f"{node!r}: mask has shape {observed_mask.shape}, expected {y.shape}")
    if not observed_mask.any():
        raise ValueError(f"{node!r}: every step is masked, so there is nothing to fit")

    scaler = mechanism.scaler_for(y[observed_mask])
    scaled = scaler.transform(np.where(observed_mask, y, np.nan))
    parent_paths = {str(k): np.asarray(v, dtype=np.float64) for k, v in (parents or {}).items()}

    t_len = int(y.shape[0])
    with pm.Model() as model:
        mechanism.build(
            node,
            obs=scaled,
            parents=parent_paths,
            T=t_len,
            horizon=0,
            mask=observed_mask,
        )
    backend = nuts_sampler or resolve_nuts_sampler()
    with model:
        trace = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            # Named, never defaulted: PyMC picks nutpie when this is omitted, which
            # the engine spike measured at 3.8x the wall clock of the JAX path.
            nuts_sampler=backend,
            target_accept=target_accept,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": False},
        )

    diagnostics = _diagnostics(node, trace, chains=chains)
    _warn_on_breach(diagnostics)
    return NodeFit(
        node=node,
        mechanism=mechanism,
        scaler=scaler,
        trace=trace,
        diagnostics=diagnostics,
        obs_scaled=scaled,
        mask=observed_mask,
        parents=parent_paths,
    )


# -- forecasting --------------------------------------------------------------


def forecast_node(
    fit: NodeFit,
    horizon: int,
    *,
    parents: Mapping[str, Array] | None = None,
    seed: int = 0,
) -> NodeForecast:
    """Continue a fitted node ``horizon`` steps, replaying its posterior.

    Builds a second model over ``T + horizon`` steps, checks that every site the
    trace has to supply is there and that every site meant to be simulated is
    not, then runs the posterior-predictive pass. The returned draws come back
    through the node's own scaler, so they are in the units the data arrived in.

    ``parents`` are the full ``T + horizon`` driver paths. Without them the
    fitted parents are extended by nothing, which is why a node with parents
    needs its parents' futures resolved before this is called.
    """
    pm = require_pymc()

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    paths = (
        {str(k): np.asarray(v, dtype=np.float64) for k, v in parents.items()}
        if parents is not None
        else dict(fit.parents)
    )
    t_len = fit.n_obs
    with pm.Model() as model:
        fit.mechanism.build(
            fit.node,
            obs=fit.obs_scaled,
            parents=paths,
            T=t_len,
            horizon=horizon,
            mask=fit.mask,
        )

    _audit_sites(model, fit.trace)

    predictive = fit.mechanism.predictive_site(fit.node)
    wanted = [predictive, *(v.name for v in model.deterministics)]
    with model:
        ppc = pm.sample_posterior_predictive(
            fit.trace,
            var_names=wanted,
            predictions=True,
            progressbar=False,
            random_seed=seed,
        )
    drawn = _group_dataset(ppc, "predictions")
    scaled_draws = _flatten_draws(drawn[predictive])
    components = {
        str(name): _flatten_draws(drawn[name]) for name in drawn.data_vars if name != predictive
    }
    return NodeForecast(
        node=fit.node,
        draws=np.asarray(fit.scaler.inverse(scaled_draws), dtype=np.float64),
        components=components,
    )


def in_sample_median(
    fit: NodeFit,
    view: NodeForecast | None = None,
    *,
    seed: int = 0,
) -> Array:
    """The fitted mean path over the training window, ``(T,)`` in the data's own units.

    Where the fitted values come from is a design decision worth stating, because
    the obvious route is closed. A fitting model registers no deterministics at
    all (S2's finding: a deterministic sitting in the trace is read back *from*
    the trace by the posterior-predictive pass rather than recomputed, so a
    fit-time ``mu`` of length ``T`` would be poured into a forecast-time slot of
    length ``T + H``), so there is no ``mu`` in the trace to read.

    But the forecast model already computes one. Its ``mu`` runs over
    ``T + horizon`` steps, and the first ``T`` of them are a pure replay: they
    depend only on parameters and innovations that came out of the trace, never
    on the fresh horizon innovations, because the recursion runs forward in time.
    So the in-sample fitted path is the forecast model's own ``mu[:T]``, free of
    charge when a forecast has already been made, and one small predictive pass
    when it has not.

    Pass ``view`` when :func:`forecast_node` has already run for this fit; the
    two agree exactly, since the horizon seed cannot reach the training rows.
    """
    seen = view if view is not None else forecast_node(fit, 1, seed=seed)
    if seen.node != fit.node:
        raise ValueError(
            f"the forecast is for {seen.node!r} and the fit is for {fit.node!r}; "
            "in-sample values have to come from this node's own model"
        )
    name = fit.mechanism.mean_site(fit.node)
    if name not in seen.components:
        raise ValueError(
            f"the forecast of {fit.node!r} carries no {name!r} deterministic, so it has no "
            "fitted mean path to read. A mechanism has to register its mean path under "
            "mean_site() in forecasting models for in-sample values to exist."
        )
    mu = np.asarray(seen.components[name], dtype=np.float64)
    fitted = np.median(mu[:, : fit.n_obs], axis=0)
    return np.asarray(fit.scaler.inverse(fitted), dtype=np.float64)


def _audit_sites(model: Any, trace: Any) -> None:
    """Check the forecast model against the trace, in both directions, before any ppc.

    Three rules, and every one of them guards a way the run could look fine and
    be wrong:

    1. A free variable the trace holds must hold it at the same shape. A shape
       that drifted would be broadcast into the model rather than refused.
    2. A free variable the trace does not hold must be one of the forecast sites
       (its name ends in ``_fc``). Any other missing name would be drawn from its
       prior, silently turning a fitted forecast into a prior sample.
    3. A forecast site must *not* be in the trace, or it would be replayed
       instead of simulated and the horizon would inherit the training window's
       innovations.

    Deterministics get the same treatment as rule 3: one stored in the trace is
    read back from it rather than recomputed, so a fitting model that recorded
    its own ``mu`` would feed a ``T``-long array into a ``T + H``-long slot.
    """
    posterior = _group_dataset(trace, "posterior")
    held = {str(name): tuple(int(s) for s in posterior[name].shape[2:]) for name in posterior}

    for rv in model.free_RVs:
        name = str(rv.name)
        if name.endswith(_FC_SUFFIX):
            if name in held:
                raise ValueError(
                    f"forecast site {name!r} is in the trace; it would be replayed instead "
                    "of simulated, and the horizon would reuse the training innovations"
                )
            continue
        # Only now: reading a shape can mean walking the graph, and a forecast
        # site's graph runs through the whole scan.
        shape = _static_shape(rv)
        if name not in held:
            raise ValueError(
                f"site {name!r} is in the forecast model but not in the trace; "
                "sample_posterior_predictive would draw it from its prior without "
                "saying so. The two models must register the same names."
            )
        if held[name] != shape:
            raise ValueError(
                f"site {name!r} has shape {shape} in the forecast model and {held[name]} "
                "in the trace; the posterior cannot be replayed into it"
            )

    for det in model.deterministics:
        name = str(det.name)
        if name in held:
            raise ValueError(
                f"deterministic {name!r} is in the trace; the posterior-predictive pass "
                "reads a stored deterministic back rather than recomputing it, so the "
                "fitting model must not record it"
            )


def _static_shape(rv: Any) -> tuple[int, ...]:
    """A model variable's shape, from the type when it is known and by evaluation when not."""
    declared = tuple(rv.type.shape)
    if all(dim is not None for dim in declared):
        return tuple(int(dim) for dim in declared)
    return tuple(int(dim) for dim in rv.shape.eval())


# -- diagnostics --------------------------------------------------------------


def _diagnostics(node: str, trace: Any, *, chains: int) -> NodeDiagnostics:
    """Divergences, minimum bulk ESS over the scalar sites, r-hat when there are chains."""
    import arviz as az

    posterior = _group_dataset(trace, "posterior")
    scalars = [str(name) for name in posterior.data_vars if posterior[name].ndim == 2]
    ess_by_site: dict[str, float] = {}
    if scalars:
        # az.ess(ds, var_names=[...]) resolves only the first variable in arviz 1.x
        # and raises on the rest, so the selection happens on the dataset first.
        ess = az.ess(posterior[scalars])
        ess_by_site = {name: float(np.asarray(ess[name])) for name in scalars}
    min_ess = min(ess_by_site.values()) if ess_by_site else float("nan")

    max_rhat: float | None = None
    if chains >= 2 and scalars:
        rhat = az.rhat(posterior[scalars])
        max_rhat = max(float(np.asarray(rhat[name])) for name in scalars)

    divergences = 0
    try:
        stats = _group_dataset(trace, "sample_stats")
    except KeyError:
        stats = None
    if stats is not None and "diverging" in stats:
        divergences = int(np.asarray(stats["diverging"]).sum())

    first = next(iter(posterior.data_vars))
    n_draws = int(np.prod(posterior[first].shape[:2]))
    return NodeDiagnostics(
        node=node,
        divergences=divergences,
        n_draws=n_draws,
        min_ess=min_ess,
        max_rhat=max_rhat,
        ess_by_site=ess_by_site,
    )


def _warn_on_breach(diagnostics: NodeDiagnostics) -> None:
    """Say so when a fit crossed a threshold. Warn, never raise."""
    share = diagnostics.divergence_share
    if share > DIVERGENCE_WARN:
        warnings.warn(
            f"{diagnostics.node}: {diagnostics.divergences} of {diagnostics.n_draws} draws "
            f"diverged ({share:.2%}); the posterior may be biased. Raising target_accept or "
            "reparameterizing the node is the usual fix.",
            UserWarning,
            stacklevel=3,
        )
    if diagnostics.min_ess < MIN_ESS_WARN:
        warnings.warn(
            f"{diagnostics.node}: minimum bulk ESS is {diagnostics.min_ess:.0f}, below "
            f"{MIN_ESS_WARN:.0f}; intervals from this fit are noisy. More draws is the "
            "usual fix.",
            UserWarning,
            stacklevel=3,
        )


# -- trace plumbing -----------------------------------------------------------


def _group_dataset(trace: Any, group: str) -> Any:
    """One group of a sampler result as an xarray Dataset.

    PyMC 6 hands back an xarray ``DataTree`` where earlier versions returned an
    arviz ``InferenceData``. Both index by group name; only one of them needs the
    extra ``to_dataset`` step. This is the only place that difference lives.
    """
    node = trace[group] if group in trace else getattr(trace, group, None)
    if node is None:
        raise KeyError(f"the sampler result has no {group!r} group")
    return node.to_dataset() if hasattr(node, "to_dataset") else node


def _flatten_draws(array: Any) -> Array:
    """``(chain, draw, ...)`` to ``(S, ...)``, the shape the result object reads."""
    values = np.asarray(array, dtype=np.float64)
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def sorted_seeds(nodes: Sequence[str], seed: int) -> dict[str, int]:
    """One seed per node: ``seed`` plus the node's position in sorted order.

    Sorted rather than given order, so the same set of series gets the same seeds
    however the caller happened to arrange the columns.
    """
    return {node: seed + index for index, node in enumerate(sorted(nodes))}
