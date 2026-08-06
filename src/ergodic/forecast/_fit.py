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

from ._mechanism import is_symbolic, require_pymc

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

    from ._mechanism import TemporalMechanism
    from ._scalers import Scaler

    Array = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

__all__ = [
    "FrozenParent",
    "JointFit",
    "JointNode",
    "NodeDiagnostics",
    "NodeFit",
    "NodeForecast",
    "ParentColumn",
    "ReplayRecipe",
    "align_draws",
    "fit_joint",
    "fit_node",
    "forecast_joint",
    "forecast_node",
    "in_sample_median",
    "inject_futures",
    "lag_align",
    "posterior_draws",
    "zero_sites",
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
class ReplayRecipe:
    """Everything :func:`forecast_node` needs to run one forecast again, unchanged.

    A replay attribution zeroes a factor's sites in a copy of the trace and asks
    for the same forecast back, so it has to be able to reproduce the call: the
    same horizon, the same driver paths, the same handoffs, and above all the
    same seed. With the seed held the fresh innovations come out identical, so
    differencing two replays leaves the factor's effect and nothing else, rather
    than the factor's effect plus a new draw of the noise.

    ``None`` on a :class:`NodeForecast` means the forecast cannot be reproduced
    on its own, which is what a joint fit is: one predictive pass produces every
    node at once, and there is no per-node call to repeat.
    """

    horizon: int
    seed: int
    parents: Mapping[str, Array] = field(default_factory=dict)
    frozen: tuple[FrozenParent, ...] = ()


@dataclass(frozen=True)
class NodeForecast:
    """One node's horizon: predictive draws in the data's own units, plus the pieces.

    ``draws`` is ``(S, H)`` on the original scale. ``components`` holds the
    model's deterministics (the trend, the season, the regression, one array per
    driver) as they came out of the posterior-predictive pass, on the scaled
    space the mechanism works in. :meth:`ForecastResult.explain` reads them, which
    is what makes an attribution cost no sampling on the structural path.
    ``replay`` is the recipe for running this same forecast again, which is what
    the fallback path needs.
    """

    node: str
    draws: Array
    components: Mapping[str, Array]
    replay: ReplayRecipe | None = None


# -- the lag-shift design -----------------------------------------------------


def lag_align(history: Array, future: Any | None, lag: int, horizon: int) -> Any:
    """One parent column, shifted ``lag`` steps right and padded at the front.

    An edge ``x[t-k] -> y[t]`` says row ``t`` of the child's design reads the
    parent at ``t - k``. The first ``k`` rows reach back before the series starts,
    so they are **front-padded with the parent's first observed value**. That
    choice is worth stating because it is an assumption: it says the driver was
    where it started before we began watching, which is the mildest thing to
    assume and is wrong in exactly one direction, understating the trend over the
    first ``k`` steps. The alternative, masking the first ``max_lag`` steps out of
    the likelihood, is a real option and is deliberately not taken in v1: with
    ``k`` small against ``T`` the padded rows are a fraction of a percent of the
    information, and dropping rows makes two nodes at different lags fit
    different windows.

    ``history`` is the parent over the training window, ``(T,)``, on the parent's
    own scaled space. ``future`` is the parent over the horizon, ``(H,)``, either
    a numpy array (a known future) or a tensor of the model being built (a frozen
    parent's ``{base}_fc`` site, or an upstream node's mean path). The result runs
    ``T + horizon`` steps, and it is a tensor exactly when the future it needed
    was one.

    A lag at or past the horizon never reads the future at all: every horizon row
    still falls inside the training window, so the column comes back pure numpy.
    """
    hist = np.asarray(history, dtype=np.float64)
    if hist.ndim != 1 or hist.size == 0:
        raise ValueError(f"a parent history must be a non-empty 1-D array, got shape {hist.shape}")
    if int(lag) < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if int(horizon) < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    t_len = int(hist.shape[0])
    n_total = t_len + int(horizon)
    head = np.concatenate([np.full(int(lag), hist[0]), hist])[:n_total]
    tail_len = n_total - min(n_total, t_len + int(lag))
    if tail_len == 0:
        return head
    if future is None:
        raise ValueError(
            f"a parent at lag {lag} needs {tail_len} step(s) of its own future to cover a "
            f"horizon of {horizon}, and none was given"
        )
    if is_symbolic(future):
        pm = require_pymc()
        return pm.math.concatenate([head, future[:tail_len]])
    tail = np.asarray(future, dtype=np.float64)
    if tail.ndim != 1 or tail.shape[0] < tail_len:
        raise ValueError(
            f"a parent at lag {lag} needs {tail_len} future step(s), and the future given "
            f"has shape {tail.shape}"
        )
    return np.concatenate([head, tail[:tail_len]])


@dataclass(frozen=True)
class ParentColumn:
    """One regressor of one node: which variable, at which lag, under which name.

    ``key`` is the design name the mechanism sees, and therefore the ``beta_{key}``
    site it registers. It is the parent's plain variable name when that parent
    drives this node at a single lag, which is the ordinary case and keeps the S2
    naming contract intact; when the same parent drives at several lags the
    shortest keeps the plain name and the others read ``{base}_lag{k}``.

    ``history`` and ``future`` are on the **parent's** scaled space, which is the
    space the child's regression reads it in. ``future`` is ``None`` for a parent
    whose horizon arrives as posterior draws instead (see :class:`FrozenParent`)
    and for a parent that is another node of a joint model.
    """

    key: str
    base: str
    lag: int
    history: Array | None = None
    future: Array | None = None
    upstream: bool = False


@dataclass(frozen=True)
class FrozenParent:
    """A parent whose horizon arrives as its own posterior draws.

    This is the staged handoff. The child's forecast model declares one fresh site
    ``{base}_fc`` of length ``H``; the parent's stored forecast draws, transformed
    into the parent's scaled space, are written into a copy of the child's trace
    under that name; and the posterior-predictive pass replays them by name like
    any other fitted site. Nothing collapses to a point: every draw of the child
    sees the matching draw of its parent, so the parent's whole forecast
    distribution reaches the child's band.

    ``draws`` is ``(S, H)`` on the parent's scaled space, with ``S`` equal to the
    child's own draw count. ``columns`` lists the design keys that read this
    parent and the lag each reads it at.
    """

    base: str
    draws: Array
    history: Array
    columns: tuple[tuple[str, int], ...]

    @property
    def site(self) -> str:
        """The name of the fresh site the child's forecast model declares."""
        return f"{self.base}{_FC_SUFFIX}"


def align_draws(draws: Array, n_target: int, *, seed: int) -> Array:
    """Make a parent's ``(S, H)`` draws match the child's draw count.

    Equal by construction whenever one ``draws=`` sets the budget for the whole
    graph, which is the normal path and costs nothing. When they differ (a parent
    fitted separately, at another budget) the parent is resampled with
    replacement under an explicit seed, which keeps the handoff a sample from the
    parent's posterior rather than a reshaped one.
    """
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"parent draws must be 2-D (S, H), got shape {values.shape}")
    if int(values.shape[0]) == int(n_target):
        return values
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.shape[0], size=int(n_target))
    return np.asarray(values[picks], dtype=np.float64)


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
    frozen: Iterable[FrozenParent] | None = None,
    seed: int = 0,
) -> NodeForecast:
    """Continue a fitted node ``horizon`` steps, replaying its posterior.

    Builds a second model over ``T + horizon`` steps, checks that every site the
    trace has to supply is there and that every site meant to be simulated is
    not, then runs the posterior-predictive pass. The returned draws come back
    through the node's own scaler, so they are in the units the data arrived in.

    ``parents`` are the full ``T + horizon`` driver paths, on each parent's own
    scaled space, for parents whose future is known outright. ``frozen`` carries
    the parents whose future is a posterior instead: each declares a fresh
    ``{base}_fc`` site here, gets its draws written into a copy of this node's
    trace, and is replayed from there, so the parent's uncertainty propagates
    into this node's band rather than collapsing to a path.
    """
    pm = require_pymc()

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    handoffs = tuple(frozen or ())
    paths: dict[str, Any] = (
        {str(k): np.asarray(v, dtype=np.float64) for k, v in parents.items()}
        if parents is not None
        else dict(fit.parents)
    )
    t_len = fit.n_obs
    # Capture the recipe before the handoffs turn their columns into tensors of
    # this model: what a replay needs is the data paths plus the handoffs
    # themselves, from which the tensors get rebuilt exactly the same way.
    recipe = ReplayRecipe(
        horizon=int(horizon),
        seed=int(seed),
        parents={key: np.asarray(value, dtype=np.float64) for key, value in paths.items()},
        frozen=handoffs,
    )
    with pm.Model() as model:
        for handoff in handoffs:
            # The prior here is never used: the audit below refuses to run the
            # predictive pass unless every one of these names is in the trace, and
            # a name in the trace is read from it rather than sampled. It exists so
            # the site is an ordinary free variable of the model.
            future = pm.Normal(handoff.site, mu=0.0, sigma=1.0, shape=horizon)
            for key, lag in handoff.columns:
                paths[key] = lag_align(handoff.history, future, lag, horizon)
        fit.mechanism.build(
            fit.node,
            obs=fit.obs_scaled,
            parents=paths,
            T=t_len,
            horizon=horizon,
            mask=fit.mask,
        )

    trace = fit.trace
    injected = tuple(handoff.site for handoff in handoffs)
    if handoffs:
        trace = inject_futures(
            trace,
            {handoff.site: handoff.draws for handoff in handoffs},
        )
    _audit_sites(model, trace, injected=injected)

    predictive = fit.mechanism.predictive_site(fit.node)
    wanted = [predictive, *(v.name for v in model.deterministics)]
    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
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
        replay=recipe,
    )


def inject_futures(trace: Any, futures: Mapping[str, Array]) -> Any:
    """Write frozen parent futures into a copy of a trace, ready to be replayed.

    Returns an augmented posterior as an ``xarray`` Dataset, which is one of the
    shapes ``pm.sample_posterior_predictive`` reads. The trace handed in is never
    touched: a fitted posterior is a record, and one node's handoff must not
    change what another node replays.

    Each ``(S, H)`` array is folded back onto the trace's own ``(chain, draw)``
    grid in the order :func:`_flatten_draws` unfolded it, so draw ``s`` of the
    parent lands on draw ``s`` of the child. Under staged fitting those two
    posteriors are independent by construction, so the pairing is a coupling of
    two exchangeable samples rather than a joint draw, and any pairing would do.
    What matters is that there *is* one: the child's forecast is then evaluated
    once per parent draw, which is what carries the parent's whole distribution
    into the child's band instead of averaging it away.
    """
    posterior = _group_dataset(trace, "posterior")
    augmented = posterior.copy()
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    for name, values in futures.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"the future for {name!r} must be 2-D (S, H), got {array.shape}")
        if int(array.shape[0]) != n_chain * n_draw:
            raise ValueError(
                f"the future for {name!r} carries {array.shape[0]} draws and the trace holds "
                f"{n_chain * n_draw}; align the draw counts before injecting"
            )
        augmented[name] = (
            ("chain", "draw", f"{name}_dim_0"),
            array.reshape(n_chain, n_draw, int(array.shape[1])),
        )
    return augmented


def zero_sites(trace: Any, names: Iterable[str]) -> Any:
    """Set the named sites to zero in a copy of a trace, leaving everything else alone.

    This is how the replay engine switches a factor off. Zero is the neutral
    value by construction under the pinned parameterization: a coefficient of
    zero contributes nothing, and a zeroed ``season_raw`` gives
    ``12 * softmax(0)``, which is exactly one, so the seasonal factor lands on
    its identity rather than near it. The Fourier form is additive, so zeroing
    its coefficients removes the term outright.

    The trace handed in is never touched. A name that is not in the posterior
    raises, because silently zeroing nothing would hand back a bar of zero for a
    factor that is still fully switched on.
    """
    import xarray as xr

    posterior = _posterior_of(trace)
    wanted = list(dict.fromkeys(str(name) for name in names))
    missing = [name for name in wanted if name not in posterior.data_vars]
    if missing:
        raise KeyError(
            f"the trace has no site(s) {missing}, so they cannot be zeroed; it holds "
            f"{sorted(str(name) for name in posterior.data_vars)}"
        )
    augmented = posterior.copy()
    for name in wanted:
        augmented[name] = xr.zeros_like(posterior[name])
    return augmented


def posterior_draws(trace: Any) -> dict[str, Array]:
    """Every posterior site as ``(S, ...)`` draws, the chains stacked onto one axis.

    The flat view a decomposition reads: a mechanism whose composition rule needs
    a fitted parameter as well as the recorded pieces looks it up here by site
    name.
    """
    posterior = _posterior_of(trace)
    return {str(name): _flatten_draws(posterior[name]) for name in posterior.data_vars}


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


# -- the joint fit ------------------------------------------------------------


@dataclass(frozen=True)
class JointNode:
    """One node's slot in a joint model: what to build and what it reads."""

    node: str
    mechanism: TemporalMechanism
    scaler: Scaler
    obs_scaled: Array
    mask: BoolArray
    parents: tuple[ParentColumn, ...] = ()


@dataclass(frozen=True)
class JointFit:
    """One posterior over every node of a joint model.

    ``sites`` maps each node to the names it registered, recorded while the model
    was built rather than guessed from a name prefix, so two nodes whose names
    share a prefix are still told apart exactly.
    """

    nodes: tuple[JointNode, ...]
    trace: Any
    sites: Mapping[str, tuple[str, ...]]
    diagnostics: Mapping[str, NodeDiagnostics]

    @property
    def n_obs(self) -> int:
        """The number of time steps the model was built over."""
        return int(self.nodes[0].obs_scaled.shape[0])

    def node_of(self, name: str) -> JointNode:
        """One node's slot, by name."""
        for slot in self.nodes:
            if slot.node == name:
                return slot
        raise KeyError(f"no node {name!r} in this joint fit")


def fit_joint(
    nodes: Sequence[JointNode],
    *,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 1,
    seed: int = 0,
    target_accept: float = 0.99,
    nuts_sampler: str | None = None,
) -> JointFit:
    """Fit every node of a graph in one model, so their posteriors stay correlated.

    ``nodes`` arrives in topological order: a node's parents are built before it,
    and a parent that is another node of this model feeds the child through its
    own **mean path** rather than through its observed series. That is the whole
    difference from staged fitting. The child's likelihood therefore says
    something about the parent's latent states, and the parent's uncertainty
    reaches the child as a joint posterior instead of as a handoff, which is what
    "the correlation staging drops" means concretely.

    It costs what one sampler over everything costs, which is why the facade
    keeps it for small graphs.
    """
    pm = require_pymc()

    slots = tuple(nodes)
    if not slots:
        raise ValueError("a joint fit needs at least one node")
    t_len = int(slots[0].obs_scaled.shape[0])
    for slot in slots:
        if int(slot.obs_scaled.shape[0]) != t_len:
            raise ValueError(
                f"{slot.node!r} was built over {slot.obs_scaled.shape[0]} steps and "
                f"{slots[0].node!r} over {t_len}; a joint model shares one time window"
            )

    with pm.Model() as model:
        sites = _build_joint(slots, t_len, horizon=0)
    backend = nuts_sampler or resolve_nuts_sampler()
    with model:
        trace = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            # Named, never defaulted: the reason is at fit_node's own call site.
            nuts_sampler=backend,
            target_accept=target_accept,
            compute_convergence_checks=False,
            idata_kwargs={"log_likelihood": False},
        )

    diagnostics: dict[str, NodeDiagnostics] = {}
    for slot in slots:
        record = _diagnostics(slot.node, trace, chains=chains, sites=sites[slot.node])
        _warn_on_breach(record)
        diagnostics[slot.node] = record
    return JointFit(nodes=slots, trace=trace, sites=sites, diagnostics=diagnostics)


def forecast_joint(fit: JointFit, horizon: int, *, seed: int = 0) -> dict[str, NodeForecast]:
    """Continue every node of a joint fit at once, in one predictive pass.

    The horizon's driver values are the drivers' own forecasts, computed inside
    the same model from the same posterior draw, so a child never sees a parent
    path that its own parameters did not come from.
    """
    pm = require_pymc()

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    t_len = fit.n_obs
    with pm.Model() as model:
        sites = _build_joint(fit.nodes, t_len, horizon=horizon)
    _audit_sites(model, fit.trace)

    predictive = {slot.node: slot.mechanism.predictive_site(slot.node) for slot in fit.nodes}
    wanted = [*predictive.values(), *(v.name for v in model.deterministics)]
    with model:
        ppc = pm.sample_posterior_predictive(
            fit.trace,
            var_names=wanted,
            predictions=True,
            progressbar=False,
            random_seed=seed,
        )
    drawn = _group_dataset(ppc, "predictions")
    out: dict[str, NodeForecast] = {}
    for slot in fit.nodes:
        name = predictive[slot.node]
        scaled = _flatten_draws(drawn[name])
        components = {
            str(site): _flatten_draws(drawn[site])
            for site in sites[slot.node]
            if site != name and site in drawn.data_vars
        }
        out[slot.node] = NodeForecast(
            node=slot.node,
            draws=np.asarray(slot.scaler.inverse(scaled), dtype=np.float64),
            components=components,
        )
    return out


def _build_joint(
    nodes: Sequence[JointNode], t_len: int, *, horizon: int
) -> dict[str, tuple[str, ...]]:
    """Build every node into the current model context and record whose sites are whose."""
    means: dict[str, Any] = {}
    sites: dict[str, tuple[str, ...]] = {}
    model = require_pymc().modelcontext(None)
    seen = _model_names(model)
    for slot in nodes:
        paths: dict[str, Any] = {}
        for column in slot.parents:
            if column.upstream:
                if column.base not in means:
                    raise ValueError(
                        f"{slot.node!r} reads {column.base!r}, which this joint model has not "
                        "built yet; the nodes have to arrive in topological order"
                    )
                paths[column.key] = _shift_mean(means[column.base], column.lag, t_len + horizon)
            else:
                if column.history is None:
                    raise ValueError(
                        f"the column {column.key!r} of {slot.node!r} is neither upstream nor "
                        "carries a history, so there is nothing to read"
                    )
                paths[column.key] = lag_align(column.history, column.future, column.lag, horizon)
        means[slot.node] = slot.mechanism.build(
            slot.node,
            obs=slot.obs_scaled,
            parents=paths,
            T=t_len,
            horizon=horizon,
            mask=slot.mask,
        )
        after = _model_names(model)
        sites[slot.node] = tuple(sorted(after - seen))
        seen = after
    return sites


def _model_names(model: Any) -> set[str]:
    """Every name registered in a model so far, free variables and deterministics alike."""
    return {str(v.name) for v in (*model.free_RVs, *model.observed_RVs, *model.deterministics)}


def _shift_mean(mean: Any, lag: int, n_total: int) -> Any:
    """An upstream node's mean path, shifted ``lag`` steps and front-padded with its first value.

    The tensor sibling of :func:`lag_align`, and the padding rule is the same one.
    """
    if int(lag) == 0:
        return mean
    pm = require_pymc()
    head = pm.math.ones(int(lag)) * mean[0]
    return pm.math.concatenate([head, mean[: n_total - int(lag)]])


def _audit_sites(model: Any, trace: Any, *, injected: Iterable[str] = ()) -> None:
    """Check the forecast model against the trace, in both directions, before any ppc.

    Four rules, and every one of them guards a way the run could look fine and
    be wrong:

    1. A free variable the trace holds must hold it at the same shape. A shape
       that drifted would be broadcast into the model rather than refused.
    2. A free variable the trace does not hold must be one of the forecast sites
       (its name ends in ``_fc``). Any other missing name would be drawn from its
       prior, silently turning a fitted forecast into a prior sample.
    3. A forecast site must *not* be in the trace, or it would be replayed
       instead of simulated and the horizon would inherit the training window's
       innovations.
    4. No deterministic of this model may be in the trace: one stored there is
       read back rather than recomputed, so a fitting model that recorded its own
       ``mu`` would feed a ``T``-long array into a ``T + H``-long slot.

    ``injected`` names the frozen parent futures the caller has just written into
    the trace, and it flips rule 3 into rule 1 for exactly those names: a
    ``{base}_fc`` that was injected has to be there, at the right shape, because
    being replayed is the point of it. Every other ``_fc`` site stays
    required-fresh.

    **That is a caller invariant, and it is the one thing here that can be
    undone from outside.** :func:`forecast_node` derives the declared sites and
    the injected list from the same tuple of :class:`FrozenParent` records, in
    one pass, which is what makes them impossible to disagree. Anyone who
    separates the two (declaring a site in one place and assembling the injected
    names in another) re-opens exactly the hole this function exists to close: a
    handoff that was declared but not written would be a ``_fc`` name absent from
    both the trace and this list, rule 2 would wave it through as
    required-fresh, and the child would forecast against a driver drawn from a
    standard normal prior without anything saying so. Never build this list by
    hand.
    """
    posterior = _posterior_of(trace)
    held = {str(name): tuple(int(s) for s in posterior[name].shape[2:]) for name in posterior}
    replayed_fc = {str(name) for name in injected}
    declared = {str(rv.name) for rv in model.free_RVs}
    orphans = sorted(replayed_fc - declared)
    if orphans:
        raise ValueError(
            f"the futures {orphans} were injected into the trace but this model never "
            "declared them, so nothing would read them and the node would forecast "
            "without its driver"
        )

    for rv in model.free_RVs:
        name = str(rv.name)
        if name.endswith(_FC_SUFFIX) and name not in replayed_fc:
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


def _diagnostics(
    node: str, trace: Any, *, chains: int, sites: Sequence[str] | None = None
) -> NodeDiagnostics:
    """Divergences, minimum bulk ESS over the scalar sites, r-hat when there are chains.

    ``sites`` restricts the reading to one node's own names, which is what a joint
    fit needs: one trace holds every node, and mixing another node's ESS into this
    node's minimum would report a problem in the wrong place. Divergences stay
    model-wide, because a divergence is a property of the joint posterior rather
    than of any one node in it.
    """
    import arviz as az

    posterior = _group_dataset(trace, "posterior")
    wanted = None if sites is None else set(sites)
    scalars = [
        str(name)
        for name in posterior.data_vars
        if posterior[name].ndim == 2 and (wanted is None or str(name) in wanted)
    ]
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


def _posterior_of(trace: Any) -> Any:
    """The posterior of a sampler result, or the dataset itself when it already is one.

    :func:`inject_futures` hands back a bare posterior Dataset (which is one of the
    shapes the predictive pass reads), so anything that reads a trace has to take
    both shapes.
    """
    import xarray as xr

    if isinstance(trace, xr.Dataset):
        return trace
    return _group_dataset(trace, "posterior")


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
