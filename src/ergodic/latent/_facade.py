# SPDX-License-Identifier: Apache-2.0
"""The facade: a series and a likelihood in, a recovered switch out.

Three verbs, three questions. :func:`recover` answers WHEN the regime was on
and WHICH named mechanism block changed. :func:`certify` answers WHETHER a
regime exists at all, vocabulary-free, by placing the observed two-state
improvement inside a null distribution simulated from the likelihood's own
fitted one-state model. :func:`calibrate` measures whether a configured
detector is honest: its fire rate on control worlds that have no regime.
Keeping the verbs apart is the point of the pillar: a delta interval that
excludes zero says which block moved *given* a regime, and reading existence
off it would let a model that can imagine a switch anywhere certify its own
imagination. :func:`consensus` then lays several likelihoods' verdicts on the
same series side by side, block by block, because each model speaks its own
vocabulary and a disagreement between them is information, not noise.

Everything user-facing is validated here, before anything samples, because a
refusal after a NUTS run is a refusal that cost minutes. The engine behind
the facade assumes clean inputs.

:func:`latent_model` is the configuration-as-value helper (the forecast
pillar's ``Forecaster`` pattern): every keyword :func:`recover` takes except
the data and the target, validated eagerly and frozen, so calibration can
deep-copy one spec per control world and a typo costs an exception now rather
than at the twentieth world.

The implicit graph is one hidden regime driving one observed series. A
``graph=`` keyword exists so the multi-series form has a named seam, and
anything except ``None`` is refused; internally the engine already takes a
list of emission tensors, so the seam is real rather than decorative.
"""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import HierarchicalTSData, PanelData, TimeSeriesData, timeseries
from ._engine import fit_recover, improvement_statistic
from ._likelihood import MarkovChain, RegimeLikelihood
from ._result import CalibrationReport, CalibrationWorld, CertifyResult, Consensus, LatentResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    Array = npt.NDArray[np.float64]

__all__ = [
    "LatentModel",
    "calibrate",
    "certify",
    "consensus",
    "latent_model",
    "recover",
    "wilson",
]


def _validated_series(
    data: TimeSeriesData,
    target: str,
    model: RegimeLikelihood,
    chain: MarkovChain | None,
    *,
    verb: str,
) -> tuple[Array, MarkovChain]:
    """The checks :func:`recover` and :func:`certify` share, refusing by name.

    Returns the finite float series and the resolved chain (the likelihood's
    own default when ``chain`` is None).
    """
    if isinstance(data, HierarchicalTSData):
        raise TypeError(
            f"{verb}() reads one series; a HierarchicalTSData is a tree of them. "
            "Select the level you mean and build a TimeSeriesData."
        )
    if isinstance(data, PanelData):
        raise TypeError(
            f"{verb}() reads one series; a PanelData holds one per entity. "
            "Pooling entities over a shared switch is not supported; select "
            "one entity and build a TimeSeriesData."
        )
    if not isinstance(data, TimeSeriesData):
        raise TypeError(
            f"{verb}() needs a TimeSeriesData, got {type(data).__name__}; build "
            "one with ergodic.timeseries(frame, time=...)"
        )
    if not isinstance(model, RegimeLikelihood):
        raise TypeError(
            f"model must be a RegimeLikelihood, got {type(model).__name__}; use "
            "spectral_whittle(), switching_ar(), or subclass RegimeLikelihood"
        )
    if chain is None:
        chain = model.default_chain()
    elif not isinstance(chain, MarkovChain):
        raise TypeError(
            f"chain must be a MarkovChain, got {type(chain).__name__}; build one "
            "with MarkovChain(stay_alpha, stay_beta)"
        )
    if target not in data.variables:
        raise KeyError(
            f"target {target!r} is not a modeled column; the data holds {list(data.variables)}"
        )
    if not data.schema[target].is_continuous:
        raise TypeError(
            f"target {target!r} is {data.schema[target].kind.value}, and the "
            "likelihoods read a continuous series; there are no count or "
            "categorical emissions"
        )
    y = np.asarray(data.column(target), dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"target {target!r} has non-finite entries; the switching likelihoods "
            "read a gapless series, so drop or fill the holes first "
            "(data.dropna() drops the rows)"
        )
    return y, chain


def recover(
    data: TimeSeriesData,
    target: str,
    *,
    model: RegimeLikelihood,
    chain: MarkovChain | None = None,
    graph: Any | None = None,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    seed: int = 0,
    decode_draws: int = 150,
) -> LatentResult:
    """Recover a hidden two-state regime from one observed series.

    Fits the switching model ``model`` describes (its emission, its declared
    mechanism deltas, its gates) with the hidden path marginalized out, then
    decodes the path per posterior draw and aligns the labels. The result
    carries the posterior state marginals, the hard path, the sign-aligned
    delta draws, and the familywise-corrected verdicts.

    Parameters
    ----------
    data:
        A :class:`~ergodic.data.TimeSeriesData`; rows are already in time
        order. Panel and hierarchical inputs are refused by name.
    target:
        The continuous column to read the regime from.
    model:
        A :class:`RegimeLikelihood`: :func:`~ergodic.latent.likelihoods.spectral_whittle`
        or :func:`~ergodic.latent.likelihoods.switching_ar`, or your own subclass.
    chain:
        The transition prior as a :class:`MarkovChain`. ``None`` means the
        likelihood's own default at its emission resolution, which is almost
        always what you want: dwell knowledge stated per tick is wrong by a
        factor of the frame length for a framed likelihood.
    graph:
        Not supported: one regime drives one series; pass nothing.
    draws, warmup, chains, seed:
        The sampler budget and seed. The defaults are the measured reference
        configuration (4 chains of 1000 after 1000 warmup).
    decode_draws:
        How many posterior draws to decode and align (evenly spaced over the
        posterior). The path marginals stabilize well before the full
        posterior is spent, and each decoded draw costs a forward-backward
        pass.

    Returns
    -------
    LatentResult
        Frozen: state marginals, hard path, delta draws, gates, verdicts,
        diagnostics.
    """
    if graph is not None:
        raise NotImplementedError(
            "graph= is not supported: recover reads one regime driving one "
            "series (the implicit regime -> target graph); several series "
            "sharing one switch are not supported."
        )
    y, chain = _validated_series(data, target, model, chain, verb="recover")
    for name, value in (
        ("draws", draws),
        ("warmup", warmup),
        ("chains", chains),
        ("decode_draws", decode_draws),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")

    return fit_recover(
        y,
        target,
        model,
        chain,
        draws=int(draws),
        warmup=int(warmup),
        chains=int(chains),
        seed=int(seed),
        decode_draws=int(decode_draws),
    )


@dataclass(frozen=True)
class LatentModel:
    """A validated recover() configuration, held as a value.

    Built by :func:`latent_model`; everything :func:`recover` accepts except
    the data and the target lives in ``options``. ``run`` applies it. The
    calibration harness deep-copies one of these per control world, which is
    why the configuration is a frozen value rather than a closure.
    """

    model: RegimeLikelihood
    options: Mapping[str, Any] = field(default_factory=dict)

    def run(self, data: TimeSeriesData, target: str) -> LatentResult:
        """Recover with this configuration."""
        return recover(data, target, model=self.model, **dict(self.options))

    def __repr__(self) -> str:
        body = ", ".join(f"{key}={value!r}" for key, value in sorted(self.options.items()))
        extra = f", {body}" if body else ""
        return f"LatentModel({type(self.model).__name__}{extra})"


def latent_model(*, model: RegimeLikelihood, **options: Any) -> LatentModel:
    """Configuration now, data later: :func:`recover`'s keywords as a frozen spec.

    Validated eagerly. A keyword :func:`recover` does not take is refused here,
    and so is a chain of the wrong type, because the consumer of this spec is
    :func:`calibrate`, where the first use of a bad option would otherwise be
    twenty NUTS runs in.
    """
    if not isinstance(model, RegimeLikelihood):
        raise TypeError(
            f"model must be a RegimeLikelihood, got {type(model).__name__}; use "
            "spectral_whittle(), switching_ar(), or subclass RegimeLikelihood"
        )
    allowed = {
        key
        for key, value in inspect.signature(recover).parameters.items()
        if key not in ("data", "target", "model") and value.kind is not value.VAR_KEYWORD
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(f"recover() takes no argument(s) {unknown}; it accepts {sorted(allowed)}")
    chain = options.get("chain")
    if chain is not None and not isinstance(chain, MarkovChain):
        raise TypeError(
            f"chain must be a MarkovChain, got {type(chain).__name__}; build one "
            "with MarkovChain(stay_alpha, stay_beta)"
        )
    return LatentModel(model=model, options=dict(options))


def certify(
    data: TimeSeriesData,
    target: str,
    *,
    model: RegimeLikelihood,
    chain: MarkovChain | None = None,
    n_null: int = 99,
    seed: int = 0,
    refit_draws: int = 200,
    refit_warmup: int = 200,
) -> CertifyResult:
    """WHETHER a regime exists at all: a simulated-null likelihood-ratio test.

    The statistic is twice the log-likelihood gain of the two-state model
    over the likelihood's own fitted one-state null: the two-state marginal
    log-likelihood (hidden path summed out by the forward algorithm) at a
    cheap point estimate of the parameters, minus ``model.null_loglik``. The
    null distribution is measured, not assumed: ``n_null`` control worlds are
    drawn from the fitted one-state model (``model.simulate_null``) and each
    is pushed through the identical computation. The p-value is
    ``(1 + #{null >= observed}) / (1 + n_null)``, so with the default 99
    worlds the finest attainable evidence is 0.01, and reporting less would
    claim resolution the simulation did not buy.

    The cheap refit is a short single-chain posterior sample (``refit_draws``
    after ``refit_warmup``, seed fixed across worlds), with the statistic
    evaluated at the posterior-mean parameters. The alternative, a
    ``pm.find_MAP`` point, was measured on one world of each likelihood and
    rejected: on the spectral model the Laplace shrinkage pulls the MAP's
    deltas to zero and the "improvement" came out negative on a genuine
    level world (-7.3 against the short sample's +25.8) while also running
    about seven times slower (30 s against 4 s per world, the optimizer
    recompiling per world where the short sample amortizes); on the AR model
    both were adequate (91.2 against 90.2 on the same world) and only there
    was MAP faster. The calibration argument for the choice is that the
    observed series and every null world go through exactly the same
    procedure, budget and seed included, so under the null the observed
    statistic is exchangeable with the simulated ones and the p-value is
    honest by construction; a fancier fit for the observed side alone would
    inflate the observed statistic relative to its null and manufacture
    significance.

    Budget honestly stated: one short fit per world, so the default
    ``n_null=99`` costs about 100 short fits (minutes on the spectral
    flagship, longer on the tick-resolution AR). Trim ``n_null`` for a
    coarser but faster answer; the p-value's floor rises with it.

    Returns a :class:`CertifyResult` with the observed statistic, the null
    array, and the p-value.
    """
    y, chain = _validated_series(data, target, model, chain, verb="certify")
    if int(n_null) < 1:
        raise ValueError(f"n_null must be >= 1, got {n_null}")
    for name, value in (("refit_draws", refit_draws), ("refit_warmup", refit_warmup)):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")

    prepared = model.prepare(y)
    observed = improvement_statistic(
        model,
        target,
        chain,
        prepared,
        method="sample",
        draws=int(refit_draws),
        warmup=int(refit_warmup),
        seed=int(seed),
    )
    rng = np.random.default_rng(seed + 1)
    null = np.empty(int(n_null))
    for i in range(int(n_null)):
        world = model.simulate_null(prepared, rng)
        null[i] = improvement_statistic(
            model,
            target,
            chain,
            model.prepare(world),
            method="sample",
            draws=int(refit_draws),
            warmup=int(refit_warmup),
            seed=int(seed),
        )
    p_value = (1.0 + float(np.sum(null >= observed))) / (1.0 + int(n_null))
    return CertifyResult(observed=float(observed), null=null, p_value=p_value)


def calibrate(
    spec: LatentModel,
    data: TimeSeriesData,
    target: str,
    *,
    n_controls: int = 20,
    seed: int = 0,
) -> CalibrationReport:
    """Is the configured detector honest: measured fire-rate on control worlds.

    Fits the likelihood's one-state null on the observed series, simulates
    ``n_controls`` regime-free worlds from it, runs the full :func:`recover`
    pipeline (the spec's own budgets and options) on each, and reports how
    often any verdict fired, with a Wilson interval on the rate. World ``i``
    runs on a deep copy of the spec with sampler seed ``seed + i``, so the
    worlds are independent and the whole report reproduces from its seed.

    This is the harness that turned the spectral likelihood from a detector
    that fired on 45% of control worlds into one that fired on 0 of 20 (the
    ENBW deflation and the per-frame detrend were the fixes, and this
    measurement is how they were found). Run it once per configuration you
    intend to trust, on a series from the domain you intend to trust it on:
    the control worlds inherit the observed series' own memory and scale, so
    the rate is about your detector on your data, not a generic promise.
    """
    if not isinstance(spec, LatentModel):
        raise TypeError(
            f"spec must be a LatentModel, got {type(spec).__name__}; build one "
            "with latent_model(model=...)"
        )
    y, _ = _validated_series(data, target, spec.model, spec.options.get("chain"), verb="calibrate")
    if int(n_controls) < 1:
        raise ValueError(f"n_controls must be >= 1, got {n_controls}")

    prepared = spec.model.prepare(y)
    worlds: list[CalibrationWorld] = []
    for i in range(int(n_controls)):
        world_seed = int(seed) + i
        world_spec = LatentModel(
            model=copy.deepcopy(spec.model),
            options={**dict(spec.options), "seed": world_seed},
        )
        simulated = spec.model.simulate_null(prepared, np.random.default_rng(world_seed))
        control = timeseries({"t": np.arange(len(simulated)), target: simulated}, time="t")
        result = world_spec.run(control, target)
        fired_blocks = tuple(name for name, hit in result.verdicts().items() if hit)
        worlds.append(
            CalibrationWorld(
                world=i,
                seed=world_seed,
                fired=bool(fired_blocks),
                fired_blocks=fired_blocks,
            )
        )
    fires = sum(w.fired for w in worlds)
    ci_lo, ci_hi = wilson(fires, int(n_controls))
    return CalibrationReport(
        n=int(n_controls),
        fires=fires,
        rate=fires / int(n_controls),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        worlds=tuple(worlds),
    )


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval: honest at 0/n and n/n.

    The textbook Wald interval collapses to a point at the boundaries, and a
    calibration run's most hoped-for outcome is exactly 0 of n, so the
    boundary is the common case rather than the corner. Ported from the
    reference lab's scoring code unchanged.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def consensus(results: Sequence[LatentResult], *, level: float = 0.99) -> Consensus:
    """Several likelihoods' verdicts on one series, block by block.

    Each model speaks its own vocabulary, so the table distinguishes three
    things a flat comparison would blur: a block both models fired, a block
    they disagree on, and a block one model cannot express at all (the
    switching AR's structural amplitude blindness is an absence, not a
    non-fire, and is shown as ``None``). Verdicts are recomputed from each
    result's delta draws at the shared ``level``, familywise-corrected and
    gated exactly as :meth:`LatentResult.verdicts` reports them.

    One cross-model rule is not available: adjudicating a spectral amplitude
    fire against a volatility model's spread verdict (a louder wave against a
    noisier floor) needs a ``vol_garch`` likelihood that ergodic does not have,
    so :meth:`Consensus.amp_vs_spread` raises by name.
    """
    results = list(results)
    if len(results) < 2:
        raise ValueError(
            f"consensus() compares verdicts across models and got {len(results)} "
            "result(s); pass at least two LatentResults for the same series"
        )
    for result in results:
        if not isinstance(result, LatentResult):
            raise TypeError(f"consensus() reads LatentResults, got {type(result).__name__}")
    targets = {result.target for result in results}
    if len(targets) != 1:
        raise ValueError(
            f"consensus() compares models of one series, and these results name "
            f"different targets: {sorted(targets)}"
        )

    models: list[str] = []
    counts: dict[str, int] = {}
    for result in results:
        base = type(result.likelihood).__name__
        counts[base] = counts.get(base, 0) + 1
        models.append(base if counts[base] == 1 else f"{base}#{counts[base]}")

    per_model = [result.verdicts(level) for result in results]
    blocks: list[str] = []
    for verdicts in per_model:
        for name in verdicts:
            if name not in blocks:
                blocks.append(name)
    table = {name: tuple(verdicts.get(name) for verdicts in per_model) for name in blocks}
    return Consensus(
        target=results[0].target, models=tuple(models), verdicts=table, level=float(level)
    )
