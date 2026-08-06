# SPDX-License-Identifier: Apache-2.0
"""The forecast pillar's mechanisms and fitter, against planted truths.

Every test here runs a real NUTS fit on a seeded generative world whose
parameters are known, and asks whether the posterior found them. Nothing under
test is mocked. The worlds are written at the scale a monthly business series
actually lives at (a level near 200, a season of plus or minus 15 percent,
observation noise of about 2 percent), because a model that recovers a planted
truth on standardized toy data has not been shown to survive real units.

Skipped wholesale when pymc is absent, the ``test_scm_pymc.py`` precedent, so the
bare install never needs it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("pymc")

from ergodic import timeseries
from ergodic.forecast import (
    NodeDiagnostics,
    SeasonalNaive,
    StructuralMechanism,
    fit_node,
    forecast_node,
    mape,
)
from ergodic.forecast._fit import _audit_sites, _group_dataset, _warn_on_breach

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.forecast import NodeFit

    Array = npt.NDArray[np.float64]

# -- the sampler budget: small enough to run in a test, big enough to mean something
WARMUP = 400
DRAWS = 600
SMALL_WARMUP = 300
SMALL_DRAWS = 400

WORLD_SEED = 20260805


# -- the planted worlds -------------------------------------------------------


def planted_factors(amplitude: float = 0.15) -> Array:
    """The twelve monthly factors, a clean seasonal shape around one."""
    return np.asarray(1.0 + amplitude * np.sin(2.0 * np.pi * np.arange(12) / 12.0))


def world(
    seed: int,
    n: int,
    *,
    phi: float = 0.85,
    slope0: float = 0.6,
    level0: float = 200.0,
    sigma_level: float = 0.4,
    sigma_slope: float = 0.15,
    sigma_obs: float = 4.0,
    amplitude: float = 0.15,
) -> tuple[Array, Array, Array]:
    """A damped-trend seasonal series in original units: ``(y, mu, levels)``.

    A level near 200 growing about 0.3 percent a month, a slope that decays at
    ``phi`` and gets nudged, a seasonal factor between 0.85 and 1.15, and 2
    percent observation noise. The slope stands well clear of the level
    innovations on purpose: when it does not, the trend state stops being
    identifiable and every forecast band is too narrow for reasons that have
    nothing to do with the code.
    """
    rng = np.random.default_rng(seed)
    factors = planted_factors(amplitude)
    level, slope = level0, slope0
    levels = np.empty(n, dtype=np.float64)
    for t in range(n):
        levels[t] = level
        level = level + slope + sigma_level * rng.normal()
        slope = phi * slope + sigma_slope * rng.normal()
    mu = levels * factors[np.arange(n) % 12]
    y = mu + rng.normal(0.0, sigma_obs, n)
    return y, mu, levels


def fit_or_skip(*args: Any, **kwargs: Any) -> NodeFit:
    """Fit, or skip the test when this sandbox cannot compile a model."""
    try:
        return fit_node(*args, **kwargs)
    except Exception as exc:  # pytensor needs a C compiler and Python headers
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise


# -- World A: recovery --------------------------------------------------------

A_TOTAL = 96
A_HOLDOUT = 12
A_TRAIN = A_TOTAL - A_HOLDOUT


@pytest.fixture(scope="module")
def world_a() -> tuple[Array, Array, Array]:
    return world(WORLD_SEED, A_TOTAL)


@pytest.fixture(scope="module")
def fit_a(world_a: tuple[Array, Array, Array]) -> NodeFit:
    """The World A fit: 84 monthly steps, warmup 400, draws 600, seed 7."""
    y, _, _ = world_a
    return fit_or_skip(
        "sales",
        y[:A_TRAIN],
        StructuralMechanism(period=12),
        draws=DRAWS,
        warmup=WARMUP,
        chains=1,
        seed=7,
    )


@pytest.fixture(scope="module")
def forecast_a(fit_a: NodeFit) -> Any:
    return forecast_node(fit_a, A_HOLDOUT, seed=1007)


def test_damping_is_recovered(fit_a: NodeFit) -> None:
    """Criterion 1: the posterior median of phi lands in [0.6, 0.98]."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    phi = np.asarray(posterior["sales_phi"]).reshape(-1)
    median = float(np.median(phi))
    assert 0.6 <= median <= 0.98, f"phi median {median:.3f} outside [0.6, 0.98]"


def test_seasonal_shape_is_recovered(fit_a: NodeFit) -> None:
    """Criterion 2: the twelve posterior factor means correlate with the truth above 0.9."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    raw = np.asarray(posterior["sales_season_raw"]).reshape(-1, 12)
    # the same 12 * softmax the mechanism applies, so this reads the factors
    # themselves rather than the unconstrained parameters behind them
    factors = 12.0 * np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
    correlation = float(np.corrcoef(factors.mean(axis=0), planted_factors())[0, 1])
    assert correlation > 0.9, f"seasonal correlation {correlation:.3f} below 0.9"


def test_the_band_covers_the_noiseless_continuation(
    world_a: tuple[Array, Array, Array], forecast_a: Any
) -> None:
    """Criterion 3: at least 10 of 12 noiseless steps sit inside the 90% band."""
    _, mu, _ = world_a
    lo, hi = np.percentile(forecast_a.draws, [5.0, 95.0], axis=0)
    truth = mu[A_TRAIN:]
    inside = int(((truth >= lo) & (truth <= hi)).sum())
    assert inside >= 10, f"only {inside}/12 steps of the true mean path are inside the band"


def test_the_forecast_beats_seasonal_naive(
    world_a: tuple[Array, Array, Array], forecast_a: Any
) -> None:
    """Criterion 4: median MAPE over the holdout beats SeasonalNaive(12) on the same split."""
    y, _, _ = world_a
    actual = y[A_TRAIN:]
    train = timeseries(
        {"t": np.arange(A_TRAIN, dtype=float), "sales": y[:A_TRAIN]},
        time="t",
    )
    naive = SeasonalNaive(12).fit(train, seed=0).forecast(A_HOLDOUT, seed=0)
    ours = float(mape(actual, forecast_a.draws))
    theirs = float(mape(actual, naive.draws("sales")))
    assert ours < theirs, f"model MAPE {ours:.2f}% did not beat seasonal naive {theirs:.2f}%"


# -- World B: calibration -----------------------------------------------------

B_TOTAL = 60
B_HOLDOUT = 6
B_TRAIN = B_TOTAL - B_HOLDOUT
B_WORLDS = 20


@pytest.fixture(scope="module")
def calibration() -> tuple[int, int]:
    """Fit twenty seeded worlds and count how many held-out points the band covers.

    This is the one criterion that needs many worlds rather than one: a single
    world's coverage is a coin flip, and the question is whether the 90 percent
    band is a 90 percent band. Twenty small fits is what buys the answer.
    """
    covered = 0
    total = 0
    for index in range(B_WORLDS):
        y, _, _ = world(WORLD_SEED + 100 + index, B_TOTAL)
        fitted = fit_or_skip(
            "sales",
            y[:B_TRAIN],
            StructuralMechanism(period=12),
            draws=SMALL_DRAWS,
            warmup=SMALL_WARMUP,
            chains=1,
            seed=1000 + index,
        )
        drawn = forecast_node(fitted, B_HOLDOUT, seed=2000 + index).draws
        lo, hi = np.percentile(drawn, [5.0, 95.0], axis=0)
        actual = y[B_TRAIN:]
        covered += int(((actual >= lo) & (actual <= hi)).sum())
        total += int(actual.size)
    return covered, total


def test_the_ninety_percent_band_is_a_ninety_percent_band(calibration: tuple[int, int]) -> None:
    """Criterion 6: pooled coverage over 120 held-out points sits in [0.80, 0.97]."""
    covered, total = calibration
    share = covered / total
    assert total == B_WORLDS * B_HOLDOUT
    assert 0.80 <= share <= 0.97, f"pooled 90% coverage {share:.3f} outside [0.80, 0.97]"


# -- World C: robustness ------------------------------------------------------


#: Where the spikes go: right at the forecast origin, and measured rather than
#: guessed. A state-space smoother absorbs a mid-window outlier into sigma_obs
#: and the forecast never feels it (spikes at 20, 41 and 63 move the Normal
#: fit's sigma_obs from 4.16 to 7.43 and its holdout MAE not at all), so
#: mid-window contamination cannot exercise robustness at all. The margin grows
#: the closer the shock lands to the origin, because that is where it bends the
#: level and slope that get extrapolated: mid-window is no contest, steps 70, 76
#: and 82 give plain 7.77 against robust 6.07, and these give plain 14.77
#: against robust 5.63. The last of those is the only placement that holds on
#: every seed tried, so it is the one pinned.
OUTLIER_POSITIONS = (79, 81, 83)


@pytest.fixture(scope="module")
def world_c(world_a: tuple[Array, Array, Array]) -> Array:
    """World A with three eight-sigma spikes planted at the forecast origin."""
    y, _, _ = world_a
    contaminated = y.copy()
    for position in OUTLIER_POSITIONS:
        contaminated[position] += 8.0 * 4.0
    return contaminated


def test_student_t_survives_outliers_better(
    world_a: tuple[Array, Array, Array], world_c: Array
) -> None:
    """Criterion 7: robust=True has lower holdout MAE and a smaller sigma_obs."""
    y, _, _ = world_a
    actual = y[A_TRAIN:]
    results = {}
    for robust in (False, True):
        fitted = fit_or_skip(
            "sales",
            world_c[:A_TRAIN],
            StructuralMechanism(period=12, robust=robust),
            draws=DRAWS,
            warmup=WARMUP,
            chains=1,
            seed=11,
        )
        drawn = forecast_node(fitted, A_HOLDOUT, seed=1011).draws
        posterior = _group_dataset(fitted.trace, "posterior")
        results[robust] = (
            float(np.mean(np.abs(np.median(drawn, axis=0) - actual))),
            float(np.median(np.asarray(posterior["sales_sigma_obs"]))),
        )
    robust_mae, robust_sigma = results[True]
    plain_mae, plain_sigma = results[False]
    assert robust_mae <= plain_mae, f"robust MAE {robust_mae:.2f} above plain {plain_mae:.2f}"
    assert robust_sigma < plain_sigma, (
        f"robust sigma_obs {robust_sigma:.4f} not below plain {plain_sigma:.4f}"
    )


# -- the parent slot ----------------------------------------------------------


def test_a_parent_coefficient_is_recovered() -> None:
    """Criterion 8: with one driver and its future given, beta comes back near 0.7."""
    n_total, horizon = 132, 12
    train = n_total - horizon
    beta = 0.7
    rng = np.random.default_rng(WORLD_SEED + 7)
    driver = np.empty(n_total, dtype=np.float64)
    driver[0] = rng.normal(0.0, 0.2)
    for t in range(1, n_total):
        driver[t] = 0.8 * driver[t - 1] + rng.normal(0.0, 0.12)

    _, _, levels = world(WORLD_SEED + 8, n_total)
    factors = planted_factors()
    # the driver acts inside the seasonal factor, exactly as the mechanism composes it
    signal = (levels / 200.0 + beta * driver) * factors[np.arange(n_total) % 12]
    y = 200.0 * signal + rng.normal(0.0, 4.0, n_total)

    fitted = fit_or_skip(
        "sales",
        y[:train],
        StructuralMechanism(period=12),
        parents={"pmi": driver[:train]},
        draws=DRAWS,
        warmup=WARMUP,
        chains=1,
        seed=13,
    )
    posterior = _group_dataset(fitted.trace, "posterior")
    found = float(np.mean(np.asarray(posterior["sales_beta_pmi"])))
    assert abs(found - beta) < 0.2, f"beta posterior mean {found:.3f} is far from {beta}"

    # the slot survives the handoff: the horizon needs the driver's future
    continued = forecast_node(fitted, horizon, parents={"pmi": driver}, seed=1013)
    assert continued.draws.shape == (DRAWS, horizon)
    assert "sales_reg_pmi" in continued.components
    with pytest.raises(ValueError, match="expected"):
        forecast_node(fitted, horizon, parents={"pmi": driver[:train]}, seed=1013)


# -- the mask -----------------------------------------------------------------


def test_masked_steps_never_enter_the_likelihood() -> None:
    """Criterion 10: a masked step's recorded value cannot touch the posterior.

    Two fits of one world. In the first the masked steps are NaN, in the second
    they hold 1e6, and both are handed the same mask. The posteriors have to come
    out bit for bit identical: if a masked value reached the likelihood or the
    scaler, a million would be impossible to miss.
    """
    y, _, _ = world(WORLD_SEED + 3, 60)
    rng = np.random.default_rng(5)
    mask = rng.random(y.size) > 0.10
    mask[0] = mask[-1] = True

    with_nan = np.where(mask, y, np.nan)
    with_sentinel = np.where(mask, y, 1e6)
    traces = []
    scalers = []
    for values in (with_nan, with_sentinel):
        fitted = fit_or_skip(
            "sales",
            values,
            StructuralMechanism(period=12),
            mask=mask,
            draws=SMALL_DRAWS,
            warmup=SMALL_WARMUP,
            chains=1,
            seed=17,
        )
        traces.append(_group_dataset(fitted.trace, "posterior"))
        scalers.append(fitted.scaler)

    assert int(mask.sum()) < mask.size, "the world should have masked steps to test"
    assert scalers[0] == scalers[1], "a masked sentinel moved the scale"
    for name in ("sales_phi", "sales_sigma_obs", "sales_level0", "sales_z_level"):
        left = np.asarray(traces[0][name])
        right = np.asarray(traces[1][name])
        assert np.array_equal(left, right), f"{name} differs between the NaN and sentinel fits"


# -- the site audit -----------------------------------------------------------


def _forecast_model(fit: NodeFit, horizon: int = 6) -> Any:
    """Rebuild the forecasting model the ppc pass would use."""
    import pymc as pm

    with pm.Model() as model:
        fit.mechanism.build(
            fit.node,
            obs=fit.obs_scaled,
            parents={},
            T=fit.n_obs,
            horizon=horizon,
            mask=fit.mask,
        )
    return model


def test_the_audit_passes_on_a_matching_pair(fit_a: NodeFit) -> None:
    """The audit is silent when every replayed name is present and every fresh one is not."""
    _audit_sites(_forecast_model(fit_a), fit_a.trace)


def test_a_missing_replayed_site_raises(fit_a: NodeFit) -> None:
    """Criterion 11: drop one replayed site and the audit refuses the ppc."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    broken = {"posterior": posterior.drop_vars("sales_z_level")}
    with pytest.raises(ValueError, match="not in the trace"):
        _audit_sites(_forecast_model(fit_a), broken)


def test_the_ppc_path_is_unreachable_without_the_audit(fit_a: NodeFit) -> None:
    """The audit is not advisory: forecast_node raises before it samples anything."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    broken = replace(fit_a, trace={"posterior": posterior.drop_vars("sales_sigma_obs")})
    with pytest.raises(ValueError, match="not in the trace"):
        forecast_node(broken, 6, seed=1)


def test_a_forecast_site_in_the_trace_raises(fit_a: NodeFit) -> None:
    """A replayed forecast innovation would reuse the training window; the audit says so."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    smuggled = posterior.assign({"sales_z_level_fc": posterior["sales_phi"]})
    with pytest.raises(ValueError, match="would be replayed instead of simulated"):
        _audit_sites(_forecast_model(fit_a), {"posterior": smuggled})


def test_a_stored_deterministic_raises(fit_a: NodeFit) -> None:
    """A fit-time deterministic in the trace is read back rather than recomputed."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    smuggled = posterior.assign({"sales_mu": posterior["sales_phi"]})
    with pytest.raises(ValueError, match="deterministic"):
        _audit_sites(_forecast_model(fit_a), {"posterior": smuggled})


def test_the_fit_records_no_deterministics(fit_a: NodeFit) -> None:
    """The fitting model stores its free variables and nothing else, which is what makes it safe."""
    posterior = _group_dataset(fit_a.trace, "posterior")
    stored = set(map(str, posterior.data_vars))
    assert "sales_mu" not in stored
    assert "sales_season_factor" not in stored
    assert {"sales_phi", "sales_z_level", "sales_season_raw"} <= stored


def test_the_forecast_model_registers_the_named_sites(fit_a: NodeFit) -> None:
    """The naming contract, checked as names: later sprints match on these exactly."""
    model = _forecast_model(fit_a)
    free = {str(rv.name) for rv in model.free_RVs}
    deterministics = {str(det.name) for det in model.deterministics}
    assert {
        "sales_phi",
        "sales_sigma_level",
        "sales_sigma_slope",
        "sales_sigma_obs",
        "sales_level0",
        "sales_slope0",
        "sales_z_level",
        "sales_z_slope",
        "sales_season_raw",
        "sales_z_level_fc",
        "sales_z_slope_fc",
        "sales_y_fc",
    } <= free
    assert {
        "sales_mu",
        "sales_trend",
        "sales_season_factor",
        "sales_reg",
        "sales_season",
    } <= deterministics


# -- diagnostics --------------------------------------------------------------


def test_diagnostics_are_carried_on_every_fit(fit_a: NodeFit) -> None:
    """Criterion 12: every field is checked against what the trace actually holds.

    Each assertion here is one the code could fail. A diagnostic that reported a
    plausible number rather than the sampler's own would pass a test that only
    asked for a non-negative count, so nothing below asks for that.
    """
    diagnostics = fit_a.diagnostics
    posterior = _group_dataset(fit_a.trace, "posterior")
    stats = _group_dataset(fit_a.trace, "sample_stats")

    assert isinstance(diagnostics, NodeDiagnostics)
    assert diagnostics.node == "sales"

    # the draw count is the trace's own chain x draw product, not the argument
    assert diagnostics.n_draws == int(np.prod(np.asarray(posterior["sales_phi"]).shape[:2]))
    assert diagnostics.n_draws == DRAWS

    # the divergence count is the sampler's, to the unit
    assert diagnostics.divergences == int(np.asarray(stats["diverging"]).sum())
    assert diagnostics.divergence_share == diagnostics.divergences / diagnostics.n_draws

    # min_ess is a real number and it is the minimum over the scalar sites, so a
    # site that mixed badly cannot be hidden by one that mixed well
    assert np.isfinite(diagnostics.min_ess)
    assert diagnostics.min_ess == min(diagnostics.ess_by_site.values())
    scalars = {str(name) for name in posterior.data_vars if posterior[name].ndim == 2}
    assert set(diagnostics.ess_by_site) == scalars
    assert "sales_phi" in scalars
    # the innovation vectors are not scalars and must not be in there: they are
    # T-long and would swamp the reading
    assert "sales_z_level" not in diagnostics.ess_by_site
    assert "sales_season_raw" not in diagnostics.ess_by_site

    assert diagnostics.max_rhat is None, "one chain cannot have an r-hat"


def test_two_chains_give_an_rhat_and_stack_their_draws() -> None:
    """Criterion 12, the other branch: r-hat exists once there is more than one chain."""
    y, _, _ = world(WORLD_SEED + 21, 48)
    fitted = fit_or_skip(
        "sales",
        y,
        StructuralMechanism(period=12),
        draws=200,
        warmup=SMALL_WARMUP,
        chains=2,
        seed=23,
    )
    diagnostics = fitted.diagnostics
    assert diagnostics.n_draws == 400, "two chains of 200 draws stack into one draw axis"
    assert isinstance(diagnostics.max_rhat, float)
    assert np.isfinite(diagnostics.max_rhat)
    assert 0.9 < diagnostics.max_rhat < 1.5, f"r-hat {diagnostics.max_rhat} is not a sane value"
    # and the forecast reads the stacked axis, not one chain of it
    assert forecast_node(fitted, 4, seed=1023).draws.shape == (400, 4)


def test_the_warning_fires_on_a_breach() -> None:
    """Criterion 12: a breach warns rather than raising, in both directions."""
    divergent = NodeDiagnostics(node="sales", divergences=50, n_draws=600, min_ess=400.0)
    with pytest.warns(UserWarning, match="diverged"):
        _warn_on_breach(divergent)
    starved = NodeDiagnostics(node="sales", divergences=0, n_draws=600, min_ess=12.0)
    with pytest.warns(UserWarning, match="bulk ESS"):
        _warn_on_breach(starved)
    assert not divergent.healthy
    assert not starved.healthy
    healthy = NodeDiagnostics(node="sales", divergences=1, n_draws=600, min_ess=400.0)
    assert healthy.healthy
