# SPDX-License-Identifier: Apache-2.0
"""certify(): the simulated-null improvement test, on budgets a suite can afford.

The one sampling test runs the full loop on a small level world: 240 ticks
(the dwell is ~240, so the switch flips about once, which is exactly enough
signal for a 40-point improvement), five null worlds, and the short refit at
120 draws after 120 warmup. Measured while choosing the refit method: the
observed statistic on this world sits near +40 while every null world lands
below +1, so the assertion is on the p-value's exact floor, not on a margin
that MCMC noise could erode. The p floor with five nulls is 1/6; a tighter
p needs more worlds and more wall clock than a test should spend.

The pure-numpy null model (OLS fit, closed-form log-likelihood, the rolled
control world) is held to hand computations here as well; those need no
sampler but live in this gated file because they are certify's machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("pymc")

from ergodic import certify, timeseries
from ergodic.latent import CertifyResult
from ergodic.latent.likelihoods import spectral_whittle, switching_ar

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]

# -- a compact level world (the golden generator, level arm only) -------------

DWELL = 240
N = 240


def _level_world(n: int = N, seed: int = 0, shift: float = 1.0) -> Array:
    """An EMA signal chasing a switched level, the golden world's level arm."""
    rng = np.random.default_rng(seed)
    mean = float(DWELL)
    var = (mean / 3.0) ** 2
    r = mean**2 / (var - mean)
    p_nb = mean / var
    states: list[int] = []
    current = 0
    while len(states) < n:
        run = max(1, int(rng.negative_binomial(r, p_nb)) + 1)
        states.extend([current] * run)
        current = 1 - current
    switch = np.array(states[:n], dtype=np.int64)
    target = 100.0 + shift * switch
    shocks = rng.normal(0.0, 1.0, size=n)
    signal = np.zeros(n)
    signal[0] = target[0]
    for t in range(1, n):
        signal[t] = 0.3 * target[t] + 0.7 * signal[t - 1] + shocks[t] * 0.1
    return signal


# -- the full loop, once ------------------------------------------------------


def test_certify_level_world_beats_its_null() -> None:
    """A planted level regime clears every simulated no-switch world."""
    signal = _level_world()
    data = timeseries({"t": np.arange(N), "signal": signal}, time="t")
    result = certify(
        data,
        "signal",
        model=switching_ar(),
        n_null=5,
        seed=0,
        refit_draws=120,
        refit_warmup=120,
    )
    assert isinstance(result, CertifyResult)
    assert result.null.shape == (5,)
    assert result.observed > float(result.null.max())
    # The p-value floor: (1 + 0) / (1 + 5). Measured on this world the
    # observed statistic is ~+40 against nulls below +1.
    assert result.p_value == pytest.approx(1.0 / 6.0)
    assert result.observed > 10.0
    assert "n_null=5" in repr(result)


# -- the AR null model, held to hand computations -----------------------------


def test_ar_null_loglik_matches_the_ols_closed_form() -> None:
    likelihood = switching_ar()
    prepared = likelihood.prepare(_level_world())
    y = prepared.obs["y"]
    base = np.column_stack([np.ones(len(y)), prepared.obs["lags"]])
    beta, *_ = np.linalg.lstsq(base, y, rcond=None)
    var = max(float(np.mean((y - base @ beta) ** 2)), 1e-12)
    by_hand = -0.5 * len(y) * (np.log(2.0 * np.pi * var) + 1.0)
    assert likelihood.null_loglik(prepared) == pytest.approx(by_hand)


def test_ar_null_world_starts_at_the_real_opening_and_reproduces() -> None:
    likelihood = switching_ar()
    prepared = likelihood.prepare(_level_world())
    world = likelihood.simulate_null(prepared, np.random.default_rng(3))
    assert world.shape == (N,)
    assert np.all(np.isfinite(world))
    # The opening ticks are the observed series' own, in standardized space.
    np.testing.assert_allclose(world[:3], prepared.obs["lags"][0][::-1])
    again = likelihood.simulate_null(prepared, np.random.default_rng(3))
    np.testing.assert_allclose(world, again)
    other = likelihood.simulate_null(prepared, np.random.default_rng(4))
    assert not np.allclose(world[3:], other[3:])


# -- the spectral null model --------------------------------------------------


def test_spectral_null_fit_and_synthesis_are_coherent() -> None:
    """The one-state MAP fit describes the world its own synthesis produces.

    A strong planted wave (amplitude 0.8 on a 0.1 noise floor) must come back
    as line power well above the continuum, and the synthesized control world
    must carry a periodogram peak near that fitted power: the amplitude is
    derived from the line by inverting the exact frame pipeline, so a large
    mismatch would mean the inversion lies.
    """
    likelihood = spectral_whittle()
    rng = np.random.default_rng(0)
    n = 1440
    t = np.arange(n)
    signal = 0.8 * np.sin(2.0 * np.pi * 0.05 * t) + rng.normal(0.0, 0.1, n)
    prepared = likelihood.prepare(signal)

    params, loglik = likelihood._null_fit(prepared)
    assert np.isfinite(loglik)
    assert params["f0"] == pytest.approx(0.05, rel=0.1)
    assert params["line"] > 10.0 * params["sigma"] ** 2

    world = likelihood.simulate_null(prepared, np.random.default_rng(1))
    assert world.shape == (prepared.T_eff * likelihood.window,)
    resim = likelihood.prepare(world)
    peak_sim = float(resim.obs["pgram"].mean(axis=0).max())
    peak_obs = float(prepared.obs["pgram"].mean(axis=0).max())
    assert peak_sim == pytest.approx(peak_obs, rel=0.5)


def test_spectral_unit_line_peak_is_near_the_hann_closed_form() -> None:
    """A bin-centered unit sinusoid's peak response is about W/6 under Hann."""
    likelihood = spectral_whittle()
    q = likelihood._unit_line_peak(10.0 / 96.0)
    assert 0.5 * 96.0 / 6.0 < q < 2.0 * 96.0 / 6.0
