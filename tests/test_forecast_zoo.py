# SPDX-License-Identifier: Apache-2.0
"""The two dynamics that read a driver as a force: CausalOU and Bass.

World D3 asks the question a level regression cannot answer. A driver steps up
and stays up; where does the series end up, and how long does it take to get
there? The attractor answers the first and ``kappa`` answers the second, and the
test plants both and asks for them back.

World D4 is a launch: forty periods of noisy adoption from a planted diffusion
curve, and the two things a launch plan wants are the size of the market and when
the peak lands.

Nothing under test is mocked. Skipped wholesale when pymc is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("pymc")

from ergodic import forecast, timeseries
from ergodic.forecast import (
    Bass,
    CausalOU,
    ScenarioMechanism,
    StructuralMechanism,
    forecast_node,
    window_graph,
)
from ergodic.forecast._fit import _group_dataset

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData
    from ergodic.forecast import ForecastResult

    Array = npt.NDArray[np.float64]

WORLD_SEED = 20260805


def monthly(columns: dict[str, Array], n: int) -> TimeSeriesData:
    stamps = np.array([np.datetime64("2018-01-01") + np.timedelta64(31 * i, "D") for i in range(n)])
    return timeseries({"month": stamps, **{k: v[:n] for k, v in columns.items()}}, time="month")


def forecast_or_skip(*args: Any, **kwargs: Any) -> ForecastResult:
    """Forecast, or skip the test when this sandbox cannot compile a model."""
    try:
        return forecast(*args, **kwargs)
    except Exception as exc:  # pytensor needs a C compiler and Python headers
        message = str(exc).lower()
        if "compil" in message or "python.h" in message:
            pytest.skip(f"pytensor cannot compile here: {exc}")
        raise


def posterior(result: ForecastResult, node: str, site: str) -> Array:
    assert result.fits is not None
    trace = result.fits[node].trace
    return np.asarray(_group_dataset(trace, "posterior")[site]).reshape(-1)


# -- World D3: the drivers move the attractor ---------------------------------

KAPPA = 0.3
BETA_OU = 1.0
MU0 = 20.0
D3_TRAIN = 80
D3_HORIZON = 12
#: The driver holds a level for ten steps and then jumps.
#:
#: Measured against two smooth alternatives (an AR(1) driver at 0.7 and at 0.9),
#: on two world seeds each, posterior mean beta against the planted 1.0:
#: steps 1.024 and 1.064, AR(0.7) 0.934 and 1.109, AR(0.9) 0.942 and 1.133. The
#: step driver wins because it is the one that separates the two parameters: a
#: jump exercises the transient, whose curvature is kappa, and then a plateau,
#: whose height is beta. A driver that drifts at about the speed the series
#: reverts confounds the two, which is not a numerical problem but the shape of
#: the likelihood.
D3_HOLD = 10


def d3_world(seed: int, n: int = D3_TRAIN + D3_HORIZON) -> tuple[Array, Array]:
    """A capacity index that steps, and a rate that reverts toward where it points."""
    rng = np.random.default_rng(seed)
    plateaus = rng.normal(0.0, 1.0, n // D3_HOLD + 1)
    driver = np.repeat(plateaus, D3_HOLD)[:n]
    driver = (driver - driver[:D3_TRAIN].mean()) / driver[:D3_TRAIN].std()
    # The index is back at its baseline when the scenario starts. That is the
    # experimental condition the reversion test needs rather than a convenience:
    # "step the driver up by two" only says something about reversion when the
    # driver was not already up there, and on a world that happens to end high
    # the series reverts *downward* toward the new attractor instead.
    driver[D3_TRAIN - D3_HOLD : D3_TRAIN] = 0.0
    level = MU0
    levels = np.empty(n, dtype=np.float64)
    for t in range(n):
        attractor = MU0 + BETA_OU * driver[t]
        level = level + KAPPA * (attractor - level) + rng.normal(0.0, 0.2)
        levels[t] = level
    return driver, levels + rng.normal(0.0, 0.15, n)


@pytest.fixture(scope="module")
def d3_truth() -> tuple[Array, Array]:
    return d3_world(WORLD_SEED)


@pytest.fixture(scope="module")
def d3_result(d3_truth: tuple[Array, Array]) -> ForecastResult:
    """The driver stepped two units above its own mean and held there for the horizon."""
    driver, rate = d3_truth
    data = monthly({"z": driver, "y": rate}, D3_TRAIN)
    return forecast_or_skip(
        data,
        D3_HORIZON,
        graph=window_graph([("z", "y", 0)]),
        mechanisms={"y": CausalOU(seasonality=None), "z": ScenarioMechanism()},
        scenario={"z": np.full(D3_HORIZON, 2.0)},
        fit="staged",
        draws=400,
        warmup=300,
        seed=5,
    )


def test_the_attractor_coefficient_is_recovered(
    d3_result: ForecastResult, d3_truth: tuple[Array, Array]
) -> None:
    """Criterion 6: the posterior mean of beta lands within 0.3 of the planted 1.0."""
    from ergodic.forecast import ZScoreScaler, auto_scaler

    assert d3_result.fits is not None
    child_scale = float(d3_result.fits["y"].scaler.median)
    # A scenario node is never fitted, so it has no NodeFit to read a scaler off.
    # The scale its coefficient is written against is the automatic one on its own
    # history, which is a public function of the data rather than a second guess.
    driver, _ = d3_truth
    driver_scaler = auto_scaler(driver[:D3_TRAIN])
    assert isinstance(driver_scaler, ZScoreScaler)
    driver_scale = float(driver_scaler.std)
    beta = float(np.mean(posterior(d3_result, "y", "y_beta_z"))) * child_scale / driver_scale
    assert abs(beta - BETA_OU) <= 0.3, f"attractor beta {beta:.4f}, planted {BETA_OU}"
    # The rest of the attractor comes back too, which is what makes beta readable.
    mu0 = float(np.mean(posterior(d3_result, "y", "y_mu0"))) * child_scale
    assert abs(mu0 - MU0) <= 1.0, f"mu0 {mu0:.4f}, planted {MU0}"
    # kappa is the weakly identified one of the three, and the band says so
    # honestly rather than pretending otherwise: measured over four world seeds it
    # came back at 0.460, 0.352, 0.246 and 0.363 against a planted 0.3. Each jump
    # in the driver buys only a few informative steps of transient, and the level
    # noise is a fifth of a typical jump, so the reversion rate is estimated from
    # much less than the eighty steps suggest. beta and mu0, which the plateaus
    # measure directly, come back sharply.
    kappa = float(np.mean(posterior(d3_result, "y", "y_kappa")))
    assert 0.15 <= kappa <= 0.65, f"kappa {kappa:.4f}, planted {KAPPA}"


def test_the_series_reverts_toward_the_shifted_attractor(d3_result: ForecastResult) -> None:
    """Criterion 7: hold the driver two units up and the series walks to ``mu0 + 2 beta``."""
    median = d3_result.median("y")
    target = MU0 + 2.0 * BETA_OU
    assert median[-1] > median[0], (
        f"the median at h=12 ({median[-1]:.3f}) should be above h=1 ({median[0]:.3f}): "
        "a shifted attractor is approached, not jumped to"
    )
    assert abs(median[-1] - target) < abs(median[0] - target), (
        "the twelfth step should be closer to the new attractor than the first"
    )
    lo, hi = d3_result.interval("y", 0.9)
    assert lo[-1] <= target <= hi[-1], (
        f"the shifted attractor {target} sits outside the h=12 band [{lo[-1]:.3f}, {hi[-1]:.3f}]"
    )
    # A driver held flat is not a driver that stops mattering: the level keeps
    # moving until it arrives, which is the whole difference from a level
    # regression, where the step would land in one period.
    assert median[0] < target - 0.5


def test_causal_ou_refuses_what_it_cannot_do() -> None:
    """The constructor's own guards."""
    with pytest.raises(ValueError, match="seasonality must be one of"):
        CausalOU(seasonality="weekly")
    with pytest.raises(ValueError, match="period must be at least 2"):
        CausalOU(period=1)
    with pytest.raises(ValueError, match="reg_scale must be positive"):
        CausalOU(reg_scale=0.0)
    with pytest.raises(ValueError, match="monthly seasonality is twelve factors"):
        CausalOU(seasonality="monthly", period=4)


# -- World D4: the S-curve ----------------------------------------------------

BASS_P = 0.03
BASS_Q = 0.4
BASS_M = 1000.0
D4_TRAIN = 40
D4_HORIZON = 6


def d4_world(seed: int, n: int = D4_TRAIN, sigma: float = 6.0) -> Array:
    """Forty periods of noisy adoption from the planted diffusion curve."""
    rng = np.random.default_rng(seed)
    cumulative = 0.0
    incremental = np.empty(n, dtype=np.float64)
    for t in range(n):
        increment = (BASS_P + BASS_Q * cumulative) * (1.0 - cumulative)
        cumulative += increment
        incremental[t] = BASS_M * increment
    return np.maximum(incremental + rng.normal(0.0, sigma, n), 0.0)


@pytest.fixture(scope="module")
def d4_result() -> ForecastResult:
    data = monthly({"units": d4_world(WORLD_SEED)}, D4_TRAIN)
    return forecast_or_skip(
        data,
        D4_HORIZON,
        mechanisms={"units": Bass()},
        draws=500,
        warmup=400,
        seed=3,
    )


def test_the_market_size_is_recovered(d4_result: ForecastResult) -> None:
    """Criterion 8: the posterior median of M lands within 25% of the planted 1000."""
    assert d4_result.fits is not None
    scale = float(d4_result.fits["units"].scaler.median)
    market = float(np.median(posterior(d4_result, "units", "units_M"))) * scale
    assert abs(market - BASS_M) <= 0.25 * BASS_M, f"M posterior median {market:.1f}"
    # The other two parameters come back as well, which is what makes M readable
    # rather than a number the prior supplied.
    assert float(np.median(posterior(d4_result, "units", "units_p"))) == pytest.approx(
        BASS_P, abs=0.02
    )
    assert float(np.median(posterior(d4_result, "units", "units_q"))) == pytest.approx(
        BASS_Q, abs=0.15
    )


def test_the_adoption_peak_lands_where_it_was_planted(d4_result: ForecastResult) -> None:
    """Criterion 9: the peak of the modeled path is within three periods of ``ln(q/p)/(p+q)``."""
    planted = float(np.log(BASS_Q / BASS_P) / (BASS_P + BASS_Q))
    assert d4_result.fits is not None
    view = forecast_node(d4_result.fits["units"], D4_HORIZON, seed=21)
    path = np.median(np.asarray(view.components["units_mu"]), axis=0)
    peak = int(np.argmax(path))
    assert abs(peak - planted) <= 3.0, f"the modeled peak is at step {peak}, planted {planted:.2f}"

    # The same reading through the closed form the mechanism exposes.
    p_hat = float(np.median(posterior(d4_result, "units", "units_p")))
    q_hat = float(np.median(posterior(d4_result, "units", "units_q")))
    assert abs(Bass().peak_step(p_hat, q_hat) - planted) <= 3.0


def test_the_bass_forecast_is_the_curve_running_out(d4_result: ForecastResult) -> None:
    """Forty periods in, the diffusion is over, so the horizon is the tail and not a trend."""
    median = d4_result.median("units")
    assert float(np.max(np.abs(median))) < 0.05 * BASS_M
    lo, hi = d4_result.interval("units", 0.9)
    assert float(np.mean(hi - lo)) > 0.0


def test_bass_refuses_what_it_cannot_do() -> None:
    """A driver, a launch past the window, and malformed priors."""
    data = monthly(
        {"units": d4_world(WORLD_SEED), "pmi": np.linspace(90.0, 110.0, D4_TRAIN)}, D4_TRAIN
    )
    with pytest.raises(NotImplementedError, match="pure diffusion"):
        forecast(
            data,
            3,
            graph=window_graph([("pmi", "units", 1)]),
            mechanisms={"units": Bass(), "pmi": StructuralMechanism()},
            fit="staged",
            draws=10,
            warmup=10,
        )
    with pytest.raises(ValueError, match="launch_index must be"):
        Bass(launch_index=-1)
    with pytest.raises(ValueError, match="q_prior must be two positive"):
        Bass(q_prior=(2.0, -1.0))
    with pytest.raises(ValueError, match="m_prior must be"):
        Bass(m_prior=(1.0, 0.0))


def test_the_bass_scaler_anchors_on_the_mean() -> None:
    """The median of an S-curve is in its tail, which is why the mean is the anchor."""
    from ergodic.forecast import RatioScaler

    adoption = d4_world(WORLD_SEED)
    scaler = Bass().scaler_for(adoption)
    assert isinstance(scaler, RatioScaler)
    assert float(scaler.median) == pytest.approx(float(np.mean(adoption[adoption > 0])), rel=1e-9)
    assert float(scaler.floor) == 0.0
    scaled = scaler.transform(adoption)
    assert float(np.mean(scaled[scaled > 0])) == pytest.approx(1.0, rel=1e-9)
    assert np.allclose(scaler.inverse(scaled), adoption)

    # The size of the difference, which is the reason for the override. The market
    # size on this scale is the sum of the scaled series, so it is tens; on the
    # automatic median scale it is hundreds here and thousands on a curve whose
    # tail is not blurred by noise, where a prior written for one would mean
    # something else entirely for the other.
    by_median = RatioScaler().fit(adoption)
    assert float(np.sum(scaled)) < float(np.sum(by_median.transform(adoption)))
    clean = np.array(
        [BASS_M * increment for increment in _bass_increments(BASS_P, BASS_Q, D4_TRAIN)]
    )
    assert float(np.sum(Bass().scaler_for(clean).transform(clean))) < 2.0 * D4_TRAIN
    assert float(np.sum(RatioScaler().fit(clean).transform(clean))) > 20.0 * D4_TRAIN


def _bass_increments(p: float, q: float, n: int) -> list[float]:
    """The noiseless incremental-adoption fractions, for the scaling contrast above."""
    cumulative = 0.0
    out: list[float] = []
    for _ in range(n):
        increment = (p + q * cumulative) * (1.0 - cumulative)
        cumulative += increment
        out.append(increment)
    return out


def test_a_launch_index_holds_adoption_at_zero_until_it_fires() -> None:
    """A run of pre-launch zeros is modeled rather than fitted around."""
    adoption = np.concatenate([np.zeros(6), d4_world(WORLD_SEED, 34)])
    data = monthly({"units": adoption}, 40)
    result = forecast_or_skip(
        data,
        4,
        mechanisms={"units": Bass(launch_index=6)},
        draws=250,
        warmup=250,
        seed=8,
    )
    assert result.fits is not None
    view = forecast_node(result.fits["units"], 4, seed=9)
    path = np.median(np.asarray(view.components["units_mu"]), axis=0)
    assert np.allclose(path[:6], 0.0)
    assert float(np.max(path)) > 0.0
    with pytest.raises(ValueError, match="at or past the end"):
        forecast(data, 4, mechanisms={"units": Bass(launch_index=40)}, draws=10, warmup=10)
