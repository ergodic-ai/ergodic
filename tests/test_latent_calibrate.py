# SPDX-License-Identifier: Apache-2.0
"""calibrate(): the detector's fire rate on control worlds, at test budgets.

Three control worlds through the full recover pipeline is the smallest run
that still exercises everything decision 9 specifies: the one-state null fit
on the observed series, per-world simulation, the deep-copied spec with
per-world seeds, the any-verdict fire rule, and the Wilson interval. The
spectral likelihood carries the run because its fit is seconds per world (the
reason it is the flagship); the assertion is zero fires, which is the
measured behavior of the shipped configuration on control worlds (0 of 20 at
the full reference budget) and holds at the reduced budget here because a
smaller sample widens the intervals rather than narrowing them.

The observed series is a plain stationary AR(1)-flavored signal rather than
the golden generator's control arm: calibrate's contract is about worlds
simulated from the fitted null, so any regime-free series is an honest seat
for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("pymc")

from ergodic import calibrate, latent_model, timeseries
from ergodic.latent import CalibrationReport

if TYPE_CHECKING:
    import numpy.typing as npt

    from ergodic.data import TimeSeriesData

    Array = npt.NDArray[np.float64]

N = 480  # 5 frames of 96: the smallest world the spectral chain reads well


def _control_series(n: int = N, seed: int = 0) -> TimeSeriesData:
    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    shocks = rng.normal(0.0, 0.1, size=n)
    for t in range(1, n):
        signal[t] = 0.7 * signal[t - 1] + shocks[t]
    return timeseries({"t": np.arange(n), "signal": signal + 100.0}, time="t")


def test_calibrate_control_worlds_stay_silent() -> None:
    """Zero fires on three control worlds, and the report holds its shape."""
    from ergodic.latent.likelihoods import spectral_whittle

    spec = latent_model(model=spectral_whittle(), draws=250, warmup=250, chains=1)
    report = calibrate(spec, _control_series(), "signal", n_controls=3, seed=0)

    assert isinstance(report, CalibrationReport)
    assert report.n == 3
    assert report.fires == 0
    assert report.rate == 0.0
    assert report.ci_lo == 0.0
    assert 0.0 < report.ci_hi < 1.0
    assert len(report.worlds) == 3
    assert [w.seed for w in report.worlds] == [0, 1, 2]
    assert all(not w.fired and w.fired_blocks == () for w in report.worlds)

    frame = report.to_frame()
    assert list(frame.columns) == ["world", "seed", "fired", "fired_blocks"]
    assert len(frame) == 3
    assert "0 of 3" in report.summary()

    # The caller's spec is untouched: each world ran on a deep copy.
    assert "seed" not in spec.options
