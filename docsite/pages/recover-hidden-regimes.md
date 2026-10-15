# Recover hidden regimes

Find when a switch that nobody logged was on in one time series, and which part of
the series' behavior it changed. You'll fit a two-state switching model and read its
path and verdicts.

!!! need "You'll need"
    - ergodic with the latent extra: `pip install "ergodic[latent]"`.
    - Optionally `pip install jax numpyro` for the fast sampler. Without them,
      sampling falls back to PyMC's C backend, which is slower and warns once.
    - One continuous series with no gaps, in time order.

## Run the program

The series is a simulated sensor. A hidden switch is on from tick 500 to tick 999,
and while it is on the noise is three times larger. The level never changes.

```python file="hidden_regimes.py" timeout="240"
import numpy as np
from ergodic import recover, timeseries
from ergodic.latent.likelihoods import spectral_whittle

# 1. Simulate 1,440 ticks of a sensor. A switch nobody logged is on from
#    tick 500 to tick 999, and while it is on the noise is three times larger.
rng = np.random.default_rng(0)
switch = np.zeros(1440, dtype=int)
switch[500:1000] = 1
signal = 100 + rng.normal(0, 0.1, 1440) * np.where(switch == 1, 3.0, 1.0)
data = timeseries({"tick": np.arange(1440), "signal": signal}, time="tick")

# 2. Recover the switch with the spectral likelihood.
result = recover(
    data, "signal", model=spectral_whittle(), draws=400, warmup=400, chains=2, seed=0
)
print(result)
print(result.chain)

# 3. Which mechanism moved: the posterior delta per block, then the verdicts.
for block, (low, mean, high) in result.deltas().items():
    print(f"{block:<7}{mean:+.2f}  [{low:+.2f}, {high:+.2f}]")
print(result.verdicts())

# 4. When the switch was on: one state per 96-tick frame.
print(result.path())
ticks = result.tick_path()
agreement = np.mean(ticks == switch[: ticks.size])
print(f"agreement with the planted switch: {max(agreement, 1 - agreement):.2f}")
```

```output from="hidden_regimes.py" numbers="vary"
LatentResult('signal', steps=15, decoded=150/800, verdicts=[spread])
MarkovChain(stay_alpha=4.0, stay_beta=2.0)
level  +0.09  [-0.02, +0.21]
spread -0.93  [-1.04, -0.83]
freq   -0.00  [-0.57, +0.52]
amp    -0.06  [-1.31, +1.32]
{'level': False, 'spread': True, 'freq': False, 'amp': False}
[1 1 1 1 1 0 0 0 0 0 0 1 1 1 1]
agreement with the planted switch: 0.95
```

It takes about 10 seconds with jax and numpyro installed.

- **Step 2.** The spectral likelihood cuts the series into 15 frames of 96 ticks, so
  the path has 15 steps. It decodes 150 of the 800 posterior draws. The chain prior
  is `Beta(4, 2)` on the probability of staying in a state from one frame to the
  next.
- **Step 3.** Each block is a named mechanism: `level` (the frame mean), `spread`
  (the log noise scale), `freq` (the rhythm of a periodic component) and `amp` (its
  amplitude). Only `spread` has an interval far from zero, and it's the only verdict
  that fires.
- **Step 4.** The path splits the frames into two stretches. It matches the planted
  switch on 95% of the ticks; the frames that straddle ticks 500 and 1000 are the
  ones it can't place exactly.

## Read the labels

The two states have no built-in meaning. The model is symmetric, so "state 1" is
whichever state the label alignment settled on. Here it named the quiet stretch
state 1. That's why the path reads `0` from tick 480 to tick 1055, and why the
spread delta is negative.

A delta is state 1 minus state 0. A spread delta of −0.93 says the noise in state 1
is about e^−0.93 ≈ 0.39 times the noise in state 0. The planted ratio is 1/3, a log
of −1.10. Frames that mix both states pull the estimate toward zero.

`result.deltas()` reports a 94% interval per block, with no correction.
`result.verdicts()` is the decision. It widens each block's interval so that all
blocks together hold a 99% familywise level (99.75% each with four blocks), and a
block fires only when its interval excludes zero. Then the likelihood's gates can
veto blocks. The spectral model's `wave_present` gate vetoes `freq` and `amp` when no
periodic component stands above the noise, as here.

!!! caution "A verdict doesn't prove there's a switch"
    `recover` always fits two states. A delta that excludes zero says which block
    differs, given that there are two. Whether two states fit better than one is a
    separate test: see [Certify and calibrate](certify-and-calibrate.md).

## Choose a likelihood

The likelihood decides how a regime shows up in the data. Two are built in, in
`ergodic.latent.likelihoods`:

- `spectral_whittle(window=96, bin_lo=1, bin_hi=41)` reads each frame's mean and
  spectrum. Its blocks are `level`, `spread`, `freq` and `amp`. It needs at least
  four frames (384 ticks at the default window), and its path moves in whole frames.
  Its calibration was measured at the default settings.
- `switching_ar(p=3)` is an autoregression of order `p` whose intercept, lag
  coefficients and noise scale switch. Its blocks are `level`, `rhythm` and
  `spread`, and its path has one state per tick after the first `p`. It needs at
  least 32 ticks after the lags. A change in amplitude alone doesn't change its
  parameters, so it has no `amp` block. It's also much slower.

## Tell it how long regimes last

`chain=` sets the prior on the hidden chain as a `MarkovChain(stay_alpha,
stay_beta)`: each state's probability of staying put has a `Beta(stay_alpha,
stay_beta)` prior. The result reports the chain it used as `result.chain`.

Leave it as `None` unless you know the dwell time. Each likelihood has a default at
its own step size: `Beta(60, 1)` per tick for `switching_ar` (a mean stay
probability of 60/61 ≈ 0.98), and `Beta(4, 2)` per 96-tick frame for
`spectral_whittle`. A per-tick prior is wrong for the spectral model by a factor of
the frame length. Dwell times are geometric: a chain with an explicit dwell-time
distribution isn't supported.

<details markdown="1"><summary>What does recover refuse, and what do the diagnostics mean?</summary>

- `data` must be a `TimeSeriesData`. A `PanelData` or `HierarchicalTSData` raises
  `TypeError`: select one series first.
- The target must be continuous and finite. Non-finite values raise `ValueError`;
  drop or fill them first.
- Only two states and one series are supported.
- `result.divergences` counts divergent transitions, and the fit warns when they
  exceed 0.5% of the draws. `result.rhat_max` is the worst split R-hat over the
  deltas after label alignment, by chain, and the fit warns when it passes 1.05.
  `result.rhat_max_unaligned` is the same reading on the raw draws: chains that
  settle in mirror-image labelings push it high (1.83 on one well-fitted world whose
  aligned value was 1.01), so judge the fit by `rhat_max`.
- `result.to_frame()` gives the path as a table with the first tick of each step.

</details>

## Next steps

<div class="cards" markdown="1">

- [Certify and calibrate](certify-and-calibrate.md): test whether two states beat one, and how often the detector fires on noise.
- [Forecast with causal drivers](forecast-with-drivers.md): model a series forward instead of backward.
- [Python reference](reference-latent.md): every option of `recover` and `LatentResult`.

</div>
