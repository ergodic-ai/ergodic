# Certify and calibrate

Test whether two states fit a series better than one, measure how often your
detector fires on series without a switch, and compare two models' verdicts. These are three separate
questions, and `recover` answers none of them.

!!! need "You'll need"
    - ergodic with the latent extra, `pip install "ergodic[latent]"`, plus jax and
      numpyro for the fast sampler.
    - [Recover hidden regimes](recover-hidden-regimes.md), which explains blocks,
      deltas and verdicts.
    - A few minutes: each control world is a fresh model fit.

## Run the program

The series is simulated. While a hidden switch is on, from tick 500 to tick 999, the
level is 0.5 higher.

```python file="certify_regime.py" timeout="400"
import numpy as np
from ergodic import calibrate, certify, consensus, latent_model, timeseries
from ergodic.latent.likelihoods import spectral_whittle, switching_ar

# 1. Simulate 1,440 ticks. While a hidden switch is on (ticks 500 to 999),
#    the level is 0.5 higher.
rng = np.random.default_rng(0)
switch = np.zeros(1440, dtype=int)
switch[500:1000] = 1
signal = 100 + 0.5 * switch + rng.normal(0, 0.5, 1440)
data = timeseries({"tick": np.arange(1440), "signal": signal}, time="tick")

# 2. Fix the detector's configuration once, then recover with it.
spec = latent_model(model=spectral_whittle(), draws=400, warmup=400, chains=2)
print(spec)
spectral = spec.run(data, "signal")
print(spectral.verdicts())
gate = spectral.gates[0]
print(gate.name, gate.blocks, f"{gate.value:.1f}", gate.passed)

# 3. Does a regime exist at all? Compare against 9 simulated no-switch worlds.
test = certify(
    data, "signal", model=spectral_whittle(), n_null=9, refit_draws=100, refit_warmup=100
)
print(test)

# 4. How often does this configuration fire on worlds with no switch?
report = calibrate(spec, data, "signal", n_controls=3)
print(report.summary())

# 5. Ask a second likelihood and compare the verdicts block by block.
ar = latent_model(model=switching_ar(), draws=200, warmup=200, chains=1).run(data, "signal")
table = consensus([spectral, ar])
print(table)
print(table.to_frame().to_string(index=False))
```

```output from="certify_regime.py" numbers="vary"
LatentModel(SpectralWhittle, chains=2, draws=400, warmup=400)
{'level': True, 'spread': False, 'freq': False, 'amp': False}
wave_present ('freq', 'amp') -5.9 False
CertifyResult(observed=28.73, null_q95=5.41, p_value=0.1000, n_null=9)
fired on 0 of 3 control worlds (0%, 95% Wilson [0%, 56%])
Consensus('signal', models=[SpectralWhittle, SwitchingAR], agreed_fires=[level], contested=[none])
 block SpectralWhittle SwitchingAR consensus
 level            True        True     fires
spread           False       False     quiet
  freq           False        None     quiet
   amp           False        None     quiet
rhythm            None       False     quiet
```

It takes one to two minutes, most of it in step 3.

- **Step 2.** `latent_model` holds the configuration as a value, a `LatentModel`, so
  the same settings run on the real series and on every control world. The spectral
  model fires `level`. Its `wave_present` gate failed, because no periodic component
  stands above the noise, so it vetoes `freq` and `amp` whatever their intervals say.
- **Step 3.** `certify` asks whether two states fit better than the model's own
  one-state version, by more than chance. The statistic on the real series is 28.7.
  On the simulated worlds with no switch, 95% of the values sit below 5.4.
- **Step 4.** On three worlds with no switch, the configured detector never fired.
- **Step 5.** Both models fire `level` and agree on everything they can both
  express.

## Certify that two states fit better than one

`certify` fits the likelihood's own one-state model to the series, simulates
`n_null` control worlds from it, and pushes the real series and every world through
the same short fit. The statistic is twice the log-likelihood gain of two states
over one.

The p-value is `(1 + #{null ≥ observed}) / (1 + n_null)`. Here the observed
statistic beat all 9 worlds, and 0.1 is the smallest p-value 9 worlds can give, so
p = 0.1 is all this run can claim. Run the default, `n_null=99`, to claim p ≤ 0.01.
It costs about 100 short fits. Keep `refit_draws` and
`refit_warmup` the same for every comparison you report: the test is honest because
the real series and the control worlds go through exactly the same procedure.

A small p-value says two states fit better than this one-state model. It doesn't
prove a switch: a one-state model that misses something else in the series, such as
a trend, heavy tails or longer memory, can lose to two states too. Look at the
recovered path and deltas before you call it a regime.

`test.null` holds the simulated statistics, and `test.to_frame()` gives the observed
and null values in one table, ready for a histogram.

## Calibrate the detector

A detector that fires on series without a regime is uninformative, however good its
hits look. `calibrate(spec, data, target, n_controls=20)` simulates regime-free
worlds that keep your series' own scale and memory, runs the full `recover` with
your spec on each (world `i` gets seed `seed + i`), and counts how many fired any
verdict.

`report.summary()` gives the rate with a 95% Wilson interval. The interval stays
meaningful at 0 of n: 0 of 3 still allows a true rate up to 56%, which is why the
default is 20 worlds. `report.to_frame()` lists each world and the blocks it fired.

!!! caution "Calibrate the configuration you will use"
    The rate belongs to one configuration on one kind of series. Change the
    likelihood, its settings or the sampler budget, and the rate can change. Rerun
    `calibrate` with the exact spec you'll trust.

## Compare two models

`consensus([result_a, result_b], level=0.99)` recomputes each result's verdicts at a
shared level and lays them side by side. A cell is `True` or `False` when the model
has that block, and `None` when it doesn't: `switching_ar` has no `amp` block, and
`spectral_whittle` has no `rhythm` block. Missing is not the same as quiet.

`table.agreed` maps each block to the verdict when every model that has it agrees.
`table.contested` lists the blocks where they disagree. A contested block is where
the models' views of the mechanism differ, so look at both results' deltas there.

<details markdown="1"><summary>What are the limits?</summary>

- `consensus` needs at least two results for the same target.
- `consensus` can't tell a louder wave from a noisier floor: no built-in model
  pairs an amplitude verdict with a volatility verdict.
- `certify` and `calibrate` validate the series the same way `recover` does, and
  refuse the same inputs.
- Writing your own likelihood means subclassing `RegimeLikelihood`. It declares its
  blocks as `DeltaBlock`s and its vetoes as `Gate`s, inside the `EmissionSpec` it
  returns, and provides `null_loglik` and `simulate_null` for `certify` and
  `calibrate`. A result reports each gate as a `GateOutcome` (name, vetoed blocks,
  statistic, passed).

</details>

## Next steps

<div class="cards" markdown="1">

- [Recover hidden regimes](recover-hidden-regimes.md): the path, the deltas and the chain prior.
- [Assumptions behind each step](assumptions.md): what each part of ergodic takes for granted.
- [Python reference](reference-latent.md): `certify`, `calibrate`, `consensus` and the custom-likelihood types.

</div>
