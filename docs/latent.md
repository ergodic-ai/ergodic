# Hidden regimes

Somewhere in a building there is a switch, and nobody logged it. The only trace it left is one measured series that behaves differently while the switch is on. `ergodic.latent` reads the switch back out of that series, and it is careful to split what sounds like one question into the three questions it actually is:

- `recover` answers **when** the regime was on and **which** named mechanism block changed. It returns a posterior over the hidden on/off path plus one credible interval per block: level, spread, rhythm, amplitude.
- `certify` answers **whether** a regime exists at all, with no vocabulary attached. It compares the switching model against control worlds simulated from a fitted no-switch model, so the null distribution is measured rather than assumed.
- `calibrate` answers whether your configured detector can be trusted: its measured fire rate on worlds that contain no regime.

Keeping the verbs apart is the design. A fitted switching model always finds *a* path, because that is what fitting does, and a delta interval that excludes zero only says which block moved *given* that a regime is there. Reading existence off it would let a model that can imagine a switch anywhere certify its own imagination.

The model underneath is one hidden two-state Markov chain. The discrete path is never sampled; the forward algorithm sums it out exactly, so NUTS only ever sees continuous parameters, and the path comes back afterwards by forward-backward, one posterior draw at a time. What varies between models is the emission, packaged as a `RegimeLikelihood` that declares its own mechanism vocabulary.

PyMC runs the sampling and it is optional: the bare install never imports it. `pip install ergodic[latent]` adds it, and installing `jax` and `numpyro` alongside puts sampling on the fast path, about four times the C backend on these models.

## A world with a sealed switch

Here is the running example, and it runs: the world the pillar was measured on. A hidden switch with a mean dwell of about 240 ticks moves a signal's level by +1.0 while it is on, and an EMA sensor chases that level through noise. The switch is generated and then sealed away; every fit below sees only the signal column.

```python
import numpy as np

def generate(n, seed):
    rng = np.random.default_rng(seed)
    mean, var = 240.0, 80.0**2          # negative-binomial dwells, mean ~240
    r, p = mean**2 / (var - mean), mean / var
    states, current = [], 0
    while len(states) < n:
        run = max(1, int(rng.negative_binomial(r, p)) + 1)
        states.extend([current] * run)
        current = 1 - current
    switch = np.array(states[:n])
    target = 100.0 + 1.0 * switch        # the mechanism: a level shift
    shocks = rng.normal(0.0, 1.0, size=n)
    signal = np.zeros(n)
    signal[0] = target[0]
    for t in range(1, n):
        signal[t] = 0.3 * target[t] + 0.7 * signal[t - 1] + shocks[t] * 0.1
    return signal, switch

signal, switch = generate(1440, seed=4)  # switch is sealed from here on
```

## Recovering the switch

```python
from ergodic import recover, timeseries
from ergodic.latent.likelihoods import spectral_whittle

data = timeseries({"t": np.arange(1440), "signal": signal}, time="t")
result = recover(data, "signal", model=spectral_whittle(),
                 draws=400, warmup=400, chains=2, seed=4)
result
# LatentResult('signal', steps=15, decoded=150/800, verdicts=[level])
```

The result carries the posterior path and the mechanism deltas:

```python
result.gamma()          # (15, 2) posterior state marginals per emission step
result.path()           # the hard path at frame resolution
result.tick_path()      # the same path repeated onto the original tick clock
result.deltas()
# {'level':  (-1.89, -1.75, -1.57),
#  'spread': (-0.07,  0.02,  0.12),
#  'freq':   (-1.33, -0.05,  0.73),
#  'amp':    (-0.97,  0.12,  1.12)}
result.verdicts()
# {'level': True, 'spread': False, 'freq': False, 'amp': False}
```

Exactly one block fires, and it is the planted one. Unsealing the switch to score the run (scoring only, never fitting), the recovered path agrees with the truth on 96% of ticks. The sign of the level delta is arbitrary: "state 1" is whichever regime the aligned path calls state 1, and a two-state model is symmetric in its labels.

That label symmetry is handled by alignment rather than by an ordering constraint. Each decoded draw's trajectory is correlated against an iterated reference, mirror-mode draws are flipped, and their delta draws are sign-corrected. An ordering constraint like `mu_0 < mu_1` would break on exactly the mechanism that did not touch the ordered parameter, and it biases small deltas away from zero, which is poison for a verdict defined as zero-exclusion.

`verdicts(level=0.99)` is deliberately stricter than `deltas()`. Each block's interval runs at the Bonferroni share of the familywise level, so with four blocks the chance that any inert block fires stays at or under one percent, and the likelihood's own gates can veto blocks on top (the spectral wave gate refuses frequency and amplitude verdicts when the fitted line never clears the noise continuum under its own peak).

## Two likelihoods, two vocabularies

`spectral_whittle()` is the flagship. It chops the series into non-overlapping 96-tick Hann frames and models each frame's mean and periodogram: an AR(1) continuum plus one spectral line per regime, with Whittle's exponential approximation per bin. Level lives in the frame mean, spread in the continuum, rhythm in the line's frequency, amplitude in its power. Phase never enters, which is what makes an amplitude change visible here. A fit takes seconds, because the forward pass is fifteen frames rather than 1440 ticks.

`switching_ar(p=3)` is the second implementation, the one that proves the contract generalizes: an AR(p) emission with level, rhythm, and spread deltas, a completely different preparation, and the same engine end to end. It sees a frequency change the classical way, in the lag coefficients. It is blind to amplitude, and the blindness is structural rather than a tuning problem: the AR coefficients that reproduce a sinusoid depend on its frequency alone, so doubling the wave changes nothing the model can see. On the measured worlds the switching AR fired 6 of 6 on level, spread, and rhythm arms and 0 of 2 on amplitude, while the spectral model catches amplitude. Its docstring says so, and "amplitude" is simply absent from its vocabulary, because a verdict slot that can never fire is worse than no slot.

Both likelihoods are frozen dataclasses over the same four-method contract (`prepare`, `emissions`, `emissions_np`, and the one-state null pair below), so a new emission model is a new likelihood and no engine changes.

## Certifying that a regime exists

The verdicts above assumed a regime and asked which block moved. `certify` drops the assumption:

```python
from ergodic import certify

cert = certify(data, "signal", model=spectral_whittle(), n_null=19, seed=0)
cert
# CertifyResult(observed=25.84, null_q95=13.89, p_value=0.0500, n_null=19)
```

The statistic is twice the log-likelihood gain of two states over one: the two-state marginal log-likelihood at a cheap point estimate of the parameters, minus the likelihood's own fitted one-state null (`null_loglik`). What makes the test honest is where the null distribution comes from. Nineteen control worlds are simulated from that fitted one-state model (`simulate_null`), each goes through the identical computation, and the p-value is `(1 + #{null >= observed}) / (1 + n_null)`. With 19 worlds the finest attainable evidence is 0.05, which is what the real world scores here: it beat all nineteen. Reporting a smaller number would claim resolution the simulation did not buy; buy more resolution with more worlds (`n_null=99` is the default and floors p at 0.01).

The measured alternative to simulating the null is BIC, which prices an extra state at a fixed `ln n` per parameter. On these worlds the reference lab found that price wrong in both directions, which is the argument for measuring the null instead of assuming it.

Two implementation notes worth knowing. The cheap refit inside certify is a short single-chain posterior sample with the statistic taken at the posterior-mean parameters, and the observed series goes through exactly the same short fit as every null world, budget and seed included, so the statistic is exchangeable under the null by construction. The other candidate, a `pm.find_MAP` point, was measured and rejected: on the spectral model the Laplace shrinkage pulls the MAP deltas to zero and the improvement came out negative on a genuine level world, at about seven times the wall clock. And the null worlds are honest about what the one-state model does and does not pin down: the spectral null resynthesizes in the time domain from the fitted continuum, line power, and mean, with the sinusoid's phase drawn uniformly, because phase is not a parameter of a Whittle model and a null world must not pretend to know it.

Budget, stated plainly: certify runs one short fit per world, so the default 99 worlds cost minutes on the spectral flagship and more on the tick-resolution AR.

## Calibrating the detector

`calibrate` turns the same control worlds on `recover` itself. Freeze your configuration as a value, then measure how often it fires on nothing:

```python
from ergodic import calibrate, latent_model

spec = latent_model(model=spectral_whittle(), draws=250, warmup=250, chains=1)
report = calibrate(spec, data, "signal", n_controls=3, seed=0)
report
# CalibrationReport(fired on 0 of 3 control worlds (0%, 95% Wilson [0%, 56%]))
report.to_frame()
#  world  seed  fired fired_blocks
#      0     0  False
#      1     1  False
#      2     2  False
```

Each world runs on a deep copy of the spec with seed `seed + i`, fires when any verdict comes back true, and the rate gets a Wilson interval, which stays honest at 0 of n, the outcome you are hoping for, where the textbook interval collapses to a point. Three worlds is a smoke check; twenty is the measured configuration, and at the full budget the shipped spectral model fires on 0 of 20 control worlds at the 99% familywise level.

That number is the pillar's origin story, so it is worth telling straight. An intermediate version of the spectral model fired on 45% of control worlds while looking excellent on effect worlds. Two habits caused it: treating periodogram bins as independent overcounts the evidence (a Hann window smears each frequency over about 1.5 bins), and a frame that still carries the signal's slow drift reads as low-frequency power that a spread delta happily absorbs. The shipped model deflates the Whittle sum by 1/1.5 and removes a per-frame linear trend, and calibration is how both fixes were found. Run it once per configuration you intend to trust, on a series from your own domain: the control worlds inherit the observed series' memory and scale, so the rate is about your detector on your data.

## When two models disagree

Run two likelihoods on the same series and lay their verdicts side by side:

```python
from ergodic import consensus
from ergodic.latent.likelihoods import switching_ar

result_ar = recover(data, "signal", model=switching_ar(), draws=250, warmup=250, chains=1)
consensus([result, result_ar]).to_frame()
#    block  SpectralWhittle  SwitchingAR  consensus
#    level             True         True      fires
#   spread            False        False      quiet
#     freq            False         None      quiet
#      amp            False         None      quiet
#   rhythm             None        False      quiet
```

The table keeps three things apart that a flat comparison would blur: a block both models settle the same way, a block they contest, and a block one model cannot express at all, shown as `None`. The AR model's missing amplitude entry is an absence, and treating it as a non-fire would count structural blindness as evidence.

One cross-model rule is planned and its slot ships explicitly empty. A spectral amplitude fire against a volatility model's spread verdict would separate "the wave got louder" from "the noise floor rose", which no single v1 likelihood can decide alone. That rule needs a GARCH-style volatility likelihood (`vol_garch`), which is deferred, and `Consensus.amp_vs_spread()` raises by name until it lands.

## What to keep in mind

The decoded path moves in 96-tick steps under the spectral model; that resolution is the price of non-overlapping frames, and overlap is not a fix, because a rolling window hands consecutive steps shared data and that manufactured memory reads as regime persistence. At reduced sampling budgets the level and amplitude arms are seed-sensitive (the golden tests pin their seeds and say why); the full reference budget of 4 chains of 1000 after 1000 warmup is the measured configuration. And the whole page ran on a series where the regime is real; on your data, run `certify` before quoting `recover`'s verdicts, and `calibrate` before trusting either.
