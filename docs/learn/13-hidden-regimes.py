# %% [markdown]
# # Hidden regimes: a switch nobody logged
#
# A building's power meter reports one reading a minute. Somewhere in the plant room a
# chiller flips between two modes every few hours, and nobody logs when. The facilities
# team has three questions. When was the other mode on? What does it change about the
# reading? And the one people skip: is there a second mode at all, or are we seeing
# patterns in noise?
#
# Those are three different questions, and `ergodic.latent` gives each its own function.
# `recover` answers *when* and *which*: the posterior path of the hidden switch, and an
# interval per named mechanism (level, spread, rhythm, amplitude). `certify` answers
# *whether*: does a two-regime story beat a world that never had a switch? `calibrate`
# answers *would it have fired anyway*: how often does this detector claim a regime on
# worlds that have none?
#
# Keeping them apart matters. A fitted switching model always finds *some* path through
# the data it was fitted to, so "which mechanism moved" can never double as "a regime
# exists".
#
# As in every guide, the answer goes into the simulation first. This time it's sealed:
# nothing after the next cell reads the switch until the lines marked as grading.

# %%
import logging

import numpy as np
import pandas as pd

from ergodic import timeseries

# PyMC logs every sampler run; this keeps the notebook's output readable.
for name in ("pymc", "pymc.sampling", "pymc.stats.convergence"):
    logging.getLogger(name).setLevel(logging.ERROR)

N = 1440  # one day of one-minute readings


def simulate(level_shift: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A meter that chases a target level, and a hidden switch that moves the target."""
    rng = np.random.default_rng(seed)
    # The switch dwells for about 240 minutes at a time (a negative binomial dwell).
    mean, var = 240.0, (240.0 / 3.0) ** 2
    r, p = mean**2 / (var - mean), mean / var
    states: list[int] = []
    current = 0
    while len(states) < n:
        states.extend([current] * max(1, int(rng.negative_binomial(r, p)) + 1))
        current = 1 - current
    switch = np.array(states[:n])

    target = 100.0 + level_shift * switch
    shocks = rng.normal(0.0, 1.0, size=n)
    reading = np.empty(n)
    reading[0] = target[0]
    for t in range(1, n):
        reading[t] = 0.3 * target[t] + 0.7 * reading[t - 1] + 0.1 * shocks[t]
    return reading, switch


# The answer key: in this world the second mode raises the level by 1.0.
reading, switch = simulate(level_shift=1.0, n=N, seed=4)
# A control world built the same way with no switch effect at all.
control, _ = simulate(level_shift=0.0, n=N, seed=0)


def as_series(values: np.ndarray):
    return timeseries(
        pd.DataFrame({"minute": np.arange(len(values)), "power": values}), time="minute"
    )


# %% [markdown]
# Here's the trace the analysis sees. The shading is the sealed switch, drawn only so you
# can appreciate the problem. A shift of 1.0 is the same size as the minute-to-minute
# wobble, so you'd struggle to mark the switch by eye without the shading.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(reading, lw=0.6, color="#33415c")
low, high = ax.get_ylim()
ax.fill_between(np.arange(N), low, high, where=switch == 1, color="#9aa5b1", alpha=0.25)
ax.set_xlabel("minute")
ax.set_ylabel("power reading")
ax.set_title("what the analysis sees (sealed switch shaded)")
fig.tight_layout()

# %% [markdown]
# ## `recover`: when, and which mechanism
#
# `recover` takes a time series, the column to read, and a likelihood: the description of
# how a regime shows up in the data. `spectral_whittle` is the fast one that ships. It
# cuts the series into 96-minute frames and reads each frame's mean and spectrum, so each
# mechanism is one named parameter: level is the frame mean, spread the noise floor,
# rhythm (`freq`) the frequency of a periodic component, and amplitude (`amp`) its power.
#
# The hidden path is summed out exactly, so the sampler only explores the continuous
# parameters. The sampler budget below is smaller than the default to keep the notebook
# quick; raise it for real work.

# %%
from ergodic.latent import recover
from ergodic.latent.likelihoods import spectral_whittle

budget = {"chains": 2, "draws": 400, "warmup": 400}
result = recover(as_series(reading), "power", model=spectral_whittle(), seed=4, **budget)
print(result)
for block, (lo, mid, hi) in result.deltas().items():
    print(f"  {block:<7} delta {mid:+.2f}  94% interval [{lo:+.2f}, {hi:+.2f}]")
print("verdicts:", result.verdicts())

# %% [markdown]
# Only `level` fires. Its delta is in units of the standardized series, and its interval
# sits far from zero while the other three straddle it. The sign depends on which state
# the model calls first: the two labels are arbitrary, so read the size, not the sign.
#
# The printed intervals are the descriptive 94% ones. A verdict is stricter: each block's
# interval runs at 99.75%, so the four blocks together stay at a 99% familywise level, and
# testing four mechanisms doesn't buy four chances to fire.
#
# `gamma()` gives the posterior probability of each state per frame, and `tick_path()`
# spreads the decoded path back onto minutes. Now, and only now, unseal the switch to
# grade it.

# %%
path = result.tick_path()
# GRADING ONLY: compare with the sealed switch. Labels are arbitrary, so take the better
# of the two matchings.
match = float(np.mean(path == switch[: len(path)]))
print(f"agreement with the sealed switch: {max(match, 1.0 - match):.1%}")

state = result.gamma()[:, 1]
edges = np.arange(len(state) + 1) * (N // len(state))
fig, ax = plt.subplots(figsize=(9, 2.5))
ax.stairs(state, edges, fill=True, color="#33415c", alpha=0.45, label="P(second state)")
ax.plot(switch, color="#d1495b", lw=1, label="sealed switch (grading only)")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("minute")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()

# %% [markdown]
# The decoded path agrees with the switch on about 96% of minutes. The misses sit at the
# edges: the path moves in 96-minute frames, so a switch that flips inside a frame can
# only be placed to the nearest frame. That's the price of reading spectra, and it buys
# speed and the amplitude mechanism, which a tick-by-tick autoregression can't see.
#
# Run the same call on the control world, where the switch does nothing:

# %%
result_control = recover(as_series(control), "power", model=spectral_whittle(), seed=0, **budget)
print("control verdicts:", result_control.verdicts())

# %% [markdown]
# Nothing fires. That's the behavior you want, but one quiet world isn't evidence of much.
# The next two functions measure it.
#
# ## `certify`: is there a regime at all
#
# `certify` asks the question without any mechanism vocabulary. It measures how much better
# a two-state model fits than the likelihood's own one-state model, then repeats the same
# computation on `n_null` worlds simulated from the one-state fit. The p-value is where the
# real improvement lands among the simulated ones.
#
# With 9 simulated worlds, the smallest p-value you can get is 1 / (1 + 9) = 0.1. The
# default is 99 worlds, a floor of 0.01, at about ten times the cost.

# %%
from ergodic.latent import certify

cert = certify(as_series(reading), "power", model=spectral_whittle(), n_null=9, seed=0)
print(cert)
print(
    f"null worlds at or above the observed improvement: {int(np.sum(cert.null >= cert.observed))}"
)

# %% [markdown]
# The observed improvement is far above every simulated null world, so the p-value sits at
# its floor of 0.1. That's the strongest statement nine worlds can make. To claim more,
# buy more worlds.
#
# ## `calibrate`: would it have fired on nothing?
#
# A detector that fires on worlds without a regime is uninformative, however good it looks
# on worlds with one. `calibrate` takes a configured detector (`latent_model` bundles the
# likelihood and the budget), simulates `n_controls` regime-free worlds from the observed
# series' own one-state fit, runs the full `recover` on each, and counts the fires.

# %%
from ergodic.latent import calibrate, latent_model

detector = latent_model(model=spectral_whittle(), **budget)
report = calibrate(detector, as_series(reading), "power", n_controls=5, seed=0)
print(report)
report.to_frame()

# %% [markdown]
# No fires on five control worlds. Read the interval, not just the count: zero of five is
# consistent with a false-fire rate as high as 43%. Five worlds keep this notebook fast;
# twenty or more is what you'd run before trusting a detector on your own data. The
# control worlds inherit your series' memory and scale, so the rate describes this
# detector on data like yours, not a generic promise.
#
# ## What to take away
#
# Ask the three questions separately. `recover` tells you when and which, given that a
# regime exists. `certify` tells you whether one exists, against a null it simulates
# rather than assumes. `calibrate` tells you how often the detector would have fired on
# nothing. A switching model that skips the last two will always find a story.
#
# The guides cover the rest:
# [recovering hidden regimes](../../docsite/pages/recover-hidden-regimes.md) explains the
# likelihoods and how to swap in `switching_ar`, and
# [certify and calibrate](../../docsite/pages/certify-and-calibrate.md) covers the budgets.
# The [next guide](14-causal-forests-and-policy-trees.py) returns to effects that differ
# from customer to customer.
