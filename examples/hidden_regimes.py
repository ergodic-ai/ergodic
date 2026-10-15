# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # Hidden regimes: a tour of the latent pillar
#
# Somewhere inside a home, a switch flips on a timescale of days. We never see the
# switch. We see one continuous trace, 1440 ticks of it, and we are told nothing
# about what the switch does to that trace, or whether it exists at all.
#
# In a simulation we can write the switch into the code, where everyone can read
# it, then seal it away from the analysis and grade the answers at the end. Real
# data never tells you whether your inference was right; a simulated world does.
# That is the move this notebook makes, once per question:
#
# 1. `recover()`: when was the regime on, and which named mechanism changed?
# 2. `certify()`: is there a regime at all, asked without any mechanism vocabulary?
# 3. `calibrate()`: would this detector have fired on worlds with no regime in them?
#
# Run with:
#
#     uv run --extra latent --group docs python examples/hidden_regimes.py
#
# (`--group docs` only supplies matplotlib; without it the notebook runs and
# simply skips the figures.) Budgets here are reduced for iteration speed; the
# whole notebook samples for about three minutes. Open it as a notebook in
# VS Code or Jupytext: the `# %%` markers are cells.

# %%
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ergodic import timeseries
from ergodic.latent import calibrate, certify, latent_model, recover
from ergodic.latent.likelihoods import spectral_whittle

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

N = 1440
BUDGET = {"chains": 2, "draws": 400, "warmup": 400}

# %% [markdown]
# ## The sealed world
#
# The generator below is the answer key. The switch is semi-Markov with dwells
# around 240 ticks, the trace chases a target with first-order lag, and one
# mechanism wires the switch to the trace: a level shift, a noise-scale change,
# a frequency change, or an amplitude change. `control` wires it to nothing,
# which is the arm every honest detector must stay silent on.
#
# Nothing past this cell reads `switch` until the final grading lines, and those
# are marked.


# %%
@dataclass(frozen=True)
class Mechanism:
    level_shift: float = 0.0
    spread_ratio: float = 1.0
    freq_ratio: float = 1.0
    amplitude_ratio: float = 1.0
    amplitude: float = 0.0
    freq_off: float = 0.05


ARMS = {
    "control": Mechanism(),
    "level": Mechanism(level_shift=1.0),
    "spread": Mechanism(spread_ratio=3.0),
    "rhythm": Mechanism(amplitude=0.8, freq_ratio=0.4),
    "amplitude": Mechanism(amplitude=0.8, amplitude_ratio=2.0),
}
# Worlds are seeded where the golden tests pinned them (their docstrings record
# why: renaming the RNG renamed the worlds, and some seeds are unlucky).
SEEDS = {"control": 0, "level": 4, "spread": 0, "rhythm": 0, "amplitude": 25}


def generate(mechanism: Mechanism, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean, var = 240.0, (240.0 / 3.0) ** 2
    r, p = mean**2 / (var - mean), mean / var
    states: list[int] = []
    current = 0
    while len(states) < n:
        states.extend([current] * max(1, int(rng.negative_binomial(r, p)) + 1))
        current = 1 - current
    switch = np.array(states[:n])

    level = 100.0 + mechanism.level_shift * switch
    if mechanism.amplitude == 0.0:
        target = level.astype(float)
    else:
        freq = mechanism.freq_off * (1.0 + (mechanism.freq_ratio - 1.0) * switch)
        amp = mechanism.amplitude * (1.0 + (mechanism.amplitude_ratio - 1.0) * switch)
        target = level + amp * np.sin(np.cumsum(2.0 * np.pi * freq))

    sd = 0.1 * (1.0 + (mechanism.spread_ratio - 1.0) * switch)
    # Shocks are pre-drawn in one vectorized call, matching the golden tests'
    # draw order exactly, so the pinned seeds above name the same worlds.
    shocks = rng.normal(0.0, 1.0, size=n)
    signal = np.empty(n)
    signal[0] = target[0]
    for t in range(1, n):
        signal[t] = 0.3 * target[t] + 0.7 * signal[t - 1] + shocks[t] * sd[t]
    return signal, switch


def ts_of(signal: np.ndarray):
    frame = pd.DataFrame({"t": np.arange(len(signal)), "signal": signal})
    return timeseries(frame, time="t")


def agreement(path: np.ndarray, truth: np.ndarray) -> float:
    """SCORING ONLY: label-invariant accuracy against the sealed switch."""
    match = float(np.mean(path == truth[: len(path)]))
    return max(match, 1.0 - match)


worlds = {arm: generate(mech, N, SEEDS[arm]) for arm, mech in ARMS.items()}

# %% [markdown]
# What the analysis actually sees is the top trace alone. The shading is the
# sealed switch, drawn here only so you can appreciate the problem: on the level
# arm the shift is visible once you know where to look, and invisible before.

# %%
if plt is not None:
    signal, switch = worlds["level"]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, lw=0.6, color="#333333")
    ax.fill_between(np.arange(N), *ax.get_ylim(), where=switch == 1, alpha=0.12, color="tab:blue")
    ax.set_title("level world: the trace (sealed switch shaded)")
    ax.set_xlabel("tick")
    plt.show()

# %% [markdown]
# ## `recover()`: when, and which
#
# One call. The likelihood here is `spectral_whittle()`, the fast flagship: it
# summarizes the trace as non-overlapping spectral frames, so every mechanism in
# its vocabulary is a named parameter of the local spectrum (level: frame mean,
# spread: noise floor, rhythm: peak location, amplitude: peak height). Swap in
# `switching_ar()` and the same call answers from a different vocabulary; that
# swap is the modularity the pillar exists for.

# %%
signal, switch = worlds["level"]
result = recover(ts_of(signal), "signal", model=spectral_whittle(), seed=SEEDS["level"], **BUDGET)
print(result)
print()
for block, (lo, mean, hi) in result.deltas().items():
    print(f"  d_{block:<7} [{lo:+.2f}, {mean:+.2f}, {hi:+.2f}]")
print()
print("verdicts:", result.verdicts())

# %%
if plt is not None:
    gamma = result.gamma()[:, 1]
    edges = np.arange(len(gamma) + 1) * (N // len(gamma))
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.stairs(gamma, edges, fill=True, alpha=0.5)
    ax.plot(switch, lw=1.0, color="#333333", label="sealed switch (grading only)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("posterior P(regime) per frame vs the sealed switch; labels are arbitrary")
    ax.legend(loc="upper right")
    plt.show()

# UNSEALED, grading only:
print(f"path agreement vs sealed switch: {agreement(result.tick_path(), switch):.2f}")

# %% [markdown]
# ## The grid: one detector, five sealed worlds
#
# The table below is the notebook's centerpiece: which delta blocks fire on each
# arm, and how well the recovered path matches the sealed switch. `control` must
# stay silent, and the mechanism arms should each light up their own block.

# %%
rows = []
for arm, (signal, switch) in worlds.items():
    res = recover(ts_of(signal), "signal", model=spectral_whittle(), seed=SEEDS[arm], **BUDGET)
    fired = [block for block, hit in res.verdicts().items() if hit]
    rows.append(
        {
            "arm": arm,
            "fires": "+".join(fired) or "-",
            "agreement": round(agreement(res.tick_path(), switch), 3),  # UNSEALED, grading
        }
    )
grid = pd.DataFrame(rows).set_index("arm")
print(grid)

# %% [markdown]
# Read the grid with its known blemishes in view: co-fires happen at frame
# boundaries, where a regime change inside one spectral frame leaks
# low-frequency energy that level and amplitude can each mistake for the other
# (the likelihood's docstring documents this confusability and its window-length
# cause), and rhythm's amplitude echo, a real consequence of the lag filter
# attenuating faster waves, appears or not depending on budget.
#
# The amplitude arm deserves a sentence, because it is the hard one. An
# autoregression predicts the next tick from the last few, and a sinusoid
# satisfies the same recursion at any amplitude, so `switching_ar()` is
# structurally blind to this arm (its docstring argues why). The spectral
# likelihood sees amplitude because a frame's band energy is a block statistic,
# not a one-step prediction. The vocabulary that names mechanisms is also the
# aperture that can miss them; that is why the next verb exists.
#
# ## `certify()`: is there a regime at all
#
# No vocabulary, one number: how much better a two-regime story fits than the
# best one-regime story, compared against worlds simulated from that one-regime
# fit. With nine nulls the finest attainable p is 0.1; buy resolution with
# `n_null` when it matters.

# %%
cert = certify(ts_of(worlds["level"][0]), "signal", model=spectral_whittle(), n_null=9, seed=0)
print("level world:  ", cert)
cert0 = certify(ts_of(worlds["control"][0]), "signal", model=spectral_whittle(), n_null=9, seed=0)
print("control world:", cert0)

# %% [markdown]
# ## `calibrate()`: would it have fired on nothing?
#
# The verb the pillar refuses to skip. The first spectral prototype looked
# excellent and fired on 45% of regime-free worlds; the shipped likelihood was
# repaired until a 20-world calibration ran at its nominal familywise rate. Five
# worlds here for speed; the report's Wilson interval says how little five
# worlds prove, which is exactly the point of printing it.

# %%
spec = latent_model(model=spectral_whittle(), **BUDGET)
report = calibrate(spec, ts_of(worlds["level"][0]), "signal", n_controls=5, seed=0)
print(report)
print(report.to_frame())

# %% [markdown]
# ## Where to go next
#
# Swap the likelihood (`switching_ar()`, or subclass `RegimeLikelihood` and
# implement `prepare`/`emissions`: the engine owns the chain, the forward pass,
# the decoding, and the label alignment). Tighten or loosen the dwell prior with
# `chain=MarkovChain(...)`. Raise the budgets for real runs. Compare two
# likelihoods on one trace with `consensus()`; its cross-model amplitude rule is
# an explicitly empty seam awaiting the volatility-switching likelihood. The
# hidden-regime guides (https://ergodic.dev/recover-hidden-regimes.html and
# https://ergodic.dev/certify-and-calibrate.html) document this surface.
