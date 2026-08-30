# SPDX-License-Identifier: Apache-2.0
"""A hidden switch, recovered blind: the latent pillar's three verbs on one world.

A hidden two-state switch dwells for ~240 ticks at a time and shifts a
signal's level while it is on; an EMA sensor chases that level through noise.
The script seals the switch away, hands only the signal to the pillar, and
then asks the three questions in order: recover() for WHEN the switch was on
and WHICH mechanism block moved, certify() for WHETHER a regime exists at all
(against a null distribution that is simulated, not assumed), and a small
calibrate() run for whether this detector would have fired on worlds with no
switch in them. The sealed switch reappears only at the end, to score the
recovered path -- scoring, never fitting.

Needs the latent extra (PyMC): uv sync --extra latent
Run with: uv run --extra latent python examples/latent_switch.py
Budgets are modest on purpose; expect a few minutes. At these reduced budgets
the level arm is seed-sensitive, so the recover() call pins the sampler seed
the golden tests measured (world 4, sampler seed 4); the full reference
budget (4 chains of 1000 after 1000) does not need the pin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ergodic import calibrate, certify, latent_model, recover, timeseries
from ergodic.latent.likelihoods import spectral_whittle

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]

# The sealed world: the golden generator's level arm (dwell ~240, EMA
# reaction 0.3, noise 0.1, level shift +1.0). Seed 4 is the measured level
# world the golden tests pin: the shift is visible but never labeled.
DWELL = 240
N = 1440
SEED = 4
LEVEL_SHIFT = 1.0


def generate(n: int, seed: int) -> tuple[Array, IntArray]:
    """One home's trajectory, plus the sealed switch (for scoring only)."""
    rng = np.random.default_rng(seed)
    mean = float(DWELL)
    var = (mean / 3.0) ** 2
    r = mean**2 / (var - mean)
    p = mean / var
    states: list[int] = []
    current = 0
    while len(states) < n:
        run = max(1, int(rng.negative_binomial(r, p)) + 1)
        states.extend([current] * run)
        current = 1 - current
    switch = np.array(states[:n], dtype=np.int64)
    target = 100.0 + LEVEL_SHIFT * switch
    shocks = rng.normal(0.0, 1.0, size=n)
    signal = np.zeros(n)
    signal[0] = target[0]
    for t in range(1, n):
        signal[t] = 0.3 * target[t] + 0.7 * signal[t - 1] + shocks[t] * 0.1
    return signal, switch


def agreement(path: IntArray, truth: IntArray) -> float:
    """Accuracy up to a label swap: state indices are arbitrary."""
    match = float(np.mean(path == truth))
    return max(match, 1.0 - match)


def main() -> None:
    signal, switch = generate(N, SEED)
    data = timeseries({"t": np.arange(N), "signal": signal}, time="t")
    print(f"SEALED: the switch dwells ~{DWELL} ticks and moves the level by +1.0.")
    print(f"The pillar sees one column of {N} ticks and nothing else.\n")

    # -- recover: WHEN was it on, WHICH block moved ---------------------------
    result = recover(
        data, "signal", model=spectral_whittle(), draws=400, warmup=400, chains=2, seed=SEED
    )
    print("recover():", result)
    print("  deltas (94% equal-tailed):")
    for name, (lo, mean, hi) in result.deltas().items():
        print(f"    {name:7s} [{lo:+.2f}, {mean:+.2f}, {hi:+.2f}]")
    print("  verdicts (99% familywise, gated):", result.verdicts())

    # -- certify: WHETHER a regime exists at all ------------------------------
    cert = certify(data, "signal", model=spectral_whittle(), n_null=19, seed=0)
    print("\ncertify():", cert)
    print(
        f"  observed improvement {cert.observed:.1f} against "
        f"{cert.n_null} simulated no-switch worlds "
        f"(their max: {float(cert.null.max()):.1f}) -> p = {cert.p_value:.3f}"
    )

    # -- calibrate: would this detector fire on nothing? ----------------------
    spec = latent_model(model=spectral_whittle(), draws=250, warmup=250, chains=1)
    report = calibrate(spec, data, "signal", n_controls=3, seed=0)
    print("\ncalibrate():", report)
    print(report.to_frame().to_string(index=False))

    # -- unseal and score (scoring only, never fitting) -----------------------
    path = result.tick_path()
    score = agreement(path, switch[: len(path)])
    print(f"\nUNSEALED: recovered path agrees with the hidden switch at {score:.2f}.")


if __name__ == "__main__":
    main()
