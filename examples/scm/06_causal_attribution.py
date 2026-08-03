# SPDX-License-Identifier: Apache-2.0
"""Causal multi-touch attribution: the best-looking channel can be the worst one.

Marketing attribution usually credits a channel with the conversions of the customers
who touched it. The problem is that channels do not reach customers at random. A
channel that happens to find high-intent buyers will look spectacular, whether or not
it changed a single decision. You end up pouring budget into the channel that is best
at *predicting* conversions rather than *causing* them.

Here a hidden-intent proxy (G, prior engagement) drives both which channels reach a
customer and whether they convert. Ads are shown heavily to high-intent customers, so
ads correlate strongly with conversion, but their true causal lift is tiny. Email is
targeted more gently and moves the needle.

The naive lift (convert rate among touched minus untouched) ranks ads above email. The
SCM, adjusting for intent, reverses it: email's causal effect is several times the ad's.
That is the difference between an attribution number you can spend against and one that
will quietly waste your budget.

Run with: uv run python examples/scm/06_causal_attribution.py
"""

from __future__ import annotations

import logging

import numpy as np

from ergodic import dag, tabular
from ergodic.scm import SCM, SCMResult


def _quiet_pymc() -> None:
    for name in ("pymc", "pymc.sampling"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_pymc()

    rng = np.random.default_rng(3)
    n = 5000
    g = rng.normal(0, 1, n)  # latent intent, observed here as prior engagement
    email = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.8 * g)))).astype(int)
    ad = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.5 * g)))).astype(int)
    # TRUE causal lifts: email 0.35, ad 0.05. Intent G adds a big confounded 1.2.
    conversion = 0.35 * email + 0.05 * ad + 1.2 * g + rng.normal(0, 0.5, n)

    data = tabular(
        {"G": g, "Email": email, "Ad": ad, "Conversion": conversion},
        types={"Email": "categorical", "Ad": "categorical"},
    )
    graph = dag(
        ["G -> Email", "G -> Ad", "G -> Conversion", "Email -> Conversion", "Ad -> Conversion"]
    )
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=3).fit()

    truth = {"Email": 0.35, "Ad": 0.05}

    def naive_lift(channel: str, col: np.ndarray) -> float:
        return float(conversion[col == 1].mean() - conversion[col == 0].mean())

    def causal_lift(channel: str) -> float:
        ate = scm.ate(channel, "Conversion")
        assert isinstance(ate, SCMResult)
        return ate.mean()

    naive = {"Email": naive_lift("Email", email), "Ad": naive_lift("Ad", ad)}
    causal = {"Email": causal_lift("Email"), "Ad": causal_lift("Ad")}

    print(__doc__.strip().splitlines()[0])
    print("=" * 70)
    print(f"{'channel':<10}{'naive lift':>12}{'causal lift':>14}{'true':>8}")
    print("-" * 70)
    for ch in ("Email", "Ad"):
        print(f"{ch:<10}{naive[ch]:>12.3f}{causal[ch]:>14.3f}{truth[ch]:>8.2f}")
    print("=" * 70)
    naive_winner = max(naive, key=naive.get)
    causal_winner = max(causal, key=causal.get)
    te, ta = truth["Email"], truth["Ad"]
    print(
        f"Naive attribution crowns {naive_winner!r} (it reaches the keen buyers); the causal\n"
        f"read crowns {causal_winner!r}. True lifts: email {te:.2f}, ad {ta:.2f}, so the\n"
        f"naive ranking is exactly backwards. Spending against it backs the weaker channel."
    )


if __name__ == "__main__":
    main()
