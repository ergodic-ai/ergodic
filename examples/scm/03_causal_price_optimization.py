# SPDX-License-Identifier: Apache-2.0
"""Price optimization off the causal demand curve, not the observed one.

You want the price that maximizes profit. Profit is (price - cost) times demand, so
you need the demand curve: how much quantity moves as you change price. The trap is
that the demand curve you can *see* in historical data is not the one you would get if
you changed the price.

Here a promotion calendar (S) is the saboteur. When there is a promotion, price is low
and demand is high, for reasons that have nothing to do with the price elasticity. So
the *observed* price-quantity relationship looks far steeper than the truth: it says
customers are price sensitive, when the promotions were moving both knobs at once.
Pricing off that observed curve underprices badly.

The SCM recovers the real demand curve by asking E[demand | do(price = p)], which
adjusts for the promotion calendar by construction. The true elasticity is gentle
(demand falls 0.8 per unit price), so the profit-maximizing price is high. The naive
curve thinks demand is elastic and sets a much lower price, sacrificing a big slice of
profit.

Run with: uv run python examples/scm/03_causal_price_optimization.py
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

    # S = promotion intensity (observed). Promotions push price DOWN and demand UP,
    # which is confounding, not elasticity. The true elasticity is only 0.8.
    alpha, beta, cost = 20.0, 0.8, 2.0  # true demand = alpha - beta*price (+ promo)
    rng = np.random.default_rng(11)
    n = 4000
    s = rng.normal(0, 1, n)
    price = 10.0 - 2.0 * s + rng.normal(0, 0.5, n)
    demand = alpha - beta * price + 3.0 * s + rng.normal(0, 1.0, n)

    data = tabular({"S": s, "Price": price, "Demand": demand})
    graph = dag(["S -> Price", "S -> Demand", "Price -> Demand"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=11).fit()

    grid = np.arange(6.0, 18.01, 0.5)

    # CAUSAL demand curve: E[Demand | do(Price = p)], promotion calendar adjusted out.
    causal_demand = np.array(
        [scm.interventional("Demand", {"Price": float(p)}, n=1500).mean() for p in grid]
    )
    causal_profit = (grid - cost) * causal_demand
    p_causal = float(grid[int(np.argmax(causal_profit))])

    # NAIVE demand curve: the observed price-quantity regression, confounded by S.
    obs_slope = float(np.cov(price, demand)[0, 1] / np.var(price))
    obs_intercept = float(np.mean(demand) - obs_slope * np.mean(price))
    naive_demand = obs_intercept + obs_slope * grid
    naive_profit = (grid - cost) * naive_demand
    p_naive = float(grid[int(np.argmax(naive_profit))])

    # TRUTH: evaluate each recommended price against the real (planted) demand law.
    def true_profit(p: float) -> float:
        return (p - cost) * (alpha - beta * p)

    p_star = (alpha + beta * cost) / (2.0 * beta)  # analytic profit-max price
    best = true_profit(p_star)
    lost = (best - true_profit(p_naive)) / best

    print(__doc__.strip().splitlines()[0])
    print("=" * 74)
    slope_abs = abs(obs_slope)
    print(f"True price elasticity (demand drop per +1 price):  {beta:.2f}")
    print(
        f"Observed slope (confounded by promos): {slope_abs:.2f}, "
        f"looks {slope_abs / beta:.1f}x as elastic\n"
    )
    print(f"{'method':<34}{'price':>8}{'true profit':>14}")
    print(f"{'-' * 56}")
    print(f"{'naive (off observed correlation)':<34}{p_naive:>8.2f}{true_profit(p_naive):>14.1f}")
    print(
        f"{'causal SCM (do-price demand curve)':<34}{p_causal:>8.2f}{true_profit(p_causal):>14.1f}"
    )
    print(f"{'true profit-maximizing price':<34}{p_star:>8.2f}{best:>14.1f}")
    print("=" * 74)
    print(
        f"Pricing off the correlation sets ${p_naive:.2f}, believing demand is elastic,\n"
        f"and leaves {lost:.0%} of profit on the table. The causal curve sets ${p_causal:.2f},\n"
        f"a grid step from the true optimum ${p_star:.2f}."
    )

    assert isinstance(scm.interventional("Demand", {"Price": 10.0}), SCMResult)


if __name__ == "__main__":
    main()
