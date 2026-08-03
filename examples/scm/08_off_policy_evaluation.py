# SPDX-License-Identifier: Apache-2.0
"""Off-policy evaluation: score a targeting policy you never ran.

A retention team can send a save offer. Sending it to everyone is expensive; sending it
to no one loses customers. The right policy sends it to the customers it will save,
the ones who would churn without it and stay with it. The trouble is you cannot
see who those are in the log, and you cannot A/B test every policy you might consider.

This is off-policy evaluation, and a fitted SCM does it directly. For every customer it
runs the twin network twice, under do(offer = 1) and do(offer = 0) reusing the same
abducted luck, and reads whether the offer flips them from churn to stay. That per-unit
label is exactly the "saveable" set, and it lets you score any policy without ever
deploying it.

The punchline is the gap between policies. The historical policy here rewarded loyal,
low-churn customers (a common and wasteful habit), so it spent heavily and saved few.
Blanket-offering saves everyone saveable but pays for thousands of wasted offers. The
targeted policy saves the same customers at a fraction of the cost, and the SCM names
them.

The "saveable" label is one twin-network draw per customer, a hard yes or no rather
than a probability, so a different seed would relabel a few borderline customers. Every
policy is scored against the same labeling, so the comparison between them stays fair.

Run with: uv run python examples/scm/08_off_policy_evaluation.py
"""

from __future__ import annotations

import logging

import numpy as np

from ergodic import dag, tabular
from ergodic.scm import SCM, CategoricalResult


def _quiet_pymc() -> None:
    for name in ("pymc", "pymc.sampling"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_pymc()

    rng = np.random.default_rng(13)
    n = 3000
    tenure = rng.normal(0, 1, n)
    # the HISTORICAL (logging) policy: reward high-tenure, loyal customers.
    offer = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.2 * tenure)))).astype(int)
    # churn logit: low-tenure churn more; the offer lowers the logit by ~1.0. The
    # offer's effect on churn PROBABILITY is heterogeneous (logistic), biggest for
    # customers sitting near the boundary.
    churn_logit = 0.3 - 1.3 * tenure - 1.0 * offer
    churn = rng.binomial(1, 1.0 / (1.0 + np.exp(-churn_logit))).astype(int)

    data = tabular(
        {"Tenure": tenure, "Offer": offer, "Churn": churn},
        types={"Offer": "categorical", "Churn": "categorical"},
    )
    graph = dag(["Tenure -> Offer", "Tenure -> Churn", "Offer -> Churn"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=13).fit()

    # per-customer twin network: churn under each offer arm, same abducted luck.
    cf1 = scm.counterfactual(data, {"Offer": 1}, "Churn")
    cf0 = scm.counterfactual(data, {"Offer": 0}, "Churn")
    assert isinstance(cf1, CategoricalResult) and isinstance(cf0, CategoricalResult)
    cats = list(cf1.categories)
    churn_code, stay_code = cats.index(1), cats.index(0)
    under_offer = cf1.per_unit()
    under_none = cf0.per_unit()
    assert under_offer is not None and under_none is not None

    # saveable: would churn with no offer, stays with the offer.
    saveable = (under_none == churn_code) & (under_offer == stay_code)
    n_saveable = int(saveable.sum())

    value_save, cost_offer = 300.0, 25.0  # dollars

    def value(offered: np.ndarray) -> tuple[int, int, float]:
        saved = int((saveable & offered).sum())
        n_off = int(offered.sum())
        return n_off, saved, value_save * saved - cost_offer * n_off

    policies = {
        "none": np.zeros(n, dtype=bool),
        "historical (the log)": offer.astype(bool),
        "blanket (offer all)": np.ones(n, dtype=bool),
        "targeted (SCM saveables)": saveable,
    }

    print(__doc__.strip().splitlines()[0])
    print("=" * 72)
    print(f"Customers: {n:,}     saveable by the offer: {n_saveable:,}")
    print(f"Each save is worth ${value_save:.0f}; each offer costs ${cost_offer:.0f}.\n")
    print(f"{'policy':<28}{'offers':>9}{'saved':>8}{'net value':>14}")
    print("-" * 72)
    results = {}
    for name, mask in policies.items():
        n_off, saved, net = value(mask)
        results[name] = (n_off, saved, net)
        print(f"{name:<28}{n_off:>9,}{saved:>8,}{net:>14,.0f}")
    print("=" * 72)
    hist_off, hist_saved, _ = results["historical (the log)"]
    print(
        f"The historical policy made {hist_off:,} offers to save {hist_saved:,}; targeted policy\n"
        f"makes one offer per save ({n_saveable:,} offers, {n_saveable:,} saved). Targeted and\n"
        f"blanket save the same people, but targeted skips the wasted spend, winning on\n"
        f"net value, chosen and scored entirely from the fitted model."
    )


if __name__ == "__main__":
    main()
