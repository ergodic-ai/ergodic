# SPDX-License-Identifier: Apache-2.0
"""But-for accounting: the incremental revenue a campaign caused.

The CFO's question about a promotion is not "what did promoted customers spend" but
"what would they have spent had we not promoted them?" The difference, summed over the
promoted base, is the incremental revenue, the only number that belongs in an ROI
calculation. The naive version (promoted spend minus unpromoted spend) is almost always
too generous, because the customers you choose to promote are not a random sample.

Here loyal customers both receive the promotion and spend more anyway. The naive
incremental credits the promotion with all of that loyalty-driven spend. The SCM builds
the genuine but-for world: for every promoted customer it abducts their own situation,
then asks what their spend would have been under do(promo = 0), holding everything else
fixed. Summed up, that is the honest incremental, and it lands on the planted truth of
$50 per customer while the naive number more than doubles it.

This is counterfactual accounting at portfolio scale: one twin-network query per
customer, aggregated into a single line a finance team can defend. In this
homogeneous-effect world the total equals the back-door ATT that estimate_effect would
report; the SCM's extra is the per-customer ledger beneath the total. When the effect
varies by customer, that per-customer view stops being optional (see 08).

Run with: uv run python examples/scm/07_but_for_accounting.py
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

    rng = np.random.default_rng(21)
    n = 4000
    loyalty = rng.normal(0, 1, n)
    promo = rng.binomial(1, 1.0 / (1.0 + np.exp(-(1.0 * loyalty)))).astype(int)
    true_effect = 50.0  # dollars of incremental spend the promo causes
    spend = true_effect * promo + 80.0 * loyalty + rng.normal(0, 20.0, n)

    data = tabular(
        {"Loyalty": loyalty, "Promo": promo, "Spend": spend}, types={"Promo": "categorical"}
    )
    graph = dag(["Loyalty -> Promo", "Loyalty -> Spend", "Promo -> Spend"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=21).fit()

    promoted = np.where(promo == 1)[0]
    n_promo = len(promoted)

    # the but-for world: each promoted customer's spend under do(Promo = 0).
    units = data.take(promoted.tolist())
    but_for = scm.counterfactual(units, {"Promo": 0}, "Spend", use_draws=False)
    assert isinstance(but_for, SCMResult)
    per_customer_incremental = spend[promoted] - but_for.draws()  # one value per customer
    causal_incremental = float(per_customer_incremental.sum())

    # the naive incremental: promoted spend minus unpromoted spend, times the base.
    naive_per = float(spend[promo == 1].mean() - spend[promo == 0].mean())
    naive_incremental = naive_per * n_promo
    true_incremental = true_effect * n_promo

    print(__doc__.strip().splitlines()[0])
    print("=" * 72)
    print(f"Promoted customers: {n_promo:,}     true effect: ${true_effect:.0f} each\n")
    print(f"{'measure':<40}{'per customer':>14}{'incremental':>16}")
    print("-" * 72)
    causal_per = causal_incremental / n_promo
    print(f"{'naive (promoted minus unpromoted)':<40}{naive_per:>13.0f}{naive_incremental:>15,.0f}")
    print(
        f"{'causal but-for (SCM twin network)':<40}{causal_per:>13.0f}{causal_incremental:>15,.0f}"
    )
    print(f"{'planted truth':<40}{true_effect:>13.0f}{true_incremental:>15,.0f}")
    print("=" * 72)
    overcredit = naive_incremental - causal_incremental
    ratio = naive_per / causal_per
    print(
        f"The naive read over-credits the promotion by ${overcredit:,.0f} ({ratio:.1f}x the\n"
        f"true per-customer lift), because loyal customers were always going to spend more.\n"
        f"The but-for world strips that out, customer by customer."
    )


if __name__ == "__main__":
    main()
