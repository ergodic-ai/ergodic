# SPDX-License-Identifier: Apache-2.0
"""Value a targeting rule: RATE, the Qini, and a treat/don't decision in dollars.

Heterogeneity is only worth measuring if you can act on it. The action is a targeting
rule: rank customers by their predicted effect and treat from the top down. The
Rank-Weighted ATE (RATE) values exactly that rule.

  ``.rate("autoc")``  the AUTOC: most powerful when a small subgroup has large
                      effects (weight everyone equally along the curve).
  ``.rate("qini")``   the Qini: most powerful when the benefit is diffuse (weight
                      by how many you treat).

A positive RATE means the ranking puts the high-benefit customers on top, so
targeting beats blanket treatment. The one-sided ``rejects()`` is also a clean HTE
test that directly targets rank quality.

The world plants ``tau(z) = 1 + z``, so a discount that costs 0.5 spend-units HURTS
the low-z customers (negative net) and pays off for high-z ones. We use the predicted
effect to decide who to treat and price the decision against blanket-treating
everyone, on a 100,000-customer book.

Note: the RATE interval here is the two-sided normal CI, which the literature flags as
anti-conservative for RATE; use ``rejects()`` (the one-sided test) for a significance
call, and read the CI as a rough band.

Run with: uv run python examples/cate/05_targeting_with_rate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import cate, dag, tabular


def main() -> None:
    # --- tau(z) = 1 + z: the discount hurts low-z, helps high-z --------------
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(0, 1, n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + 1.0 * z) * a + 1.5 * z + rng.normal(size=n)

    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    result = cate(graph, data, "A", "Y", random_state=0)
    h = result.explain()

    # --- the rank-weighted value of the predicted-effect ranking -------------
    autoc = h.rate("autoc", random_state=0)
    qini = h.rate("qini", random_state=0)

    # --- the decision: treat where predicted lift beats the cost -------------
    cost = 0.5  # cost of the discount, in spend-units, per customer
    tau = result.values
    target = tau > cost  # treat only where the predicted effect clears the cost
    blanket_net = float(np.mean(tau)) - cost  # treat everyone
    targeted_net = float(np.sum(np.maximum(tau - cost, 0.0))) / n  # treat the top set
    book = 100_000  # customers

    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted tau(z) = 1 + z; discount cost 0.5/customer; book of 100,000.\n")
    print("RATE -- is the predicted-effect ranking a good targeting rule")
    lo, hi = autoc.interval()
    print(
        f"  AUTOC {autoc.estimate:6.3f}   one-sided p {autoc.pvalue:.4f}"
        f"   rejects 'no HTE'? {autoc.rejects()}   (CI [{lo:.2f}, {hi:.2f}])"
    )
    print(f"  Qini  {qini.estimate:6.3f}   one-sided p {qini.pvalue:.4f}\n")
    print("THE DECISION -- net value per customer, scaled to the book")
    print(
        f"  blanket  (treat all):           {blanket_net:6.3f}/customer"
        f"   -> {blanket_net * book:>12,.0f} total"
    )
    print(
        f"  targeted (predicted lift>cost): {targeted_net:6.3f}/customer"
        f"   -> {targeted_net * book:>12,.0f} total"
    )
    print(f"  treated fraction under targeting: {float(np.mean(target)):.1%}")
    gain = (targeted_net - blanket_net) * book
    print(f"\n  targeting is worth ~{gain:,.0f} more than blanket on this book.")
    print("=" * 78)
    print(
        "A positive RATE confirms the ranking is good; the decision turns it into money\n"
        "by sparing the customers the discount would lose you."
    )


if __name__ == "__main__":
    main()
