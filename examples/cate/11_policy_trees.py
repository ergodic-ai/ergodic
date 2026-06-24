# SPDX-License-Identifier: Apache-2.0
"""From a ranking to a rule you can act on: the optimal policy tree.

RATE (script 05) tells you a predicted-effect *ranking* is good. But a ranking is
not yet a decision -- you still have to draw the line. A policy tree draws it for
you, and draws it interpretably: a shallow decision tree that reads the
doubly-robust scores and returns a treat / don't-treat RULE, the one that maximizes
welfare ``W(pi) = mean[(2 pi(X) - 1) * Gamma_i]`` (Athey-Wager 2021). At depth <= 2
the search over thresholds is exhaustive, so the rule is globally optimal, not
greedy.

The world plants a SIGN-FLIPPING effect: a discount's lift on spend is
``tau(tenure) = 2*tenure``. It HELPS loyal customers (tenure > 0) and HURTS new
ones (tenure < 0), so the optimal rule is exactly "treat iff tenure > 0" and the
population average effect (~0) hides it completely. Tenure also drives who got the
offer historically, so the graph says to adjust.

What the script shows:
  - ``describe()`` prints the learned if-else rule -- a split on tenure near 0.
  - the policy VALUE beats both treat-all and treat-none (regret >= 0), with an
    honest interval from the influence terms.
  - priced on a book of customers, the rule is worth real money over blanket.
  - RATE ranks; the policy tree is the rule that ranking implies.

Run with: uv run python examples/cate/11_policy_trees.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import cate, dag, tabular


def main() -> None:
    # --- a sign-flipping world: tau(tenure) = 2*tenure -----------------------
    # optimal rule: treat iff tenure > 0. The average effect (~0) hides it.
    rng = np.random.default_rng(0)
    n = 5000
    tenure = rng.normal(size=n)
    p_offer = 1.0 / (1.0 + np.exp(-0.6 * tenure))  # historically targeted (confounding)
    offer = (rng.uniform(size=n) < p_offer).astype(float)
    tau = 2.0 * tenure  # the planted per-unit effect, sign-flips at 0
    spend = tau * offer + 1.0 * tenure + rng.normal(size=n)

    data = tabular(pd.DataFrame({"offer": offer, "tenure": tenure, "spend": spend}))
    graph = dag(["tenure -> offer", "tenure -> spend", "offer -> spend"])

    # --- estimate the CATE, then learn the policy off its scores -------------
    result = cate(graph, data, "offer", "spend", method="dr_learner", random_state=0, n_boot=100)
    h = result.explain()
    tree = h.policy_tree("tenure", depth=1)  # the depth-1 globally optimal rule

    # the ranking (RATE) the rule is built on, for the tie-in
    autoc = h.rate("autoc", random_state=0)

    # --- price the decision --------------------------------------------------
    book = 200_000  # customers eligible; the policy value is in spend-units/customer
    gain_vs_all = (tree.value - tree.treat_all_value) * book
    lo, hi = tree.interval(0.9)

    # --- report --------------------------------------------------------------
    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted truth: tau(tenure) = 2*tenure   (optimal rule: treat iff tenure > 0)\n")

    print("THE RANKING (RATE) -- is the predicted-effect ordering worth acting on?")
    print(
        f"  AUTOC {autoc.estimate:.3f}  (one-sided p {autoc.pvalue:.4f}, "
        f"rejects 'no HTE'? {autoc.rejects()})  -> yes, the ranking is good"
    )

    print("\nTHE RULE (policy tree) -- the decision that ranking implies")
    for line in tree.describe().splitlines():
        print(f"  {line}")
    assert tree.root.threshold is not None
    print(f"  -> splits tenure at {tree.root.threshold:+.3f}, the planted flip point ~ 0.")

    print("\nIS THE RULE WORTH IT? (policy value vs the trivial baselines)")
    print(f"  policy value (the learned rule) .. {tree.value:6.3f}   90% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  treat everyone ................... {tree.treat_all_value:6.3f}")
    print(f"  treat no one ..................... {tree.treat_none_value:6.3f}")
    print(
        f"  regret vs best trivial ........... {tree.regret('best_trivial'):6.3f}   "
        "(>= 0: the rule is at least as good)"
    )
    print(f"  treated fraction under the rule .. {float(tree.assign().mean()):.1%}")

    print(f"\nPRICED (book of {book:,})")
    print(f"  the rule is worth ~{gain_vs_all:,.0f} more than blanket-treating everyone.")

    print("=" * 78)
    print(
        "RATE said the ranking was good; the policy tree turned it into a one-line\n"
        "rule -- treat the positive-tenure customers -- and priced it against blanket."
    )


if __name__ == "__main__":
    main()
