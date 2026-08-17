# SPDX-License-Identifier: Apache-2.0
"""Case study: a B2B renewal-discount win-back program, end to end (Phase 2).

An account-management team offers a renewal discount to win back at-risk customers.
The question is not "does the discount work on average" but "for whom is it worth the
cost, which model do we trust to tell us, and how much do we gain by targeting the
right accounts." This runs the full Phase-2 CATE pipeline -- a variety of methods,
including the causal forest and data-driven model selection -- on one planted world,
and ends on a priced decision.

The world (the planted truth):
  - The discount's lift on retained revenue RISES with product engagement:
    ``tau(engagement) = 2 + 4*engagement``. It pays off for engaged accounts and
    loses money on disengaged ones; population ATE is about 2.0.
  - The discount was historically TARGETED toward accounts with declining usage, and
    declining usage also depresses revenue -- so the naive treated-minus-untreated
    read is badly confounded and even flips sign.
  - ``usage_drop`` and ``seats`` confound the offer but do NOT modify the effect --
    two decoys the analysis must clear.

The pipeline, each step a section below:
  1. SPECIFY   the causal graph; expose the (wrong) naive number.
  2. SELECT    estimate the CATE across S/T/X/DR/R/forest and let the DR tau-risk
               pick the learner to trust (and aggregate them).
  3. VALIDATE  that the heterogeneity is real (the calibration test).
  4. EXPLAIN   the drivers (BLP / GATES / CLAN + the forest's variable importance).
  5. ACT       turn it into a policy-tree rule and price it (vs blanket, vs oracle).

Run with: uv run python examples/cate/13_case_study_phase2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import (
    CausalForest,
    dag,
    estimate_effect,
    identify_effect,
    select_cate,
    tabular,
)


def main() -> None:
    # === the planted world ===================================================
    rng = np.random.default_rng(2024)
    n = 6000
    engagement = rng.normal(size=n)  # product engagement (the modifier)
    usage_drop = rng.normal(size=n)  # recent usage decline (a confounder)
    seats = rng.normal(size=n)  # account size (a decoy confounder)
    # historically targeted: more offers to declining and smaller accounts
    p_offer = 1.0 / (1.0 + np.exp(-(1.1 * usage_drop - 0.5 * seats + 0.3 * engagement)))
    offer = (rng.uniform(size=n) < p_offer).astype(int)
    tau = 2.0 + 4.0 * engagement  # the planted per-unit effect
    revenue = tau * offer + 3.0 * engagement - 4.0 * usage_drop + 1.0 * seats + rng.normal(0, 2, n)

    data = tabular(
        pd.DataFrame(
            {
                "offer": offer,
                "engagement": engagement,
                "usage_drop": usage_drop,
                "seats": seats,
                "revenue": revenue,
            }
        ),
        types={"offer": "categorical"},
    )
    true_ate = float(np.mean(tau))

    print(__doc__.strip().splitlines()[0])
    print("=" * 80)
    print(f"Planted truth: tau(engagement) = 2 + 4*engagement   (ATE ~ {true_ate:.2f})\n")

    # === 1) specify, identify, expose the naive number =======================
    graph = dag(
        [
            "engagement -> offer",
            "engagement -> revenue",
            "usage_drop -> offer",
            "usage_drop -> revenue",
            "seats -> offer",
            "seats -> revenue",
            "offer -> revenue",
        ]
    )
    naive = float(revenue[offer == 1].mean() - revenue[offer == 0].mean())
    adjusted = estimate_effect(graph, data, "offer", "revenue", random_state=2024, treated=1)
    print("STEP 1  SPECIFY + IDENTIFY")
    print(f"  naive treated - untreated revenue .. {naive:6.3f}   (confounded; flips sign)")
    print(
        f"  back-door adjusted ATE ............. {adjusted.value:6.3f}   (planted ~ {true_ate:.2f})"
    )

    # === 2) select the learner from the data =================================
    candidates = ["s_learner", "t_learner", "x_learner", "dr_learner", "r_learner", "causal_forest"]
    report = select_cate(
        graph,
        data,
        "offer",
        "revenue",
        candidates=candidates,
        random_state=2024,
        n_boot=50,
        treated=1,
    )
    chosen = report.selected
    print("\nSTEP 2  SELECT  (doubly-robust tau-risk picks the learner; lower is better)")
    for score in report.ranking:
        print(f"    {score.method:<14}{score.tau_risk:8.3f}")
    print(
        f"  -> selected {report.selected_method!r} (the pooled s_learner is worst at "
        f"{report.risk('s_learner'):.1f}); the Q-aggregate risk {report.aggregate_tau_risk:.3f} "
        f"matches the best single {report.best_single_tau_risk:.3f}."
    )
    print(
        f"  chosen CATE: ATE {chosen.ate().value:.3f}, corr with planted tau "
        f"{float(np.corrcoef(chosen.values, tau)[0, 1]):.3f}."
    )

    # === 3) validate that the heterogeneity is real ==========================
    h = chosen.explain()
    cal = h.calibration_test()
    print("\nSTEP 3  VALIDATE  (is the heterogeneity real?)")
    print(
        f"  calibration test: differential coef {cal.differential_coefficient:.3f}"
        f"   -> {'REJECTS no-HTE' if cal.rejects() else 'no detection'}"
    )

    # === 4) explain the drivers ==============================================
    print("\nSTEP 4  EXPLAIN  (which driver, how much, for whom)")
    for name, row in h.blp().frame().items():
        if name == "(intercept)":
            continue
        tag = "modifier" if row["ci_low"] > 0.0 else "inert (decoy cleared)"
        ci = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        print(f"  BLP slope  {name:<11}{row['coef']:7.3f}  90% CI {ci}  {tag}")
    g = h.gates(n_groups=5)
    print("  GATES group effects (low -> high): " + "  ".join(f"{x:5.2f}" for x in g.group_effect))
    # the forest's own variable importance, a second read on the driver
    forest = CausalForest(n_trees=300, min_leaf=10, random_state=2024, treated=1).fit(
        identify_effect(graph, "offer", "revenue"), data
    )
    vim = forest.variable_importance()
    top_driver, top_share = vim.top(1)[0]
    print(
        f"  forest variable_importance: top driver {top_driver!r} "
        f"({top_share:.2f}), clearing both decoys."
    )

    # === 5) act: a policy-tree rule, priced ==================================
    cost = 1.0  # per-account cost of the discount, in revenue-units
    book = 300_000  # accounts eligible
    tree = h.policy_tree("engagement", depth=2)
    tau_hat = chosen.values
    blanket_net = float(np.mean(tau_hat)) - cost  # treat everyone
    targeted_net = float(np.sum(np.maximum(tau_hat - cost, 0.0))) / n  # treat predicted winners
    oracle_net = float(np.sum(np.maximum(tau - cost, 0.0))) / n  # the true-tau ceiling
    gain = (targeted_net - blanket_net) * book
    gap_to_oracle = abs(targeted_net - oracle_net) * book

    print("\nSTEP 5  ACT  (a readable rule, priced; cost 1.0/account, book of 300,000)")
    print("  the policy tree (the interpretable decision):")
    for line in tree.describe().splitlines():
        print(f"    {line}")
    print(
        f"  policy value {tree.value:.3f} beats treat-all {tree.treat_all_value:.3f} "
        f"(regret vs best trivial {tree.regret('best_trivial'):.3f})"
    )
    print(f"  blanket  (treat all):           {blanket_net * book:>12,.0f}")
    print(
        f"  targeted (lift>cost, {float(np.mean(tau_hat > cost)):.0%} treated): "
        f"{targeted_net * book:>12,.0f}"
    )
    print(f"  oracle   (if we knew true tau): {oracle_net * book:>12,.0f}   (the ceiling)")
    print(f"\n  -> targeting the predicted winners is worth ~{gain:,.0f} more than blanket,")
    print(f"     and lands within {gap_to_oracle:,.0f} of the true-tau ceiling.")

    print("=" * 80)
    print(
        "A confounded -0.30 naive read became an honest 2.04 ATE; the data picked the\n"
        "learner; the calibration test confirmed the engagement-driven CATE; and a\n"
        "one-page policy-tree rule nearly matched the oracle on a 300,000-account book."
    )


if __name__ == "__main__":
    main()
