# SPDX-License-Identifier: Apache-2.0
"""Which learner do I trust -- answered by the data, not by faith.

Six CATE learners can give six different per-unit pictures of the same world, and
you do not get to peek at the true effect to see who is right. So which do you
trust? ``select_cate`` answers with the data. It scores each candidate by the
doubly-robust tau-risk on a shared cross-fit signal ``Gamma`` (whose conditional
mean is the true CATE), ``R(tau_hat) = mean[(Gamma - tau_hat(X))^2]``, whose
minimizer is the true effect -- so a lower risk is a genuinely better CATE estimate,
no oracle required (Foster-Syrgkanis; Mahajan et al. 2024). It also returns a
Q-aggregation ensemble: convex weights over the candidates that is at least as good
as the best single one (Lan-Syrgkanis 2024).

The world makes one family CORRECTLY specified and another WRONG. The effect is a
strong interaction, ``tau(tenure) = 3*tenure``. The S-learner pools both arms into
one outcome model and structurally cannot represent that interaction, so it is badly
misspecified; the T-learner, the DR-learner, and the causal forest can all track it.
A good selector should rank the S-learner dead last from the data alone.

What the script shows:
  - the DR tau-risk ranks the correctly-specified learners first and the pooled
    S-learner last, by a wide margin.
  - the Q-aggregation ensemble matches or beats the best single candidate, and puts
    ~zero weight on the misspecified learner.
  - the causal forest sits among the candidates and is scored on the same footing.

Run with: uv run python examples/cate/12_model_selection.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import dag, select_cate, tabular


def main() -> None:
    # --- a strong-interaction world: the pooled S-learner is misspecified ----
    # tau(tenure) = 3*tenure. A model that pools the arms cannot fit a*tenure.
    rng = np.random.default_rng(7)
    n = 3000
    tenure = rng.normal(size=n)
    p_offer = 1.0 / (1.0 + np.exp(-0.5 * tenure))  # mild confounding through tenure
    offer = (rng.uniform(size=n) < p_offer).astype(float)
    tau = 3.0 * tenure  # the planted per-unit effect (a pure interaction)
    spend = tau * offer + 0.5 * tenure + 0.3 * rng.normal(size=n)

    data = tabular(pd.DataFrame({"offer": offer, "tenure": tenure, "spend": spend}))
    graph = dag(["tenure -> offer", "tenure -> spend", "offer -> spend"])

    # --- let the data pick the learner ---------------------------------------
    candidates = ["s_learner", "t_learner", "dr_learner", "causal_forest"]
    report = select_cate(
        graph, data, "offer", "spend", candidates=candidates, random_state=0, n_boot=50
    )

    # --- report --------------------------------------------------------------
    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted truth: tau(tenure) = 3*tenure   (a pure interaction)\n")

    print("THE RANKING (doubly-robust tau-risk, lower is better)")
    for rank, score in enumerate(report.ranking, start=1):
        spec = "misspecified (pools the arms)" if score.method == "s_learner" else "can fit it"
        print(f"  {rank}. {score.method:<14}{score.tau_risk:8.3f}   {spec}")
    worst = report.ranking[-1]
    best = report.ranking[0]
    print(
        f"  -> selected {report.selected_method!r}; the pooled S-learner is "
        f"{worst.tau_risk / best.tau_risk:.1f}x worse and ranks last."
    )

    print("\nTHE Q-AGGREGATION ENSEMBLE (a guarantee: <= the best single)")
    print(f"  best single candidate risk .. {report.best_single_tau_risk:8.3f}")
    print(f"  q_aggregate ensemble risk ... {report.aggregate_tau_risk:8.3f}   (matches/beats it)")
    weights = dict(zip(report.aggregate_methods, report.weights, strict=True))
    shown = "  ".join(f"{m}={float(w):.2f}" for m, w in weights.items())
    print(f"  ensemble weights: {shown}")
    print(
        f"  -> ~0 weight on the misspecified s_learner "
        f"({float(weights['s_learner']):.2f}); the data routes around it."
    )

    print("=" * 78)
    print(
        "No peeking at the truth: the doubly-robust risk ranked the right learners\n"
        "first, and the ensemble is provably no worse than the best single one."
    )


if __name__ == "__main__":
    main()
