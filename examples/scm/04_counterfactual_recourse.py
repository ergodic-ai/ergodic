# SPDX-License-Identifier: Apache-2.0
"""Algorithmic recourse: the minimal real change that would flip a decision.

A loan applicant is denied. The honest, actionable question is not "which feature
explains the score" (the SHAP question) but "what would this person have to change,
and by how little, to be approved?" That is a counterfactual about one
individual, and it has to respect how the world works: changing one feature pushes its
downstream features around too.

The world: income drives savings (up) and debt ratio (down), and all three feed a
credit score. We take a denied applicant and ask the SCM for the smallest change to
each lever that lifts their counterfactual score over the approval line, holding their
own idiosyncrasies fixed (the twin network).

The non-obvious part is the ranking of the levers. Income looks like a weak lever if
you read the coefficient a scorecard fits on it with the other features held fixed,
which is exactly what a feature-attribution recourse tool reads. But intervening on
income also raises savings and cuts debt, and those gains flow into the score too. The SCM prices
in the whole cascade, so income turns out to be the *most* efficient single change,
roughly half what its own coefficient would suggest. Recourse that ignores the causal
graph asks people to change more than they need to.

Run with: uv run python examples/scm/04_counterfactual_recourse.py
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

    rng = np.random.default_rng(4)
    n = 3000
    income = rng.normal(0, 1, n)
    savings = 0.5 * income + rng.normal(0, 0.5, n)
    debt = -0.3 * income + rng.normal(0, 0.5, n)
    score = 0.4 * income + 0.6 * savings - 0.8 * debt + rng.normal(0, 0.3, n)

    data = tabular({"Income": income, "Savings": savings, "Debt": debt, "Score": score})
    graph = dag(
        [
            "Income -> Savings",
            "Income -> Debt",
            "Income -> Score",
            "Savings -> Score",
            "Debt -> Score",
        ]
    )
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=4).fit()

    # approval line at the 55th percentile; take a clearly-denied applicant.
    threshold = float(np.quantile(score, 0.55))
    denied = np.where(score < threshold)[0]
    idx = int(denied[np.argsort(score[denied])[len(denied) // 2]])  # a middling denial
    unit = data.take([idx])
    score_obs = float(score[idx])
    deficit = threshold - score_obs  # how far below the line, in score points

    def cf_score(lever: str, new_value: float) -> float:
        res = scm.counterfactual(unit, {lever: new_value}, "Score", use_draws=False)
        assert isinstance(res, SCMResult)
        return res.mean()

    # the SCM's effective per-unit slope of Score w.r.t. each lever (one-unit nudge),
    # which already includes every downstream cascade through the graph.
    levers = {
        "Income": float(income[idx]),
        "Savings": float(savings[idx]),
        "Debt": float(debt[idx]),
    }
    base = {lev: cf_score(lev, val) for lev, val in levers.items()}  # == factual score
    slope = {lev: cf_score(lev, val + 1.0) - base[lev] for lev, val in levers.items()}

    # the naive slope reads the score equation's own coefficient, blind to the cascade.
    naive_coef = {"Income": 0.4, "Savings": 0.6, "Debt": -0.8}

    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print(f"Applicant #{idx}: score {score_obs:.2f}, line {threshold:.2f}, short {deficit:.2f}\n")
    print(f"{'lever':<10}{'causal slope':>14}{'change needed':>16}{'naive change':>16}")
    print("-" * 78)
    for lev in ("Income", "Savings", "Debt"):
        need_causal = deficit / slope[lev]
        need_naive = deficit / naive_coef[lev]
        # verify the causal recourse clears the line (linear, so exact).
        cleared = cf_score(lev, levers[lev] + need_causal)
        assert abs(cleared - threshold) < 0.02, f"{lev}: {cleared} vs {threshold}"
        print(f"{lev:<10}{slope[lev]:>14.3f}{need_causal:>+16.3f}{need_naive:>+16.3f}")
    print("=" * 78)
    eff = deficit / slope["Income"]
    naive_eff = deficit / naive_coef["Income"]
    ratio = naive_eff / eff
    print(
        f"Income is the cheapest lever: a {eff:+.2f} change clears the line, because the\n"
        f"gain cascades into savings and debt too. Reading the scorecard coefficient (0.4)\n"
        f"would tell them to raise income by {naive_eff:+.2f}, about {ratio:.1f}x more than needed."
    )


if __name__ == "__main__":
    main()
