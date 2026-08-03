# SPDX-License-Identifier: Apache-2.0
"""Counterfactual fairness: would the decision change if this person's group flipped?

Comparing average outcomes across groups cannot tell you whether a decision is fair,
because the groups differ for legitimate reasons too. Counterfactual fairness (Kusner
et al., 2017) asks the sharper question, one individual at a time: hold everything
about this person fixed, flip only their protected attribute, and see if the decision
moves. If it does, the protected attribute is doing work it should not.

The SCM is built for exactly this. It is a twin-network query: abduct the applicant's
own noise, flip the attribute on the action branch, read the new score. And because it
is a structural model, it can split the group gap into the part that flows through a
legitimate channel (here, qualification) and the part that runs *directly* from the
attribute to the score (the illegitimate part). You pin qualification to its factual
value with a second ``do`` and the legitimate path is severed, leaving only the direct
effect.

The direct effect is the quantity a counterfactually fair model has to remove. Here it
is about 0.5 of score (the same for every applicant, since the mechanism is linear; a
nonlinear score would give a per-person direct effect), on top of a legitimate
through-qualification gap of about 0.8.

Run with: uv run python examples/scm/05_counterfactual_fairness.py
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

    # A = protected group (binary). Q = qualification (a legitimate, resolving path).
    # Score gets a legitimate 1.0*Q and an ILLEGITIMATE 0.5*A direct path.
    rng = np.random.default_rng(7)
    n = 3000
    a = rng.binomial(1, 0.5, n).astype(int)
    q = 0.8 * a + rng.normal(0, 0.7, n)
    score = 1.0 * q + 0.5 * a + rng.normal(0, 0.4, n)

    data = tabular({"A": a, "Q": q, "Score": score}, types={"A": "categorical"})
    graph = dag(["A -> Q", "A -> Score", "Q -> Score"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=7).fit()

    threshold = float(np.quantile(score, 0.6))  # approve the top 40%

    def cf(unit: object, do: dict[str, float]) -> float:
        res = scm.counterfactual(unit, do, "Score", use_draws=False)
        assert isinstance(res, SCMResult)
        return res.mean()

    # cohort-level total disparity: E[Score | do(A=1)] - E[Score | do(A=0)].
    tot_hi = scm.interventional("Score", {"A": 1})
    tot_lo = scm.interventional("Score", {"A": 0})
    assert isinstance(tot_hi, SCMResult) and isinstance(tot_lo, SCMResult)
    total_disparity = tot_hi.mean() - tot_lo.mean()

    print(__doc__.strip().splitlines()[0])
    print("=" * 76)
    print(f"Cohort total group disparity in score:  {total_disparity:+.3f}")
    print("  splits into a legitimate through-qualification part and a direct part.\n")

    # per-applicant counterfactual fairness audit on a few representative people.
    picks = [int(np.where(a == 0)[0][0]), int(np.where(a == 1)[0][0])]
    # add a borderline applicant sitting just under the line.
    borderline = int(np.argmin(np.abs(score - threshold)))
    picks.append(borderline)

    print(
        f"{'applicant':<12}{'group':>6}{'score':>8}"
        f"{'flip group':>12}{'direct part':>13}{'decision':>12}"
    )
    print("-" * 76)
    directs = []
    for i in picks:
        unit = data.take([i])
        q_i = float(q[i])
        # total individual effect of the flip: change A, let Q recompute.
        cf_flip = cf(unit, {"A": 1 - int(a[i])})
        # direct-only effect: change A but PIN Q to this person's factual value.
        cf_direct_to = cf(unit, {"A": 1 - int(a[i]), "Q": q_i})
        cf_direct_from = cf(unit, {"A": int(a[i]), "Q": q_i})
        direct = cf_direct_to - cf_direct_from
        directs.append(abs(direct))
        # does the approve/deny decision flip under the counterfactual group?
        flips = (float(score[i]) >= threshold) != (cf_flip >= threshold)
        print(
            f"#{i:<11}{int(a[i]):>6}{float(score[i]):>8.2f}{cf_flip:>12.2f}{direct:>+13.3f}"
            f"{('flips' if flips else 'stable'):>12}"
        )
    print("=" * 76)
    mean_direct = float(np.mean(directs))
    print(
        f"The direct effect of the protected attribute is about {mean_direct:.2f} of score for\n"
        f"every applicant, qualification held fixed. That is the illegitimate part: a\n"
        f"counterfactually fair model is one whose decision does not move when only the\n"
        f"group flips and qualification stays put."
    )


if __name__ == "__main__":
    main()
