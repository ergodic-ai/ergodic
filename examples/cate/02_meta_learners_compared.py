# SPDX-License-Identifier: Apache-2.0
"""Five learners, one truth: S, T, X, DR, R, and DML on the same world.

A meta-learner turns any regression model into a CATE estimator. There is more than
one way to do it, and they differ in how robust they are and in what they can see.
This runs six of them on ONE planted heterogeneous world, where the true per-unit
effect is ``tau(z) = 1 + z`` (average 1.0, rising with the modifier z), and prints
each one's recovered CATE against that truth.

The lineup:

  S-learner    one model with treatment as a feature.
  T-learner    a separate model per arm, differenced.
  X-learner    the T-learner's cross-imputed refinement.
  DR-learner   regress the doubly-robust score on the covariates (the default).
  R-learner    Robinson residual-on-residual, the R-loss.
  DML          double machine learning, a single final coefficient.

The lesson in the numbers: with a plain linear learner in the slots, the S-learner
and DML return a CONSTANT effect (they fit no treatment-by-covariate interaction, so
they recover the average and call everyone the same), while T / X / DR / R all
recover the ``1 + z`` gradient. None is wrong about the *average*; they differ on
whether they see the heterogeneity at all. Pick the method with ``method=`` on the
``cate()`` facade.

Run with: uv run python examples/cate/02_meta_learners_compared.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import cate, dag, tabular


def main() -> None:
    # --- one planted heterogeneous world: tau(z) = 1 + z ---------------------
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(0, 1, n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + 1.0 * z) * a + 1.5 * z + rng.normal(size=n)

    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    truth = 1.0 + z  # the planted per-unit effect

    methods = ["s_learner", "t_learner", "x_learner", "dr_learner", "r_learner", "dml"]

    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted truth: tau(z) = 1 + z   (population ATE 1.000, effect rises with z)\n")
    print(f"  {'method':<12} {'ATE':>7} {'corr':>7} {'sd(CATE)':>9}   reads heterogeneity?")
    print("  " + "-" * 72)
    for m in methods:
        res = cate(graph, data, "A", "Y", method=m, random_state=0)
        corr = float(np.corrcoef(res.values, truth)[0, 1])
        sd = float(np.std(res.values))
        sees = "yes" if sd > 0.1 else "no  (constant -> homogeneous)"
        print(f"  {m:<12} {res.ate().value:7.3f} {corr:14.3f} {sd:9.3f}   {sees}")
    print("=" * 78)
    print(
        "All six recover the average ~1.0. But with a linear learner the S-learner and\n"
        "DML report ONE effect for everyone (sd 0), while T/X/DR/R track tau(z)=1+z\n"
        "(corr ~1.0). Same truth, different ability to see who it helps."
    )


if __name__ == "__main__":
    main()
