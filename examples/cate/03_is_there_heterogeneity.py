# SPDX-License-Identifier: Apache-2.0
"""Prove it's there before you explain it: the omnibus heterogeneity test.

A CATE estimator always returns a per-unit vector, even when the true effect is the
same for everyone. The vector will have spread just from estimation noise, and it is
tempting to read that spread as real heterogeneity and start "explaining" it. That is
the road to a story about noise.

The discipline is to test first. ``CATEResult.explain().calibration_test()`` is the
GRF omnibus test: it regresses the doubly-robust score on the predicted effect and
asks whether the *differential* part carries real signal. Its one-sided p-value is an
honest test for "is there usable heterogeneity at all." The Best Linear Projection
(``.blp()``) is the companion read: each covariate slope is a valid (HC3) test of
whether that covariate moves the effect.

This runs both on two worlds that are identical except for one planted constant:

  HOMOGENEOUS:   tau(z) = 1 + 0*z   the effect is 1.0 for everyone.
  HETEROGENEOUS: tau(z) = 1 + 1*z   the effect rises with z.

The test does not reject on the null (slope CI covers zero, p large) and rejects
hard on the real gradient. Honest non-detection is the feature.

Run with: uv run python examples/cate/03_is_there_heterogeneity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import cate, dag, tabular


def _world(b: float, seed: int) -> tuple:
    """tau(z) = 1 + b*z; b=0 is the homogeneous null, b>0 is real heterogeneity."""
    rng = np.random.default_rng(seed)
    n = 4000
    z = rng.normal(0, 1, n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + b * z) * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    return data, dag(["Z -> A", "Z -> Y", "A -> Y"])


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print("=" * 76)

    worlds = [("HOMOGENEOUS  tau = 1 + 0*z", 0.0, 3), ("HETEROGENEOUS tau = 1 + 1*z", 1.0, 0)]
    for label, b, seed in worlds:
        data, graph = _world(b, seed)
        h = cate(graph, data, "A", "Y", random_state=seed).explain()
        cal = h.calibration_test()
        z_slope = h.blp().frame()["Z"]

        print(f"\n{label}")
        print(
            f"  calibration test:  differential coef {cal.differential_coefficient:6.3f}"
            f"   one-sided p {cal.heterogeneity_pvalue:.4f}"
        )
        verdict = "REJECTS 'no heterogeneity'" if cal.rejects() else "does not reject (honest)"
        print(f"                     -> {verdict}")
        excludes = "excludes 0" if z_slope["ci_low"] > 0.0 else "covers 0"
        print(
            f"  BLP slope on Z:    {z_slope['coef']:6.3f}"
            f"   90% CI [{z_slope['ci_low']:6.3f}, {z_slope['ci_high']:6.3f}]  ({excludes})"
        )

    print("\n" + "=" * 76)
    print(
        "On the null the test does NOT reject and the slope CI covers zero, even though\n"
        "the raw CATE vector has noise spread. On the real gradient both fire. Test for\n"
        "heterogeneity before you tell a story about it."
    )


if __name__ == "__main__":
    main()
