# SPDX-License-Identifier: Apache-2.0
"""Where, and for whom: BLP, GATES, and CLAN read the heterogeneity.

Once the calibration test says heterogeneity is real, three tools say where it lives
and who is in the high-effect group. All three read the SAME doubly-robust scores, so
their inference is honest (the CATE predictions are only a fixed ranking, never the
object of inference).

  ``.blp()``    Best Linear Projection: regress the effect on the covariates. Each
                slope is "how much does this covariate move the effect," with a valid
                CI. The intercept is the effect at covariates all zero (the ATE
                only when the covariates are centred).
  ``.gates()``  Sorted Group ATEs: bin units by predicted effect and report the true
                effect per bin, plus the top-minus-bottom contrast. "Does the effect
                rise across the predicted groups."
  ``.clan()``   Classification Analysis: compare covariate averages in the most- vs
                least-affected group. "Who is in the high-effect group."

The world plants TWO covariates but only one real modifier. The discount lift rises
with ``tenure`` (``tau = 1 + tenure``); ``age`` confounds treatment and spend but does
NOT modify the effect. A good characterization must finger tenure and clear age.

Run with: uv run python examples/cate/04_where_is_the_effect_higher.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import cate, dag, tabular


def main() -> None:
    # --- two covariates, ONE real modifier (tenure); age is inert ------------
    rng = np.random.default_rng(0)
    n = 5000
    tenure = rng.normal(0, 1, n)  # the real effect modifier
    age = rng.normal(0, 1, n)  # a confounder, but not a modifier
    discount = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.6 * tenure + 0.4 * age)))).astype(
        float
    )
    spend = (1.0 + 1.0 * tenure) * discount + 1.0 * tenure + 0.8 * age + rng.normal(size=n)

    data = tabular(
        pd.DataFrame({"discount": discount, "tenure": tenure, "age": age, "spend": spend})
    )
    graph = dag(
        [
            "tenure -> discount",
            "tenure -> spend",
            "age -> discount",
            "age -> spend",
            "discount -> spend",
        ]
    )
    h = cate(graph, data, "discount", "spend", random_state=0).explain()

    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted: lift rises with tenure (tau = 1 + tenure); age confounds but is inert.\n")

    print("BLP -- which covariate moves the effect")
    for name, row in h.blp().frame().items():
        if name == "(intercept)":
            tag = ""
        else:
            tag = "  <- modifier" if row["ci_low"] > 0.0 else "  (inert)"
        print(
            f"  {name:<12} slope {row['coef']:6.3f}"
            f"   90% CI [{row['ci_low']:6.3f}, {row['ci_high']:6.3f}]{tag}"
        )

    g = h.gates(n_groups=5)
    lo, hi = g.interval()
    print("\nGATES -- does the true effect rise across predicted groups")
    print("  group effects (low -> high): " + "  ".join(f"{x:5.2f}" for x in g.group_effect))
    print(f"  monotone increasing? {g.is_monotone()}")
    print(f"  top-minus-bottom contrast: {g.difference:5.2f}   90% CI [{lo:.2f}, {hi:.2f}]")

    print("\nCLAN -- who is in the high-effect group")
    for name, row in h.clan().frame().items():
        nonzero = row["ci_low"] > 0.0 or row["ci_high"] < 0.0
        excl = "excludes 0  <- profiles it" if nonzero else "covers 0"
        print(
            f"  {name:<12} high-group mean {row['high_mean']:6.3f}"
            f"   low-group mean {row['low_mean']:6.3f}"
            f"   diff {row['difference']:6.3f}  ({excl})"
        )

    print("=" * 78)
    print(
        "All three agree: tenure drives the effect (slope ~1, GATES climbs, the high\n"
        "group is high-tenure), and age is correctly cleared. Where, and for whom."
    )


if __name__ == "__main__":
    main()
