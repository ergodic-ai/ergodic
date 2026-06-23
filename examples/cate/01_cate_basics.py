# SPDX-License-Identifier: Apache-2.0
"""From one number to a function: the CATE and its per-unit result surface.

``estimate_effect`` answers "what is the average effect?" with one number. But an
average hides who the effect helps and who it does not. The conditional average
treatment effect (CATE) is the next step up: the effect *as a function of who you
are*, one number per unit.

This builds a retention world with a KNOWN heterogeneous truth. A discount is
offered, and how much it lifts spend depends on customer tenure: the true per-unit
effect is ``tau(tenure) = 1 + tenure``, so the population average is 1.0 but the
effect runs from strongly negative for new customers to strongly positive for loyal
ones. Tenure also drives who gets the discount and baseline spend, so it confounds
the naive read; the graph says to adjust for it.

The point is the result object. ``cate()`` returns a ``CATEResult`` that *is* a
per-unit vector: ``.values`` is one effect per customer, ``.interval()`` is the
band, ``.ate()`` bridges back to the population number, ``.to_frame()`` is the
table, and ``__array__`` makes ``np.mean(result)`` just work. One call, the whole
heterogeneity surface.

Run with: uv run python examples/cate/01_cate_basics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import CATEResult, cate, dag, estimate_effect, tabular


def main() -> None:
    # --- a retention world with a KNOWN heterogeneous truth -------------------
    # tau(tenure) = 1 + tenure: average lift 1.0, but it depends on the customer.
    rng = np.random.default_rng(0)
    n = 4000
    tenure = rng.normal(0, 1, n)  # standardized months as a customer
    # tenure drives who takes the discount (confounding) ...
    discount = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * tenure))).astype(float)
    # ... and the lift it produces, plus a direct effect on spend.
    spend = (1.0 + 1.0 * tenure) * discount + 1.5 * tenure + rng.normal(size=n)

    data = tabular(pd.DataFrame({"discount": discount, "tenure": tenure, "spend": spend}))
    graph = dag(["tenure -> discount", "tenure -> spend", "discount -> spend"])

    # --- one call gives the per-unit effect function -------------------------
    result = cate(graph, data, "discount", "spend", random_state=0)
    assert isinstance(result, CATEResult)

    truth = 1.0 + tenure  # the planted per-unit effect

    # the population number, the old answer, recovered from the per-unit one.
    pop = result.ate()
    direct_ate = estimate_effect(graph, data, "discount", "spend", random_state=0)

    # the per-unit band, and the drop-in array behaviour.
    lo, hi = result.interval(0.9)
    frame = result.to_frame()

    # --- report --------------------------------------------------------------
    print(__doc__.strip().splitlines()[0])
    print("=" * 74)
    print("Planted per-unit truth: tau(tenure) = 1 + tenure   (population ATE 1.000)\n")
    print("ONE NUMBER (the average effect)")
    print(f"  result.ate() ......................... {pop.value:6.3f}   95% band aside")
    print(f"  estimate_effect(), for comparison .... {direct_ate.value:6.3f}\n")
    print("A FUNCTION (one effect per customer)")
    print(f"  result.values: shape {result.values.shape}, runs across customers")
    print(
        f"    min {result.values.min():6.3f}   median {np.median(result.values):6.3f}"
        f"   max {result.values.max():6.3f}"
    )
    corr = float(np.corrcoef(result.values, truth)[0, 1])
    print(f"  correlation with the planted tau(tenure) ... {corr:.3f}")
    print(f"  np.mean(result) works (the __array__ drop-in) .. {float(np.mean(result)):.3f}\n")
    print("PER-UNIT UNCERTAINTY (the band around each)")
    print(f"  result.interval(0.9): lo/hi each shape {lo.shape} / {hi.shape}")
    print("  result.to_frame() (first three customers):")
    for unit, row in frame.head(3).iterrows():
        print(
            f"    customer {unit:>2}:  effect {row['effect']:6.3f}"
            f"   [{row['ci_low']:6.3f}, {row['ci_high']:6.3f}]"
        )
    print("=" * 74)
    print(
        "The average is 1.0, but the discount HURTS new customers and helps loyal\n"
        "ones. One number could not say that; the CATE function does."
    )


if __name__ == "__main__":
    main()
