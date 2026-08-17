# SPDX-License-Identifier: Apache-2.0
"""Per-unit effects with honest, native error bars, and what drives them.

The meta-learners borrow a regressor and bootstrap it to get a band. A causal
forest is different: the forest *is* the CATE learner, and it carries its own
per-unit standard errors. Those come from the infinitesimal jackknife (the
bootstrap-of-little-bags), which is valid because the forest is *honest* -- each
tree splits on one half of its subsample and fills its leaves on the other, so a
unit's outcome never both places a split and is averaged into the effect there.
The result is an analytic, per-unit band, not a bootstrap one.

This builds a retention world with a KNOWN heterogeneous truth. A discount lifts
spend, and the lift rises with tenure: ``tau(tenure) = 1 + 1.5*tenure``, so the
population average is about 1.0 but the effect is strongly negative for new
customers and strongly positive for loyal ones. A second covariate, ``region``, is
a decoy: it shifts baseline spend but does NOT modify the effect. Tenure also drives
who gets the offer, so the naive read is confounded; the graph says to adjust.

Three things to watch:
  - ``cate(..., method="causal_forest")`` recovers ``tau(tenure)`` (corr ~0.99).
  - the forest's per-unit band is NATIVE and analytic (``kind="analytic"``), and its
    width VARIES across units -- contrast the bootstrap band a meta-learner reports.
  - ``variable_importance()`` fingers tenure as the driver and clears the decoy.

Run with: uv run python examples/cate/10_causal_forest.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import CausalForest, cate, dag, identify_effect, tabular


def main() -> None:
    # --- a retention world: tau(tenure) = 1 + 1.5*tenure, region is a decoy ---
    rng = np.random.default_rng(0)
    n = 4000
    tenure = rng.normal(size=n)  # the true modifier (standardized months)
    region = rng.normal(size=n)  # a decoy: shifts baseline, not the effect
    p_offer = 1.0 / (1.0 + np.exp(-0.8 * tenure))  # tenure drives the offer (confounding)
    offer = (rng.uniform(size=n) < p_offer).astype(float)
    tau = 1.0 + 1.5 * tenure  # the planted per-unit effect
    spend = 1.5 * tenure + 0.7 * region + tau * offer + rng.normal(size=n)

    data = tabular(
        pd.DataFrame({"offer": offer, "tenure": tenure, "region": region, "spend": spend})
    )
    graph = dag(["tenure -> offer", "tenure -> spend", "region -> spend", "offer -> spend"])
    truth = 1.0 + 1.5 * tenure

    # --- the honest causal forest --------------------------------------------
    forest_res = cate(
        graph,
        data,
        "offer",
        "spend",
        method="causal_forest",
        n_trees=400,
        min_leaf=15,
        random_state=0,
    )
    f_lo, f_hi = forest_res.interval(0.9)
    f_cov = float(np.mean((f_lo <= truth) & (truth <= f_hi)))
    assert forest_res.std_error is not None

    # --- a bootstrap-band meta-learner, for contrast -------------------------
    boot_res = cate(graph, data, "offer", "spend", method="t_learner", random_state=0, n_boot=200)
    b_lo, b_hi = boot_res.interval(0.9)

    # --- variable importance: which covariate drives the effect --------------
    fitted = CausalForest(n_trees=400, min_leaf=15, random_state=0).fit(
        identify_effect(graph, "offer", "spend"), data
    )
    vim = fitted.variable_importance()
    importance = vim.to_dict()

    # --- report --------------------------------------------------------------
    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Planted truth: tau(tenure) = 1 + 1.5*tenure   (population ATE ~ 1.00)\n")

    print("RECOVERING THE PER-UNIT EFFECT (the causal forest)")
    print(f"  forest ATE ................ {forest_res.ate().value:6.3f}   (planted ~ 1.00)")
    print(
        f"  corr(tau_hat, true tau) ... {float(np.corrcoef(forest_res.values, truth)[0, 1]):6.3f}"
    )

    print("\nHONEST, NATIVE ERROR BARS (infinitesimal jackknife, not bootstrap)")
    print(f"  forest band kind .......... {forest_res.kind!r}   (per-unit, analytic)")
    print(
        f"  forest 90% band: widths run {float((f_hi - f_lo).min()):.3f} -> "
        f"{float((f_hi - f_lo).max()):.3f}  (they VARY per unit; coverage {f_cov:.2f})"
    )
    print(
        f"  contrast a meta-learner ({boot_res.source}): band kind {boot_res.kind!r}, "
        f"width ~{float(np.mean(b_hi - b_lo)):.3f}"
    )

    print("\nWHAT DRIVES THE EFFECT (depth-weighted split frequency)")
    for name, value in sorted(importance.items(), key=lambda kv: -kv[1]):
        tag = "the modifier" if name == "tenure" else "decoy (cleared)"
        print(f"  variable_importance  {name:<8}{value:6.3f}   {tag}")
    print(f"  -> top driver: {vim.top(1)[0][0]!r}  (tenure outweighs the region decoy)")

    print("=" * 78)
    print(
        "The forest recovers tau(tenure), carries its own honest per-unit band whose\n"
        "width changes from customer to customer, and names tenure as the driver."
    )


if __name__ == "__main__":
    main()
