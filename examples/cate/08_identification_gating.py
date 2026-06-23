# SPDX-License-Identifier: Apache-2.0
"""cate() only answers when the graph licenses it: the honesty layer.

A per-unit effect is still a causal effect, so it inherits the same precondition as
any causal estimate: the effect must be identified from the graph. ``cate()`` is
gated exactly like ``estimate_effect``. It refuses, with a pointer, rather than
returning a confident-looking number off an unlicensed query.

This walks the four cases:

  1. ADJUSTMENT-IDENTIFIED   a confounder you measured, with a valid back-door set.
                             ``cate()`` runs and returns the per-unit effects.
  2. NON-IDENTIFIED (a bow)  treatment and outcome share an unmeasured confounder and
                             nothing blocks the back door. No adjustment set exists, so
                             ``cate()`` raises "not identifiable."
  3. FRONT-DOOR FAMILY       the effect IS identified (through a mediator), but not by
                             adjustment. There is no per-unit meta-learner for the
                             front door yet, so ``cate()`` raises and says so. (The
                             population effect is available via ``estimate_effect`` /
                             the SCM's latent-district fit.)
  4. NEURAL / FOUNDATION     ``method="dragonnet"`` / ``"causal_pfn"`` are reserved
                             seams; they raise a clear "needs the optional extra."

The point: the refusals are the feature. A CATE you cannot identify is a CATE you
should not report.

Run with: uv run python examples/cate/08_identification_gating.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ergodic import cate, dag, tabular


def main() -> None:
    rng = np.random.default_rng(0)
    print(__doc__.strip().splitlines()[0])
    print("=" * 78)

    # --- 1) adjustment-identified: cate() runs -------------------------------
    n = 2000
    z = rng.normal(0, 1, n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.7 * z))).astype(float)
    y = (1.0 + z) * a + 1.5 * z + rng.normal(size=n)
    data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])
    res = cate(graph, data, "A", "Y", random_state=0)
    print("\n1. ADJUSTMENT-IDENTIFIED  (Z measured, back-door valid)")
    print(f"   -> cate() runs.  ate = {res.ate().value:.3f}, {len(res)} per-unit effects")

    # --- 2) a bow: not identifiable ------------------------------------------
    m = 500
    u = rng.normal(size=m)
    a2 = (rng.uniform(size=m) < 0.5).astype(float)
    y2 = a2 + u + rng.normal(size=m)
    data2 = tabular(pd.DataFrame({"A": a2, "Y": y2}))
    graph2 = dag(["A -> Y", "U -> A", "U -> Y"], latent=["U"])
    print("\n2. NON-IDENTIFIED  (A <-> Y via an unmeasured U, no adjustment set)")
    try:
        cate(graph2, data2, "A", "Y")
    except ValueError as exc:
        print(f"   -> refuses: {str(exc).split(':')[0]}")

    # --- 3) front-door family: identified, but not by adjustment -------------
    k = 800
    u3 = rng.normal(size=k)
    a3 = (rng.uniform(size=k) < 1.0 / (1.0 + np.exp(-u3))).astype(int)
    md = (rng.uniform(size=k) < 1.0 / (1.0 + np.exp(-(a3)))).astype(int)
    y3 = (rng.uniform(size=k) < 1.0 / (1.0 + np.exp(-(md + u3)))).astype(int)
    data3 = tabular(
        pd.DataFrame({"A": a3, "M": md, "Y": y3}),
        types={"A": "categorical", "M": "categorical", "Y": "categorical"},
    )
    graph3 = dag(["A -> M", "M -> Y", "U -> A", "U -> Y"], latent=["U"])
    print("\n3. FRONT-DOOR FAMILY  (identified through mediator M, not by adjustment)")
    try:
        cate(graph3, data3, "A", "Y")
    except ValueError:
        print("   -> refuses: CATE needs adjustment-identification (no per-unit front-door yet)")

    # --- 4) the neural / foundation seams ------------------------------------
    print("\n4. RESERVED SEAMS  (neural / foundation methods)")
    for method, extra in [("dragonnet", "neural"), ("causal_pfn", "foundation")]:
        try:
            cate(graph, data, "A", "Y", method=method)
        except NotImplementedError:
            print(f"   -> method={method!r:13} raises: needs the optional {extra!r} extra")

    print("=" * 78)
    print(
        "cate() answers in case 1 and refuses in 2-4, each with a pointer. Identification\n"
        "is the gate; the per-unit surface does not lower the bar."
    )


if __name__ == "__main__":
    main()
