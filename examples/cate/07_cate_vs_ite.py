# SPDX-License-Identifier: Apache-2.0
"""The average for people like you, vs your own counterfactual: CATE vs ITE.

These two per-unit quantities look the same and are not. They answer different
questions and come from different pillars, yet they share one result surface.

  CATE (the conditional average treatment effect): "for people with covariates like
        yours, what is the average effect?" An average over the unobserved noise.
        A meta-learner's ``cate()`` estimates it from data (the inference pillar).

  ITE  (the individual treatment effect): "for YOU specifically, what would have
        happened in the other world?" The same person's two counterfactual worlds,
        sharing that person's abducted noise. ``SCM.ite()`` reads it off the fitted
        structural model's twin network (the SCM pillar).

This builds one world with a binary treatment, a binary outcome, and a real
interaction (the treatment helps more as z rises). Both quantities are computed on
it. The CATE comes out smooth: each unit's *probability* lift, varying continuously
with z. The ITE comes out as each unit's actual counterfactual *flip* -- in
{-1, 0, 1}, because a binary outcome either changes or it does not. Their averages
agree (both estimate the ATE), but they are different objects: a smooth conditional
expectation versus a discrete individual event.

Both are a ``PerUnitEffect``, so ``.values``, ``.ate()``, ``.interval()``, and
``np.mean(...)`` work identically on each. And ``ITEResult.explain()`` raises with a
pointer: the Tier-B characterization is defined over the CATE's doubly-robust scores,
not the structural ITE.

Needs the ``scm`` extra (PyMC). Pin BLAS threads when you run it:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    uv run python examples/cate/07_cate_vs_ite.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ergodic import CATEResult, ITEResult, PerUnitEffect, cate, dag, tabular
from ergodic.scm import SCM


def _quiet_pymc() -> None:
    for name in ("pymc", "pymc.sampling"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_pymc()

    # --- binary treatment, binary outcome, real interaction (effect rises in z) ---
    rng = np.random.default_rng(3)
    n = 3000
    z = rng.normal(0, 1, n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(int)
    logit = -0.2 + 1.5 * a + 0.5 * z + 1.2 * a * z
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    data = tabular(
        pd.DataFrame({"A": a, "Z": z, "Y": y}),
        types={"A": "categorical", "Y": "categorical"},
    )
    graph = dag(["Z -> A", "Z -> Y", "A -> Y"])

    # --- CATE: the conditional average, from a meta-learner -------------------
    cate_res = cate(graph, data, "A", "Y", method="dr_learner", random_state=3, treated=1)
    assert isinstance(cate_res, CATEResult)

    # --- ITE: the individual counterfactual flip, from the fitted SCM ---------
    scm = SCM(graph, data, draws=300, tune=300, chains=2, random_state=3).fit()
    ite_res = scm.ite(data, "A", "Y", levels=(0, 1))
    assert isinstance(ite_res, ITEResult)

    cate_vals = cate_res.values
    ite_vals = ite_res.values
    flips = sorted({float(v) for v in np.round(ite_vals, 6)})

    print(__doc__.strip().splitlines()[0])
    print("=" * 78)
    print("Binary treatment + binary outcome + interaction (effect rises with z).\n")
    print("CATE (conditional average over noise) -- a smooth probability lift")
    print(
        f"  ate {cate_res.ate().value:6.3f}   per-unit sd {float(np.std(cate_vals)):.3f}"
        f"   correlation with z {float(np.corrcoef(cate_vals, z)[0, 1]):.3f}"
    )
    print(f"  values are continuous: e.g. {np.round(cate_vals[:4], 3).tolist()}\n")
    print("ITE (this person's own counterfactual) -- a discrete flip")
    print(f"  mean {float(np.mean(ite_vals)):6.3f}   per-unit sd {float(np.std(ite_vals)):.3f}")
    print(f"  values are individual events in {{-1, 0, 1}}: distinct values seen = {flips}")
    print(f"  (its factual outcome rate was {float(np.mean(ite_res.factual)):.3f})\n")
    print("SAME SURFACE, DIFFERENT OBJECTS")
    both = isinstance(cate_res, PerUnitEffect) and isinstance(ite_res, PerUnitEffect)
    print(f"  both subclass PerUnitEffect: {both}")
    print(
        f"  the averages agree (both estimate the ATE): "
        f"CATE {cate_res.ate().value:.3f}  vs  ITE {float(np.mean(ite_vals)):.3f}"
    )
    try:
        ite_res.explain()
    except NotImplementedError:
        print("  ITEResult.explain() raises -> the Tier-B tools need the CATE's DR scores")
    print("=" * 78)
    print(
        "The CATE is the smooth average for people like you; the ITE is your own\n"
        "yes/no counterfactual. They average to the same ATE and live behind one\n"
        "result type, but they answer two different questions."
    )


if __name__ == "__main__":
    main()
