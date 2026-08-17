# SPDX-License-Identifier: Apache-2.0
"""The three rungs of the causal ladder on one fitted model: see, do, imagine.

A structural causal model is the only object that answers all three causal
questions from a single fit. This script fits one and asks all three of a simple
confounded world Z -> X, Z -> Y, X -> Y, where the true effect of X on Y is 1.5
but Z (a common cause) inflates the *observed* association to about 2.9.

  See     (association):   what the data alone reports, E[Y | X]. Confounded: ~2.9.
  Do      (intervention):  E[Y | do(X=1)] - E[Y | do(X=0)], the back-door-adjusted
                           causal effect. The SCM severs Z's influence: ~1.5.
  Imagine (counterfactual): for one specific unit, "what would this person's Y have
                           been had their X been one unit higher, holding everything
                           else about them fixed?" The twin network answers it
                           exactly.

The point: see, do, and imagine are three different numbers, and only the model
that carries the structural noise can give you the third one (a statement about an
individual, not a population).

The counterfactual delta here equals the structural slope exactly because the
mechanism is linear and additive, so the abducted noise cancels. A nonlinear mechanism
would still answer the same question, just without that clean equality.

Run with: uv run python examples/scm/01_three_rungs.py
"""

from __future__ import annotations

import logging

import numpy as np

from ergodic import dag, tabular
from ergodic.scm import SCM, SCMResult


def _quiet_pymc() -> None:
    """Silence PyMC's per-chain sampling chatter so the narrative reads cleanly."""
    for name in ("pymc", "pymc.sampling"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_pymc()

    # --- a confounded world with a KNOWN truth --------------------------------
    a, b, c = 0.8, 1.5, 2.0  # Z->X, X->Y (the truth we want back), Z->Y
    rng = np.random.default_rng(7)
    n = 3000
    z = rng.normal(0, 1, n)
    x = a * z + rng.normal(0, 0.7, n)
    y = b * x + c * z + rng.normal(0, 0.5, n)

    data = tabular({"Z": z, "X": x, "Y": y})
    graph = dag(["Z -> X", "Z -> Y", "X -> Y"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=7).fit()

    # --- RUNG 1: SEE. The association the data reports, confounded by Z. -------
    observational = float(np.cov(x, y)[0, 1] / np.var(x))

    # --- RUNG 2: DO. The intervention severs the back-door through Z. ----------
    hi = scm.interventional("Y", {"X": 1.0})
    lo = scm.interventional("Y", {"X": 0.0})
    assert isinstance(hi, SCMResult) and isinstance(lo, SCMResult)
    do_contrast = hi.mean() - lo.mean()

    ate = scm.ate("X", "Y")  # routed through identification first
    assert isinstance(ate, SCMResult)
    lo_ci, hi_ci = ate.interval(0.95)

    # --- RUNG 3: IMAGINE. One unit's but-for outcome, noise held fixed. --------
    u = 17
    unit = data.take([u])
    x_obs, y_obs = float(x[u]), float(y[u])
    cf = scm.counterfactual(unit, {"X": x_obs + 1.0}, "Y", use_draws=False)
    assert isinstance(cf, SCMResult)
    y_cf = cf.mean()

    # --- report ----------------------------------------------------------------
    print(__doc__.strip().splitlines()[0])
    print("=" * 72)
    print(f"True structural effect of X on Y:           {b:.3f}\n")
    print("RUNG 1  SEE      E[Y|X] slope (data alone)")
    print(f"                 association ............... {observational:6.3f}   (confounded by Z)")
    print("RUNG 2  DO       E[Y|do(X=1)] - E[Y|do(X=0)]")
    print(f"                 interventional contrast ... {do_contrast:6.3f}")
    ci = f"[{lo_ci:.2f}, {hi_ci:.2f}]"
    print(f"                 ate(), identified ......... {ate.mean():6.3f}   95% CrI {ci}")
    print(f"RUNG 3  IMAGINE  unit #{u}: X={x_obs:.2f}, Y={y_obs:.2f}")
    print(f"                 had X been {x_obs + 1.0:.2f} instead .... Y* = {y_cf:6.3f}")
    print(f"                 individual but-for delta .. {y_cf - y_obs:6.3f}   (structural slope)")
    print("=" * 72)
    print(
        "The correlation says ~2.9; the cause is ~1.5; and for this one unit the\n"
        "counterfactual moves Y by exactly the structural slope. One fit, three rungs."
    )


if __name__ == "__main__":
    main()
