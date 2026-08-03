# SPDX-License-Identifier: Apache-2.0
"""Identify an effect through an unmeasured confounder, via the front door.

Adjustment needs you to measure the confounders. What do you do when the confounder
is unmeasured and unmeasurable? The classic example: does smoking cause cancer, when
some hidden genotype might drive both the urge to smoke and the cancer risk? You can
never adjust for a variable you cannot see.

The front-door criterion is the way through, when a mediator sits on the causal path
and is itself unconfounded with the outcome. Smoking (X) deposits tar (M), and tar
causes cancer (Y). The hidden genotype (U) confounds X and Y but does not touch the
tar-to-cancer step. That structure identifies the smoking-to-cancer effect from the
two clean legs, even though X and Y are confounded.

This is the SCM's deep moat. It fits the confounded {X, Y} district with an explicit
latent, so the structural tar-to-cancer mechanism fits clean, and ``do(X)`` severs
smoking from the latent. The mediated effect drops out: 1.5 (tar per cigarette) times
0.8 (cancer per unit tar) = 1.2. The naive smoking-cancer regression, blind to the
genotype, reports a much larger and wrong number.

Run with: uv run python examples/scm/10_frontdoor_unmeasured_confounder.py
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

    # X (smoking) -> M (tar) -> Y (cancer); U (genotype) -> X and -> Y, U LATENT.
    b_xm, b_my, a_x, a_y = 1.5, 0.8, 1.3, 1.1
    fd_effect = b_xm * b_my  # the front-door truth: 1.2
    rng = np.random.default_rng(20240616)
    n = 4000
    u = rng.normal(0, 1, n)
    x = a_x * u + rng.normal(0, 0.6, n)
    m = b_xm * x + rng.normal(0, 0.5, n)
    y = b_my * m + a_y * u + rng.normal(0, 0.5, n)

    data = tabular({"X": x, "M": m, "Y": y})
    graph = dag(["X -> M", "M -> Y", "U -> X", "U -> Y"], latent=["U"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=20240616).fit()

    # the wrong answer: regress cancer on smoking, blind to the genotype.
    naive = float(np.cov(x, y)[0, 1] / np.var(x))

    # the right answer: front-door, computed from the latent-district fit.
    ate = scm.ate("X", "Y")
    assert isinstance(ate, SCMResult)
    lo, hi = ate.interval(0.95)
    ci = f"[{lo:.2f}, {hi:.2f}]"

    print(__doc__.strip().splitlines()[0])
    print("=" * 74)
    print(f"True front-door effect (1.5 tar/cig x 0.8 cancer/tar):   {fd_effect:.3f}\n")
    print(
        f"Naive smoking->cancer regression (genotype hidden) ..... {naive:6.3f}   wrong, confounded"
    )
    print(f"scm.ate('X','Y'), front-door identified ................ {ate.mean():6.3f}   CrI {ci}")
    print(f"  identified despite an UNMEASURED confounder? .......... {ate.identified}\n")
    print("Under the hood (how it does what adjustment cannot):")
    print(
        f"  confounded district fit jointly with a latent .... {scm._latent.confounded_districts}"
    )
    print(f"  clean mediator fit on its own .................... {scm._latent.singletons}")
    print(f"  the explicit latent standing in for the genotype . {scm._latent.latents}")
    print("=" * 74)
    print(
        "No adjustment set exists (you cannot condition on a variable you never\n"
        "measured), yet the effect is identified and computed. That is the front door."
    )


if __name__ == "__main__":
    main()
