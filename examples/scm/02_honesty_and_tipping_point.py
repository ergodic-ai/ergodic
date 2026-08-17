# SPDX-License-Identifier: Apache-2.0
"""The honesty layer: when an effect is not identified, you get a curve, not a number.

Most causal tooling will hand you a confident point estimate whether or not the data
can support one. The ergodic SCM will not. When the effect you ask for is not a
function of the data alone, ``ate()`` returns a :class:`Sensitivity` object that has
*no* scalar accessor at all. You literally cannot read a naked number off it. You get
the effect as a function of the assumed confounding instead.

The world here is a bow: an unmeasured U drives both X and Y, and X also affects Y
directly (the true direct effect is 0.7). Because U is latent, no adjustment set
exists, so the effect is not identified. The same ``scm.ate("X", "Y")`` call that
returns a posterior point on an identified world returns a Sensitivity here. The
return *type* is the message.

That curve is not a shrug. It is a decision tool. It tells you exactly how strong the
hidden confounding would have to be to overturn your conclusion. Here the effect stays
positive unless the confounding channel explains *all* of the observed covariance, so
"X helps" is a robust read even though its magnitude is not pinned.

Run with: uv run python examples/scm/02_honesty_and_tipping_point.py
"""

from __future__ import annotations

import logging

import numpy as np

from ergodic import dag, tabular
from ergodic.scm import SCM, SCMResult, Sensitivity


def _quiet_pymc() -> None:
    for name in ("pymc", "pymc.sampling"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_pymc()

    # --- a bow: U latent, so the X -> Y effect is NOT identified --------------
    a_x, a_y, b = 1.3, 1.1, 0.7  # U->X, U->Y, X->Y (the true, unidentified effect)
    rng = np.random.default_rng(20240615)
    n = 600
    u = rng.normal(0, 1, n)
    x = a_x * u + rng.normal(0, 0.6, n)
    y = b * x + a_y * u + rng.normal(0, 0.5, n)

    data = tabular({"X": x, "Y": y})
    graph = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
    scm = SCM(graph, data, draws=300, tune=300, chains=2, random_state=20240615).fit()

    res = scm.ate("X", "Y")
    assert isinstance(res, Sensitivity)

    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
    lo, hi = res.range()

    # The sign tipping point: b(c) = (cov - c)/var = 0 at c = cov_xy, so the effect
    # stays positive until the confounding channel explains the whole covariance.
    c_zero = cov_xy
    b0 = res.identified_at_zero  # the effect if you assume zero confounding
    half_c = 0.5 * cov_xy
    e_half = res.at(half_c)  # the effect if half the covariance is confounding
    e_tip = res.at(c_zero)  # the effect at the sign tipping point (= 0)

    print(__doc__.strip().splitlines()[0])
    print("=" * 74)
    print(f"True (unidentified) direct effect of X on Y:  {b:.3f}\n")
    print(f"scm.ate('X','Y') returned:  {type(res).__name__}")
    print(f"  has a .value accessor?    {hasattr(res, 'value')}   (a naked point is impossible)")
    print(f"  identified?               {res.identified}\n")
    print("The effect as a function of the assumed confounding strength c:")
    print(f"  c = 0     (assume no confounding) ...... {b0:6.3f}   the confounded slope")
    print(f"  c = {half_c:4.2f}  (half the covariance) ...... {e_half:6.3f}")
    print(f"  c = {c_zero:4.2f}  (ALL the covariance) ....... {e_tip:6.3f}   the sign tips here")
    print(f"  swept range of the effect .............. [{lo:.3f}, {hi:.3f}]")
    print(f"  the truth {b:.2f} lies inside this range:  {lo <= b <= hi}\n")
    print("DECISION READ:")
    print(
        f"  To overturn 'X helps Y', the hidden confounder would have to explain the\n"
        f"  ENTIRE observed X-Y covariance (c >= {c_zero:.2f}). Attribute even half of it\n"
        f"  to confounding and the effect is still +{e_half:.2f}. The sign is robust;\n"
        f"  the magnitude is not pinned, and the model refuses to pretend otherwise."
    )

    # --- contrast: the SAME call on an IDENTIFIED world returns a point --------
    rng2 = np.random.default_rng(1)
    zc = rng2.normal(0, 1, n)
    xi = 0.8 * zc + rng2.normal(0, 0.6, n)
    yi = b * xi + 1.0 * zc + rng2.normal(0, 0.5, n)
    ident = SCM(
        dag(["Z -> X", "Z -> Y", "X -> Y"]),
        tabular({"Z": zc, "X": xi, "Y": yi}),
        draws=300,
        tune=300,
        chains=2,
        random_state=1,
    ).fit()
    point = ident.ate("X", "Y")
    assert isinstance(point, SCMResult)
    print("=" * 74)
    print(
        f"Same ate() call, identified world (Z observed): {type(point).__name__}"
        f"  ->  {point.mean():.3f}\n"
        "The type you get back tells you whether the data can answer the question."
    )


if __name__ == "__main__":
    main()
