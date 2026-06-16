# SPDX-License-Identifier: Apache-2.0
"""Exogenous noise and abduction: recover a unit's structural noise.

A structural causal model writes every node as an equation over an explicit
exogenous-noise variable: ``X = f(PA) + U`` (additive, invertible) or
``X = g(PA, U)`` (general, e.g. the Gumbel-max categorical). A counterfactual
*abducts* that noise on the factual graph, then *predicts* on an action branch
that reuses it. This module owns the abduction, the step the design's
explicit-noise twin-network construction turns on.

Two forms, matching the two capability tiers in :mod:`._mechanism`:

- Invertible additive Gaussian: the noise is the exact residual ``U = X - f(PA)``,
  per draw of the parameter posterior. No MCMC, machine-precision counterfactuals
  (the pressure-test's ``Y_cf = y + b(x' - x)`` to ~1e-15).
- Gumbel-max categorical: the truncated-Gumbel abduction (Maddison; Oberst &
  Sontag 2019). The observed category pins the winner's Gumbel value at the
  sampled max; the losers draw from a Gumbel truncated above by it. Reusing the
  abducted Gumbel vector under shifted logits gives the categorical
  counterfactual.

Everything here is pure numpy: no pymc import, so the bare install keeps working
and abduction is fast and deterministic under a seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]


def gaussian_residual(value: Array, mean: Array) -> Array:
    """The exact additive-noise residual ``U = X - f(PA)``.

    ``value`` is the node's observed value and ``mean`` the structural mean
    ``f(PA)``; both broadcast, so a ``(S, n)`` posterior of means against an
    ``(n,)`` observation gives the per-draw, per-unit residual. This is the
    invertible abduction: no sampling, exact.
    """
    return np.asarray(value, dtype=np.float64) - np.asarray(mean, dtype=np.float64)


def gumbel_max_abduct(
    logits: Array,
    observed: IntArray,
    rng: np.random.Generator,
) -> Array:
    """Abduct the per-unit Gumbel noise consistent with the observed category.

    ``logits`` is ``(n, K)`` (one row of class logits per unit), ``observed`` the
    ``(n,)`` observed category codes. Returns the ``(n, K)`` Gumbel vector ``g``
    such that ``argmax(logits + g) == observed`` exactly, drawn from its posterior
    given the observation: the winner's total value is the unconditional max
    ``M ~ Gumbel(logsumexp(logits), 1)``, each loser a Gumbel truncated above by
    ``M``. The structural counterfactual reuses this ``g`` under shifted logits.

    This is the invertible categorical abduction; it needs no custom distribution,
    only the exact truncated-Gumbel inverse-CDF sampler.
    """
    logits = np.asarray(logits, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.int64)
    n, k = logits.shape
    if obs.shape != (n,):
        raise ValueError(f"observed must have shape ({n},), got {obs.shape}")
    if obs.size and (int(obs.min()) < 0 or int(obs.max()) >= k):
        raise ValueError(f"observed categories must be in [0, {k})")

    lse = np.log(np.exp(logits).sum(axis=1))  # (n,) logsumexp per row
    g = np.zeros((n, k), dtype=np.float64)
    rows = np.arange(n)
    # winner's total value M ~ Gumbel(lse, 1), via inverse CDF.
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=n)
    max_val = lse - np.log(-np.log(u))
    g[rows, obs] = max_val - logits[rows, obs]
    # each loser: a Gumbel(location=logits_k) truncated above by M.
    for col in range(k):
        loser = obs != col
        if not loser.any():
            continue
        loc = logits[loser, col]
        upper = max_val[loser]
        cdf_u = np.exp(-np.exp(-(upper - loc)))
        uu = rng.uniform(0.0, 1.0, size=loc.shape) * cdf_u
        uu = np.clip(uu, 1e-12, 1.0 - 1e-12)
        g[loser, col] = (loc - np.log(-np.log(uu))) - loc
    return g


def gumbel_max_decide(logits: Array, gumbel: Array) -> IntArray:
    """The forward Gumbel-max mechanism: ``argmax(logits + g)`` per row."""
    total = np.asarray(logits, dtype=np.float64) + np.asarray(gumbel, dtype=np.float64)
    return np.asarray(np.argmax(total, axis=1), dtype=np.int64)
