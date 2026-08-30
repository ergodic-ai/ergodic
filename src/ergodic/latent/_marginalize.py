# SPDX-License-Identifier: Apache-2.0
"""Marginalizing the hidden path: forward in the model, forward-backward outside it.

NUTS cannot sample a discrete path, and it does not have to. The forward
algorithm sums the path out exactly, leaving a marginal log-likelihood that is
a smooth function of the continuous parameters, and that scalar enters the
model as a single ``pm.Potential``. The sampler then explores stay
probabilities and emission parameters only, and the path comes back afterwards
per posterior draw with the numpy forward-backward here.

Everything runs in log space. A 1400-step chain underflows a probability-space
forward within a few hundred steps, and rescaling tricks are just log-space
arithmetic wearing a disguise, so the recursions here are written in logs from
the start.

The symbolic forward comes in two gears, both measured in the reference
implementations this pillar ports. A short chain (the spectral model's 15
frames) is unrolled: fifteen explicit steps compile faster and run faster than
a scan of fifteen, and cheap fits are the point of framing the series in the
first place, because calibration reruns the whole pipeline over many control
worlds. A long chain (the autoregression's ~1400 steps) goes through
``pytensor.scan``, where unrolling would blow up the graph. The cutover is
:data:`UNROLL_MAX`; ``unroll=`` overrides it either way.

Emissions arrive as ``(T, K)``, or as ``(T, K, K)`` for a density that
conditions on the previous state as well as the current one. The first step of
a pairwise emission has no previous state, so it reads the diagonal
``loglik[0, j, j]``: when the pairwise tensor does not actually depend on the
previous state, that convention makes the two shapes agree exactly, which the
tests hold executably.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]

__all__ = ["UNROLL_MAX", "forward_backward", "log_forward", "viterbi"]

#: Chains at or under this many steps are unrolled; longer ones go through scan.
UNROLL_MAX = 32


def log_forward(
    loglik: Any,
    log_trans: Any,
    log_pi0: Any,
    n_steps: int,
    *,
    unroll: bool | None = None,
) -> Any:
    """The marginal log-likelihood of the observations, path summed out.

    ``loglik`` is a pytensor tensor of shape ``(T, K)`` or ``(T, K, K)``,
    ``log_trans`` the ``(K, K)`` log transition matrix with rows indexed by the
    previous state, ``log_pi0`` the ``(K,)`` log initial distribution.
    ``n_steps`` must be the concrete ``T``, known at build time; it picks the
    unrolled or scanned recursion unless ``unroll`` says otherwise. Returns a
    scalar tensor for ``pm.Potential``.
    """
    import importlib

    # The typed-Any surface: pytensor's own ops trip mypy's no-untyped-call
    # (the SCM notes the same and routes through pm.math; this module has no
    # pymc in play, so it takes the module through importlib instead).
    pt: Any = importlib.import_module("pytensor.tensor")
    scan: Any = importlib.import_module("pytensor").scan

    if loglik.ndim not in (2, 3):
        raise ValueError(f"loglik must be (T, K) or (T, K, K), got a tensor of ndim {loglik.ndim}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    pairwise = loglik.ndim == 3

    def lse0(x: Any) -> Any:
        m = pt.max(x, axis=0)
        return m + pt.log(pt.sum(pt.exp(x - m), axis=0))

    first = pt.diagonal(loglik[0]) if pairwise else loglik[0]

    unrolled = unroll if unroll is not None else n_steps <= UNROLL_MAX
    if unrolled:
        # The spectral reference's loop: renormalize each step and accumulate
        # the normalizers, so the running vector stays near zero.
        lp = log_pi0 + first
        total = pt.constant(0.0)
        for t in range(1, n_steps):
            z = lse0(lp)
            total = total + z
            step = (lp - z)[:, None] + log_trans
            if pairwise:
                step = step + loglik[t]
            lp = lse0(step)
            if not pairwise:
                lp = lp + loglik[t]
        return total + lse0(lp)

    # The switching-AR reference's scan: plain log-space forward, no
    # renormalization needed because log values grow only linearly in T.
    if pairwise:

        def step_pair(le_t: Any, log_alpha: Any, log_p: Any) -> Any:
            return lse0(log_alpha[:, None] + log_p + le_t)

        log_alpha = scan(
            step_pair,
            sequences=[loglik[1:]],
            outputs_info=[log_pi0 + first],
            non_sequences=[log_trans],
            return_updates=False,
        )
    else:

        def step_state(le_t: Any, log_alpha: Any, log_p: Any) -> Any:
            return lse0(log_alpha[:, None] + log_p) + le_t

        log_alpha = scan(
            step_state,
            sequences=[loglik[1:]],
            outputs_info=[log_pi0 + first],
            non_sequences=[log_trans],
            return_updates=False,
        )
    return lse0(log_alpha[-1])


def _lse(x: Array, axis: int) -> Array:
    m = x.max(axis=axis, keepdims=True)
    out: Array = (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)
    return out


def forward_backward(loglik: Array, trans: Array, pi0: Array) -> tuple[Array, float]:
    """Posterior state marginals and the marginal log-likelihood, in log space.

    ``loglik`` is ``(T, K)`` log emission densities for one posterior draw,
    ``trans`` the ``(K, K)`` transition matrix, ``pi0`` the initial
    distribution, both in probability space. Returns ``(gamma, loglik)`` with
    ``gamma`` of shape ``(T, K)`` summing to one per row.
    """
    ll = np.asarray(loglik, dtype=np.float64)
    if ll.ndim != 2:
        raise ValueError(f"loglik must be (T, K), got shape {ll.shape}")
    n_obs, k = ll.shape
    log_t = np.log(np.maximum(np.asarray(trans, dtype=np.float64), 1e-300))
    log_pi0 = np.log(np.maximum(np.asarray(pi0, dtype=np.float64), 1e-300))

    alpha = np.zeros((n_obs, k))
    alpha[0] = log_pi0 + ll[0]
    for t in range(1, n_obs):
        alpha[t] = _lse(alpha[t - 1][:, None] + log_t, axis=0) + ll[t]
    beta = np.zeros((n_obs, k))
    for t in range(n_obs - 2, -1, -1):
        beta[t] = _lse(log_t + (ll[t + 1] + beta[t + 1])[None, :], axis=1)

    lg = alpha + beta
    lg -= _lse(lg, axis=1)[:, None]
    m = alpha[-1].max()
    total = float(m + np.log(np.exp(alpha[-1] - m).sum()))
    return np.exp(lg), total


def viterbi(loglik: Array, trans: Array, pi0: Array) -> IntArray:
    """The single most probable path, in log space.

    Same inputs as :func:`forward_backward`. The decode accessors report the
    marginal path (``gamma > 0.5`` per step), which minimizes per-step error;
    Viterbi answers the different question of the best whole path, and is here
    for callers who want that one.
    """
    ll = np.asarray(loglik, dtype=np.float64)
    if ll.ndim != 2:
        raise ValueError(f"loglik must be (T, K), got shape {ll.shape}")
    n_obs, k = ll.shape
    log_t = np.log(np.maximum(np.asarray(trans, dtype=np.float64), 1e-300))
    log_pi0 = np.log(np.maximum(np.asarray(pi0, dtype=np.float64), 1e-300))

    score = np.zeros((n_obs, k))
    back = np.zeros((n_obs, k), dtype=np.int64)
    score[0] = log_pi0 + ll[0]
    for t in range(1, n_obs):
        cand = score[t - 1][:, None] + log_t
        back[t] = cand.argmax(axis=0)
        score[t] = cand.max(axis=0) + ll[t]

    path = np.zeros(n_obs, dtype=np.int64)
    path[-1] = int(score[-1].argmax())
    for t in range(n_obs - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return path
