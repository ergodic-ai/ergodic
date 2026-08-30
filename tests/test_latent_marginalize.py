# SPDX-License-Identifier: Apache-2.0
"""The forward pass and the decode recursions, held to brute-force enumeration.

Chains this small (T <= 8, K = 2) can be summed over every one of their 2^T
paths, so the exact marginal likelihood and the exact posterior marginals are
available as ground truth with no approximation anywhere. The symbolic
forward is checked in both gears (unrolled and scanned), in both emission
shapes, and against the numpy forward-backward that decoding uses.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.latent._marginalize import forward_backward, log_forward, viterbi

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]


def _world(n_steps: int, seed: int, *, pairwise: bool = False) -> tuple[Array, Array, Array]:
    """Random emissions, transitions, and start for a tiny K=2 chain."""
    rng = np.random.default_rng(seed)
    shape = (n_steps, 2, 2) if pairwise else (n_steps, 2)
    loglik = rng.normal(0.0, 1.5, size=shape)
    trans = rng.dirichlet(np.ones(2), size=2)
    pi0 = rng.dirichlet(np.ones(2))
    return loglik, trans, pi0


def _path_score(path: tuple[int, ...], loglik: Array, trans: Array, pi0: Array) -> float:
    """The joint log-probability of one hidden path with its emissions."""
    pairwise = loglik.ndim == 3
    first = loglik[0, path[0], path[0]] if pairwise else loglik[0, path[0]]
    total = float(np.log(pi0[path[0]]) + first)
    for t in range(1, len(path)):
        emit = loglik[t, path[t - 1], path[t]] if pairwise else loglik[t, path[t]]
        total += float(np.log(trans[path[t - 1], path[t]]) + emit)
    return total


def _enumerate(loglik: Array, trans: Array, pi0: Array) -> tuple[float, Array]:
    """Exact marginal log-likelihood and posterior state marginals by enumeration."""
    n_steps = loglik.shape[0]
    scores = {
        path: _path_score(path, loglik, trans, pi0)
        for path in itertools.product(range(2), repeat=n_steps)
    }
    values = np.array(list(scores.values()))
    top = values.max()
    total = float(top + np.log(np.exp(values - top).sum()))
    gamma = np.zeros((n_steps, 2))
    for path, score in scores.items():
        weight = np.exp(score - total)
        for t, state in enumerate(path):
            gamma[t, state] += weight
    return total, gamma


@pytest.mark.parametrize("unroll", [True, False])
def test_log_forward_matches_enumeration(unroll: bool) -> None:
    pytest.importorskip("pytensor")
    loglik, trans, pi0 = _world(6, seed=1)
    expected, _ = _enumerate(loglik, trans, pi0)
    total = log_forward(
        _tensor(loglik), _tensor(np.log(trans)), _tensor(np.log(pi0)), 6, unroll=unroll
    )
    assert float(total.eval()) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("unroll", [True, False])
def test_log_forward_pairwise_matches_enumeration(unroll: bool) -> None:
    pytest.importorskip("pytensor")
    loglik, trans, pi0 = _world(5, seed=2, pairwise=True)
    expected, _ = _enumerate(loglik, trans, pi0)
    total = log_forward(
        _tensor(loglik), _tensor(np.log(trans)), _tensor(np.log(pi0)), 5, unroll=unroll
    )
    assert float(total.eval()) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("unroll", [True, False])
def test_pairwise_reduces_to_state_form_when_pair_independent(unroll: bool) -> None:
    """A (T, K, K) emission that ignores the previous state is a (T, K) emission."""
    pytest.importorskip("pytensor")
    loglik, trans, pi0 = _world(7, seed=3)
    expanded = np.repeat(loglik[:, None, :], 2, axis=1)  # (T, K_prev, K)
    args = (_tensor(np.log(trans)), _tensor(np.log(pi0)))
    flat = log_forward(_tensor(loglik), *args, 7, unroll=unroll)
    pair = log_forward(_tensor(expanded), *args, 7, unroll=unroll)
    assert float(pair.eval()) == pytest.approx(float(flat.eval()), abs=1e-10)


def test_log_forward_refusals() -> None:
    pytest.importorskip("pytensor")
    loglik, trans, pi0 = _world(4, seed=4)
    with pytest.raises(ValueError, match="ndim"):
        log_forward(_tensor(loglik[0]), _tensor(np.log(trans)), _tensor(np.log(pi0)), 4)
    with pytest.raises(ValueError, match="n_steps"):
        log_forward(_tensor(loglik), _tensor(np.log(trans)), _tensor(np.log(pi0)), 0)


def test_forward_backward_matches_enumeration() -> None:
    loglik, trans, pi0 = _world(8, seed=5)
    expected_total, expected_gamma = _enumerate(loglik, trans, pi0)
    gamma, total = forward_backward(loglik, trans, pi0)
    assert gamma.shape == (8, 2)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(gamma, expected_gamma, atol=1e-10)
    assert total == pytest.approx(expected_total, abs=1e-10)


def test_forward_backward_refuses_bad_shape() -> None:
    loglik, trans, pi0 = _world(4, seed=6, pairwise=True)
    with pytest.raises(ValueError, match=r"\(T, K\)"):
        forward_backward(loglik, trans, pi0)


def test_viterbi_beats_every_other_path() -> None:
    """The Viterbi path's joint score tops enumeration, so also any random path."""
    loglik, trans, pi0 = _world(8, seed=7)
    path = viterbi(loglik, trans, pi0)
    best = _path_score(tuple(int(s) for s in path), loglik, trans, pi0)
    for other in itertools.product(range(2), repeat=8):
        assert best >= _path_score(other, loglik, trans, pi0) - 1e-12


def _tensor(values: Array) -> object:
    import pytensor.tensor as pt

    return pt.as_tensor_variable(values)
