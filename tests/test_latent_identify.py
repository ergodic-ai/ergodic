# SPDX-License-Identifier: Apache-2.0
"""Label alignment: planted flips come back out, and flat draws stay put."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ergodic.latent._identify import align_gammas

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]


def _trajectory(n_steps: int = 60) -> Array:
    """A structured state-1 trajectory: two on-blocks with soft edges."""
    gamma = np.full(n_steps, 0.05)
    gamma[10:25] = 0.95
    gamma[40:52] = 0.9
    return gamma


def _draws(flipped: Array, seed: int = 0, noise: float = 0.05) -> Array:
    """Noisy copies of the base trajectory, mirrored where ``flipped`` says."""
    rng = np.random.default_rng(seed)
    base = _trajectory()
    gammas = np.clip(base[None, :] + rng.normal(0.0, noise, (len(flipped), len(base))), 0, 1)
    return np.where(flipped[:, None], 1.0 - gammas, gammas)


def test_planted_flips_are_recovered() -> None:
    planted = np.zeros(24, dtype=bool)
    planted[[3, 7, 8, 15, 21]] = True
    aligned, flips = align_gammas(_draws(planted))
    # Draw 0 is unflipped, so the reference mode is the planted one and the
    # flip record must match exactly, and every aligned draw must correlate
    # positively with every other.
    assert flips.tolist() == planted.tolist()
    mean = aligned.mean(axis=0)
    for row in aligned:
        assert np.corrcoef(row, mean)[0, 1] > 0.9


def test_alignment_is_mode_symmetric() -> None:
    """Starting inside the mirror mode aligns to it: labels are conventions.

    When draw 0 is itself flipped, self-consistency lands on the mirror
    labeling, and the flip record is the planted one inverted. Both answers
    are equally right, which is exactly why deltas are reported sign-aligned.
    """
    planted = np.zeros(24, dtype=bool)
    planted[[0, 1, 5, 6, 11]] = True
    aligned, flips = align_gammas(_draws(planted))
    assert flips.tolist() == (~planted).tolist()
    mean = aligned.mean(axis=0)
    for row in aligned:
        assert np.corrcoef(row, mean)[0, 1] > 0.9


def test_half_flipped_draws_align_to_one_mode() -> None:
    """An even split across the modes still collapses to one, not to mush."""
    planted = np.arange(30) % 2 == 0
    aligned, _ = align_gammas(_draws(planted))
    mean = aligned.mean(axis=0)
    base = _trajectory()
    corr = np.corrcoef(mean, base)[0, 1]
    assert abs(corr) > 0.95  # one mode or its mirror, never the average of both
    assert mean.std() > 0.3  # averaging across modes would flatten this toward 0


def test_flat_draws_never_flip() -> None:
    gammas = _draws(np.zeros(6, dtype=bool))
    gammas[2] = 0.5  # a draw with no label information at all
    aligned, flips = align_gammas(gammas)
    assert not flips[2]
    np.testing.assert_array_equal(aligned[2], gammas[2])


def test_align_refuses_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(S, T\)"):
        align_gammas(np.zeros(10))
