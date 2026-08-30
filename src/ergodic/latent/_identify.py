# SPDX-License-Identifier: Apache-2.0
"""Label alignment: fixing the switch's name after the fit, never during it.

A two-state model is symmetric in its labels. Swap "state 0" and "state 1"
everywhere (flip the path, swap the stay probabilities, negate every delta)
and the likelihood is unchanged, so the posterior has two mirror-image modes
and each draw lands in whichever one its chain wandered into. Averaging draws
across modes cancels the signal: half the draws say the switch was on where
the other half say off, and the mean path flattens to one half everywhere.

The textbook fix is an ordering constraint, ``mu_0 < mu_1`` or a positivity
constraint on one delta. That is the wrong tool here, twice over. It breaks
exactly when the ordered parameter is the one the mechanism did not touch (a
spread regime with no level shift leaves ``mu`` unordered in truth, and the
constraint then scrambles rather than fixes the labels), and it is not
vocabulary-free: choosing which parameter to order presumes which mechanism
fired, which is the question being asked. Ordering also distorts the
posterior near the boundary, biasing small deltas away from zero, which is
poison for a verdict defined as zero-exclusion.

So the engine aligns after the fact instead, on the one quantity every
likelihood shares: the decoded path. Each draw's posterior state-1 trajectory
is correlated against a reference; a negative correlation means that draw
lives in the mirror mode, and it is flipped (path reflected, and downstream,
its delta draws sign-flipped). The reference starts as the first draw and is
iterated to the mean of the aligned set, which makes the outcome independent
of which mode the first draw happened to sit in. Three sweeps suffice in
practice; the reference implementations this ports used exactly three and
measured stable alignments. A draw whose trajectory is flat carries no label
information and is left unflipped, since a correlation against a constant is
undefined rather than negative.

None of this ever consults the ground truth. Alignment is self-consistency
across draws, which is available in the field; agreement against a known
switch is a test-bench statistic only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    Array = npt.NDArray[np.float64]
    BoolArray = npt.NDArray[np.bool_]

__all__ = ["align_gammas"]

#: Trajectories with standard deviation under this carry no label information.
_FLAT = 1e-9

#: Alignment sweeps. The reference implementations used three and measured
#: stable references; more sweeps never changed a flip in their runs.
_SWEEPS = 3


def align_gammas(gammas: Array) -> tuple[Array, BoolArray]:
    """Align per-draw state trajectories across the label symmetry.

    ``gammas`` is ``(S, T)``: each row one decoded draw's posterior
    probability of state 1 per step. Returns the aligned rows (flipped rows
    replaced by ``1 - gamma``) and the boolean flip record, which downstream
    code uses to sign-correct that draw's deltas.
    """
    gammas = np.asarray(gammas, dtype=np.float64)
    if gammas.ndim != 2:
        raise ValueError(f"gammas must be (S, T), got shape {gammas.shape}")
    n_draws = gammas.shape[0]

    ref = gammas[0].copy()
    flips = np.zeros(n_draws, dtype=bool)
    aligned = gammas
    for _ in range(_SWEEPS):
        for j in range(n_draws):
            gj = gammas[j]
            if gj.std() < _FLAT or ref.std() < _FLAT:
                flips[j] = False
                continue
            flips[j] = bool(np.corrcoef(gj, ref)[0, 1] < 0)
        aligned = np.where(flips[:, None], 1.0 - gammas, gammas)
        ref = aligned.mean(axis=0)
    return aligned, flips
