# SPDX-License-Identifier: Apache-2.0
"""The G squared test, natively categorical.

The likelihood-ratio (mutual-information) test for discrete data: stratify
the rows by the joint configuration of the conditioning variables, build the
x-by-y contingency table inside each stratum, and accumulate
``G2 = 2 * sum(obs * log(obs / expected))`` against a chi-square reference.
Runs on integer codes, no encoding to floats anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._ci import CITest
from .._stats import _chi2_sf

if TYPE_CHECKING:
    import numpy.typing as npt


class Discrete(CITest):
    """The G squared conditional-independence test on contingency tables.

    Degrees of freedom are adjusted for sparsity, bnlearn-style: each
    **nonempty** stratum of the conditioning configuration contributes
    ``(card_x - 1) * (card_y - 1)``, and configurations with no observed rows
    contribute nothing. The unadjusted product over the full configuration
    space would overstate the dof badly on sparse tables and make the test
    blind. When the dof comes out zero (a single-level endpoint, or no rows),
    the query is vacuous and the result is ``(0.0, 1.0, 0.0)``.
    """

    name: ClassVar[str] = "discrete"
    needs: ClassVar[str] = "codes"

    def statistic(
        self,
        x: npt.NDArray[np.int64],
        y: npt.NDArray[np.int64],
        z: tuple[npt.NDArray[np.int64], ...],
        cards: tuple[int, int, tuple[int, ...]],
    ) -> tuple[float, float, float | None]:
        card_x, card_y, _ = cards
        n = x.shape[0]
        if z:
            _, strata = np.unique(np.stack(z, axis=1), axis=0, return_inverse=True)
            strata = strata.reshape(-1)
            n_strata = int(strata.max()) + 1 if strata.size else 0
        else:
            strata = np.zeros(n, dtype=np.int64)
            n_strata = 1 if n else 0
        per_stratum = (card_x - 1) * (card_y - 1)
        g2 = 0.0
        dof = 0.0
        for s in range(n_strata):
            mask = strata == s
            counts = np.bincount(x[mask] * card_y + y[mask], minlength=card_x * card_y)
            table = counts.reshape(card_x, card_y).astype(np.float64)
            expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
            seen = table > 0
            g2 += 2.0 * float(np.sum(table[seen] * np.log(table[seen] / expected[seen])))
            dof += per_stratum
        if dof == 0.0:
            return (0.0, 1.0, 0.0)
        return g2, _chi2_sf(g2, dof), dof
