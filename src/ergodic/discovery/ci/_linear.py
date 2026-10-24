# SPDX-License-Identifier: Apache-2.0
"""Partial correlation with the Fisher z transform, for continuous data.

The classic test behind PC on Gaussian data: residualize the two endpoints on
the conditioning block, correlate the residuals, and map the correlation
through ``atanh`` so the statistic is asymptotically standard normal. Exact
under joint Gaussianity, and a dependable linear-signal detector well beyond
it. Endpoints must be scalar columns; categorical endpoints expand to one-hot
blocks, and those belong to :class:`~ergodic.discovery.ci._mixed.Mixed`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._ci import CITest
from .._stats import _norm_two_sided

if TYPE_CHECKING:
    import numpy.typing as npt

_CLIP = 1.0 - 1e-12


def _residualize(
    block: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The least-squares residual of ``block`` on an intercept plus ``z``."""
    design = np.hstack([np.ones((block.shape[0], 1), dtype=np.float64), z])
    coef, *_ = np.linalg.lstsq(design, block, rcond=None)
    return np.asarray(block - design @ coef, dtype=np.float64)


class Linear(CITest):
    """Partial correlation through the Fisher z transform.

    Residualizes x and y on ``[1, z]``, computes the correlation ``r`` of the
    residuals, and reports ``sqrt(n - dz - 3) * atanh(r)`` against the
    standard normal, two-sided. The sign of ``r`` survives in the statistic;
    the p-value does not depend on it. Tails use erfc to avoid CDF subtraction
    cancellation; below binary64 range (e.g. |z|=40), p still becomes zero.
    """

    # Two-sided tails use erfc; below binary64 range (e.g. |z|=40), p is zero.
    name: ClassVar[str] = "linear"
    needs: ClassVar[str] = "numeric"

    def statistic(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> tuple[float, float, float | None]:
        n, dz = x.shape[0], z.shape[1]
        if x.shape[1] != 1 or y.shape[1] != 1:
            raise TypeError(
                "Linear needs scalar endpoints; a categorical endpoint expands to a "
                "one-hot block, use Mixed for categorical endpoints"
            )
        if n - dz - 3 <= 0:
            raise ValueError(
                "too few rows for the conditioning size: the Fisher z transform needs "
                f"n - dz - 3 > 0, got n={n} with dz={dz} conditioning columns"
            )
        rx = _residualize(x, z)[:, 0]
        ry = _residualize(y, z)[:, 0]
        denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
        r = float(rx @ ry) / denom if denom > 0.0 else 0.0
        r = min(_CLIP, max(-_CLIP, r))
        stat = math.sqrt(n - dz - 3) * math.atanh(r)
        p = _norm_two_sided(stat)
        return stat, p, None
