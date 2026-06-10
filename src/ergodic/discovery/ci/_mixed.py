# SPDX-License-Identifier: Apache-2.0
"""The degenerate-Gaussian test for mixed data (Andrews, Ramsey, Cooper 2019).

Categorical variables enter as their one-hot blocks (the base's numeric
encoding, first level dropped) and continuous variables as themselves, so
every query is a likelihood-ratio test between two column blocks given a
third. The statistic comes from the partial canonical correlations of the two
residualized blocks through Bartlett's chi-square approximation. This is the
default test when the columns mix kinds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._ci import CITest
from .._stats import _chi2_sf
from ._linear import _CLIP, _residualize

if TYPE_CHECKING:
    import numpy.typing as npt


class Mixed(CITest):
    """Partial canonical correlations with Bartlett's chi-square.

    Residualizes the x and y blocks on ``[1, z]``, takes the reduced QR of
    each residual block, and reads the canonical correlations ``rho_i`` off
    the singular values of ``Qx.T @ Qy`` (clipped below 1). Bartlett's
    statistic ``-(n - dz - (dx + dy + 3) / 2) * sum(log(1 - rho_i^2))`` is
    referred to a chi-square with ``dx * dy`` degrees of freedom.

    With scalar endpoints (``dx == dy == 1``) the single canonical correlation
    is the absolute partial correlation, so Mixed agrees with Linear
    asymptotically. A zero-width block (a single-level categorical) carries no
    information; that query is vacuous and returns ``(0.0, 1.0, 0.0)``.
    """

    name: ClassVar[str] = "mixed"
    needs: ClassVar[str] = "numeric"

    def statistic(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> tuple[float, float, float | None]:
        n = x.shape[0]
        dx, dy, dz = x.shape[1], y.shape[1], z.shape[1]
        if dx == 0 or dy == 0:
            return (0.0, 1.0, 0.0)
        if n - dz - (dx + dy) - 1 <= 0:
            raise ValueError(
                "too few rows for the block sizes: the Bartlett statistic needs "
                f"n - dz - (dx + dy) - 1 > 0, got n={n} with dx={dx}, dy={dy}, dz={dz}"
            )
        qx, _ = np.linalg.qr(_residualize(x, z), mode="reduced")
        qy, _ = np.linalg.qr(_residualize(y, z), mode="reduced")
        rho = np.clip(np.linalg.svd(qx.T @ qy, compute_uv=False), 0.0, _CLIP)
        stat = -(n - dz - (dx + dy + 3) / 2.0) * float(np.sum(np.log1p(-(rho**2))))
        dof = float(dx * dy)
        return stat, _chi2_sf(stat, dof), dof
