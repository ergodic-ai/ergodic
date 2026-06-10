# SPDX-License-Identifier: Apache-2.0
"""The degenerate-Gaussian BIC: one-hot blocks scored as jointly Gaussian.

Andrews, Ramsey, and Cooper (2019) score a mixed system by replacing each
discrete variable with its one-hot indicator block and treating the stacked
design as jointly Gaussian. The covariance of an indicator block is
degenerate, hence the name, but the resulting score is consistent for model
selection. A local is a block Schur complement on the covariance the parent
``BIC`` already prepares at bind, with the log-determinant read off the
eigenvalues, each clamped at ``1e-12`` so a degenerate direction costs a
constant that cancels across parent sets.

Unlike conditional-Gaussian scores there is no edge restriction: discrete and
continuous variables mix freely on both sides of an edge. On all-continuous
data every block has one column, the determinant is the residual variance,
and the score agrees with ``BIC`` exactly, penalty included: the parameter
count ``dy dx + dy (dy + 1) / 2`` reduces to ``dx + 1`` at ``dy = 1``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ._bic import BIC

if TYPE_CHECKING:
    import numpy.typing as npt

_EIGENVALUE_FLOOR = 1e-12


class MixedBIC(BIC):
    """The degenerate-Gaussian BIC, decomposable and score-equivalent.

    Any column kind is a valid node: a continuous node is its single column
    and a categorical node is its one-hot block (first level dropped), scored
    jointly. ``local = -(n/2) log det Sigma_{Y|X} - penalty * p/2 * log n``
    with ``p = dy dx + dy (dy + 1) / 2`` for a ``dy``-column node and ``dx``
    total parent columns.

    Args:
        data: Bind at construction, or later through ``bind``.
        penalty: A multiplier on the complexity term, inherited from ``BIC``.
            ``1.0`` is the textbook BIC; smaller values prune less.
    """

    name: ClassVar[str] = "bic_mixed"

    # -- the covariance path ----------------------------------------------------

    def _local_value(self, node: str, parents: frozenset[str]) -> float:
        """The local by block Schur complement on the precomputed covariance."""
        node_slice = self._slices[node]
        y_columns = list(range(node_slice.start, node_slice.stop))
        x_columns = [
            column
            for parent in sorted(parents)
            for column in range(self._slices[parent].start, self._slices[parent].stop)
        ]
        s_yy = self._cov[np.ix_(y_columns, y_columns)]
        if x_columns:
            s_xx = self._cov[np.ix_(x_columns, x_columns)]
            s_xy = self._cov[np.ix_(x_columns, y_columns)]
            conditional = s_yy - s_xy.T @ np.linalg.pinv(s_xx, hermitian=True) @ s_xy
        else:
            conditional = s_yy
        return self._mixed_value(
            np.asarray(conditional, dtype=np.float64),
            dy=len(y_columns),
            dx=len(x_columns),
            n=self._n,
        )

    # -- the array path (user contract, and the cross-check) ---------------------

    def local_from_arrays(self, y: npt.NDArray[np.float64], X: npt.NDArray[np.float64]) -> float:
        """The same local by direct least squares on ``(n, dy)`` ``y`` and ``(n, k)`` ``X``.

        Agrees with the covariance path to numerical precision; the tests hold
        the two to ``1e-8``.
        """
        if y.ndim != 2 or y.shape[1] < 1:
            raise TypeError("MixedBIC scores column blocks; y must have shape (n, dy), dy >= 1")
        n = int(y.shape[0])
        if n == 0:
            raise ValueError("MixedBIC needs at least one row")
        centered_y = y - y.mean(axis=0, keepdims=True)
        if X.shape[1]:
            centered_x = X - X.mean(axis=0, keepdims=True)
            beta, *_ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
            residual = centered_y - centered_x @ beta
        else:
            residual = centered_y
        conditional = np.asarray(residual.T @ residual, dtype=np.float64) / float(n)
        return self._mixed_value(conditional, dy=int(y.shape[1]), dx=int(X.shape[1]), n=n)

    # -- the shared tail -----------------------------------------------------------

    def _mixed_value(
        self, conditional: npt.NDArray[np.float64], *, dy: int, dx: int, n: int
    ) -> float:
        eigenvalues = np.linalg.eigvalsh(conditional)
        logdet = float(np.log(np.maximum(eigenvalues, _EIGENVALUE_FLOOR)).sum())
        size = float(n)
        fit = -(size / 2.0) * logdet
        parameters = dy * dx + dy * (dy + 1) / 2.0
        complexity = self._penalty * (parameters / 2.0) * math.log(size)
        return fit - complexity
