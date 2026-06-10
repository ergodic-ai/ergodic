# SPDX-License-Identifier: Apache-2.0
"""The Gaussian BIC score: one covariance pass, every local by Schur complement.

For a linear-Gaussian node the maximized log-likelihood depends on the data
only through the residual variance of the node given its parents, and that
variance can be read off a covariance matrix without touching the rows again.
Binding therefore stacks the per-variable numeric blocks into one centered
design and computes its MLE covariance once; ``local(node, parents)`` is then
a Schur complement on it.

The convention: ``local = -(n/2) log sigma2 - penalty * ((k + 1)/2) log n``,
where ``k`` is the total parent column count (a categorical parent contributes
``cardinality - 1`` one-hot columns) and the ``+ 1`` counts the residual
variance. Terms that are equal across parent sets, the Gaussian normalizer
``-(n/2)(1 + log 2 pi)`` and the intercept parameter, are dropped: they shift
every local of a node by the same amount, so no comparison a search makes can
see them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .._score import Score

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData

_VARIANCE_FLOOR = 1e-12


class BIC(Score):
    """The Gaussian BIC, decomposable and score-equivalent.

    Nodes must be single numeric columns (continuous, datetime, or binary);
    scoring a wider categorical block raises ``TypeError`` lazily, at the
    first query against that node. Categorical *parents* are fine: their
    one-hot columns enter the regression and each column counts toward ``k``.

    Args:
        data: Bind at construction, or later through ``bind``.
        penalty: A multiplier on the complexity term ``((k + 1)/2) log n``.
            ``1.0`` is the textbook BIC; smaller values prune less.
    """

    name: ClassVar[str] = "bic"
    decomposable: ClassVar[bool] = True
    score_equivalent: ClassVar[bool] = True

    def __init__(self, data: TabularData | None = None, *, penalty: float = 1.0) -> None:
        if penalty <= 0.0:
            raise ValueError(f"penalty must be positive, got {penalty}")
        self._penalty = float(penalty)
        self._slices: dict[str, slice] = {}
        self._cov: npt.NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        super().__init__(data)

    @property
    def penalty(self) -> float:
        """The multiplier on the complexity term."""
        return self._penalty

    # -- one-time sufficient statistics ---------------------------------------

    def _prepare(self) -> None:
        if self._n == 0:
            raise ValueError("BIC needs at least one complete row; the data has none")
        slices: dict[str, slice] = {}
        parts: list[npt.NDArray[np.float64]] = []
        start = 0
        for name in self.variables:
            block = self._blocks[name]
            stop = start + block.shape[1]
            slices[name] = slice(start, stop)
            parts.append(block)
            start = stop
        design = np.hstack(parts) if parts else np.empty((self._n, 0), dtype=np.float64)
        centered = design - design.mean(axis=0, keepdims=True)
        self._slices = slices
        self._cov = np.asarray(centered.T @ centered, dtype=np.float64) / float(self._n)

    # -- the covariance path ---------------------------------------------------

    def _local_value(self, node: str, parents: frozenset[str]) -> float:
        """The local by Schur complement on the precomputed covariance."""
        node_slice = self._slices[node]
        width = node_slice.stop - node_slice.start
        if width != 1:
            raise TypeError(
                f"BIC scores single-column nodes; {node!r} encodes to {width} columns. "
                "The mixed score lands in D2."
            )
        target = node_slice.start
        columns = [
            column
            for parent in sorted(parents)
            for column in range(self._slices[parent].start, self._slices[parent].stop)
        ]
        syy = float(self._cov[target, target])
        if columns:
            spp = self._cov[np.ix_(columns, columns)]
            spy = self._cov[columns, target]
            sigma2 = syy - float(spy @ np.linalg.pinv(spp, hermitian=True) @ spy)
        else:
            sigma2 = syy
        return self._bic_value(sigma2, k=len(columns), n=self._n)

    # -- the array path (user contract, and the cross-check) -------------------

    def local_from_arrays(self, y: npt.NDArray[np.float64], X: npt.NDArray[np.float64]) -> float:
        """The same local by direct least squares on ``(n, 1)`` ``y`` and ``(n, k)`` ``X``.

        Agrees with the covariance path to numerical precision; the tests hold
        the two to ``1e-8``.
        """
        if y.ndim != 2 or y.shape[1] != 1:
            raise TypeError(
                "BIC scores single-column nodes; y must have shape (n, 1). "
                "The mixed score lands in D2."
            )
        n = int(y.shape[0])
        if n == 0:
            raise ValueError("BIC needs at least one row")
        centered_y = y - y.mean(axis=0, keepdims=True)
        if X.shape[1]:
            centered_x = X - X.mean(axis=0, keepdims=True)
            beta, *_ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
            residual = centered_y - centered_x @ beta
        else:
            residual = centered_y
        sigma2 = float((residual * residual).sum()) / n
        return self._bic_value(sigma2, k=int(X.shape[1]), n=n)

    # -- the shared tail --------------------------------------------------------

    def _bic_value(self, sigma2: float, *, k: int, n: int) -> float:
        bounded = max(sigma2, _VARIANCE_FLOOR)
        size = float(n)
        fit = -(size / 2.0) * math.log(bounded)
        complexity = self._penalty * ((k + 1) / 2.0) * math.log(size)
        return fit - complexity
