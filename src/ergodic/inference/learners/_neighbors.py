# SPDX-License-Identifier: Apache-2.0
"""Native nearest-neighbor learners: a flexible nuisance model with no dependency.

Features are standardized by the training mean and standard deviation, distances
are Euclidean, and a prediction averages (regressor) or votes (classifier) over
the ``k`` nearest training rows, weighted by ``sample_weight`` when ``fit`` gets
one. With no features every row is a tie, so the prediction falls back to the
(weighted) training mean or class frequency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._linear import _set_known_params

if TYPE_CHECKING:
    import numpy.typing as npt


def _2d(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


class _KNNBase:
    def __init__(self, *, k: int = 5) -> None:
        self.k = k
        self._x: npt.NDArray[np.float64] | None = None
        self._y: npt.NDArray[np.float64] | None = None
        self._w: npt.NDArray[np.float64] | None = None
        self._mean: npt.NDArray[np.float64] | None = None
        self._scale: npt.NDArray[np.float64] | None = None

    def _standardize_fit(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self._mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale[scale == 0.0] = 1.0
        self._scale = scale
        return np.asarray((x - self._mean) / self._scale, dtype=np.float64)

    def _standardize(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._mean is None or self._scale is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")
        return np.asarray((x - self._mean) / self._scale, dtype=np.float64)

    def _neighbor_mean(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """For each query row, the (weighted) mean target of its ``k`` nearest rows."""
        if self._x is None or self._y is None or self._w is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")
        idx = self._neighbor_index(x)
        weights = self._w[idx]
        totals = weights.sum(axis=1)
        plain = np.mean(self._y[idx], axis=1)
        weighted = np.sum(weights * self._y[idx], axis=1) / np.where(totals > 0.0, totals, 1.0)
        # a neighborhood whose weights are all zero falls back to the plain mean
        return np.asarray(np.where(totals > 0.0, weighted, plain), dtype=np.float64)

    def _neighbor_index(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """For each query row, the indices of its ``k`` nearest training rows (m, k)."""
        assert self._x is not None
        query = self._standardize(_2d(x))
        if self._x.shape[1] == 0:
            # with no features every training row ties, so all of them are neighbors
            return np.tile(np.arange(self._x.shape[0]), (query.shape[0], 1))
        sq = (
            np.sum(query**2, axis=1)[:, None]
            + np.sum(self._x**2, axis=1)[None, :]
            - 2.0 * query @ self._x.T
        )
        k = min(self.k, self._x.shape[0])
        return np.asarray(np.argpartition(sq, kth=k - 1, axis=1)[:, :k], dtype=np.int64)

    def _fit_xy(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None,
    ) -> None:
        self._x = self._standardize_fit(_2d(X))
        self._y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = self._y.shape[0]
        if sample_weight is None:
            self._w = np.ones(n, dtype=np.float64)
            return
        weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weight.shape[0] != n:
            raise ValueError(f"sample_weight has {weight.shape[0]} entries for {n} rows")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("sample_weight must be finite and non-negative")
        self._w = weight

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameter ``k``."""
        return {"k": self.k}

    def set_params(self, **params: Any) -> _KNNBase:
        """Set parameters by name and return the learner itself.

        Raises ``ValueError`` on a name :meth:`get_params` does not list.
        """
        _set_known_params(self, params)
        return self


class KNNRegressor(_KNNBase):
    """K-nearest-neighbors regression: the prediction is the neighbor mean."""

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> KNNRegressor:
        """Store the standardized training rows, their targets, and their weights.

        ``sample_weight`` (finite, non-negative, one per row) weights each row in the
        neighbor average; without it every row counts once.

        Returns:
            The fitted learner itself.
        """
        self._fit_xy(X, y, sample_weight)
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the (weighted) mean target of the ``k`` nearest training rows, per row.

        Raises ``RuntimeError`` before :meth:`fit`.
        """
        return self._neighbor_mean(X)


class KNNClassifier(_KNNBase):
    """K-nearest-neighbors classification for a binary 0/1 label."""

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> KNNClassifier:
        """Store the standardized training rows, their 0/1 labels, and their weights.

        ``sample_weight`` (finite, non-negative, one per row) weights each row in the
        neighbor vote; without it every row counts once.

        Returns:
            The fitted learner itself.
        """
        self._fit_xy(X, y, sample_weight)
        return self

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return class probabilities as an ``(n, 2)`` array.

        Column 1 is the (weighted) share of 1s among the ``k`` nearest training rows.
        Raises ``RuntimeError`` before :meth:`fit`.
        """
        prob = self._neighbor_mean(X)
        return np.asarray(np.column_stack([1.0 - prob, prob]), dtype=np.float64)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the 0/1 label, 1 where the neighbor share of 1s is at least 0.5."""
        return np.asarray(self.predict_proba(X)[:, 1] >= 0.5, dtype=np.float64)
