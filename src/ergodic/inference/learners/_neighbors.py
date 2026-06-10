# SPDX-License-Identifier: Apache-2.0
"""Native nearest-neighbor learners: a flexible nuisance model with no dependency.

Features are standardized by the training mean and standard deviation, distances
are Euclidean, and a prediction averages (regressor) or votes (classifier) over
the ``k`` nearest training rows. With no features every row is a tie, so the
prediction falls back to the training mean or class frequency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

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
        self._mean: npt.NDArray[np.float64] | None = None
        self._scale: npt.NDArray[np.float64] | None = None

    def _standardize_fit(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self._mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale[scale == 0.0] = 1.0
        self._scale = scale
        return np.asarray((x - self._mean) / self._scale, dtype=np.float64)

    def _standardize(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self._mean is not None and self._scale is not None
        return np.asarray((x - self._mean) / self._scale, dtype=np.float64)

    def _neighbor_targets(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """For each query row, the targets of its ``k`` nearest training rows (m, k)."""
        assert self._x is not None and self._y is not None
        query = self._standardize(_2d(x))
        if self._x.shape[1] == 0:
            # with no features every training row ties, so all of them are neighbors
            idx = np.tile(np.arange(self._x.shape[0]), (query.shape[0], 1))
            return np.asarray(self._y[idx], dtype=np.float64)
        sq = (
            np.sum(query**2, axis=1)[:, None]
            + np.sum(self._x**2, axis=1)[None, :]
            - 2.0 * query @ self._x.T
        )
        k = min(self.k, self._x.shape[0])
        nearest = np.argpartition(sq, kth=k - 1, axis=1)[:, :k]
        return np.asarray(self._y[nearest], dtype=np.float64)

    def _fit_xy(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        self._x = self._standardize_fit(_2d(X))
        self._y = np.asarray(y, dtype=np.float64).reshape(-1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"k": self.k}

    def set_params(self, **params: Any) -> _KNNBase:
        for key, value in params.items():
            setattr(self, key, value)
        return self


class KNNRegressor(_KNNBase):
    """K-nearest-neighbors regression: the prediction is the neighbor mean."""

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> KNNRegressor:
        self._fit_xy(X, y)
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.mean(self._neighbor_targets(X), axis=1), dtype=np.float64)


class KNNClassifier(_KNNBase):
    """K-nearest-neighbors classification for a binary 0/1 label."""

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> KNNClassifier:
        self._fit_xy(X, y)
        return self

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        prob = np.mean(self._neighbor_targets(X), axis=1)
        return np.asarray(np.column_stack([1.0 - prob, prob]), dtype=np.float64)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(self.predict_proba(X)[:, 1] >= 0.5, dtype=np.float64)
