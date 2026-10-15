# SPDX-License-Identifier: Apache-2.0
"""Native learners: ordinary least squares and logistic regression.

These cover the default nuisance slots, so a zero-dependency install can still
estimate. Both follow the sklearn estimator surface (``fit``, ``predict``,
``predict_proba`` for the classifier, ``get_params``, ``set_params``), so any
sklearn or xgboost estimator drops into the same slots with no adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._stats import _sigmoid

if TYPE_CHECKING:
    import numpy.typing as npt


def _design(x: npt.NDArray[np.float64], fit_intercept: bool) -> npt.NDArray[np.float64]:
    """A 2-D float design matrix, with a leading column of ones when asked."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if fit_intercept:
        ones = np.ones((arr.shape[0], 1), dtype=np.float64)
        return np.asarray(np.hstack([ones, arr]), dtype=np.float64)
    return arr


class OLS:
    """Ordinary least squares, with optional ridge penalty and sample weights."""

    def __init__(self, *, fit_intercept: bool = True, ridge: float = 0.0) -> None:
        self.fit_intercept = fit_intercept
        self.ridge = ridge
        self._coef: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> OLS:
        """Fit the coefficients by (weighted, optionally ridge-penalized) least squares.

        The intercept is never penalized.

        Args:
            X: A 2-D feature matrix (a 1-D array is read as one feature).
            y: The target, one value per row.
            sample_weight: Optional per-row weights.

        Returns:
            The fitted learner itself.
        """
        design = _design(X, self.fit_intercept)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        n, k = design.shape
        weight = (
            np.ones(n, dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        root = np.sqrt(weight)
        xw = design * root[:, None]
        yw = target * root
        if self.ridge > 0.0:
            penalty = np.sqrt(self.ridge) * np.eye(k, dtype=np.float64)
            if self.fit_intercept:
                penalty[0, 0] = 0.0
            xw = np.vstack([xw, penalty])
            yw = np.concatenate([yw, np.zeros(k, dtype=np.float64)])
        solution = np.linalg.lstsq(xw, yw, rcond=None)
        self._coef = np.asarray(solution[0], dtype=np.float64)
        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the fitted linear prediction for the rows of ``X``.

        Raises:
            RuntimeError: The learner has not been fitted.
        """
        if self._coef is None:
            raise RuntimeError("OLS.predict called before fit")
        return np.asarray(_design(X, self.fit_intercept) @ self._coef, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters, ``fit_intercept`` and ``ridge``."""
        return {"fit_intercept": self.fit_intercept, "ridge": self.ridge}

    def set_params(self, **params: Any) -> OLS:
        """Set parameters by name and return the learner itself."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LogisticRegression:
    """Binary logistic regression by Newton-Raphson (IRLS), ridge-stabilized."""

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        l2: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self._coef: npt.NDArray[np.float64] | None = None

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> LogisticRegression:
        """Fit the coefficients by Newton-Raphson with an L2 ridge on the slopes.

        Iteration stops when the largest step falls under ``tol`` or after ``max_iter``
        steps, with no warning when it stops on ``max_iter``.

        Args:
            X: A 2-D feature matrix (a 1-D array is read as one feature).
            y: A 0/1 label, one value per row.
            sample_weight: Optional per-row weights.

        Returns:
            The fitted learner itself.
        """
        design = _design(X, self.fit_intercept)
        label = np.asarray(y, dtype=np.float64).reshape(-1)
        n, k = design.shape
        weight = (
            np.ones(n, dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        penalty = self.l2 * np.eye(k, dtype=np.float64)
        if self.fit_intercept:
            penalty[0, 0] = 0.0
        beta = np.zeros(k, dtype=np.float64)
        for _ in range(self.max_iter):
            prob = _sigmoid(design @ beta)
            grad = design.T @ (weight * (label - prob)) - penalty @ beta
            reweight = weight * prob * (1.0 - prob)
            hessian = design.T @ (design * reweight[:, None]) + penalty
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                step = np.asarray(np.linalg.lstsq(hessian, grad, rcond=None)[0], dtype=np.float64)
            beta = beta + step
            if float(np.max(np.abs(step))) < self.tol:
                break
        self._coef = np.asarray(beta, dtype=np.float64)
        return self

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return class probabilities as an ``(n, 2)`` array: column 1 is P(y = 1).

        Raises:
            RuntimeError: The learner has not been fitted.
        """
        if self._coef is None:
            raise RuntimeError("LogisticRegression.predict_proba called before fit")
        prob = _sigmoid(_design(X, self.fit_intercept) @ self._coef)
        return np.asarray(np.column_stack([1.0 - prob, prob]), dtype=np.float64)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the 0/1 label, 1 where P(y = 1) is at least 0.5."""
        return np.asarray(self.predict_proba(X)[:, 1] >= 0.5, dtype=np.float64)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters: ``fit_intercept``, ``l2``, ``max_iter``, ``tol``."""
        return {
            "fit_intercept": self.fit_intercept,
            "l2": self.l2,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params: Any) -> LogisticRegression:
        """Set parameters by name and return the learner itself."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
