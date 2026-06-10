# SPDX-License-Identifier: Apache-2.0
"""K-fold cross-fitting for the orthogonal estimators.

Fit each nuisance on the other folds and predict on the held-out fold. The
out-of-fold predictions remove the bias from fitting and predicting on the same
rows, so a flexible learner keeps valid inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._learner import _mean_predict

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ._learner import Learner


def _folds(n: int, n_folds: int, rng: np.random.Generator) -> list[npt.NDArray[np.int64]]:
    """Shuffled test-fold index arrays; the train set is every other fold."""
    k = max(2, min(n_folds, n))
    order = rng.permutation(n)
    return [np.asarray(fold, dtype=np.int64) for fold in np.array_split(order, k)]


def _train_mask(n: int, test: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    mask = np.ones(n, dtype=bool)
    mask[test] = False
    return mask


def _fit_predict(
    make_one: Callable[[], Learner],
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Fit a fresh learner on the train rows and return its mean prediction on the test rows."""
    return _mean_predict(make_one().fit(x_train, y_train), x_test)
