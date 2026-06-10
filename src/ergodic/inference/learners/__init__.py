# SPDX-License-Identifier: Apache-2.0
"""Native learners for the nuisance slots."""

from __future__ import annotations

from ._linear import OLS, LogisticRegression
from ._neighbors import KNNClassifier, KNNRegressor
from ._pymc import PyMCClassifier, PyMCLearner

__all__ = [
    "OLS",
    "KNNClassifier",
    "KNNRegressor",
    "LogisticRegression",
    "PyMCClassifier",
    "PyMCLearner",
]
