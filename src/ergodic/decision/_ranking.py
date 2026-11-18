# SPDX-License-Identifier: Apache-2.0
"""Probability means and conservative, outcome-independent targeting validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..inference._learner import _fresh
from ..inference.learners import LogisticRegression

FloatArray = npt.NDArray[np.float64]


class ProbabilityMean:
    """Expose the probability of event 1 as a regression mean, never hard labels."""

    def __init__(self, classifier: Any = None) -> None:
        self.classifier = classifier if classifier is not None else LogisticRegression()

    def __sklearn_clone__(self) -> ProbabilityMean:
        return type(self)(_fresh(self.classifier))

    def fit(self, x: FloatArray, y: FloatArray) -> ProbabilityMean:
        y = np.asarray(y, float)
        if not np.isin(y, [0, 1]).all():
            raise ValueError("probability outcome means require binary event indicators")
        self.constant = float(y[0]) if len(np.unique(y)) == 1 else None
        if self.constant is None:
            self.model = _fresh(self.classifier).fit(x, y)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        if self.constant is not None:
            return np.full(len(x), self.constant)
        classes = np.asarray(
            [0, 1] if isinstance(self.model, LogisticRegression) else self.model.classes_
        )
        p = np.asarray(self.model.predict_proba(x), float)
        if (
            classes.shape != (2,)
            or set(classes) != {0, 1}
            or p.shape != (len(x), 2)
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-7)
        ):
            raise ValueError("outcome classifier must return normalized probabilities for 0 and 1")
        return np.asarray(p[:, int(np.flatnonzero(classes == 1)[0])], dtype=float)


def top_weights(scores: FloatArray, fraction: float = 0.2) -> FloatArray:
    """Exactly fraction*n selected mass, splitting all boundary ties equally."""
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("ranking requires finite scalar scores")
    mass = fraction * len(scores)
    cut = np.sort(scores)[-int(np.ceil(mass))]
    w = (scores > cut).astype(float)
    tied = scores == cut
    w[tied] = (mass - w.sum()) / tied.sum()
    return np.asarray(w, dtype=float)


def assess_targeting(
    predictions: dict[str, FloatArray],
    a: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    probability: float,
    family_size: int,
    direction: int = 1,
    default: str = "plugin",
) -> dict[str, Any]:
    """Finite-cohort Hoeffding bounds under explicitly declared Bernoulli assignment.

    Binary potential outcomes, independent assignments with constant known p,
    frozen models and pretreatment validation covariates. Not population policy
    intervals, calibrated individual-effect intervals, or evidence across reuse.

    Selection keeps ``default`` unless another candidate's paired gain over the
    default and its own gain over random both have a positive lower bound; among
    such candidates the largest paired lower bound wins. ``family_size`` must
    count every bound that may be reported (candidates plus paired differences).
    """
    a, y = np.asarray(a, float), np.asarray(y, float)
    if (
        len(a) < 20
        or y.shape != a.shape
        or not np.isin(a, [0, 1]).all()
        or not np.isin(y, [0, 1]).all()
        or not 0.01 < probability < 0.99
        or family_size < 1
        or direction not in (-1, 1)
    ):
        raise ValueError("validation needs >=20 binary rows and known assignment p in (.01,.99)")
    z = direction * (a * y / probability - (1 - a) * y / (1 - probability))
    weights = {k: top_weights(direction * np.asarray(v)) / 0.2 - 1 for k, v in predictions.items()}
    if any(v.shape != a.shape for v in weights.values()):
        raise ValueError("validation predictions must align with outcomes")

    def bound(h: FloatArray) -> dict[str, Any]:
        width = np.abs(h) * (1 / probability + 1 / (1 - probability))
        radius = float(
            np.sqrt(np.log(2 * family_size / 0.05) * np.sum(width**2) / (2 * len(a) ** 2))
        )
        gain = float(np.mean(h * z))
        lo, hi = gain - radius, gain + radius
        return {
            "gain": gain,
            "interval": [lo, hi],
            "status": ("supported" if lo > 0 else "evidence_against" if hi < 0 else "inconclusive"),
        }

    reported = len(weights) + (len(weights) - 1 if len(weights) > 1 else 0)
    if family_size < reported:
        raise ValueError("family_size must count every reported bound")
    report = {k: bound(h) for k, h in weights.items()}
    if len(predictions) == 1:
        selected, paired = next(iter(predictions)), {}
    else:
        if default not in weights:
            raise ValueError("the default ranking method must be among the candidates")
        paired = {k: bound(h - weights[default]) for k, h in weights.items() if k != default}
        better = {
            k: b["interval"][0]
            for k, b in paired.items()
            if b["interval"][0] > 0 and report[k]["interval"][0] > 0
        }
        selected = max(better, key=lambda k: better[k]) if better else default
    return {
        "candidates": report,
        "default": default if len(predictions) > 1 else selected,
        "paired_minus_default": paired,
        "selected": selected,
        "protocol": "finite-cohort-bernoulli-hoeffding_v1",
        "fraction": 0.2,
        "family_size": family_size,
        "alpha": 0.05,
        "n_validation": len(a),
        "probability": probability,
        "estimand": "finite_validation_cohort_targeting_gain",
        "assumptions": [
            "independent Bernoulli assignments",
            "disjoint context and validation units; truthful unit identifiers",
            "no interference between units",
            "binary potential outcomes",
            "pretreatment covariates",
            "no validation-driven model tuning or reuse",
        ],
    }
