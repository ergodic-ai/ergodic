# SPDX-License-Identifier: Apache-2.0
"""Augmented inverse propensity weighting (AIPW), the doubly-robust estimator."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .._estimator import Slot, Task, _AdjustmentBase, _dirichlet_weights
from .._learner import _mean_predict, _mean_predict_draws
from .._overlap import _propensity_diagnostics, _validate_overlap
from .._scores import _crossfit_nuisances
from .._scores import aipw_scores as _aipw_ate_scores
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._estimator import _FitState
    from .._learner import Learner
    from .._result import EffectEstimate


def _aipw_att_scores(
    a: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mu0: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    p1: float,
) -> npt.NDArray[np.float64]:
    """Influence scores for the ATT; their mean is the AIPW-ATT point."""
    scores = (a * (y - mu0) - (1.0 - a) * (e / (1.0 - e)) * (y - mu0)) / p1
    return np.asarray(scores, dtype=np.float64)


class AIPW(_AdjustmentBase):
    """Combine an outcome model and a propensity model.

    ``mode="auto"`` cross-fits frequentist nuisances (``n_folds=5``). A Bayesian
    slot instead uses the historical in-sample posterior/mean path, reported in
    diagnostics. ``mode="in_sample"`` explicitly selects parametric in-sample
    inference; ``mode="cross_fitted"`` refuses Bayesian slots.

    ATE inference uses the cross-fitted AIPW influence scores, not partially
    linear DML. Consistency needs at least one correct nuisance; valid intervals
    also need suitable nuisance rates, overlap and independent observations.
    Cross-fitting alone does not qualify arbitrary learners. ATT uses bootstrap.
    ``overlap="warn"`` warns on weak estimated overlap and refuses severely
    degenerate propensities; ``"raise"`` refuses both and ``"ignore"`` explicitly
    overrides the diagnostic guard (never invalid probability validation).
    """

    name = "aipw"
    slots = (Slot("outcome", Task.REGRESSION), Slot("propensity", Task.CLASSIFICATION))
    targets = frozenset({EffectTarget.ATE, EffectTarget.ATT})

    def __init__(
        self,
        learners: Any = None,
        *,
        clip: float = 1e-3,
        bayesian_bootstrap: bool = True,
        mode: str = "auto",
        n_folds: int = 5,
        overlap: str = "warn",
        overlap_threshold: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(learners, **kwargs)
        _validate_overlap(clip, overlap, overlap_threshold)
        if mode not in {"auto", "cross_fitted", "in_sample"}:
            raise ValueError("mode must be 'auto', 'cross_fitted', or 'in_sample'")
        if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError("n_folds must be an integer >= 2")
        self.clip = clip
        self.bayesian_bootstrap = bayesian_bootstrap
        self.mode = mode
        self.n_folds = n_folds
        self.overlap = overlap
        self.overlap_threshold = overlap_threshold
        self._resolved_mode = "in_sample"
        self._nuisance_cache: (
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] | None
        ) = None

    def fit(self, identification: Identification, data: TabularData) -> Self:
        self._nuisance_cache = None
        super().fit(identification, data)
        bayesian = self._has_bayesian_slot(self._state)
        if self.mode == "cross_fitted" and bayesian:
            raise ValueError(
                "cross_fitted AIPW does not implement Bayesian posteriors; use mode='auto' "
                "or 'in_sample'"
            )
        self._resolved_mode = (
            "in_sample" if self.mode == "in_sample" or bayesian else "cross_fitted"
        )
        a = self._state.arrays["A"]
        if not np.any(a == 0) or not np.any(a == 1):
            raise ValueError("AIPW requires both treatment arms")
        if self._resolved_mode == "cross_fitted":
            if self.n_folds > a.size:
                raise ValueError("n_folds must not exceed n_rows")
            if min(np.sum(a == 0), np.sum(a == 1)) < 2:
                raise ValueError(
                    "cross-fitting needs at least two observations in each treatment arm"
                )
        self._state.diagnostics.update(
            {
                "cross_fitted": float(self._resolved_mode == "cross_fitted"),
                "n_folds": float(
                    min(self.n_folds, max(np.sum(a == 0), np.sum(a == 1)))
                    if self._resolved_mode == "cross_fitted"
                    else 1
                ),
                "n_folds_requested": float(self.n_folds),
                "bayesian_in_sample": float(bayesian),
                "crossfit_seed": float(self._random_state if self._random_state is not None else 0),
            }
        )
        return self

    def effect(self, *, target: EffectTarget | str | None = None) -> EffectEstimate:
        result = super().effect(target=target)
        assumptions: tuple[str, ...] = (
            "Adjustment identification, consistency, no interference and independent observations.",
            "Overlap and suitable nuisance convergence are required; cross-fitting and "
            "clipping do not guarantee valid inference for arbitrary learners.",
        )
        if self._has_bayesian_slot(self._state):
            assumptions += (
                "Bayesian nuisance fitting uses the in-sample model posterior; this is not "
                "cross-fitted frequentist inference.",
            )
        elif self._resolved_mode == "in_sample":
            assumptions += (
                "In-sample nuisance inference requires appropriate "
                "parametric/empirical-process conditions; unrestricted interpolating "
                "learners are not qualified.",
            )
        return replace(
            result,
            diagnostics={**result.diagnostics, **self._state.diagnostics},
            assumptions=assumptions,
        )

    def _overlap_diagnostics(
        self, a: npt.NDArray[np.float64], raw: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        return _propensity_diagnostics(
            a, raw, clip=self.clip, overlap=self.overlap, threshold=self.overlap_threshold
        )

    def _nuisance(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        original = arrays is self._state.arrays
        if original and self._nuisance_cache is not None:
            return self._nuisance_cache
        a = arrays["A"]
        y = arrays["Y"]
        z = arrays["Z"]
        n = a.shape[0]
        if self._resolved_mode == "cross_fitted":
            raw = np.empty(n, dtype=np.float64)
            result = _crossfit_nuisances(
                a,
                y,
                z,
                make_outcome=make["outcome"],
                make_propensity=make["propensity"],
                n_folds=self.n_folds,
                clip=self.clip,
                random_state=self._random_state if self._random_state is not None else 0,
                raw_propensity=raw,
            )
            diagnostics = self._overlap_diagnostics(a, raw)
            if original:
                self._state.diagnostics.update(diagnostics)
                self._nuisance_cache = result
            return result
        outcome = make["outcome"]().fit(np.hstack([a.reshape(-1, 1), z]), y)
        ones = np.ones((n, 1), dtype=np.float64)
        zeros = np.zeros((n, 1), dtype=np.float64)
        mu1 = _mean_predict(outcome, np.hstack([ones, z]))
        mu0 = _mean_predict(outcome, np.hstack([zeros, z]))
        propensity = make["propensity"]().fit(z, a)
        raw = _mean_predict(propensity, z)
        diagnostics = self._overlap_diagnostics(a, raw)
        e = np.clip(raw, self.clip, 1.0 - self.clip)
        result = (mu0, mu1, e)
        if original:
            self._state.diagnostics.update(diagnostics)
            self._nuisance_cache = result
        return result

    def _scores(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> npt.NDArray[np.float64]:
        a = arrays["A"]
        y = arrays["Y"]
        mu0, mu1, e = self._nuisance(arrays, make)
        if target is EffectTarget.ATT:
            p1 = float(np.mean(a))
            if p1 <= 0.0:
                raise ValueError("no treated units for ATT")
            return _aipw_att_scores(a, y, mu0, e, p1)
        return _aipw_ate_scores(a, y, mu0, mu1, e)

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        return float(np.mean(self._scores(arrays, target, make)))

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        if target is not EffectTarget.ATE:
            return None
        psi = self._scores(arrays, target, make)
        return float(np.mean(psi)), psi

    def _posterior(self, state: _FitState, target: EffectTarget) -> npt.NDArray[np.float64] | None:
        """The doubly-robust posterior: the influence scores per nuisance draw.

        Fit the outcome and propensity once, read their ``(S, n)`` draws, and for
        each draw evaluate the same influence scores as the point path. With
        ``bayesian_bootstrap`` (the default) reweight the rows by a Dirichlet(1)
        draw, so the spread carries the nuisance posterior and the sampling
        uncertainty together. Needs both nuisances Bayesian; a mix returns ``None``.
        """
        if not all(state.bayesian.values()):
            return None
        arrays = state.arrays
        a, y, z = arrays["A"], arrays["Y"], arrays["Z"]
        n = a.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        zeros = np.zeros((n, 1), dtype=np.float64)
        outcome = state.make["outcome"]().fit(np.hstack([a.reshape(-1, 1), z]), y)
        mu1d = _mean_predict_draws(outcome, np.hstack([ones, z]))
        mu0d = _mean_predict_draws(outcome, np.hstack([zeros, z]))
        propensity = state.make["propensity"]().fit(z, a)
        raw_draws = _mean_predict_draws(propensity, z)
        if not np.all(np.isfinite(raw_draws)) or np.any((raw_draws < 0) | (raw_draws > 1)):
            raise ValueError("propensity draws must be finite probabilities in [0, 1]")
        self._state.diagnostics.update(self._overlap_diagnostics(a, np.mean(raw_draws, axis=0)))
        self._state.diagnostics["posterior_draw_clipped_fraction"] = float(
            np.mean((raw_draws < self.clip) | (raw_draws > 1.0 - self.clip))
        )
        self._state.diagnostics["posterior_draw_tail_fraction"] = float(
            np.mean(
                (raw_draws < self.overlap_threshold) | (raw_draws > 1.0 - self.overlap_threshold)
            )
        )
        ed = np.clip(raw_draws, self.clip, 1.0 - self.clip)
        n_draws = min(mu1d.shape[0], mu0d.shape[0], ed.shape[0])
        p1 = float(np.mean(a))
        if target is EffectTarget.ATT and p1 <= 0.0:
            raise ValueError("no treated units for ATT")
        rng = np.random.default_rng(self._random_state)
        draws = np.empty(n_draws, dtype=np.float64)
        for s in range(n_draws):
            w = _dirichlet_weights(n, rng) if self.bayesian_bootstrap else None
            if target is EffectTarget.ATT:
                if w is None:
                    phi = _aipw_att_scores(a, y, mu0d[s], ed[s], p1)
                else:
                    # the treated share must follow the same reweighted measure
                    p1_w = float(np.sum(w * a))
                    if p1_w <= 0.0:
                        raise ValueError("a Bayesian-bootstrap draw left no treated weight for ATT")
                    phi = _aipw_att_scores(a, y, mu0d[s], ed[s], p1_w)
            else:
                phi = _aipw_ate_scores(a, y, mu0d[s], mu1d[s], ed[s])
            draws[s] = float(np.mean(phi)) if w is None else float(np.sum(w * phi))
        return draws
