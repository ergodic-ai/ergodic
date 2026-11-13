# SPDX-License-Identifier: Apache-2.0
"""Cross-fitted kernel DR for scalar continuous treatment (API-219).

Population score: Colangelo & Lee (2022), equation (1),
https://arxiv.org/html/2004.03036v7 . Both nuisance functions are evaluated
at the requested dose, not at the observed treatment. Double robustness is
asymptotic as bandwidth shrinks; finite bandwidth introduces smoothing bias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from .._contrast import DoseContrast, DoseEffects, _in_range, _levels
from .._crossfit import _folds, _train_mask
from .._estimator import (
    Slot,
    Task,
    _AdjustmentBase,
    _covariate_matrix,
    _outcome_vector,
    _require_complete,
    _single,
)
from .._learner import _fresh, _is_classifier, _mean_predict
from .._target import EffectTarget
from ..learners import OLS

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from .._learner import Learner


class TreatmentDensity(Protocol):
    """Conditional density learner; ``pdf(X, treatment)`` returns one density/row.

    Density is in inverse treatment units, not a probability or predicted mean.
    Objects must support fresh cloning (get_params or deepcopy), as learners do.
    """

    def fit(self, X: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]) -> Any: ...

    def pdf(
        self, X: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...


class GaussianTreatmentDensity:
    """Homoskedastic Gaussian treatment model, with an OLS conditional mean.

    This explicit parametric assumption is a baseline, not a universal density
    model. Supply a different TreatmentDensity for non-Gaussian/heteroskedastic
    treatment. Fit only receives the outer training fold.
    """

    def __init__(self) -> None:
        self._model: OLS | None = None
        self._scale = 0.0

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {}

    def fit(self, X: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]) -> Self:
        self._model = None
        model = OLS().fit(X, treatment)
        residual = treatment - model.predict(X)
        rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X])))
        if len(X) <= rank:
            raise ValueError("not enough rows to estimate conditional treatment density")
        variance = float(np.sum(residual**2) / (len(X) - rank))
        if not np.isfinite(variance) or variance <= np.finfo(float).eps * np.var(treatment):
            raise ValueError("no residual variation for conditional treatment density")
        self._scale = float(np.sqrt(variance))
        self._model = model
        return self

    def pdf(
        self, X: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("treatment density is not fitted")
        residual = (treatment - self._model.predict(X)) / self._scale
        return np.asarray(np.exp(-0.5 * residual**2) / (np.sqrt(2 * np.pi) * self._scale))


class ContinuousDR(_AdjustmentBase):
    """Scalar continuous-dose DR, with explicit ``contrast(baseline, value)``.

    ``learners`` supplies the outcome learner or an ``outcome``/``final`` map.
    ``density`` is a separate conditional-density learner. The final regressor is
    used only for conditional effects on new rows (``effects``); these are point
    predictions without per-row confidence intervals.

    Epanechnikov bandwidth defaults to sd(T)*n**(-1/4), an undersmoothing rule,
    not a tuned optimum. Density below ``min_density`` (inverse treatment units)
    refuses the contrast rather than silently clipping weights. Support checks
    cannot prove positivity. No effect, probability or interval is clipped.
    """

    name = "continuous_dr"
    slots = (Slot("outcome", Task.REGRESSION), Slot("final", Task.REGRESSION))
    targets = frozenset({EffectTarget.ATE})

    def __init__(
        self,
        learners: Any = None,
        *,
        density: TreatmentDensity | None = None,
        bandwidth: float | None = None,
        n_folds: int = 5,
        min_density: float = 1e-6,
        min_effective_n: float = 10.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(learners, random_state=random_state, n_boot=0)
        if bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError("bandwidth must be finite and positive")
        if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
            raise ValueError("n_folds must be an integer >= 2")
        if not np.isfinite(min_density) or min_density <= 0:
            raise ValueError("min_density must be finite and positive")
        if not np.isfinite(min_effective_n) or min_effective_n < 2:
            raise ValueError("min_effective_n must be finite and >= 2")
        self.density = density if density is not None else GaussianTreatmentDensity()
        if not callable(getattr(self.density, "pdf", None)):
            raise TypeError("density needs fit and pdf; a treatment regressor is not a density")
        self.bandwidth = bandwidth
        self.n_folds = n_folds
        self.min_density = min_density
        self.min_effective_n = min_effective_n
        self.bandwidth_: float | None = None
        self._models: list[tuple[npt.NDArray[np.int64], Learner, TreatmentDensity]] = []
        self._training_data: TabularData | None = None
        self._scores_cache: dict[float, tuple[npt.NDArray[np.float64], dict[str, float]]] = {}
        self._final_models: dict[tuple[float, float], Learner] = {}

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        if not data.schema[treatment].is_continuous:
            raise ValueError("continuous_dr requires a continuous numerical treatment")
        covariates = identification.adjustment_set or frozenset()
        _require_complete(data, {treatment, outcome, *covariates})
        self._covariates = tuple(sorted(covariates))
        self._treatment_name, self._outcome_name = treatment, outcome
        a = np.asarray(data.column(treatment), dtype=np.float64)
        y, binary = _outcome_vector(data, outcome)
        if (
            binary
            and data.schema[outcome].is_discrete
            and not set(data.schema[outcome].categories or ()) <= {0, 1}
        ):
            raise ValueError("encode the desired binary outcome as 1 and the other as 0")
        # Category codes depend on declared order; the event contract does not.
        if binary:
            y = np.asarray(data.column(outcome), dtype=np.float64)
        elif set(np.unique(y)) == {0.0, 1.0}:
            binary = True
        z = _covariate_matrix(data, covariates)
        if any(not np.isfinite(arr).all() for arr in (a, y, z)):
            raise ValueError("continuous_dr needs finite treatment, outcome and covariates")
        if len(np.unique(a)) < 3 or np.std(a) <= 0:
            raise ValueError("continuous_dr needs at least three distinct treatment values")
        if data.n_rows < max(4, self.n_folds):
            raise ValueError("not enough context rows for n_folds")
        return {"A": a, "Y": y, "Z": z}, binary, {"n_covariates": float(z.shape[1])}

    def fit(self, identification: Identification, data: TabularData) -> Self:
        self._fit_state = None
        self._models = []
        self._scores_cache = {}
        self._final_models = {}
        self._training_data = None
        self.bandwidth_ = None
        try:
            super().fit(identification, data)
            state = self._state
            a, y, z = (state.arrays[key] for key in ("A", "Y", "Z"))
            if _is_classifier(state.make["final"]()):
                raise TypeError("final learner must regress numerical effect scores")
            self.bandwidth_ = (
                float(self.bandwidth)
                if self.bandwidth is not None
                else float(np.std(a, ddof=1) * len(a) ** (-0.25))
            )
            for test in _folds(len(a), self.n_folds, np.random.default_rng(self._random_state)):
                train = _train_mask(len(a), test)
                model = state.make["outcome"]().fit(np.column_stack([a[train], z[train]]), y[train])
                density = cast("TreatmentDensity", _fresh(cast("Learner", self.density)))
                density.fit(z[train], a[train])
                self._models.append((test, model, density))
            state.learner_names["density"] = type(self.density).__name__
            self._training_data = data
        except Exception:
            self._fit_state = None
            self._models = []
            self.bandwidth_ = None
            raise
        return self

    def _density_at(
        self, density: TreatmentDensity, z: npt.NDArray[np.float64], dose: float
    ) -> npt.NDArray[np.float64]:
        f = np.asarray(density.pdf(z, np.full(len(z), dose)), dtype=np.float64)
        if f.shape != (len(z),) or not np.isfinite(f).all() or np.any(f < self.min_density):
            raise ValueError("insufficient conditional treatment-density support at requested dose")
        return f

    def _dose_scores(self, dose: float) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        state = self._state
        if dose in self._scores_cache:
            return self._scores_cache[dose]
        a, y, z = (state.arrays[key] for key in ("A", "Y", "Z"))
        _in_range(a, dose)
        h = self.bandwidth_
        assert h is not None
        if dose - h < a.min() or dose + h > a.max():
            raise ValueError("dose is too close to the observed boundary for this bandwidth")
        u = (a - dose) / h
        kernel = 0.75 * np.maximum(1.0 - u**2, 0.0) / h
        scores, weights = np.empty(len(a)), np.empty(len(a))
        min_density = float("inf")
        for test, model, density in self._models:
            f = self._density_at(density, z[test], dose)
            mu = np.asarray(
                _mean_predict(model, np.column_stack([np.full(len(test), dose), z[test]]))
            )
            if mu.shape != (len(test),) or not np.isfinite(mu).all():
                raise ValueError("outcome learner must return one finite mean per row")
            weights[test] = kernel[test] / f
            scores[test] = mu + weights[test] * (y[test] - mu)
            min_density = min(min_density, float(f.min()))
        squared = float(np.sum(weights**2))
        effective_n = float(np.sum(weights) ** 2 / squared) if squared else 0.0
        if effective_n < self.min_effective_n:
            raise ValueError("insufficient effective local sample size at requested dose")
        if not np.isfinite(scores).all():
            raise ValueError("continuous DR scores are not finite")
        diagnostics = {"effective_n": effective_n, "min_density": min_density}
        self._scores_cache[dose] = scores, diagnostics
        return scores, diagnostics

    def _contrast_scores(
        self, baseline: float, value: float
    ) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        low, high = self._dose_scores(baseline), self._dose_scores(value)
        diagnostics = {
            **self._state.diagnostics,
            "bandwidth": float(cast("float", self.bandwidth_)),
            "n_folds": float(self.n_folds),
            "baseline_effective_n": low[1]["effective_n"],
            "value_effective_n": high[1]["effective_n"],
            "min_density": min(low[1]["min_density"], high[1]["min_density"]),
            "smoothing_bias_corrected": 0.0,
            "graph_selection_accounted_for": 0.0,
        }
        return high[0] - low[0], diagnostics

    def contrast(self, *, baseline: float, value: float) -> DoseContrast:
        """Population point-dose contrast, with a pointwise asymptotic SE.

        The SE uses paired cross-fitted scores; no independent-dose assumption.
        Valid inference additionally requires negligible smoothing bias and the
        density/outcome convergence rates in the reference, not merely one
        correct nuisance. Bandwidth selection is not repeated for each dose.
        """
        baseline, value = _levels(baseline, value)
        scores, diagnostics = self._contrast_scores(baseline, value)
        return DoseContrast(
            target=EffectTarget.ATE,
            estimator=self.name,
            learners=dict(self._state.learner_names),
            n=len(scores),
            kind="analytic",
            diagnostics=diagnostics,
            _point=float(np.mean(scores)),
            _std_error=float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
            baseline=baseline,
            treatment_value=value,
            treatment=self._treatment_name,
            outcome=self._outcome_name,
            protocol="continuous-dr-epanechnikov-v1",
        )

    def _query_design(self, data: TabularData) -> npt.NDArray[np.float64]:
        _ = self._state
        assert self._training_data is not None
        _require_complete(data, set(self._covariates))
        blocks = []
        for name in self._covariates:
            col = self._training_data.schema[name]
            raw = data.column(name)
            if col.is_discrete:
                categories = tuple(col.categories or ())
                if not all(v in categories for v in raw):
                    raise ValueError(f"unknown covariate category for {name!r}")
                if len(categories) > 1:
                    blocks.append(np.column_stack([raw == c for c in categories[1:]]))
            elif col.is_datetime:
                if not data.schema[name].is_datetime:
                    raise ValueError(f"query covariate {name!r} must be datetime")
                stamps = np.asarray(raw, dtype="datetime64[ns]")
                numeric = stamps.view(np.int64).astype(np.float64)
                numeric[np.isnat(stamps)] = np.nan
                blocks.append(numeric[:, None])
            else:
                # Never use the query's inferred category codes for a fitted
                # numerical feature (e.g. integer [2, 3] must stay [2, 3]).
                blocks.append(np.asarray(raw, dtype=np.float64)[:, None])
        z = (
            np.asarray(np.column_stack(blocks), dtype=np.float64)
            if blocks
            else np.empty((data.n_rows, 0))
        )
        if not np.isfinite(z).all():
            raise ValueError("query covariates must be finite")
        return z

    def effects(self, data: TabularData, *, baseline: float, value: float) -> DoseEffects:
        """Conditional effects on query rows; final-regression point estimates.

        Fit the final regressor to cross-fitted contrast scores, then predict
        with only the adjustment covariates from the query. No query outcomes or
        treatments are used. Query means are not transport-adjusted population
        DR estimates. Graph/density/final-model assumptions remain explicit.
        """
        baseline, value = _levels(baseline, value)
        scores, diagnostics = self._contrast_scores(baseline, value)
        z = self._query_design(data)
        for _, _, density in self._models:
            self._density_at(density, z, baseline)
            self._density_at(density, z, value)
        key = (baseline, value)
        if key not in self._final_models:
            self._final_models[key] = self._state.make["final"]().fit(
                self._state.arrays["Z"], scores
            )
        final = self._final_models[key]
        values = np.asarray(final.predict(z), dtype=np.float64)
        if values.shape != (len(z),) or not np.isfinite(values).all():
            raise ValueError("final learner must return one finite effect per query row")
        return DoseEffects(
            treatment=self._treatment_name,
            outcome=self._outcome_name,
            covariates=self._covariates,
            n=len(z),
            kind="point",
            source=self.name,
            diagnostics=diagnostics,
            _values=values,
            identification=self._identification,
            baseline=baseline,
            treatment_value=value,
            protocol="continuous-dr-epanechnikov-v1",
        )

    def effect(self, *, target: EffectTarget | str | None = None) -> DoseContrast:
        raise ValueError(
            "continuous_dr needs explicit doses; use contrast(baseline=..., value=...)"
        )

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        raise ValueError("continuous_dr needs explicit doses; use contrast")
