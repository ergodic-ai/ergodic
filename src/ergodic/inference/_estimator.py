# SPDX-License-Identifier: Apache-2.0
"""The estimator base: consume an estimand, fill nuisance slots, return an effect.

A :class:`CausalEstimator` declares which estimand family it consumes, the named
nuisance slots it exposes, and the targets it reports. The base resolves a learner
for each slot (the user's or the default), enforces the slot contract (a
classification slot needs ``predict_proba``), and runs the shared standard-error
path: an analytic influence function when the estimator has one, bootstrap
otherwise. Bootstrap fills the result draws, so the distribution-shaped result is
exercised without a probabilistic backend.

The concrete estimators live in :mod:`ergodic.inference.estimators` and supply two
methods: ``_point`` (fit fresh learners on the given arrays and return the
estimate) and, when an influence function is known, ``_analytic``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ._crossfit import _folds, _train_mask
from ._learner import _as_learner, _fresh, _is_bayesian, _is_classifier
from ._target import EffectTarget, as_target
from .learners import OLS, LogisticRegression

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Self

    import numpy.typing as npt

    from ..data import TabularData
    from ..data._schema import ColumnType
    from ..identification import Identification
    from ._learner import Learner
    from ._perunit import CATEResult
    from ._result import EffectEstimate
    from ._scores import DRScores


# Estimand families, matching ``Identification.strategy``.
ADJUSTMENT = "adjustment"
FRONTDOOR = "frontdoor"
IV = "iv"
NULL = "null"


class Task(Enum):
    """What a nuisance slot asks of its learner."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"  # propensity or a discrete outcome

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Slot:
    """A named nuisance position with the learning task it requires.

    ``weighted`` marks a slot whose learner is fit with ``sample_weight``. A learner
    that declares ``supports_sample_weight = False`` (a remote learner whose fits
    take no row weights) is refused there when the slots resolve, before any fit.
    """

    name: str
    task: Task
    weighted: bool = False


@dataclass
class _FitState:
    arrays: dict[str, npt.NDArray[np.float64]]
    make: dict[str, Callable[[], Learner]]
    learner_names: dict[str, str]
    n: int
    diagnostics: dict[str, float]
    bayesian: dict[str, bool]


# -- extraction helpers ------------------------------------------------------


def _single(names: frozenset[str], role: str) -> str:
    if len(names) != 1:
        raise NotImplementedError(f"these estimators support a single {role}; got {sorted(names)}")
    return next(iter(names))


def _require_complete(data: TabularData, names: set[str]) -> None:
    missing = data.missing()
    dirty = sorted(name for name in names if missing.get(name, 0) > 0)
    if dirty:
        raise ValueError(f"columns {dirty} have missing values; call data.dropna() first")


def _binary_vector(data: TabularData, name: str, role: str) -> npt.NDArray[np.float64]:
    col = data.schema[name]
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64)
    if col.is_continuous:
        values = np.asarray(data.column(name), dtype=np.float64)
        if set(np.unique(values).tolist()) <= {0.0, 1.0}:
            return values
    raise NotImplementedError(
        f"this estimator needs a binary {role}; {name!r} is not binary. For a "
        "continuous treatment use strategy='dml' or 'pliv'; a multi-level "
        "treatment is not supported, so recode it to two levels"
    )


def _numeric_vector(data: TabularData, name: str, role: str) -> npt.NDArray[np.float64]:
    col = data.schema[name]
    if col.is_continuous:
        return np.asarray(data.column(name), dtype=np.float64)
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64)
    raise NotImplementedError(
        f"this estimator needs a continuous or binary {role}; {name!r} is {col.kind.value}"
    )


def _coded_treatment(data: TabularData, name: str, treated: Any) -> npt.NDArray[np.float64]:
    """Code the treatment 1 where it equals ``treated`` and 0 on the other level."""
    col = data.schema[name]
    raw = data.column(name)
    if col.is_discrete and col.cardinality == 2:
        levels: tuple[Any, ...] = tuple(col.categories or ())
    elif col.is_continuous:
        uniques = np.unique(np.asarray(raw, dtype=np.float64))
        if uniques.shape[0] != 2:
            raise ValueError(
                f"treated= needs a two-level treatment; {name!r} has {uniques.shape[0]} level(s)"
            )
        levels = tuple(uniques.tolist())
    else:
        raise ValueError(f"treated= needs a two-level treatment; {name!r} is {col.kind}")
    if treated not in levels:
        raise ValueError(
            f"treated={treated!r} is not a level of {name!r}; observed levels are {list(levels)}"
        )
    return np.asarray(np.asarray(raw) == treated, dtype=np.float64)


def _outcome_vector(data: TabularData, name: str) -> tuple[npt.NDArray[np.float64], bool]:
    col = data.schema[name]
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64), True
    if col.is_continuous:
        return np.asarray(data.column(name), dtype=np.float64), False
    raise NotImplementedError(
        f"the estimators support a continuous or binary outcome; {name!r} is {col.kind.value}"
    )


def _covariate_matrix(
    data: TabularData,
    names: frozenset[str] | list[str],
    *,
    fitted_columns: tuple[ColumnType, ...] | None = None,
) -> npt.NDArray[np.float64]:
    """A float design, using the training types and vocabulary when supplied."""
    from ..data._schema import _encode

    chosen = sorted(names)
    missing = sorted(set(chosen) - set(data.schema.names))
    if missing:
        raise ValueError(f"CATE prediction is missing fitted covariate columns: {missing}")
    _require_complete(data, set(chosen))
    columns = {col.name: col for col in fitted_columns} if fitted_columns is not None else {}
    if not chosen:
        return np.empty((data.n_rows, 0), dtype=np.float64)
    blocks: list[npt.NDArray[np.float64]] = []
    for name in chosen:
        current = data.schema[name]
        col = columns.get(name, current)
        if current.kind is not col.kind:
            raise ValueError(
                f"covariate {name!r} has type {current.kind.value!r}; "
                f"the fitted type is {col.kind.value!r}"
            )
        if col.is_continuous or col.is_datetime:
            blocks.append(np.asarray(data.matrix([name]), dtype=np.float64))
        elif col.is_discrete:
            codes = _encode(data.column(name), col.categories, col.ordered, name)
            width = col.cardinality or 0
            onehot = (codes[:, None] == np.arange(width)[None, :]).astype(np.float64)
            if onehot.shape[1] > 1:
                blocks.append(onehot[:, 1:])
        else:
            raise ValueError(f"covariate {name!r} is excluded; it has no numeric form")
    if not blocks:
        return np.empty((data.n_rows, 0), dtype=np.float64)
    return np.asarray(np.hstack(blocks), dtype=np.float64)


def _covariate_columns(
    data: TabularData, names: frozenset[str] | list[str] | tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Name every column :func:`_covariate_matrix` builds, and the variable it encodes.

    A continuous column, or a discrete one that encodes to a single indicator (two
    levels), keeps its variable name. A discrete covariate with ``k >= 3`` levels
    encodes to ``k - 1`` indicators (the first level dropped), named ``"C==level"``.
    Returns ``(column_names, column_variables)``, aligned to the design's columns.
    """
    columns: list[str] = []
    variables: list[str] = []
    for name in sorted(names):
        col = data.schema[name]
        if col.is_continuous or col.is_datetime:
            columns.append(name)
            variables.append(name)
        elif col.is_discrete:
            width = col.cardinality or 0
            if width <= 1:
                continue
            if width == 2:
                columns.append(name)
                variables.append(name)
                continue
            categories = col.categories or tuple(range(width))
            for level in categories[1:width]:
                columns.append(f"{name}=={level}")
                variables.append(name)
    return tuple(columns), tuple(variables)


def _dirichlet_weights(n: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Bayesian-bootstrap weights: one Dirichlet(1, ..., 1) draw over ``n`` rows.

    The weights sum to one, so a weighted average over them is the posterior of the
    empirical mean under a flat Dirichlet prior. Pairing one such draw with each
    nuisance posterior draw carries the sampling uncertainty alongside the model
    uncertainty.
    """
    weight = rng.standard_gamma(1.0, size=n)
    return np.asarray(weight / weight.sum(), dtype=np.float64)


# -- the base ----------------------------------------------------------------


class CausalEstimator(ABC):
    """Base for the strategies. Stateful: ``fit`` then ``effect``."""

    name: ClassVar[str]
    consumes: ClassVar[frozenset[str]]
    slots: ClassVar[tuple[Slot, ...]] = ()
    targets: ClassVar[frozenset[EffectTarget]]
    kind: ClassVar[str] = "analytic"

    def __init__(
        self,
        learners: Any = None,
        *,
        random_state: int | None = None,
        n_boot: int = 1000,
        treated: Any | None = None,
    ) -> None:
        self._spec = learners
        self._random_state = random_state
        self._n_boot = n_boot
        self._treated = treated
        self._fit_state: _FitState | None = None
        self._covariates: tuple[str, ...] = ()
        self._covariate_schema: tuple[ColumnType, ...] = ()
        self._identification: Identification | None = None
        self._treatment_name: str = ""
        self._outcome_name: str = ""

    # -- learner resolution --------------------------------------------------

    def _slot_specs(self) -> dict[str, Any]:
        names = [s.name for s in self.slots]
        spec = self._spec
        if spec is None:
            return dict.fromkeys(names)
        if isinstance(spec, Mapping):
            unknown = set(spec) - set(names)
            if unknown:
                raise ValueError(f"{self.name} has no slot(s) {sorted(unknown)}; slots are {names}")
            return {name: spec.get(name) for name in names}
        return dict.fromkeys(names, spec)

    def _default_learner(self, slot: Slot, y_binary: bool) -> Learner:
        if slot.task is Task.CLASSIFICATION:
            return LogisticRegression()
        if slot.name == "outcome" and y_binary:
            return LogisticRegression()
        return OLS()

    def _resolve(
        self, y_binary: bool
    ) -> tuple[dict[str, Callable[[], Learner]], dict[str, str], dict[str, bool]]:
        specs = self._slot_specs()
        make: dict[str, Callable[[], Learner]] = {}
        names: dict[str, str] = {}
        bayesian: dict[str, bool] = {}
        for slot in self.slots:
            spec = specs[slot.name]
            want_classifier = slot.task is Task.CLASSIFICATION
            learner = (
                self._default_learner(slot, y_binary)
                if spec is None
                else _as_learner(spec, want_classifier=want_classifier)
            )
            if slot.task is Task.CLASSIFICATION and not _is_classifier(learner):
                raise TypeError(
                    f"slot {slot.name!r} needs a classifier with predict_proba; "
                    f"got {type(learner).__name__}"
                )
            if slot.weighted and getattr(learner, "supports_sample_weight", True) is False:
                raise TypeError(
                    f"slot {slot.name!r} needs a learner whose fit accepts sample_weight; "
                    f"{type(learner).__name__} declares it does not, so nothing was fit"
                )
            make[slot.name] = partial(_fresh, learner)
            names[slot.name] = type(learner).__name__
            bayesian[slot.name] = _is_bayesian(learner)
        return make, names, bayesian

    # -- fit and effect ------------------------------------------------------

    def fit(self, identification: Identification, data: TabularData) -> Self:
        """Read the estimand and the data, resolve learners, store the fit state."""
        if not identification.identifiable:
            raise ValueError("effect is not identifiable; nothing to estimate")
        if identification.strategy not in self.consumes:
            raise ValueError(
                f"{self.name} consumes {sorted(self.consumes)}, "
                f"but the effect was identified by {identification.strategy!r}"
            )
        self._identification = identification
        arrays, y_binary, diagnostics = self._extract(identification, data)
        self._covariate_schema = tuple(data.schema[name] for name in self._covariates)
        make, learner_names, bayesian = self._resolve(y_binary)
        self._fit_state = _FitState(
            arrays=arrays,
            make=make,
            learner_names=learner_names,
            n=data.n_rows,
            diagnostics=diagnostics,
            bayesian=bayesian,
        )
        return self

    @property
    def _state(self) -> _FitState:
        if self._fit_state is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit first")
        return self._fit_state

    def _cate_covariates(self, data: TabularData | None) -> npt.NDArray[np.float64]:
        """The covariate design for CATE prediction: in-sample, or extracted from ``data``."""
        if data is None:
            return self._state.arrays["Z"]
        return _covariate_matrix(
            data, list(self._covariates), fitted_columns=self._covariate_schema
        )

    def _treatment_vector(
        self, data: TabularData, name: str, *, allow_continuous: bool = False
    ) -> npt.NDArray[np.float64]:
        """The treatment as floats, honoring ``treated`` and warning on an implicit coding."""
        if self._treated is not None:
            return _coded_treatment(data, name, self._treated)
        col = data.schema[name]
        if col.is_discrete and col.cardinality == 2:
            cats = tuple(col.categories or ())
            # {0, 1} also admits booleans, since False == 0 and True == 1
            if not set(cats) <= {0, 1}:
                warnings.warn(
                    f"treatment {name!r} was coded 1 for level {cats[1]!r} and 0 for "
                    f"{cats[0]!r} by level order; pass treated=... to choose",
                    UserWarning,
                    stacklevel=2,
                )
        if allow_continuous:
            return _numeric_vector(data, name, "treatment")
        return _binary_vector(data, name, "treatment")

    def effect(self, *, target: EffectTarget | str | None = None) -> EffectEstimate:
        """Report the effect, with an analytic standard error or a bootstrap one."""
        from ._result import EffectEstimate

        chosen = self._pick_target(target)
        state = self._state
        if self._has_bayesian_slot(state):
            return self._bayesian_effect(state, chosen)
        analytic = self._analytic(state.arrays, chosen, state.make)
        if analytic is not None:
            point, influence = analytic
            se = float(np.std(influence, ddof=1) / np.sqrt(influence.shape[0]))
            return EffectEstimate(
                target=chosen,
                estimator=self.name,
                learners=dict(state.learner_names),
                n=state.n,
                kind="analytic",
                diagnostics=dict(state.diagnostics),
                _point=point,
                _std_error=se,
            )
        rng = np.random.default_rng(self._random_state)
        point = self._point(state.arrays, chosen, state.make)
        draws = self._bootstrap(state, chosen, rng)
        diagnostics = {
            **state.diagnostics,
            "n_boot_requested": float(self._n_boot),
            "n_boot_returned": float(draws.shape[0]),
        }
        return EffectEstimate(
            target=chosen,
            estimator=self.name,
            learners=dict(state.learner_names),
            n=state.n,
            kind="bootstrap",
            diagnostics=diagnostics,
            _point=point,
            _draws=draws,
        )

    # -- the posterior path --------------------------------------------------

    def _has_bayesian_slot(self, state: _FitState) -> bool:
        """True when any resolved slot carries a posterior."""
        return any(state.bayesian.values())

    def _bayesian_effect(self, state: _FitState, target: EffectTarget) -> EffectEstimate:
        """Estimate with a Bayesian slot present, never bootstrapping.

        A bootstrap loop would refit the (expensive) Bayesian nuisance once per
        resample, so this path avoids it. If the estimator implements ``_posterior``
        for this target, return the posterior. Otherwise fall back to the analytic
        result when one exists, else a single point with no interval, with a note in
        the diagnostics. The point and the analytic both fit the nuisances once and
        read their posterior means.
        """
        from ._result import EffectEstimate

        draws = self._posterior(state, target)
        if draws is not None:
            return EffectEstimate(
                target=target,
                estimator=self.name,
                learners=dict(state.learner_names),
                n=state.n,
                kind="posterior",
                diagnostics=dict(state.diagnostics),
                _draws=draws,
            )
        diagnostics = {**state.diagnostics, "posterior_available": 0.0}
        analytic = self._analytic(state.arrays, target, state.make)
        if analytic is not None:
            point, influence = analytic
            se = float(np.std(influence, ddof=1) / np.sqrt(influence.shape[0]))
            return EffectEstimate(
                target=target,
                estimator=self.name,
                learners=dict(state.learner_names),
                n=state.n,
                kind="analytic",
                diagnostics=diagnostics,
                _point=point,
                _std_error=se,
            )
        point = self._point(state.arrays, target, state.make)
        return EffectEstimate(
            target=target,
            estimator=self.name,
            learners=dict(state.learner_names),
            n=state.n,
            kind="point",
            diagnostics=diagnostics,
            _point=point,
        )

    def _pick_target(self, target: EffectTarget | str | None) -> EffectTarget:
        if target is None:
            if EffectTarget.ATE in self.targets:
                return EffectTarget.ATE
            return sorted(self.targets, key=str)[0]
        chosen = as_target(target)
        if chosen not in self.targets:
            raise ValueError(
                f"{self.name} reports {sorted(str(t) for t in self.targets)}, not {chosen}"
            )
        return chosen

    def _bootstrap(
        self, state: _FitState, target: EffectTarget, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        n = state.n
        draws = np.empty(self._n_boot, dtype=np.float64)
        filled = 0
        attempts = 0
        budget = self._n_boot * 5
        while filled < self._n_boot and attempts < budget:
            attempts += 1
            idx = rng.integers(0, n, n)
            sub = {key: arr[idx] for key, arr in state.arrays.items()}
            try:
                value = self._point(sub, target, state.make)
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                continue
            if not np.isfinite(value):
                continue
            draws[filled] = value
            filled += 1
        if filled == 0:
            raise RuntimeError(
                f"bootstrap for {self.name} produced no finite draws in {attempts} attempts; "
                "every resample failed to estimate"
            )
        return draws[:filled]

    # -- per-estimator -------------------------------------------------------

    @abstractmethod
    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        """Pull the arrays the estimator needs, the outcome-is-binary flag, diagnostics."""

    @abstractmethod
    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        """Fit fresh learners on ``arrays`` and return the effect estimate."""

    def _analytic(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> tuple[float, npt.NDArray[np.float64]] | None:
        """Return ``(point, influence)`` when an influence function is known, else ``None``."""
        return None

    def _posterior(self, state: _FitState, target: EffectTarget) -> npt.NDArray[np.float64] | None:
        """Posterior draws of the effect when the estimator supports them.

        Called only when a slot is Bayesian. The default is ``None`` (no posterior
        path for this estimator or target), which sends ``_bayesian_effect`` to its
        analytic-or-point fallback. Implementers fit the nuisances once, read their
        ``predict_draws``, and evaluate the estimator's functional per draw.
        """
        return None


# -- family bases ------------------------------------------------------------


class _AdjustmentBase(CausalEstimator):
    consumes = frozenset({ADJUSTMENT})

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        covariates = identification.adjustment_set or frozenset()
        _require_complete(data, {treatment, outcome, *covariates})
        self._covariates = tuple(sorted(covariates))
        self._covariate_columns, self._covariate_variables = _covariate_columns(data, covariates)
        self._treatment_name = treatment
        self._outcome_name = outcome
        a = self._treatment_vector(data, treatment)
        y, y_binary = _outcome_vector(data, outcome)
        z = _covariate_matrix(data, covariates)
        arrays = {"A": a, "Y": y, "Z": z}
        n_treated = int(np.sum(a == 1.0))
        n_control = int(np.sum(a == 0.0))
        if n_treated == 0 or n_control == 0:
            raise ValueError(
                f"treatment {treatment!r} has an empty arm: "
                f"{n_treated} treated and {n_control} control rows"
            )
        diagnostics = {
            "n_treated": float(n_treated),
            "n_control": float(n_control),
            "n_covariates": float(z.shape[1]),
        }
        return arrays, y_binary, diagnostics


class _MetaLearnerBase(_AdjustmentBase):
    """Shared coherence layer for the heterogeneous-effect meta-learners.

    Factors the pattern the DR-, R-, and X-learner duplicate: fit a CATE model on
    the arrays, predict a per-unit effect on a covariate design, and report it as a
    structured :class:`~ergodic.inference.CATEResult` rather than a bare array.
    Subclasses supply two hooks, :meth:`_fit_cate` (build the model handle) and
    :meth:`_predict_cate` (per-unit prediction on a design), and the base does the
    rest: the ``cate`` surface, the per-unit uncertainty, and the doubly-robust
    scores the explainability layer reads. The ATE ``_point`` reuses the same two
    hooks, so the math is identical to the per-estimator code it replaces.

    Per-unit uncertainty, uniform across the meta-learners:

    * a Bayesian learner slot gives posterior ``(S, n)`` draws (``_cate_draws``);
    * otherwise the bootstrap path refits the CATE model on resamples and collects
      per-unit predictions into ``(S, n)`` draws (``kind="bootstrap"``);
    * the DR-learner additionally has the analytic influence-function standard error
      where the final stage is linear (``_cate_analytic_se``).
    """

    def _fit_cate(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        make: dict[str, Callable[[], Learner]],
    ) -> Any:
        """Fit the CATE model on ``arrays`` and return an opaque handle."""
        raise NotImplementedError

    def _predict_cate(self, model: Any, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Per-unit CATE on the covariate design ``z`` from a fitted handle."""
        raise NotImplementedError

    def _cate_draws(self, model: Any, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
        """Posterior per-unit CATE draws ``(S, n)`` when the slot is Bayesian, else ``None``."""
        return None

    def _cate_analytic_se(
        self, model: Any, z: npt.NDArray[np.float64], data: TabularData | None
    ) -> npt.NDArray[np.float64] | None:
        """Per-unit analytic standard errors when a closed form exists, else ``None``."""
        return None

    # -- the shared ATE path reuses the CATE model ---------------------------

    def _cate_model(self) -> Any:
        model = getattr(self, "_fitted_cate", None)
        if model is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit first")
        return model

    def fit(self, identification: Identification, data: TabularData) -> Self:
        super().fit(identification, data)
        self._fitted_cate = self._fit_cate(self._state.arrays, self._state.make)
        return self

    def _point(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
        target: EffectTarget,
        make: dict[str, Callable[[], Learner]],
    ) -> float:
        model = self._fit_cate(arrays, make)
        return float(np.mean(self._predict_cate(model, arrays["Z"])))

    # -- the uniform per-unit result -----------------------------------------

    def cate(self, data: TabularData | None = None, *, uncertainty: bool = True) -> CATEResult:
        """Per-unit CATE as a :class:`~ergodic.inference.CATEResult`.

        In-sample when ``data`` is ``None``, else scored on its covariates. The
        result carries the per-unit point, the uncertainty (posterior draws for a
        Bayesian slot, the bootstrap draws otherwise, the analytic standard error
        for the linear DR final), the identification, and the cached doubly-robust
        scores for :meth:`~ergodic.inference.CATEResult.explain`.

        Pass ``uncertainty=False`` for a fast point-only result (``kind="point"``,
        no standard error or draws): it skips the uncertainty step, which for a
        non-linear final means a full bootstrap refit of the cross-fit pipeline.
        Use it when only the effect curve is needed (ranking, targeting, plots);
        the per-unit points are identical either way. ``explain`` still works (it
        reads the cached scores, not the draws).
        """
        from ._perunit import CATEResult

        model = self._cate_model()
        z = self._cate_covariates(data)
        point = np.asarray(self._predict_cate(model, z), dtype=np.float64)
        n = point.shape[0]

        draws: npt.NDArray[np.float64] | None = None
        std_error: npt.NDArray[np.float64] | None = None
        kind = "point"
        if uncertainty:
            if self._has_bayesian_slot(self._state):
                draws = self._cate_draws(model, z)
                if draws is not None:
                    kind = "posterior"
            if draws is None:
                analytic = self._cate_analytic_se(model, z, data)
                if analytic is not None:
                    std_error = analytic
                    kind = "analytic"
                else:
                    draws = self._cate_bootstrap(z, data)
                    kind = "bootstrap"

        scores = self._cate_scores(data)
        return CATEResult(
            treatment=self._treatment_name,
            outcome=self._outcome_name,
            covariates=self._covariates,
            n=n,
            kind=kind,
            source=self.name,
            diagnostics=dict(self._state.diagnostics),
            identification=self._identification,
            _values=point,
            _index=self._cate_index(data, n),
            _std_error=std_error,
            _draws=draws,
            _scores=scores,
        )

    def _cate_bootstrap(
        self, z: npt.NDArray[np.float64], data: TabularData | None
    ) -> npt.NDArray[np.float64] | None:
        """Refit the CATE model on row resamples, predict on ``z``: ``(S, n)`` draws.

        Mirrors the scalar bootstrap (the same resample-refit machinery), but
        collects the per-unit predictions on the fixed evaluation design rather than
        a single number. ``None`` when no resample yields a finite prediction.
        """
        state = self._state
        arrays = state.arrays
        keys = list(arrays)
        nrows = arrays["Z"].shape[0]
        rng = np.random.default_rng(self._random_state)
        collected: list[npt.NDArray[np.float64]] = []
        attempts = 0
        budget = self._n_boot * 5
        while len(collected) < self._n_boot and attempts < budget:
            attempts += 1
            idx = rng.integers(0, nrows, nrows)
            sub = {key: arrays[key][idx] for key in keys}
            try:
                model = self._fit_cate(sub, state.make)
                pred = np.asarray(self._predict_cate(model, z), dtype=np.float64)
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                continue
            if not np.all(np.isfinite(pred)):
                continue
            collected.append(pred)
        if not collected:
            return None
        return np.asarray(np.vstack(collected), dtype=np.float64)

    def _cate_scores(self, data: TabularData | None) -> DRScores | None:
        """The cross-fit doubly-robust scores on the fit data, for explainability.

        Built once on the in-sample arrays, since the AIPW signal is defined over the
        observed treatment and outcome (it is not an out-of-sample prediction). When
        ``data`` is given (a fresh covariate design with no treatment/outcome), the
        scores are still those of the fit sample, so ``explain`` describes the fit
        population; that is documented on the result.
        """
        if self._identification is None or data is not None:
            return None
        from ._scores import DRScores
        from ._scores import aipw_scores as _scores_fn

        state = self._state
        a, y, z = state.arrays["A"], state.arrays["Y"], state.arrays["Z"]
        try:
            mu0, mu1, e = self._dr_nuisances(state.arrays, state.make)
        except (ValueError, np.linalg.LinAlgError):
            return None
        tau_hat, ranking_error = self._crossfit_tau_hat(state.arrays, state.make)
        return DRScores(
            scores=_scores_fn(a, y, mu0, mu1, e),
            mu0=mu0,
            mu1=mu1,
            propensity=e,
            covariates=z,
            treatment=a,
            outcome=y,
            covariate_names=self._covariate_columns,
            covariate_variables=self._covariate_variables,
            tau_hat=tau_hat,
            ranking_error=ranking_error,
        )

    def _crossfit_tau_hat(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64] | None, str | None]:
        """Out-of-fold predicted effects: a fixed proxy ranking for honest explainability.

        The literature's honesty recipe needs the ranking proxy fixed on data not used
        for the summary. Fitting the CATE model on each train fold and predicting the
        held-out fold gives an out-of-fold ``tau_hat`` that does not share rows with the
        scores it is later sorted against, so GATES, RATE, and the calibration test do
        not pick up the in-sample fit's spurious self-correlation. A failed fold
        returns no proxy and its diagnostic; ranking-based inference must refuse it.
        """
        a, z = arrays["A"], arrays["Z"]
        n = a.shape[0]
        n_folds = int(getattr(self, "n_folds", 5))
        tau = np.empty(n, dtype=np.float64)
        rng = np.random.default_rng(self._random_state)
        _fold = 0
        try:
            for _fold, test in enumerate(_folds(n, n_folds, rng, stratify=a), start=1):
                train = _train_mask(n, test)
                sub = {key: arr[train] for key, arr in arrays.items()}
                model = self._fit_cate(sub, make)
                tau[test] = self._predict_cate(model, z[test])
        except (ValueError, np.linalg.LinAlgError, TypeError) as exc:
            return None, f"cross-fit ranking fold {_fold}: {type(exc).__name__}: {exc}"
        if not np.all(np.isfinite(tau)):
            return None, "cross-fit ranking produced non-finite predictions"
        return tau, None

    def _dr_nuisances(
        self, arrays: dict[str, npt.NDArray[np.float64]], make: dict[str, Callable[[], Learner]]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Cross-fit ``mu0, mu1, e`` for the explainability scores (DR construction)."""
        from ._scores import _crossfit_nuisances

        n_folds = getattr(self, "n_folds", 5)
        clip = getattr(self, "clip", 1e-3)
        make_outcome = make.get("outcome") or (lambda: OLS())
        make_propensity = make.get("propensity") or (lambda: LogisticRegression())
        return _crossfit_nuisances(
            arrays["A"],
            arrays["Y"],
            arrays["Z"],
            make_outcome=make_outcome,
            make_propensity=make_propensity,
            n_folds=n_folds,
            clip=clip,
            random_state=self._random_state,
        )

    def _cate_index(self, data: TabularData | None, n: int) -> npt.NDArray[np.int64]:
        """Positional row ids for the scored units (the data carries no key index)."""
        return np.arange(n, dtype=np.int64)


class _FrontdoorBase(CausalEstimator):
    consumes = frozenset({FRONTDOOR})

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        mediator = _single(identification.mediators or frozenset(), "mediator")
        _require_complete(data, {treatment, outcome, mediator})
        a = self._treatment_vector(data, treatment)
        m = _binary_vector(data, mediator, "mediator")
        y, y_binary = _outcome_vector(data, outcome)
        arrays = {"A": a, "M": m, "Y": y}
        return arrays, y_binary, {}


class _IVBase(CausalEstimator):
    consumes = frozenset({IV})

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        if not identification.instruments:
            raise ValueError("no instruments on the identification")
        instrument_names = sorted({name for name, _ in identification.instruments})
        conditioning = sorted({w for _, given in identification.instruments for w in given})
        instrument_names, conditioning = self._iv_columns(instrument_names, conditioning)
        _require_complete(data, {treatment, outcome, *instrument_names, *conditioning})
        a = self._treatment_vector(data, treatment, allow_continuous=True)
        y, _ = _outcome_vector(data, outcome)
        zi = _covariate_matrix(data, instrument_names)
        w = _covariate_matrix(data, conditioning)
        arrays = {"A": a.reshape(-1, 1), "Y": y, "Zi": zi, "W": w}
        diagnostics = {"n_instruments": float(zi.shape[1])}
        return arrays, False, diagnostics

    def _iv_columns(
        self, instruments: list[str], conditioning: list[str]
    ) -> tuple[list[str], list[str]]:
        return instruments, conditioning
