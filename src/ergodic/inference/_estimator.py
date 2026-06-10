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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ._learner import _as_learner, _fresh, _is_bayesian, _is_classifier
from ._target import EffectTarget, as_target
from .learners import OLS, LogisticRegression

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Self

    import numpy.typing as npt

    from ..data import TabularData
    from ..identification import Identification
    from ._learner import Learner
    from ._result import EffectEstimate


# Estimand families, matching ``Identification.strategy``.
ADJUSTMENT = "adjustment"
FRONTDOOR = "frontdoor"
IV = "iv"


class Task(Enum):
    """What a nuisance slot asks of its learner."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"  # propensity or a discrete outcome

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Slot:
    """A named nuisance position with the learning task it requires."""

    name: str
    task: Task


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
        raise NotImplementedError(f"E1 supports a single {role}; got {sorted(names)}")
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
        f"E1 estimators need a binary {role}; {name!r} is not binary "
        "(continuous treatment and multi-level effects come later)"
    )


def _numeric_vector(data: TabularData, name: str, role: str) -> npt.NDArray[np.float64]:
    col = data.schema[name]
    if col.is_continuous:
        return np.asarray(data.column(name), dtype=np.float64)
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64)
    raise NotImplementedError(f"E1 needs a continuous or binary {role}; {name!r} is {col.kind}")


def _outcome_vector(data: TabularData, name: str) -> tuple[npt.NDArray[np.float64], bool]:
    col = data.schema[name]
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64), True
    if col.is_continuous:
        return np.asarray(data.column(name), dtype=np.float64), False
    raise NotImplementedError(f"E1 supports a continuous or binary outcome; {name!r} is {col.kind}")


def _covariate_matrix(
    data: TabularData, names: frozenset[str] | list[str]
) -> npt.NDArray[np.float64]:
    """A float design for the covariates: continuous as is, discrete one-hot (drop first)."""
    chosen = sorted(names)
    if not chosen:
        return np.empty((data.n_rows, 0), dtype=np.float64)
    blocks: list[npt.NDArray[np.float64]] = []
    for name in chosen:
        col = data.schema[name]
        if col.is_continuous or col.is_datetime:
            blocks.append(np.asarray(data.matrix([name]), dtype=np.float64))
        elif col.is_discrete:
            codes = data.codes(name)
            width = col.cardinality or 0
            onehot = (codes[:, None] == np.arange(width)[None, :]).astype(np.float64)
            if onehot.shape[1] > 1:
                blocks.append(onehot[:, 1:])
        else:
            raise ValueError(f"covariate {name!r} is excluded; it has no numeric form")
    if not blocks:
        return np.empty((data.n_rows, 0), dtype=np.float64)
    return np.asarray(np.hstack(blocks), dtype=np.float64)


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
    ) -> None:
        self._spec = learners
        self._random_state = random_state
        self._n_boot = n_boot
        self._fit_state: _FitState | None = None
        self._covariates: tuple[str, ...] = ()

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
        arrays, y_binary, diagnostics = self._extract(identification, data)
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
        return _covariate_matrix(data, list(self._covariates))

    def effect(
        self, *, target: EffectTarget | str | None = None, level: float = 0.95
    ) -> EffectEstimate:
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
        return EffectEstimate(
            target=chosen,
            estimator=self.name,
            learners=dict(state.learner_names),
            n=state.n,
            kind="bootstrap",
            diagnostics=dict(state.diagnostics),
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
        a = _binary_vector(data, treatment, "treatment")
        y, y_binary = _outcome_vector(data, outcome)
        z = _covariate_matrix(data, covariates)
        arrays = {"A": a, "Y": y, "Z": z}
        diagnostics = {
            "n_treated": float(np.sum(a == 1.0)),
            "n_control": float(np.sum(a == 0.0)),
            "n_covariates": float(z.shape[1]),
        }
        return arrays, y_binary, diagnostics


class _FrontdoorBase(CausalEstimator):
    consumes = frozenset({FRONTDOOR})

    def _extract(
        self, identification: Identification, data: TabularData
    ) -> tuple[dict[str, npt.NDArray[np.float64]], bool, dict[str, float]]:
        treatment = _single(identification.treatment, "treatment")
        outcome = _single(identification.outcome, "outcome")
        mediator = _single(identification.mediators or frozenset(), "mediator")
        _require_complete(data, {treatment, outcome, mediator})
        a = _binary_vector(data, treatment, "treatment")
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
        a = _numeric_vector(data, treatment, "treatment")
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
