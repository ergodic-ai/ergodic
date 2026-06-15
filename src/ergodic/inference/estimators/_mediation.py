# SPDX-License-Identifier: Apache-2.0
"""The mediation estimator: g-computation for the natural and controlled effects.

:func:`estimate_mediation` identifies the decomposition (through
:func:`~ergodic.identification.identify_mediation`), then estimates it by
g-computation (the simulation form of the mediation formula). It fits two models,
the outcome model ``mu(x, m, w) = E[Y | X=x, M=m, W=w]`` and the mediator model
``p(m | x, w) = P(M=m | X=x, W=w)``, and plugs them into the formula, summing over
the mediator's discrete levels and averaging over the empirical covariates:

    NDE = mean_i sum_m [mu(1, m, w_i) - mu(0, m, w_i)] p(m | 0, w_i)
    NIE = mean_i sum_m mu(0, m, w_i) [p(m | 1, w_i) - p(m | 0, w_i)]
    TE  = NDE + NIE
    CDE(m) = mean_i [mu(1, m, w_i) - mu(0, m, w_i)]   (per mediator level m)

It is the MVP: a binary treatment, a discrete (binary or low-cardinality)
mediator, and a continuous or binary outcome. A continuous mediator raises with a
pointer at the future multiply-robust / density path. The standard error is a
nonparametric bootstrap over the rows, refitting both models per resample, which
also carries the model uncertainty.

Reuses the inference machinery: ``_covariate_matrix`` for the covariate design,
the :class:`~ergodic.inference.Learner` resolution, and the
:class:`~ergodic.inference.EffectEstimate` shape (one per quantity). The estimator
is a self-contained function rather than a :class:`CausalEstimator`, since it
reports a bundle (NDE, NIE, TE, CDE) rather than a single contrast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...identification import identify_mediation
from .._design import Design
from .._estimator import _coded_treatment, _covariate_matrix
from .._learner import _as_learner, _fresh, _is_classifier, _mean_predict
from .._result import EffectEstimate
from .._target import EffectTarget

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData
    from ...graph import MixedGraph
    from ...identification import MediationID
    from .._learner import Learner


# -- the result --------------------------------------------------------------


@dataclass(frozen=True)
class MediationEstimate:
    """A mediation decomposition with its uncertainty.

    ``nde``, ``nie``, and ``total`` are the natural direct effect, the natural
    indirect effect, and the total effect (``nde + nie``), each a distribution-
    shaped :class:`~ergodic.inference.EffectEstimate` with a bootstrap standard
    error and interval. ``cde`` maps each mediator level to the controlled direct
    effect at that level (also an :class:`EffectEstimate`). ``design`` names the
    (untestable) assumptions the decomposition rests on, the same role
    :class:`~ergodic.inference.Design` plays for the quasi-experimental estimators.

    ``proportion_mediated`` is ``nie / total``, the share of the total effect that
    flows through the mediator. It is unstable when the total effect is near zero
    (a small denominator), so it is offered with that caveat and a ``flagged``
    diagnostic.
    """

    nde: EffectEstimate
    nie: EffectEstimate
    total: EffectEstimate
    cde: dict[Any, EffectEstimate]
    design: Design
    mediators: frozenset[str]
    covariates: frozenset[str]

    @property
    def proportion_mediated(self) -> float:
        """``NIE / TE``: the share of the total effect carried by the mediator.

        Unstable when the total effect is near zero; see
        :attr:`proportion_mediated_flagged`.
        """
        te = self.total.value
        return self.nie.value / te if te != 0.0 else float("nan")

    @property
    def proportion_mediated_flagged(self) -> bool:
        """``True`` when the total effect is too small to trust the proportion.

        The proportion mediated has the total effect in its denominator, so a
        total effect close to zero makes it numerically unstable: a tiny shift in
        either piece swings the ratio. Flagged when ``|TE|`` is below one bootstrap
        standard error (or essentially zero).
        """
        te = self.total.value
        se = self.total.std_error or 0.0
        return abs(te) <= max(se, 1e-8)

    def __repr__(self) -> str:
        pm = self.proportion_mediated
        flag = " (unstable)" if self.proportion_mediated_flagged else ""
        return (
            f"MediationEstimate(NDE={self.nde.value:.4g}, NIE={self.nie.value:.4g}, "
            f"TE={self.total.value:.4g}, proportion_mediated={pm:.3g}{flag})"
        )


# -- the g-computation core --------------------------------------------------


@dataclass(frozen=True)
class _Design:
    """The arrays and resolved learners the g-computation reads, fixed once."""

    a: npt.NDArray[np.float64]  # treatment, 0/1, shape (n,)
    m_codes: npt.NDArray[np.int64]  # mediator codes 0..K-1, shape (n,)
    y: npt.NDArray[np.float64]  # outcome, shape (n,)
    w: npt.NDArray[np.float64]  # covariate design, shape (n, p)
    levels: tuple[int, ...]  # the mediator's distinct codes
    y_binary: bool
    make_outcome: Any  # () -> Learner
    make_mediator: Any  # () -> Learner


def _fit_outcome(design: _Design, idx: npt.NDArray[np.int64]) -> Learner:
    """Fit ``mu(x, m, w)``: the outcome on treatment, the mediator one-hot, and W."""
    features = _outcome_features(design, design.a[idx], design.m_codes[idx], design.w[idx])
    model: Learner = design.make_outcome()
    model.fit(features, design.y[idx])
    return model


def _outcome_features(
    design: _Design,
    a: npt.NDArray[np.float64],
    m_codes: npt.NDArray[np.int64],
    w: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """``[a, one-hot(m) drop-first, a * one-hot(m), w]`` for the outcome model.

    The mediator enters one-hot (dropping the first level) so a multi-level
    discrete mediator is linear-model friendly; a binary mediator is then a single
    indicator column. The treatment-by-mediator interaction terms are explicit, so
    even a linear outcome model fits a distinct ``E[Y | x, m, w]`` per ``(x, m)``
    cell. Without them the controlled direct effect would be forced constant across
    mediator levels (a misspecification that also biases NDE/NIE under an
    interaction); with them the model is saturated in ``(x, m)``.
    """
    onehot = _mediator_onehot(m_codes, design.levels)
    interaction = onehot * a.reshape(-1, 1)
    return np.asarray(np.hstack([a.reshape(-1, 1), onehot, interaction, w]), dtype=np.float64)


def _mediator_onehot(
    m_codes: npt.NDArray[np.int64], levels: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """One-hot of the mediator codes over ``levels``, dropping the first level."""
    cols = [(m_codes == level).astype(np.float64) for level in levels[1:]]
    if not cols:
        return np.empty((m_codes.shape[0], 0), dtype=np.float64)
    return np.asarray(np.column_stack(cols), dtype=np.float64)


def _mu_at(
    design: _Design, model: Learner, x_level: float, m_level: int, w: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Predicted ``mu(x_level, m_level, w_i)`` for every row of ``w``."""
    n = w.shape[0]
    a = np.full(n, x_level, dtype=np.float64)
    m_codes = np.full(n, m_level, dtype=np.int64)
    return _mean_predict(model, _outcome_features(design, a, m_codes, w))


def _fit_mediator(design: _Design, idx: npt.NDArray[np.int64]) -> list[Learner]:
    """Fit ``p(m | x, w)`` as one binary classifier per mediator level (one-vs-rest).

    Each level's model predicts the indicator ``M == level`` from ``[x, w]``; the
    per-level class-1 probabilities are normalized across levels at prediction
    time. This keeps the native binary classifier in play for a low-cardinality
    mediator and reduces to a single classifier for a binary one. A level absent
    from a resample gets a constant-zero model so its probability is zero.
    """
    features = _mediator_features(design.a[idx], design.w[idx])
    m_codes = design.m_codes[idx]
    models: list[Learner] = []
    for level in design.levels:
        indicator = (m_codes == level).astype(np.float64)
        if indicator.sum() == 0.0 or indicator.sum() == indicator.shape[0]:
            models.append(_ConstantClassifier(float(indicator.mean())))
        else:
            models.append(design.make_mediator().fit(features, indicator))
    return models


def _mediator_features(
    a: npt.NDArray[np.float64], w: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """``[a, w]`` for the mediator model."""
    return np.asarray(np.hstack([a.reshape(-1, 1), w]), dtype=np.float64)


def _p_m_given(
    design: _Design, models: list[Learner], x_level: float, w: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The mediator-distribution matrix ``p(m | x_level, w_i)``, shape ``(n, K)``.

    Column ``k`` is ``P(M = levels[k] | X = x_level, W = w_i)``. The per-level
    one-vs-rest probabilities are normalized to sum to one across the levels, so
    the result is a proper distribution even when the binary classifiers do not
    agree exactly.
    """
    n = w.shape[0]
    a = np.full(n, x_level, dtype=np.float64)
    features = _mediator_features(a, w)
    raw = np.column_stack([_class1_proba(model, features) for model in models])
    total = raw.sum(axis=1, keepdims=True)
    degenerate = (total == 0.0)[:, 0]
    if degenerate.any():
        # Every one-vs-rest classifier returned 0 for this cell; fall back to a
        # uniform mediator distribution rather than an all-zero (non-unit) row.
        raw[degenerate] = 1.0
        total[degenerate] = float(raw.shape[1])
    return np.asarray(raw / total, dtype=np.float64)


def _class1_proba(model: Learner, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The class-1 probability from a classifier, shape ``(n,)``."""
    proba = getattr(model, "predict_proba", None)
    if proba is None:  # pragma: no cover - resolution guarantees a classifier
        raise TypeError("the mediator model must be a classifier with predict_proba")
    return np.asarray(proba(features), dtype=np.float64)[:, 1]


class _ConstantClassifier:
    """A degenerate classifier returning a fixed class-1 probability.

    Used for a mediator level that is constant in a resample (all-present or
    all-absent), where a real classifier has nothing to separate.
    """

    def __init__(self, p1: float) -> None:
        self._p1 = p1

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sample_weight: npt.NDArray[np.float64] | None = None,
    ) -> _ConstantClassifier:
        return self

    def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = x.shape[0]
        return np.asarray(np.column_stack([np.full(n, 1.0 - self._p1), np.full(n, self._p1)]))

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(self.predict_proba(x)[:, 1] >= 0.5, dtype=np.float64)


@dataclass(frozen=True)
class _Effects:
    """The four mediation quantities from one fit: NDE, NIE, TE, and CDE per level."""

    nde: float
    nie: float
    total: float
    cde: dict[int, float]


def _compute(design: _Design, idx: npt.NDArray[np.int64]) -> _Effects:
    """Fit the two models on rows ``idx`` and evaluate the mediation formula.

    Predictions and the outer average are taken over the *full* sample's
    covariates (``design.w``), so the target population is ``P(W)`` regardless of
    the resample; the models are what the resample perturbs.
    """
    outcome = _fit_outcome(design, idx)
    mediator = _fit_mediator(design, idx)
    w = design.w
    levels = design.levels

    mu1 = {m: _mu_at(design, outcome, 1.0, m, w) for m in levels}
    mu0 = {m: _mu_at(design, outcome, 0.0, m, w) for m in levels}
    p0 = _p_m_given(design, mediator, 0.0, w)
    p1 = _p_m_given(design, mediator, 1.0, w)

    nde_rows = np.zeros(w.shape[0], dtype=np.float64)
    nie_rows = np.zeros(w.shape[0], dtype=np.float64)
    for k, m in enumerate(levels):
        nde_rows += (mu1[m] - mu0[m]) * p0[:, k]
        nie_rows += mu0[m] * (p1[:, k] - p0[:, k])
    nde = float(np.mean(nde_rows))
    nie = float(np.mean(nie_rows))
    cde = {m: float(np.mean(mu1[m] - mu0[m])) for m in levels}
    return _Effects(nde=nde, nie=nie, total=nde + nie, cde=cde)


# -- guards and learner resolution -------------------------------------------


def _resolve_learners(learner: Any) -> tuple[Any, Any]:
    """Resolve the outcome and mediator learner factories from the ``learner`` argument.

    ``learner`` is an alias, a learner-like object, or a slot mapping with keys
    ``"outcome"`` and/or ``"mediator"``. The outcome slot defaults to OLS (or
    logistic for a binary outcome, handled at fit), the mediator slot to logistic;
    the mediator slot must be a classifier.
    """
    if isinstance(learner, dict):
        unknown = set(learner) - {"outcome", "mediator"}
        if unknown:
            raise ValueError(
                f"mediation has slots 'outcome' and 'mediator'; unknown {sorted(unknown)}"
            )
        return learner.get("outcome"), learner.get("mediator")
    return learner, learner


def _make_outcome(spec: Any, y_binary: bool) -> Any:
    from ..learners import OLS, LogisticRegression

    if spec is None:
        proto: Learner = LogisticRegression() if y_binary else OLS()
    else:
        proto = _as_learner(spec)
    return lambda: _fresh(proto)


def _make_mediator(spec: Any) -> Any:
    from ..learners import LogisticRegression

    if spec is None:
        proto: Learner = LogisticRegression()
    else:
        proto = _as_learner(spec, want_classifier=True)
        if not _is_classifier(proto):
            raise TypeError(
                f"the mediator model needs a classifier with predict_proba; "
                f"got {type(proto).__name__}"
            )
    return lambda: _fresh(proto)


_MAX_MEDIATOR_LEVELS = 10


def _build_design(
    data: TabularData,
    treatment: str,
    outcome: str,
    mediators: frozenset[str],
    covariates: frozenset[str],
    *,
    learner: Any,
    treated: Any,
) -> _Design:
    """Validate the shapes and assemble the arrays and learner factories."""
    if len(mediators) != 1:
        raise NotImplementedError(
            f"the g-computation estimator supports a single mediator; got {sorted(mediators)}. "
            "Multiple mediators come with the path-specific / multiply-robust path"
        )
    mediator = next(iter(mediators))
    for name in (treatment, outcome, mediator, *covariates):
        if name not in data.schema:
            raise ValueError(f"column {name!r} is not in the data")
        if data.missing().get(name, 0) > 0:
            raise ValueError(f"column {name!r} has missing values; call data.dropna() first")

    a = _treatment_01(data, treatment, treated)
    m_codes, levels = _mediator_codes(data, mediator)
    y, y_binary = _outcome(data, outcome)
    w = _covariate_matrix(data, covariates)

    outcome_spec, mediator_spec = _resolve_learners(learner)
    return _Design(
        a=a,
        m_codes=m_codes,
        y=y,
        w=w,
        levels=levels,
        y_binary=y_binary,
        make_outcome=_make_outcome(outcome_spec, y_binary),
        make_mediator=_make_mediator(mediator_spec),
    )


def _treatment_01(data: TabularData, name: str, treated: Any) -> npt.NDArray[np.float64]:
    """The treatment coded 0/1, a binary treatment required."""
    if treated is not None:
        return _coded_treatment(data, name, treated)
    col = data.schema[name]
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64)
    if col.is_continuous:
        values = np.asarray(data.column(name), dtype=np.float64)
        if set(np.unique(values).tolist()) <= {0.0, 1.0}:
            return values
    raise NotImplementedError(
        f"the mediation estimator needs a binary treatment; {name!r} is not binary. "
        "Pass treated=... to pick a level, or use a two-level treatment"
    )


def _mediator_codes(data: TabularData, name: str) -> tuple[npt.NDArray[np.int64], tuple[int, ...]]:
    """The mediator's integer codes and its distinct levels; discrete and low-cardinality."""
    col = data.schema[name]
    if not col.is_discrete:
        raise NotImplementedError(
            f"the g-computation estimator needs a discrete (binary or low-cardinality) "
            f"mediator; {name!r} is {col.kind.value}. A continuous mediator needs the "
            "density / imputation form (the multiply-robust path), not built"
        )
    card = col.cardinality or 0
    if card > _MAX_MEDIATOR_LEVELS:
        raise NotImplementedError(
            f"mediator {name!r} has {card} levels; the g-computation sums over the "
            f"mediator's levels and is capped at {_MAX_MEDIATOR_LEVELS}. Coarsen it "
            "or use the continuous path"
        )
    codes = data.codes(name)
    return codes, tuple(range(card))


def _outcome(data: TabularData, name: str) -> tuple[npt.NDArray[np.float64], bool]:
    """The outcome as floats and whether it is binary (continuous or binary only)."""
    col = data.schema[name]
    if col.is_discrete and col.cardinality == 2:
        return data.codes(name).astype(np.float64), True
    if col.is_continuous:
        return np.asarray(data.column(name), dtype=np.float64), False
    raise NotImplementedError(
        f"the mediation estimator supports a continuous or binary outcome; {name!r} is "
        f"{col.kind.value}"
    )


# -- the public entry --------------------------------------------------------


def _to_estimate(point: float, draws: npt.NDArray[np.float64], n: int, kind: str) -> EffectEstimate:
    """Wrap a point and its bootstrap draws as an :class:`EffectEstimate`."""
    return EffectEstimate(
        target=EffectTarget.ATE,
        estimator="mediation_gcomputation",
        learners={},
        n=n,
        kind="bootstrap",
        diagnostics={"quantity": _QUANTITY_CODE[kind], "n_boot_returned": float(draws.shape[0])},
        _point=point,
        _draws=draws,
    )


_QUANTITY_CODE = {"nde": 0.0, "nie": 1.0, "total": 2.0, "cde": 3.0}


def estimate_mediation(
    graph: MixedGraph,
    data: TabularData,
    treatment: str,
    outcome: str,
    mediators: str | frozenset[str],
    *,
    learner: Any = None,
    treated: Any | None = None,
    random_state: int | None = None,
    n_boot: int = 1000,
) -> MediationEstimate:
    """Identify and estimate a mediation decomposition by g-computation.

    Identifies the decomposition with
    :func:`~ergodic.identification.identify_mediation` (raising if the natural
    direct and indirect effects are not identifiable), then estimates NDE, NIE, the
    total effect, and the controlled direct effect at each mediator level by
    g-computation: fit ``mu(x, m, w) = E[Y | x, m, w]`` and ``p(m | x, w)``, plug
    them into the mediation formula, and average over the empirical covariates.

    For a binary treatment and a discrete (binary or low-cardinality) mediator,
    with a continuous or binary outcome. ``learner`` is an alias, a learner, or a
    ``{"outcome": ..., "mediator": ...}`` mapping (the mediator slot needs a
    classifier). ``treated`` names the treatment level coded to 1. The standard
    errors are a nonparametric bootstrap over the rows at ``random_state``.

    A continuous mediator raises (the density / multiply-robust path is not built);
    a non-binary treatment raises. Returns a :class:`MediationEstimate` with a
    :class:`~ergodic.inference.Design` record naming the (untestable) cross-world
    assumption.
    """
    mset = frozenset({mediators}) if isinstance(mediators, str) else frozenset(mediators)
    ident: MediationID = identify_mediation(graph, treatment, outcome, mset)
    if not ident.identifiable:
        raise ValueError(
            "the mediation decomposition is not identifiable: "
            f"{ident.assumptions[0]} The controlled direct effect "
            f"{'is' if ident.cde_identifiable else 'is not'} identifiable; for the "
            "natural effects, consider a path-specific analysis."
        )
    covariates = ident.covariates or frozenset()
    design = _build_design(
        data,
        treatment,
        outcome,
        mset,
        covariates,
        learner=learner,
        treated=treated,
    )
    if float(np.sum(design.a == 1.0)) == 0.0 or float(np.sum(design.a == 0.0)) == 0.0:
        raise ValueError(
            f"treatment {treatment!r} has an empty arm; both levels must appear to "
            "estimate a direct/indirect contrast"
        )

    n = data.n_rows
    full = np.arange(n, dtype=np.int64)
    point = _compute(design, full)
    boot = _bootstrap(design, n, random_state, n_boot)

    cde_levels = _mediator_level_values(data, next(iter(mset)))
    cde: dict[Any, EffectEstimate] = {}
    for code in design.levels:
        cde[cde_levels[code]] = _to_estimate(point.cde[code], boot["cde"][:, code], n, "cde")

    design_record = Design(
        strategy="mediation",
        outcome=outcome,
        treatment=treatment,
        assumptions=ident.assumptions,
        estimand=f"NDE = {ident.nde_estimand}; NIE = {ident.nie_estimand}",
    )
    return MediationEstimate(
        nde=_to_estimate(point.nde, boot["nde"], n, "nde"),
        nie=_to_estimate(point.nie, boot["nie"], n, "nie"),
        total=_to_estimate(point.total, boot["total"], n, "total"),
        cde=cde,
        design=design_record,
        mediators=mset,
        covariates=covariates,
    )


def _mediator_level_values(data: TabularData, mediator: str) -> dict[int, Any]:
    """Map each mediator code to its category value (for the CDE keys)."""
    categories = data.schema[mediator].categories or ()
    return {code: categories[code] for code in range(len(categories))}


def _bootstrap(
    design: _Design, n: int, random_state: int | None, n_boot: int
) -> dict[str, npt.NDArray[np.float64]]:
    """Resample rows, refit both models, recompute the four quantities.

    Returns arrays of draws per quantity (and a ``(n_boot, K)`` matrix for the CDE
    levels), honoring ``random_state``. A resample that fails to fit (a degenerate
    arm, a singular design) is skipped, the same discipline the other estimators'
    bootstrap uses.
    """
    rng = np.random.default_rng(random_state)
    k = len(design.levels)
    nde = np.empty(n_boot, dtype=np.float64)
    nie = np.empty(n_boot, dtype=np.float64)
    total = np.empty(n_boot, dtype=np.float64)
    cde = np.empty((n_boot, k), dtype=np.float64)
    filled = 0
    attempts = 0
    budget = n_boot * 5
    while filled < n_boot and attempts < budget:
        attempts += 1
        idx = rng.integers(0, n, n)
        if np.sum(design.a[idx] == 1.0) == 0 or np.sum(design.a[idx] == 0.0) == 0:
            continue
        try:
            eff = _compute(design, idx)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if not (np.isfinite(eff.nde) and np.isfinite(eff.nie)):
            continue
        nde[filled] = eff.nde
        nie[filled] = eff.nie
        total[filled] = eff.total
        for code in design.levels:
            cde[filled, code] = eff.cde[code]
        filled += 1
    if filled == 0:  # pragma: no cover - a resample of a positive sample rarely all-fails
        raise RuntimeError(
            f"the mediation bootstrap produced no finite draws in {attempts} attempts"
        )
    return {
        "nde": nde[:filled],
        "nie": nie[:filled],
        "total": total[:filled],
        "cde": cde[:filled],
    }
