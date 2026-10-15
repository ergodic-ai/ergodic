# SPDX-License-Identifier: Apache-2.0
"""A uniform per-unit treatment-effect result: one shape for CATE and ITE.

:class:`PerUnitEffect` is the per-unit analogue of :class:`EffectEstimate`. Where
``EffectEstimate`` is scalar (``value -> float``), this is vectorized over a unit
axis (``values -> (n,)``). It is a SIBLING of ``EffectEstimate``, not a subclass:
the scalar ``value``/``interval`` contract cannot be array-overridden without
breaking callers, so this follows the :class:`~ergodic.inference.MediationEstimate`
and :class:`~ergodic.scm.Sensitivity` precedent of a parallel record that yields an
``EffectEstimate`` for its population summary through :meth:`PerUnitEffect.ate`.

Two subclasses name the estimand and carry what genuinely differs. ``CATEResult``
(``estimand="cate"``) holds the identification record and the cached doubly-robust
scores for :meth:`~CATEResult.explain`. ``ITEResult`` (``estimand="ite"``) holds the
factual outcomes and the structural framing; its ``explain`` is deferred.

The same tagged-union storage as ``EffectEstimate``: a point with optional
per-unit standard errors, or ``(S, n)`` draws (bootstrap or posterior). Drop-in for
code doing arithmetic on the raw arrays: ``__array__`` and ``.values`` return the
per-unit point vector, so ``np.mean(result)`` works as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._result import EffectEstimate
from ._stats import _norm_ppf
from ._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    import pandas as pd

    from ..identification import Identification
    from ._scores import DRScores
    from .explain import Heterogeneity


@dataclass(frozen=True, eq=False)
class PerUnitEffect:
    """Per-unit treatment effects with their uncertainty.

    ``_values`` is the per-unit point vector ``(n,)``, ordered to the input rows.
    Either ``_std_error`` (per-unit analytic SEs) or ``_draws`` ``(S, n)`` (bootstrap
    or posterior) carries the uncertainty, and the accessors read whichever is
    present. ``index`` are explicit row ids; ``estimand`` is set by the subclass.
    """

    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    n: int
    kind: str  # "point", "analytic", "bootstrap", or "posterior"
    estimand: str = ""
    source: str = ""
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    _values: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    _index: npt.NDArray[Any] | None = field(default=None, repr=False)
    _std_error: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    _draws: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    # -- the per-unit surface ------------------------------------------------

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """The per-unit point estimates, ``(n,)``, ordered to the input rows.

        When only draws are present, the posterior or bootstrap mean per unit.
        """
        if self._values.size or self._draws is None:
            return np.asarray(self._values, dtype=np.float64)
        return np.asarray(self._draws.mean(axis=0), dtype=np.float64)

    def __array__(self, dtype: Any = None) -> npt.NDArray[np.float64]:
        """The per-unit point vector, so ``np.asarray(result)`` and arithmetic work."""
        out = self.values
        return out.astype(dtype) if dtype is not None else out

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the per-unit point vector, ``(n,)``."""
        return self.values.shape

    def __getitem__(self, item: Any) -> Any:
        """Index into the per-unit point vector (so ``result[i]`` is the unit's effect)."""
        return self.values[item]

    def __iter__(self) -> Any:
        return iter(self.values)

    # Element-wise comparisons against the point vector, so a wrapped result is a
    # drop-in for the bare array the SCM and the meta-learners returned before.
    def __eq__(self, other: Any) -> Any:
        return self.values == np.asarray(other)

    def __ne__(self, other: Any) -> Any:
        return self.values != np.asarray(other)

    def __lt__(self, other: Any) -> Any:
        return self.values < np.asarray(other)

    def __le__(self, other: Any) -> Any:
        return self.values <= np.asarray(other)

    def __gt__(self, other: Any) -> Any:
        return self.values > np.asarray(other)

    def __ge__(self, other: Any) -> Any:
        return self.values >= np.asarray(other)

    __hash__ = None  # type: ignore[assignment]

    @property
    def index(self) -> npt.NDArray[Any]:
        """The row ids for the units, positional ``arange(n)`` when none were given."""
        if self._index is not None:
            return np.asarray(self._index)
        return np.arange(self.n)

    @property
    def std_error(self) -> npt.NDArray[np.float64] | None:
        """Per-unit standard errors, or the per-unit draw standard deviations."""
        if self._std_error is not None:
            return np.asarray(self._std_error, dtype=np.float64)
        if self._draws is not None:
            return np.asarray(self._draws.std(axis=0, ddof=1), dtype=np.float64)
        return None

    def interval(
        self, level: float = 0.9
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-unit confidence or credible intervals, ``(lo[n], hi[n])`` at ``level``.

        Draw quantiles when draws are present, else the normal interval from the
        per-unit point and standard error. Raises when there is no uncertainty.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if self._draws is not None:
            lo = np.quantile(self._draws, (1.0 - level) / 2.0, axis=0)
            hi = np.quantile(self._draws, (1.0 + level) / 2.0, axis=0)
            return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        if self._std_error is not None:
            z = _norm_ppf((1.0 + level) / 2.0)
            point = self.values
            se = np.asarray(self._std_error, dtype=np.float64)
            return point - z * se, point + z * se
        raise ValueError("per-unit effect has no uncertainty to form an interval from")

    def draws(self) -> npt.NDArray[np.float64] | None:
        """The ``(S, n)`` bootstrap or posterior draws, or ``None`` for a point result."""
        if self._draws is None:
            return None
        return np.array(self._draws, copy=True)

    def ate(self) -> EffectEstimate:
        """The population summary: the average per-unit effect, as an ``EffectEstimate``.

        The bridge back to the scalar family. With draws, the per-unit draws average
        to a draw vector of the mean effect (so the population interval reflects the
        joint uncertainty). With neither draws nor standard errors, a point.

        Per-unit analytic standard errors cannot be combined into the mean's
        standard error: the units share one fitted model, so their errors are
        correlated, and treating them as independent understates the uncertainty
        by about a factor of ``sqrt(n)``. So, with analytic standard errors:

        - a homogeneous result (one effect and one standard error repeated over
          the units, as ``DML.cate()`` returns) reports that effect and its
          standard error;
        - a result carrying its cross-fit doubly-robust scores reports the AIPW
          average off them: the mean score with standard error
          ``std(scores) / sqrt(n)`` (the same scores :meth:`CATEResult.explain`
          reads). For the DR-learner's linear final stage the point equals the
          mean per-unit effect; for a non-linear final stage such as the causal
          forest it can differ slightly from the plain mean of the per-unit
          effects (``diagnostics["ate_from_dr_scores"]`` marks this path);
        - otherwise it reports the mean with no standard error (``kind="point"``).
          For an interval, use :func:`~ergodic.inference.estimate_effect`.
        """
        target = EffectTarget.CATE if self.estimand == "cate" else EffectTarget.ITE
        learners: dict[str, str] = {}
        if self._draws is not None:
            mean_draws = np.asarray(self._draws.mean(axis=1), dtype=np.float64)
            return EffectEstimate(
                target=target,
                estimator=self.source,
                learners=learners,
                n=self.n,
                kind=self.kind if self.kind in ("bootstrap", "posterior") else "bootstrap",
                diagnostics=dict(self.diagnostics),
                _draws=mean_draws,
            )
        values = self.values
        point = float(np.mean(values))
        if self._std_error is not None:
            se = np.asarray(self._std_error, dtype=np.float64)
            if values.size and np.ptp(values) == 0.0 and np.ptp(se) == 0.0:
                # one shared estimate repeated over the units: its own standard error
                return EffectEstimate(
                    target=target,
                    estimator=self.source,
                    learners=learners,
                    n=self.n,
                    kind="analytic",
                    diagnostics=dict(self.diagnostics),
                    _point=float(values[0]),
                    _std_error=float(se[0]),
                )
            scores = self._ate_scores()
            if scores is not None:
                k = scores.shape[0]
                return EffectEstimate(
                    target=target,
                    estimator=self.source,
                    learners=learners,
                    n=self.n,
                    kind="analytic",
                    diagnostics={**dict(self.diagnostics), "ate_from_dr_scores": 1.0},
                    _point=float(np.mean(scores)),
                    _std_error=float(np.std(scores, ddof=1) / np.sqrt(k)),
                )
        return EffectEstimate(
            target=target,
            estimator=self.source,
            learners=learners,
            n=self.n,
            kind="point",
            diagnostics=dict(self.diagnostics),
            _point=point,
        )

    def _ate_scores(self) -> npt.NDArray[np.float64] | None:
        """Per-unit doubly-robust scores over these same units, when carried."""
        return None

    def to_frame(self) -> pd.DataFrame:
        """A per-unit table (index, effect, and lo/hi/se when available). Pandas only here."""
        import pandas as pd

        columns: dict[str, npt.NDArray[Any]] = {"effect": self.values}
        se = self.std_error
        if se is not None:
            columns["std_error"] = se
        try:
            lo, hi = self.interval()
            columns["ci_low"] = lo
            columns["ci_high"] = hi
        except ValueError:
            pass
        return pd.DataFrame(columns, index=pd.Index(self.index, name="unit"))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        v = self.values
        head = f"{self.estimand or 'effect'}"
        body = (
            f"ate={float(np.mean(v)):.4g}, "
            f"min={float(np.min(v)):.3g}, median={float(np.median(v)):.3g}, "
            f"max={float(np.max(v)):.3g}"
        )
        name = type(self).__name__
        return f"{name}({self.source or head}, {self.kind}: {body}, n={self.n})"


@dataclass(frozen=True, eq=False)
class CATEResult(PerUnitEffect):
    """Conditional average treatment effects, with an explainability hook.

    A conditional average over the unobserved noise (the inference-pillar object).
    Carries the :class:`~ergodic.identification.Identification` it was fit under and
    the cached doubly-robust scores + covariate design, so :meth:`explain` projects
    honest summaries (BLP, GATES, CLAN, the calibration test, RATE) off the same
    orthogonal signal.
    """

    estimand: str = "cate"
    identification: Identification | None = None
    _scores: DRScores | None = field(default=None, repr=False)

    def _ate_scores(self) -> npt.NDArray[np.float64] | None:
        """The cached AIPW scores, when they cover exactly the result's units."""
        if self._scores is None:
            return None
        scores = np.asarray(self._scores.scores, dtype=np.float64)
        if scores.shape[0] != self.n or scores.shape[0] < 2:
            return None
        return scores

    def explain(self) -> Heterogeneity:
        """Tier-B characterization of the heterogeneity, on the doubly-robust scores.

        Builds a :class:`~ergodic.inference.explain.Heterogeneity` record whose
        methods (``blp``, ``gates``, ``clan``, ``calibration_test``, ``rate``) all
        consume the cached cross-fit scores, the covariate design, and the fitted
        per-unit ``tau_hat``. Needs the scores; raises when they were not cached
        (an estimator that does not build them, or a result without them).
        """
        from .explain import Heterogeneity

        if self._scores is None:
            raise RuntimeError(
                "this CATEResult carries no doubly-robust scores to explain; "
                "explainability needs the cross-fit AIPW signal (the cate() facade "
                "and the doubly-robust meta-learners cache it)"
            )
        # the out-of-fold proxy ranking when available (the honesty recipe: the
        # ranking is fixed on data not used for the summary), else the in-sample fit
        tau_hat = self._scores.tau_hat if self._scores.tau_hat is not None else self.values
        return Heterogeneity(
            scores=self._scores,
            tau_hat=np.asarray(tau_hat, dtype=np.float64),
            treatment=self.treatment,
            outcome=self.outcome,
        )


@dataclass(frozen=True, eq=False)
class ITEResult(PerUnitEffect):
    """Individual treatment effects from a structural model (the twin network).

    The individual's own counterfactual contrast (the SCM-pillar object), not a
    conditional average. Carries the factual outcome column, the contrast framing
    (``levels=`` or ``shift=``), and a structural provenance note.

    :meth:`explain` is not available: the Tier-B machinery is defined over the
    CATE's doubly-robust scores, and there is no inference-valid descriptive
    regression of abducted ITEs, so it raises with a pointer rather than faking
    inference.
    """

    estimand: str = "ite"
    framing: str = ""
    note: str = ""
    _factual: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def factual(self) -> npt.NDArray[np.float64] | None:
        """The per-unit factual outcome the contrast was taken against, when carried."""
        if self._factual is None:
            return None
        return np.asarray(self._factual, dtype=np.float64)

    def explain(self) -> Heterogeneity:
        """Not available: the Tier-B tools need the CATE's doubly-robust scores."""
        raise NotImplementedError(
            "ITEResult.explain is not available: the Tier-B characterization "
            "(BLP, GATES, RATE) is defined over the doubly-robust CATE scores, which "
            "structural ITEs do not have. Use a CATE meta-learner (cate(...)) for "
            "inference-valid heterogeneity characterization"
        )
