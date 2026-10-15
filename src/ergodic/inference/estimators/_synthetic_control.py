# SPDX-License-Identifier: Apache-2.0
"""Synthetic control: one treated unit against a weighted average of donors.

The estimator finds nonnegative donor weights that sum to one and make the
weighted donor outcome track the treated unit before treatment. The post-period
gap between the observed and the synthetic path is the effect. Inference is the
placebo permutation of Abadie, Diamond, and Hainmueller: rerun the same fit on
every donor as if it were treated, then rank the treated unit's post-to-pre
RMSPE ratio among the placebos.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from .._design import Design
from .._result import EffectEstimate
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy.typing as npt

    from ...data import PanelData

__all__ = ["SyntheticControlEffect", "synthetic_control"]


@dataclass(frozen=True)
class SyntheticControlEffect(EffectEstimate):
    """A synthetic-control fit: the weights, the two paths, and placebo inference.

    ``observed`` and ``synthetic`` run over ``periods``, and ``gaps`` is their
    difference. ``pre_rmspe`` is the tracking error before ``at``, the fit
    diagnostic to judge before trusting the effect. ``p_value`` ranks the
    treated unit's post-to-pre RMSPE ratio among the donor placebos, and
    ``placebo_gaps`` holds each donor's post-period placebo gaps. There is no
    sampling story for one treated unit, so ``std_error`` stays ``None`` on
    purpose and ``interval`` is a placebo band instead.
    """

    design: Design | None = None
    treated_unit: Any = None
    at: Any = None
    weights: Mapping[Any, float] = field(default_factory=dict)
    periods: tuple[Any, ...] = ()
    observed: tuple[float, ...] = ()
    synthetic: tuple[float, ...] = ()
    pre_rmspe: float = float("nan")
    post_rmspe: float = float("nan")
    p_value: float | None = None
    placebo_gaps: Mapping[Any, tuple[float, ...]] = field(default_factory=dict)

    @property
    def gaps(self) -> tuple[float, ...]:
        """Observed minus synthetic, one value per period."""
        return tuple(o - s for o, s in zip(self.observed, self.synthetic, strict=True))

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """A placebo-permutation band around the estimate, not a sampling CI.

        The half width is the ``level`` quantile of the absolute placebo
        post-gap means: the gap a unit with no treatment can show. An estimate
        whose band excludes zero clears the placebo bar at that level.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if not self.placebo_gaps:
            raise ValueError("estimate has no placebo gaps to form a band from")
        means = np.asarray(
            [abs(float(np.mean(gaps))) for gaps in self.placebo_gaps.values()],
            dtype=np.float64,
        )
        half = float(np.quantile(means, level))
        return (self.value - half, self.value + half)


def _project_simplex(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Euclidean projection onto the probability simplex (Duchi et al., 2008).

    Sort descending, take running sums, find the last index whose value still
    sits above the running threshold, and clip below that threshold.
    """
    descending = np.sort(v)[::-1]
    cumulative = np.cumsum(descending)
    counts = np.arange(1, v.size + 1, dtype=np.float64)
    positive = descending - (cumulative - 1.0) / counts > 0.0
    rho = int(np.nonzero(positive)[0][-1])
    theta = float((cumulative[rho] - 1.0) / (rho + 1.0))
    projected: npt.NDArray[np.float64] = np.maximum(v - theta, 0.0)
    return projected


def _simplex_weights(
    target: npt.NDArray[np.float64], donors: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Minimize ``||target - donors @ w||^2`` over the simplex by projected gradient.

    Uniform start, steps of ``1 / L`` on the half squared error with ``L`` the
    largest eigenvalue of the donor Gram matrix, the Duchi projection after
    every step, at most 5000 steps, stopping once the largest weight change
    drops below 1e-12. Deterministic: no randomness anywhere.
    """
    n_donors = donors.shape[1]
    weights = np.full(n_donors, 1.0 / n_donors, dtype=np.float64)
    gram = np.asarray(donors.T @ donors, dtype=np.float64)
    cross = np.asarray(donors.T @ target, dtype=np.float64)
    largest = float(np.linalg.eigvalsh(gram)[-1])
    step = 1.0 / max(largest, 1e-12)
    for _ in range(5000):
        gradient = gram @ weights - cross
        updated = _project_simplex(np.asarray(weights - step * gradient, dtype=np.float64))
        change = float(np.max(np.abs(updated - weights)))
        weights = updated
        if change < 1e-12:
            break
    return weights


def _treated_from_column(data: PanelData, treatment: str) -> tuple[Any, Any]:
    """Read the treated unit and its first treated period off a 0/1 column."""
    if treatment not in data.variables:
        raise ValueError(
            f"treatment column {treatment!r} is not a variable of the panel; "
            f"have {sorted(data.variables)}"
        )
    raw = data.column(treatment)
    if raw.dtype.kind not in "biuf":
        raise ValueError(
            f"treatment column {treatment!r} must be boolean or 0/1 numeric, got dtype {raw.dtype}"
        )
    values = raw.astype(np.float64)
    if np.isnan(values).any():
        raise ValueError(
            f"treatment column {treatment!r} has missing values; it must be 0/1 everywhere"
        )
    outside = values[(values != 0.0) & (values != 1.0)]
    if outside.size:
        raise ValueError(
            f"treatment column {treatment!r} must be 0/1, found value {float(outside[0])}"
        )
    units_raw = data.column(data.entity).tolist()
    times_raw = data.column(data.time).tolist()
    flags_by_unit: dict[Any, dict[Any, float]] = {}
    for unit, period, flag in zip(units_raw, times_raw, values.tolist(), strict=True):
        flags_by_unit.setdefault(unit, {})[period] = float(flag)
    treated_units: list[Any] = []
    first_treated: dict[Any, Any] = {}
    for unit in sorted(flags_by_unit):
        ordered = sorted(flags_by_unit[unit].items())
        levels = [flag for _, flag in ordered]
        if any(a > b for a, b in pairwise(levels)):
            raise ValueError(
                f"treatment column {treatment!r} must be absorbing within a unit "
                f"(once 1, it stays 1); unit {unit!r} switches back to 0"
            )
        for period, flag in ordered:
            if flag == 1.0:
                treated_units.append(unit)
                first_treated[unit] = period
                break
    if not treated_units:
        raise ValueError(f"treatment column {treatment!r} marks no unit as treated")
    if len(treated_units) > 1:
        raise ValueError(
            f"treatment column {treatment!r} marks {len(treated_units)} treated units "
            f"({treated_units[:4]}); synthetic control here takes one "
            "treated unit, so run one call per treated unit"
        )
    chosen = treated_units[0]
    return chosen, first_treated[chosen]


def synthetic_control(
    data: PanelData,
    outcome: str,
    *,
    treatment: str | None = None,
    treated: Any | None = None,
    at: Any | None = None,
    donors: Iterable[Any] | None = None,
) -> SyntheticControlEffect:
    """Fit a synthetic control for one treated unit in a panel.

    Name the treated unit in exactly one of two ways. Pass ``treatment``, an
    absorbing 0/1 column (once a unit turns 1 it stays 1) that marks exactly
    one unit as treated; its first treated period becomes ``at``. Or pass
    ``treated`` and ``at`` directly. ``donors`` defaults to every other unit.

    The donor weights minimize the squared pre-period tracking error over the
    simplex (nonnegative, summing to one) by projected gradient descent with
    the Duchi projection; the run is deterministic. The headline value is the
    ATT, the mean of ``observed - synthetic`` over the post periods
    (``t >= at``), and ``pre_rmspe`` reports the fit it rests on.

    Inference is by placebo permutation, with the treated unit excluded from
    every placebo run: each donor is treated as pseudo-treated, fit on the
    remaining donors, and scored by its post-to-pre RMSPE ratio (skipped when
    its pre RMSPE falls below 1e-12). ``p_value`` is
    ``(1 + count(ratio_j >= ratio_treated)) / (1 + n_placebos)``. There is no
    sampling standard error for one treated unit, so ``std_error`` stays
    ``None`` and ``interval`` reads the placebo gaps instead.

    Validation: ``at`` must be one of the panel's periods with at least 2 pre
    periods and 1 post period; the treated unit must be observed at every
    period; donors missing any period are dropped with a warning and at least
    2 complete donors must remain; a NaN outcome among the used rows raises.
    """
    if outcome not in data.variables:
        raise ValueError(
            f"outcome {outcome!r} is not a variable of the panel; have {sorted(data.variables)}"
        )
    pair_given = treated is not None or at is not None
    if treatment is not None and pair_given:
        raise ValueError("pass exactly one of treatment= or the (treated=, at=) pair, got both")
    if treatment is None and (treated is None or at is None):
        raise ValueError(
            "pass exactly one of treatment= or the (treated=, at=) pair; "
            f"got treated={treated!r} and at={at!r}"
        )
    if treatment is not None:
        treated, at = _treated_from_column(data, treatment)

    entities = data.entities
    periods = data.periods
    if treated not in entities:
        raise ValueError(f"treated unit {treated!r} is not an entity of the panel")
    if at not in periods:
        raise ValueError(f"at={at!r} is not one of the panel's periods")
    at_idx = periods.index(at)
    n_pre = at_idx
    n_post = len(periods) - at_idx
    if n_pre < 2:
        raise ValueError(
            f"need at least 2 pre periods before at={at!r}, got {n_pre}; "
            "the weights have nothing to track"
        )

    if donors is None:
        donor_list = [unit for unit in entities if unit != treated]
    else:
        donor_list = sorted(set(donors))
        unknown = [donor for donor in donor_list if donor not in entities]
        if unknown:
            raise ValueError(f"unknown donor units: {unknown}")
        if treated in donor_list:
            raise ValueError(f"the treated unit {treated!r} cannot be one of its own donors")

    units_raw = data.column(data.entity).tolist()
    times_raw = data.column(data.time).tolist()
    outcome_values = data.matrix([outcome])[:, 0]
    paths: dict[Any, dict[Any, float]] = {}
    for unit, period, value in zip(units_raw, times_raw, outcome_values.tolist(), strict=True):
        paths.setdefault(unit, {})[period] = float(value)

    treated_path = paths[treated]
    n_missing = sum(period not in treated_path for period in periods)
    if n_missing:
        raise ValueError(
            f"the treated unit {treated!r} is missing {n_missing} of {len(periods)} periods; "
            "synthetic control needs its full path"
        )
    kept: list[Any] = []
    dropped: list[Any] = []
    for donor in donor_list:
        if all(period in paths[donor] for period in periods):
            kept.append(donor)
        else:
            dropped.append(donor)
    if dropped:
        warnings.warn(
            f"dropped {len(dropped)} donor(s) not observed at every period: {dropped[:5]}",
            UserWarning,
            stacklevel=2,
        )
    if len(kept) < 2:
        raise ValueError(f"need at least 2 donors observed at every period, got {len(kept)}")

    y1 = np.asarray([treated_path[period] for period in periods], dtype=np.float64)
    y0 = np.asarray(
        [[paths[donor][period] for donor in kept] for period in periods], dtype=np.float64
    )
    if np.isnan(y1).any() or np.isnan(y0).any():
        raise ValueError(
            f"outcome {outcome!r} has missing values among the used rows; "
            "synthetic control needs complete paths"
        )

    w = _simplex_weights(y1[:at_idx], y0[:at_idx])
    synthetic_path = np.asarray(y0 @ w, dtype=np.float64)
    gaps = y1 - synthetic_path
    att = float(np.mean(gaps[at_idx:]))
    pre_rmspe = float(np.sqrt(np.mean(gaps[:at_idx] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gaps[at_idx:] ** 2)))

    ratios: list[float] = []
    n_skipped = 0
    placebo_gaps: dict[Any, tuple[float, ...]] = {}
    for j, donor in enumerate(kept):
        pool = [i for i in range(len(kept)) if i != j]
        w_j = _simplex_weights(y0[:at_idx, j], y0[:at_idx][:, pool])
        gaps_j = np.asarray(y0[:, j] - y0[:, pool] @ w_j, dtype=np.float64)
        placebo_gaps[donor] = tuple(float(g) for g in gaps_j[at_idx:])
        pre_j = float(np.sqrt(np.mean(gaps_j[:at_idx] ** 2)))
        post_j = float(np.sqrt(np.mean(gaps_j[at_idx:] ** 2)))
        if pre_j < 1e-12:
            n_skipped += 1
            continue
        ratios.append(post_j / pre_j)
    treated_ratio = float("inf") if pre_rmspe < 1e-12 else post_rmspe / pre_rmspe
    p_value: float | None = None
    if ratios:
        p_value = (1 + sum(ratio >= treated_ratio for ratio in ratios)) / (1 + len(ratios))

    design = Design(
        strategy="synthetic_control",
        outcome=outcome,
        treatment=treatment,
        assumptions=(
            "the treated unit's untreated path lies in the convex hull of the donors",
            "no spillovers from the treated unit to donors",
            "a good pre-period fit, judge pre_rmspe",
        ),
        estimand="ATT for the treated unit, post-period gap to its synthetic control",
    )
    return SyntheticControlEffect(
        target=EffectTarget.ATT,
        estimator="synthetic_control",
        learners={},
        n=(1 + len(kept)) * len(periods),
        kind="analytic",
        diagnostics={
            "n_donors": float(len(kept)),
            "n_pre": float(n_pre),
            "n_post": float(n_post),
            "n_placebos": float(len(ratios)),
            "n_dropped_donors": float(len(dropped)),
            "n_placebo_skipped": float(n_skipped),
            "pre_rmspe": pre_rmspe,
            "post_rmspe": post_rmspe,
        },
        _point=att,
        _std_error=None,
        design=design,
        treated_unit=treated,
        at=at,
        weights={donor: float(weight) for donor, weight in zip(kept, w, strict=True)},
        periods=periods,
        observed=tuple(float(v) for v in y1),
        synthetic=tuple(float(v) for v in synthetic_path),
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        p_value=p_value,
        placebo_gaps=placebo_gaps,
    )
