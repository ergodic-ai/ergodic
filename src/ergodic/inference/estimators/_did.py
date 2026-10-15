# SPDX-License-Identifier: Apache-2.0
"""Difference-in-differences over panel data.

Adoption timing identifies the effect. A unit adopts in the first period its
treatment turns on, treatment stays on afterwards (absorbing), and the change
in a treated unit's outcome is compared with the change among units that have
not adopted. :func:`did` returns the average effect on the treated (ATT) in one
of three flavors: ``canonical`` for a single adoption period, ``group_time``
for staggered adoption in the style of Callaway and Sant'Anna (2021), and
``twfe``, the two-way fixed-effects regression, kept for reference because
under staggered adoption with effects that change over time it mixes cell
effects with possibly negative weights (Goodman-Bacon 2021). :func:`event_study`
reuses the group-time cells to trace the effect by time since adoption, leads
as the placebo check and lags as the dynamic effect.

Identification rests on parallel trends against a comparison pool, never-treated
units by default or not-yet-treated units on request, plus no anticipation. The
assumptions ride with every result in its :class:`Design` record. Inference is
a unit bootstrap throughout: resample units with replacement and recompute.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._design import Design
from .._result import EffectEstimate
from .._target import EffectTarget

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from ...data import PanelData

__all__ = ["DiDEffect", "EventTimeEffect", "GroupTimeEffect", "did", "event_study"]

_ASSUMPTIONS = (
    "parallel trends: untreated outcomes of treated and comparison units move in parallel",
    "no anticipation: outcomes do not respond before adoption",
)


@dataclass(frozen=True)
class GroupTimeEffect:
    """One ATT(g, t) cell: the cohort adopting at ``group``, read at ``period``.

    ``event_time`` is the position of ``period`` minus the position of ``group``
    in the sorted period sequence, so -1 is the base period and never appears.
    ``n_treated`` and ``n_comparison`` count the units observed at both the base
    and the target period on each side of the comparison.
    """

    group: Any
    period: Any
    event_time: int
    estimate: float
    n_treated: int
    n_comparison: int


@dataclass(frozen=True)
class EventTimeEffect:
    """The effect at one event time, aggregated over cohorts with cohort-size weights.

    ``std_error`` and ``interval`` come from the unit bootstrap: the standard
    deviation of the draws and their 95% percentile interval. ``n_treated``
    counts the treated units contributing a cell at this event time.
    """

    event_time: int
    estimate: float
    std_error: float | None
    interval: tuple[float, float]
    n_treated: int


@dataclass(frozen=True, repr=False)
class DiDEffect(EffectEstimate):
    """A difference-in-differences result: the estimate plus the design it rests on.

    The :class:`EffectEstimate` surface (``value``, ``std_error``, ``interval``,
    ``draws``) reads the headline ATT. ``flavor`` records which estimator ran,
    ``comparison`` which pool stood in for the counterfactual, ``by_group_time``
    the per-cell detail of the group-time flavor, and ``by_event_time`` the
    event-study aggregation when :func:`event_study` built the result.
    """

    design: Design | None = None
    flavor: str = ""
    comparison: str = ""
    by_group_time: tuple[GroupTimeEffect, ...] = field(default=(), repr=False)
    by_event_time: tuple[EventTimeEffect, ...] = field(default=(), repr=False)


# -- preparation ---------------------------------------------------------------


@dataclass(frozen=True)
class _Prepared:
    """The panel unpacked into aligned arrays, validated and adoption-timed."""

    y: npt.NDArray[np.float64]
    d: npt.NDArray[np.float64]
    x: npt.NDArray[np.float64]
    unit: npt.NDArray[np.int64]
    pos: npt.NDArray[np.int64]
    rows_of_unit: tuple[npt.NDArray[np.int64], ...]
    entities: tuple[Any, ...]
    periods: tuple[Any, ...]
    adoption: npt.NDArray[np.int64]
    n_controls: int

    @property
    def n_units(self) -> int:
        return len(self.entities)

    @property
    def n_periods(self) -> int:
        return len(self.periods)


def _prepare(data: PanelData, outcome: str, treatment: str, controls: list[str]) -> _Prepared:
    """Validate the columns, code the treatment, and time each unit's adoption."""
    available = set(data.variables)
    roles = [("outcome", outcome), ("treatment", treatment)]
    roles += [("control", name) for name in controls]
    for role, name in roles:
        if name not in available:
            raise ValueError(
                f"{role} column {name!r} is not a variable of the panel; "
                f"variables are {sorted(available)}"
            )
    if outcome == treatment:
        raise ValueError(f"outcome and treatment are both {outcome!r}; they must differ")
    if len(set(controls)) != len(controls):
        raise ValueError(f"controls repeat a column: {controls}")
    overlap = sorted({outcome, treatment} & set(controls))
    if overlap:
        raise ValueError(f"controls must exclude the outcome and the treatment, got {overlap}")

    y = np.asarray(data.matrix([outcome])[:, 0], dtype=np.float64)
    if bool(np.isnan(y).any()):
        raise ValueError(
            f"outcome {outcome!r} has missing values; estimation needs complete columns"
        )
    raw = data.column(treatment)
    if raw.dtype.kind not in "biuf":
        raise ValueError(
            f"treatment {treatment!r} must be boolean or numeric coded 0/1, "
            f"got dtype {raw.dtype}; recode it"
        )
    d = raw.astype(np.float64)
    if bool(np.isnan(d).any()):
        raise ValueError(
            f"treatment {treatment!r} has missing values; estimation needs complete columns"
        )
    levels = np.unique(d)
    if not bool(np.isin(levels, (0.0, 1.0)).all()):
        raise ValueError(
            f"treatment {treatment!r} must be coded 0/1 (or boolean), "
            f"found values {levels.tolist()[:6]}"
        )
    if controls:
        x = np.asarray(data.matrix(controls), dtype=np.float64)
        for j, name in enumerate(controls):
            if bool(np.isnan(x[:, j]).any()):
                raise ValueError(
                    f"control {name!r} has missing values; estimation needs complete columns"
                )
    else:
        x = np.empty((data.n_rows, 0), dtype=np.float64)

    entities = data.entities
    periods = data.periods
    entity_pos = {value: index for index, value in enumerate(entities)}
    period_pos = {value: index for index, value in enumerate(periods)}
    unit = np.asarray(
        [entity_pos[value] for value in data.column(data.entity).tolist()], dtype=np.int64
    )
    pos = np.asarray(
        [period_pos[value] for value in data.column(data.time).tolist()], dtype=np.int64
    )

    rows_of_unit: list[npt.NDArray[np.int64]] = []
    adoption = np.full(len(entities), -1, dtype=np.int64)
    for u in range(len(entities)):
        rows = np.flatnonzero(unit == u).astype(np.int64)
        rows = rows[np.argsort(pos[rows], kind="stable")]
        rows_of_unit.append(rows)
        d_unit = d[rows]
        if bool(np.any(np.diff(d_unit) < 0.0)):
            raise ValueError(
                f"treatment {treatment!r} is not absorbing: unit {entities[u]!r} "
                "switches from 1 back to 0"
            )
        on = np.flatnonzero(d_unit == 1.0)
        if on.size:
            adoption[u] = int(pos[rows[int(on[0])]])

    return _Prepared(
        y=y,
        d=d,
        x=x,
        unit=unit,
        pos=pos,
        rows_of_unit=tuple(rows_of_unit),
        entities=entities,
        periods=periods,
        adoption=adoption,
        n_controls=len(controls),
    )


def _design_record(outcome: str, treatment: str, comparison: str) -> Design:
    return Design(
        strategy="did",
        outcome=outcome,
        treatment=treatment,
        assumptions=_ASSUMPTIONS,
        estimand=f"ATT identified by parallel trends against {comparison} units",
    )


def _check_common(comparison: str, n_boot: int) -> None:
    if comparison not in ("never_treated", "not_yet_treated"):
        raise ValueError(
            f"comparison must be 'never_treated' or 'not_yet_treated', got {comparison!r}"
        )
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")


# -- canonical -------------------------------------------------------------------


def _canonical_att(
    delta: npt.NDArray[np.float64],
    treated: npt.NDArray[np.bool_],
    xbar: npt.NDArray[np.float64],
    idx: npt.NDArray[np.int64],
) -> float | None:
    """The 2x2 ATT on the units at ``idx``; None when a resample loses an arm."""
    flags = treated[idx]
    if not bool(flags.any()) or bool(flags.all()):
        return None
    changes = delta[idx]
    if xbar.shape[1] == 0:
        return float(changes[flags].mean() - changes[~flags].mean())
    design = np.column_stack(
        [np.ones(idx.size, dtype=np.float64), flags.astype(np.float64), xbar[idx]]
    )
    beta = np.linalg.lstsq(design, changes, rcond=None)[0]
    return float(beta[1])


def _fit_canonical(
    prep: _Prepared, comparison: str, n_boot: int, rng: np.random.Generator, design: Design
) -> DiDEffect:
    adopted = prep.adoption[prep.adoption >= 0]
    g = int(adopted[0])
    if not bool((prep.adoption == -1).any()):
        raise ValueError(
            "canonical did has no comparison units: every unit adopts the treatment, "
            "so no pool of never-treated units exists"
        )

    n_units = prep.n_units
    k = prep.n_controls
    delta = np.full(n_units, np.nan, dtype=np.float64)
    xbar = np.zeros((n_units, k), dtype=np.float64)
    kept = np.zeros(n_units, dtype=np.bool_)
    for u, rows in enumerate(prep.rows_of_unit):
        pre = rows[prep.pos[rows] < g]
        post = rows[prep.pos[rows] >= g]
        if pre.size == 0 or post.size == 0:
            continue
        kept[u] = True
        delta[u] = float(prep.y[post].mean()) - float(prep.y[pre].mean())
        if k:
            xbar[u] = prep.x[pre].mean(axis=0)

    n_dropped = int((~kept).sum())
    if n_dropped:
        names = [prep.entities[u] for u in np.flatnonzero(~kept)[:5]]
        warnings.warn(
            f"canonical did dropped {n_dropped} unit(s) lacking a pre or post "
            f"observation, starting with {names}",
            UserWarning,
            stacklevel=2,
        )

    kept_idx = np.flatnonzero(kept).astype(np.int64)
    deltas = delta[kept_idx]
    treated = prep.adoption[kept_idx] == g
    pre_means = xbar[kept_idx]
    if not bool(treated.any()):
        raise ValueError("no treated unit has both a pre and a post observation")
    if bool(treated.all()):
        raise ValueError("no comparison unit has both a pre and a post observation")

    m = kept_idx.size
    point = _canonical_att(deltas, treated, pre_means, np.arange(m, dtype=np.int64))
    if point is None:  # pragma: no cover - both arms were just validated nonempty
        raise RuntimeError("canonical did lost an arm after validation")

    draws = np.empty(n_boot, dtype=np.float64)
    got = 0
    attempts = 0
    while got < n_boot and attempts < 5 * n_boot:
        attempts += 1
        value = _canonical_att(deltas, treated, pre_means, rng.integers(0, m, m, dtype=np.int64))
        if value is None:
            continue
        draws[got] = value
        got += 1
    if got == 0:
        raise RuntimeError(
            "the unit bootstrap produced no draws: every resample lacked a treated "
            "or a comparison unit"
        )

    diagnostics: dict[str, float] = {
        "n_units": float(n_units),
        "n_treated_units": float(treated.sum()),
        "n_comparison_units": float((~treated).sum()),
        "n_periods": float(prep.n_periods),
        "n_dropped_units": float(n_dropped),
    }
    if k == 0:
        treated_deltas = deltas[treated]
        comparison_deltas = deltas[~treated]
        if treated_deltas.size >= 2 and comparison_deltas.size >= 2:
            diagnostics["analytic_se"] = float(
                np.sqrt(
                    treated_deltas.var(ddof=1) / treated_deltas.size
                    + comparison_deltas.var(ddof=1) / comparison_deltas.size
                )
            )

    n_rows_used = int(sum(prep.rows_of_unit[u].size for u in kept_idx.tolist()))
    return DiDEffect(
        target=EffectTarget.ATT,
        estimator="did",
        learners={},
        n=n_rows_used,
        kind="bootstrap",
        diagnostics=diagnostics,
        _point=point,
        _draws=draws[:got],
        design=design,
        flavor="canonical",
        comparison=comparison,
    )


# -- two-way fixed effects ---------------------------------------------------------


def _group_demean(
    z: npt.NDArray[np.float64], idx: npt.NDArray[np.int64], n_groups: int
) -> npt.NDArray[np.float64]:
    """Subtract each group's column means; empty groups contribute nothing."""
    counts = np.bincount(idx, minlength=n_groups).astype(np.float64)
    counts[counts == 0.0] = 1.0
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        means = np.bincount(idx, weights=z[:, j], minlength=n_groups) / counts
        out[:, j] = z[:, j] - means[idx]
    return out


def _twfe_coef(
    z: npt.NDArray[np.float64],
    unit_idx: npt.NDArray[np.int64],
    pos_idx: npt.NDArray[np.int64],
    n_units: int,
    n_periods: int,
) -> float | None:
    """The treatment coefficient after the iterated two-way within transform.

    Columns of ``z`` are ``[y, d, controls...]``. Alternate unit and period
    demeaning until the largest change drops below 1e-10 (at most 100 sweeps),
    which handles unbalanced panels, then regress demeaned y on the rest.
    Returns None when the demeaned treatment has no variation left.
    """
    current = np.array(z, dtype=np.float64)
    for _ in range(100):
        previous = current
        current = _group_demean(current, unit_idx, n_units)
        current = _group_demean(current, pos_idx, n_periods)
        if float(np.max(np.abs(current - previous))) < 1e-10:
            break
    if float(np.max(np.abs(current[:, 1]))) < 1e-8:
        return None
    beta = np.linalg.lstsq(current[:, 1:], current[:, 0], rcond=None)[0]
    return float(beta[0])


def _fit_twfe(
    prep: _Prepared, comparison: str, n_boot: int, rng: np.random.Generator, design: Design
) -> DiDEffect:
    z = np.column_stack([prep.y, prep.d, prep.x])
    n_units = prep.n_units
    point = _twfe_coef(z, prep.unit, prep.pos, n_units, prep.n_periods)
    if point is None:
        raise ValueError(
            "twfe found no within variation in the treatment after the two-way transform"
        )

    sizes = np.asarray([rows.size for rows in prep.rows_of_unit], dtype=np.int64)
    draws = np.empty(n_boot, dtype=np.float64)
    got = 0
    attempts = 0
    while got < n_boot and attempts < 5 * n_boot:
        attempts += 1
        sample = rng.integers(0, n_units, n_units, dtype=np.int64)
        rows = np.concatenate([prep.rows_of_unit[u] for u in sample.tolist()])
        clone_ids = np.repeat(np.arange(n_units, dtype=np.int64), sizes[sample])
        value = _twfe_coef(z[rows], clone_ids, prep.pos[rows], n_units, prep.n_periods)
        if value is None:
            continue
        draws[got] = value
        got += 1
    if got == 0:
        raise RuntimeError(
            "the unit bootstrap produced no draws: every resample lost the "
            "treatment's within variation"
        )

    diagnostics: dict[str, float] = {
        "n_units": float(n_units),
        "n_treated_units": float((prep.adoption >= 0).sum()),
        "n_comparison_units": float((prep.adoption == -1).sum()),
        "n_periods": float(prep.n_periods),
        "n_dropped_units": 0.0,
    }
    return DiDEffect(
        target=EffectTarget.ATT,
        estimator="did",
        learners={},
        n=int(prep.y.size),
        kind="bootstrap",
        diagnostics=diagnostics,
        _point=point,
        _draws=draws[:got],
        design=design,
        flavor="twfe",
        comparison=comparison,
    )


# -- group-time (Callaway-Sant'Anna style) -----------------------------------------


@dataclass(frozen=True)
class _Cell:
    """One (group, period) cell with its sides frozen against the full sample."""

    g: int
    t: int
    base: int
    group_units: npt.NDArray[np.int64]
    comp_units: npt.NDArray[np.int64]
    delta_group: npt.NDArray[np.float64]
    delta_comp: npt.NDArray[np.float64]
    x_group: npt.NDArray[np.float64]
    x_comp: npt.NDArray[np.float64]


def _cell_estimate(cell: _Cell, weights: npt.NDArray[np.float64], k: int) -> float | None:
    """ATT(g, t) under per-unit bootstrap weights; None when a side has none."""
    w_group = weights[cell.group_units]
    w_comp = weights[cell.comp_units]
    total_group = float(w_group.sum())
    total_comp = float(w_comp.sum())
    if total_group <= 0.0 or total_comp <= 0.0:
        return None
    if k == 0:
        group_mean = float(w_group @ cell.delta_group) / total_group
        comp_mean = float(w_comp @ cell.delta_comp) / total_comp
        return group_mean - comp_mean
    m_group = cell.group_units.size
    m_comp = cell.comp_units.size
    indicator = np.concatenate(
        [np.ones(m_group, dtype=np.float64), np.zeros(m_comp, dtype=np.float64)]
    )
    design = np.column_stack(
        [
            np.ones(m_group + m_comp, dtype=np.float64),
            indicator,
            np.vstack([cell.x_group, cell.x_comp]),
        ]
    )
    deltas = np.concatenate([cell.delta_group, cell.delta_comp])
    scale = np.sqrt(np.concatenate([w_group, w_comp]))
    beta = np.linalg.lstsq(design * scale[:, None], deltas * scale, rcond=None)[0]
    return float(beta[1])


def _build_cells(
    prep: _Prepared, comparison: str
) -> tuple[list[npt.NDArray[np.int64]], list[list[_Cell]]]:
    """All computable (group, period) cells against the universal base period.

    The base for group ``g`` is the period just before it, so a group adopting
    in the first period owns no cells. The cell at event time -1 is the base
    itself, identically zero, and is never computed.
    """
    n_units = prep.n_units
    n_periods = prep.n_periods
    k = prep.n_controls
    never = prep.adoption == -1
    y_grid = np.full((n_units, n_periods), np.nan, dtype=np.float64)
    y_grid[prep.unit, prep.pos] = prep.y
    x_grid = np.full((n_units, n_periods, k), np.nan, dtype=np.float64)
    x_grid[prep.unit, prep.pos] = prep.x

    groups = sorted(set(prep.adoption[prep.adoption >= 0].tolist()))
    group_units_all = [np.flatnonzero(prep.adoption == g).astype(np.int64) for g in groups]
    cells_by_group: list[list[_Cell]] = []
    for g in groups:
        cells: list[_Cell] = []
        base = g - 1
        if base >= 0:
            observed_base = ~np.isnan(y_grid[:, base])
            for t in range(n_periods):
                if t == base:
                    continue
                observed = observed_base & ~np.isnan(y_grid[:, t])
                if comparison == "never_treated":
                    pool = never
                else:
                    cutoff = max(t, base)
                    pool = never | ((prep.adoption > cutoff) & (prep.adoption != g))
                group_units = np.flatnonzero(observed & (prep.adoption == g)).astype(np.int64)
                comp_units = np.flatnonzero(observed & pool).astype(np.int64)
                if group_units.size == 0 or comp_units.size == 0:
                    continue
                delta = y_grid[:, t] - y_grid[:, base]
                cells.append(
                    _Cell(
                        g=g,
                        t=t,
                        base=base,
                        group_units=group_units,
                        comp_units=comp_units,
                        delta_group=delta[group_units],
                        delta_comp=delta[comp_units],
                        x_group=x_grid[group_units, base],
                        x_comp=x_grid[comp_units, base],
                    )
                )
        cells_by_group.append(cells)
    return group_units_all, cells_by_group


def _aggregate(
    cells_by_group: list[list[_Cell]],
    group_units_all: list[npt.NDArray[np.int64]],
    weights: npt.NDArray[np.float64],
    k: int,
) -> tuple[float, dict[int, float]] | None:
    """The overall post ATT and the per-event-time means under one weight vector.

    Overall: per group, the unweighted mean of its post cells (t at or after g),
    then the group-size-weighted mean over groups. Per event time: the
    group-size-weighted mean of the cells sharing that event time. None when no
    group has a computable post cell.
    """
    overall_num = 0.0
    overall_den = 0.0
    theta_num: dict[int, float] = {}
    theta_den: dict[int, float] = {}
    for group_index, cells in enumerate(cells_by_group):
        size = float(weights[group_units_all[group_index]].sum())
        if size <= 0.0:
            continue
        post_values: list[float] = []
        for cell in cells:
            estimate = _cell_estimate(cell, weights, k)
            if estimate is None:
                continue
            event = cell.t - cell.g
            theta_num[event] = theta_num.get(event, 0.0) + size * estimate
            theta_den[event] = theta_den.get(event, 0.0) + size
            if cell.t >= cell.g:
                post_values.append(estimate)
        if post_values:
            overall_num += size * float(np.mean(post_values))
            overall_den += size
    if overall_den == 0.0:
        return None
    theta = {event: theta_num[event] / theta_den[event] for event in theta_num}
    return overall_num / overall_den, theta


def _fit_group_time(
    prep: _Prepared,
    comparison: str,
    n_boot: int,
    rng: np.random.Generator,
    design: Design,
    event_window: tuple[int | None, int | None] | None,
) -> DiDEffect:
    if comparison == "never_treated" and not bool((prep.adoption == -1).any()):
        raise ValueError(
            "comparison='never_treated' needs units that never adopt, and every unit "
            "here adopts; try comparison='not_yet_treated'"
        )
    k = prep.n_controls
    group_units_all, cells_by_group = _build_cells(prep, comparison)

    ones = np.ones(prep.n_units, dtype=np.float64)
    records: list[GroupTimeEffect] = []
    for cells in cells_by_group:
        for cell in cells:
            estimate = _cell_estimate(cell, ones, k)
            if estimate is None:  # pragma: no cover - sides are nonempty by construction
                raise RuntimeError("a full-sample cell lost its units")
            records.append(
                GroupTimeEffect(
                    group=prep.periods[cell.g],
                    period=prep.periods[cell.t],
                    event_time=cell.t - cell.g,
                    estimate=estimate,
                    n_treated=int(cell.group_units.size),
                    n_comparison=int(cell.comp_units.size),
                )
            )
    full = _aggregate(cells_by_group, group_units_all, ones, k)
    if full is None:
        raise ValueError(
            "no group-time cell could be computed: every (group, period) pair lacks "
            "treated or comparison units observed at both the base and the target period"
        )
    overall_point, theta_point = full

    overall_draws = np.empty(n_boot, dtype=np.float64)
    theta_draws: dict[int, list[float]] = {event: [] for event in theta_point}
    got = 0
    attempts = 0
    n_units = prep.n_units
    while got < n_boot and attempts < 5 * n_boot:
        attempts += 1
        counts = np.bincount(
            rng.integers(0, n_units, n_units, dtype=np.int64), minlength=n_units
        ).astype(np.float64)
        drawn = _aggregate(cells_by_group, group_units_all, counts, k)
        if drawn is None:
            continue
        overall_value, theta_value = drawn
        overall_draws[got] = overall_value
        got += 1
        for event, value in theta_value.items():
            if event in theta_draws:
                theta_draws[event].append(value)
    if got == 0:
        raise RuntimeError(
            "the unit bootstrap produced no draws: every resample lost all group-time cells"
        )

    by_event_time: tuple[EventTimeEffect, ...] = ()
    if event_window is not None:
        leads, lags = event_window
        entries: list[EventTimeEffect] = []
        for event in sorted(theta_point):
            if leads is not None and event < -leads:
                continue
            if lags is not None and event > lags:
                continue
            event_values = np.asarray(theta_draws[event], dtype=np.float64)
            if event_values.size >= 2:
                std_error: float | None = float(np.std(event_values, ddof=1))
            else:
                std_error = None
            if event_values.size:
                interval = (
                    float(np.quantile(event_values, 0.025)),
                    float(np.quantile(event_values, 0.975)),
                )
            else:  # pragma: no cover - an event time absent from every draw
                interval = (theta_point[event], theta_point[event])
            n_treated_at = sum(
                cell.group_units.size
                for cells in cells_by_group
                for cell in cells
                if cell.t - cell.g == event
            )
            entries.append(
                EventTimeEffect(
                    event_time=event,
                    estimate=theta_point[event],
                    std_error=std_error,
                    interval=interval,
                    n_treated=int(n_treated_at),
                )
            )
        by_event_time = tuple(entries)

    used = np.zeros((prep.n_units, prep.n_periods), dtype=np.bool_)
    treated_used = np.zeros(prep.n_units, dtype=np.bool_)
    comp_used = np.zeros(prep.n_units, dtype=np.bool_)
    for cells in cells_by_group:
        for cell in cells:
            used[cell.group_units, cell.t] = True
            used[cell.group_units, cell.base] = True
            used[cell.comp_units, cell.t] = True
            used[cell.comp_units, cell.base] = True
            treated_used[cell.group_units] = True
            comp_used[cell.comp_units] = True

    diagnostics: dict[str, float] = {
        "n_units": float(prep.n_units),
        "n_treated_units": float(treated_used.sum()),
        "n_comparison_units": float(comp_used.sum()),
        "n_periods": float(prep.n_periods),
        "n_cells": float(len(records)),
        "n_dropped_units": 0.0,
    }
    return DiDEffect(
        target=EffectTarget.ATT,
        estimator="did",
        learners={},
        n=int(used.sum()),
        kind="bootstrap",
        diagnostics=diagnostics,
        _point=overall_point,
        _draws=overall_draws[:got],
        design=design,
        flavor="group_time",
        comparison=comparison,
        by_group_time=tuple(records),
        by_event_time=by_event_time,
    )


# -- public entry points -----------------------------------------------------------


def did(
    data: PanelData,
    outcome: str,
    treatment: str,
    *,
    controls: Iterable[str] = (),
    comparison: str = "never_treated",
    strategy: str = "auto",
    n_boot: int = 1000,
    random_state: int | None = None,
) -> DiDEffect:
    """Difference-in-differences: the average effect of adoption on the treated (ATT).

    The panel is read through adoption timing: a unit's adoption period is the
    first period its treatment equals 1, units that never show a 1 are never
    treated, and the treatment must be absorbing (never switching from 1 back
    to 0 within a unit). Periods compare by their position in the sorted period
    sequence, so integer and datetime periods both work.

    Flavors. ``strategy="canonical"`` needs a single adoption period g: each
    unit's outcome change from its pre-period mean (periods before g) to its
    post-period mean (periods at or after g), treated minus comparison. Units
    lacking a pre or a post observation are dropped with a warning and counted
    in ``diagnostics["n_dropped_units"]``. Controls enter through a unit-level
    regression of that change on the treated flag and the unit's pre-period
    control means. ``strategy="group_time"`` handles staggered adoption in the
    style of Callaway and Sant'Anna: one ATT(g, t) per adoption cohort and
    period, each differenced against the period just before adoption, then
    aggregated over post periods with cohort-size weights; the cells land in
    ``by_group_time``. ``strategy="twfe"`` is the static two-way fixed-effects
    regression (iterated within transform, so unbalanced panels work). Under
    staggered adoption with effects that change over time it is a weighted mix
    of cell effects with possibly negative weights (Goodman-Bacon), so prefer
    ``group_time`` there. ``strategy="auto"`` picks canonical for one adoption
    period and group_time for several, never twfe.

    Comparison pools. ``comparison="never_treated"`` uses units that never
    adopt and raises when none exist. ``comparison="not_yet_treated"`` also
    admits units that adopt only after the periods a cell reads, the pool a
    panel without never-treated units needs. In the canonical flavor the two
    pools coincide because no later adopters exist, and twfe ignores the pool.

    Inference is a unit bootstrap: resample units with replacement, recompute,
    and keep ``n_boot`` draws (a failed resample is skipped, with a budget of
    five attempts per draw). The result is ``kind="bootstrap"``; the canonical
    no-controls fit also stores the two-sample analytic standard error in
    ``diagnostics["analytic_se"]``. The parallel-trends and no-anticipation
    assumptions ride along in ``design``.

    Raises ValueError for an unknown column, a missing value in the outcome,
    treatment, or controls, a treatment not coded 0/1 or boolean, a
    non-absorbing treatment, no treated units, or an empty comparison pool.
    """
    control_list = [str(name) for name in controls]
    _check_common(comparison, n_boot)
    if strategy not in ("auto", "canonical", "twfe", "group_time"):
        raise ValueError(
            f"strategy must be 'auto', 'canonical', 'twfe', or 'group_time', got {strategy!r}"
        )
    prep = _prepare(data, outcome, treatment, control_list)
    adoption_periods = sorted(set(prep.adoption[prep.adoption >= 0].tolist()))
    if not adoption_periods:
        raise ValueError(f"no treated units: treatment {treatment!r} is never 1")
    flavor = strategy
    if strategy == "auto":
        flavor = "canonical" if len(adoption_periods) == 1 else "group_time"
    if flavor == "canonical" and len(adoption_periods) > 1:
        raise ValueError(
            f"canonical did needs one adoption period, found {len(adoption_periods)} "
            f"({[prep.periods[g] for g in adoption_periods]}); use strategy='group_time'"
        )
    rng = np.random.default_rng(random_state)
    design = _design_record(outcome, treatment, comparison)
    if flavor == "canonical":
        return _fit_canonical(prep, comparison, n_boot, rng, design)
    if flavor == "twfe":
        return _fit_twfe(prep, comparison, n_boot, rng, design)
    return _fit_group_time(prep, comparison, n_boot, rng, design, event_window=None)


def event_study(
    data: PanelData,
    outcome: str,
    treatment: str,
    *,
    controls: Iterable[str] = (),
    comparison: str = "never_treated",
    leads: int | None = None,
    lags: int | None = None,
    n_boot: int = 1000,
    random_state: int | None = None,
) -> DiDEffect:
    """The effect by event time: leads as the placebo check, lags as the dynamic path.

    Runs the group-time machinery of :func:`did` and aggregates the ATT(g, t)
    cells that share an event time, the position of the period minus the
    position of the adoption period, weighting cohorts by size. Event time -1
    is the reference: every cell differences against the period just before
    adoption, so the effect there is zero by construction and it is omitted.
    ``leads`` and ``lags`` bound the reported window (``leads=2`` keeps event
    times from -2, ``lags=3`` up to 3); ``None`` keeps everything available.

    The headline ``value`` is the overall post-adoption ATT, the same aggregate
    :func:`did` reports under ``strategy="group_time"``, and ``by_event_time``
    carries one :class:`EventTimeEffect` per event time with a bootstrap
    standard error and 95% percentile interval from the same unit bootstrap
    that drives the headline draws. A lead whose interval sits away from zero
    is evidence against parallel trends.

    Validation and the comparison pools match :func:`did`.
    """
    control_list = [str(name) for name in controls]
    _check_common(comparison, n_boot)
    if leads is not None and leads < 0:
        raise ValueError(f"leads must be non-negative, got {leads}")
    if lags is not None and lags < 0:
        raise ValueError(f"lags must be non-negative, got {lags}")
    prep = _prepare(data, outcome, treatment, control_list)
    if not bool((prep.adoption >= 0).any()):
        raise ValueError(f"no treated units: treatment {treatment!r} is never 1")
    rng = np.random.default_rng(random_state)
    design = _design_record(outcome, treatment, comparison)
    return _fit_group_time(prep, comparison, n_boot, rng, design, event_window=(leads, lags))
