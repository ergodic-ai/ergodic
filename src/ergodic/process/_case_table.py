# SPDX-License-Identifier: Apache-2.0
"""Case table: one row per case, encoded so the causal stack can read it.

``case_table`` turns an :class:`~ergodic.data.EventLog` into a
:class:`~ergodic.data.TabularData` with one row per case: a binary treatment,
an outcome, and covariates measured no later than the moment the treatment
question is asked. The ``when`` mode fixes that moment, and the timestamp
gating is the whole point: nothing measured after the treatment can land in
the covariates, so the table cannot leak the outcome back into the adjustment
set.

With ``when="anytime"`` the treatment reads "did the activity ever happen in
the case", so the only safe covariates are case-start facts: the case-constant
attribute columns, the number of concurrently open cases at the first event
(``workload_at_start``), and, when the timestamp column is a datetime, the
calendar features ``start_weekday`` and ``start_hour``. A constant column is
assumed known at case start; if an attribute is recorded after the fact, it
does not belong in the covariates.

The landmark modes ask the question at a fixed point in each case's life:
``when=("after_events", k)`` at the k-th event, ``when=("after_elapsed", x)``
at ``x`` seconds after the first event. Cases that never reach the landmark
are dropped and counted in ``n_dropped_short``; cases already treated by the
landmark are dropped and counted in ``n_dropped_treated``; the treatment
becomes "did the activity happen after the landmark". Prefix features
measured at the landmark (``elapsed_at_landmark``, ``workload_at_landmark``,
``prefix_activities``, ``n_events_prefix``) then join the case-start
features. The estimand is the effect among cases that reached the landmark
untreated, and that is the honest population for a timing-aware question.

A derived covariate that is constant across the kept cases is dropped in
silence, because a zero-variance column breaks discovery. Under
``after_elapsed`` the drop is expected: ``elapsed_at_landmark`` always equals
``x`` there.

The returned :class:`CaseTable` also carries a
:class:`~ergodic.DomainKnowledge` with the tiers covariates, then treatment,
then outcome. Discovery and identification read it, so adjusting for anything
downstream of the treatment is forbidden mechanically, not by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import Ordinal, tabular
from ..knowledge import DomainKnowledge
from ._cases import case_attribute, case_bounds, case_slices, open_cases, time_seconds

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from ..data import ColumnType, EventLog, TabularData

__all__ = ["CaseTable", "case_table"]

_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_HOUR = 3600.0
_CALENDAR_NAMES = ("start_hour", "start_weekday")
_LANDMARK_NAMES = (
    "elapsed_at_landmark",
    "n_events_prefix",
    "prefix_activities",
    "workload_at_landmark",
)


@dataclass(frozen=True)
class CaseTable:
    """An event log flattened to one row per case, ready for the causal stack.

    ``data`` holds the covariates (sorted), the treatment, and the outcome, one
    row per kept case in case order. ``knowledge`` carries the tier order
    covariates, then treatment, then outcome. The drop counters are zero
    outside the landmark modes.
    """

    data: TabularData
    knowledge: DomainKnowledge
    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    when: str
    n_cases: int
    n_dropped_short: int = 0
    n_dropped_treated: int = 0

    def __repr__(self) -> str:
        dropped = ""
        if self.n_dropped_short or self.n_dropped_treated:
            dropped = (
                f", dropped_short={self.n_dropped_short}, dropped_treated={self.n_dropped_treated}"
            )
        return (
            f"CaseTable(cases={self.n_cases}, treatment={self.treatment!r}, "
            f"outcome={self.outcome!r}, covariates={len(self.covariates)}, "
            f"when={self.when!r}{dropped})"
        )


# -- input parsing -------------------------------------------------------------


def _parse_when(when: str | tuple[str, int | float]) -> tuple[str, int | float | None]:
    """Validate ``when`` and return ``(mode, parameter)``."""
    if isinstance(when, str):
        if when == "anytime":
            return "anytime", None
        if when in ("after_events", "after_elapsed"):
            raise ValueError(
                f"when={when!r} needs a parameter; pass when=({when!r}, k) with the "
                "event count or elapsed seconds"
            )
        raise ValueError(
            f"unknown when={when!r}; use 'anytime', ('after_events', k), or ('after_elapsed', x)"
        )
    if isinstance(when, tuple) and len(when) == 2:
        mode, parameter = when
        if mode == "after_events":
            if isinstance(parameter, bool) or not isinstance(parameter, int):
                raise ValueError(f"after_events needs an integer event count, got {parameter!r}")
            if parameter < 1:
                raise ValueError(f"after_events needs k >= 1, got k={parameter}")
            return mode, parameter
        if mode == "after_elapsed":
            if isinstance(parameter, bool) or not isinstance(parameter, int | float):
                raise ValueError(f"after_elapsed needs a number of seconds, got {parameter!r}")
            if not parameter > 0:
                raise ValueError(f"after_elapsed needs x > 0, got x={parameter}")
            return mode, float(parameter)
        raise ValueError(
            f"unknown landmark mode {mode!r}; use ('after_events', k) or ('after_elapsed', x)"
        )
    raise ValueError(f"when must be 'anytime' or a (mode, parameter) tuple, got {when!r}")


def _attribute_type(column: ColumnType) -> str | Ordinal:
    """The ``types=`` spec that keeps a log column's kind in the case table."""
    if column.is_ordinal and column.categories is not None:
        return Ordinal(column.categories)
    return column.kind.value


def _binary_treatment(values: npt.NDArray[Any], name: str) -> npt.NDArray[np.int64]:
    """A column treatment as int64 0/1, or an error naming the offending values."""
    if values.dtype.kind == "b":
        return values.astype(np.int64)
    if values.dtype.kind in "iuf":
        floats = np.asarray(values, dtype=np.float64)
        if np.isnan(floats).any():
            raise ValueError(f"treatment column {name!r} has missing values")
        levels = sorted(set(floats.tolist()))
        if set(levels) <= {0.0, 1.0}:
            return floats.astype(np.int64)
        raise ValueError(
            f"treatment column {name!r} must hold only 0/1 or booleans; observed {levels}"
        )
    raise ValueError(
        f"treatment column {name!r} must hold only 0/1 or booleans; its dtype is {values.dtype}"
    )


def _treatment_source(log: EventLog, treatment: str) -> str:
    """Whether ``treatment`` names an activity or a column, rejecting ambiguity."""
    as_activity = treatment in set(log.activities)
    as_column = treatment in set(log.variables)
    if as_activity and as_column:
        raise ValueError(
            f"treatment {treatment!r} is both an activity and a column; "
            "rename one of them so the treatment is unambiguous"
        )
    if as_activity:
        return "activity"
    if as_column:
        return "column"
    raise ValueError(
        f"treatment {treatment!r} is neither an activity nor a column; "
        f"activities are {sorted(log.activities)} and columns are {sorted(log.variables)}"
    )


# -- per-mode case arithmetic --------------------------------------------------


def _after_events(
    log: EventLog,
    treatment_hit: npt.NDArray[np.bool_],
    starts: npt.NDArray[np.int64],
    stamps: npt.NDArray[np.float64],
    first: npt.NDArray[np.float64],
    last: npt.NDArray[np.float64],
    k: int,
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.int64],
    dict[str, npt.NDArray[np.float64]],
    int,
    int,
]:
    """Risk set, treatment, and prefix features at each case's k-th event."""
    activities = log.column(log.activity)
    n_all = first.shape[0]
    n_events: npt.NDArray[np.int64] = np.diff(starts)
    short = n_events <= k
    treated_prefix = np.zeros(n_all, dtype=bool)
    treated = np.zeros(n_all, dtype=np.int64)
    landmark = first.copy()
    distinct = np.zeros(n_all, dtype=np.float64)
    for i in range(n_all):
        if short[i]:
            continue
        s, e = int(starts[i]), int(starts[i + 1])
        p = s + k
        treated_prefix[i] = bool(treatment_hit[s:p].any())
        treated[i] = int(bool(treatment_hit[p:e].any()))
        landmark[i] = stamps[p - 1]
        distinct[i] = float(len(set(activities[s:p].tolist())))
    keep = ~short & ~treated_prefix
    derived = {
        "elapsed_at_landmark": landmark - first,
        "n_events_prefix": np.full(n_all, float(k)),
        "prefix_activities": distinct,
        "workload_at_landmark": open_cases(first, last, landmark),
    }
    return keep, treated, derived, int(short.sum()), int((~short & treated_prefix).sum())


def _after_elapsed(
    log: EventLog,
    treatment_hit: npt.NDArray[np.bool_],
    starts: npt.NDArray[np.int64],
    stamps: npt.NDArray[np.float64],
    first: npt.NDArray[np.float64],
    last: npt.NDArray[np.float64],
    x: float,
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.int64],
    dict[str, npt.NDArray[np.float64]],
    int,
    int,
]:
    """Risk set, treatment, and prefix features at first event time plus ``x``."""
    activities = log.column(log.activity)
    n_all = first.shape[0]
    landmark = first + x
    still_open = last > landmark
    treated_by = np.zeros(n_all, dtype=bool)
    treated = np.zeros(n_all, dtype=np.int64)
    n_prefix = np.zeros(n_all, dtype=np.float64)
    distinct = np.zeros(n_all, dtype=np.float64)
    for i in range(n_all):
        s, e = int(starts[i]), int(starts[i + 1])
        in_prefix = stamps[s:e] <= landmark[i]
        hits = treatment_hit[s:e]
        treated_by[i] = bool((hits & in_prefix).any())
        treated[i] = int(bool((hits & ~in_prefix).any()))
        n_prefix[i] = float(in_prefix.sum())
        distinct[i] = float(len(set(activities[s:e][in_prefix].tolist())))
    keep = still_open & ~treated_by
    derived = {
        "elapsed_at_landmark": np.full(n_all, float(x)),
        "n_events_prefix": n_prefix,
        "prefix_activities": distinct,
        "workload_at_landmark": open_cases(first, last, landmark),
    }
    return keep, treated, derived, int((~still_open).sum()), int((still_open & treated_by).sum())


# -- covariate assembly --------------------------------------------------------


def _auto_attributes(log: EventLog, treatment: str, outcome: str) -> dict[str, npt.NDArray[Any]]:
    """Every case-constant variable column, skipping the ones that vary."""
    out: dict[str, npt.NDArray[Any]] = {}
    for name in log.variables:
        if name in (treatment, outcome):
            continue
        try:
            out[name] = case_attribute(log, name)
        except ValueError:
            continue  # varies within a case: an event attribute, not a covariate
    return out


def _named_covariates(
    log: EventLog,
    requested: Iterable[str],
    derived: dict[str, npt.NDArray[np.float64]],
    treatment: str,
    outcome: str,
    mode: str,
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, npt.NDArray[np.float64]]]:
    """Resolve an explicit covariate list into attribute columns and derived features."""
    variable_names = set(log.variables)
    attributes: dict[str, npt.NDArray[Any]] = {}
    chosen: dict[str, npt.NDArray[np.float64]] = {}
    for name in dict.fromkeys(requested):
        if name in (treatment, outcome):
            role = "treatment" if name == treatment else "outcome"
            raise ValueError(f"covariate {name!r} is the {role}; it cannot also be a covariate")
        if name in derived:
            chosen[name] = derived[name]
        elif name in variable_names:
            attributes[name] = case_attribute(log, name)
        elif name in _CALENDAR_NAMES:
            kind = log.schema[log.timestamp].kind.value
            raise ValueError(
                f"covariate {name!r} needs a datetime timestamp column; {log.timestamp!r} is {kind}"
            )
        elif name in _LANDMARK_NAMES:
            raise ValueError(
                f"covariate {name!r} is only computed by the landmark modes; when={mode!r}"
            )
        else:
            raise ValueError(
                f"unknown covariate {name!r}; columns are {sorted(variable_names)} "
                f"and derived features are {sorted(derived)}"
            )
    return attributes, chosen


def _outcome_values(
    log: EventLog,
    outcome: str,
    first: npt.NDArray[np.float64],
    last: npt.NDArray[np.float64],
    starts: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[Any], str | Ordinal]:
    """One outcome value per case in case order, with its declared type."""
    if outcome in ("cycle_time", "n_events"):
        if outcome in log.schema:
            raise ValueError(
                f"outcome {outcome!r} is a reserved derived name and also a log column; "
                "rename the column or pass it under a different outcome name"
            )
        if outcome == "cycle_time":
            return last - first, "continuous"
        return np.diff(starts).astype(np.float64), "continuous"
    if outcome in set(log.variables):
        return case_attribute(log, outcome), _attribute_type(log.schema[outcome])
    raise ValueError(
        f"outcome {outcome!r} is not 'cycle_time', 'n_events', or a case-constant "
        f"column; columns are {sorted(log.variables)}"
    )


# -- the encoder ----------------------------------------------------------------


def case_table(
    log: EventLog,
    treatment: str,
    outcome: str = "cycle_time",
    *,
    covariates: str | Iterable[str] = "auto",
    when: str | tuple[str, int | float] = "anytime",
) -> CaseTable:
    """Encode an event log as one row per case: covariates, treatment, outcome.

    Args:
        log: The event log to flatten.
        treatment: An activity name (the treatment is whether it occurred, per
            ``when``) or a case-constant 0/1 column (``when="anytime"`` only,
            since a column carries no timing). A name that is both raises.
        outcome: ``"cycle_time"`` (case duration in seconds), ``"n_events"``
            (events per case), or a case-constant column. The two derived
            names are reserved: a log column by either name raises.
        covariates: ``"auto"`` takes every case-constant variable column plus
            the mode's derived features; an iterable names columns (validated
            case-constant) and/or derived features. A derived covariate that
            is constant across the kept cases is dropped in silence.
        when: ``"anytime"``, ``("after_events", k)`` with ``k >= 1``, or
            ``("after_elapsed", x)`` with ``x > 0`` seconds. The landmark
            modes keep only cases that reached the landmark untreated, so
            their estimand is the effect in that population.

    Returns:
        A :class:`CaseTable`: the :class:`~ergodic.data.TabularData` (columns
        in the order sorted covariates, treatment, outcome; one row per kept
        case, in case order), the tiered :class:`~ergodic.DomainKnowledge`,
        and the drop counters of the landmark modes.
    """
    mode, parameter = _parse_when(when)
    if treatment == outcome:
        raise ValueError(f"treatment and outcome must differ; both are {treatment!r}")
    _, starts = case_slices(log)
    _, first, last = case_bounds(log)
    n_all = int(first.shape[0])
    if n_all == 0:
        raise ValueError("the event log has no cases")
    stamps = time_seconds(log.column(log.timestamp))
    outcome_all, outcome_type = _outcome_values(log, outcome, first, last, starts)
    source = _treatment_source(log, treatment)

    derived: dict[str, npt.NDArray[np.float64]] = {
        "workload_at_start": open_cases(first, last, first)
    }
    if log.schema[log.timestamp].is_datetime:
        days = np.floor(first / _SECONDS_PER_DAY)
        derived["start_weekday"] = np.mod(days + 3.0, 7.0)  # the epoch was a Thursday
        derived["start_hour"] = np.floor((first - days * _SECONDS_PER_DAY) / _SECONDS_PER_HOUR)

    keep = np.ones(n_all, dtype=bool)
    n_dropped_short = 0
    n_dropped_treated = 0
    if source == "column":
        if mode != "anytime":
            raise ValueError(
                f"treatment column {treatment!r} carries no event timing, so a landmark "
                "mode cannot tell whether it preceded the landmark; use when='anytime' "
                "or name a treatment activity"
            )
        treated = _binary_treatment(case_attribute(log, treatment), treatment)
    else:
        hit = np.asarray(log.column(log.activity) == treatment, dtype=bool)
        if mode == "anytime":
            treated = np.logical_or.reduceat(hit, starts[:-1]).astype(np.int64)
        elif mode == "after_events":
            assert isinstance(parameter, int)  # narrowed by _parse_when
            keep, treated, extra, n_dropped_short, n_dropped_treated = _after_events(
                log, hit, starts, stamps, first, last, parameter
            )
            derived.update(extra)
        else:
            assert isinstance(parameter, float)  # narrowed by _parse_when
            keep, treated, extra, n_dropped_short, n_dropped_treated = _after_elapsed(
                log, hit, starts, stamps, first, last, parameter
            )
            derived.update(extra)

    clash = sorted(set(derived) & set(log.schema.names))
    if clash:
        raise ValueError(
            f"columns {clash} collide with the derived feature names of when={mode!r}; "
            "rename the columns so the covariates are unambiguous"
        )

    if isinstance(covariates, str) and covariates == "auto":
        attributes = _auto_attributes(log, treatment, outcome)
        chosen_derived = dict(derived)
    else:
        requested = [covariates] if isinstance(covariates, str) else list(covariates)
        attributes, chosen_derived = _named_covariates(
            log, requested, derived, treatment, outcome, mode
        )

    if not bool(keep.any()):
        raise ValueError(
            f"no cases remain at the landmark: {n_dropped_short} never reached it "
            f"and {n_dropped_treated} were already treated"
        )

    columns: dict[str, npt.NDArray[Any]] = {}
    types: dict[str, str | Ordinal] = {}
    for name in attributes:
        columns[name] = attributes[name][keep]
        types[name] = _attribute_type(log.schema[name])
    for name, values in chosen_derived.items():
        kept_values = values[keep]
        if bool(np.all(kept_values == kept_values[0])):
            continue  # constant across the kept cases: useless and harmful downstream
        columns[name] = kept_values
        types[name] = "continuous"

    covariate_names = sorted(columns)
    ordered: dict[str, npt.NDArray[Any]] = {name: columns[name] for name in covariate_names}
    ordered[treatment] = treated[keep]
    types[treatment] = "categorical"
    ordered[outcome] = outcome_all[keep]
    types[outcome] = outcome_type
    data = tabular(ordered, types=types)

    tiers: list[list[str]] = []
    if covariate_names:
        tiers.append(list(covariate_names))
    tiers.extend(([treatment], [outcome]))
    knowledge = DomainKnowledge().with_tiers(tiers)

    return CaseTable(
        data=data,
        knowledge=knowledge,
        treatment=treatment,
        outcome=outcome,
        covariates=tuple(covariate_names),
        when=mode,
        n_cases=int(keep.sum()),
        n_dropped_short=n_dropped_short,
        n_dropped_treated=n_dropped_treated,
    )
