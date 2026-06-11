# SPDX-License-Identifier: Apache-2.0
"""Decision points: the xor splits of a process tree, read causally.

Decision mining in the classic sense classifies which branch a case will take
at an exclusive choice. This module asks the other question: what would taking
the other branch do? At an xor node the branch taken is a treatment, the
features known when the case reaches the split are the honest adjusters, and
positivity (how thinly the data covers each branch) is a first-class output
rather than a surprise in the standard errors later.

Extraction without replay leans on one invariant of mined trees: every
activity label sits in exactly one leaf, so the children of an xor node have
disjoint activity alphabets and a case's branch is observable as which
alphabet its events hit. Both entry points check the invariant and raise on a
tree that breaks it. The landmark is the moment the case reveals its branch,
the time of its first event inside the split's combined alphabet, and every
covariate is measured strictly before that moment, mirroring the landmark
mode of :func:`~ergodic.process.case_table`.

A split with a silent branch is reported by :func:`decision_points` (a case
touching no branch alphabet counts as the silent branch when the split is
always reached, and lands in ``n_skipped`` otherwise) but refused by
:func:`decision_table`: taking the silent branch leaves no event, so without
alignment-based replay the decision moment is unobservable. Alignment-based
extraction is future work.

The default outcome is ``"remaining_time"``, seconds from the landmark to the
case's last event, because time already spent before the decision is not an
effect of the branch. Estimation stays composition: run
:func:`~ergodic.discovery.discover` and :func:`~ergodic.estimate_effect` on
the returned table, with ``treated=`` naming a branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import tabular
from ..inference._estimator import _covariate_matrix
from ..inference.learners import LogisticRegression
from ..knowledge import DomainKnowledge
from ._case_table import (
    _SECONDS_PER_DAY,
    _SECONDS_PER_HOUR,
    _attribute_type,
    _auto_attributes,
    _named_covariates,
)
from ._cases import case_attribute, case_bounds, case_slices, open_cases, time_seconds
from ._mine import ProcessModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ..data import EventLog, Ordinal, TabularData
    from ._tree import ProcessTree

__all__ = [
    "DecisionPoint",
    "DecisionTable",
    "PositivityReport",
    "decision_points",
    "decision_table",
]

_RESERVED_OUTCOMES = ("remaining_time", "cycle_time", "n_events")


@dataclass(frozen=True)
class DecisionPoint:
    """One xor split of a process tree, with the branch each case took.

    ``path`` locates the xor node in the :meth:`ProcessTree.nodes` convention.
    The branch tuples line up positionally: ``branches[j]`` names the j-th
    child (a single visible leaf by its label, any other child by its sorted
    alphabet joined with ``+``, a silent child as ``TAU``), ``alphabets[j]``
    is its activity alphabet, and ``counts[j]`` and ``shares[j]`` say how many
    assigned cases took it. A case is assigned the branch owning its first
    event inside the split's combined alphabet; events in two or more branches
    make it ambiguous (``n_ambiguous``, excluded from the counts). A case
    touching no branch counts as the silent branch when the split has one and
    is always reached; otherwise it lands in ``n_skipped``, because the log
    cannot tell a silent branch from a split never reached.
    """

    path: tuple[int, ...]
    branches: tuple[str, ...]
    alphabets: tuple[tuple[str, ...], ...]
    counts: tuple[int, ...]
    shares: tuple[float, ...]
    n_cases: int
    n_skipped: int
    n_ambiguous: int

    def __repr__(self) -> str:
        body = ", ".join(
            f"{name}={count}" for name, count in zip(self.branches, self.counts, strict=True)
        )
        extra = ""
        if self.n_skipped or self.n_ambiguous:
            extra = f", skipped={self.n_skipped}, ambiguous={self.n_ambiguous}"
        return f"DecisionPoint(path={self.path}, {body}{extra})"


@dataclass(frozen=True)
class PositivityReport:
    """Per-branch coverage of a decision table: the positivity diagnostic.

    ``counts`` and ``shares`` are per branch among the included cases, and
    ``min_share`` is the smallest share. For a two-branch split,
    ``overlap_share`` is the share of cases whose in-sample propensity (a
    logistic regression of the branch on the covariate matrix) lies inside
    [0.05, 0.95]; it is None for wider splits, and 1.0 when there are no
    covariates, since the propensity is then the base rate for every case and
    the covariates cannot starve either branch.

    ``flagged`` is True when ``min_share`` falls below 0.05 (one branch holds
    under five percent of the cases), any branch holds fewer than 10 cases
    (too few for a stable branch mean), or ``overlap_share`` exists and falls
    below 0.8 (for over a fifth of the cases the covariates nearly decide the
    branch, so the comparison there rests on extrapolation). A flag is a
    warning that the branches barely overlap, not a verdict that an estimate
    is wrong.
    """

    branches: tuple[str, ...]
    counts: tuple[int, ...]
    shares: tuple[float, ...]
    min_share: float
    overlap_share: float | None
    flagged: bool


@dataclass(frozen=True)
class DecisionTable:
    """A decision point encoded for the causal stack, one row per assigned case.

    ``data`` holds the sorted covariates, the categorical ``branch`` treatment,
    and the outcome, one row per unambiguously assigned case in case order.
    ``knowledge`` carries the tier order covariates, then branch, then outcome.
    ``point`` is the underlying :class:`DecisionPoint`, ``positivity`` the
    coverage diagnostic, and ``n_dropped_ambiguous`` counts the cases excluded
    for touching two or more branches.
    """

    data: TabularData
    knowledge: DomainKnowledge
    outcome: str
    covariates: tuple[str, ...]
    point: DecisionPoint
    positivity: PositivityReport
    n_cases: int
    n_dropped_ambiguous: int
    treatment: str = "branch"

    def __repr__(self) -> str:
        dropped = ""
        if self.n_dropped_ambiguous:
            dropped = f", dropped_ambiguous={self.n_dropped_ambiguous}"
        flag = ", positivity flagged" if self.positivity.flagged else ""
        return (
            f"DecisionTable(cases={self.n_cases}, branches={self.point.branches}, "
            f"outcome={self.outcome!r}, covariates={len(self.covariates)}{dropped}{flag})"
        )


# -- tree reading ---------------------------------------------------------------


def _check_unique_labels(tree: ProcessTree) -> None:
    """Raise when an activity label sits in more than one leaf of ``tree``."""
    seen: set[str] = set()
    for _, node in tree.nodes():
        if node.op is None and node.label is not None:
            if node.label in seen:
                raise ValueError(
                    f"activity {node.label!r} appears in more than one leaf; decision "
                    "extraction reads a case's branch from which alphabet its events "
                    "hit, which needs every activity in exactly one leaf (a mined tree "
                    "has this by construction)"
                )
            seen.add(node.label)


def _xor_paths(tree: ProcessTree) -> tuple[tuple[int, ...], ...]:
    """The paths of every xor node, in depth-first order."""
    return tuple(path for path, node in tree.nodes() if node.op == "xor")


def _always_reached(tree: ProcessTree, path: tuple[int, ...]) -> bool:
    """Whether every run of the tree reaches the node at ``path``.

    A seq or par ancestor runs all its children; a loop always runs its do
    child (index 0) but may skip the redo; an xor may pick another child.
    """
    node = tree
    for i in path:
        if node.op == "xor" or (node.op == "loop" and i != 0):
            return False
        node = node.children[i]
    return True


def _branch_name(alphabet: tuple[str, ...]) -> str:
    """The branch name: the sorted alphabet joined with ``+``, or ``TAU`` when empty."""
    return "+".join(alphabet) if alphabet else "TAU"


# -- branch assignment ------------------------------------------------------------


def _assign_branches(
    log: EventLog, alphabets: tuple[tuple[str, ...], ...]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Per case: the branch index taken and the row of the first hit.

    The branch is the one owning the case's first event inside the combined
    alphabet; -1 means no event hit any branch and -2 means events hit two or
    more branches (ambiguous). The row index is -1 when nothing hit.
    """
    owner = {act: j for j, alphabet in enumerate(alphabets) for act in alphabet}
    activities = log.column(log.activity)
    _, starts = case_slices(log)
    n = starts.shape[0] - 1
    branch = np.full(n, -1, dtype=np.int64)
    hit_row = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for r in range(int(starts[i]), int(starts[i + 1])):
            j = owner.get(str(activities[r]))
            if j is None:
                continue
            if branch[i] == -1:
                branch[i] = j
                hit_row[i] = r
            elif j != int(branch[i]):
                branch[i] = -2
                break
    return branch, hit_row


def _point_at(
    log: EventLog, tree: ProcessTree, path: tuple[int, ...]
) -> tuple[DecisionPoint, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The :class:`DecisionPoint` at ``path``, with the per-case assignment arrays."""
    node = tree.subtree(path)
    alphabets = tuple(child.activities() for child in node.children)
    branches = tuple(_branch_name(alphabet) for alphabet in alphabets)
    branch_idx, hit_row = _assign_branches(log, alphabets)
    counts = [int(np.sum(branch_idx == j)) for j in range(len(alphabets))]
    n_none = int(np.sum(branch_idx == -1))
    n_ambiguous = int(np.sum(branch_idx == -2))
    silent = next((j for j, alphabet in enumerate(alphabets) if not alphabet), None)
    n_skipped = n_none
    if silent is not None and _always_reached(tree, path):
        # The split always runs, so a case with no visible hit took the silent
        # branch; when reaching the split is optional the log cannot tell the
        # two apart and the case stays in n_skipped.
        counts[silent] += n_none
        n_skipped = 0
    n_cases = int(sum(counts))
    shares = tuple((count / n_cases if n_cases else 0.0) for count in counts)
    point = DecisionPoint(
        path=path,
        branches=branches,
        alphabets=alphabets,
        counts=tuple(counts),
        shares=shares,
        n_cases=n_cases,
        n_skipped=n_skipped,
        n_ambiguous=n_ambiguous,
    )
    return point, branch_idx, hit_row


def decision_points(log: EventLog, model: ProcessTree | ProcessModel) -> tuple[DecisionPoint, ...]:
    """Locate every xor split of ``model`` and the branch each case of ``log`` took.

    ``model`` is a process tree or a mined :class:`~ergodic.process.ProcessModel`
    (its tree is read). Returns one :class:`DecisionPoint` per xor node, in
    depth-first order. The tree must carry each activity label in exactly one
    leaf (a mined tree does, by construction); a duplicated label raises, since
    branch membership would then be ambiguous by design rather than by data.
    """
    tree = model.tree if isinstance(model, ProcessModel) else model
    _check_unique_labels(tree)
    return tuple(_point_at(log, tree, path)[0] for path in _xor_paths(tree))


# -- the table encoder -------------------------------------------------------------


def _resolve_path(
    tree: ProcessTree, point: tuple[int, ...] | DecisionPoint | None
) -> tuple[int, ...]:
    """The xor path picked by ``point``, or the only one when ``point`` is None."""
    paths = _xor_paths(tree)
    if point is None:
        if len(paths) == 1:
            return paths[0]
        if not paths:
            raise ValueError("the tree has no xor node, so there is no decision to analyze")
        listing = ", ".join(str(path) for path in paths)
        raise ValueError(f"the tree has {len(paths)} xor nodes; pass point= to pick one: {listing}")
    path = point.path if isinstance(point, DecisionPoint) else tuple(point)
    if path not in paths:
        raise ValueError(f"point {path} is not an xor node; the xor paths are {list(paths)}")
    return path


def _outcome_per_case(
    log: EventLog,
    outcome: str,
    first: npt.NDArray[np.float64],
    last: npt.NDArray[np.float64],
    landmark: npt.NDArray[np.float64],
    starts: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[Any], str | Ordinal]:
    """One outcome value per case in case order, with its declared type."""
    if outcome in _RESERVED_OUTCOMES:
        if outcome in log.schema:
            raise ValueError(
                f"outcome {outcome!r} is a reserved derived name and also a log column; "
                "rename the column or pass it under a different outcome name"
            )
        if outcome == "remaining_time":
            return last - landmark, "continuous"
        if outcome == "cycle_time":
            return last - first, "continuous"
        return np.diff(starts).astype(np.float64), "continuous"
    if outcome in set(log.variables):
        return case_attribute(log, outcome), _attribute_type(log.schema[outcome])
    raise ValueError(
        f"outcome {outcome!r} is not 'remaining_time', 'cycle_time', 'n_events', or a "
        f"case-constant column; columns are {sorted(log.variables)}"
    )


def _positivity(
    data: TabularData, covariates: tuple[str, ...], branches: tuple[str, ...]
) -> PositivityReport:
    """The coverage diagnostic over the finished table."""
    labels = data.column("branch")
    n = data.n_rows
    counts = tuple(int(np.sum(labels == name)) for name in branches)
    shares = tuple(count / n for count in counts)
    min_share = float(min(shares))
    overlap: float | None = None
    if len(branches) == 2:
        design = _covariate_matrix(data, list(covariates))
        if design.shape[1] == 0:
            overlap = 1.0
        else:
            target = np.asarray(labels == branches[1], dtype=np.float64)
            propensity = LogisticRegression().fit(design, target).predict_proba(design)[:, 1]
            overlap = float(np.mean((propensity >= 0.05) & (propensity <= 0.95)))
    flagged = (
        min_share < 0.05
        or any(count < 10 for count in counts)
        or (overlap is not None and overlap < 0.8)
    )
    return PositivityReport(
        branches=branches,
        counts=counts,
        shares=shares,
        min_share=min_share,
        overlap_share=overlap,
        flagged=flagged,
    )


def decision_table(
    log: EventLog,
    model: ProcessTree | ProcessModel,
    *,
    point: tuple[int, ...] | DecisionPoint | None = None,
    outcome: str = "remaining_time",
    covariates: str | Sequence[str] = "auto",
) -> DecisionTable:
    """Encode one xor split of ``model`` as a causal table: covariates, branch, outcome.

    The landmark is per case the time of its first event inside the split's
    combined alphabet, the moment the case reveals its branch. Only cases
    assigned exactly one branch enter the table; cases touching two or more
    branches are dropped and counted in ``n_dropped_ambiguous``. A split with
    a silent (tau or empty-alphabet) branch raises: taking it leaves no event,
    so the decision moment is unobservable without alignment-based replay.

    Args:
        log: The event log the tree describes.
        model: A process tree with each activity in exactly one leaf, or a
            mined :class:`~ergodic.process.ProcessModel` (its tree is read).
        point: The xor node to analyze: its path, a :class:`DecisionPoint`, or
            None when the tree has exactly one xor node (more than one raises,
            listing the paths).
        outcome: ``"remaining_time"`` (seconds from the landmark to the case's
            last event; the default, because time already spent before the
            decision is not an effect of the branch), ``"cycle_time"``,
            ``"n_events"``, or a case-constant column. The three derived names
            are reserved: a log column by any of them raises.
        covariates: ``"auto"`` takes every case-constant variable column plus
            the derived features, each measured strictly before the landmark:
            events and elapsed seconds before it (``n_events_prefix``,
            ``prefix_activities``, ``elapsed_at_landmark``), the open-case
            workload at the start and at the landmark, and the calendar
            features when the timestamps are datetimes. A sequence restricts
            to the named columns and derived features. A derived covariate
            that is constant across the kept cases is dropped in silence.

    Returns:
        A :class:`DecisionTable`: the :class:`~ergodic.data.TabularData`
        (columns in the order sorted covariates, ``branch``, outcome; one row
        per kept case, in case order), the tiered
        :class:`~ergodic.DomainKnowledge`, the :class:`DecisionPoint`, and the
        :class:`PositivityReport`.
    """
    tree = model.tree if isinstance(model, ProcessModel) else model
    _check_unique_labels(tree)
    path = _resolve_path(tree, point)
    node = tree.subtree(path)
    if any(not child.activities() for child in node.children):
        raise ValueError(
            f"the split at {path} has a silent branch: taking it leaves no event, so "
            "the decision moment is unobservable without alignment-based replay; "
            "alignment-based extraction is future work"
        )
    if "branch" in log.schema.names:
        raise ValueError(
            "column 'branch' collides with the treatment column of the decision table; "
            "rename it so the treatment is unambiguous"
        )
    if outcome == "branch":
        raise ValueError("outcome 'branch' is the treatment column; pick another outcome")

    record, branch_idx, hit_row = _point_at(log, tree, path)
    keep = branch_idx >= 0
    if not bool(keep.any()):
        raise ValueError(
            f"no case takes the split at {path} unambiguously: {record.n_skipped} touch "
            f"no branch and {record.n_ambiguous} touch more than one"
        )

    _, starts = case_slices(log)
    _, first, last = case_bounds(log)
    stamps = time_seconds(log.column(log.timestamp))
    landmark = first.copy()
    landmark[keep] = stamps[hit_row[keep]]

    outcome_all, outcome_type = _outcome_per_case(log, outcome, first, last, landmark, starts)

    activities = log.column(log.activity)
    n_all = int(first.shape[0])
    n_prefix = np.zeros(n_all, dtype=np.float64)
    distinct = np.zeros(n_all, dtype=np.float64)
    for i in np.flatnonzero(keep):
        s, r = int(starts[i]), int(hit_row[i])
        n_prefix[i] = float(r - s)
        distinct[i] = float(len(set(activities[s:r].tolist())))

    derived: dict[str, npt.NDArray[np.float64]] = {
        "workload_at_start": open_cases(first, last, first),
        "elapsed_at_landmark": landmark - first,
        "n_events_prefix": n_prefix,
        "prefix_activities": distinct,
        "workload_at_landmark": open_cases(first, last, landmark),
    }
    if log.schema[log.timestamp].is_datetime:
        days = np.floor(first / _SECONDS_PER_DAY)
        derived["start_weekday"] = np.mod(days + 3.0, 7.0)  # the epoch was a Thursday
        derived["start_hour"] = np.floor((first - days * _SECONDS_PER_DAY) / _SECONDS_PER_HOUR)

    clash = sorted(set(derived) & set(log.schema.names))
    if clash:
        raise ValueError(
            f"columns {clash} collide with the derived covariate names of the decision "
            "table; rename the columns so the covariates are unambiguous"
        )

    if isinstance(covariates, str) and covariates == "auto":
        attributes = _auto_attributes(log, "branch", outcome)
        chosen_derived = dict(derived)
    else:
        requested = [covariates] if isinstance(covariates, str) else list(covariates)
        attributes, chosen_derived = _named_covariates(
            log, requested, derived, "branch", outcome, "decision"
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
    ordered["branch"] = np.array([record.branches[int(j)] for j in branch_idx[keep]])
    types["branch"] = "categorical"
    ordered[outcome] = outcome_all[keep]
    types[outcome] = outcome_type
    data = tabular(ordered, types=types)

    tiers: list[list[str]] = []
    if covariate_names:
        tiers.append(list(covariate_names))
    tiers.extend((["branch"], [outcome]))
    knowledge = DomainKnowledge().with_tiers(tiers)

    return DecisionTable(
        data=data,
        knowledge=knowledge,
        outcome=outcome,
        covariates=tuple(covariate_names),
        point=record,
        positivity=_positivity(data, tuple(covariate_names), record.branches),
        n_cases=int(keep.sum()),
        n_dropped_ambiguous=record.n_ambiguous,
    )
