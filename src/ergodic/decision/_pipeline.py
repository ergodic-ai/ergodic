# SPDX-License-Identifier: Apache-2.0
"""Graph-identified, single-variable decisions with context-only defaults."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..data import tabular
from ..graph import GraphKind, MixedGraph
from ..identification import identify_effect, is_valid_adjustment_set
from ..identification._criteria import _adjustment_formula
from ..inference.estimators import DML, ContinuousDR
from ._categorical import ArmDR

PROTOCOL = "decision-contrasts_v1"
PROTOCOLS = (PROTOCOL, "decision-contrasts_v2")
# Query-row ranking methods for discrete treatments under decision-contrasts_v2.
RANKING_METHODS = ("plugin", "s", "dr")
"""Version of intervention defaults, class semantics and ranking policy."""


def _label(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _levels(series: pd.Series[Any]) -> list[Any]:
    return sorted([_label(v) for v in series.unique()], key=lambda v: (type(v).__name__, str(v)))


def _kind(series: pd.Series[Any]) -> str:
    if len(series.unique()) == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "continuous"
    return "categorical"


@dataclass(frozen=True)
class Intervention:
    """One actionable variable's method and optional explicit dose contrast.

    Omitted levels use context mode for discrete variables and median for numeric
    variables, against other labels or the 25th/75th percentiles respectively.
    Declare ``kind='categorical'`` for integer-coded categories with >2 labels.
    """

    method: str | None = None
    baseline: Any = None
    values: tuple[Any, ...] | None = None
    kind: str | None = None
    bandwidth: float | None = None
    validation_probability: float | None = None


class _Design:
    def __init__(self, context: pd.DataFrame, names: list[str], types: Mapping[str, str]):
        self.names = names
        self.ranges = {
            name: (float(context[name].min()), float(context[name].max()))
            for name in names
            if types.get(name, _kind(context[name])) == "continuous"
        }
        self.categories = {
            name: _levels(context[name])
            for name in names
            if types.get(name, _kind(context[name])) != "continuous"
        }

    def matrix(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        parts = []
        for name in self.names:
            if name not in frame:
                raise ValueError(f"query is missing adjustment variable {name!r}")
            values = frame[name]
            if name in self.categories:
                categories = self.categories[name]
                if not values.isin(categories).all():
                    raise ValueError(f"unknown query category in {name!r}")
                parts.extend([(values == label).to_numpy(dtype=float) for label in categories[1:]])
            else:
                numeric = values.to_numpy(dtype=float)
                lo, hi = self.ranges[name]
                if np.any(numeric < lo) or np.any(numeric > hi):
                    raise ValueError(
                        f"query covariate {name!r} is outside observed context support"
                    )
                parts.append(numeric)
        result = np.column_stack(parts) if parts else np.empty((len(frame), 0))
        if not np.isfinite(result).all():
            raise ValueError("adjustment variables must be finite and nonmissing")
        return result


def _resolved(
    series: pd.Series[Any],
    spec: Intervention,
    protocol: str = PROTOCOL,
    target_kind: str | None = None,
) -> tuple[str, Any, list[Any], str]:
    kind = spec.kind or _kind(series)
    if kind not in ("continuous", "binary", "categorical"):
        raise ValueError("variable kind must be continuous, binary or categorical")
    labels = _levels(series)
    if len(labels) < 2 or (kind == "binary" and len(labels) != 2):
        raise ValueError("treatment has an invalid number of distinct levels")
    if kind == "continuous":
        numeric = series.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("continuous treatment must be finite")
        baseline = float(np.median(numeric)) if spec.baseline is None else spec.baseline
        values = (
            list(np.quantile(numeric, [0.25, 0.75])) if spec.values is None else list(spec.values)
        )
        method = spec.method or "partially_linear"
        if method not in ("partially_linear", "continuous_dr"):
            raise ValueError("continuous treatment supports partially_linear or continuous_dr")
    else:
        counts = series.value_counts()
        baseline = (
            next(label for label in labels if counts[label] == counts.max())
            if spec.baseline is None
            else spec.baseline
        )
        if spec.baseline is None and kind == "binary" and set(labels) == {0, 1}:
            baseline = next(label for label in labels if label == 0)
        values = (
            [label for label in labels if label != baseline]
            if spec.values is None
            else list(spec.values)
        )
        method = spec.method or (
            "plugin"
            if kind == "binary" and target_kind == "binary" and protocol == "decision-contrasts_v2"
            else "dr"
        )
        allowed = (*RANKING_METHODS, "auto") if protocol == "decision-contrasts_v2" else ("dr",)
        if method not in allowed or (method == "auto" and kind != "binary"):
            raise ValueError("unsupported discrete treatment ranking method for this protocol")
        if baseline not in labels or any(value not in labels for value in values):
            raise ValueError("requested treatment level was not observed in context")
    if not values or len(values) != len(set(values)) or baseline in values:
        raise ValueError("intervention values must be unique and different from baseline")
    return kind, _label(baseline), [_label(v) for v in values], method


def decide(
    graph: MixedGraph,
    context: pd.DataFrame,
    *,
    target: str,
    interventions: Mapping[str, Intervention | Mapping[str, Any]],
    rows: pd.DataFrame | None = None,
    objective: str = "maximize",
    target_kind: str | None = None,
    target_class: Any = None,
    types: Mapping[str, str] | None = None,
    learners: Mapping[str, Any] | None = None,
    n_folds: int = 5,
    seed: int = 0,
    min_propensity: float = 0.01,
    checkpoint: Callable[[], None] | None = None,
    protocol: str = PROTOCOL,
    learners_by_variable: Mapping[str, Mapping[str, Any]] | None = None,
    validation: pd.DataFrame | None = None,
    validation_design: str | None = None,
    unit_id: str | None = None,
) -> dict[str, Any]:
    """Identify, estimate and rank single-variable contrasts in one target's units.

    ``context`` alone determines fits and defaults. ``rows`` only supplies query
    adjustment values. Population effects refer to context; query effects are
    conditional expectations, not structural individual potential outcomes.
    Discrete targets report class-probability differences and require a ranking
    event (defaults to True/1 only for boolean or 0/1 targets). Unidentifiable or
    unsupported candidates are returned in ``skipped`` with explicit reasons.
    Graph discovery/selection uncertainty is not included in intervals.
    """
    if protocol not in PROTOCOLS:
        raise ValueError("unsupported decision protocol")
    v2 = protocol == "decision-contrasts_v2"
    if not v2 and (validation is not None or learners_by_variable or unit_id):
        raise ValueError("validation and per-variable learners require v2")
    if validation is not None and (
        unit_id is None
        or unit_id not in context
        or unit_id not in validation
        or context[unit_id].isna().any()
        or validation[unit_id].isna().any()
        or not context[unit_id].is_unique
        or not validation[unit_id].is_unique
        or set(context[unit_id]) & set(validation[unit_id])
    ):
        raise ValueError("validation requires unique disjoint unit IDs in context and validation")
    if unit_id is not None:
        if unit_id == target or unit_id in interventions or unit_id not in context:
            raise ValueError("unit_id must be an excluded identifier, not a treatment or target")
        context = context.drop(columns=[unit_id])
        if validation is not None:
            validation = validation.drop(columns=[unit_id])
    if validation is not None:
        validation = validation.copy(deep=True)
        if (
            not validation.columns.is_unique
            or set(validation.columns) != set(context.columns)
            or validation.isna().any().any()
            or len(validation) < 20
            or validation_design != "independent_bernoulli"
        ):
            raise ValueError(
                "validation requires separate complete context-schema rows "
                "and independent_bernoulli design"
            )
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if not all(isinstance(name, str) and name for name in context.columns):
        raise ValueError("context column names must be nonempty strings")
    context = context.copy(deep=True)
    rows = None if rows is None else rows.copy(deep=True)
    if (
        not context.columns.is_unique
        or target not in context
        or len(context) < 2 * n_folds
        or context.isna().any().any()
    ):
        raise ValueError(
            "context needs unique columns, the target, enough rows and no missing cells"
        )
    if set(context.columns) != set(graph.nodes):
        raise ValueError("graph nodes must exactly match context columns")
    if objective not in ("maximize", "minimize"):
        raise ValueError("objective must be maximize or minimize")
    if (
        not interventions
        or target in interventions
        or not set(interventions) <= set(context.columns)
    ):
        raise ValueError("list actionable context variables other than the target")
    if rows is not None and (len(rows) == 0 or not rows.columns.is_unique):
        raise ValueError("query rows need unique columns and at least one row")
    types, learners = dict(types or {}), dict(learners or {})
    for name, raw_spec in interventions.items():
        declared_spec = raw_spec if isinstance(raw_spec, Intervention) else Intervention(**raw_spec)
        if declared_spec.kind is not None:
            if name in types and types[name] != declared_spec.kind:
                raise ValueError("intervention kind conflicts with declared variable type")
            types[name] = declared_spec.kind
    target_kind = target_kind or types.get(target) or _kind(context[target])
    if target_kind not in ("continuous", "binary", "categorical"):
        raise ValueError("target_kind must be continuous, binary or categorical")
    classes = [] if target_kind == "continuous" else _levels(context[target])
    if classes:
        if len(classes) < 2 or (target_kind == "binary" and len(classes) != 2):
            raise ValueError("target has an invalid number of classes")
        if target_class is None and set(classes) == {0, 1}:
            target_class = next(label for label in classes if label == 1)
        if target_class not in classes:
            raise ValueError("target_class must name an observed event to optimize")
    elif target_class is not None:
        raise ValueError("continuous targets do not have target_class")
    binary_event = v2 and target_kind == "binary"
    outcomes = [target_class] if binary_event else (classes or [None])
    # Freeze multiplicity before reading validation outcomes; include every attempted contrast.
    # Per contrast: three candidate gains (plugin, s, dr) and two paired differences
    # against the default, so up to five reported bounds.
    family_size = 5 * sum(
        len(
            _resolved(
                context[name],
                replace(
                    raw if isinstance(raw, Intervention) else Intervention(**raw),
                    kind=(raw.kind if isinstance(raw, Intervention) else raw.get("kind"))
                    or types.get(name),
                ),
                protocol,
            )[2]
        )
        for name, raw in interventions.items()
    )
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    query_candidates: list[list[dict[str, Any]]] = [
        [] for _ in range(0 if rows is None else len(rows))
    ]
    for treatment, raw in interventions.items():
        if checkpoint:
            checkpoint()
        spec = raw if isinstance(raw, Intervention) else Intervention(**raw)
        # Invalid user configurations fail the request, not silently vanish from ranking.
        if spec.kind is not None and treatment in types and spec.kind != types[treatment]:
            raise ValueError("intervention kind conflicts with declared variable type")
        spec = replace(spec, kind=spec.kind or types.get(treatment))
        kind, baseline, values, method = _resolved(context[treatment], spec, protocol, target_kind)
        binary_event = v2 and target_kind == "binary" and kind != "continuous"
        outcomes = [target_class] if binary_event else (classes or [None])
        if validation is not None:
            if (
                not validation[treatment].isin(_levels(context[treatment])).all()
                and kind != "continuous"
            ):
                raise ValueError("validation treatment contains unknown labels")
            if classes and not validation[target].isin(classes).all():
                raise ValueError("validation outcome contains unknown labels")
        slots_for_variable = dict(learners)
        slots_for_variable.update((learners_by_variable or {}).get(treatment, {}))
        if v2 and target_kind == "binary" and kind != "continuous":
            from ._ranking import ProbabilityMean

            supplied = slots_for_variable.get("outcome")
            if supplied is None or hasattr(supplied, "predict_proba"):
                slots_for_variable["outcome"] = ProbabilityMean(supplied)
        if method == "auto" and (
            not binary_event or validation is None or spec.validation_probability is None
        ):
            raise ValueError(
                "auto requires binary outcome, separate randomized validation and known probability"
            )
        if spec.validation_probability is not None and (
            not v2
            or kind != "binary"
            or not np.isfinite(spec.validation_probability)
            or not 0.01 < spec.validation_probability < 0.99
        ):
            raise ValueError(
                "validation probability requires v2 binary treatment and p in (.01,.99)"
            )
        identified = identify_effect(graph, treatment, target)
        if not identified.identifiable or identified.strategy not in ("adjustment", "null"):
            skipped.append(
                {
                    "variable": treatment,
                    "reason": "not_adjustment_identifiable",
                    "detail": str(identified.estimand),
                }
            )
            continue
        if identified.strategy == "adjustment" and graph.kind == GraphKind.DAG:
            baseline_nodes = (
                set(context.columns) - set(graph.descendants(treatment)) - {treatment, target}
            )
            if is_valid_adjustment_set(graph, treatment, target, baseline_nodes):
                identified = replace(
                    identified,
                    adjustment_set=frozenset(baseline_nodes),
                    estimand=_adjustment_formula(
                        identified.treatment, identified.outcome, frozenset(baseline_nodes)
                    ),
                )
        covariates = sorted(identified.adjustment_set or [])
        design = _Design(context, covariates, types)
        x = design.matrix(context)
        q = None if rows is None else design.matrix(rows)
        validation_q = None if validation is None else design.matrix(validation)
        n_query = 0 if q is None else len(q)
        evaluation_q = q
        if validation_q is not None:
            evaluation_q = validation_q if q is None else np.concatenate([q, validation_q])
        fitted: list[Any] = []
        pending: list[dict[str, Any]] = []
        pending_queries: list[list[dict[str, Any]]] = [[] for _ in query_candidates]
        try:
            for event in outcomes:
                if checkpoint:
                    checkpoint()
                y = (
                    context[target].to_numpy(dtype=float)
                    if event is None
                    else (context[target] == event).to_numpy(dtype=float)
                )
                if not np.isfinite(y).all():
                    raise ValueError("target must be finite")
                if identified.strategy == "null":
                    fitted.append(None)
                elif kind != "continuous":
                    labels = _levels(context[treatment])
                    a = np.asarray([labels.index(_label(v)) for v in context[treatment]], float)
                    fitted.append(
                        ArmDR(
                            outcome=slots_for_variable.get("outcome"),
                            propensity=slots_for_variable.get("propensity"),
                            final=slots_for_variable.get("final"),
                            n_folds=n_folds,
                            seed=seed,
                            min_propensity=min_propensity,
                        ).fit(x, a, y)
                    )
                else:
                    # Preserve graph variable names while giving outcome a numeric indicator.
                    frame = context.copy()
                    frame[target] = y
                    declared = {
                        name: (
                            "continuous"
                            if types.get(name, _kind(frame[name])) == "continuous"
                            else "categorical"
                        )
                        for name in map(str, frame.columns)
                    }
                    declared[target] = declared[treatment] = "continuous"
                    data = tabular(frame, types=declared)
                    slots = {
                        key: model
                        for key, model in slots_for_variable.items()
                        if key
                        in (
                            ("outcome", "treatment")
                            if method == "partially_linear"
                            else ("outcome", "final")
                        )
                    }
                    estimator = (
                        DML(slots or None, n_folds=n_folds, random_state=seed)
                        if method == "partially_linear"
                        else ContinuousDR(
                            slots or None,
                            n_folds=n_folds,
                            random_state=seed,
                            bandwidth=spec.bandwidth,
                        )
                    )
                    fitted.append(estimator.fit(identified, data))
            for value in values:
                if checkpoint:
                    checkpoint()
                effects, ses, per_row, diagnostics, score_columns = [], [], [], [], []
                for estimator in fitted:
                    if estimator is None:
                        point, se, individual = (
                            0.0,
                            0.0,
                            None if rows is None else np.zeros(len(rows)),
                        )
                        diagnostic: dict[str, Any] = {"identification": "null"}
                        if v2:
                            diagnostic.update(
                                ranking_method="plugin" if method == "auto" else method,
                                population_method="identified_null",
                                ranking_validation={
                                    "status": "not_assessed",
                                    "reason": "graph_identifies_zero_effect",
                                },
                            )
                    elif kind != "continuous":
                        ranking_method = "plugin" if method == "auto" else method
                        point, se, predictions = estimator.contrast(
                            labels.index(baseline),
                            labels.index(value),
                            evaluation_q,
                            method=ranking_method,
                        )
                        diagnostic = {"minimum_propensity": estimator.minimum_propensity_}
                        if v2:
                            quality = {
                                "status": "not_assessed",
                                "reason": "independent_randomized_validation_required",
                            }
                            if (
                                validation is not None
                                and binary_event
                                and kind == "binary"
                                and spec.validation_probability is not None
                            ):
                                from ._ranking import assess_targeting

                                predictions_by_method = {ranking_method: predictions}
                                if method == "auto":
                                    for alternative in ("s", "dr"):
                                        predictions_by_method[alternative] = estimator.contrast(
                                            labels.index(baseline),
                                            labels.index(value),
                                            evaluation_q,
                                            method=alternative,
                                        )[2]
                                evaluation = assess_targeting(
                                    {k: v[n_query:] for k, v in predictions_by_method.items()},
                                    (validation[treatment] == value).to_numpy(float),
                                    (validation[target] == target_class).to_numpy(float),
                                    probability=spec.validation_probability,
                                    family_size=family_size,
                                    direction=1 if objective == "maximize" else -1,
                                    default=ranking_method,
                                )
                                if method == "auto":
                                    ranking_method = evaluation["selected"]
                                    predictions = predictions_by_method[ranking_method]
                                quality = dict(
                                    status=evaluation["candidates"][ranking_method]["status"],
                                    **evaluation,
                                )
                            scores_for_contrast = (
                                estimator._scores[:, labels.index(value)]
                                - estimator._scores[:, labels.index(baseline)]
                            )
                            diagnostic.update(
                                population_method="dr",
                                ranking_method=ranking_method,
                                ranking_validation=quality,
                                target_event=target_class,
                                direction={"baseline": baseline, "value": value},
                                stages={
                                    "design": {
                                        "n_context": len(x),
                                        "columns": x.shape[1],
                                        "covariates": covariates,
                                    },
                                    "cross_fitted_outcome_means": estimator.outcome_summary_,
                                    "corrected_scores": {
                                        "mean": float(scores_for_contrast.mean()),
                                        "std": float(scores_for_contrast.std()),
                                    },
                                    "query_scores": None
                                    if predictions is None
                                    else {
                                        "std": float(np.std(predictions[:n_query]))
                                        if n_query
                                        else None
                                    },
                                },
                                n_query=n_query,
                                n_validation=0 if validation is None else len(validation),
                                ranking_estimand="conditional_effect_model_not_structural_ite",
                            )
                        individual = None if q is None else predictions[:n_query]
                    else:
                        contrast = estimator.contrast(baseline=baseline, value=value)
                        point, se = float(contrast.value), float(contrast.std_error)
                        diagnostic = dict(contrast.diagnostics)
                        individual = None
                        if rows is not None:
                            query_types = {
                                name: (
                                    "continuous" if name not in design.categories else "categorical"
                                )
                                for name in covariates
                            }
                            query = (
                                tabular(rows[covariates], types=query_types)
                                if covariates
                                else tabular(
                                    {treatment: np.full(len(rows), baseline)},
                                    types={treatment: "continuous"},
                                )
                            )
                            individual = estimator.effects(
                                query, baseline=baseline, value=value
                            ).values
                    if classes:
                        if estimator is None:
                            score_columns.append(np.zeros(len(context)))
                        elif kind != "continuous":
                            score_columns.append(
                                estimator._scores[:, labels.index(value)]
                                - estimator._scores[:, labels.index(baseline)]
                            )
                        elif method == "continuous_dr":
                            score_columns.append(estimator._contrast_scores(baseline, value)[0])
                        else:
                            theta, ry, rt, denom = estimator._cache
                            influence = rt * (ry - theta * rt) / (denom / len(rt))
                            score_columns.append((theta + influence) * (value - baseline))
                    effects.append(point)
                    ses.append(se)
                    per_row.append(np.empty(0) if individual is None else np.asarray(individual))
                    diagnostics.append(diagnostic)
                if binary_event:
                    chosen = classes.index(target_class)
                    other = 1 - chosen
                    point, se, prediction, diagnostic = (
                        effects[0],
                        ses[0],
                        per_row[0],
                        diagnostics[0],
                    )
                    effects, ses, per_row = [0.0, 0.0], [se, se], [prediction, prediction]
                    effects[chosen], effects[other] = point, -point
                    per_row[chosen], per_row[other] = prediction, -prediction
                    diagnostics = [dict(diagnostic), dict(diagnostic)]
                    for diagnostic in diagnostics:
                        diagnostic["binary_complement_derived"] = True
                elif classes:
                    # Known identity: class-probability contrasts sum to zero. Project the
                    # joint scores, not just displayed means; preserve cross-class covariance.
                    joint = np.column_stack(score_columns)
                    joint -= joint.mean(axis=1, keepdims=True)
                    effects = joint.mean(axis=0).tolist()
                    ses = (joint.std(axis=0, ddof=1) / np.sqrt(len(context))).tolist()
                    if rows is not None:
                        query_joint = np.column_stack(per_row)
                        query_joint -= query_joint.mean(axis=1, keepdims=True)
                        per_row = [query_joint[:, j] for j in range(len(classes))]
                    for diagnostic in diagnostics:
                        diagnostic["class_contrast_zero_sum_projection"] = 1.0
                selected = classes.index(target_class) if classes else 0
                item = {
                    "variable": treatment,
                    "baseline": baseline,
                    "value": value,
                    "method": "dr" if v2 and kind != "continuous" else method,
                    "effect": effects[selected],
                    "standard_error": ses[selected],
                    "interval": [
                        effects[selected] - 1.959963984540054 * ses[selected],
                        effects[selected] + 1.959963984540054 * ses[selected],
                    ],
                    "class_effects": [
                        {"label": c, "effect": e} for c, e in zip(classes, effects, strict=False)
                    ],
                    "adjustment_set": covariates,
                    "diagnostics": diagnostics[selected],
                }
                pending.append(item)
                for index, ranking in enumerate(pending_queries):
                    ranking.append(
                        {
                            "variable": treatment,
                            "baseline": baseline,
                            "value": value,
                            "method": diagnostics[selected].get("ranking_method", method),
                            **({"diagnostics": diagnostics[selected]} if v2 else {}),
                            "effect": float(per_row[selected][index]),
                            "class_effects": [
                                {"label": c, "effect": float(v[index])}
                                for c, v in zip(classes, per_row, strict=False)
                            ],
                        }
                    )
            candidates.extend(pending)
            for ranking, additions in zip(query_candidates, pending_queries, strict=True):
                ranking.extend(additions)
        except ValueError as exc:
            # Scientific refusal never turns into a zero effect. Infrastructure errors propagate.
            skipped.append(
                {"variable": treatment, "reason": "estimation_refused", "detail": str(exc)}
            )
    sign = 1 if objective == "maximize" else -1

    def rank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(items, key=lambda item: -sign * item["effect"])
        return [
            dict(item, rank=i + 1, improvement=sign * item["effect"])
            for i, item in enumerate(ordered)
        ]

    return {
        "protocol": protocol,
        "target": target,
        "target_kind": target_kind,
        "target_class": _label(target_class),
        "objective": objective,
        "population": rank(candidates),
        "individual": [
            {"row": i, "interventions": rank(items)} for i, items in enumerate(query_candidates)
        ],
        "skipped": skipped,
        "graph": graph.to_dict(),
        "n_context": len(context),
        "assumptions": [
            "effects conditional on the analysis graph and no unmeasured confounding",
            "query effects are conditional expectations, not structural ITEs",
            "intervals exclude graph selection, simultaneous ranking and smoothing bias",
        ],
        "configuration": {
            "n_folds": n_folds,
            "seed": seed,
            "min_propensity": min_propensity,
            **(
                {
                    "validation_design": validation_design,
                    "validation_family_size": family_size,
                    "unit_id": unit_id,
                }
                if v2
                else {}
            ),
            "interventions": {
                k: {
                    key: value
                    for key, value in (
                        asdict(v) if isinstance(v, Intervention) else dict(v)
                    ).items()
                    if v2 or key != "validation_probability"
                }
                for k, v in interventions.items()
            },
        },
    }
