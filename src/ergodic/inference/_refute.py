# SPDX-License-Identifier: Apache-2.0
"""Refutation: stress-test an estimate by re-running it under a perturbation.

An estimate is only as good as the assumptions behind it, and a single number on
its own tells you nothing about how fragile it is. These checks are the cheap,
estimator-agnostic ones. Scramble the treatment and the effect should vanish. Add
an irrelevant covariate and the effect should not move. Drop a random slice of the
rows and the effect should hold. None of them proves the estimate is right, but a
failure is a real warning, and they cost nothing beyond re-fitting.

This is the model-agnostic layer. Sensitivity to a specific unobserved confounder
(the Cinelli-Hazlett partial-R-squared, the E-value) is a separate, sharper tool
that is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data import Ordinal, tabular
from ._estimate import estimate_effect

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from ..data import TabularData
    from ..graph import MixedGraph

_METHODS = ("placebo_treatment", "random_common_cause", "data_subset")


@dataclass(frozen=True)
class Refutation:
    """The result of one refutation check.

    ``estimand_value`` is the effect on the real data. ``refuted_value`` is the
    mean effect over the perturbed runs, with ``refuted_std`` their spread.
    ``passed`` is the convenience verdict (the rule is in the method's section of
    :func:`refute_effect`); read the numbers, not just the flag. ``p_value`` is
    the permutation tail for the placebo check and ``None`` for the others.
    """

    method: str
    estimand_value: float
    refuted_value: float
    refuted_std: float
    p_value: float | None
    passed: bool
    n_simulations: int
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        verdict = "passed" if self.passed else "FAILED"
        body = (
            f"{self.method}: estimate={self.estimand_value:.4g}, "
            f"under perturbation={self.refuted_value:.4g}"
        )
        if self.p_value is not None:
            body += f", p={self.p_value:.3g}"
        return f"Refutation({body}, {verdict}, n={self.n_simulations})"


def _rebuilt(template: TabularData, frame: pd.DataFrame) -> TabularData:
    """Rebuild a tabular dataset from a modified frame, keeping the column types."""
    types: dict[str, Any] = {}
    exclude: list[str] = []
    for column in template.schema:
        if column.name not in frame.columns:
            continue
        if column.is_excluded:
            exclude.append(column.name)
        elif column.is_ordinal and column.categories is not None:
            types[column.name] = Ordinal(list(column.categories))
        else:
            types[column.name] = column.kind.value
    return tabular(frame, types=types, exclude=exclude or None)


def refute_effect(
    graph: MixedGraph,
    data: TabularData,
    treatment: str,
    outcome: str,
    *,
    method: str = "placebo_treatment",
    strategy: str | None = None,
    learner: Any = None,
    treated: Any | None = None,
    estimand_value: float | None = None,
    n_simulations: int = 100,
    subset_fraction: float = 0.8,
    random_state: int | None = None,
    **kwargs: Any,
) -> Refutation:
    """Stress-test the effect of ``treatment`` on ``outcome`` under a perturbation.

    Runs the same identify-then-estimate path as :func:`estimate_effect`, so pass
    the same ``strategy``, ``learner``, and ``treated`` you used for the estimate.
    ``estimand_value`` skips re-computing the effect on the real data when you
    already have it.

    The three methods:

    - ``"placebo_treatment"`` shuffles the treatment column, breaking its link to
      everything, and re-estimates. The placebo effect should sit near zero.
      ``passed`` when the mean placebo effect is under 20 percent of the real one;
      ``p_value`` is the share of placebo runs whose effect matches or beats the
      real one in size.
    - ``"random_common_cause"`` adds an independent random covariate as a parent of
      both treatment and outcome and re-estimates. The effect should not move.
      ``passed`` when the mean stays within 10 percent of the original.
    - ``"data_subset"`` re-estimates on a random ``subset_fraction`` of the rows.
      The effect should hold. ``passed`` when the original sits within two standard
      deviations of the subset spread.

    Flexible or cross-fit estimators make each run a full fit, so lower
    ``n_simulations`` (or pick a faster ``strategy``) when the estimator is heavy.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown refutation method {method!r}; use one of {', '.join(_METHODS)}")
    if n_simulations < 2:
        raise ValueError(f"n_simulations must be at least 2, got {n_simulations}")
    rng = np.random.default_rng(random_state)

    def run(g: MixedGraph, d: TabularData) -> float:
        return estimate_effect(
            g,
            d,
            treatment,
            outcome,
            strategy=strategy,
            learner=learner,
            treated=treated,
            random_state=random_state,
            **kwargs,
        ).value

    if estimand_value is None:
        estimand_value = run(graph, data)

    if method == "placebo_treatment":
        base = data.to_pandas()
        column = base[treatment].to_numpy()
        effects = np.array(
            [
                run(graph, _rebuilt(data, base.assign(**{treatment: rng.permutation(column)})))
                for _ in range(n_simulations)
            ]
        )
        refuted_value = float(effects.mean())
        refuted_std = float(effects.std(ddof=1))
        p_value: float | None = float(
            (1 + np.sum(np.abs(effects) >= abs(estimand_value))) / (1 + n_simulations)
        )
        passed = abs(refuted_value) <= 0.2 * abs(estimand_value)

    elif method == "random_common_cause":
        noise = "_random_common_cause"
        if noise in data.names:
            raise ValueError(f"the data already has a column named {noise!r}")
        augmented = (
            graph.add_edge(f"{noise} -> {treatment}")
            .add_edge(f"{noise} -> {outcome}")
            .as_kind(graph.kind)
        )
        base = data.to_pandas()
        effects = np.array(
            [
                run(
                    augmented,
                    _rebuilt(data, base.assign(**{noise: rng.standard_normal(len(base))})),
                )
                for _ in range(n_simulations)
            ]
        )
        refuted_value = float(effects.mean())
        refuted_std = float(effects.std(ddof=1))
        p_value = None
        passed = abs(refuted_value - estimand_value) <= 0.1 * abs(estimand_value) + 1e-9

    else:  # data_subset
        n = data.n_rows
        size = max(2, round(subset_fraction * n))
        effects = np.array(
            [
                run(graph, data.take(np.sort(rng.choice(n, size=size, replace=False))))
                for _ in range(n_simulations)
            ]
        )
        refuted_value = float(effects.mean())
        refuted_std = float(effects.std(ddof=1))
        p_value = None
        passed = abs(estimand_value - refuted_value) <= 2.0 * refuted_std + 1e-9

    return Refutation(
        method=method,
        estimand_value=float(estimand_value),
        refuted_value=refuted_value,
        refuted_std=refuted_std,
        p_value=p_value,
        passed=passed,
        n_simulations=n_simulations,
        diagnostics={"effect_min": float(effects.min()), "effect_max": float(effects.max())},
    )


__all__ = ["Refutation", "refute_effect"]
