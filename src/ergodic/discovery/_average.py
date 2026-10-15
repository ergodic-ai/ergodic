# SPDX-License-Identifier: Apache-2.0
"""Effects averaged over a posterior over DAGs (DEC-082 decision 6).

Two questions a single graph cannot answer, read off :class:`GraphSamples`:

- **Is this adjustment set valid?** :func:`adjustment_probability` is the posterior
  mass of the DAGs in which the set satisfies the adjustment criterion for the
  effect (ergodic's complete criterion, :func:`~ergodic.identification.is_valid_adjustment_set`).
- **What is the effect, averaged over the graphs?** :func:`average_effect` asks
  each distinct DAG which adjustment set identifies the effect, groups the DAGs by
  that set, estimates the effect once per set with ergodic's estimators, and
  mixes the estimates by posterior mass.

Three kinds of mass are kept apart. A DAG with no directed path from the
treatment to the outcome has effect exactly 0; that mass enters the mixture at 0
and is reported as ``zero_effect_mass``. A DAG in which no adjustment set over
the data's columns is valid (only possible when the posterior has variables the
data lacks, which are then latent) is ``unidentified_mass``. Sets beyond the
``max_sets`` heaviest are ``truncated_mass``. The mixture is over the identified,
estimated mass only, renormalized, and the result says how much it leaves out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..graph import dag
from ..identification import (
    Identification,
    adjustment_set,
    is_valid_adjustment_set,
    minimal_adjustment_set,
)
from ._posterior import POSTERIOR_NOTE, FeatureProbability, GraphSamples

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from ..data import TabularData
    from ..graph import MixedGraph
    from ..inference import EffectEstimate

DEFAULT_MAX_SETS: Final = 20
_RULES: Final = ("canonical", "minimal")


@dataclass(frozen=True)
class AdjustmentGroup:
    """The DAGs that share one adjustment set, and the effect estimated with it.

    Attributes:
        adjustment_set: The set, or ``None`` for the DAGs with no directed path
            from the treatment to the outcome (effect 0).
        mass: The posterior mass of those DAGs under the method.
        mass_error: Its Monte Carlo standard error (0 when exact).
        n_dags: How many distinct DAGs share the set.
        value: The estimated effect (0 for the no-path group).
        std_error: The estimate's standard error (0 for the no-path group;
            ``None`` when the estimator gives none).
        estimate: The full :class:`~ergodic.inference.EffectEstimate`, or ``None``
            for the no-path group.
        bound: Whether ``mass_error`` is the rule-of-three bound ``3 / draws``
            (every draw agreed), a one-sided 95% bound, not a standard error.
    """

    adjustment_set: frozenset[str] | None
    mass: float
    mass_error: float
    n_dags: int
    value: float
    std_error: float | None
    estimate: EffectEstimate | None = field(default=None, repr=False)
    bound: bool = False


@dataclass(frozen=True, eq=False)
class AveragedEffect:
    """An effect averaged over a posterior over DAGs; see :func:`average_effect`.

    Attributes:
        treatment: The treatment variable.
        outcome: The outcome variable.
        groups: One :class:`AdjustmentGroup` per estimated set (and the no-path
            group when it has mass), heaviest first.
        mean: The mixture mean over the identified, estimated mass.
        mean_mc_error: The mean's Monte Carlo standard error from the posterior masses
            alone (the estimates held fixed; their own sampling error is in
            ``within_variance``); 0 when the masses are exact.
        variance: The mixture variance, ``within + between``.
        within_variance: The mass-weighted mean of the squared standard errors.
        between_variance: The mass-weighted variance of the group estimates.
        identified_mass: The mass the mixture covers (including the zero-effect mass).
        zero_effect_mass: The mass of DAGs with no directed path (effect 0).
        unidentified_mass: The mass of DAGs where no adjustment set over the data
            is valid; left out of the mixture.
        truncated_mass: The mass of identified sets beyond ``max_sets``; left out.
        exact: Whether the masses are exact (every DAG enumerated).
        diagnostics: How it was computed.
    """

    treatment: str
    outcome: str
    groups: tuple[AdjustmentGroup, ...] = field(repr=False)
    mean: float
    mean_mc_error: float
    variance: float | None
    within_variance: float | None
    between_variance: float
    identified_mass: float
    zero_effect_mass: float
    unidentified_mass: float
    truncated_mass: float
    exact: bool
    diagnostics: dict[str, Any] = field(repr=False)

    @property
    def std(self) -> float | None:
        """The mixture's standard deviation, ``sqrt(variance)``."""
        return None if self.variance is None else math.sqrt(self.variance)

    def to_frame(self) -> pd.DataFrame:
        """The groups as a dataframe, one row per group.

        Columns: the set, its mass, the mass's error, whether that error is the
        rule-of-three bound, the DAG count, the estimate and its standard error.

        The set reads ``"(empty)"`` for no adjustment and ``"(no directed path)"``
        for the zero-effect group.
        """
        import pandas as pd

        rows = [
            {
                "adjustment_set": "(no directed path)"
                if g.adjustment_set is None
                else ", ".join(sorted(g.adjustment_set)) or "(empty)",
                "mass": g.mass,
                "mass_error": g.mass_error,
                "bound": g.bound,
                "n_dags": g.n_dags,
                "estimate": g.value,
                "std_error": g.std_error,
            }
            for g in self.groups
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "adjustment_set",
                "mass",
                "mass_error",
                "bound",
                "n_dags",
                "estimate",
                "std_error",
            ],
        )

    def __repr__(self) -> str:
        spread = "" if self.std is None else f" (sd {self.std:.4g})"
        left = max(0.0, 1.0 - self.identified_mass)
        left = 0.0 if left < 1e-12 else left
        return (
            f"AveragedEffect({self.treatment} -> {self.outcome}: {self.mean:.4g}{spread} "
            f"over {self.identified_mass:.3g} of the mass, {len(self.groups)} groups, "
            f"{left:.3g} left out)"
        )


def adjustment_probability(
    samples: GraphSamples,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    adjustment: str | Iterable[str],
) -> FeatureProbability:
    """The posterior probability that ``adjustment`` is a valid adjustment set.

    Each distinct DAG is checked with the complete adjustment criterion
    (:func:`~ergodic.identification.is_valid_adjustment_set`), and the masses of
    the DAGs where it holds are summed. Exact (``mc_error`` 0) when every DAG was
    enumerated; otherwise an estimate from the draws (exact draws or MCMC) with
    its Monte Carlo error. A posterior probability under the method. On
    observational data these probabilities typically stay well below 1, because
    only the Markov equivalence class is identified; in the registered B4 run
    (simulated linear-Gaussian graphs, the true parents of the treatment under the
    BGe posterior) the medians were 0.37 at n = 200 and 0.48 at n = 1,000.

    Args:
        samples: A posterior from :func:`graph_posterior`.
        treatment: The treatment variable(s).
        outcome: The outcome variable(s).
        adjustment: The candidate set, a name or names (empty for no adjustment).

    Raises:
        ValueError: A name is not one of the posterior's variables, or the
            treatment, outcome and set overlap.
    """
    if not isinstance(samples, GraphSamples):
        raise TypeError("adjustment_probability reads GraphSamples from graph_posterior")
    x, y = _names(treatment), _names(outcome)
    z = frozenset([adjustment]) if isinstance(adjustment, str) else frozenset(adjustment)
    _check_names(samples.variables, x | y | z)
    if (x & y) or (z & (x | y)):
        raise ValueError("the treatment, the outcome and the adjustment set must not overlap")
    values = np.asarray(
        [
            is_valid_adjustment_set(samples.graph(i), x, y, z)
            for i in range(samples.adjacency.shape[0])
        ],
        dtype=np.float64,
    )
    return samples._estimate(values)


def average_effect(
    samples: GraphSamples,
    data: TabularData,
    treatment: str,
    outcome: str,
    *,
    rule: str = "canonical",
    strategy: str | None = None,
    learner: Any = None,
    max_sets: int = DEFAULT_MAX_SETS,
    seed: int = 0,
    n_boot: int = 1000,
    **kwargs: Any,
) -> AveragedEffect:
    """The effect of ``treatment`` on ``outcome``, averaged over a posterior over DAGs.

    For each distinct DAG: no directed path from the treatment to the outcome
    means the effect is 0; otherwise ``rule`` picks an adjustment set over the
    data's columns (``"canonical"``, the set :func:`~ergodic.identify_effect`
    uses, or ``"minimal"``). A variable of the posterior that ``data`` lacks is
    treated as latent, so a DAG can have no valid set; its mass is
    ``unidentified_mass``. The DAGs are grouped by set, and the ``max_sets``
    heaviest sets are each estimated once with :func:`~ergodic.estimate`
    (``strategy`` and ``learner`` as there; the default is the family default,
    the doubly-robust AIPW, which needs a binary treatment; use
    ``strategy="dml"`` for a continuous one), seeded with ``seed``.

    The result mixes the group estimates by posterior mass, renormalized over the
    mass it covers: the zero-effect mass (at 0, standard error 0) plus the
    estimated sets. Its variance is the law of total variance, the mass-weighted
    mean of the squared standard errors (within) plus the mass-weighted variance
    of the estimates (between), reading each estimate as a normal around its value.
    Unidentified and truncated mass are left out and reported, never averaged in
    silently. Masses are posterior probabilities under the method, exact when every
    DAG was enumerated and otherwise from the draws, each with its Monte Carlo error.

    Args:
        samples: A posterior from :func:`graph_posterior`.
        data: The data to estimate on; it must hold the treatment and the outcome.
        treatment: The treatment variable.
        outcome: The outcome variable.
        rule: ``"canonical"`` or ``"minimal"``.
        strategy: The estimator, as in :func:`~ergodic.estimate`.
        learner: The learners, as in :func:`~ergodic.estimate`.
        max_sets: The most adjustment sets to estimate, heaviest first.
        seed: The estimators' ``random_state``.
        n_boot: Bootstrap draws for estimators that bootstrap.
        **kwargs: Passed to the estimator (``treated``, ...).

    Returns:
        An :class:`AveragedEffect`.

    Raises:
        TypeError: ``samples`` or ``data`` has the wrong type.
        ValueError: Unknown variables, a bad ``rule`` or ``max_sets``, or no mass
            left to average.
        NotImplementedError: From the estimator: the default (AIPW) and the other
            binary-treatment estimators refuse a treatment that is not binary; pass
            ``strategy="dml"`` (or ``"pliv"``) for a continuous one.
    """
    from ..data import TabularData
    from ..inference import estimate

    if not isinstance(samples, GraphSamples):
        raise TypeError("average_effect reads GraphSamples from graph_posterior")
    if not isinstance(data, TabularData):
        raise TypeError(f"data must be a TabularData, got {type(data).__name__}")
    if rule not in _RULES:
        raise ValueError(f"rule must be one of {list(_RULES)}, got {rule!r}")
    if type(max_sets) is not int or max_sets < 1:
        raise ValueError(f"max_sets must be a positive int, got {max_sets!r}")
    if not isinstance(treatment, str) or not isinstance(outcome, str) or treatment == outcome:
        raise ValueError("treatment and outcome must be two different variable names")
    names = samples.variables
    _check_names(names, {treatment, outcome})
    columns = set(data.variables)
    missing = {treatment, outcome} - columns
    if missing:
        raise ValueError(f"data lacks the treatment or outcome: {sorted(missing)}")
    latent = tuple(name for name in names if name not in columns)
    picker = adjustment_set if rule == "canonical" else minimal_adjustment_set
    keys: list[frozenset[str] | str] = []
    for index in range(samples.adjacency.shape[0]):
        graph = _with_latent(samples, index, latent)
        if treatment not in graph.ancestors(outcome):
            keys.append("no_path")
            continue
        chosen = picker(graph, treatment, outcome)
        keys.append("unidentified" if chosen is None else chosen)
    distinct: dict[frozenset[str] | str, list[int]] = {}
    for index, key in enumerate(keys):
        distinct.setdefault(key, []).append(index)

    def mass_of(members: list[int]) -> FeatureProbability:
        values = np.zeros(samples.adjacency.shape[0])
        values[members] = 1.0
        return samples._estimate(values)

    zero = mass_of(distinct.get("no_path", []))
    unidentified = mass_of(distinct.get("unidentified", []))
    sets = [
        (key, mass_of(members), len(members))
        for key, members in distinct.items()
        if isinstance(key, frozenset)
    ]
    sets.sort(key=lambda item: (-item[1].probability, sorted(item[0])))
    kept, dropped = sets[:max_sets], sets[max_sets:]
    truncated = float(sum(item[1].probability for item in dropped))
    groups: list[AdjustmentGroup] = []
    if zero.probability > 0.0:
        groups.append(
            AdjustmentGroup(
                None,
                zero.probability,
                zero.mc_error,
                len(distinct["no_path"]),
                0.0,
                0.0,
                bound=zero.bound,
            )
        )
    x, y = frozenset([treatment]), frozenset([outcome])
    for key, mass, count in kept:
        identification = Identification(
            x,
            y,
            True,
            "adjustment",
            adjustment_set=key,
            estimand=f"adjust for {{{', '.join(sorted(key))}}}",
        )
        effect = estimate(
            identification,
            data,
            strategy=strategy,
            learner=learner,
            random_state=seed,
            n_boot=n_boot,
            **kwargs,
        )
        groups.append(
            AdjustmentGroup(
                key,
                mass.probability,
                mass.mc_error,
                count,
                effect.value,
                effect.std_error,
                effect,
                bound=mass.bound,
            )
        )
    groups.sort(key=lambda g: -g.mass)
    covered = float(sum(g.mass for g in groups))
    if covered <= 0.0:
        raise ValueError(
            "no posterior mass is identified by adjustment over the data's columns, so "
            "there is nothing to average"
        )
    weights = np.asarray([g.mass for g in groups]) / covered
    values = np.asarray([g.value for g in groups])
    mean = float(weights @ values)
    # The mean is a ratio of two posterior expectations; linearized, its Monte Carlo
    # error is that of the per-DAG value (effect - mean) * [covered] / covered mass.
    per_dag = np.zeros(samples.adjacency.shape[0])
    for group in groups:
        members = distinct["no_path" if group.adjustment_set is None else group.adjustment_set]
        per_dag[members] = (group.value - mean) / covered
    mean_error = samples._estimate(per_dag, indicator=False).mc_error
    between = float(weights @ (values - mean) ** 2)
    errors = [g.std_error for g in groups]
    within: float | None
    if any(e is None for e in errors):
        within = None
    else:
        within = float(weights @ np.asarray(errors, dtype=np.float64) ** 2)
    diagnostics: dict[str, Any] = {
        "rule": rule,
        "strategy": strategy,
        "seed": seed,
        "max_sets": max_sets,
        "n_sets": len(sets),
        "n_estimated": len(kept),
        "latent": latent,
        "posterior_method": samples.method,
        "masses_exact": samples.exhaustive,
        "zero_effect_mass_error": zero.mc_error,
        "unidentified_mass_error": unidentified.mc_error,
        "zero_effect_mass_error_is_bound": zero.bound,
        "unidentified_mass_error_is_bound": unidentified.bound,
        "note": POSTERIOR_NOTE,
    }
    if within is None:
        diagnostics["variance_note"] = "an estimator gave no standard error, so no variance"
    return AveragedEffect(
        treatment=treatment,
        outcome=outcome,
        groups=tuple(groups),
        mean=mean,
        mean_mc_error=mean_error,
        variance=None if within is None else within + between,
        within_variance=within,
        between_variance=between,
        identified_mass=covered,
        zero_effect_mass=zero.probability,
        unidentified_mass=unidentified.probability,
        truncated_mass=truncated,
        exact=samples.exhaustive,
        diagnostics=diagnostics,
    )


def _names(value: str | Iterable[str]) -> frozenset[str]:
    return frozenset([value]) if isinstance(value, str) else frozenset(value)


def _check_names(variables: tuple[str, ...], names: Iterable[str]) -> None:
    unknown = sorted(set(names) - set(variables))
    if unknown:
        raise ValueError(f"not variables of the posterior: {unknown}")


def _with_latent(samples: GraphSamples, index: int, latent: tuple[str, ...]) -> MixedGraph:
    if not latent:
        return samples.graph(index)
    names = samples.variables
    matrix = samples.adjacency[index]
    sources, targets = np.nonzero(matrix)
    return dag(
        [(names[int(s)], names[int(t)]) for s, t in zip(sources, targets, strict=True)],
        nodes=names,
        latent=latent,
    )
