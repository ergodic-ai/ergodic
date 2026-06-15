# SPDX-License-Identifier: Apache-2.0
"""The discrete plug-in estimator: evaluate an ID functional on the empirical joint.

A general do-calculus functional (``Identification.strategy == "id"``) has no
named estimator, but on discrete data it needs none: the functional is a
sum-product of conditional frequencies, so estimate every term by its empirical
count and evaluate the tree. This is exact in the population and consistent on a
sample, the same reference path the soundness-against-truth test exercises.

The estimator builds the empirical joint ``P(V)`` over every variable the
functional reads (the treatment, the outcome, and any adjustment or mediator
variables it sums over, including the ones bound by a marginal), evaluates the
functional with the shared :mod:`ergodic.identification._evaluate` engine, reads
off ``P(Y | do(X=1))`` and ``P(Y | do(X=0))``, and contrasts them into the ATE. A
bootstrap over the rows gives the standard error.

It is honest about its limits. A continuous column raises (the SCM path handles a
continuous functional). A non-binary treatment raises (the ATE is a two-level
contrast). A zero-count conditioning cell raises with the offending assignment,
the discrete analog of the overlap check, rather than dividing by zero or
smoothing in silence.

numpy only, no new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...identification._evaluate import Factor, evaluate, marginalize
from ...identification._functional import Marginal, Prob, Product, Ratio
from .._result import EffectEstimate
from .._target import EffectTarget, as_target

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...data import TabularData
    from ...identification import Identification
    from ...identification._functional import Functional


_PLUGIN_TARGETS = frozenset({EffectTarget.ATE})


@dataclass(frozen=True)
class _Problem:
    """Everything the contrast and the bootstrap need, resolved once from the inputs."""

    functional: Functional
    scope: tuple[str, ...]
    treatment: str
    outcome: str
    domains: dict[str, int]
    levels: npt.NDArray[np.float64]


def plugin_estimate(
    identification: Identification,
    data: TabularData,
    *,
    target: EffectTarget | str | None = None,
    treated: Any | None = None,
    random_state: int | None = None,
    n_boot: int = 1000,
) -> EffectEstimate:
    """Estimate a general ID functional on discrete data, returning an ATE.

    ``identification`` must carry a :class:`~ergodic.identification.Functional`
    (``strategy == "id"``). ``treated`` names the treatment level that codes to 1
    for the ``do(X=1) - do(X=0)`` contrast; without it the codes' level order
    decides. The result is distribution-shaped with a bootstrap standard error,
    reusing :class:`~ergodic.inference.EffectEstimate`.
    """
    chosen = EffectTarget.ATE if target is None else as_target(target)
    if chosen not in _PLUGIN_TARGETS:
        raise ValueError(f"the plug-in estimator reports ATE, not {chosen}")
    functional = identification.functional
    if functional is None:  # pragma: no cover - guarded by the dispatch
        raise ValueError("the plug-in estimator needs an ID functional; none is set")

    treatment = _one(identification.treatment, "treatment")
    outcome = _one(identification.outcome, "outcome")
    scope = tuple(sorted(_referenced(functional) | {treatment, outcome}))
    _check_discrete(data, scope)
    _check_binary_treatment(data, treatment)
    problem = _Problem(
        functional=functional,
        scope=scope,
        treatment=treatment,
        outcome=outcome,
        domains={name: _cardinality(data, name) for name in scope},
        levels=_outcome_levels(data, outcome),
    )
    coded = _codes(data, scope, treatment, treated)

    point = _ate(coded, data.n_rows, problem)
    rng = np.random.default_rng(random_state)
    draws = _bootstrap(coded, data.n_rows, problem, rng, n_boot)
    diagnostics = {
        "n_levels_outcome": float(len(problem.levels)),
        "n_boot_requested": float(n_boot),
        "n_boot_returned": float(draws.shape[0]),
    }
    return EffectEstimate(
        target=chosen,
        estimator="plug_in",
        learners={},
        n=data.n_rows,
        kind="bootstrap",
        diagnostics=diagnostics,
        _point=point,
        _draws=draws,
    )


# -- roles and guards --------------------------------------------------------


def _one(names: frozenset[str], role: str) -> str:
    if len(names) != 1:
        raise NotImplementedError(
            f"the plug-in estimator supports a single {role}; got {sorted(names)}"
        )
    return next(iter(names))


def _referenced(functional: Functional) -> frozenset[str]:
    """Every variable named anywhere in the functional, bound indices included.

    ``free_vars`` drops the marginalized variables (the mediators, the adjustment
    set), but the empirical joint must span them too, since the inner terms read
    them. So this collects the full variable scope: the heads and givens of every
    atom, plus the indices the marginals sum over.
    """
    if isinstance(functional, Prob):
        return functional.head | functional.given
    if isinstance(functional, Product):
        out: frozenset[str] = frozenset()
        for factor in functional.factors:
            out |= _referenced(factor)
        return out
    if isinstance(functional, Marginal):
        return _referenced(functional.body) | functional.index
    if isinstance(functional, Ratio):
        return _referenced(functional.num) | _referenced(functional.den)
    raise TypeError(f"not a Functional: {functional!r}")  # pragma: no cover


def _check_discrete(data: TabularData, free: tuple[str, ...]) -> None:
    """Every free variable must be discrete; a continuous one points at the SCM."""
    for name in free:
        if name not in data.schema:
            raise ValueError(f"column {name!r} from the functional is not in the data")
        col = data.schema[name]
        if not col.is_discrete:
            raise NotImplementedError(
                f"the discrete plug-in needs categorical variables; {name!r} is "
                f"{col.kind.value}. A continuous functional is estimated by the SCM's "
                "fitted mechanisms (its own pillar), not the plug-in"
            )


def _check_binary_treatment(data: TabularData, treatment: str) -> None:
    if data.schema[treatment].cardinality != 2:
        raise ValueError(
            f"the ATE is a two-level contrast; treatment {treatment!r} has "
            f"{data.schema[treatment].cardinality} levels"
        )


def _cardinality(data: TabularData, name: str) -> int:
    card = data.schema[name].cardinality
    assert card is not None  # discreteness already checked
    return card


def _outcome_levels(data: TabularData, outcome: str) -> npt.NDArray[np.float64]:
    """The numeric value of each outcome code, for ``E[Y | do(x)] = sum_k v_k P(k)``.

    A binary 0/1 outcome reduces to ``P(Y=1 | do(x))``. A multi-level discrete
    numeric outcome weights each level's probability by the level's value. Codes
    are 0..K-1 in category order; the value at code ``k`` is the ``k``-th
    category, which must be numeric.
    """
    categories = data.schema[outcome].categories or ()
    try:
        return np.asarray([float(c) for c in categories], dtype=np.float64)
    except (TypeError, ValueError):
        raise NotImplementedError(
            f"the plug-in reports a numeric ATE; outcome {outcome!r} has non-numeric "
            f"levels {list(categories)}. Recode it to numbers or pick a binary outcome"
        ) from None


# -- the empirical joint -----------------------------------------------------


def _codes(
    data: TabularData, free: tuple[str, ...], treatment: str, treated: Any | None
) -> dict[str, npt.NDArray[np.int64]]:
    """Integer codes per free variable, the treatment honoring ``treated=``.

    Every variable is coded 0..K-1. For the treatment, ``treated=`` forces the
    named level to code 1 (the rest to 0), so the ``do(X=1)`` arm is that level;
    without it the schema's level order decides, matching ``data.codes``.
    """
    out: dict[str, npt.NDArray[np.int64]] = {}
    for name in free:
        if name == treatment and treated is not None:
            out[name] = _coded_to_treated(data, name, treated)
        else:
            out[name] = data.codes(name)
    return out


def _coded_to_treated(data: TabularData, name: str, treated: Any) -> npt.NDArray[np.int64]:
    """Code ``name`` to 1 where it equals ``treated`` and 0 otherwise (a binary level)."""
    categories = tuple(data.schema[name].categories or ())
    if treated not in categories:
        raise ValueError(
            f"treated={treated!r} is not a level of {name!r}; observed levels are "
            f"{list(categories)}"
        )
    raw = data.column(name)
    return np.asarray(raw == treated, dtype=np.int64)


def _empirical_joint(
    coded: dict[str, npt.NDArray[np.int64]],
    n: int,
    free: tuple[str, ...],
    domains: dict[str, int],
) -> Factor:
    """The empirical ``P(free)``: normalized co-occurrence counts of the codes."""
    shape = tuple(domains[name] for name in free)
    counts = np.zeros(shape, dtype=np.float64)
    index = tuple(coded[name] for name in free)
    np.add.at(counts, index, 1.0)
    return Factor(free, counts / float(n))


# -- the contrast ------------------------------------------------------------


def _ate(coded: dict[str, npt.NDArray[np.int64]], n: int, problem: _Problem) -> float:
    """``E[Y | do(X=1)] - E[Y | do(X=0)]`` from the empirical joint."""
    joint = _empirical_joint(coded, n, problem.scope, problem.domains)
    _check_positivity(problem, joint)
    effect = evaluate(problem.functional, joint, problem.domains)
    e1 = _mean_outcome(effect, problem.outcome, problem.treatment, 1, problem.levels)
    e0 = _mean_outcome(effect, problem.outcome, problem.treatment, 0, problem.levels)
    return e1 - e0


def _mean_outcome(
    effect: Factor,
    outcome: str,
    treatment: str,
    x_level: int,
    levels: npt.NDArray[np.float64],
) -> float:
    """``E[Y | do(X=x_level)]`` read off the evaluated ``P(Y | do(X))`` factor.

    The factor is ``P(Y | do(X))`` over the outcome, the treatment, and possibly
    extra variables it is provably constant in. Slice the treatment to
    ``x_level``, fix every extra variable to 0 (the factor does not depend on
    them, asserted), then weight the outcome distribution by the level values.
    """
    sliced = _slice(effect, treatment, x_level)
    extra = [a for a in sliced.axes if a != outcome]
    if extra:
        _assert_constant(sliced, outcome)
        for axis in extra:
            sliced = _slice(sliced, axis, 0)
    if sliced.axes != (outcome,):  # pragma: no cover - shape invariant
        raise AssertionError(f"expected a factor over {outcome!r}, got {sliced.axes}")
    return float(np.dot(sliced.values, levels))


def _slice(factor: Factor, axis: str, level: int) -> Factor:
    """Fix ``axis`` to ``level``, dropping it from the factor."""
    pos = factor.axes.index(axis)
    values = np.take(factor.values, level, axis=pos)
    remaining = tuple(a for a in factor.axes if a != axis)
    return Factor(remaining, values)


def _assert_constant(factor: Factor, outcome: str) -> None:
    """Assert the factor does not vary across its non-outcome axes.

    ``P(Y | do(X))`` evaluated at a fixed treatment must be constant in any extra
    free variable the algorithm folded in. If it is not, the functional or the
    join is wrong and fixing the extra to 0 would silently bias the answer, so we
    raise loudly instead.
    """
    out_pos = factor.axes.index(outcome)
    moved = np.moveaxis(factor.values, out_pos, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    if flat.shape[0] > 1 and not np.allclose(flat, flat[0], atol=1e-9):
        raise AssertionError(
            f"P({outcome} | do) is not constant in the extra variables "
            f"{[a for a in factor.axes if a != outcome]}; the functional is malformed"
        )


# -- positivity --------------------------------------------------------------


def _check_positivity(problem: _Problem, joint: Factor) -> None:
    """Raise on an empty stratum that the functional genuinely divides by.

    Two checks, the discrete analog of the overlap condition.

    First, both treatment arms must be populated: a contrast ``do(X=1) -
    do(X=0)`` is undefined without rows at each level, so ``P(X=0)`` and
    ``P(X=1)`` must both be positive (the same empty-arm guard the adjustment
    estimators carry).

    Second, every conditional ``P(head | given)`` whose ``given`` mentions the
    treatment must be defined at each cell it is read with positive weight. The
    cell ``given=g`` is read with positive weight exactly when its non-treatment
    part has support, ``P(given \\ {X}) > 0`` there (an outer ``sum`` keeps it),
    yet the stratum itself is empty, ``P(given=g)=0``: an empty treated-or-control
    stratum inside a populated covariate cell. That is the adjustment/front-door
    positivity failure, and the one the evaluator's silent zero-fill would bias.
    A ``given`` cell empty only because its whole covariate slice is empty carries
    no weight and is not flagged. The walk recurses into the subterms.

    The guard is scoped to the treatment-conditioning terms, which covers the
    adjustment and front-door forms shipped today. A general nested ID functional
    can in principle divide by a pure-covariate marginal that goes to zero with
    positive weight; that cell is not flagged here and the evaluator zero-fills
    it. No such case is reachable in the current forms, so the guard is
    best-effort overlap, not a proof of it.
    """
    _check_arms(problem, joint)
    _walk_conditionals(problem.functional, problem.treatment, joint)


def _check_arms(problem: _Problem, joint: Factor) -> None:
    """Both levels of the binary treatment must appear."""
    margin = marginalize(joint, frozenset(joint.axes) - {problem.treatment})
    for level in (0, 1):
        if margin.values[level] == 0.0:
            raise ValueError(
                f"positivity violated: treatment {problem.treatment!r} has no rows at "
                f"level {level}, so the do(X={level}) arm of the contrast is empty"
            )


def _walk_conditionals(functional: Functional, treatment: str, joint: Factor) -> None:
    if isinstance(functional, Prob):
        if treatment in functional.given:
            _check_stratum(functional, treatment, joint)
    elif isinstance(functional, Product):
        for factor in functional.factors:
            _walk_conditionals(factor, treatment, joint)
    elif isinstance(functional, Marginal):
        _walk_conditionals(functional.body, treatment, joint)
    elif isinstance(functional, Ratio):
        _walk_conditionals(functional.num, treatment, joint)
        _walk_conditionals(functional.den, treatment, joint)


def _check_stratum(term: Prob, treatment: str, joint: Factor) -> None:
    """For a treatment-conditioning term, flag an empty stratum inside a populated cell."""
    given = term.given
    covariates = given - {treatment}
    p_given = marginalize(joint, frozenset(joint.axes) - given)
    if covariates:
        p_cov = marginalize(joint, frozenset(joint.axes) - covariates)
    else:
        # no covariates: the "cell" is the whole sample, populated by construction
        p_cov = Factor((), np.array(1.0))
    for cell in np.argwhere(p_given.values == 0.0):
        assignment = {axis: int(cell[i]) for i, axis in enumerate(p_given.axes)}
        cov_cell = tuple(assignment[a] for a in p_cov.axes)
        if float(p_cov.values[cov_cell] if p_cov.axes else p_cov.values) > 0.0:
            raise ValueError(
                f"positivity violated: the stratum {assignment} has no rows while its "
                f"covariate cell is populated, so {term.render()} is undefined there. The "
                "discrete plug-in does not smooth an empty stratum; cover it, drop the "
                "variable, or coarsen it"
            )


# -- bootstrap ---------------------------------------------------------------


def _bootstrap(
    coded: dict[str, npt.NDArray[np.int64]],
    n: int,
    problem: _Problem,
    rng: np.random.Generator,
    n_boot: int,
) -> npt.NDArray[np.float64]:
    """Resample rows, rebuild the empirical joint, re-evaluate the ATE."""
    draws = np.empty(n_boot, dtype=np.float64)
    filled = 0
    attempts = 0
    budget = n_boot * 5
    while filled < n_boot and attempts < budget:
        attempts += 1
        idx = rng.integers(0, n, n)
        sub = {name: arr[idx] for name, arr in coded.items()}
        try:
            value = _ate(sub, n, problem)
        except (ValueError, ZeroDivisionError, AssertionError):
            continue
        if not np.isfinite(value):
            continue
        draws[filled] = value
        filled += 1
    if filled == 0:  # pragma: no cover - a resample of a positive sample rarely all-fails
        raise RuntimeError(
            f"the plug-in bootstrap produced no finite draws in {attempts} attempts; "
            "every resample hit a positivity violation or a non-finite value"
        )
    return draws[:filled]
