# SPDX-License-Identifier: Apache-2.0
"""A small discrete-factor evaluator for the estimand functional.

Given the observational joint ``P(V)`` as a dense factor (a numpy array over the
variables' integer domains), this evaluates any :class:`Functional` to a factor
over its free variables. It is the engine behind the soundness-against-truth test
(evaluate ID's functional on ``P(V)``, compare to the true interventional) and
the reference the discrete plug-in estimator reuses.

The pieces are the textbook discrete-factor operations: a product that aligns
named axes and broadcast-multiplies, a marginalization that sums over axes, a
ratio that divides with a zero-guard, and ``P(head | given)`` read from the joint
by marginalizing to the head-and-given variables and dividing by the marginal
over ``given``.

numpy only, no new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._functional import Functional, Marginal, Prob, Product, Ratio

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class Factor:
    """A discrete factor: a tuple of axis names and a numpy array over their domains.

    ``axes[i]`` names dimension ``i`` of ``values``; ``values.shape[i]`` is that
    variable's cardinality. A factor with empty ``axes`` is a scalar (a
    zero-dimensional array). Axes carry no inherent order beyond their position
    here; the operations realign by name.
    """

    axes: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != len(self.axes):
            raise ValueError(
                f"factor over {self.axes} needs a {len(self.axes)}-d array, "
                f"got shape {self.values.shape}"
            )
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"factor axes must be distinct, got {self.axes}")

    def _aligned_to(self, order: tuple[str, ...]) -> np.ndarray:
        """Return ``values`` transposed and broadcast to the axis order ``order``.

        ``order`` is a superset of ``self.axes``. Missing axes become size-1
        dimensions, so the result broadcasts against any factor laid out the
        same way.
        """
        present = [a for a in order if a in self.axes]
        perm = [self.axes.index(a) for a in present]
        arr = np.transpose(self.values, perm)
        shape = [self.values.shape[self.axes.index(a)] if a in self.axes else 1 for a in order]
        return arr.reshape(shape)


def _domains_for(factor: Factor, domains: Mapping[str, int]) -> dict[str, int]:
    return {a: factor.values.shape[i] for i, a in enumerate(factor.axes)} | dict(domains)


def product(factors: list[Factor], domains: Mapping[str, int]) -> Factor:
    """Multiply factors, aligning shared axes by name and broadcasting the rest."""
    if not factors:
        return Factor((), np.array(1.0))
    order: tuple[str, ...] = ()
    for f in factors:
        order = order + tuple(a for a in f.axes if a not in order)
    shape = tuple(_card(a, factors, domains) for a in order)
    out = np.ones(shape, dtype=float)
    for f in factors:
        out = out * f._aligned_to(order)
    return Factor(order, out)


def _card(axis: str, factors: list[Factor], domains: Mapping[str, int]) -> int:
    for f in factors:
        if axis in f.axes:
            return int(f.values.shape[f.axes.index(axis)])
    return int(domains[axis])


def marginalize(factor: Factor, index: frozenset[str]) -> Factor:
    """Sum ``factor`` over the axes named in ``index`` (axes not present are ignored)."""
    drop = tuple(a for a in factor.axes if a in index)
    if not drop:
        return factor
    keep = tuple(a for a in factor.axes if a not in index)
    axis_nums = tuple(factor.axes.index(a) for a in drop)
    return Factor(keep, factor.values.sum(axis=axis_nums))


def ratio(num: Factor, den: Factor) -> Factor:
    """Divide ``num`` by ``den``, aligning ``den``'s axes and guarding zeros.

    Where the denominator is zero the result is zero, not a NaN: an unreachable
    conditioning cell contributes nothing rather than blowing up. The result
    spans the union of the two factors' axes.
    """
    order: tuple[str, ...] = num.axes + tuple(a for a in den.axes if a not in num.axes)
    domains = _domains_for(num, _domains_for(den, {}))
    shape = tuple(domains[a] for a in order)
    n = np.broadcast_to(num._aligned_to(order), shape)
    d = np.broadcast_to(den._aligned_to(order), shape)
    out = np.zeros(shape, dtype=float)
    np.divide(n, d, out=out, where=d != 0)
    return Factor(order, out)


def _prob(joint: Factor, head: frozenset[str], given: frozenset[str]) -> Factor:
    """``P(head | given)`` read from the joint factor."""
    target = head | given
    pj = marginalize(joint, frozenset(joint.axes) - target)
    if not given:
        return pj
    pg = marginalize(joint, frozenset(joint.axes) - given)
    return ratio(pj, pg)


def evaluate(functional: Functional, joint: Factor, domains: Mapping[str, int]) -> Factor:
    """Evaluate ``functional`` on the observational ``joint``, returning its factor.

    The result is a factor over ``functional.free_vars()``. ``joint`` is the
    dense observational ``P(V)`` (its axes are the observed variables), and
    ``domains`` maps every variable name to its cardinality (a fallback for
    variables a sub-factor never realizes; the joint's own shape is preferred).
    """
    if isinstance(functional, Prob):
        return _prob(joint, functional.head, functional.given)
    if isinstance(functional, Product):
        parts = [evaluate(f, joint, domains) for f in functional.factors]
        return product(parts, domains)
    if isinstance(functional, Marginal):
        body = evaluate(functional.body, joint, domains)
        return marginalize(body, functional.index)
    if isinstance(functional, Ratio):
        return ratio(
            evaluate(functional.num, joint, domains),
            evaluate(functional.den, joint, domains),
        )
    raise TypeError(f"not a Functional: {functional!r}")  # pragma: no cover


def read_off(
    factor: Factor,
    targets: Mapping[str, int],
    given: Mapping[str, int] | None = None,
) -> float:
    """Read ``P(targets | given=...)`` off a result factor at a fixed assignment.

    ``factor`` is the evaluated functional (already a conditional or a joint over
    its free variables); ``targets`` and ``given`` assign each named axis an
    integer level. Every axis of ``factor`` must be named in ``targets`` or
    ``given``.
    Returns the scalar at that cell.
    """
    assignment = dict(targets) | dict(given or {})
    missing = set(factor.axes) - assignment.keys()
    if missing:
        raise ValueError(f"no assignment for axes {sorted(missing)}")
    idx = tuple(assignment[a] for a in factor.axes)
    return float(factor.values[idx]) if factor.axes else float(factor.values)
