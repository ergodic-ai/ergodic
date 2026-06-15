# SPDX-License-Identifier: Apache-2.0
"""The estimand functional: a small, frozen, sealed algebra.

This is what the ID algorithm returns and what every downstream consumer reads.
An estimand is a sum-product of conditional and marginal terms, with a ratio for
the conditional (IDC) case. The string the user used to see is now
``functional.render()``, in the same style as the named-criterion estimands, so
the readable-estimand convention survives.

PROVISIONAL: the surface (the subclasses and their fields) is public but marked
provisional while the ID layer settles. Consumers should read through
:meth:`render`, :meth:`free_vars`, and :meth:`simplify` where they can.

The algebra is deliberately minimal: it expresses exactly the functionals ID and
IDC produce. Difference and contrast terms (the ATE contrast, the mediation
decompositions) are not here; they are estimands-of-estimands and live one level
up, in the estimators and the mediation record.
"""

from __future__ import annotations

from dataclasses import dataclass


def _join(names: frozenset[str]) -> str:
    return ", ".join(sorted(names))


class Functional:
    """The sealed base of the estimand algebra. Use the frozen subclasses below.

    Three operations are common to every node: :meth:`render` (the readable
    estimand string), :meth:`free_vars` (the variable scope, used by recognition
    and the plug-in evaluator), and :meth:`simplify` (conservative rewrites
    only). Subclasses are frozen value objects, so two structurally equal
    functionals are equal and hash alike.
    """

    def render(self) -> str:
        """The readable estimand string, e.g. ``sum_z P(Y | X, z) P(z)``."""
        raise NotImplementedError

    def free_vars(self) -> frozenset[str]:
        """The variable scope: heads and givens, minus bound marginal indices."""
        raise NotImplementedError

    def simplify(self) -> Functional:
        """A conservatively simplified functional (safe rewrites only).

        Flatten nested products, drop a marginal over an empty index, drop a
        factor that is identically one. No cancellations, no clever algebra: two
        runs render the same estimand.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Prob(Functional):
    """An atomic term ``P(head | given)`` (a marginal when ``given`` is empty)."""

    head: frozenset[str]
    given: frozenset[str] = frozenset()

    def render(self) -> str:
        if self.given:
            return f"P({_join(self.head)} | {_join(self.given)})"
        return f"P({_join(self.head)})"

    def free_vars(self) -> frozenset[str]:
        return self.head | self.given

    def simplify(self) -> Functional:
        return self

    def _is_one(self) -> bool:
        """``P(s | s)`` and ``P()`` are identically one and drop from a product."""
        return not self.head or self.head <= self.given


@dataclass(frozen=True)
class Product(Functional):
    """A product of factors."""

    factors: tuple[Functional, ...]

    def render(self) -> str:
        if not self.factors:
            return "1"
        return " ".join(f.render() for f in self.factors)

    def free_vars(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for f in self.factors:
            out |= f.free_vars()
        return out

    def simplify(self) -> Functional:
        flat: list[Functional] = []
        for f in self.factors:
            s = f.simplify()
            if isinstance(s, Product):
                flat.extend(s.factors)  # flatten nested products
            elif isinstance(s, Prob) and s._is_one():
                continue  # drop an identically-one factor
            else:
                flat.append(s)
        if len(flat) == 1:
            return flat[0]
        return Product(tuple(flat))


@dataclass(frozen=True)
class Marginal(Functional):
    """Sum ``index`` out of ``body`` (a marginalization)."""

    index: frozenset[str]
    body: Functional

    def render(self) -> str:
        if not self.index:
            return self.body.render()
        sums = " ".join(f"sum_{v.lower()}" for v in sorted(self.index))
        return f"{sums} {self.body.render()}"

    def free_vars(self) -> frozenset[str]:
        return self.body.free_vars() - self.index

    def simplify(self) -> Functional:
        body = self.body.simplify()
        # Only sum over indices the body actually mentions; if nothing remains,
        # the marginal is the identity.
        index = self.index & body.free_vars()
        if not index:
            return body
        if isinstance(body, Marginal):
            return Marginal(index | body.index, body.body)  # merge nested sums
        return Marginal(index, body)


@dataclass(frozen=True)
class Ratio(Functional):
    """A ratio ``num / den`` (the IDC normalization ``P_x(y, z) / P_x(z)``)."""

    num: Functional
    den: Functional

    def render(self) -> str:
        return f"[{self.num.render()}] / [{self.den.render()}]"

    def free_vars(self) -> frozenset[str]:
        return self.num.free_vars() | self.den.free_vars()

    def simplify(self) -> Functional:
        return Ratio(self.num.simplify(), self.den.simplify())


# -- helper constructors (used by the ID algorithm) --------------------------


def marg(body: Functional, index: frozenset[str]) -> Functional:
    """``Marginal(index, body)``: sum ``index`` out of ``body``."""
    return Marginal(index, body)


def cond(p: Functional, head: frozenset[str], given: frozenset[str]) -> Functional:
    """``P(head | given)`` under the functional ``p`` whose scope is ``p.free_vars()``.

    The general form is a ratio of marginals over ``p``'s own scope: the
    numerator sums every scope variable outside ``head`` and ``given`` out of
    ``p``, the denominator sums every scope variable outside ``given``. This
    is load-bearing for ID: the c-factor conditionals in steps 5b and 5c are
    conditionals of the *current* functional, not of the raw observational
    distribution. When ``p`` is a plain joint ``P(V)`` (a ``Prob`` over its whole
    scope with no conditioning) the ratio is just the plain conditional
    ``P(head | given)``, so that case is built directly and renders cleanly. At
    the top level of ID ``p`` is exactly that joint, so the estimand reads in
    the familiar form; after step 2 leaves a ``Marginal`` (or step 5c a
    ``Product``), ``cond`` falls to the exact ratio of marginals.
    """
    scope = p.free_vars()
    if isinstance(p, Prob) and not p.given and p.head == scope:
        return Prob(head, given)
    num = marg(p, scope - (head | given))
    den = marg(p, scope - given)
    return Ratio(num=num, den=den)
