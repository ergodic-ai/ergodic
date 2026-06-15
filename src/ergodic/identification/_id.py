# SPDX-License-Identifier: Apache-2.0
"""The general do-calculus identification algorithm: ID, IDC, and the hedge.

The Shpitser-Pearl ID algorithm (2006) decides, for a semi-Markovian model (a
DAG or ADMG, possibly with bows), whether ``P(y | do(x))`` is identifiable from
the observational distribution, and if so returns the estimand as a
:class:`~ergodic.identification._functional.Functional`. IDC extends it to
conditional effects ``P(y | do(x), z)``. When the effect is not identifiable the
algorithm exhibits a :class:`Hedge`, the structural witness (a C-forest pair)
that proves it.

The recursion carries the estimand as it builds: ``P`` is a ``Functional`` whose
scope tracks the active variables, so each step rewrites the functional rather
than a string. The one subtle point (and the classic implementation bug) is that
the c-factor conditionals in the terminal steps are conditionals of the *current*
functional, via :func:`~ergodic.identification._functional.cond`, not bare
``Prob`` terms; the soundness-against-truth test is what proves this is right.
A second, related subtlety lives in :func:`_c_factor`: when a deeper recursion
folds intervened-away do-targets into ``P`` (the c-factor ``Q[S']`` is a
distribution over ``S'`` parameterized by the rest), those carried-over context
variables must stay conditioned in every term, never integrated out. Both the
soundness-against-truth and the verdict cross-check against an independent
Tian-Pearl oracle hold to machine precision.

This module is the engine. The public ``identify_distribution`` wrapper, the
``identify_effect`` fallthrough, and ``ida`` are wired on top of it elsewhere.
The internal runner here, :func:`_identify_admg`, latent-projects its input and
returns an :class:`IDResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..graph import districts, latent_projection, topological_order
from ..graph._surgery import _cut_outgoing, intervene
from ._functional import Functional, Prob, Product, Ratio, cond, marg

if TYPE_CHECKING:
    from ..graph import MixedGraph


# -- the non-identifiability witness -----------------------------------------


@dataclass(frozen=True)
class Hedge(Exception):
    """The structural witness that an effect is not identifiable (a C-forest pair).

    ID fails at a subproblem whose whole graph is a single c-component over
    ``nodes`` and whose treatment-free part is the district ``district``. That
    pair is a hedge in the sense of Shpitser-Pearl: it is the obstruction that
    proves ``P(y | do(x))`` is not a function of the observational distribution.
    The SCM bounds layer reads this object to decide where to return bounds
    instead of a point.

    The witness is the ``(nodes, district)`` summary of the failing subproblem,
    enough to mark non-identifiability and to seed the bounds. The explicit
    minimal C-forest pair (the rooted forests of Shpitser-Pearl) is not extracted
    yet; that is a refinement for later.
    """

    nodes: frozenset[str]
    district: frozenset[str]

    def render(self) -> str:
        """A human-readable reason."""
        return (
            f"not identifiable: a hedge formed by {sorted(self.nodes)} "
            f"and the c-component {sorted(self.district)}"
        )

    def __str__(self) -> str:
        return self.render()


# -- the result record -------------------------------------------------------


@dataclass(frozen=True)
class IDResult:
    """The outcome of running ID or IDC on a single graph.

    ``identifiable`` is the verdict; ``functional`` is the estimand when it
    holds (``None`` otherwise); ``hedge`` is the witness when it fails (``None``
    otherwise). ``query`` records the ``(outcome, do, given)`` that was asked.
    """

    identifiable: bool
    functional: Functional | None
    hedge: Hedge | None
    query: tuple[frozenset[str], frozenset[str], frozenset[str]]

    def __bool__(self) -> bool:
        return self.identifiable

    def render(self) -> str:
        if self.identifiable and self.functional is not None:
            return self.functional.render()
        if self.hedge is not None:
            return self.hedge.render()
        return "not identifiable"

    def __repr__(self) -> str:
        y, x, z = self.query
        cond_part = f", given={sorted(z)}" if z else ""
        head = f"P({sorted(y)} | do({sorted(x)}){cond_part})"
        if not self.identifiable:
            return f"IDResult(not identifiable, {head})"
        return f"IDResult({head} = {self.render()})"


# -- helpers -----------------------------------------------------------------


def _ancestors(graph: MixedGraph, nodes: frozenset[str]) -> frozenset[str]:
    """The ancestors of ``nodes`` in ``graph``, inclusive of ``nodes`` themselves."""
    out: set[str] = set(nodes)
    for n in nodes:
        out |= graph.ancestors(n)
    return frozenset(out)


def _preds(v: str, order: tuple[str, ...], scope: frozenset[str]) -> frozenset[str]:
    """Nodes strictly before ``v`` in ``order``, intersected with ``scope``."""
    out: set[str] = set()
    for node in order:
        if node == v:
            break
        out.add(node)
    return frozenset(out) & scope


def _c_factor(
    p: Functional,
    region: frozenset[str],
    order: tuple[str, ...],
    nodes: frozenset[str],
) -> Functional:
    """The c-factor of ``region``: ``prod_{v in region} cond(P, {v}, preds(v))``.

    Each factor is a conditional of the *current* functional ``p`` (via
    :func:`cond`), not of the raw observational distribution. The conditioning
    set of each term is the predecessors of ``v`` in ``order`` (the current
    graph's topological order) restricted to ``p``'s scope, *together with the
    carried-over context*: any variable in ``p``'s scope that is not a node of
    the current graph (``nodes``).

    That context is the do-targets a deeper recursion folded into ``p`` as a
    c-factor (``Q[S']`` is ``P(S' | do(V\\S'))``, a distribution over ``S'``
    parameterized by the intervened-away rest). Those variables stay in ``p``'s
    scope and must stay in the conditioning of every term, never integrated out,
    so the conditional reads ``P(v | predecessors, context)``. Dropping them is
    the classic ID bug: it marginalizes a do-target the sub-problem is solved
    within, and the estimand comes out wrong. The context is resolved later, at
    the outer recursion level where those variables are graph nodes again.
    """
    scope = p.free_vars()
    context = scope - nodes
    factors = tuple(
        cond(p, frozenset({v}), _preds(v, order, scope) | context) for v in sorted(region)
    )
    return Product(factors)


# -- ID ----------------------------------------------------------------------


def _id(y: frozenset[str], x: frozenset[str], p: Functional, g: MixedGraph) -> Functional:
    """The Shpitser-Pearl ID recursion. Returns the estimand or raises :class:`Hedge`.

    ``p`` is a ``Functional`` whose scope equals the nodes of ``g`` at entry;
    ``g`` is an ADMG (it may carry bows from latent projection). The seven-step
    form matches Tikka-Karvanen's ``causaleffect``.
    """
    v = g.nodes

    # 1. nothing left to intervene on: marginalize the functional to y.
    if not x:
        return marg(p, v - y)

    # 2. restrict to the ancestors of y: nodes off every causal path to y drop.
    anc = _ancestors(g, y)
    if anc != v:
        return _id(y, x & anc, marg(p, v - anc), g.induced_subgraph(anc))

    # 3. add to the intervention set anything not a post-intervention ancestor of y.
    anc_cut = _ancestors(intervene(g, x), y)
    w = (v - x) - anc_cut
    if w:
        return _id(y, x | w, p, g)

    # 4. split into the c-components of G \ x.
    components = districts(g.induced_subgraph(v - x))
    if len(components) > 1:
        parts = tuple(_id(s, v - s, p, g) for s in sorted(components, key=sorted))
        return marg(Product(parts), v - (y | x))

    # one district S remains; look at the c-components of the whole graph.
    s = next(iter(components))
    whole = districts(g)
    order = topological_order(g)

    # 5a. the whole graph is one c-component: the hedge, not identifiable.
    if whole == frozenset({v}):
        raise Hedge(nodes=v, district=s)

    # 5b. S is itself a c-component of G: terminate with its c-factor.
    if s in whole:
        return marg(_c_factor(p, s, order, v), s - y)

    # 5c. S sits strictly inside a larger district S': recurse on G[S'].
    s_prime = next(d for d in whole if s < d)
    return _id(y, x & s_prime, _c_factor(p, s_prime, order, v), g.induced_subgraph(s_prime))


# -- IDC ---------------------------------------------------------------------


def _idc(
    y: frozenset[str], x: frozenset[str], z: frozenset[str], p: Functional, g: MixedGraph
) -> Functional:
    """The Shpitser-Pearl IDC recursion for ``P(y | do(x), z)``.

    For each conditioning ``Z``, if the do-calculus rule-2 separation holds (in
    the graph with edges into ``x`` cut and edges out of ``Z`` cut), move ``Z``
    into the intervention. Otherwise normalize the joint interventional
    ``P(y, z | do(x))`` by its marginal over ``y``. A hedge from the inner ID
    propagates.
    """
    intervened = intervene(g, x)
    for z_node in sorted(z):
        cut = _cut_outgoing(intervened, frozenset({z_node}))
        if cut.m_separated(y, frozenset({z_node}), x | (z - {z_node})):
            return _idc(y, x | {z_node}, z - {z_node}, p, g)
    joint = _id(y | z, x, p, g)
    return Ratio(num=joint, den=marg(joint, y))


# -- the internal runner -----------------------------------------------------


def _identify_admg(
    graph: MixedGraph,
    outcome: frozenset[str],
    do: frozenset[str],
    given: frozenset[str] = frozenset(),
) -> IDResult:
    """Run ID (or IDC) on ``graph`` after latent-projecting it to an ADMG.

    The input may be a DAG or ADMG with ``latent=`` designations or bows;
    :func:`latent_projection` turns it into a clean ADMG over the observed
    margin first, so latents and bows are handled uniformly. Returns an
    :class:`IDResult`: identifiable with the functional, or not identifiable
    with the hedge. Name validation and the single-graph-face check are the
    caller's job (the public wrapper does them); this runner assumes a DAG or
    ADMG and projects.
    """
    projected = latent_projection(graph)
    v = projected.nodes
    init = Prob(v)
    query = (outcome, do, given)
    try:
        if given:
            functional = _idc(outcome, do, given, init, projected)
        else:
            functional = _id(outcome, do, init, projected)
    except Hedge as hedge:
        return IDResult(identifiable=False, functional=None, hedge=hedge, query=query)
    return IDResult(
        identifiable=True,
        functional=functional.simplify(),
        hedge=None,
        query=query,
    )
