# SPDX-License-Identifier: Apache-2.0
"""The public ID/IDC wrapper and the ``identify_effect`` ID fallthrough.

:func:`identify_distribution` is the raw entry to the general do-calculus engine:
any interventional or conditional-interventional query ``P(y | do(x), z)`` on a
single graph (a DAG or ADMG), returning an
:class:`~ergodic.identification._id.IDResult`. It validates names and the single
graph face, then hands off to :func:`~ergodic.identification._id._identify_admg`,
which latent-projects and runs ID (or IDC when ``given`` is nonempty).

:func:`id_fallthrough` is what :func:`~ergodic.identification._criteria.identify_effect`
calls when its named ladder (null, adjustment, front-door, instruments) misses on
a single graph. It runs general ID, recognizes a named closed form when there is
one (so today's estimators still run), and otherwise reports ``strategy="id"``
with the typed functional. A hedge becomes a not-identifiable
:class:`~ergodic.identification._record.Identification` carrying the witness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._criteria import (
    _adjustment_formula,
    _as_set,
    _frontdoor_formula,
    _is_class_kind,
    _validate,
)
from ._id import _identify_admg
from ._recognize import recognize
from ._record import Identification

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..graph import MixedGraph
    from ._id import IDResult


def identify_distribution(
    graph: MixedGraph,
    outcome: str | Iterable[str],
    do: str | Iterable[str],
    given: str | Iterable[str] = frozenset(),
) -> IDResult:
    """Run the general do-calculus ID (or IDC) algorithm on a single graph.

    Decides whether ``P(outcome | do(do), given)`` is identifiable from the
    observational distribution and, when it is, returns the estimand as a typed
    functional inside an :class:`~ergodic.identification._id.IDResult`. With
    ``given`` empty this is ID; with ``given`` nonempty it is the conditional
    algorithm IDC. The graph may be a DAG or an ADMG, and may carry ``latent=``
    designations or bows (it is latent-projected to an ADMG first).

    A CPDAG, MAG, or PAG raises: ID is defined on a single graph, and the class
    is covered by :func:`~ergodic.identification.ida`, which enumerates members.
    A ``mixed``-face graph raises, like the rest of identification.
    """
    y = _as_set(outcome)
    x = _as_set(do)
    z = _as_set(given)
    _validate(graph, x, y, z)
    if x & z:
        raise ValueError("the do-set and the conditioning set must be disjoint")
    if y & z:
        raise ValueError("the outcome and the conditioning set must be disjoint")
    if _is_class_kind(graph):
        raise NotImplementedError(
            f"identify_distribution runs the ID algorithm on a single graph, not on "
            f"an equivalence class ({graph.kind.value}); use ida() to enumerate the "
            "class and identify per member, or extract a member first"
        )
    return _identify_admg(graph, y, x, z)


def id_fallthrough(graph: MixedGraph, x: frozenset[str], y: frozenset[str]) -> Identification:
    """The general-ID step ``identify_effect`` runs after its named ladder misses.

    Runs ID for ``P(y | do(x))``. If it identifies, the functional passes through
    recognition (:func:`~ergodic.identification._recognize.recognize`): a
    recognized named form returns that strategy and set (so the existing
    estimators consume it), otherwise ``strategy="id"`` with the functional set
    and ``estimand=functional.render()``. If ID fails, the result is
    not-identifiable with the hedge witness and a hedge-based reason.

    The caller (:func:`identify_effect`) has already settled names, the single
    graph face, the null case, adjustment, the front-door, and instruments; this
    is the last word on a single graph.
    """
    result = _identify_admg(graph, y, x)
    if not result.identifiable:
        assert result.hedge is not None  # not identifiable implies a hedge
        return Identification(
            x,
            y,
            False,
            "none",
            estimand=result.hedge.render(),
            hedge=result.hedge,
        )
    assert result.functional is not None  # identifiable implies a functional
    named = recognize(graph, x, y, result.functional)
    if named is not None:
        strategy, chosen = named
        if strategy == "adjustment":
            return Identification(
                x,
                y,
                True,
                "adjustment",
                adjustment_set=chosen,
                estimand=_adjustment_formula(x, y, chosen),
            )
        if strategy == "frontdoor":
            return Identification(
                x,
                y,
                True,
                "frontdoor",
                mediators=chosen,
                estimand=_frontdoor_formula(x, y, chosen),
            )
    return Identification(
        x,
        y,
        True,
        "id",
        estimand=result.functional.render(),
        functional=result.functional,
    )
