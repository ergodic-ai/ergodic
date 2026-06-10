# SPDX-License-Identifier: Apache-2.0
"""The functional facade: data in, a discovered graph out.

``discover`` resolves the method, hands it the matching ingredient (a test
for the constraint algorithms, a score for the score-based ones), fits, and
returns the :class:`DiscoveryResult`. The class API underneath stays the
expressive path; this is the one-liner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._registry import algorithm_for
from .algorithms._astar import AStar
from .algorithms._pc import PC

if TYPE_CHECKING:
    from ..data import TabularData
    from ..knowledge import DomainKnowledge
    from ._result import DiscoveryResult


def discover(
    data: TabularData,
    method: str = "pc",
    *,
    test: Any = None,
    score: Any = None,
    knowledge: DomainKnowledge | None = None,
    alpha: float = 0.05,
    **kwargs: Any,
) -> DiscoveryResult:
    """Learn causal structure from ``data`` with the named ``method``.

    ``test`` feeds a constraint method (a :class:`CITest` instance, bound or
    not, or an alias like ``"linear"``); ``score`` feeds a score-based method
    (a :class:`Score` instance or ``"bic"``). Extra keyword arguments pass to
    the algorithm (``max_cond_size``, ``max_parents``, ``node_limit``).
    """
    cls = algorithm_for(method)
    if cls is PC:
        if score is not None:
            raise ValueError(f"method {method!r} consumes a test, not a score")
        chosen_test = test if test is not None else "auto"
        return PC(test=chosen_test, alpha=alpha, **kwargs).fit(data, knowledge=knowledge)
    if test is not None:
        raise ValueError(f"method {method!r} consumes a score, not a test")
    chosen_score = score if score is not None else "bic"
    return AStar(score=chosen_score, **kwargs).fit(data, knowledge=knowledge)
