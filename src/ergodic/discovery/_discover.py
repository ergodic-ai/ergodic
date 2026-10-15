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

if TYPE_CHECKING:
    from ..data import PanelData, TabularData, TimeSeriesData
    from ..knowledge import DomainKnowledge
    from ._result import DiscoveryResult


def discover(
    data: TabularData | TimeSeriesData | PanelData,
    method: str = "pc",
    *,
    test: Any = None,
    score: Any = None,
    knowledge: DomainKnowledge | None = None,
    alpha: float = 0.05,
    **kwargs: Any,
) -> DiscoveryResult:
    """Learn causal structure from ``data`` with the named ``method``.

    ``test`` feeds a constraint method (a :class:`CITest` instance built
    without data, since ``discover`` binds it to ``data``, or an alias like
    ``"linear"``; a bound test runs through the algorithm class, as in
    ``PC(test=GCM(data)).fit()``); ``score`` feeds a score-based method
    (a :class:`Score` instance or an alias like ``"bic"``, ``"bdeu"``); both
    default to ``"auto"``, picked from the column schema. Each algorithm
    declares which ingredient it consumes. Extra keyword arguments pass to
    the algorithm (``max_cond_size``, ``max_parents``, ``node_limit``,
    ``max_degree``, ``tabu``, ``max_lag``).

    The iid methods read a :class:`TabularData` and reject temporal shapes
    (rows are not exchangeable); ``method="pcmci"`` reads a
    :class:`TimeSeriesData` or :class:`PanelData` and returns a window graph
    over lagged nodes like ``"X[t-1]"``.

    A few methods consume neither ingredient because the machinery is the
    algorithm: ``"lingam"`` (regression and residual independence) and
    ``"glasso"`` (a sparse precision estimate). They reject ``test=`` and
    ``score=``, and ``alpha`` does not apply (their own knobs, like ``lam``
    or ``prune``, pass through the keyword arguments).

    ``method="welt:<model>"`` (for example ``"welt:arrow"``) runs a hosted Welt
    model through :func:`ergodic.welt.discover`, one paid Welt job. It needs
    ``pip install "ergodic[welt]"``, takes no test, score or ``alpha``, and
    passes only ``client``, ``seed``, ``timeout`` and ``decoder`` through the
    keyword arguments; anything else raises before any request. ``decoder=``
    asks for local decoding of the model's edge scores (:func:`ergodic.decode`),
    which is what lets a non-DAG model such as ``"welt:avici"`` return a
    (decoded) graph; ``knowledge`` applies to that decoding and, without
    ``decoder=``, implies greedy.
    """
    if method.lower().startswith("welt:"):
        return _welt(
            data, method, test=test, score=score, knowledge=knowledge, alpha=alpha, **kwargs
        )
    cls = algorithm_for(method)
    result: DiscoveryResult
    if cls.ingredient is None:
        if test is not None or score is not None:
            raise ValueError(f"method {method!r} consumes neither a test nor a score")
        result = cls(**kwargs).fit(data, knowledge=knowledge)
        return result
    if cls.ingredient == "test":
        if score is not None:
            raise ValueError(f"method {method!r} consumes a test, not a score")
        chosen_test = test if test is not None else "auto"
        result = cls(test=chosen_test, alpha=alpha, **kwargs).fit(data, knowledge=knowledge)
        return result
    if test is not None:
        raise ValueError(f"method {method!r} consumes a score, not a test")
    chosen_score = score if score is not None else "auto"
    result = cls(score=chosen_score, **kwargs).fit(data, knowledge=knowledge)
    return result


def _welt(
    data: Any,
    method: str,
    *,
    test: Any,
    score: Any,
    knowledge: DomainKnowledge | None,
    alpha: float,
    **kwargs: Any,
) -> DiscoveryResult:
    """Hand a ``welt:<model>`` method to :mod:`ergodic.welt`, imported only now."""
    if test is not None or score is not None:
        raise ValueError(f"method {method!r} runs a Welt model; it takes no test or score")
    if alpha != 0.05:
        raise ValueError(
            f"method {method!r} runs a Welt model; alpha is a local test setting and does not apply"
        )
    local = sorted(set(kwargs) - {"client", "seed", "timeout", "decoder"})
    if local:
        raise ValueError(
            f"method {method!r} runs a Welt model and takes only client, seed, timeout "
            f"and decoder; got {local}"
        )
    try:
        from ..welt import discover as welt_discover
    except ImportError as error:
        raise ImportError(
            f"method {method!r} needs welt-client: pip install 'ergodic[welt]'"
        ) from error
    model = method.split(":", 1)[1]
    return welt_discover(data, model, knowledge=knowledge, **kwargs)
