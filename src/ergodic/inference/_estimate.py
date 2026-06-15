# SPDX-License-Identifier: Apache-2.0
"""The functional facade: graph and data in, an effect estimate out.

``estimate_effect`` runs identification then estimation. ``estimate`` takes an
``Identification`` you already hold. Both are thin wrappers over a stateful
``CausalEstimator``: resolve the estimator for the family, fill its slots with the
requested learner (or the default), fit, and report the target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..identification import identify_effect
from ._registry import default_target, estimator_for

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..data import TabularData
    from ..graph import MixedGraph
    from ..identification import Identification
    from ._result import EffectEstimate
    from ._target import EffectTarget


def estimate(
    identification: Identification,
    data: TabularData,
    *,
    strategy: str | None = None,
    learner: Any = None,
    target: EffectTarget | str | None = None,
    treated: Any | None = None,
    random_state: int | None = None,
    n_boot: int = 1000,
    **kwargs: Any,
) -> EffectEstimate:
    """Estimate an identified effect over tabular data.

    ``strategy`` names the estimator (else the family default). ``learner`` is an
    alias, a learner, or a slot-to-learner mapping (else the slot defaults).
    ``treated`` names the treatment level that codes to 1 for a two-level
    treatment (else level order decides, with a warning when the levels are not
    0/1). Extra keyword arguments pass to the estimator (``per_arm``, ``scheme``,
    ``clip``).
    """
    if not identification.identifiable:
        raise ValueError(f"effect is not identifiable: {identification.estimand}")
    if identification.strategy == "id":
        raise NotImplementedError(
            "the effect is identified only by the general do-calculus ID algorithm, "
            f"with the functional {identification.estimand!r}; no named estimator "
            "consumes a general functional. The estimation paths are the discrete "
            "plug-in (evaluating the functional over empirical conditionals on "
            "categorical data) and the SCM's fitted mechanisms (continuous), "
            "neither wired in yet"
        )
    estimator_cls = estimator_for(identification, strategy)
    estimator = estimator_cls(
        learner, random_state=random_state, n_boot=n_boot, treated=treated, **kwargs
    )
    estimator.fit(identification, data)
    chosen = target if target is not None else default_target(identification)
    return estimator.effect(target=chosen)


def estimate_effect(
    graph: MixedGraph,
    data: TabularData,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    *,
    strategy: str | None = None,
    learner: Any = None,
    target: EffectTarget | str | None = None,
    treated: Any | None = None,
    random_state: int | None = None,
    n_boot: int = 1000,
    **kwargs: Any,
) -> EffectEstimate:
    """Identify the effect of ``treatment`` on ``outcome`` in ``graph``, then estimate it.

    ``treated`` names the treatment level that codes to 1 (see :func:`estimate`).
    """
    identification = identify_effect(graph, treatment, outcome)
    if not identification.identifiable:
        raise ValueError(
            f"effect of {treatment} on {outcome} is not identifiable: {identification.estimand}"
        )
    return estimate(
        identification,
        data,
        strategy=strategy,
        learner=learner,
        target=target,
        treated=treated,
        random_state=random_state,
        n_boot=n_boot,
        **kwargs,
    )
