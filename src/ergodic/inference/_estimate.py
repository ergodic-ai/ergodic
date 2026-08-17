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
from ._registry import cate_estimator_for, cate_kwargs_for, default_target, estimator_for
from .estimators._plugin import plugin_estimate

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..data import TabularData
    from ..graph import MixedGraph
    from ..identification import Identification
    from ._perunit import CATEResult
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
        if strategy is not None:
            raise ValueError(
                f"a general ID functional has no named estimator to select; got "
                f"strategy={strategy!r}. The discrete plug-in evaluates it directly"
            )
        return plugin_estimate(
            identification,
            data,
            target=target,
            treated=treated,
            random_state=random_state,
            n_boot=n_boot,
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


def cate(
    graph: MixedGraph,
    data: TabularData,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    *,
    method: str = "auto",
    learner: Any = None,
    treated: Any | None = None,
    random_state: int | None = None,
    n_boot: int = 1000,
    **kwargs: Any,
) -> CATEResult:
    """Identify the effect of ``treatment`` on ``outcome``, then estimate the CATE.

    Identification-gated exactly like :func:`estimate_effect`: identify the effect,
    select a CATE-capable estimator for the family, fit it, and return a
    :class:`~ergodic.inference.CATEResult`. ``method`` names the meta-learner
    (``"dr_learner"`` by default, the doubly-robust choice that shares the
    explainability score helper; also ``"r_learner"``, ``"x_learner"``,
    ``"t_learner"``, ``"s_learner"``, ``"dml"``). ``learner`` fills the nuisance
    slots (an alias, a learner, or a slot mapping). The result carries per-unit
    uncertainty (a posterior with a Bayesian slot, the bootstrap otherwise, the
    analytic standard error for the linear DR final) and the cached doubly-robust
    scores, so :meth:`~ergodic.inference.CATEResult.explain` runs the Tier-B
    characterization.
    """
    identification = identify_effect(graph, treatment, outcome)
    if not identification.identifiable:
        raise ValueError(
            f"effect of {treatment} on {outcome} is not identifiable: {identification.estimand}"
        )
    estimator_cls = cate_estimator_for(method)
    if identification.strategy not in estimator_cls.consumes:
        raise ValueError(
            f"the CATE estimator {estimator_cls.name!r} consumes "
            f"{sorted(estimator_cls.consumes)}, but the effect was identified by "
            f"{identification.strategy!r}; CATE estimation needs an adjustment-identified "
            "effect (back-door). Front-door, instruments, and the general-ID functional "
            "have no per-unit meta-learner yet"
        )
    from typing import cast

    from ._registry import HasCate

    method_kwargs = {**cate_kwargs_for(method), **kwargs}
    estimator = estimator_cls(
        learner, random_state=random_state, n_boot=n_boot, treated=treated, **method_kwargs
    )
    estimator.fit(identification, data)
    return cast("HasCate", estimator).cate()
