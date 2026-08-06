# SPDX-License-Identifier: Apache-2.0
"""The forecast result: one shape for every model, from a naive baseline up.

:class:`ForecastResult` is the trajectory sibling of
:class:`~ergodic.inference.EffectEstimate`. Where an effect estimate is a scalar
with uncertainty, a forecast is a set of named series, each carried as ``(S, H)``
posterior or bootstrap draws. The accessors read the cloud: :meth:`median` is the
point path, :meth:`interval` the equal-tailed band, :meth:`draws` the raw sample.

Provenance fields (the mechanisms, the window graph, the training schedule, the
sampler diagnostics, the reconciliation strategy) default to ``None`` and get
filled by the later sprints. A baseline fills none of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    import pandas as pd

    from ..data import Hierarchy
    from ..graph import MixedGraph

__all__ = ["FitStrategy", "ForecastResult", "coherence_gap"]


@dataclass(frozen=True)
class FitStrategy:
    """How a forecast was fitted, and why that one.

    ``strategy`` is what ran: ``"independent"`` (one model per series, nothing
    crossing between them), ``"staged"`` (one model per node, posteriors handed
    down the graph), or ``"joint"`` (one model over every node, so their
    posteriors stay correlated). ``requested`` is what the caller asked for, which
    is ``"auto"`` most of the time.

    ``n_mcmc_nodes`` and ``auto_threshold`` are the arithmetic behind an
    ``"auto"`` choice: a graph with no more modeled nodes than the threshold is
    fitted jointly, and anything larger is staged. Carrying both means the choice
    can be read off a result rather than reasoned about.
    """

    strategy: str
    requested: str = "auto"
    n_mcmc_nodes: int = 0
    auto_threshold: int | None = None

    def __repr__(self) -> str:
        body = self.strategy
        if self.requested == "auto" and self.auto_threshold is not None:
            body += f" (auto: {self.n_mcmc_nodes} node(s), joint at or under {self.auto_threshold})"
        elif self.requested != self.strategy:
            body += f" (requested {self.requested!r})"
        return f"FitStrategy({body})"


@dataclass(frozen=True)
class ForecastResult:
    """Forecast draws for one or more series.

    ``series`` maps a node name to its draws, shape ``(S, H)``: ``S`` sample
    paths of length ``horizon``. Every node carries the same ``S`` and the same
    ``H``. ``kind`` records where the draws came from (``"draws"`` for a sampled
    forecast). ``start`` is the label of the first forecast step when the data
    made it computable, else ``None``.

    Each array is copied on the way in and frozen, so the result owns its draws:
    the caller's array stays writable, and nothing the caller does to it
    afterwards can reach inside the result.

    Draws are permissive about NaN: a masked observation window can legitimately
    leave one, and a metric that meets it says so rather than the constructor.
    """

    series: Mapping[str, npt.NDArray[np.float64]]
    horizon: int
    kind: str = "draws"
    start: Any | None = None
    mechanisms: Mapping[str, Any] | None = None
    graph: MixedGraph | None = None
    schedule: Any | None = None
    #: How the fit ran and why, as a :class:`FitStrategy`. ``None`` for a model
    #: that does no sampling.
    strategy: FitStrategy | None = None
    #: Per-node fitted handles (a ``NodeFit`` each), the posterior behind the
    #: draws. A joint fit gives every node the same trace, which is what a joint
    #: fit is. ``None`` for a model that does no sampling.
    fits: Mapping[str, Any] | None = None
    #: Per-node sampler diagnostics, one record each (divergences, minimum bulk
    #: ESS, r-hat when there is more than one chain). A model that does no
    #: sampling leaves it None.
    diagnostics: Mapping[str, Any] | None = None
    reconciliation: str | None = None
    coherent: bool | None = None
    #: Each node's in-sample one-step residual second moment, when the fits
    #: computed it: the mean of the squared residuals, uncentered, which is what
    #: MinT wants (its W estimates ``E[e e']``, not the spread of e around its own
    #: mean, so a biased fit is charged for its bias). It rides along so the whole
    #: reconciliation menu stays available after the fact: MinT-shrink weights are
    #: a property of the fit rather than of the draws, and without them here an
    #: unreconciled result could not be reconciled that way without sampling
    #: everything again.
    residual_var: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not self.series:
            raise ValueError("a forecast needs at least one series")
        cleaned: dict[str, npt.NDArray[np.float64]] = {}
        n_draws: int | None = None
        for name, draws in self.series.items():
            # Copy before freezing: asarray would hand back the caller's own array,
            # so setflags would freeze an object we do not own, and a view passed in
            # would leave its base writable underneath us.
            arr = np.array(draws, dtype=np.float64, copy=True)
            if arr.ndim != 2:
                raise ValueError(f"series {name!r} must be 2-D (S, H), got shape {arr.shape}")
            if arr.shape[1] != self.horizon:
                raise ValueError(
                    f"series {name!r} has {arr.shape[1]} columns, expected horizon={self.horizon}"
                )
            if n_draws is None:
                n_draws = int(arr.shape[0])
            elif int(arr.shape[0]) != n_draws:
                raise ValueError(
                    f"series {name!r} has {arr.shape[0]} draws, expected {n_draws}; "
                    "every node shares one draw axis"
                )
            arr.setflags(write=False)
            cleaned[str(name)] = arr
        object.__setattr__(self, "series", cleaned)

    # -- shape ---------------------------------------------------------------

    @property
    def nodes(self) -> tuple[str, ...]:
        """The forecast series names, in insertion order."""
        return tuple(self.series)

    @property
    def n_draws(self) -> int:
        """The number of sample paths per node."""
        first = next(iter(self.series.values()))
        return int(first.shape[0])

    def _one(self, node: str | None) -> str:
        """Resolve the node argument, which may be omitted for a single series."""
        if node is None:
            if len(self.series) != 1:
                raise ValueError(
                    f"node= is required when the result holds more than one series: "
                    f"{list(self.series)}"
                )
            return next(iter(self.series))
        if node not in self.series:
            raise KeyError(f"no series {node!r}; the result holds {list(self.series)}")
        return node

    # -- the distribution ----------------------------------------------------

    def draws(self, node: str | None = None) -> npt.NDArray[np.float64]:
        """The raw draws for one node, shape ``(S, H)``."""
        return self.series[self._one(node)]

    def median(self, node: str | None = None) -> npt.NDArray[np.float64]:
        """The per-step posterior median, shape ``(H,)``."""
        return np.asarray(np.median(self.draws(node), axis=0), dtype=np.float64)

    def mean(self, node: str | None = None) -> npt.NDArray[np.float64]:
        """The per-step posterior mean, shape ``(H,)``."""
        return np.asarray(np.mean(self.draws(node), axis=0), dtype=np.float64)

    def interval(
        self, node: str | None = None, level: float = 0.9
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The equal-tailed interval at ``level``, two arrays of shape ``(H,)``."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        draws = self.draws(node)
        tail = (1.0 - level) / 2.0
        lo = np.percentile(draws, tail * 100.0, axis=0)
        hi = np.percentile(draws, (1.0 - tail) * 100.0, axis=0)
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    # -- the hierarchy -------------------------------------------------------

    def coherence_error(self, hierarchy: Hierarchy, *, relative: bool = True) -> float:
        """The worst gap between a parent and its children, over every draw and step.

        Zero means the forecast keeps the hierarchy's arithmetic promise exactly.
        The default reading is **relative**: the gap divided by the size of the
        parent it belongs to, so the number means the same thing on a series in
        units and on the same series in billions. That is the honest contract,
        because the residual reconciliation leaves behind is rounding, and
        rounding is relative: adding four float64 numbers near 1e9 cannot do
        better than about 1e-7 absolute, and calling that incoherent would be
        wrong. Anything a strategy leaves should read as machine epsilon here,
        around 1e-16, whatever the series is measured in.

        Pass ``relative=False`` for the gap in the units of the series, which is
        what to look at when a real disagreement needs sizing.

        Every node of the tree has to be in the result, since a missing one is a
        gap that cannot be measured rather than a gap of zero.
        """
        return coherence_gap(self.series, hierarchy, relative=relative)

    def to_frame(self, level: float = 0.9) -> pd.DataFrame:
        """A tidy frame: one row per node and step, with the median and the band."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for name in self.series:
            median = self.median(name)
            lo, hi = self.interval(name, level=level)
            for step in range(self.horizon):
                rows.append(
                    {
                        "node": name,
                        "step": step + 1,
                        "median": float(median[step]),
                        "lo": float(lo[step]),
                        "hi": float(hi[step]),
                    }
                )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        names = ", ".join(self.nodes)
        body = f"{len(self.series)} series [{names}], horizon={self.horizon}, S={self.n_draws}"
        if self.coherent is not None:
            body += f", coherent={self.coherent}"
        return f"ForecastResult({self.kind}: {body})"


def coherence_gap(
    series: Mapping[str, npt.NDArray[np.float64]],
    hierarchy: Hierarchy,
    *,
    relative: bool = True,
) -> float:
    """The largest ``|parent - aggregate(children)|`` over parents, draws, and steps.

    Relative by default: each parent's gap is divided by its own size, so a
    branch is judged against itself rather than against the biggest number in the
    tree, and the answer does not change when the series changes units. The
    divisor is floored at one, which keeps the reading absolute for a series that
    lives near zero, where a relative gap would be noise over noise.

    Free-standing so a set of draws can be measured before it is wrapped in a
    result, which is what the reconciliation layer does on its way out.
    """
    missing = sorted(set(hierarchy.nodes) - set(series))
    if missing:
        raise ValueError(
            f"coherence cannot be measured: {missing} have no draws. Every node of the "
            "hierarchy has to be present, since a missing series is an unknown gap rather "
            "than a gap of zero."
        )
    worst = 0.0
    for parent in hierarchy.internal_nodes:
        kids = hierarchy.children_of(parent)
        stacked = np.stack([np.asarray(series[kid], dtype=np.float64) for kid in kids], axis=0)
        expected = stacked.mean(axis=0) if hierarchy.aggregation == "mean" else stacked.sum(axis=0)
        top = np.asarray(series[parent], dtype=np.float64)
        gap = float(np.max(np.abs(top - expected)))
        if relative:
            gap /= max(1.0, float(np.max(np.abs(top))))
        worst = max(worst, gap)
    return worst
