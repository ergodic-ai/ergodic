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

    from ..graph import MixedGraph

__all__ = ["ForecastResult"]


@dataclass(frozen=True)
class ForecastResult:
    """Forecast draws for one or more series.

    ``series`` maps a node name to its draws, shape ``(S, H)``: ``S`` sample
    paths of length ``horizon``. Every node carries the same ``S`` and the same
    ``H``. ``kind`` records where the draws came from (``"draws"`` for a sampled
    forecast). ``start`` is the label of the first forecast step when the data
    made it computable, else ``None``.

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
    diagnostics: Mapping[str, float] | None = None
    reconciliation: str | None = None
    coherent: bool | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not self.series:
            raise ValueError("a forecast needs at least one series")
        cleaned: dict[str, npt.NDArray[np.float64]] = {}
        n_draws: int | None = None
        for name, draws in self.series.items():
            arr = np.asarray(draws, dtype=np.float64)
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
