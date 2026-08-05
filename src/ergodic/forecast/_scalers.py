# SPDX-License-Identifier: Apache-2.0
"""Per-node scalers: put a series on a scale the sampler can work with.

A revenue series in the millions and a rate series near zero cannot share one
set of priors. Each node carries its own scaler, so the mechanism writes its
priors on a standardized scale once and every node reads back in its own units.

Four of them. :class:`RatioScaler` divides by the median of the positive values,
which centers an always-positive series near one and keeps multiplicative
structure intact. :class:`ZScoreScaler` centers and standardizes, which is the
right choice for something that can go negative. :class:`IdentityScaler` leaves
data that already sits on a natural scale alone. :class:`LogScaler` handles a
heavily skewed positive series.

They are immutable, like the rest of the package: :meth:`Scaler.fit` returns a
new fitted scaler rather than changing the one you called it on. An unfitted
scaler is the identity, so nothing silently half-transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import Self

    import numpy.typing as npt

__all__ = ["IdentityScaler", "LogScaler", "RatioScaler", "Scaler", "ZScoreScaler"]


def _values(x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class Scaler:
    """The scaler contract: ``fit`` returns a fitted copy, ``transform`` and ``inverse`` undo."""

    def fit(self, x: npt.NDArray[Any] | Any) -> Self:
        """Learn the scale from ``x`` and return the fitted scaler."""
        return self

    def transform(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        """Map data onto the model scale."""
        return _values(x)

    def inverse(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        """Map model-scale values back to the data's own units."""
        return _values(x)


@dataclass(frozen=True)
class RatioScaler(Scaler):
    """Divide by the median of the positive values, so the series sits near one.

    Zeros and negatives are ignored when the median is taken, so a series that
    starts with a run of zeros does not end up with a tiny divisor. ``floor``
    clips the transformed values from below, which keeps a positive-only
    likelihood away from zero; a value at or below ``floor * median`` therefore
    does not survive a round trip.

    Two magnitude boundaries worth knowing. The guard against a zero divisor is
    absolute (a fitted median at or below 1e-8 falls back to 1.0), so a series
    genuinely living at that scale is left unscaled rather than blown up. And
    the floor is relative to the median, so a series with a huge dynamic range
    loses its smallest values to the floor.
    """

    floor: float = 0.01
    median: float = 1.0

    def fit(self, x: npt.NDArray[Any] | Any) -> Self:
        values = _values(x)
        positive = values[np.isfinite(values) & (values > 0)]
        found = float(np.median(positive)) if positive.size else 1.0
        return replace(self, median=found if abs(found) > 1e-8 else 1.0)

    def transform(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        return np.maximum(_values(x) / self.median, self.floor)

    def inverse(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        return _values(x) * self.median


@dataclass(frozen=True)
class ZScoreScaler(Scaler):
    """Center and standardize. The default for a series that can go negative."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, x: npt.NDArray[Any] | Any) -> Self:
        values = _values(x)
        spread = float(np.nanstd(values))
        return replace(self, mean=float(np.nanmean(values)), std=spread if spread > 1e-8 else 1.0)

    def transform(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        return (_values(x) - self.mean) / self.std

    def inverse(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        return _values(x) * self.std + self.mean


@dataclass(frozen=True)
class IdentityScaler(Scaler):
    """Leave the data alone. For anything already on a natural scale, like an index."""


@dataclass(frozen=True)
class LogScaler(Scaler):
    """``log(x + offset)``, for a positive series with a long right tail.

    ``offset`` keeps the log away from zero. A value at or below ``-offset``
    has no log, so :meth:`transform` refuses it rather than quietly returning
    negative infinity or NaN.

    The round trip is exact to floating point only where the log is well
    conditioned. Far below the offset, ``exp(log(x + offset)) - offset`` loses
    the small difference between two large numbers, so relative accuracy decays
    to roughly 1e-7 rather than the 1e-12 the other scalers hold.
    """

    offset: float = 1.0

    def transform(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        shifted = _values(x) + self.offset
        if bool(np.any(shifted <= 0)):
            smallest = float(np.min(shifted)) - self.offset
            raise ValueError(
                f"LogScaler(offset={self.offset}) needs values above {-self.offset}; "
                f"the smallest is {smallest}"
            )
        return np.asarray(np.log(shifted), dtype=np.float64)

    def inverse(self, x: npt.NDArray[Any] | Any) -> npt.NDArray[np.float64]:
        return np.asarray(np.exp(_values(x)) - self.offset, dtype=np.float64)
