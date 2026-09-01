# SPDX-License-Identifier: Apache-2.0
"""Control-baseline residual evidence for aligned multichannel days.

This module is deliberately source agnostic and NumPy only.  A baseline is
fitted from six ordered control days, then a query is compared with the
nearest daily-centered control prototype.  Evidence is the strongest mean
residual over a fixed, edge-eligible duration scan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ._signed_evidence import (
    SIGNED_EVIDENCE_DURATIONS,
    _candidate_is_better,
    _finite_float,
    _finite_tuple,
)
from ._signed_profile import SIGNED_PROFILE_CHANNELS, _robust_diff_scales, _validate_channels

__all__ = [
    "ControlBaselineModel",
    "ControlResidualEvidence",
    "fit_control_baseline",
    "score_control_residual_evidence",
]

_DAY_LENGTH = 1_440
_CONTROL_COUNT = 6
_FORMULA_VERSION = "control-baseline-residual/v1"
_FLOAT_TOLERANCE = 1e-12


def _validate_durations(durations: object, *, context: str) -> tuple[int, ...]:
    if not isinstance(durations, tuple):
        raise TypeError(f"{context} durations must be an immutable tuple")
    if durations != SIGNED_EVIDENCE_DURATIONS:
        raise ValueError(f"{context} durations must equal SIGNED_EVIDENCE_DURATIONS")
    return durations


def _validate_formula_version(value: object, *, context: str) -> str:
    if value != _FORMULA_VERSION:
        raise ValueError(f"{context} formula_version must equal {_FORMULA_VERSION!r}")
    return _FORMULA_VERSION


def _immutable_prototypes(value: object, channels: tuple[str, ...], *, context: str) -> np.ndarray:
    if not isinstance(value, tuple):
        raise TypeError(f"{context} centered_prototypes must be immutable nested tuples")
    if len(value) != _CONTROL_COUNT:
        raise ValueError(f"{context} centered_prototypes must contain exactly six prototypes")
    if any(not isinstance(prototype, tuple) for prototype in value):
        raise TypeError(f"{context} centered_prototypes must be immutable nested tuples")
    if any(
        len(prototype) != _DAY_LENGTH or any(not isinstance(row, tuple) for row in prototype)
        for prototype in value
    ):
        raise ValueError(
            f"{context} centered_prototypes must contain six 1,440-row immutable prototypes"
        )
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} centered_prototypes must contain finite real numbers") from exc
    expected_shape = (_CONTROL_COUNT, _DAY_LENGTH, len(channels))
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{context} centered_prototypes must have shape {expected_shape} and be finite"
        )
    return array


def _prototype_tuples(values: np.ndarray) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(tuple(float(value) for value in row) for row in prototype) for prototype in values
    )


def _copy_day(
    channels: Mapping[str, np.ndarray], channel_order: tuple[str, ...], *, context: str
) -> np.ndarray:
    if not isinstance(channels, Mapping):
        raise TypeError(f"{context} must be a mapping")
    if set(channels) != set(channel_order) or len(channels) != len(channel_order):
        if channel_order == SIGNED_PROFILE_CHANNELS:
            raise ValueError(
                f"{context} must contain exactly the six public signed-profile channels"
            )
        raise ValueError(f"{context} must contain exactly the channels in channel_order")

    arrays: list[np.ndarray] = []
    for name in channel_order:
        try:
            values = np.array(channels[name], dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} channel {name!r} must be a numeric 1-D array") from exc
        if values.ndim != 1:
            raise ValueError(f"{context} channel {name!r} must be a 1-D array")
        if values.shape != (_DAY_LENGTH,):
            raise ValueError(f"{context} channel {name!r} must contain exactly 1,440 samples")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{context} channel {name!r} contains non-finite values")
        arrays.append(values)
    return np.column_stack(arrays)


def _prototype_digest(prototype: np.ndarray) -> str:
    """Return a platform-stable digest of one numeric prototype."""
    shape = np.asarray(prototype.shape, dtype="<u8").tobytes(order="C")
    values = np.asarray(prototype, dtype="<f8", order="C").tobytes(order="C")
    return sha256(shape + values).hexdigest()


@dataclass(frozen=True)
class ControlBaselineModel:
    """Six ordered, daily-centered normal-day prototypes and fixed scales."""

    channels: tuple[str, ...]
    durations: tuple[int, ...]
    centered_prototypes: tuple[tuple[tuple[float, ...], ...], ...]
    scales: tuple[float, ...]
    formula_version: str

    def __post_init__(self) -> None:
        channels = _validate_channels(self.channels, context="baseline")
        _validate_durations(self.durations, context="baseline")
        _immutable_prototypes(self.centered_prototypes, channels, context="baseline")
        _finite_tuple(self.scales, len(channels), context="baseline scales", positive=True)
        _validate_formula_version(self.formula_version, context="baseline")


@dataclass(frozen=True)
class ControlResidualEvidence:
    """Strongest edge-eligible residual episode against a control baseline."""

    channels: tuple[str, ...]
    durations: tuple[int, ...]
    score: float
    start_tick: int
    end_tick: int
    duration: int
    scales: tuple[float, ...]
    signed_deltas: tuple[float, ...]
    query_centers: tuple[float, ...]
    selected_prototype_position: int
    selected_prototype_digest: str
    prototype_loss: float
    formula_version: str

    def __post_init__(self) -> None:
        channels = _validate_channels(self.channels, context="residual evidence")
        _validate_durations(self.durations, context="residual evidence")
        for name, value in (
            ("start_tick", self.start_tick),
            ("end_tick", self.end_tick),
            ("duration", self.duration),
            ("selected_prototype_position", self.selected_prototype_position),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"residual evidence {name} must be an integer")
        if self.duration not in self.durations:
            raise ValueError("residual evidence duration must be one of the scanned durations")
        if self.start_tick < 0:
            raise ValueError("residual evidence start_tick must be nonnegative")
        if self.end_tick != self.start_tick + self.duration:
            raise ValueError("residual evidence end_tick must equal start_tick + duration")
        if self.end_tick > _DAY_LENGTH:
            raise ValueError("residual evidence interval must end by tick 1,440")
        if not 0 <= self.selected_prototype_position < _CONTROL_COUNT:
            raise ValueError("residual evidence selected_prototype_position must be in [0, 6)")
        if (
            not isinstance(self.selected_prototype_digest, str)
            or len(self.selected_prototype_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.selected_prototype_digest
            )
        ):
            raise ValueError(
                "residual evidence selected_prototype_digest must be a lowercase SHA-256 digest"
            )
        scales = _finite_tuple(
            self.scales, len(channels), context="residual evidence scales", positive=True
        )
        deltas = _finite_tuple(
            self.signed_deltas, len(channels), context="residual evidence signed_deltas"
        )
        _finite_tuple(self.query_centers, len(channels), context="residual evidence query_centers")
        score = _finite_float(self.score, context="residual evidence score", nonnegative=True)
        _finite_float(
            self.prototype_loss,
            context="residual evidence prototype_loss",
            nonnegative=True,
        )
        expected = float(np.linalg.norm(deltas / scales) / np.sqrt(len(channels)))
        if not np.isclose(score, expected, rtol=_FLOAT_TOLERANCE, atol=_FLOAT_TOLERANCE):
            raise ValueError("residual evidence score must equal the normalized signed-delta norm")
        _validate_formula_version(self.formula_version, context="residual evidence")


def fit_control_baseline(
    control_days: Sequence[Mapping[str, np.ndarray]],
    *,
    channel_order: tuple[str, ...] = SIGNED_PROFILE_CHANNELS,
) -> ControlBaselineModel:
    """Fit the frozen baseline from exactly six ordered 1,440-sample controls.

    The sequence order is retained as prototype order.  Each prototype is
    centered by its own full-day channel medians.  Per-control robust first-
    difference scales are calculated with the signed-profile scale routine,
    and the baseline scale is their channelwise median.
    """
    if isinstance(control_days, (str, bytes)) or not isinstance(control_days, Sequence):
        raise TypeError("control_days must be a sequence")
    if len(control_days) != _CONTROL_COUNT:
        raise ValueError("control_days must contain exactly six control days")
    accepted_order = _validate_channels(channel_order, context="channel_order")

    centered: list[np.ndarray] = []
    control_scales: list[np.ndarray] = []
    for index, day in enumerate(control_days):
        matrix = _copy_day(day, accepted_order, context=f"control_days[{index}]")
        daily_centers = np.median(matrix, axis=0)
        prototype = matrix - daily_centers[None, :]
        with np.errstate(over="ignore", invalid="ignore"):
            _, scales = _robust_diff_scales(matrix)
        if not np.all(np.isfinite(prototype)) or not np.all(np.isfinite(scales)):
            raise ValueError("control baseline calculation produced non-finite values")
        if np.any(scales <= 0.0):
            raise ValueError("control baseline calculation produced non-positive scales")
        centered.append(prototype)
        control_scales.append(scales)

    prototypes = np.stack(centered)
    scales = np.median(np.stack(control_scales), axis=0)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("control baseline calculation produced invalid channel scales")
    return ControlBaselineModel(
        channels=accepted_order,
        durations=SIGNED_EVIDENCE_DURATIONS,
        centered_prototypes=_prototype_tuples(prototypes),
        scales=tuple(float(value) for value in scales),
        formula_version=_FORMULA_VERSION,
    )


def score_control_residual_evidence(
    channels: Mapping[str, np.ndarray], baseline: ControlBaselineModel
) -> ControlResidualEvidence:
    """Score one aligned query against its nearest frozen control prototype.

    Prototype ties select the earlier input position.  Every interval
    ``[start, start + duration)`` within the 1,440-sample day is eligible;
    exact score ties choose the shorter duration and then the earlier start.
    """
    if not isinstance(baseline, ControlBaselineModel):
        raise TypeError("baseline must be a ControlBaselineModel")
    ControlBaselineModel.__post_init__(baseline)
    matrix = _copy_day(channels, baseline.channels, context="channels")
    prototypes = np.asarray(baseline.centered_prototypes, dtype=np.float64)
    scales = np.asarray(baseline.scales, dtype=np.float64)

    query_centers = np.median(matrix, axis=0)
    centered_query = matrix - query_centers[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        standardized = (centered_query[None, :, :] - prototypes) / scales[None, None, :]
        per_tick_loss = np.sqrt(np.mean(standardized**2, axis=2))
        prototype_losses = np.median(per_tick_loss, axis=1)
    if not np.all(np.isfinite(prototype_losses)):
        raise ValueError("control prototype selection produced a non-finite loss")
    selected = int(np.argmin(prototype_losses))
    residual = centered_query - prototypes[selected]

    best_key: tuple[float, int, int] | None = None
    best_deltas: np.ndarray | None = None
    for duration in baseline.durations:
        windows = np.lib.stride_tricks.sliding_window_view(residual, window_shape=duration, axis=0)
        deltas = np.mean(windows, axis=-1)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.linalg.norm(deltas / scales, axis=1) / np.sqrt(len(baseline.channels))
        if not np.all(np.isfinite(scores)):
            raise ValueError("control residual evidence calculation produced a non-finite score")
        index = int(np.argmax(scores))
        candidate = (float(scores[index]), duration, index)
        if _candidate_is_better(candidate, best_key):
            best_key = candidate
            best_deltas = deltas[index].copy()

    assert best_key is not None and best_deltas is not None
    score, duration, start = best_key
    return ControlResidualEvidence(
        channels=baseline.channels,
        durations=baseline.durations,
        score=score,
        start_tick=start,
        end_tick=start + duration,
        duration=duration,
        scales=baseline.scales,
        signed_deltas=tuple(float(value) for value in best_deltas),
        query_centers=tuple(float(value) for value in query_centers),
        selected_prototype_position=selected,
        selected_prototype_digest=_prototype_digest(prototypes[selected]),
        prototype_loss=float(prototype_losses[selected]),
        formula_version=baseline.formula_version,
    )
