# SPDX-License-Identifier: Apache-2.0
"""Source-free evidence for finite multichannel disturbance episodes.

The evidence score compares each candidate episode with equally long shoulders
on both sides.  It deliberately knows nothing about source templates or labels;
calibration is a separate finite-control rank calculation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._signed_profile import (
    SIGNED_PROFILE_CHANNELS,
    _robust_diff_scales,
    _validate_channels,
)

if TYPE_CHECKING:
    from ._control_residual import ControlResidualEvidence

__all__ = [
    "SIGNED_EVIDENCE_DURATIONS",
    "NullEvidenceDecision",
    "NullEvidenceGate",
    "SignedDisturbanceEvidence",
    "apply_null_evidence_gate",
    "calibrate_null_evidence_gate",
    "score_signed_disturbance_evidence",
]

SIGNED_EVIDENCE_DURATIONS = (30, 60, 120, 240)
_FLOAT_TOLERANCE = 1e-12


def _finite_float(
    value: object,
    *,
    context: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if positive and nonnegative:
        raise ValueError("a number cannot require both positive and nonnegative validation")
    qualifier = "positive " if positive else "nonnegative " if nonnegative else ""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite {qualifier}number") from exc
    if not np.isfinite(number) or (positive and number <= 0.0) or (nonnegative and number < 0.0):
        raise ValueError(f"{context} must be a finite {qualifier}number")
    return number


def _finite_tuple(
    values: object,
    length: int,
    *,
    context: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    if positive and nonnegative:
        raise ValueError("values cannot require both positive and nonnegative validation")
    if not isinstance(values, tuple):
        raise TypeError(f"{context} must be an immutable tuple")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain finite real numbers") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} must contain {length} finite real numbers")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{context} must contain positive numbers")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{context} must contain nonnegative numbers")
    return array


@dataclass(frozen=True)
class SignedDisturbanceEvidence:
    """Strongest source-free inside-versus-shoulders episode contrast."""

    channels: tuple[str, ...]
    durations: tuple[int, ...]
    score: float
    start_tick: int
    end_tick: int
    duration: int
    scales: tuple[float, ...]
    signed_deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        channels = _validate_channels(self.channels, context="evidence")
        if self.durations != SIGNED_EVIDENCE_DURATIONS:
            raise ValueError("evidence durations must equal SIGNED_EVIDENCE_DURATIONS")
        for name, value in (
            ("start_tick", self.start_tick),
            ("end_tick", self.end_tick),
            ("duration", self.duration),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"evidence {name} must be an integer")
        if self.duration not in self.durations:
            raise ValueError("evidence duration must be one of the scanned durations")
        if self.start_tick < self.duration:
            raise ValueError("evidence interval must have a full left shoulder")
        if self.end_tick != self.start_tick + self.duration:
            raise ValueError("evidence end_tick must equal start_tick + duration")
        scales = _finite_tuple(self.scales, len(channels), context="evidence scales", positive=True)
        deltas = _finite_tuple(self.signed_deltas, len(channels), context="evidence signed_deltas")
        score = _finite_float(self.score, context="evidence score", nonnegative=True)
        expected = float(np.linalg.norm(deltas / scales) / np.sqrt(len(channels)))
        if not np.isclose(score, expected, rtol=_FLOAT_TOLERANCE, atol=_FLOAT_TOLERANCE):
            raise ValueError("evidence score must equal the normalized signed-delta norm")


@dataclass(frozen=True)
class NullEvidenceGate:
    """Order-invariant empirical null calibration from control evidence."""

    calibration_scores: tuple[float, ...]
    threshold: float
    calibration_count: int
    alpha: float

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_scores, tuple):
            raise TypeError("gate calibration_scores must be an immutable tuple")
        if not self.calibration_scores:
            raise ValueError("gate calibration_scores must be non-empty")
        scores = _finite_tuple(
            self.calibration_scores,
            len(self.calibration_scores),
            context="gate calibration_scores",
            nonnegative=True,
        )
        if tuple(float(value) for value in np.sort(scores)) != self.calibration_scores:
            raise ValueError("gate calibration_scores must be sorted")
        if isinstance(self.calibration_count, bool) or not isinstance(self.calibration_count, int):
            raise TypeError("gate calibration_count must be an integer")
        if self.calibration_count != len(scores):
            raise ValueError("gate calibration_count must equal the number of scores")
        threshold = _finite_float(self.threshold, context="gate threshold", nonnegative=True)
        if threshold != float(scores[-1]):
            raise ValueError("gate threshold must equal the largest calibration score")
        alpha = _finite_float(self.alpha, context="gate alpha", positive=True)
        expected_alpha = 1.0 / (self.calibration_count + 1)
        if alpha != expected_alpha:
            raise ValueError("gate alpha must equal 1 / (calibration_count + 1)")


@dataclass(frozen=True)
class NullEvidenceDecision:
    """Conservative empirical-null rank decision for one evidence value."""

    supported: bool
    p_null: float
    rank_numerator: int
    rank_denominator: int
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise TypeError("decision supported must be a boolean")
        for name, value in (
            ("rank_numerator", self.rank_numerator),
            ("rank_denominator", self.rank_denominator),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"decision {name} must be an integer")
        if self.rank_denominator < 2:
            raise ValueError("decision rank_denominator must be at least two")
        if not 1 <= self.rank_numerator <= self.rank_denominator:
            raise ValueError("decision rank_numerator must be within the denominator")
        p_null = _finite_float(self.p_null, context="decision p_null", positive=True)
        if p_null != self.rank_numerator / self.rank_denominator:
            raise ValueError("decision p_null must equal rank_numerator / rank_denominator")
        _finite_float(self.threshold, context="decision threshold", nonnegative=True)
        expected_supported = self.rank_numerator == 1
        if self.supported != expected_supported:
            raise ValueError("decision supported must equal the conservative rank verdict")
        expected_reason = (
            "evidence_exceeds_null_threshold"
            if self.supported
            else "evidence_not_above_null_threshold"
        )
        if self.reason != expected_reason:
            raise ValueError("decision reason does not match supported")


def _copy_channel_matrix(
    channels: Mapping[str, np.ndarray], channel_order: tuple[str, ...]
) -> np.ndarray:
    if not isinstance(channels, Mapping):
        raise TypeError("channels must be a mapping")
    accepted_order = _validate_channels(channel_order, context="channel_order")
    if set(channels) != set(accepted_order) or len(channels) != len(accepted_order):
        if accepted_order == SIGNED_PROFILE_CHANNELS:
            raise ValueError("channels must contain exactly the six public signed-profile channels")
        raise ValueError("channels must contain exactly the channels in channel_order")

    arrays: list[np.ndarray] = []
    length: int | None = None
    for name in accepted_order:
        try:
            values = np.array(channels[name], dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"channel {name!r} must be a numeric 1-D array") from exc
        if values.ndim != 1:
            raise ValueError(f"channel {name!r} must be a 1-D array")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"channel {name!r} contains non-finite values")
        if length is None:
            length = values.size
        elif values.size != length:
            raise ValueError("all channels must have equal lengths")
        arrays.append(values)
    minimum_length = 3 * max(SIGNED_EVIDENCE_DURATIONS)
    if length is None or length < minimum_length:
        raise ValueError(f"channels need at least {minimum_length} samples")
    return np.column_stack(arrays)


def _candidate_is_better(
    candidate: tuple[float, int, int], incumbent: tuple[float, int, int] | None
) -> bool:
    """Compare ``(score, duration, start)`` using the frozen exact tie rules."""
    if incumbent is None:
        return True
    score, duration, start = candidate
    best_score, best_duration, best_start = incumbent
    return score > best_score or (
        score == best_score and (duration, start) < (best_duration, best_start)
    )


def score_signed_disturbance_evidence(
    channels: Mapping[str, np.ndarray],
    *,
    channel_order: tuple[str, ...] = SIGNED_PROFILE_CHANNELS,
) -> SignedDisturbanceEvidence:
    """Score the strongest fixed-duration episode against symmetric shoulders.

    Inputs are defensively copied and ordered explicitly.  Each channel is
    detrended by its median first difference, then scaled by the same robust,
    positive scale used by :func:`extract_signed_temporal_profile`.  Exact
    score ties choose the shorter duration and then the earlier start.

    At least 720 samples are required so every frozen duration is eligible.
    Completely flat or purely linear traces return zero evidence at the
    shortest duration and earliest legal start; zero is not affirmative
    support and still has to exceed the calibrated null maximum.
    """
    accepted_order = _validate_channels(channel_order, context="channel_order")
    matrix = _copy_channel_matrix(channels, accepted_order)
    diff_centers, scales = _robust_diff_scales(matrix)
    ticks = np.arange(matrix.shape[0], dtype=np.float64)[:, None]
    detrended = matrix - ticks * diff_centers[None, :]

    best_key: tuple[float, int, int] | None = None
    best_deltas: np.ndarray | None = None
    for duration in SIGNED_EVIDENCE_DURATIONS:
        starts = np.arange(duration, matrix.shape[0] - 2 * duration + 1)
        # Direct window reductions avoid subtracting two large cumulative sums,
        # which can erase a real episode near machine precision on a large
        # baseline.  ``sliding_window_view`` is a view, so this remains bounded
        # and vectorized while following the frozen mean formula literally.
        windows = np.lib.stride_tricks.sliding_window_view(detrended, window_shape=duration, axis=0)
        means = np.mean(windows, axis=-1)
        left = means[starts - duration]
        inside = means[starts]
        right = means[starts + duration]
        deltas = inside - 0.5 * (left + right)
        scores = np.linalg.norm(deltas / scales, axis=1) / np.sqrt(matrix.shape[1])
        index = int(np.argmax(scores))
        candidate = (float(scores[index]), duration, int(starts[index]))
        if _candidate_is_better(candidate, best_key):
            best_key = candidate
            best_deltas = deltas[index].copy()

    assert best_key is not None and best_deltas is not None
    score, duration, start = best_key
    if not np.isfinite(score):
        raise ValueError("disturbance evidence calculation produced a non-finite score")
    return SignedDisturbanceEvidence(
        channels=accepted_order,
        durations=SIGNED_EVIDENCE_DURATIONS,
        score=score,
        start_tick=start,
        end_tick=start + duration,
        duration=duration,
        scales=tuple(float(value) for value in scales),
        signed_deltas=tuple(float(value) for value in best_deltas),
    )


def calibrate_null_evidence_gate(
    calibration_evidence: Sequence[SignedDisturbanceEvidence | ControlResidualEvidence],
) -> NullEvidenceGate:
    """Calibrate the strict empirical-null maximum from non-empty controls."""
    if isinstance(calibration_evidence, (str, bytes)) or not isinstance(
        calibration_evidence, Sequence
    ):
        raise TypeError("calibration_evidence must be a sequence")
    if not calibration_evidence:
        raise ValueError("calibration_evidence must contain at least one value")
    scores: list[float] = []
    for index, evidence in enumerate(calibration_evidence):
        _validate_gate_evidence(evidence, context=f"calibration_evidence[{index}]")
        scores.append(float(evidence.score))
    scores.sort()
    count = len(scores)
    return NullEvidenceGate(
        calibration_scores=tuple(scores),
        threshold=scores[-1],
        calibration_count=count,
        alpha=1.0 / (count + 1),
    )


def apply_null_evidence_gate(
    evidence: SignedDisturbanceEvidence | ControlResidualEvidence, gate: NullEvidenceGate
) -> NullEvidenceDecision:
    """Apply a conservative ``>=`` null rank; threshold ties fail closed."""
    _validate_gate_evidence(evidence, context="evidence")
    if not isinstance(gate, NullEvidenceGate):
        raise TypeError("gate must be a NullEvidenceGate")
    NullEvidenceGate.__post_init__(gate)
    numerator = 1 + sum(score >= evidence.score for score in gate.calibration_scores)
    denominator = gate.calibration_count + 1
    supported = numerator == 1
    reason = "evidence_exceeds_null_threshold" if supported else "evidence_not_above_null_threshold"
    return NullEvidenceDecision(
        supported=supported,
        p_null=numerator / denominator,
        rank_numerator=numerator,
        rank_denominator=denominator,
        threshold=gate.threshold,
        reason=reason,
    )


def _validate_gate_evidence(
    evidence: object, *, context: str
) -> SignedDisturbanceEvidence | ControlResidualEvidence:
    """Validate either supported evidence representation without an import cycle."""
    from ._control_residual import ControlResidualEvidence

    if isinstance(evidence, SignedDisturbanceEvidence):
        SignedDisturbanceEvidence.__post_init__(evidence)
        return evidence
    if isinstance(evidence, ControlResidualEvidence):
        ControlResidualEvidence.__post_init__(evidence)
        return evidence
    raise TypeError(f"{context} must be a SignedDisturbanceEvidence or ControlResidualEvidence")
