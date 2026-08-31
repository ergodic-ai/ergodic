# SPDX-License-Identifier: Apache-2.0
"""Observable profiles and conservative matching for recovered latent regimes.

A profile summarizes which likelihood-declared delta blocks carry standardized
posterior evidence.  The matching vector deliberately discards delta signs:
hidden-state labels are arbitrary, whereas the descriptive intervals retain
the engine's sign-aligned posterior coordinates for reporting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._result import LatentResult

__all__ = [
    "ObservableProfile",
    "ProfileInterval",
    "ProfileMatch",
    "match_profile",
    "profile",
]

_NORM_TOLERANCE = 1e-12
_SCALE_FLOOR = 1e-12


@dataclass(frozen=True)
class ProfileInterval:
    """One block's sign-aligned descriptive posterior interval."""

    block: str
    low: float
    mean: float
    high: float

    def __post_init__(self) -> None:
        if not isinstance(self.block, str) or not self.block.strip():
            raise ValueError("profile interval block must be a non-empty string")
        values = np.asarray((self.low, self.mean, self.high), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"profile interval {self.block!r} must contain finite values")
        if self.low > self.high:
            raise ValueError(
                f"profile interval {self.block!r} must satisfy low <= high, "
                f"got {self.low} > {self.high}"
            )


@dataclass(frozen=True)
class ObservableProfile:
    """An immutable, unit-length profile over named observable blocks."""

    blocks: tuple[str, ...]
    weights: tuple[float, ...]
    intervals: tuple[ProfileInterval, ...]

    def __post_init__(self) -> None:
        _validate_profile(self)


@dataclass(frozen=True)
class ProfileMatch:
    """The best library alignment, including an explicit abstention reason."""

    status: Literal["identified", "unattributed"]
    source: str | None
    similarity: float
    runner_up_similarity: float | None
    margin: float | None
    threshold: float
    min_margin: float
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"identified", "unattributed"}:
            raise ValueError("profile match status must be 'identified' or 'unattributed'")
        similarity = _validate_probability(self.similarity, name="similarity")
        threshold = _validate_probability(self.threshold, name="threshold")
        min_margin = _validate_probability(self.min_margin, name="min_margin")

        if self.runner_up_similarity is None:
            if self.margin is not None:
                raise ValueError("profile match margin must be null when runner-up is null")
            runner_up = None
            margin = None
        else:
            runner_up = _validate_probability(
                self.runner_up_similarity, name="runner_up_similarity"
            )
            if self.margin is None:
                raise ValueError("profile match margin is required when runner-up is present")
            margin = _validate_probability(self.margin, name="margin")
            expected_margin = similarity - runner_up
            if expected_margin < 0.0 or abs(margin - expected_margin) > _NORM_TOLERANCE:
                raise ValueError(
                    "profile match margin must be non-negative and equal "
                    "similarity - runner_up_similarity"
                )

        if self.status == "identified":
            if not isinstance(self.source, str) or not self.source.strip():
                raise ValueError("identified profile match requires a non-empty source")
            if self.reason != "threshold_and_margin_passed":
                raise ValueError(
                    "identified profile match reason must be 'threshold_and_margin_passed'"
                )
            if runner_up is None or margin is None:
                raise ValueError("identified profile match requires a runner-up and margin")
            if similarity < threshold or margin < min_margin:
                raise ValueError("identified profile match must pass threshold and minimum margin")
            return

        if self.source is not None:
            raise ValueError("unattributed profile match source must be null")
        if self.reason == "below_similarity_threshold":
            if similarity >= threshold:
                raise ValueError("below-threshold reason requires similarity < threshold")
        elif self.reason == "no_runner_up_for_margin":
            if similarity < threshold or runner_up is not None:
                raise ValueError(
                    "no-runner-up reason requires threshold passage and a null runner-up"
                )
        elif self.reason == "insufficient_runner_up_margin":
            if similarity < threshold or runner_up is None or margin is None:
                raise ValueError(
                    "insufficient-margin reason requires threshold passage and a runner-up"
                )
            if margin >= min_margin:
                raise ValueError("insufficient-margin reason requires margin < min_margin")
        else:
            raise ValueError("unattributed profile match has an unknown reason")


def _validate_blocks(blocks: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(blocks, tuple):
        raise TypeError(f"{context} blocks must be an immutable tuple")
    if not blocks:
        raise ValueError(f"{context} must contain at least one block")
    for block in blocks:
        if not isinstance(block, str) or not block.strip():
            raise ValueError(f"{context} block names must be non-empty strings")
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"{context} block names must be unique")
    return blocks


def _validate_profile(value: object, *, context: str = "profile") -> ObservableProfile:
    if not isinstance(value, ObservableProfile):
        raise TypeError(f"{context} must be an ObservableProfile")

    blocks = _validate_blocks(value.blocks, context=context)
    if not isinstance(value.weights, tuple):
        raise TypeError(f"{context} weights must be an immutable tuple")
    if not isinstance(value.intervals, tuple):
        raise TypeError(f"{context} intervals must be an immutable tuple")
    if len(value.weights) != len(blocks) or len(value.intervals) != len(blocks):
        raise ValueError(f"{context} blocks, weights, and intervals must have equal lengths")

    try:
        weights = np.asarray(value.weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} weights must be real numbers") from exc
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError(f"{context} weights must be finite real numbers")
    if np.any(weights < 0.0):
        raise ValueError(f"{context} weights must be non-negative")
    norm = float(np.linalg.norm(weights))
    if not np.isfinite(norm) or abs(norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(
            f"{context} weight norm must be within 1 +/- {_NORM_TOLERANCE:g}, got {norm}"
        )

    for expected_block, interval in zip(blocks, value.intervals, strict=True):
        if not isinstance(interval, ProfileInterval):
            raise TypeError(f"{context} intervals must contain ProfileInterval values")
        # Revalidate rather than trusting construction: frozen dataclasses can
        # still arrive from untrusted deserializers or object-level mutation.
        ProfileInterval.__post_init__(interval)
        if interval.block != expected_block:
            raise ValueError(
                f"{context} interval blocks must equal blocks in order: "
                f"expected {expected_block!r}, got {interval.block!r}"
            )
    return value


def _unit_interval(block: str, values: tuple[float, float, float]) -> ProfileInterval:
    return ProfileInterval(block=block, low=values[0], mean=values[1], high=values[2])


def profile(result: LatentResult) -> ObservableProfile:
    """Extract a label-invariant standardized profile from one recovery.

    For each likelihood-declared block, the weight before normalization is
    ``abs(mean(draws)) / max(std(draws, ddof=1), 1e-12)``.  The intervals are
    the result's 94% sign-aligned descriptive posterior intervals and do not
    participate in matching.
    """
    if not isinstance(result, LatentResult):
        raise TypeError(f"profile() reads a LatentResult, got {type(result).__name__}")

    blocks = _validate_blocks(result.blocks, context="LatentResult")
    standardized: list[float] = []
    for block in blocks:
        draws = np.asarray(result.delta_draws[block], dtype=np.float64)
        if draws.ndim != 1 or draws.size < 2:
            raise ValueError(f"delta block {block!r} requires at least two draws")
        if not np.all(np.isfinite(draws)):
            raise ValueError(f"delta block {block!r} contains non-finite draws")
        mean = float(np.mean(draws))
        scale = max(float(np.std(draws, ddof=1)), _SCALE_FLOOR)
        score = abs(mean) / scale
        if not np.isfinite(score):
            raise ValueError(f"delta block {block!r} produced a non-finite standardized weight")
        standardized.append(score)

    raw = np.asarray(standardized, dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("profile has zero or non-finite standardized weight norm")
    weights = tuple(float(value) for value in raw / norm)

    descriptive = result.deltas(level=0.94)
    intervals = tuple(_unit_interval(block, descriptive[block]) for block in blocks)
    return ObservableProfile(blocks=blocks, weights=weights, intervals=intervals)


def _validate_probability(value: float, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return number


def _aligned_similarity(left: ObservableProfile, right: ObservableProfile) -> float:
    union = tuple(sorted(set(left.blocks) | set(right.blocks)))
    left_by_block = dict(zip(left.blocks, left.weights, strict=True))
    right_by_block = dict(zip(right.blocks, right.weights, strict=True))
    left_vector = np.asarray([left_by_block.get(block, 0.0) for block in union])
    right_vector = np.asarray([right_by_block.get(block, 0.0) for block in union])
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    similarity = float(np.dot(left_vector, right_vector) / denominator)
    # Non-negative unit vectors have cosine in [0, 1]; clip only floating-point
    # roundoff so decision thresholds receive a stable value.
    return float(np.clip(similarity, 0.0, 1.0))


def match_profile(
    query: ObservableProfile,
    library: Mapping[str, ObservableProfile],
    *,
    threshold: float = 0.80,
    min_margin: float = 0.15,
) -> ProfileMatch:
    """Match a profile to a fixed library, or conservatively abstain.

    Identification requires both the cosine threshold and a distinct runner-up
    separated by ``min_margin``.  Similarities align on the sorted union of
    block names, with missing blocks filled by zero.
    """
    accepted_threshold = _validate_probability(threshold, name="threshold")
    accepted_margin = _validate_probability(min_margin, name="min_margin")
    _validate_profile(query, context="query profile")
    if not isinstance(library, Mapping):
        raise TypeError("library must be a mapping of source names to profiles")
    if not library:
        raise ValueError("library must contain at least one profile")

    candidates: list[tuple[str, float]] = []
    for source, candidate in library.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("library source names must be non-empty strings")
        _validate_profile(candidate, context=f"library profile {source!r}")
        candidates.append((source, _aligned_similarity(query, candidate)))
    candidates.sort(key=lambda item: (-item[1], item[0]))

    best_source, best_similarity = candidates[0]
    runner_up_similarity = candidates[1][1] if len(candidates) > 1 else None
    margin = best_similarity - runner_up_similarity if runner_up_similarity is not None else None

    if best_similarity < accepted_threshold:
        status: Literal["identified", "unattributed"] = "unattributed"
        source: str | None = None
        reason = "below_similarity_threshold"
    elif runner_up_similarity is None:
        status = "unattributed"
        source = None
        reason = "no_runner_up_for_margin"
    elif margin is not None and margin < accepted_margin:
        status = "unattributed"
        source = None
        reason = "insufficient_runner_up_margin"
    else:
        status = "identified"
        source = best_source
        reason = "threshold_and_margin_passed"

    return ProfileMatch(
        status=status,
        source=source,
        similarity=best_similarity,
        runner_up_similarity=runner_up_similarity,
        margin=margin,
        threshold=accepted_threshold,
        min_margin=accepted_margin,
        reason=reason,
    )
