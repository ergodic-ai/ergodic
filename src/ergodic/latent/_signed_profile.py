# SPDX-License-Identifier: Apache-2.0
"""Public-signal temporal profiles for deterministic source matching.

This module is deliberately NumPy-only.  It reads six named sensor arrays,
finds their strongest joint public change, and preserves the relative signs
of short- and medium-horizon contrasts.  A global sign is treated as an
orientation rather than source identity: entering and leaving the same regime
can therefore match one template without erasing cross-channel direction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "SIGNED_PROFILE_CHANNELS",
    "SignedProfileMatch",
    "SignedProfileTemplate",
    "SignedTemporalProfile",
    "build_signed_template",
    "extract_signed_temporal_profile",
    "match_signed_profile",
]

SIGNED_PROFILE_CHANNELS = (
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "pm25_ug_m3",
    "voc_ppb",
    "noise_db",
)
_SHORT_HORIZON = 5
_MEDIUM_HORIZON = 30
_SCAN_BURN_IN = 2 * _MEDIUM_HORIZON
_NORM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SignedTemporalProfile:
    """One inferred public change and its unit-length signed feature vector."""

    channels: tuple[str, ...]
    change_tick: int
    scales: tuple[float, ...]
    contrasts_5: tuple[float, ...]
    contrasts_30: tuple[float, ...]
    vector: tuple[float, ...]
    magnitude: float

    def __post_init__(self) -> None:
        _validate_profile(self)


@dataclass(frozen=True)
class SignedProfileTemplate:
    """A deterministic, globally sign-invariant source template."""

    source: str
    channels: tuple[str, ...]
    vector: tuple[float, ...]
    n_profiles: int

    def __post_init__(self) -> None:
        _validate_source(self.source)
        _validate_channels(self.channels, context="template")
        _validate_unit_vector(self.vector, 2 * len(self.channels), context="template")
        if isinstance(self.n_profiles, bool) or not isinstance(self.n_profiles, int):
            raise TypeError("template n_profiles must be an integer")
        if self.n_profiles < 1:
            raise ValueError("template n_profiles must be positive")


@dataclass(frozen=True)
class SignedProfileMatch:
    """Best sign-invariant cosine match, including explicit abstention."""

    status: Literal["identified", "unattributed"]
    source: str | None
    similarity: float
    runner_up_similarity: float | None
    margin: float | None
    orientation: Literal[-1, 1]
    threshold: float
    min_margin: float
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"identified", "unattributed"}:
            raise ValueError("signed profile match status must be 'identified' or 'unattributed'")
        similarity = _probability(self.similarity, "similarity")
        threshold = _probability(self.threshold, "threshold")
        min_margin = _probability(self.min_margin, "min_margin")
        if self.orientation not in {-1, 1}:
            raise ValueError("signed profile match orientation must be -1 or 1")
        if self.runner_up_similarity is None:
            if self.margin is not None:
                raise ValueError("margin must be null when runner-up similarity is null")
            runner_up = None
        else:
            runner_up = _probability(self.runner_up_similarity, "runner_up_similarity")
            if self.margin is None:
                raise ValueError("margin is required when runner-up similarity is present")
            margin = _probability(self.margin, "margin")
            if abs(margin - (similarity - runner_up)) > _NORM_TOLERANCE:
                raise ValueError("margin must equal similarity - runner_up_similarity")
        if self.status == "identified":
            _validate_source(self.source)
            if runner_up is None or similarity < threshold or self.margin is None:
                raise ValueError("identified match must pass threshold and have a runner-up")
            if self.margin < min_margin:
                raise ValueError("identified match must pass minimum margin")
        elif self.source is not None:
            raise ValueError("unattributed match source must be null")


def _validate_source(source: object) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    return source


def _validate_channels(channels: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(channels, tuple):
        raise TypeError(f"{context} channels must be an immutable tuple")
    if channels != SIGNED_PROFILE_CHANNELS:
        raise ValueError(f"{context} channels must equal SIGNED_PROFILE_CHANNELS in order")
    return channels


def _as_finite_vector(values: object, length: int, *, context: str) -> np.ndarray:
    if not isinstance(values, tuple):
        raise TypeError(f"{context} must be an immutable tuple")
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain real numbers") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{context} must contain {length} finite real numbers")
    return vector


def _validate_unit_vector(values: object, length: int, *, context: str) -> np.ndarray:
    vector = _as_finite_vector(values, length, context=f"{context} vector")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > _NORM_TOLERANCE:
        raise ValueError(f"{context} vector must have unit norm")
    return vector


def _validate_profile(value: object, *, context: str = "profile") -> SignedTemporalProfile:
    if not isinstance(value, SignedTemporalProfile):
        raise TypeError(f"{context} must be a SignedTemporalProfile")
    channels = _validate_channels(value.channels, context=context)
    if isinstance(value.change_tick, bool) or not isinstance(value.change_tick, int):
        raise TypeError(f"{context} change_tick must be an integer")
    if value.change_tick < _MEDIUM_HORIZON:
        raise ValueError(f"{context} change_tick is too close to the start")
    scales = _as_finite_vector(value.scales, len(channels), context=f"{context} scales")
    if np.any(scales <= 0.0):
        raise ValueError(f"{context} scales must be positive")
    _as_finite_vector(value.contrasts_5, len(channels), context=f"{context} contrasts_5")
    _as_finite_vector(value.contrasts_30, len(channels), context=f"{context} contrasts_30")
    _validate_unit_vector(value.vector, 2 * len(channels), context=context)
    try:
        magnitude = float(value.magnitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} magnitude must be finite and positive") from exc
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError(f"{context} magnitude must be finite and positive")
    return value


def _probability(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return number


def _robust_diff_scales(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    differences = np.diff(values, axis=0)
    centers = np.median(differences, axis=0)
    deviations = np.abs(differences - centers)
    mad = 1.4826 * np.median(deviations, axis=0)
    scales = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        positive = deviations[:, column][deviations[:, column] > 0.0]
        resolution_floor = 0.1 * float(np.median(positive)) if positive.size else 0.0
        level_floor = np.sqrt(np.finfo(np.float64).eps) * max(
            1.0, float(np.median(np.abs(values[:, column])))
        )
        scales[column] = max(float(mad[column]), resolution_floor, level_floor)
    return centers, scales


def _window_means(cumulative: np.ndarray, ticks: np.ndarray, horizon: int) -> np.ndarray:
    before = (cumulative[ticks] - cumulative[ticks - horizon]) / horizon
    after = (cumulative[ticks + horizon] - cumulative[ticks]) / horizon
    return after - before


def extract_signed_temporal_profile(
    channels: Mapping[str, np.ndarray],
) -> SignedTemporalProfile:
    """Extract the strongest public six-channel temporal change.

    Only the six exact sensor names in :data:`SIGNED_PROFILE_CHANNELS` are
    accepted.  Arrays are copied before calculation, must be finite 1-D
    series of equal length, and need at least 61 samples for the 30-sample
    before/after scan.  Ties are resolved by NumPy's first-maximum rule, so
    the earliest tick wins deterministically.
    """
    if not isinstance(channels, Mapping):
        raise TypeError("channels must be a mapping")
    if set(channels) != set(SIGNED_PROFILE_CHANNELS):
        raise ValueError("channels must contain exactly the six public signed-profile channels")

    arrays: list[np.ndarray] = []
    length: int | None = None
    for name in SIGNED_PROFILE_CHANNELS:
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
    if length is None or length < 2 * _MEDIUM_HORIZON + 1:
        raise ValueError("channels need at least 61 samples")

    matrix = np.column_stack(arrays)
    diff_centers, scales = _robust_diff_scales(matrix)
    # A second medium window establishes a public baseline before scoring.
    # Short inputs retain the only mathematically available 30-sample guard.
    scan_start = _SCAN_BURN_IN if length >= _SCAN_BURN_IN + _MEDIUM_HORIZON else _MEDIUM_HORIZON
    ticks = np.arange(scan_start, length - _MEDIUM_HORIZON + 1)
    trend = np.arange(length, dtype=np.float64)[:, None] * diff_centers[None, :]
    detrended = matrix - trend
    cumulative = np.vstack((np.zeros((1, matrix.shape[1])), np.cumsum(detrended, axis=0)))
    contrasts_5 = _window_means(cumulative, ticks, _SHORT_HORIZON) / scales
    contrasts_30 = _window_means(cumulative, ticks, _MEDIUM_HORIZON) / scales
    scores = np.linalg.norm(np.concatenate((contrasts_5, contrasts_30), axis=1), axis=1)
    best = int(np.argmax(scores))
    magnitude = float(scores[best])
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("channels contain no non-zero temporal change")
    short = contrasts_5[best]
    medium = contrasts_30[best]
    vector = np.concatenate((short, medium)) / magnitude
    return SignedTemporalProfile(
        channels=SIGNED_PROFILE_CHANNELS,
        change_tick=int(ticks[best]),
        scales=tuple(float(value) for value in scales),
        contrasts_5=tuple(float(value) for value in short),
        contrasts_30=tuple(float(value) for value in medium),
        vector=tuple(float(value) for value in vector),
        magnitude=magnitude,
    )


def _canonical_orientation(vector: np.ndarray) -> np.ndarray:
    anchor = int(np.argmax(np.abs(vector)))
    return vector if vector[anchor] >= 0.0 else -vector


def build_signed_template(
    source: str, profiles: Sequence[SignedTemporalProfile]
) -> SignedProfileTemplate:
    """Build an input-order- and global-sign-invariant source template."""
    accepted_source = _validate_source(source)
    if isinstance(profiles, (str, bytes)) or not isinstance(profiles, Sequence):
        raise TypeError("profiles must be a sequence of SignedTemporalProfile values")
    if not profiles:
        raise ValueError("profiles must contain at least one profile")
    vectors: list[np.ndarray] = []
    for index, item in enumerate(profiles):
        profile = _validate_profile(item, context=f"profiles[{index}]")
        vectors.append(_canonical_orientation(np.asarray(profile.vector, dtype=np.float64)))
    vectors.sort(key=lambda value: tuple(float(item) for item in value))
    combined = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(combined))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("profiles cannot form a non-zero signed template")
    vector = combined / norm
    vector = _canonical_orientation(vector)
    return SignedProfileTemplate(
        source=accepted_source,
        channels=SIGNED_PROFILE_CHANNELS,
        vector=tuple(float(value) for value in vector),
        n_profiles=len(vectors),
    )


def match_signed_profile(
    profile: SignedTemporalProfile,
    templates: Mapping[str, SignedProfileTemplate],
    threshold: float = 0.80,
    min_margin: float = 0.15,
) -> SignedProfileMatch:
    """Match by absolute cosine, returning orientation or explicit abstention."""
    query = _validate_profile(profile, context="profile")
    accepted_threshold = _probability(threshold, "threshold")
    accepted_margin = _probability(min_margin, "min_margin")
    if not isinstance(templates, Mapping):
        raise TypeError("templates must be a mapping")
    if not templates:
        raise ValueError("templates must contain at least one template")
    query_vector = np.asarray(query.vector, dtype=np.float64)
    candidates: list[tuple[str, float, Literal[-1, 1]]] = []
    for source, template in templates.items():
        accepted_source = _validate_source(source)
        if not isinstance(template, SignedProfileTemplate):
            raise TypeError(f"template {accepted_source!r} must be a SignedProfileTemplate")
        SignedProfileTemplate.__post_init__(template)
        if template.source != accepted_source:
            raise ValueError("template mapping key must equal template source")
        signed_cosine = float(np.dot(query_vector, np.asarray(template.vector)))
        orientation: Literal[-1, 1] = 1 if signed_cosine >= 0.0 else -1
        similarity = float(np.clip(abs(signed_cosine), 0.0, 1.0))
        candidates.append((accepted_source, similarity, orientation))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    best_source, similarity, orientation = candidates[0]
    runner_up = candidates[1][1] if len(candidates) > 1 else None
    margin = similarity - runner_up if runner_up is not None else None
    if similarity < accepted_threshold:
        status: Literal["identified", "unattributed"] = "unattributed"
        source: str | None = None
        reason = "below_similarity_threshold"
    elif runner_up is None:
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
    return SignedProfileMatch(
        status=status,
        source=source,
        similarity=similarity,
        runner_up_similarity=runner_up,
        margin=margin,
        orientation=orientation,
        threshold=accepted_threshold,
        min_margin=accepted_margin,
        reason=reason,
    )
