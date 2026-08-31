# SPDX-License-Identifier: Apache-2.0
"""Fast checks for source-free signed disturbance evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from ergodic.latent import (
    SIGNED_EVIDENCE_DURATIONS,
    SIGNED_PROFILE_CHANNELS,
    NullEvidenceDecision,
    NullEvidenceGate,
    SignedDisturbanceEvidence,
    apply_null_evidence_gate,
    calibrate_null_evidence_gate,
    score_signed_disturbance_evidence,
)
from ergodic.latent._signed_evidence import _candidate_is_better


def _channels(*, length: int = 900, start: int = 360, duration: int = 60) -> dict[str, np.ndarray]:
    ticks = np.arange(length, dtype=np.float64)
    signs = np.array([-2.0, -0.5, -30.0, 2.0, 5.0, 4.0])
    channels: dict[str, np.ndarray] = {}
    for index, name in enumerate(SIGNED_PROFILE_CHANNELS):
        # Nonconstant differences give a stable, positive robust scale while
        # the small oscillation remains negligible beside the planted episode.
        values = 0.002 * (index + 1) * ticks + 0.01 * np.sin(ticks / 7.0 + index)
        values[start : start + duration] += signs[index]
        channels[name] = values
    return channels


def _evidence(score: float) -> SignedDisturbanceEvidence:
    return SignedDisturbanceEvidence(
        channels=("x",),
        durations=SIGNED_EVIDENCE_DURATIONS,
        score=score,
        start_tick=30,
        end_tick=60,
        duration=30,
        scales=(1.0,),
        signed_deltas=(score,),
    )


def test_score_uses_exact_inside_shoulders_arithmetic_and_signed_deltas() -> None:
    channels = _channels()
    evidence = score_signed_disturbance_evidence(channels)

    assert evidence.channels == SIGNED_PROFILE_CHANNELS
    assert evidence.durations == (30, 60, 120, 240)
    assert evidence.start_tick == 360
    assert evidence.end_tick == 420
    assert evidence.duration == 60

    matrix = np.column_stack([channels[name] for name in SIGNED_PROFILE_CHANNELS])
    slopes = np.median(np.diff(matrix, axis=0), axis=0)
    detrended = matrix - np.arange(matrix.shape[0])[:, None] * slopes
    expected_delta = detrended[360:420].mean(axis=0) - 0.5 * (
        detrended[300:360].mean(axis=0) + detrended[420:480].mean(axis=0)
    )
    np.testing.assert_allclose(evidence.signed_deltas, expected_delta)
    expected_score = np.linalg.norm(expected_delta / evidence.scales) / np.sqrt(6)
    assert evidence.score == pytest.approx(expected_score)
    assert evidence.signed_deltas[0] < 0.0
    assert evidence.signed_deltas[3] > 0.0


def test_score_ties_choose_shorter_duration_then_earlier_start() -> None:
    incumbent = (3.0, 60, 100)
    assert _candidate_is_better((3.0, 30, 200), incumbent)
    assert _candidate_is_better((3.0, 60, 99), incumbent)
    assert not _candidate_is_better((3.0, 60, 101), incumbent)
    assert not _candidate_is_better((2.9, 30, 1), incumbent)

    channels = {}
    for index, name in enumerate(SIGNED_PROFILE_CHANNELS):
        values = np.tile([0.0, float(index + 1)], 480)
        values[300:360] += float(index + 1)
        values[600:660] += float(index + 1)
        channels[name] = values
    assert score_signed_disturbance_evidence(channels).start_tick == 300


def test_score_copies_inputs_and_uses_explicit_order() -> None:
    channels = _channels()
    reverse = {name: channels[name] for name in reversed(SIGNED_PROFILE_CHANNELS)}
    left = score_signed_disturbance_evidence(channels)
    right = score_signed_disturbance_evidence(reverse)
    for values in channels.values():
        values[:] = np.nan
    assert left == right

    order = (*SIGNED_PROFILE_CHANNELS, "illuminance_lux")
    seven = _channels()
    seven["illuminance_lux"] = np.sin(np.arange(900) / 5.0)
    seven["illuminance_lux"][360:420] -= 100.0
    result = score_signed_disturbance_evidence(seven, channel_order=order)
    assert result.channels == order
    assert len(result.scales) == 7
    assert result.signed_deltas[-1] < 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.pop("noise_db"), "exactly"),
        (lambda c: c.__setitem__("truth", np.zeros(900)), "exactly"),
        (lambda c: c.__setitem__("noise_db", np.zeros((900, 1))), "1-D"),
        (lambda c: c.__setitem__("noise_db", np.zeros(899)), "equal lengths"),
        (lambda c: c["noise_db"].__setitem__(0, np.inf), "non-finite"),
    ],
)
def test_score_rejects_invalid_public_arrays(mutate: object, message: str) -> None:
    channels = _channels()
    mutate(channels)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        score_signed_disturbance_evidence(channels)


def test_score_rejects_short_but_returns_deterministic_zero_for_no_episode() -> None:
    with pytest.raises(ValueError, match="720"):
        score_signed_disturbance_evidence({name: np.zeros(719) for name in SIGNED_PROFILE_CHANNELS})
    flat = score_signed_disturbance_evidence(
        {name: np.zeros(720) for name in SIGNED_PROFILE_CHANNELS}
    )
    assert flat.score == 0.0
    assert flat.duration == 30
    assert flat.start_tick == 30
    assert flat.end_tick == 60
    assert flat.signed_deltas == (0.0,) * 6

    ticks = np.arange(720, dtype=np.float64)
    for slope, offset in ((1.0, 0.0), (0.1, 17.3), (1e6, -2.4)):
        linear = score_signed_disturbance_evidence(
            {
                name: (index + 1) * slope * ticks + offset
                for index, name in enumerate(SIGNED_PROFILE_CHANNELS)
            }
        )
        assert np.isfinite(linear.score)
        assert linear.score < 2e-6


def test_near_affine_tiny_episode_is_not_erased_and_matches_brute_force() -> None:
    ticks = np.arange(900, dtype=np.float64)
    values = 1000.0 + 0.1 * ticks
    values[360:390] += 1e-11

    evidence = score_signed_disturbance_evidence({"x": values}, channel_order=("x",))

    slope = float(np.median(np.diff(values)))
    detrended = values - ticks * slope
    scale = evidence.scales[0]
    candidates: list[tuple[float, int, int, float]] = []
    for duration in SIGNED_EVIDENCE_DURATIONS:
        for start in range(duration, len(values) - 2 * duration + 1):
            delta = float(
                np.mean(detrended[start : start + duration])
                - 0.5
                * (
                    np.mean(detrended[start - duration : start])
                    + np.mean(detrended[start + duration : start + 2 * duration])
                )
            )
            candidates.append((abs(delta / scale), duration, start, delta))
    expected_score = max(candidate[0] for candidate in candidates)
    expected = min(
        (candidate for candidate in candidates if candidate[0] == expected_score),
        key=lambda candidate: (candidate[1], candidate[2]),
    )

    assert evidence.score > 0.0
    assert evidence.score == expected[0]
    assert evidence.duration == expected[1]
    assert evidence.start_tick == expected[2]
    assert evidence.signed_deltas == (expected[3],)


def test_evidence_is_immutable_and_validates_derived_values() -> None:
    evidence = score_signed_disturbance_evidence(_channels())
    with pytest.raises(FrozenInstanceError):
        evidence.score = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="normalized signed-delta norm"):
        replace(evidence, score=evidence.score + 1.0)
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(evidence, scales=list(evidence.scales))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonnegative"):
        _evidence(-0.1)


def test_calibration_is_generic_sorted_and_order_invariant() -> None:
    evidence = [_evidence(3.0), _evidence(1.0), _evidence(2.0)]
    left = calibrate_null_evidence_gate(evidence)
    right = calibrate_null_evidence_gate(list(reversed(evidence)))

    assert left == right
    assert left.calibration_scores == (1.0, 2.0, 3.0)
    assert left.threshold == 3.0
    assert left.calibration_count == 3
    assert left.alpha == 1 / 4


def test_gate_rank_uses_greater_equal_and_threshold_ties_fail_closed() -> None:
    gate = calibrate_null_evidence_gate([_evidence(1.0), _evidence(2.0), _evidence(3.0)])

    tied = apply_null_evidence_gate(_evidence(3.0), gate)
    assert tied == NullEvidenceDecision(
        supported=False,
        p_null=2 / 4,
        rank_numerator=2,
        rank_denominator=4,
        threshold=3.0,
        reason="evidence_not_above_null_threshold",
    )

    passed = apply_null_evidence_gate(_evidence(3.1), gate)
    assert passed.supported
    assert passed.p_null == 1 / 4
    assert passed.rank_numerator == 1
    assert passed.reason == "evidence_exceeds_null_threshold"

    middle = apply_null_evidence_gate(_evidence(1.5), gate)
    assert middle.rank_numerator == 3
    assert middle.p_null == 3 / 4


def test_zero_score_calibration_and_query_use_strict_greater_than_support() -> None:
    gate = calibrate_null_evidence_gate([_evidence(0.0), _evidence(0.0), _evidence(0.0)])
    assert gate.calibration_scores == (0.0, 0.0, 0.0)
    assert gate.threshold == 0.0

    tied = apply_null_evidence_gate(_evidence(0.0), gate)
    assert not tied.supported
    assert tied.rank_numerator == 4
    assert tied.p_null == 1.0

    positive = apply_null_evidence_gate(_evidence(0.1), gate)
    assert positive.supported
    assert positive.rank_numerator == 1
    assert positive.p_null == 1 / 4


def test_gate_rejects_empty_wrong_types_and_invalid_direct_construction() -> None:
    with pytest.raises(ValueError, match="at least one"):
        calibrate_null_evidence_gate([])
    with pytest.raises(TypeError, match=r"calibration_evidence\[0\]"):
        calibrate_null_evidence_gate([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="sorted"):
        NullEvidenceGate((2.0, 1.0), 2.0, 2, 1 / 3)
    with pytest.raises(ValueError, match="largest"):
        NullEvidenceGate((1.0, 2.0), 1.0, 2, 1 / 3)
    with pytest.raises(ValueError, match="alpha"):
        NullEvidenceGate((1.0, 2.0), 2.0, 2, 0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        NullEvidenceGate((-1.0, 2.0), 2.0, 2, 1 / 3)
