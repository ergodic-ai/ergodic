# SPDX-License-Identifier: Apache-2.0
"""Focused checks for immutable control-baseline residual evidence."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from ergodic.latent import (
    SIGNED_EVIDENCE_DURATIONS,
    SIGNED_PROFILE_CHANNELS,
    ControlBaselineModel,
    ControlResidualEvidence,
    NullEvidenceDecision,
    apply_null_evidence_gate,
    calibrate_null_evidence_gate,
    fit_control_baseline,
    score_control_residual_evidence,
)

_SEVEN_CHANNELS = (*SIGNED_PROFILE_CHANNELS, "indoor_illuminance_lux")


def _control_days(
    *, channel_order: tuple[str, ...] = ("temperature_c", "humidity_pct")
) -> list[dict[str, np.ndarray]]:
    ticks = np.arange(1_440, dtype=np.float64)
    days: list[dict[str, np.ndarray]] = []
    for day_index in range(6):
        day: dict[str, np.ndarray] = {}
        for channel_index, name in enumerate(channel_order):
            wave = np.sin(ticks / (19.0 + day_index + channel_index))
            ripple = 0.07 * np.cos(ticks / (3.0 + channel_index))
            day[name] = (
                100.0 * (day_index + 1) + 11.0 * channel_index + (channel_index + 1) * wave + ripple
            )
        days.append(day)
    return days


def _flat_baseline(*, channel_order: tuple[str, ...] = ("temperature_c",)) -> ControlBaselineModel:
    days = [{name: np.full(1_440, 10.0 * index) for name in channel_order} for index in range(6)]
    return fit_control_baseline(days, channel_order=channel_order)


def _residual_evidence(score: float) -> ControlResidualEvidence:
    return ControlResidualEvidence(
        channels=("x",),
        durations=SIGNED_EVIDENCE_DURATIONS,
        score=score,
        start_tick=0,
        end_tick=30,
        duration=30,
        scales=(1.0,),
        signed_deltas=(score,),
        query_centers=(0.0,),
        selected_prototype_position=0,
        selected_prototype_digest="0" * 64,
        prototype_loss=0.0,
        formula_version="control-baseline-residual/v1",
    )


def test_fit_uses_exact_daily_centering_and_control_scale_arithmetic() -> None:
    days = _control_days()
    snapshots = [{name: values.copy() for name, values in day.items()} for day in days]

    baseline = fit_control_baseline(days, channel_order=("temperature_c", "humidity_pct"))

    assert baseline.channels == ("temperature_c", "humidity_pct")
    assert baseline.durations == (30, 60, 120, 240)
    assert baseline.formula_version == "control-baseline-residual/v1"
    assert len(baseline.centered_prototypes) == 6
    for index, snapshot in enumerate(snapshots):
        matrix = np.column_stack([snapshot[name] for name in baseline.channels])
        expected = matrix - np.median(matrix, axis=0)
        np.testing.assert_allclose(baseline.centered_prototypes[index], expected, rtol=0, atol=0)

    per_control_scales: list[np.ndarray] = []
    for snapshot in snapshots:
        matrix = np.column_stack([snapshot[name] for name in baseline.channels])
        differences = np.diff(matrix, axis=0)
        diff_center = np.median(differences, axis=0)
        deviation = np.abs(differences - diff_center)
        mad_scale = 1.4826 * np.median(deviation, axis=0)
        expected_scales = []
        for column in range(matrix.shape[1]):
            positive = deviation[:, column][deviation[:, column] > 0.0]
            resolution = 0.1 * np.median(positive) if positive.size else 0.0
            level_floor = np.sqrt(np.finfo(np.float64).eps) * max(
                1.0, np.median(np.abs(matrix[:, column]))
            )
            expected_scales.append(max(mad_scale[column], resolution, level_floor))
        per_control_scales.append(np.asarray(expected_scales))
    np.testing.assert_allclose(
        baseline.scales,
        np.median(np.stack(per_control_scales), axis=0),
        rtol=0,
        atol=0,
    )

    for day in days:
        for values in day.values():
            values[:] = np.nan
    assert np.all(np.isfinite(np.asarray(baseline.centered_prototypes)))


def test_query_selects_nearest_prototype_by_frozen_loss_and_ties_choose_earlier() -> None:
    days = _control_days(channel_order=("x",))
    baseline = fit_control_baseline(days, channel_order=("x",))
    query = {"x": days[4]["x"] + 4_000.0}

    evidence = score_control_residual_evidence(query, baseline)

    matrix = query["x"][:, None]
    centered_query = matrix - np.median(matrix, axis=0)
    prototypes = np.asarray(baseline.centered_prototypes)
    scales = np.asarray(baseline.scales)
    expected_losses = np.median(
        np.sqrt(np.mean(((centered_query - prototypes) / scales) ** 2, axis=2)), axis=1
    )
    assert evidence.selected_prototype_position == int(np.argmin(expected_losses)) == 4
    assert evidence.prototype_loss == pytest.approx(expected_losses[4])

    flat = _flat_baseline()
    tied = score_control_residual_evidence({"temperature_c": np.zeros(1_440)}, flat)
    assert tied.selected_prototype_position == 0


def test_residual_score_uses_edge_intervals_and_flat_ties_choose_shortest_earliest() -> None:
    baseline = _flat_baseline()
    flat = score_control_residual_evidence({"temperature_c": np.full(1_440, 88.0)}, baseline)

    assert flat.score == 0.0
    assert flat.start_tick == 0
    assert flat.end_tick == 30
    assert flat.duration == 30
    assert flat.signed_deltas == (0.0,)

    first = np.zeros(1_440)
    first[:30] = 5.0
    edge = score_control_residual_evidence({"temperature_c": first}, baseline)
    assert edge.start_tick == 0
    assert edge.end_tick == 30
    assert edge.duration == 30
    assert edge.signed_deltas[0] > 0.0

    last = np.zeros(1_440)
    last[-30:] = -5.0
    other_edge = score_control_residual_evidence({"temperature_c": last}, baseline)
    assert other_edge.start_tick == 1_410
    assert other_edge.end_tick == 1_440


def test_constant_offsets_do_not_change_evidence_and_slow_residual_survives_centering() -> None:
    baseline = _flat_baseline(channel_order=("temperature_c", "humidity_pct"))
    query = {
        "temperature_c": np.zeros(1_440),
        "humidity_pct": np.zeros(1_440),
    }
    ramp = np.linspace(0.0, 8.0, 240)
    query["temperature_c"][600:840] = ramp

    original = score_control_residual_evidence(query, baseline)
    shifted = score_control_residual_evidence(
        {
            "temperature_c": query["temperature_c"] + 23.5,
            "humidity_pct": query["humidity_pct"] - 81.25,
        },
        baseline,
    )

    assert original.score > 0.0
    assert original.signed_deltas[0] > 0.0
    assert original.selected_prototype_position == shifted.selected_prototype_position
    assert original.duration == shifted.duration
    assert original.start_tick == shifted.start_tick
    assert original.score == pytest.approx(shifted.score)
    np.testing.assert_allclose(original.signed_deltas, shifted.signed_deltas, atol=1e-14)


def test_explicit_seven_channel_schema_and_mapping_order_are_deterministic() -> None:
    days = _control_days(channel_order=_SEVEN_CHANNELS)
    shuffled = [{name: day[name] for name in reversed(_SEVEN_CHANNELS)} for day in days]

    left = fit_control_baseline(days, channel_order=_SEVEN_CHANNELS)
    right = fit_control_baseline(shuffled, channel_order=_SEVEN_CHANNELS)
    evidence = score_control_residual_evidence(days[2], left)

    assert left == right
    assert left.channels == _SEVEN_CHANNELS
    assert evidence.channels == _SEVEN_CHANNELS
    assert len(evidence.scales) == len(_SEVEN_CHANNELS)
    assert len(evidence.signed_deltas) == len(_SEVEN_CHANNELS)


def test_model_and_evidence_are_immutable_defensive_and_stably_serializable() -> None:
    baseline = _flat_baseline()
    duplicate = _flat_baseline()
    evidence = score_control_residual_evidence({"temperature_c": np.zeros(1_440)}, baseline)

    with pytest.raises(FrozenInstanceError):
        baseline.scales = (2.0,)  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable nested tuples"):
        replace(
            baseline,
            centered_prototypes=list(baseline.centered_prototypes),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(evidence, query_centers=[0.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalized signed-delta norm"):
        replace(evidence, score=1.0)

    left_json = json.dumps(asdict(baseline), sort_keys=True, separators=(",", ":"))
    right_json = json.dumps(asdict(duplicate), sort_keys=True, separators=(",", ":"))
    assert left_json == right_json
    assert (
        evidence.selected_prototype_digest
        == score_control_residual_evidence(
            {"temperature_c": np.zeros(1_440)}, duplicate
        ).selected_prototype_digest
    )


@pytest.mark.parametrize(
    ("days", "error", "message"),
    [
        (_control_days()[:5], ValueError, "exactly six"),
        ([*_control_days(), _control_days()[0]], ValueError, "exactly six"),
        (object(), TypeError, "sequence"),
    ],
)
def test_fit_rejects_wrong_control_count_or_container(
    days: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        fit_control_baseline(days, channel_order=("temperature_c", "humidity_pct"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda days: days[0].pop("humidity_pct"), "exactly"),
        (lambda days: days[0].__setitem__("truth", np.zeros(1_440)), "exactly"),
        (lambda days: days[0].__setitem__("humidity_pct", np.zeros((1_440, 1))), "1-D"),
        (lambda days: days[0].__setitem__("humidity_pct", np.zeros(1_439)), "1,440"),
        (lambda days: days[0]["humidity_pct"].__setitem__(0, np.inf), "non-finite"),
    ],
)
def test_fit_rejects_invalid_control_arrays(mutate: object, message: str) -> None:
    days = _control_days()
    mutate(days)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        fit_control_baseline(days, channel_order=("temperature_c", "humidity_pct"))


def test_query_rejects_schema_length_finiteness_and_invalid_baseline() -> None:
    baseline = _flat_baseline()
    with pytest.raises(ValueError, match="exactly"):
        score_control_residual_evidence(
            {"temperature_c": np.zeros(1_440), "truth": np.zeros(1_440)}, baseline
        )
    with pytest.raises(ValueError, match="1,440"):
        score_control_residual_evidence({"temperature_c": np.zeros(1_439)}, baseline)
    invalid = np.zeros(1_440)
    invalid[-1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        score_control_residual_evidence({"temperature_c": invalid}, baseline)
    with pytest.raises(TypeError, match="ControlBaselineModel"):
        score_control_residual_evidence({"temperature_c": np.zeros(1_440)}, object())  # type: ignore[arg-type]


def test_residual_evidence_uses_unchanged_null_rank_arithmetic() -> None:
    gate = calibrate_null_evidence_gate(
        [_residual_evidence(3.0), _residual_evidence(1.0), _residual_evidence(2.0)]
    )
    assert gate.calibration_scores == (1.0, 2.0, 3.0)

    tied = apply_null_evidence_gate(_residual_evidence(3.0), gate)
    assert tied == NullEvidenceDecision(
        supported=False,
        p_null=2 / 4,
        rank_numerator=2,
        rank_denominator=4,
        threshold=3.0,
        reason="evidence_not_above_null_threshold",
    )
    passed = apply_null_evidence_gate(_residual_evidence(3.1), gate)
    assert passed.supported
    assert passed.rank_numerator == 1
    assert passed.rank_denominator == 4
    assert passed.p_null == 1 / 4
