# SPDX-License-Identifier: Apache-2.0
"""Fast checks for public signed temporal profiles."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ergodic.latent import (
    SIGNED_PROFILE_CHANNELS,
    build_signed_template,
    extract_signed_temporal_profile,
    match_signed_profile,
)


def _channels(*, first: int = 60, second: int | None = None) -> dict[str, np.ndarray]:
    time = np.arange(140, dtype=np.float64)
    result = {
        name: 0.001 * (index + 1) * time for index, name in enumerate(SIGNED_PROFILE_CHANNELS)
    }
    signs = np.array([-2.0, -0.5, -30.0, 2.0, 5.0, 4.0])
    for index, name in enumerate(SIGNED_PROFILE_CHANNELS):
        result[name][first:] += signs[index]
        if second is not None:
            result[name][second:] -= signs[index]
    return result


def _flipped(profile: object) -> object:
    return replace(
        profile,
        contrasts_5=tuple(-value for value in profile.contrasts_5),
        contrasts_30=tuple(-value for value in profile.contrasts_30),
        vector=tuple(-value for value in profile.vector),
    )


def test_extract_finds_public_change_and_preserves_signed_contrasts() -> None:
    value = extract_signed_temporal_profile(_channels())

    assert value.channels == SIGNED_PROFILE_CHANNELS
    assert value.change_tick == 60
    assert value.contrasts_5[0] < 0.0
    assert value.contrasts_5[3] > 0.0
    assert value.contrasts_30[2] < 0.0
    assert value.magnitude > 0.0
    assert np.linalg.norm(value.vector) == pytest.approx(1.0)


def test_extract_breaks_equal_on_off_change_tie_at_earliest_tick() -> None:
    channels = {name: np.zeros(140) for name in SIGNED_PROFILE_CHANNELS}
    for index, name in enumerate(SIGNED_PROFILE_CHANNELS, start=1):
        channels[name][65:105] = float(index)
    value = extract_signed_temporal_profile(channels)
    assert value.change_tick == 65


def test_extract_copies_inputs_and_ignores_mapping_order() -> None:
    channels = _channels()
    reverse = {name: channels[name] for name in reversed(SIGNED_PROFILE_CHANNELS)}
    left = extract_signed_temporal_profile(channels)
    right = extract_signed_temporal_profile(reverse)
    for values in channels.values():
        values[:] = np.nan
    assert left == right
    assert all(np.isfinite(left.vector))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.pop("noise_db"), "exactly"),
        (lambda c: c.__setitem__("truth", np.zeros(140)), "exactly"),
        (lambda c: c.__setitem__("noise_db", np.zeros((140, 1))), "1-D"),
        (lambda c: c.__setitem__("noise_db", np.zeros(139)), "equal lengths"),
        (lambda c: c["noise_db"].__setitem__(0, np.nan), "non-finite"),
    ],
)
def test_extract_rejects_invalid_public_arrays(mutate: object, message: str) -> None:
    channels = _channels()
    mutate(channels)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        extract_signed_temporal_profile(channels)


def test_extract_rejects_short_and_constant_series() -> None:
    with pytest.raises(ValueError, match="61"):
        extract_signed_temporal_profile({name: np.zeros(60) for name in SIGNED_PROFILE_CHANNELS})
    with pytest.raises(ValueError, match="non-zero"):
        extract_signed_temporal_profile({name: np.zeros(80) for name in SIGNED_PROFILE_CHANNELS})


def test_template_is_invariant_to_global_sign_and_input_order() -> None:
    profile = extract_signed_temporal_profile(_channels())
    flipped = _flipped(profile)

    left = build_signed_template("window", [profile, flipped])
    right = build_signed_template("window", [flipped, profile])

    assert left == right
    assert left.n_profiles == 2
    np.testing.assert_allclose(left.vector, build_signed_template("window", [profile]).vector)


def test_match_identifies_both_orientations_and_reports_margin() -> None:
    window = extract_signed_temporal_profile(_channels())
    heat_channels = _channels()
    for name in SIGNED_PROFILE_CHANNELS:
        heat_channels[name] *= 0.0
    heat_channels["temperature_c"][70:] = 4.0
    heat_channels["humidity_pct"][70:] = -2.0
    heat = extract_signed_temporal_profile(heat_channels)
    templates = {
        "heat": build_signed_template("heat", [heat]),
        "window": build_signed_template("window", [window]),
    }

    forward = match_signed_profile(window, templates, 0.8, 0.05)
    reversed_match = match_signed_profile(_flipped(window), templates, 0.8, 0.05)

    assert forward.status == "identified"
    assert forward.source == "window"
    assert reversed_match.source == "window"
    assert reversed_match.orientation == -forward.orientation
    assert forward.similarity == pytest.approx(1.0)
    assert forward.margin is not None and forward.margin >= 0.05


def test_match_abstains_below_threshold_and_without_margin() -> None:
    query = extract_signed_temporal_profile(_channels())
    template = build_signed_template("window", [query])
    one = match_signed_profile(query, {"window": template}, 0.8, 0.0)
    assert one.status == "unattributed"
    assert one.reason == "no_runner_up_for_margin"
    assert one.source is None

    tied = match_signed_profile(
        query,
        {"a": replace(template, source="a"), "b": replace(template, source="b")},
        0.8,
        0.01,
    )
    assert tied.status == "unattributed"
    assert tied.reason == "insufficient_runner_up_margin"
    assert tied.margin == pytest.approx(0.0)


def test_match_validates_thresholds_templates_and_mapping_binding() -> None:
    profile = extract_signed_temporal_profile(_channels())
    template = build_signed_template("window", [profile])
    with pytest.raises(ValueError, match="threshold"):
        match_signed_profile(profile, {"window": template}, np.nan, 0.1)
    with pytest.raises(ValueError, match="mapping key"):
        match_signed_profile(profile, {"other": template})
    with pytest.raises(ValueError, match="at least one"):
        match_signed_profile(profile, {})
