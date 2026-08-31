# SPDX-License-Identifier: Apache-2.0
"""Pure profile extraction and conservative observable-profile matching."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ergodic.latent import (
    LatentResult,
    MarkovChain,
    ObservableProfile,
    ProfileInterval,
    ProfileMatch,
    match_profile,
    profile,
)


def _result(delta_draws: dict[str, np.ndarray]) -> LatentResult:
    n_decoded = len(next(iter(delta_draws.values()))) if delta_draws else 2
    return LatentResult(
        target="signal",
        gammas=np.full((n_decoded, 4), 0.5),
        flips=np.zeros(n_decoded, dtype=bool),
        delta_draws=delta_draws,
        gates=(),
        tick_multiplier=1,
        offset=0,
        chain=MarkovChain(),
        likelihood=object(),
        n_draws=n_decoded,
    )


def _observable(blocks: tuple[str, ...], weights: tuple[float, ...]) -> ObservableProfile:
    return ObservableProfile(
        blocks=blocks,
        weights=weights,
        intervals=tuple(ProfileInterval(block, 0.0, 0.0, 0.0) for block in blocks),
    )


def test_profile_uses_declared_order_standardized_weights_and_intervals() -> None:
    result = _result(
        {
            "spread": np.array([1.0, 3.0, 5.0, 7.0]),
            "level": np.array([-2.0, -1.0, -3.0, -2.0]),
        }
    )

    value = profile(result)

    spread = abs(np.mean(result.delta_draws["spread"])) / np.std(
        result.delta_draws["spread"], ddof=1
    )
    level = abs(np.mean(result.delta_draws["level"])) / np.std(result.delta_draws["level"], ddof=1)
    expected = np.array([spread, level])
    expected /= np.linalg.norm(expected)
    assert value.blocks == ("spread", "level")
    np.testing.assert_allclose(value.weights, expected, rtol=0.0, atol=1e-15)
    assert tuple(interval.block for interval in value.intervals) == value.blocks
    assert value.intervals[0].mean == pytest.approx(4.0)
    assert value.intervals[1].mean == pytest.approx(-2.0)
    assert np.linalg.norm(value.weights) == pytest.approx(1.0, abs=1e-15)


def test_profile_matching_coordinates_are_label_invariant() -> None:
    draws = {
        "level": np.array([0.8, 1.0, 1.2, 1.0]),
        "spread": np.array([-0.2, 0.2, -0.1, 0.1]),
    }
    forward = profile(_result(draws))
    flipped = profile(_result({name: -values for name, values in draws.items()}))

    assert forward.blocks == flipped.blocks
    np.testing.assert_allclose(forward.weights, flipped.weights, rtol=0.0, atol=1e-15)
    assert forward.intervals != flipped.intervals


@pytest.mark.parametrize(
    ("draws", "message"),
    [
        ({"level": np.array([1.0])}, "at least two"),
        ({"level": np.array([1.0, np.nan])}, "non-finite"),
        ({"level": np.array([0.0, 0.0])}, "zero"),
        ({"": np.array([1.0, 2.0])}, "non-empty"),
    ],
)
def test_profile_refuses_invalid_draws_and_blocks(
    draws: dict[str, np.ndarray], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        profile(_result(draws))


def test_profile_refuses_non_result() -> None:
    with pytest.raises(TypeError, match="LatentResult"):
        profile(object())  # type: ignore[arg-type]


def test_profile_values_are_frozen_and_tuple_only() -> None:
    interval = ProfileInterval("level", -1.0, 0.0, 1.0)
    value = _observable(("level",), (1.0,))
    match = ProfileMatch(
        "unattributed",
        None,
        0.5,
        None,
        None,
        0.8,
        0.15,
        "below_similarity_threshold",
    )

    with pytest.raises(FrozenInstanceError):
        interval.mean = 2.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        value.weights = (0.0,)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        match.reason = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable tuple"):
        ObservableProfile(  # type: ignore[arg-type]
            blocks=["level"], weights=(1.0,), intervals=(interval,)
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        ObservableProfile(  # type: ignore[arg-type]
            blocks=("level",), weights=[1.0], intervals=(interval,)
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        ObservableProfile(  # type: ignore[arg-type]
            blocks=("level",), weights=(1.0,), intervals=[interval]
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ProfileInterval("", 0.0, 0.0, 1.0), "non-empty"),
        (lambda: ProfileInterval("x", np.nan, 0.0, 1.0), "finite"),
        (lambda: ProfileInterval("x", 2.0, 0.0, 1.0), "low <= high"),
        (
            lambda: ObservableProfile(
                ("x", "x"),
                (2**-0.5, 2**-0.5),
                (ProfileInterval("x", 0.0, 0.0, 0.0),) * 2,
            ),
            "unique",
        ),
        (
            lambda: ObservableProfile(("x",), (1.0, 0.0), (ProfileInterval("x", 0.0, 0.0, 0.0),)),
            "equal lengths",
        ),
        (
            lambda: ObservableProfile(("x",), (-1.0,), (ProfileInterval("x", 0.0, 0.0, 0.0),)),
            "non-negative",
        ),
        (
            lambda: ObservableProfile(("x",), (0.5,), (ProfileInterval("x", 0.0, 0.0, 0.0),)),
            "weight norm",
        ),
        (
            lambda: ObservableProfile(("x",), (1.0,), (ProfileInterval("y", 0.0, 0.0, 0.0),)),
            "equal blocks in order",
        ),
    ],
)
def test_profile_value_validation(factory: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


def test_match_aligns_sorted_union_and_identifies_with_threshold_and_margin() -> None:
    query = _observable(("spread", "level"), (0.6, 0.8))
    library = {
        "simulated_spread_change": _observable(("level", "spread", "freq"), (0.0, 1.0, 0.0)),
        "simulated_level_shift": _observable(("freq", "level"), (0.0, 1.0)),
    }

    match = match_profile(query, library, threshold=0.8, min_margin=0.15)

    assert match.status == "identified"
    assert match.source == "simulated_level_shift"
    assert match.similarity == pytest.approx(0.8)
    assert match.runner_up_similarity == pytest.approx(0.6)
    assert match.margin == pytest.approx(0.2)
    assert match.reason == "threshold_and_margin_passed"


def test_match_threshold_failure_precedes_margin_rules() -> None:
    query = _observable(("level", "amp"), (0.6, 0.8))
    one_entry = {"level": _observable(("level",), (1.0,))}

    match = match_profile(query, one_entry, threshold=0.8)

    assert match.status == "unattributed"
    assert match.source is None
    assert match.similarity == pytest.approx(0.6)
    assert match.runner_up_similarity is None
    assert match.margin is None
    assert match.reason == "below_similarity_threshold"


def test_match_requires_a_runner_up_even_when_threshold_passes() -> None:
    match = match_profile(
        _observable(("level",), (1.0,)),
        {"level": _observable(("level",), (1.0,))},
    )

    assert match.status == "unattributed"
    assert match.reason == "no_runner_up_for_margin"


def test_match_abstains_when_runner_up_margin_is_too_small() -> None:
    query = _observable(("a", "b"), (2**-0.5, 2**-0.5))
    library = {
        "b": _observable(("b",), (1.0,)),
        "a": _observable(("a",), (1.0,)),
    }

    match = match_profile(query, library, threshold=0.7, min_margin=0.01)

    assert match.status == "unattributed"
    assert match.source is None
    assert match.margin == pytest.approx(0.0)
    assert match.reason == "insufficient_runner_up_margin"


def test_match_ties_are_broken_by_source_name() -> None:
    query = _observable(("x", "y"), (2**-0.5, 2**-0.5))
    match = match_profile(
        query,
        {
            "z-source": _observable(("y",), (1.0,)),
            "a-source": _observable(("x",), (1.0,)),
        },
        threshold=0.0,
        min_margin=0.0,
    )
    assert match.status == "identified"
    assert match.source == "a-source"


@pytest.mark.parametrize(
    ("threshold", "min_margin", "message"),
    [
        (-0.1, 0.1, "threshold"),
        (1.1, 0.1, "threshold"),
        (np.nan, 0.1, "threshold"),
        (0.8, -0.1, "min_margin"),
        (0.8, 1.1, "min_margin"),
        (0.8, np.inf, "min_margin"),
    ],
)
def test_match_refuses_invalid_decision_parameters(
    threshold: float, min_margin: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        match_profile(
            _observable(("x",), (1.0,)),
            {"x": _observable(("x",), (1.0,))},
            threshold=threshold,
            min_margin=min_margin,
        )


def test_match_refuses_empty_library_and_bad_sources() -> None:
    query = _observable(("x",), (1.0,))
    with pytest.raises(ValueError, match="at least one"):
        match_profile(query, {})
    with pytest.raises(ValueError, match="source names"):
        match_profile(query, {"": query})


def test_match_revalidates_forged_query_and_library_profiles() -> None:
    query = _observable(("x",), (1.0,))
    candidate = _observable(("x",), (1.0,))
    object.__setattr__(query, "weights", (np.nan,))
    with pytest.raises(ValueError, match="finite"):
        match_profile(query, {"x": candidate})

    query = _observable(("x",), (1.0,))
    object.__setattr__(candidate, "blocks", ())
    with pytest.raises(ValueError, match="at least one"):
        match_profile(query, {"x": candidate})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "other"}, "status"),
        ({"similarity": np.nan}, "similarity"),
        ({"similarity": -0.1}, "similarity"),
        ({"runner_up_similarity": np.inf}, "runner_up_similarity"),
        ({"threshold": 1.1}, "threshold"),
        ({"min_margin": -0.1}, "min_margin"),
        ({"margin": 0.1}, "margin must be null"),
        (
            {"runner_up_similarity": 0.4, "margin": None},
            "margin is required",
        ),
        (
            {"runner_up_similarity": 0.6, "margin": 0.1},
            "non-negative and equal",
        ),
        (
            {"runner_up_similarity": 0.4, "margin": 0.2},
            "non-negative and equal",
        ),
        ({"source": "named"}, "source must be null"),
        (
            {"reason": "no_runner_up_for_margin"},
            "threshold passage",
        ),
        (
            {
                "similarity": 0.9,
                "threshold": 0.8,
                "reason": "below_similarity_threshold",
            },
            "similarity < threshold",
        ),
        (
            {
                "similarity": 0.9,
                "threshold": 0.8,
                "runner_up_similarity": 0.7,
                "margin": 0.2,
                "reason": "insufficient_runner_up_margin",
            },
            "margin < min_margin",
        ),
        ({"reason": "unknown"}, "unknown reason"),
    ],
)
def test_profile_match_rejects_invalid_public_values(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "status": "unattributed",
        "source": None,
        "similarity": 0.5,
        "runner_up_similarity": None,
        "margin": None,
        "threshold": 0.8,
        "min_margin": 0.15,
        "reason": "below_similarity_threshold",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ProfileMatch(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": None}, "non-empty source"),
        ({"source": ""}, "non-empty source"),
        ({"reason": "below_similarity_threshold"}, "reason"),
        ({"runner_up_similarity": None, "margin": None}, "runner-up"),
        ({"similarity": 0.79, "runner_up_similarity": 0.5, "margin": 0.29}, "pass"),
        ({"runner_up_similarity": 0.76, "margin": 0.14}, "pass"),
    ],
)
def test_identified_profile_match_rejects_invalid_public_values(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "status": "identified",
        "source": "simulated_level_shift",
        "similarity": 0.9,
        "runner_up_similarity": 0.6,
        "margin": 0.3,
        "threshold": 0.8,
        "min_margin": 0.15,
        "reason": "threshold_and_margin_passed",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ProfileMatch(**values)  # type: ignore[arg-type]


def test_profile_match_accepts_margin_arithmetic_within_tolerance() -> None:
    value = ProfileMatch(
        status="identified",
        source="simulated_level_shift",
        similarity=0.9,
        runner_up_similarity=0.6,
        margin=0.3 + 5e-13,
        threshold=0.8,
        min_margin=0.15,
        reason="threshold_and_margin_passed",
    )
    assert value.margin == pytest.approx(0.3)


def test_public_profile_names_are_exported() -> None:
    import ergodic.latent as latent

    assert latent.ProfileInterval is ProfileInterval
    assert latent.ObservableProfile is ObservableProfile
    assert latent.ProfileMatch is ProfileMatch
    assert latent.profile is profile
    assert latent.match_profile is match_profile
