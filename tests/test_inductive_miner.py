# SPDX-License-Identifier: Apache-2.0
"""Tests for the inductive miner: planted trees, the infrequent filter, provenance."""

from __future__ import annotations

import pandas as pd
import pytest

from ergodic.data import EventLog, event_log
from ergodic.process._mine import discover_model
from ergodic.process._tree import TAU, loop, par, seq, xor

# -- fixtures ------------------------------------------------------------------


def _log(variants: dict[tuple[str, ...], int]) -> EventLog:
    """One case per trace copy, events at float timestamps 0, 1, 2, ..."""
    cases: list[str] = []
    acts: list[str] = []
    stamps: list[float] = []
    n = 0
    for trace, count in variants.items():
        for _ in range(count):
            name = f"c{n:03d}"
            n += 1
            for t, act in enumerate(trace):
                cases.append(name)
                acts.append(act)
                stamps.append(float(t))
    df = pd.DataFrame({"case": cases, "act": acts, "ts": stamps})
    return event_log(df, case="case", activity="act", timestamp="ts")


_Variants = dict[tuple[str, ...], int]

# Clean worlds with planted trees.
_NESTED_CLEAN: _Variants = {("r", "x", "n"): 20, ("r", "y", "n"): 20}
_STRUCTURELESS: _Variants = {
    ("a", "b", "c"): 2,
    ("c", "a", "b"): 2,
    ("a", "c"): 2,
    ("b", "c", "a"): 2,
}

# The nested world plus one swapped pair: r and x traded places in one case.
_NESTED_NOISY: _Variants = {("r", "x", "n"): 20, ("r", "y", "n"): 20, ("x", "r", "n"): 1}

# Two clean branches plus one swapped pair and one stray cross-branch step:
# the brief's skipped step cannot be absorbed by the edge-only filter (a seq
# projection always hands the skipped part an empty trace and mints a TAU
# option), so the second noisy trace is a branch-one case contaminated by a
# branch-two event, which exercises the other absorber: the choice
# projection's event drop.
_BRANCHED_CLEAN: _Variants = {("a", "b", "c"): 20, ("d", "e", "f"): 20}
_BRANCHED_NOISY: _Variants = {
    ("a", "b", "c"): 20,
    ("d", "e", "f"): 20,
    ("b", "a", "c"): 1,
    ("a", "e", "b", "c"): 1,
}

_ALL_WORLDS: list[_Variants] = [
    {("a", "b", "c"): 4},
    {("a",): 3, ("b",): 3},
    {("a", "b"): 5, ("b", "a"): 5},
    {("a",): 3, ("a", "b", "a"): 3, ("a", "b", "a", "b", "a"): 3},
    {("a", "b", "c"): 3, ("a", "c"): 2},
    {("a",): 2, ("a", "a"): 2, ("a", "a", "a"): 1},
    {("a",): 2, ("a", "b", "a"): 2, ("a", "c", "a"): 2},
    {("s", "b", "s", "t"): 4},
    _NESTED_CLEAN,
    _STRUCTURELESS,
    _NESTED_NOISY,
    _BRANCHED_NOISY,
]

# -- one planted tree per operator ---------------------------------------------


def test_sequence_world() -> None:
    model = discover_model(_log({("a", "b", "c"): 4}))
    assert model.tree == seq("a", "b", "c")
    assert model.fallthroughs == ()
    assert model.n_filtered_arcs == 0
    assert model.n_dropped_events == 0
    assert model.dropped_share == 0.0
    assert model.noise == 0.0


def test_choice_world() -> None:
    assert discover_model(_log({("a",): 3, ("b",): 3})).tree == xor("a", "b")


def test_parallel_world() -> None:
    assert discover_model(_log({("a", "b"): 5, ("b", "a"): 5})).tree == par("a", "b")


def test_loop_world() -> None:
    log = _log({("a",): 3, ("a", "b", "a"): 3, ("a", "b", "a", "b", "a"): 3})
    assert discover_model(log).tree == loop("a", "b")


def test_loop_with_two_redo_parts() -> None:
    # Our loop is binary, so several redo parts land under one xor.
    log = _log({("a",): 2, ("a", "b", "a"): 2, ("a", "c", "a"): 2})
    assert discover_model(log).tree == loop("a", xor("b", "c"))


def test_loop_cut_boundary_rule_rejects_candidate() -> None:
    # At the root, s -> b leaves the do body from a non-end activity, so b
    # merges into the do part, the loop cut dies, and the sequence cut wins;
    # the loop reappears one level down, where s is both start and end.
    log = _log({("s", "b", "s", "t"): 4})
    assert discover_model(log).tree == seq(loop("s", "b"), "t")


# -- nesting, optionality, and the bases ----------------------------------------


def test_nested_world() -> None:
    assert discover_model(_log(_NESTED_CLEAN)).tree == seq("r", xor("x", "y"), "n")


def test_optional_step() -> None:
    log = _log({("a", "b", "c"): 3, ("a", "c"): 2})
    assert discover_model(log).tree == seq("a", xor(TAU, "b"), "c")


def test_single_activity_repeats() -> None:
    log = _log({("a",): 2, ("a", "a"): 2, ("a", "a", "a"): 1})
    assert discover_model(log).tree == loop("a", TAU)


def test_structureless_log_falls_through_to_flower() -> None:
    model = discover_model(_log(_STRUCTURELESS))
    assert model.tree == loop(TAU, xor("a", "b", "c"))
    assert model.fallthroughs == ((),)


# -- the infrequent filter -------------------------------------------------------


def test_infrequent_filter_recovers_nested_world() -> None:
    clean = discover_model(_log(_NESTED_CLEAN)).tree
    noisy = _log(_NESTED_NOISY)

    filtered = discover_model(noisy, noise=0.3)
    assert filtered.tree == clean
    assert filtered.n_filtered_arcs == 2  # the swap's x -> r and r -> n edges
    assert filtered.n_dropped_events == 0
    assert filtered.noise == 0.3

    literal = discover_model(noisy)
    assert literal.tree != clean
    # The swap's back edge welds r and x into one strongly connected block,
    # so the literal miner reads them as parallel with x and y optional.
    assert literal.tree == seq(par("r", xor(TAU, "x")), xor(TAU, "y"), "n")
    assert literal.n_filtered_arcs == 0


def test_infrequent_filter_drops_stray_cross_branch_event() -> None:
    clean = discover_model(_log(_BRANCHED_CLEAN)).tree
    assert clean == xor(seq("a", "b", "c"), seq("d", "e", "f"))
    noisy = _log(_BRANCHED_NOISY)

    filtered = discover_model(noisy, noise=0.3)
    assert filtered.tree == clean
    assert filtered.n_filtered_arcs == 6  # four edges at the root, two inside the branch
    assert filtered.n_dropped_events == 1  # the stray e, dropped by the choice projection
    assert filtered.dropped_share == 1 / noisy.n_rows
    assert filtered.fallthroughs == ()

    assert discover_model(noisy).tree != clean


# -- invariants ------------------------------------------------------------------


def test_activities_match_log() -> None:
    for variants in _ALL_WORLDS:
        log = _log(variants)
        for noise in (0.0, 0.3):
            tree = discover_model(log, noise=noise).tree
            assert set(tree.activities()) == set(log.activities)


def test_determinism() -> None:
    for variants in (_NESTED_NOISY, _BRANCHED_NOISY, _STRUCTURELESS):
        log = _log(variants)
        assert discover_model(log) == discover_model(log)
        assert discover_model(log, noise=0.3) == discover_model(log, noise=0.3)


def test_net_is_the_tree_net() -> None:
    model = discover_model(_log(_NESTED_CLEAN))
    assert model.net == model.tree.to_net()


# -- the record ------------------------------------------------------------------


def test_repr() -> None:
    plain = discover_model(_log({("a", "b"): 2}))
    assert repr(plain) == "ProcessModel(tree=seq('a', 'b'), noise=0.0)"

    noisy = discover_model(_log(_BRANCHED_NOISY), noise=0.3)
    assert "filtered_arcs=6" in repr(noisy)
    assert "dropped_events=1" in repr(noisy)

    flower = discover_model(_log(_STRUCTURELESS))
    assert "flowers=" in repr(flower)


def test_dropped_share_arithmetic() -> None:
    noisy = _log(_BRANCHED_NOISY)
    model = discover_model(noisy, noise=0.3)
    assert model.dropped_share == model.n_dropped_events / noisy.n_rows


def test_noise_out_of_range_raises() -> None:
    log = _log({("a", "b"): 2})
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="noise"):
            discover_model(log, noise=bad)
