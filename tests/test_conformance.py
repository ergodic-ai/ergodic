# SPDX-License-Identifier: Apache-2.0
"""Tests for conformance checking: token replay and optimal alignments."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from ergodic.data import EventLog, event_log
from ergodic.process._conformance import (
    ConformanceResult,
    VariantConformance,
    _align_variant,
    _alignment_fitness,
    _by_label,
    _model_path_cost,
    conformance,
)
from ergodic.process._net import WorkflowNet
from ergodic.process._tree import TAU, loop, par, seq, xor

# -- fixtures ------------------------------------------------------------------


def _log_of(*traces: tuple[str, ...]) -> EventLog:
    """One case per trace, events at t = 0, 1, 2, ..."""
    cases: list[str] = []
    acts: list[str] = []
    stamps: list[float] = []
    for i, trace in enumerate(traces):
        for j, act in enumerate(trace):
            cases.append(f"c{i:03d}")
            acts.append(act)
            stamps.append(float(j))
    df = pd.DataFrame({"case": cases, "act": acts, "ts": stamps})
    return event_log(df, case="case", activity="act", timestamp="ts")


def _duplicate_label_net() -> WorkflowNet:
    """Two transitions labeled 'a': t0 leads on to 'b', t1 jumps straight to the sink."""
    return WorkflowNet(
        places=("source", "p0", "sink"),
        transitions=(("t0", "a"), ("t1", "a"), ("t2", "b")),
        arcs=(
            ("source", "t0"),
            ("t0", "p0"),
            ("p0", "t2"),
            ("t2", "sink"),
            ("source", "t1"),
            ("t1", "sink"),
        ),
        source="source",
        sink="sink",
    )


def _deadlocking_net() -> WorkflowNet:
    """A parallel split whose branches each feed the sink: {sink: 1} is never reached."""
    return WorkflowNet(
        places=("source", "p1", "p2", "sink"),
        transitions=(("t0", None), ("t1", "a"), ("t2", "b")),
        arcs=(
            ("source", "t0"),
            ("t0", "p1"),
            ("t0", "p2"),
            ("p1", "t1"),
            ("t1", "sink"),
            ("p2", "t2"),
            ("t2", "sink"),
        ),
        source="source",
        sink="sink",
    )


# -- fitting behavior ------------------------------------------------------------


def test_fitting_log_is_perfect() -> None:
    log = _log_of(("a", "b", "c"), ("a", "b", "c"), ("a", "b", "c"))
    res = conformance(log, seq("a", "b", "c"))
    assert res.fitness == 1.0
    assert res.alignment_fitness == 1.0
    assert res.n_cases == 3
    assert res.n_variants == 1
    assert res.n_fitting == 3
    assert res.share_fitting == 1.0
    v = res.variants[0]
    assert isinstance(v, VariantConformance)
    assert v.sequence == ("a", "b", "c")
    assert v.n_cases == 3
    assert (v.missing, v.remaining, v.consumed, v.produced) == (0, 0, 4, 4)
    assert v.replay_fitness == 1.0
    assert v.cost == 0
    assert v.worst_cost == 6  # 3 trace events plus the 3-visible-step model run
    assert v.alignment_fitness == 1.0
    assert v.alignment == (("sync", "a"), ("sync", "b"), ("sync", "c"))


def test_optional_step_takes_the_silent_branch() -> None:
    # seq(a, xor(TAU, b), c): the trace (a, c) runs through tau at no cost.
    res = conformance(_log_of(("a", "c")), seq("a", xor(TAU, "b"), "c"))
    assert res.fitness == 1.0
    assert res.alignment_fitness == 1.0
    assert res.n_fitting == 1
    v = res.variants[0]
    assert (v.missing, v.remaining, v.consumed, v.produced) == (0, 0, 4, 4)
    assert v.cost == 0
    assert v.worst_cost == 4  # 2 trace events plus the 2-visible run a, tau, c
    assert v.alignment == (("sync", "a"), ("sync", "c"))  # the silent move is omitted


def test_parallel_interleavings_both_fit() -> None:
    res = conformance(_log_of(("a", "b"), ("b", "a")), par("a", "b"))
    assert res.fitness == 1.0
    assert res.alignment_fitness == 1.0
    assert res.n_fitting == 2
    by_seq = {v.sequence: v for v in res.variants}
    for sequence in (("a", "b"), ("b", "a")):
        v = by_seq[sequence]
        # split and join are silent firings: they count 1+2 and 2+1 tokens.
        assert (v.missing, v.remaining, v.consumed, v.produced) == (0, 0, 6, 6)
        assert v.cost == 0
        assert v.worst_cost == 4
    assert by_seq[("a", "b")].alignment == (("sync", "a"), ("sync", "b"))
    assert by_seq[("b", "a")].alignment == (("sync", "b"), ("sync", "a"))


def test_loop_replays_zero_and_one_redo_rounds() -> None:
    res = conformance(_log_of(("a",), ("a", "b", "a")), loop("a", "b"))
    assert res.fitness == 1.0
    assert res.alignment_fitness == 1.0
    assert res.n_fitting == 2
    by_seq = {v.sequence: v for v in res.variants}
    short, long = by_seq[("a",)], by_seq[("a", "b", "a")]
    assert (short.missing, short.remaining, short.consumed, short.produced) == (0, 0, 4, 4)
    assert short.cost == 0
    assert short.worst_cost == 2  # 1 event plus the 1-visible run tau, a, tau
    assert short.alignment == (("sync", "a"),)
    assert (long.missing, long.remaining, long.consumed, long.produced) == (0, 0, 6, 6)
    assert long.cost == 0
    assert long.worst_cost == 4
    assert long.alignment == (("sync", "a"), ("sync", "b"), ("sync", "a"))


# -- deviations, hand-computed ----------------------------------------------------


def test_skipped_step_is_one_model_move() -> None:
    # seq(a, b, c) is the chain source -> a -> p0 -> b -> p1 -> c -> sink.
    # Replaying (a, c): c's pre-place p1 is empty, so 1 missing token; after the
    # final consume the token produced by a is still on p0, so 1 remaining.
    # Consumed: a, c, and the environment's take = 3. Produced: the initial
    # token plus a's and c's outputs = 3.
    res = conformance(_log_of(("a", "c")), seq("a", "b", "c"))
    v = res.variants[0]
    assert (v.missing, v.remaining, v.consumed, v.produced) == (1, 1, 3, 3)
    assert v.replay_fitness == pytest.approx(2 / 3)  # 0.5*(1 - 1/3) + 0.5*(1 - 1/3)
    assert res.fitness == pytest.approx(2 / 3)
    assert v.cost == 1
    assert v.alignment == (("sync", "a"), ("model", "b"), ("sync", "c"))
    assert v.alignment.count(("model", "b")) == 1
    assert v.worst_cost == 5  # 2 trace events plus the 3-visible model run
    assert v.alignment_fitness == pytest.approx(0.8)
    assert res.n_fitting == 0
    assert res.share_fitting == 0.0


def test_off_model_label_is_one_log_move() -> None:
    # x exists in the trace but no transition carries it: the replay counts
    # one missing and one consumed token (the unmatchable consume), so
    # consumed is 5 (a, b, x, c, environment) while produced stays 4.
    res = conformance(_log_of(("a", "b", "x", "c")), seq("a", "b", "c"))
    v = res.variants[0]
    assert (v.missing, v.remaining, v.consumed, v.produced) == (1, 0, 5, 4)
    assert v.replay_fitness == pytest.approx(0.9)  # 0.5*(1 - 1/5) + 0.5*(1 - 0/4)
    assert res.fitness == pytest.approx(0.9)
    assert v.cost == 1
    assert v.alignment == (("sync", "a"), ("sync", "b"), ("log", "x"), ("sync", "c"))
    assert v.worst_cost == 7
    assert v.alignment_fitness == pytest.approx(6 / 7)


def test_swapped_pair_costs_two() -> None:
    # (b, a) on seq(a, b): two optimal alignments exist at cost 2; the
    # deterministic tie-break lands on model-a, sync-b, log-a.
    res = conformance(_log_of(("b", "a")), seq("a", "b"))
    v = res.variants[0]
    assert v.cost == 2
    assert Counter(v.alignment) == Counter([("model", "a"), ("sync", "b"), ("log", "a")])
    assert v.worst_cost == 4
    assert v.alignment_fitness == 0.5
    assert (v.missing, v.remaining, v.consumed, v.produced) == (1, 1, 3, 3)
    assert v.replay_fitness == pytest.approx(2 / 3)


def test_duplicate_labels_pick_the_lowest_id_among_cheapest() -> None:
    # Both t0 and t1 carry 'a' and both are enabled, so the tid tie-break
    # fires t0; the trace (a, b) then fits perfectly. Firing t1 instead
    # would strand a token and miss one for b.
    res = conformance(_log_of(("a", "b")), _duplicate_label_net())
    assert res.fitness == 1.0
    v = res.variants[0]
    assert (v.missing, v.remaining, v.consumed, v.produced) == (0, 0, 3, 3)
    assert v.cost == 0
    assert v.worst_cost == 3  # the cheapest model run is t1 alone: 1 visible firing
    assert v.alignment == (("sync", "a"), ("sync", "b"))


# -- aggregation -----------------------------------------------------------------


def test_weighted_aggregation_over_variants() -> None:
    # Three fitting (a, b, c) cases and one (a, c) case. Rozinat's log
    # fitness aggregates the token sums: missing = 3*0 + 1*1 = 1 against
    # consumed = 3*4 + 1*3 = 15, and the same on the remaining/produced side,
    # so fitness = 0.5*(1 - 1/15) + 0.5*(1 - 1/15) = 14/15. The alignment
    # side is the case-weighted mean: (3*1.0 + 1*0.8)/4 = 0.95.
    log = _log_of(("a", "b", "c"), ("a", "b", "c"), ("a", "b", "c"), ("a", "c"))
    res = conformance(log, seq("a", "b", "c"))
    assert res.fitness == pytest.approx(14 / 15)
    assert res.alignment_fitness == pytest.approx(0.95)
    assert res.n_cases == 4
    assert res.n_variants == 2
    assert res.n_fitting == 3
    assert res.share_fitting == 0.75
    assert res.variants[0].sequence == ("a", "b", "c")  # most frequent first
    assert res.variants[0].n_cases == 3
    assert res.variants[1].sequence == ("a", "c")
    assert res.variants[1].n_cases == 1


def test_variant_ties_order_by_sequence() -> None:
    res = conformance(_log_of(("b", "a"), ("a", "b")), seq("a", "b"))
    assert [v.sequence for v in res.variants] == [("a", "b"), ("b", "a")]


# -- inputs and guards -----------------------------------------------------------


def test_tree_and_net_inputs_agree() -> None:
    log = _log_of(("a", "b", "c"), ("a", "c"))
    tree = seq("a", "b", "c")
    assert conformance(log, tree) == conformance(log, tree.to_net())


def test_other_model_types_raise() -> None:
    with pytest.raises(TypeError, match="ProcessModel, a ProcessTree, or a WorkflowNet"):
        conformance(_log_of(("a",)), "not a model")  # type: ignore[arg-type]


def test_max_states_cap_raises() -> None:
    with pytest.raises(ValueError, match="max_states"):
        conformance(_log_of(("a", "b", "c")), seq("a", "b", "c"), max_states=1)


def test_model_that_cannot_complete_raises() -> None:
    with pytest.raises(ValueError, match="final marking"):
        conformance(_log_of(("a", "b")), _deadlocking_net())


def test_empty_trace_guards() -> None:
    # Log cases always hold at least one event; the internal pieces still
    # handle an empty trace on a model that can run silently end to end.
    net = xor(TAU, "b").to_net()
    index = net.index()
    assert _model_path_cost(net, index, 100) == 0
    assert _align_variant(net, index, _by_label(index), (), 100) == (0, ())
    assert _alignment_fitness(0, 0) == 1.0


# -- repr and determinism ---------------------------------------------------------


def test_repr_reports_fitness_and_fitting_share() -> None:
    log = _log_of(("a", "b", "c"), ("a", "b", "c"), ("a", "b", "c"), ("a", "c"))
    res = conformance(log, seq("a", "b", "c"))
    text = repr(res)
    assert text.startswith("ConformanceResult(")
    assert "fitness=0.933" in text
    assert "alignment_fitness=0.950" in text
    assert "fitting=3/4 cases" in text


def test_repeated_calls_give_equal_objects() -> None:
    log = _log_of(("a", "b", "c"), ("a", "c"), ("b", "a", "c"))
    tree = seq("a", "b", "c")
    first = conformance(log, tree)
    second = conformance(log, tree)
    assert isinstance(first, ConformanceResult)
    assert first == second
    assert first.variants == second.variants
