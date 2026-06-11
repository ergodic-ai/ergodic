# SPDX-License-Identifier: Apache-2.0
"""Tests for workflow-net soundness, and the pins for the tree/net contracts."""

from __future__ import annotations

import pytest

from ergodic.process._net import WorkflowNet, marking, net_of_tree
from ergodic.process._soundness import is_sound
from ergodic.process._tree import TAU, ProcessTree, loop, par, seq, xor

# -- fixtures ------------------------------------------------------------------


def _shape(net: WorkflowNet) -> tuple[int, int, int, int]:
    """(places, visible transitions, silent transitions, arcs) of ``net``."""
    n_silent = sum(1 for _, label in net.transitions if label is None)
    return (len(net.places), len(net.transitions) - n_silent, n_silent, len(net.arcs))


def _chain_net() -> WorkflowNet:
    """Hand-built strictly sequential net: source -> a -> p -> b -> sink."""
    return WorkflowNet(
        places=("source", "p", "sink"),
        transitions=(("t_a", "a"), ("t_b", "b")),
        arcs=(("source", "t_a"), ("t_a", "p"), ("p", "t_b"), ("t_b", "sink")),
        source="source",
        sink="sink",
    )


def _par_split_xor_join() -> WorkflowNet:
    """Parallel split joined by an exclusive join: both branches run, the join
    fires once per token, and a run ends with a token left behind."""
    return WorkflowNet(
        places=("source", "p1", "p2", "p3", "sink"),
        transitions=(("t_split", None), ("t_a", "a"), ("t_b", "b"), ("t_end", None)),
        arcs=(
            ("source", "t_split"),
            ("t_split", "p1"),
            ("t_split", "p2"),
            ("p1", "t_a"),
            ("t_a", "p3"),
            ("p2", "t_b"),
            ("t_b", "p3"),
            ("p3", "t_end"),
            ("t_end", "sink"),
        ),
        source="source",
        sink="sink",
    )


def _xor_split_par_join() -> WorkflowNet:
    """Exclusive split joined by a parallel join: only one of p1/p2 ever holds
    a token, so the join is dead and the net deadlocks."""
    return WorkflowNet(
        places=("source", "p1", "p2", "sink"),
        transitions=(("t_a", "a"), ("t_b", "b"), ("t_join", None)),
        arcs=(
            ("source", "t_a"),
            ("t_a", "p1"),
            ("source", "t_b"),
            ("t_b", "p2"),
            ("p1", "t_join"),
            ("p2", "t_join"),
            ("t_join", "sink"),
        ),
        source="source",
        sink="sink",
    )


def _dead_transition_net() -> WorkflowNet:
    """The chain of :func:`_chain_net` plus ``t_d``, a silent transition with
    pre {source, p} and post {p}: structurally on a source-to-sink path, but
    one token can never sit on both pre places, so it never fires."""
    return WorkflowNet(
        places=("source", "p", "sink"),
        transitions=(("t_a", "a"), ("t_b", "b"), ("t_d", None)),
        arcs=(
            ("source", "t_a"),
            ("t_a", "p"),
            ("p", "t_b"),
            ("t_b", "sink"),
            ("source", "t_d"),
            ("p", "t_d"),
            ("t_d", "p"),
        ),
        source="source",
        sink="sink",
    )


# -- net_of_tree shapes ----------------------------------------------------------


def test_net_of_tree_shapes() -> None:
    cases = [
        (seq("a", "b"), (3, 2, 0, 4)),
        (xor("a", "b"), (2, 2, 0, 4)),
        (xor("a", TAU), (2, 1, 1, 4)),
        (par("a", "b"), (6, 2, 2, 10)),
        (loop("a", "b"), (4, 2, 2, 8)),
        (seq("a", xor("b", par("c", "d"))), (7, 4, 2, 14)),
    ]
    for tree, expected in cases:
        net = tree.to_net()
        assert _shape(net) == expected, repr(tree)
        assert net.source == "source"
        assert net.sink == "sink"


def test_xor_net_transition_labels() -> None:
    net = xor("a", "b").to_net()
    assert {label for _, label in net.transitions} == {"a", "b"}


def test_par_and_loop_add_exactly_two_silent_transitions() -> None:
    for tree in (par("a", "b"), par("a", "b", "c"), loop("a", "b")):
        net = tree.to_net()
        assert sum(1 for _, label in net.transitions if label is None) == 2, repr(tree)


def test_net_of_tree_matches_to_net_and_is_deterministic() -> None:
    tree = seq("a", xor("b", par("c", "d")))
    assert net_of_tree(tree) == tree.to_net()
    assert tree.to_net() == tree.to_net()


# -- soundness -------------------------------------------------------------------


def test_tree_conversions_are_sound() -> None:
    trees = [
        seq("a", "b"),
        xor("a", "b"),
        xor("a", TAU),
        par("a", "b"),
        loop("a", "b"),
        seq("a", xor("b", par("c", "d"))),
        loop(TAU, xor("a", "b", "c")),  # the flower
    ]
    for tree in trees:
        assert is_sound(tree.to_net()) is True, repr(tree)


def test_hand_built_chain_is_sound() -> None:
    assert is_sound(_chain_net()) is True


def test_par_split_xor_join_is_unsound() -> None:
    # Proper completion and option to complete both fail: the only runs end
    # at markings like (p3, sink) or (sink, 2), never at exactly (sink, 1).
    assert is_sound(_par_split_xor_join()) is False


def test_xor_split_par_join_is_unsound() -> None:
    # The final marking is unreachable and t_join is dead.
    assert is_sound(_xor_split_par_join()) is False


def test_dead_transition_alone_breaks_soundness() -> None:
    net = _dead_transition_net()
    idx = net.index()
    # Walk the whole reachable space by hand: three markings, t_d in none.
    m0 = net.initial_marking
    assert idx.enabled(m0) == ("t_a",)
    m1 = idx.fire(m0, "t_a")
    assert m1 == (("p", 1),)
    assert idx.enabled(m1) == ("t_b",)
    m2 = idx.fire(m1, "t_b")
    assert m2 == net.final_marking
    assert idx.enabled(m2) == ()
    # So every marking reaches the final marking and only the final marking
    # touches the sink: clauses (a) and (b) hold, and the False below can
    # only come from clause (c), the dead transition.
    assert is_sound(net) is False
    # The same chain without t_d is sound.
    assert is_sound(_chain_net()) is True


def test_state_cap_raises() -> None:
    with pytest.raises(ValueError, match="max_states"):
        is_sound(par("a", "b").to_net(), max_states=2)


def test_is_sound_is_deterministic() -> None:
    sound = seq("a", xor("b", par("c", "d"))).to_net()
    assert is_sound(sound) is is_sound(sound) is True
    broken = _xor_split_par_join()
    assert is_sound(broken) is is_sound(broken) is False


# -- WorkflowNet validation ------------------------------------------------------


def test_net_duplicate_arcs_raise() -> None:
    with pytest.raises(ValueError, match="duplicate arcs"):
        WorkflowNet(
            places=("source", "sink"),
            transitions=(("t", "a"),),
            arcs=(("source", "t"), ("source", "t"), ("t", "sink")),
            source="source",
            sink="sink",
        )


def test_net_place_to_place_arc_raises() -> None:
    with pytest.raises(ValueError, match="place and a transition"):
        WorkflowNet(
            places=("source", "sink"),
            transitions=(("t", "a"),),
            arcs=(("source", "t"), ("t", "sink"), ("source", "sink")),
            source="source",
            sink="sink",
        )


def test_net_name_collision_raises() -> None:
    with pytest.raises(ValueError, match="both place and transition"):
        WorkflowNet(
            places=("source", "sink", "x"),
            transitions=(("x", "a"),),
            arcs=(),
            source="source",
            sink="sink",
        )


def test_net_stranded_node_raises() -> None:
    with pytest.raises(ValueError, match="orphan"):
        WorkflowNet(
            places=("source", "sink", "orphan"),
            transitions=(("t", "a"),),
            arcs=(("source", "t"), ("t", "sink")),
            source="source",
            sink="sink",
        )


def test_net_arc_into_source_raises() -> None:
    with pytest.raises(ValueError, match="incoming"):
        WorkflowNet(
            places=("source", "sink"),
            transitions=(("t", "a"),),
            arcs=(("source", "t"), ("t", "sink"), ("t", "source")),
            source="source",
            sink="sink",
        )


def test_net_arc_out_of_sink_raises() -> None:
    with pytest.raises(ValueError, match="outgoing"):
        WorkflowNet(
            places=("source", "sink"),
            transitions=(("t", "a"),),
            arcs=(("source", "t"), ("t", "sink"), ("sink", "t")),
            source="source",
            sink="sink",
        )


def test_net_source_equals_sink_raises() -> None:
    with pytest.raises(ValueError, match="different"):
        WorkflowNet(
            places=("source",),
            transitions=(),
            arcs=(),
            source="source",
            sink="source",
        )


# -- marking and NetIndex --------------------------------------------------------


def test_marking_canonicalizes() -> None:
    assert marking({"b": 1, "a": 2, "c": 0}) == (("a", 2), ("b", 1))
    assert marking([("b", 1), ("a", 2), ("c", 0)]) == (("a", 2), ("b", 1))
    assert marking([("a", 1), ("a", 2)]) == (("a", 3),)
    assert marking({}) == ()


def test_marking_negative_count_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        marking({"a": -1})


def test_index_enabled_and_fire() -> None:
    net = seq("a", "b").to_net()
    idx = net.index()
    assert idx.enabled(net.initial_marking) == ("t0",)
    assert idx.labels["t0"] == "a"
    after = idx.fire(net.initial_marking, "t0")
    assert after == (("p0", 1),)
    assert idx.enabled(after) == ("t1",)
    assert idx.fire(after, "t1") == net.final_marking


def test_index_fire_not_enabled_raises() -> None:
    net = seq("a", "b").to_net()
    with pytest.raises(ValueError, match="not enabled"):
        net.index().fire(net.initial_marking, "t1")


# -- ProcessTree validation and semantics ----------------------------------------


def test_tree_loop_with_three_children_raises() -> None:
    leaf = ProcessTree(label="a")
    with pytest.raises(ValueError, match="exactly two"):
        ProcessTree(op="loop", children=(leaf, leaf, leaf))


def test_tree_operator_with_one_child_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        seq("a")
    with pytest.raises(ValueError, match="at least two"):
        xor("a")


def test_tree_leaf_with_children_raises() -> None:
    with pytest.raises(ValueError, match="leaf has no children"):
        ProcessTree(label="a", children=(TAU,))


def test_tree_unknown_op_raises() -> None:
    with pytest.raises(ValueError, match="unknown operator"):
        ProcessTree(op="opt", children=(ProcessTree(label="a"), ProcessTree(label="b")))


def test_tree_xor_and_par_children_canonicalize() -> None:
    assert xor("b", "a") == xor("a", "b")
    assert par("b", "a") == par("a", "b")
    # seq and loop keep semantic order.
    assert seq("b", "a") != seq("a", "b")
    assert loop("b", "a") != loop("a", "b")


def test_tree_repr() -> None:
    assert repr(seq("a", xor("b", TAU))) == "seq('a', xor(TAU, 'b'))"
