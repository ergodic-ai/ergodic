# SPDX-License-Identifier: Apache-2.0
"""Tests for the PAG toolkit: definite-status separation and the conversions."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest

from ergodic.graph import (
    Edge,
    GraphKind,
    InvalidGraphError,
    Mark,
    MixedGraph,
    admg,
    cpdag_of_dag,
    dag,
    dags_of_cpdag,
    fci_closure,
    mag_of_dag,
    mag_of_pag,
    mags_of_pag,
    mixed,
    pag,
    pag_of_mag,
    possible_ancestors,
    possible_descendants,
    visible_edges,
)
from ergodic.graph._separation import _definitely_connected, _plainly_connected

# -- m-separation on PAGs ------------------------------------------------------


def test_a_circle_collider_blocks_marginally_and_opens_when_conditioned() -> None:
    graph = pag(["X o-> Y", "Z o-> Y"])
    assert graph.m_separated("X", "Z")
    assert not graph.m_separated("X", "Z", "Y")


def test_a_collider_opens_through_a_directed_descendant() -> None:
    graph = pag(["X o-> Y", "Z o-> Y", "Y -> W"])
    assert graph.m_separated("X", "Z")
    assert not graph.m_separated("X", "Z", "W")


def test_an_unshielded_circle_vertex_is_a_definite_non_collider() -> None:
    graph = pag(["X o-o Y", "Y o-o Z"])
    assert not graph.m_separated("X", "Z")
    assert graph.m_separated("X", "Z", "Y")


def test_a_vertex_without_definite_status_blocks() -> None:
    # Arrow on one side, circle on the other, unshielded: not definite, so the
    # conservative walker blocks. Built with mixed(): this graph is not closed
    # under R1, so it is not a valid complete PAG.
    graph = mixed(["X o-> V", "V o-o Y"])
    assert graph.m_separated("X", "Y")


def test_pag_separation_agrees_with_a_member_of_the_class() -> None:
    graph = pag(["X o-> Y", "Z o-> Y"])
    member = dag(["X -> Y", "Z -> Y"])
    for given in [frozenset(), frozenset({"Y"})]:
        assert graph.m_separated("X", "Z", given) == member.m_separated("X", "Z", given)


def test_pag_queries_validate_names_and_disjointness() -> None:
    graph = pag(["X o-o Y"])
    with pytest.raises(ValueError, match="unknown node"):
        graph.m_separated("X", "Q")
    with pytest.raises(ValueError, match="disjoint"):
        graph.m_separated("X", "Y", "X")


def _random_admg(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        draw = rng.uniform()
        if draw < 0.30:
            edges.append(Edge.make(names[i], Mark.TAIL, names[j], Mark.ARROW))
        elif draw < 0.40:
            edges.append(Edge.make(names[i], Mark.ARROW, names[j], Mark.ARROW))
    return admg(edges, nodes=names)


def test_the_definite_walker_reduces_to_the_plain_walker_without_circles() -> None:
    rng = np.random.default_rng(0)
    for _ in range(30):
        graph = _random_admg(rng, 6)
        nodes = graph.node_list()
        for x, y in combinations(nodes, 2):
            rest = [n for n in nodes if n not in (x, y)]
            for size in (0, 1, 2):
                for given in combinations(rest, size):
                    sources, targets = frozenset({x}), frozenset({y})
                    conditioned = frozenset(given)
                    assert _definitely_connected(
                        graph, sources, targets, conditioned
                    ) == _plainly_connected(graph, sources, targets, conditioned)


# -- mag_of_dag ----------------------------------------------------------------


def test_a_pure_latent_confounder_projects_to_a_bidirected_edge() -> None:
    graph = dag(["L -> X", "L -> Y"], latent=["L"])
    mag = mag_of_dag(graph)
    assert mag.nodes == frozenset({"X", "Y"})
    edge = mag.get_edge("X", "Y")
    assert edge is not None and edge.is_bidirected


def test_a_confounded_direct_effect_keeps_its_directed_edge() -> None:
    graph = dag(["L -> X", "L -> Y", "X -> Y"], latent=["L"])
    edge = mag_of_dag(graph).get_edge("X", "Y")
    assert edge is not None and edge.directed_pair() == ("X", "Y")


def test_a_latent_into_one_node_only_vanishes() -> None:
    graph = dag(["Z -> X", "X -> Y", "L -> Y"], latent=["L"])
    mag = mag_of_dag(graph)
    assert {e.directed_pair() for e in mag.edges} == {("Z", "X"), ("X", "Y")}


def test_the_instrument_graph_projects_to_the_complete_mag() -> None:
    graph = dag(["Z -> X", "X -> Y", "L -> X", "L -> Y"], latent=["L"])
    mag = mag_of_dag(graph)
    pairs = {
        ("X", "Y"): ("X", "Y"),
        ("X", "Z"): ("Z", "X"),
        ("Y", "Z"): ("Z", "Y"),
    }
    assert len(mag.edges) == 3
    for (a, b), directed in pairs.items():
        edge = mag.get_edge(a, b)
        assert edge is not None and edge.directed_pair() == directed


def test_mag_of_dag_without_latents_is_the_dag_itself() -> None:
    graph = dag(["X -> M", "M -> Y"])
    assert mag_of_dag(graph).edges == graph.edges


def _random_dag_with_latents(rng: np.random.Generator, p: int, n_latent: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        if rng.uniform() < 0.4:
            edges.append((names[i], names[j]))
    latent = list(rng.choice(names, size=n_latent, replace=False))
    return dag(edges, nodes=names, latent=latent)


def test_the_projection_preserves_every_separation_over_the_observed() -> None:
    rng = np.random.default_rng(1)
    for _ in range(25):
        graph = _random_dag_with_latents(rng, 6, 2)
        mag = mag_of_dag(graph)
        observed = sorted(mag.nodes)
        for x, y in combinations(observed, 2):
            rest = [n for n in observed if n not in (x, y)]
            for size in range(len(rest) + 1):
                for given in combinations(rest, size):
                    assert graph.m_separated(x, y, given) == mag.m_separated(x, y, given)


def test_mag_of_dag_rejects_non_admg_and_bad_selection_inputs() -> None:
    with pytest.raises(ValueError, match="DAG or ADMG"):
        mag_of_dag(mixed(["X -- Y"]))
    stray = MixedGraph(
        nodes=frozenset({"X", "Y"}),
        edges=dag(["X -> Y"]).edges,
        selection=frozenset({"S"}),
    )
    with pytest.raises(ValueError, match="selection names"):
        mag_of_dag(stray)
    both = MixedGraph(
        nodes=frozenset({"X", "Y", "S"}),
        edges=dag(["X -> Y"]).edges,
        latent=frozenset({"S"}),
        selection=frozenset({"S"}),
    )
    with pytest.raises(ValueError, match="both latent and selection"):
        mag_of_dag(both)


def test_mag_of_dag_accepts_an_admg_with_its_implicit_latents() -> None:
    # A bidirected edge is an implicit latent confounder; this ADMG is already
    # maximal and ancestral, so it projects to itself.
    graph = admg(["X <-> Y", "Y -> Z"])
    assert mag_of_dag(graph).edges == graph.edges


# -- the FCI rule closure --------------------------------------------------------


def _never(d: str, a: str, b: str, c: str) -> bool | None:
    return None


def _mark(graph: MixedGraph, a: str, b: str, at: str) -> Mark | None:
    return graph.mark_at(a, b, at)


def test_r1_orients_into_the_unshielded_circle() -> None:
    out = fci_closure(mixed(["A o-> B", "B o-o C"]), discriminating=_never)
    assert _mark(out, "B", "C", "B") is Mark.TAIL
    assert _mark(out, "B", "C", "C") is Mark.ARROW
    assert _mark(out, "A", "B", "A") is Mark.CIRCLE  # untouched


def test_r2_first_form_adds_the_arrowhead_along_a_directed_chain() -> None:
    out = fci_closure(mixed(["A -> B", "B <-> C", "A o-o C"]), discriminating=_never)
    assert _mark(out, "A", "C", "C") is Mark.ARROW
    assert _mark(out, "A", "C", "A") is Mark.CIRCLE


def test_r2_second_form_adds_the_arrowhead_too() -> None:
    out = fci_closure(mixed(["A o-> B", "B -> C", "A o-o C"]), discriminating=_never)
    assert _mark(out, "A", "C", "C") is Mark.ARROW
    assert _mark(out, "A", "C", "A") is Mark.CIRCLE


def test_r3_orients_the_middle_vertex_into_the_collider() -> None:
    out = fci_closure(
        mixed(["A o-> B", "C o-> B", "A o-o D", "D o-o C", "D o-o B"]),
        discriminating=_never,
    )
    assert _mark(out, "D", "B", "B") is Mark.ARROW
    assert _mark(out, "D", "B", "D") is Mark.CIRCLE


def test_r4_asks_the_callback_and_orients_the_non_collider_branch() -> None:
    asked: list[tuple[str, str, str, str]] = []

    def saw(d: str, a: str, b: str, c: str) -> bool | None:
        asked.append((d, a, b, c))
        return False

    out = fci_closure(mixed(["D o-> A", "A <-> B", "A -> C", "B o-o C"]), discriminating=saw)
    assert ("D", "A", "B", "C") in asked
    assert _mark(out, "B", "C", "B") is Mark.TAIL
    assert _mark(out, "B", "C", "C") is Mark.ARROW


def test_r4_collider_branch_orients_both_bidirected_edges() -> None:
    out = fci_closure(
        mixed(["D o-> A", "A <-> B", "A -> C", "B o-o C"]),
        discriminating=lambda d, a, b, c: True,
    )
    edge_ab = out.get_edge("A", "B")
    edge_bc = out.get_edge("B", "C")
    assert edge_ab is not None and edge_ab.is_bidirected
    assert edge_bc is not None and edge_bc.is_bidirected


def test_r8_turns_the_circle_into_a_tail_behind_a_directed_chain() -> None:
    out = fci_closure(mixed(["A -> B", "B -> C", "A o-> C"]), discriminating=_never)
    edge = out.get_edge("A", "C")
    assert edge is not None and edge.directed_pair() == ("A", "C")


def test_r9_uses_an_uncovered_possibly_directed_path() -> None:
    out = fci_closure(mixed(["A o-> C", "A o-o M", "M o-o N", "N -> C"]), discriminating=_never)
    assert _mark(out, "A", "C", "A") is Mark.TAIL


def test_r10_uses_two_unshielded_possibly_directed_first_steps() -> None:
    out = fci_closure(
        mixed(["A o-> C", "A o-o B", "A o-o D", "B -> C", "D -> C"]),
        discriminating=_never,
    )
    assert _mark(out, "A", "C", "A") is Mark.TAIL


def test_the_closure_never_moves_a_definite_mark() -> None:
    graph = mixed(["A -> B", "B <-> C", "A o-o C"])
    out = fci_closure(graph, discriminating=_never)
    assert _mark(out, "A", "B", "A") is Mark.TAIL
    assert _mark(out, "A", "B", "B") is Mark.ARROW
    assert _mark(out, "B", "C", "B") is Mark.ARROW


# -- the conversions and the round trip ------------------------------------------


def test_the_canonical_pag_of_the_collider_keeps_its_arrowheads_only() -> None:
    out = pag_of_mag(dag(["X -> Y", "Z -> Y"]))
    assert {e.glyph() for e in out.edges} == {"X o-> Y", "Y <-o Z"}


def test_the_canonical_pag_of_the_chain_is_all_circles() -> None:
    out = pag_of_mag(dag(["X -> Y", "Y -> Z"]))
    assert {e.glyph() for e in out.edges} == {"X o-o Y", "Y o-o Z"}


def test_pag_of_mag_exercises_the_discriminating_path_collider_branch() -> None:
    mag = mixed(["D -> A", "A <-> B", "A -> C", "B <-> C"]).as_kind(GraphKind.MAG)
    out = pag_of_mag(mag)
    assert {e.glyph() for e in out.edges} == {"A <-o D", "A <-> B", "A --> C", "B <-> C"}


def test_pag_of_mag_exercises_the_discriminating_path_tail_branch() -> None:
    mag = mixed(["D -> A", "A <-> B", "A -> C", "B -> C"]).as_kind(GraphKind.MAG)
    out = pag_of_mag(mag)
    # The (A, B) mark at B stays a circle: B -> A in place of A <-> B gives a
    # Markov-equivalent MAG (same skeleton, colliders, and discriminating
    # status for B), so only the arrowhead at A is invariant.
    assert {e.glyph() for e in out.edges} == {"A <-o D", "A <-o B", "A --> C", "B --> C"}


def test_mag_of_pag_extracts_the_collider_exactly() -> None:
    member = mag_of_pag(pag(["X o-> Y", "Z o-> Y"]))
    assert {e.directed_pair() for e in member.edges} == {("X", "Y"), ("Z", "Y")}


def test_mag_of_pag_orients_the_circle_component_without_new_colliders() -> None:
    member = mag_of_pag(pag(["X o-o Y", "Y o-o Z"]))
    assert member.is_mag()
    collider = sum(
        1
        for z in member.nodes
        if len(member.parents(z)) == 2 and not member.has_edge(*sorted(member.parents(z)))
    )
    assert collider == 0


def test_mag_of_pag_resolves_selection_marks() -> None:
    # Undirected edges are invariant and stay; a circle-tail edge directs out
    # of its tail (Zhang's arrowhead augmentation).
    kept = mag_of_pag(mixed(["X -- Y"])).get_edge("X", "Y")
    assert kept is not None and kept.is_undirected
    out_of_x = mag_of_pag(mixed(["X --o Y"])).get_edge("X", "Y")
    assert out_of_x is not None and out_of_x.directed_pair() == ("X", "Y")
    out_of_y = mag_of_pag(mixed(["X o-- Y"])).get_edge("X", "Y")
    assert out_of_y is not None and out_of_y.directed_pair() == ("Y", "X")


def test_the_round_trip_holds_over_random_latent_projections() -> None:
    rng = np.random.default_rng(7)
    for _ in range(40):
        graph = _random_dag_with_latents(rng, 7, 2)
        mag = mag_of_dag(graph)
        canonical = pag_of_mag(mag)
        member = mag_of_pag(canonical)
        assert member.is_mag()
        assert {frozenset((e.u, e.v)) for e in member.edges} == {
            frozenset((e.u, e.v)) for e in mag.edges
        }
        assert pag_of_mag(member).edges == canonical.edges
        assert canonical.is_pag()


def test_definite_separation_on_the_canonical_pag_answers_for_the_class() -> None:
    rng = np.random.default_rng(11)
    for _ in range(15):
        graph = _random_dag_with_latents(rng, 6, 2)
        mag = mag_of_dag(graph)
        canonical = pag_of_mag(mag)
        observed = sorted(mag.nodes)
        for x, y in combinations(observed, 2):
            rest = [n for n in observed if n not in (x, y)]
            for size in (0, 1, 2):
                for given in combinations(rest, size):
                    assert canonical.m_separated(x, y, given) == mag.m_separated(x, y, given)


def test_is_pag_rejects_a_graph_not_closed_under_the_rules() -> None:
    assert not mixed(["X o-> Y", "Y o-o Z"]).is_pag()  # R1 would orient Y -> Z
    assert not mixed(["X -- Y"]).is_pag()  # a lone tail-tail is not invariant
    assert not dag(["X -> Y"]).is_pag()  # the class leaves both ends as circles
    with pytest.raises(InvalidGraphError):
        mixed(["X o-> Y", "Y o-o Z"]).as_kind(GraphKind.PAG)


def test_the_closure_is_deterministic() -> None:
    mag = mag_of_dag(dag(["Z -> X", "X -> Y", "L -> X", "L -> Y"], latent=["L"]))
    assert pag_of_mag(mag).edges == pag_of_mag(mag).edges


# -- selection bias --------------------------------------------------------------


def _random_dag_with_margin(
    rng: np.random.Generator, p: int, n_latent: int, n_selection: int
) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = []
    for i, j in combinations(range(p), 2):
        if rng.uniform() < 0.4:
            edges.append((names[i], names[j]))
    hidden = list(rng.choice(names, size=n_latent + n_selection, replace=False))
    return dag(edges, nodes=names, latent=hidden[:n_latent], selection=hidden[n_latent:])


_CYCLE_WORLD = [
    "A -> S1",
    "B -> S1",
    "B -> S2",
    "C -> S2",
    "C -> S3",
    "D -> S3",
    "D -> S4",
    "A -> S4",
]


def test_selection_on_a_collider_projects_to_an_undirected_edge() -> None:
    graph = dag(["X -> S", "Y -> S"], selection=["S"])
    mag = mag_of_dag(graph)
    assert sorted(mag.nodes) == ["X", "Y"]
    edge = mag.get_edge("X", "Y")
    assert edge is not None and edge.is_undirected
    # A lone edge carries no invariant marks, so the class shows circles.
    canonical = pag_of_mag(mag)
    pag_edge = canonical.get_edge("X", "Y")
    assert pag_edge is not None and pag_edge.glyph() == "X o-o Y"


def test_an_ancestor_of_selection_keeps_its_tail() -> None:
    # X -> Y -> S: X is an ancestor of Y and Y is an ancestor of selection,
    # so both ends are tails and the directed edge projects as undirected.
    graph = dag(["X -> Y", "Y -> S"], selection=["S"])
    edge = mag_of_dag(graph).get_edge("X", "Y")
    assert edge is not None and edge.is_undirected


def test_is_mag_reads_the_ancestral_conditions_on_undirected_edges() -> None:
    assert mixed(["X -- Y"]).is_mag()
    assert mixed(["X -- Y", "Y -- Z"]).is_mag()
    assert mixed(["X -- Y", "X -> Z"]).is_mag()  # an arrow out of X is fine
    assert not mixed(["X -- Y", "Z -> X"]).is_mag()  # a parent at an undirected endpoint
    assert not mixed(["X -- Y", "Z <-> Y"]).is_mag()  # a spouse at an undirected endpoint
    assert not mixed(["X o-o Y"]).is_mag()  # circles still fail


def test_the_selection_projection_preserves_separation_given_selection() -> None:
    # Richardson and Spirtes (2002), Theorem 4.18: m-separation in MAG(G) is
    # d-separation in G given Z union S.
    rng = np.random.default_rng(11)
    for _ in range(25):
        graph = _random_dag_with_margin(rng, 6, 1, 1)
        mag = mag_of_dag(graph)
        observed = sorted(mag.nodes)
        for x, y in combinations(observed, 2):
            rest = [n for n in observed if n not in (x, y)]
            for size in range(len(rest) + 1):
                for given in combinations(rest, size):
                    expected = graph.m_separated(x, y, frozenset(given) | graph.selection)
                    assert mag.m_separated(x, y, given) == expected


def test_the_round_trip_holds_over_random_selection_projections() -> None:
    rng = np.random.default_rng(13)
    n_undirected = 0
    for _ in range(40):
        graph = _random_dag_with_margin(rng, 7, 1, 2)
        mag = mag_of_dag(graph)
        n_undirected += any(e.is_undirected for e in mag.edges)
        canonical = pag_of_mag(mag)
        assert canonical.is_pag()
        member = mag_of_pag(canonical)
        assert member.is_mag()
        assert {frozenset((e.u, e.v)) for e in member.edges} == {
            frozenset((e.u, e.v)) for e in mag.edges
        }
        assert pag_of_mag(member).edges == canonical.edges
    assert n_undirected >= 10  # the worlds actually exercise selection


def test_r5_fires_on_the_chordless_circle_cycle() -> None:
    # Four pair-colliders, each selected: the MAG is the chordless undirected
    # four-cycle, and every mark in its PAG is an invariant tail, which only
    # R5 can conclude.
    graph = dag(_CYCLE_WORLD, selection=["S1", "S2", "S3", "S4"])
    mag = mag_of_dag(graph)
    glyphs = sorted(e.glyph() for e in mag.edges)
    assert glyphs == ["A --- B", "A --- D", "B --- C", "C --- D"]
    assert pag_of_mag(mag).edges == mag.edges
    assert pag(["A -- B", "B -- C", "C -- D", "D -- A"]).is_pag()


def test_r6_leaves_a_circle_tail_edge_and_the_round_trip_holds() -> None:
    # The four-cycle world plus B -> E: B's end of the new edge is a tail in
    # every member (R6), E's end varies, so the complete PAG carries B --o E.
    graph = dag([*_CYCLE_WORLD, "B -> E"], selection=["S1", "S2", "S3", "S4"])
    canonical = pag_of_mag(mag_of_dag(graph))
    edge = canonical.get_edge("B", "E")
    assert edge is not None and edge.glyph() == "B --o E"
    assert canonical.is_pag()
    member_edge = mag_of_pag(canonical).get_edge("B", "E")
    assert member_edge is not None and member_edge.directed_pair() == ("B", "E")


def test_mags_of_pag_selection_marks_need_the_wider_class() -> None:
    cycle = pag(["A -- B", "B -- C", "C -- D", "D -- A"])
    with pytest.raises(ValueError, match="selection=True"):
        mags_of_pag(cycle)
    members = mags_of_pag(cycle, selection=True)
    assert members == (mag_of_pag(cycle),)


# -- possible ancestry ----------------------------------------------------------


def _random_dag(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = [(names[i], names[j]) for i, j in combinations(range(p), 2) if rng.uniform() < 0.45]
    return dag(edges, nodes=names)


def test_possible_descendants_on_a_dag_are_plain_descendants() -> None:
    g = dag(["A -> B", "B -> C", "D -> C"])
    assert possible_descendants(g, "A") == g.descendants("A")
    assert possible_ancestors(g, "C") == g.ancestors("C")
    assert possible_descendants(g, ["A", "D"]) == frozenset({"B", "C"})


def test_possible_reachability_matches_the_members_of_a_cpdag() -> None:
    rng = np.random.default_rng(7)
    for _ in range(15):
        cp = cpdag_of_dag(_random_dag(rng, int(rng.choice([3, 4, 5]))))
        members = dags_of_cpdag(cp)
        for v in sorted(cp.nodes):
            assert possible_descendants(cp, v) == frozenset().union(
                *[m.descendants(v) for m in members]
            )
            assert possible_ancestors(cp, v) == frozenset().union(
                *[m.ancestors(v) for m in members]
            )


def test_possible_reachability_matches_the_member_mags_of_a_pag() -> None:
    rng = np.random.default_rng(11)
    for _ in range(8):
        p = pag_of_mag(mag_of_dag(_random_dag_with_latents(rng, 4, 1)))
        members = mags_of_pag(p)
        for v in sorted(p.nodes):
            assert possible_descendants(p, v) == frozenset().union(
                *[m.descendants(v) for m in members]
            )


def test_possible_reachability_validates_names() -> None:
    g = dag(["A -> B"])
    with pytest.raises(ValueError, match="unknown node names"):
        possible_descendants(g, "Q")


# -- visibility -----------------------------------------------------------------


def test_visible_edges_by_the_direct_clause() -> None:
    # C *-> A with C not adjacent to B certifies A -> B.
    m = mag_of_dag(dag(["C -> A", "A -> B"]))
    assert visible_edges(m) == frozenset({("A", "B")})


def test_visible_edges_by_the_collider_path_clause() -> None:
    # c *-> q <-> a with q a parent of b and c not adjacent to b certifies a -> b.
    m = mixed(["c -> q", "q <-> a", "q -> b", "a -> b"]).as_kind(GraphKind.MAG)
    assert ("a", "b") in visible_edges(m)
    assert ("q", "b") in visible_edges(m)  # the direct clause, via c *-> q
    assert ("c", "q") not in visible_edges(m)  # nothing points into c


def test_a_two_node_directed_edge_is_invisible() -> None:
    assert visible_edges(mag_of_dag(dag(["A -> B"]))) == frozenset()


def test_a_visible_edge_is_never_confounded_in_the_generating_truth() -> None:
    # Soundness against ground truth: if a -> b is visible in the projected
    # MAG, no latent in the generating DAG is a common cause of a and b
    # (reaching b off a). The truth is one member of the class, so a visible
    # edge that hid a confounder there would falsify the certificate.
    rng = np.random.default_rng(13)
    for _ in range(60):
        truth = _random_dag_with_latents(rng, 5, 2)
        m = mag_of_dag(truth)
        for a, b in visible_edges(m):
            for u in truth.latent:
                confounds = a in truth.descendants(u) and b in truth.induced_subgraph(
                    truth.nodes - {a}
                ).descendants(u)
                assert not confounds, (sorted(e.glyph() for e in truth.edges), (a, b), u)


# -- member enumeration ---------------------------------------------------------


def _brute_members(cp: MixedGraph) -> set[frozenset[Edge]]:
    undirected = sorted((e.u, e.v) for e in cp.edges if e.is_undirected)
    rest = [e.glyph() for e in cp.edges if not e.is_undirected]
    out: set[frozenset[Edge]] = set()
    for orient in product((0, 1), repeat=len(undirected)):
        specs = rest + [
            f"{u} -> {v}" if o == 0 else f"{v} -> {u}"
            for (u, v), o in zip(undirected, orient, strict=True)
        ]
        try:
            cand = dag(specs, nodes=sorted(cp.nodes))
        except InvalidGraphError:
            continue
        if cpdag_of_dag(cand).edges == cp.edges:
            out.add(cand.edges)
    return out


def test_dags_of_cpdag_enumerates_exactly_the_consistent_extensions() -> None:
    rng = np.random.default_rng(17)
    for _ in range(20):
        truth = _random_dag(rng, int(rng.choice([3, 4, 5])))
        cp = cpdag_of_dag(truth)
        members = dags_of_cpdag(cp)
        assert {m.edges for m in members} == _brute_members(cp)
        assert len({m.edges for m in members}) == len(members)
        assert truth.edges in {m.edges for m in members}
        assert all(m.kind is GraphKind.DAG for m in members)


def test_dags_of_cpdag_respects_the_member_cap() -> None:
    chain = cpdag_of_dag(dag(["A -> B", "B -> C"]))  # three members
    assert len(dags_of_cpdag(chain, max_members=3)) == 3
    with pytest.raises(ValueError, match="more than 2 members"):
        dags_of_cpdag(chain, max_members=2)


def test_mags_of_pag_lists_the_two_node_class() -> None:
    members = mags_of_pag(pag(["X o-o Y"]))
    glyphs = sorted(next(iter(m.edges)).glyph() for m in members)
    assert glyphs == ["X --> Y", "X <-- Y", "X <-> Y"]
    with pytest.raises(ValueError, match="more than 2 members"):
        mags_of_pag(pag(["X o-o Y"]), max_members=2)
    # The full Markov class also carries the undirected selection member.
    widened = mags_of_pag(pag(["X o-o Y"]), selection=True)
    glyphs = sorted(next(iter(m.edges)).glyph() for m in widened)
    assert glyphs == ["X --- Y", "X --> Y", "X <-- Y", "X <-> Y"]


def test_mags_of_pag_members_round_trip_and_include_the_extracted_one() -> None:
    rng = np.random.default_rng(19)
    for _ in range(8):
        truth = _random_dag_with_latents(rng, 4, 1)
        mag = mag_of_dag(truth)
        p = pag_of_mag(mag)
        members = mags_of_pag(p)
        assert mag.edges in {m.edges for m in members}
        assert mag_of_pag(p).edges in {m.edges for m in members}
        for m in members:
            assert m.is_mag()
            assert pag_of_mag(m).edges == p.edges


def test_the_enumerators_validate_their_input() -> None:
    # A single directed edge is not a CPDAG: its class leaves the edge
    # undirected. The collider A -> B <- C, by contrast, is its own CPDAG.
    with pytest.raises(ValueError, match="needs a CPDAG"):
        dags_of_cpdag(dag(["A -> B"]))
    with pytest.raises(ValueError, match="needs a complete PAG"):
        mags_of_pag(mixed(["X o-> Y", "Y o-o Z"]))
