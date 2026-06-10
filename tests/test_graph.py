# SPDX-License-Identifier: Apache-2.0
"""Tests for the mixed-graph core, its kinds, separation, surgery, and interop."""

from __future__ import annotations

import pytest

from ergodic.graph import (
    Edge,
    GraphKind,
    InvalidGraphError,
    Mark,
    MixedGraph,
    admg,
    cpdag,
    dag,
    mixed,
    pag,
)

# -- edges -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "directed", "bidirected", "undirected", "circle"),
    [
        ("X -> Y", True, False, False, False),
        ("X <- Y", True, False, False, False),
        ("X <-> Y", False, True, False, False),
        ("X -- Y", False, False, True, False),
        ("X o-> Y", False, False, False, True),
        ("X o-o Y", False, False, False, True),
    ],
)
def test_edge_string_marks(
    text: str, directed: bool, bidirected: bool, undirected: bool, circle: bool
) -> None:
    edge = mixed([text]).edge_list()[0]
    assert edge.is_directed is directed
    assert edge.is_bidirected is bidirected
    assert edge.is_undirected is undirected
    assert edge.has_circle is circle


def test_edge_make_normalizes_order() -> None:
    forward = Edge.make("X", Mark.TAIL, "Y", Mark.ARROW)
    backward = Edge.make("Y", Mark.ARROW, "X", Mark.TAIL)
    assert forward == backward
    assert forward.directed_pair() == ("X", "Y")


def test_edge_glyph_round_trip() -> None:
    # the parser takes "A -> B" or "A --> B"; glyph() renders the canonical shaft form
    for text in ("A --> B", "A <-> B", "A o-> B", "A --- B"):
        assert mixed([text]).edge_list()[0].glyph() == text


def test_self_loop_rejected() -> None:
    with pytest.raises(ValueError, match="self-loop"):
        Edge.make("X", Mark.TAIL, "X", Mark.ARROW)


# -- kinds and validation ----------------------------------------------------


def test_dag_accepts_directed_and_rejects_bidirected() -> None:
    dag(["X -> Y", "Y -> Z"])
    with pytest.raises(InvalidGraphError):
        dag(["X <-> Y"])


def test_dag_rejects_cycle() -> None:
    with pytest.raises(InvalidGraphError):
        dag(["X -> Y", "Y -> Z", "Z -> X"])


def test_admg_allows_bidirected_but_not_circle() -> None:
    admg(["X -> Y", "Y <-> Z"])
    with pytest.raises(InvalidGraphError):
        admg(["X o-> Y"])


def test_cpdag_allows_undirected_but_not_bidirected() -> None:
    cpdag(["X -- Y", "Y -> Z"])
    with pytest.raises(InvalidGraphError):
        cpdag(["X <-> Y"])


def test_pag_allows_circles() -> None:
    graph = pag(["X o-> Y", "Y o-o Z"])
    assert graph.kind is GraphKind.PAG


def test_as_kind_validates() -> None:
    graph = mixed(["X <-> Y"])
    with pytest.raises(InvalidGraphError):
        graph.as_kind(GraphKind.DAG)
    assert graph.as_kind(GraphKind.ADMG).kind is GraphKind.ADMG


def test_mag_rejects_almost_directed_cycle() -> None:
    bad = mixed(["X -> Y", "Y -> Z", "X <-> Z"])
    assert bad.is_admg()
    assert not bad.is_mag()
    good = mixed(["X <-> Y", "Y <-> Z"])
    assert good.is_mag()


# -- immutability and equality -----------------------------------------------


def test_edits_return_new_graphs() -> None:
    base = dag(["X -> Y"])
    grown = base.add_edge("Y -> Z")
    assert "Z" not in base
    assert "Z" in grown
    assert base == dag(["X -> Y"])
    assert base != grown
    assert len({base, grown, dag(["X -> Y"])}) == 2


def test_remove_and_orient() -> None:
    graph = mixed(["X -- Y"]).orient("X", "Y")
    assert graph.get_edge("X", "Y") is not None
    assert graph.get_edge("X", "Y").directed_pair() == ("X", "Y")  # type: ignore[union-attr]
    assert graph.remove_edge("X", "Y").get_edge("X", "Y") is None


# -- queries -----------------------------------------------------------------


def test_family_queries() -> None:
    graph = dag(["A -> X", "B -> X", "X -> C", "D -> C"])
    assert graph.parents("X") == {"A", "B"}
    assert graph.children("X") == {"C"}
    assert graph.ancestors("C") == {"A", "B", "D", "X"}
    assert graph.descendants("A") == {"X", "C"}
    assert graph.markov_blanket("X") == {"A", "B", "C", "D"}


def test_spouses_and_markov_blanket_guard() -> None:
    graph = admg(["X <-> Y"])
    assert graph.spouses("X") == {"Y"}
    with pytest.raises(NotImplementedError):
        graph.markov_blanket("X")


# -- d-separation ------------------------------------------------------------


def test_chain_fork_collider() -> None:
    chain = dag(["X -> Y", "Y -> Z"])
    assert not chain.d_separated("X", "Z")
    assert chain.d_separated("X", "Z", "Y")

    fork = dag(["Y -> X", "Y -> Z"])
    assert not fork.d_separated("X", "Z")
    assert fork.d_separated("X", "Z", "Y")

    collider = dag(["X -> Y", "Z -> Y"])
    assert collider.d_separated("X", "Z")
    assert not collider.d_separated("X", "Z", "Y")


def test_conditioning_on_collider_descendant_opens_path() -> None:
    graph = dag(["X -> Y", "Z -> Y", "Y -> W"])
    assert graph.d_separated("X", "Z")
    assert not graph.d_separated("X", "Z", "W")


def test_d_separated_requires_dag() -> None:
    with pytest.raises(ValueError, match="requires a DAG"):
        admg(["X <-> Y"]).d_separated("X", "Y")


# -- m-separation ------------------------------------------------------------


def test_m_separation_with_bidirected_collider() -> None:
    graph = admg(["X -> Y", "Y <-> Z"])
    assert graph.m_separated("X", "Z")
    assert not graph.m_separated("X", "Z", "Y")


def test_m_separation_rejects_circles() -> None:
    with pytest.raises(NotImplementedError):
        pag(["X o-> Y", "Y o-> Z"]).m_separated("X", "Z")


def test_separation_requires_disjoint_sets() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        dag(["X -> Y"]).d_separated("X", "Y", "X")


# -- surgery -----------------------------------------------------------------


def test_do_cuts_incoming_edges() -> None:
    collider = dag(["X -> Y", "Z -> Y"])
    assert collider.do("Y").edges == frozenset()

    chain = dag(["X -> Y", "Y -> Z"])
    after = chain.do("Y")
    assert after.parents("Y") == frozenset()
    assert after.children("Y") == {"Z"}


def test_do_cuts_bidirected_confounding() -> None:
    graph = admg(["X -> Y", "X <-> Y"])  # the bidirected edge wins the pair
    assert graph.do("Y").get_edge("X", "Y") is None


def test_induced_and_ancestral_subgraph() -> None:
    graph = dag(["X -> Y", "Y -> Z", "W -> Z"])
    induced = graph.induced_subgraph({"X", "Y"})
    assert induced.nodes == {"X", "Y"}
    assert induced.has_edge("X", "Y")

    ancestral = graph.ancestral_subgraph({"Y"})
    assert ancestral.nodes == {"X", "Y"}


# -- interop -----------------------------------------------------------------


def test_dict_round_trip() -> None:
    graph = admg(["X -> Y", "Y <-> Z"], nodes=["Q"], latent=["L"])
    assert MixedGraph.from_dict(graph.to_dict()) == graph


def test_networkx_round_trip_preserves_edges() -> None:
    graph = admg(["X -> Y", "Y <-> Z"])
    back = MixedGraph.from_networkx(graph.to_networkx())
    assert back.edges == graph.edges
    assert back.nodes == graph.nodes


def test_adjacency_matrix() -> None:
    matrix, labels = dag(["X -> Y", "Y -> Z"]).to_adjacency()
    assert labels == ["X", "Y", "Z"]
    assert matrix[0, 1] == 1
    assert matrix[1, 2] == 1
    assert matrix.sum() == 2
