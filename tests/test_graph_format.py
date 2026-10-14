# SPDX-License-Identifier: Apache-2.0
"""The versioned dict format: ``to_dict`` writes it, ``from_dict`` checks it.

Every graph kind round-trips, a dict without the field reads as version 1
(older output and hand-written payloads such as Welt's), and an unknown format
is refused by name.
"""

from __future__ import annotations

import json

import pytest

import ergodic
from ergodic.graph import (
    GRAPH_FORMAT,
    GraphKind,
    MixedGraph,
    admg,
    cpdag,
    dag,
    mag_of_dag,
    mixed,
    pag,
    pag_of_mag,
)


def _graphs() -> dict[str, MixedGraph]:
    # A DAG with a latent confounder and a selection node, projected to its MAG
    # and canonical PAG. The projection drops the designations and writes the
    # selection into the marks (undirected and circle-circle edges).
    world = dag(["L -> X", "L -> Y", "X -> Z", "Y -> S", "Z -> S"], latent=["L"], selection=["S"])
    world_mag = mag_of_dag(world)
    return {
        "dag": dag(["X -> Y", "Y -> Z"]),
        "dag_latent": dag(["L -> X", "L -> Y", "X -> Y"], latent=["L"]),
        "dag_latent_selection": world,
        "admg": admg(["X -> Y", "Y <-> Z"], nodes=["Q"], latent=["L"]),
        "cpdag": cpdag(["X -- Y", "Y -- Z"]),
        "pag": pag(["X o-> Y", "Z o-> Y"]),
        "mag": mixed(["X -> Y", "Y <-> Z"]).as_kind("mag"),
        "mag_selection": world_mag,
        "pag_selection": pag_of_mag(world_mag),
        "mixed": mixed(["X o-o Y", "Y -> Z", "Z <-> W"], nodes=["Q"]),
        "mixed_latent_selection": mixed(["X -> Y", "Y -> S"], latent=["L"], selection=["S"]),
    }


GRAPHS = _graphs()


def test_graph_format_is_exported_at_top_level_and_from_graph() -> None:
    assert GRAPH_FORMAT == "ergodic.graph/1"
    assert ergodic.GRAPH_FORMAT == GRAPH_FORMAT
    assert "GRAPH_FORMAT" in ergodic.__all__
    assert "GRAPH_FORMAT" in ergodic.graph.__all__


def test_the_fixtures_cover_every_kind() -> None:
    assert {g.kind for g in GRAPHS.values()} == set(GraphKind)
    assert any(g.latent and g.selection for g in GRAPHS.values() if g.kind is GraphKind.DAG)
    assert any(g.latent and g.selection for g in GRAPHS.values() if g.kind is GraphKind.MIXED)
    assert any(g.latent for g in GRAPHS.values() if g.kind is GraphKind.ADMG)
    assert any(e.is_undirected for e in GRAPHS["mag_selection"].edges)
    assert GRAPHS["pag_selection"].kind is GraphKind.PAG


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_round_trip_through_json(name: str) -> None:
    graph = GRAPHS[name]
    data = graph.to_dict()
    assert data["format"] == GRAPH_FORMAT
    back = MixedGraph.from_dict(json.loads(json.dumps(data)))
    assert back == graph
    assert back.kind is graph.kind
    assert back.latent == graph.latent
    assert back.selection == graph.selection


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_missing_format_reads_as_version_one(name: str) -> None:
    graph = GRAPHS[name]
    data = graph.to_dict()
    del data["format"]
    assert MixedGraph.from_dict(data) == graph


def test_to_dict_layout_is_unchanged_apart_from_the_format_field() -> None:
    graph = admg(["L -> X", "X -> Y", "Y <-> Z"], nodes=["Q"], latent=["L"])
    assert graph.to_dict() == {
        "format": "ergodic.graph/1",
        "kind": "admg",
        "nodes": ["L", "Q", "X", "Y", "Z"],
        "edges": [
            {"u": "L", "v": "X", "u_mark": "tail", "v_mark": "arrow"},
            {"u": "X", "v": "Y", "u_mark": "tail", "v_mark": "arrow"},
            {"u": "Y", "v": "Z", "u_mark": "arrow", "v_mark": "arrow"},
        ],
        "latent": ["L"],
        "selection": [],
    }
    assert list(graph.to_dict()) == ["format", "kind", "nodes", "edges", "latent", "selection"]


def test_minimal_payload_without_format_latent_or_selection() -> None:
    # The shape welt-client sends: kind, nodes, and {u, v, u_mark, v_mark} edges.
    payload = {
        "kind": "dag",
        "nodes": ["X", "Y", "Z"],
        "edges": [
            {"u": "X", "v": "Y", "u_mark": "tail", "v_mark": "arrow"},
            {"u": "Y", "v": "Z", "u_mark": "tail", "v_mark": "arrow"},
        ],
    }
    assert MixedGraph.from_dict(payload) == dag(["X -> Y", "Y -> Z"])
    assert MixedGraph.from_dict({**payload, "format": GRAPH_FORMAT}) == dag(["X -> Y", "Y -> Z"])


@pytest.mark.parametrize("fmt", ["ergodic.graph/2", "ergodic.graph/0", "", None, 1])
def test_unknown_format_raises_and_names_the_supported_one(fmt: object) -> None:
    data = dag(["X -> Y"]).to_dict()
    data["format"] = fmt
    with pytest.raises(ValueError) as info:
        MixedGraph.from_dict(data)
    assert str(info.value) == (
        f"unsupported graph format {fmt!r}; this version of ergodic reads "
        "'ergodic.graph/1' (a dict without a 'format' field is read as 'ergodic.graph/1')"
    )
