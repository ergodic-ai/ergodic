# SPDX-License-Identifier: Apache-2.0
"""Interop and serialization: networkx, adjacency matrices, and plain dicts.

networkx lives at this boundary only. The graph algorithms elsewhere in the
package do not depend on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._graph import Edge, GraphKind, MixedGraph, _merge_edges
from ._marks import Mark

if TYPE_CHECKING:
    import numpy.typing as npt


def _reject_parallel(graph: MixedGraph, op: str) -> None:
    """Raise if any node pair carries two edges (a bow from latent_projection).

    The marks model holds one edge per pair, so serialization and adjacency
    cannot represent a bow (a directed and a bidirected edge on one pair). A
    projected ADMG is an internal ID input; fail loudly here instead of dropping
    or mangling the bidirected edge. A normal ADMG or MAG with a single ``<->``
    edge per pair is unaffected.
    """
    seen: set[tuple[str, str]] = set()
    for e in graph.edges:
        if (e.u, e.v) in seen:
            raise ValueError(
                f"{op} cannot represent the parallel-edge pair ({e.u!r}, {e.v!r}): "
                "this graph carries a bow, which latent_projection can produce. The "
                "marks model holds one edge per pair, so a projected ADMG is an "
                "internal ID input, not a serializable graph."
            )
        seen.add((e.u, e.v))


def to_networkx(graph: MixedGraph) -> Any:
    """Return a ``networkx.MultiDiGraph`` carrying the marks as edge data.

    Each edge becomes one ``u -> v`` arc with ``u_mark`` and ``v_mark`` data, so
    :func:`from_networkx` can round-trip it. Node ``latent`` and ``selection``
    flags are stored as boolean node data.
    """
    _reject_parallel(graph, "to_networkx")
    import networkx as nx

    out = nx.MultiDiGraph()
    for node in graph.nodes:
        out.add_node(node, latent=node in graph.latent, selection=node in graph.selection)
    for e in graph.edges:
        out.add_edge(e.u, e.v, u_mark=e.u_mark.value, v_mark=e.v_mark.value)
    return out


def from_networkx(graph: Any, *, kind: GraphKind | None = None) -> MixedGraph:
    """Build a :class:`MixedGraph` from a networkx graph.

    Edges with ``u_mark``/``v_mark`` data keep those marks. A plain edge is read
    as directed (``u -> v``). With no ``kind`` given, the result is tagged ``DAG``
    when every edge is directed and acyclic, otherwise ``MIXED``. Two arcs that
    disagree about one node pair (say ``X -> Y`` and ``Y -> X``) raise
    ``ValueError``, since a :class:`MixedGraph` holds one edge per pair.
    """
    specs: list[Edge] = []
    for u, v, data in graph.edges(data=True):
        if "u_mark" in data and "v_mark" in data:
            specs.append(Edge.make(u, Mark(data["u_mark"]), v, Mark(data["v_mark"])))
        else:
            specs.append(Edge.make(u, Mark.TAIL, v, Mark.ARROW))
    latent = {n for n, d in graph.nodes(data=True) if d.get("latent")}
    selection = {n for n, d in graph.nodes(data=True) if d.get("selection")}
    out = MixedGraph(
        nodes=frozenset(graph.nodes),
        edges=_merge_edges(specs),
        latent=frozenset(latent),
        selection=frozenset(selection),
        kind=GraphKind.MIXED,
    )
    if kind is not None:
        return out.as_kind(kind)
    if out.is_dag():
        return out.as_kind(GraphKind.DAG)
    return out


def to_adjacency(
    graph: MixedGraph, order: list[str] | None = None
) -> tuple[npt.NDArray[np.int_], list[str]]:
    """Return the directed adjacency matrix and its node order.

    ``matrix[i, j] == 1`` when there is a directed edge from node ``i`` to node
    ``j``. Bidirected, undirected, and circle edges are not represented here.
    """
    _reject_parallel(graph, "to_adjacency")
    labels = order if order is not None else graph.node_list()
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for e in graph.edges:
        pair = e.directed_pair()
        if pair is not None and pair[0] in index and pair[1] in index:
            matrix[index[pair[0]], index[pair[1]]] = 1
    return matrix, labels


def to_dict(graph: MixedGraph) -> dict[str, Any]:
    """Return a JSON-ready dict that :func:`from_dict` can read back."""
    _reject_parallel(graph, "to_dict")
    return {
        "kind": graph.kind.value,
        "nodes": graph.node_list(),
        "edges": [
            {"u": e.u, "v": e.v, "u_mark": e.u_mark.value, "v_mark": e.v_mark.value}
            for e in graph.edge_list()
        ],
        "latent": sorted(graph.latent),
        "selection": sorted(graph.selection),
    }


def from_dict(data: dict[str, Any]) -> MixedGraph:
    """Rebuild a graph from :func:`to_dict` output.

    Two entries that disagree about one node pair raise ``ValueError``.
    """
    edges = _merge_edges(
        Edge.make(e["u"], Mark(e["u_mark"]), e["v"], Mark(e["v_mark"])) for e in data["edges"]
    )
    graph = MixedGraph(
        nodes=frozenset(data["nodes"]),
        edges=edges,
        latent=frozenset(data.get("latent", ())),
        selection=frozenset(data.get("selection", ())),
        kind=GraphKind(data.get("kind", "mixed")),
    )
    graph.validate()
    return graph
