# SPDX-License-Identifier: Apache-2.0
"""Tests for the identification graph primitives.

``districts`` (c-components), ``topological_order`` (Kahn, name-tie-broken),
and ``latent_projection`` (the ADMG projection onto the observed margin). The
anchor is the m-separation-preservation property of ``latent_projection`` over
random latent DAGs, the same shape as the RS Theorem 4.18 property test.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pytest

from ergodic.graph import (
    GraphKind,
    MixedGraph,
    admg,
    dag,
    district_of,
    districts,
    latent_projection,
    mixed,
    topological_order,
)


def _random_dag(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = [(names[i], names[j]) for i, j in combinations(range(p), 2) if rng.uniform() < 0.5]
    return dag(edges, nodes=names)


def _random_dag_with_latents(rng: np.random.Generator, p: int, n_latent: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges = [(names[i], names[j]) for i, j in combinations(range(p), 2) if rng.uniform() < 0.4]
    latent = list(rng.choice(names, size=n_latent, replace=False))
    return dag(edges, nodes=names, latent=latent)


def _random_admg(rng: np.random.Generator, p: int) -> MixedGraph:
    names = [f"V{i}" for i in range(p)]
    edges: list[tuple[str, str] | str] = []
    for i, j in combinations(range(p), 2):
        u = rng.uniform()
        if u < 0.30:
            edges.append((names[i], names[j]))
        elif u < 0.45:
            edges.append(f"{names[i]} <-> {names[j]}")
    return admg(edges, nodes=names)


# -- districts ----------------------------------------------------------------


def test_districts_are_singletons_on_a_dag() -> None:
    g = dag(["X -> Y", "Y -> Z", "X -> Z"])
    assert districts(g) == frozenset({frozenset({"X"}), frozenset({"Y"}), frozenset({"Z"})})


def test_districts_find_the_bidirected_components() -> None:
    g = admg(["A <-> B", "B <-> C", "D <-> E", "X -> Y"])
    assert districts(g) == frozenset(
        {
            frozenset({"A", "B", "C"}),
            frozenset({"D", "E"}),
            frozenset({"X"}),
            frozenset({"Y"}),
        }
    )


def test_districts_partition_the_nodes() -> None:
    rng = np.random.default_rng(11)
    for _ in range(50):
        g = _random_admg(rng, int(rng.choice([4, 5, 6, 7])))
        parts = districts(g)
        # every node in exactly one district
        flat = [n for d in parts for n in d]
        assert Counter(flat) == Counter(dict.fromkeys(g.nodes, 1))
        assert frozenset(flat) == g.nodes
        # district_of agrees with the part containing the node
        for n in g.nodes:
            (only,) = [d for d in parts if n in d]
            assert district_of(g, n) == only


def test_district_of_rejects_an_unknown_node() -> None:
    g = admg(["A <-> B"])
    with pytest.raises(ValueError, match="unknown node name"):
        district_of(g, "Z")


def test_district_walks_bidirected_only_not_directed() -> None:
    # a directed chain leaves every node its own district
    g = admg(["A -> B", "B -> C"])
    assert district_of(g, "A") == frozenset({"A"})
    assert district_of(g, "B") == frozenset({"B"})


# -- topological_order --------------------------------------------------------


def test_topological_order_respects_every_directed_edge() -> None:
    rng = np.random.default_rng(13)
    for _ in range(60):
        g = _random_dag(rng, int(rng.choice([3, 4, 5, 6])))
        order = topological_order(g)
        assert frozenset(order) == g.nodes
        pos = {n: i for i, n in enumerate(order)}
        for e in g.edges:
            pair = e.directed_pair()
            assert pair is not None
            assert pos[pair[0]] < pos[pair[1]]


def test_topological_order_is_deterministic() -> None:
    g = dag(["X -> M", "M -> Y", "X -> Y", "A -> X", "A -> Y"])
    assert topological_order(g) == topological_order(g)
    # the name tie-break is stable and known: A and X are forced before the rest
    assert topological_order(g) == ("A", "X", "M", "Y")


def test_topological_order_ignores_bidirected_edges() -> None:
    # bidirected edges impose no order; the directed edges still constrain
    g = admg(["X -> Y", "Z <-> X", "Y <-> W"])
    order = topological_order(g)
    pos = {n: i for i, n in enumerate(order)}
    assert pos["X"] < pos["Y"]
    assert frozenset(order) == g.nodes


def test_topological_order_raises_on_a_cycle() -> None:
    g = mixed(["A -> B", "B -> C", "C -> A"])
    with pytest.raises(ValueError, match="cycle"):
        topological_order(g)


# -- latent_projection --------------------------------------------------------


def test_latent_projection_chain_gives_a_directed_edge() -> None:
    proj = latent_projection(dag(["X -> L", "L -> Y"], latent=["L"]))
    assert proj.kind is GraphKind.ADMG
    assert proj.latent == frozenset()
    assert proj.nodes == frozenset({"X", "Y"})
    assert {e.glyph() for e in proj.edges} == {"X --> Y"}


def test_latent_projection_fork_gives_a_bidirected_edge() -> None:
    proj = latent_projection(dag(["L -> X", "L -> Y"], latent=["L"]))
    assert proj.nodes == frozenset({"X", "Y"})
    assert {e.glyph() for e in proj.edges} == {"X <-> Y"}


def test_latent_projection_keeps_both_edges_of_a_bow() -> None:
    # L confounds X and Y while X -> Y directly: the projection is a bow, and
    # m-separation needs both edges, so both are emitted.
    proj = latent_projection(dag(["L -> X", "L -> Y", "X -> Y"], latent=["L"]))
    assert {e.glyph() for e in proj.edges} == {"X --> Y", "X <-> Y"}
    assert proj.is_admg()


def test_latent_projection_defaults_keep_to_observed_nodes() -> None:
    g = dag(["A -> B", "B -> C", "L -> B", "L -> C"], latent=["L"])
    proj = latent_projection(g)
    assert proj.nodes == frozenset({"A", "B", "C"})
    # A -> B -> C survives, and L confounds B and C
    glyphs = {e.glyph() for e in proj.edges}
    assert "A --> B" in glyphs
    assert "B --> C" in glyphs
    assert "B <-> C" in glyphs


def test_latent_projection_honours_an_explicit_keep() -> None:
    g = dag(["X -> M", "M -> Y"])
    proj = latent_projection(g, keep=["X", "Y"])
    assert proj.nodes == frozenset({"X", "Y"})
    assert {e.glyph() for e in proj.edges} == {"X --> Y"}


def test_latent_projection_rejects_a_non_admg() -> None:
    g = mixed(["X o-> Y"])
    with pytest.raises(ValueError, match="DAG or ADMG"):
        latent_projection(g)


def test_latent_projection_rejects_unknown_keep_names() -> None:
    with pytest.raises(ValueError, match="unknown node"):
        latent_projection(dag(["X -> Y"]), keep=["X", "Z"])


def test_latent_projection_preserves_m_separation_over_random_latent_dags() -> None:
    # The anchor. For disjoint observed A, B, Z (none projected out), the
    # projection and the source DAG agree on m-separation; assert agreement
    # over many random worlds and many triples, with a not-vacuous guard that
    # both separated and connected cases were actually exercised.
    rng = np.random.default_rng(2024)
    checked = separated = connected = bows = 0
    for _ in range(400):
        p = int(rng.choice([4, 5, 6, 7, 8]))
        n_latent = int(rng.choice([1, 2, 3]))
        g = _random_dag_with_latents(rng, p, n_latent)
        proj = latent_projection(g)
        assert proj.nodes == g.nodes - g.latent
        pair_counts = Counter(frozenset((e.u, e.v)) for e in proj.edges)
        bows += sum(1 for v in pair_counts.values() if v > 1)
        observed = sorted(proj.nodes)
        if len(observed) < 3:
            continue
        for _ in range(10):
            perm = list(rng.permutation(observed))
            n_a = int(rng.choice([1, 1, 2]))
            n_b = int(rng.choice([1, 1, 2]))
            a = frozenset(perm[:n_a])
            b = frozenset(perm[n_a : n_a + n_b])
            rest = perm[n_a + n_b :]
            mask = rng.uniform(size=len(rest)) < 0.4
            z = frozenset(n for n, m in zip(rest, mask, strict=True) if m)
            if not a or not b:
                continue
            truth = g.m_separated(a, b, z)
            got = proj.m_separated(a, b, z)
            assert truth == got, (
                sorted(e.glyph() for e in g.edge_list()),
                sorted(g.latent),
                sorted(a),
                sorted(b),
                sorted(z),
            )
            checked += 1
            separated += int(truth)
            connected += int(not truth)
    assert checked > 1000
    assert separated > 0  # not vacuous: real separations were exercised
    assert connected > 0  # not vacuous: real connections were exercised
    assert bows > 0  # the bow path was actually hit


def test_latent_projection_preserves_m_separation_over_random_admgs() -> None:
    # The same property with bidirected edges already present in the source and
    # an arbitrary kept margin (the rest is the projected-out set H).
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(250):
        g = _random_admg(rng, int(rng.choice([5, 6, 7, 8])))
        nodes = sorted(g.nodes)
        mask = rng.uniform(size=len(nodes)) < 0.6
        keep = [n for n, m in zip(nodes, mask, strict=True) if m]
        if len(keep) < 3:
            continue
        proj = latent_projection(g, keep=keep)
        assert proj.nodes == frozenset(keep)
        for _ in range(8):
            perm = list(rng.permutation(keep))
            a, b = perm[0], perm[1]
            rest = perm[2:]
            z = frozenset(
                n for n, m in zip(rest, rng.uniform(size=len(rest)) < 0.4, strict=True) if m
            )
            assert g.m_separated({a}, {b}, z) == proj.m_separated({a}, {b}, z)
            checked += 1
    assert checked > 800
