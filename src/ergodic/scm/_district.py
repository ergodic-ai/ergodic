# SPDX-License-Identifier: Apache-2.0
"""The latent structure of an ADMG: one explicit latent per bidirected edge.

The SCM grows from a Markovian DAG to an ADMG by realizing each ``<->`` as an
explicit per-unit latent common cause (the ``latent_projection`` semantics the
identification layer already reads). This module is the pure, pymc-free analysis
that turns the projected ADMG into the bookkeeping the fit and the generative model
need: which latents exist, which nodes each touches, which node anchors each
latent's sign gauge, and which districts are confounded (so fit jointly) versus
singletons (so fit per node, the MVP path).

One latent per bidirected edge, named ``L__{u}__{v}`` for the sorted endpoint pair.
The gauge anchor is the first endpoint ``u``: its loading on ``L__u__v`` is pinned
positive, killing the ``L -> -L`` reflection. A node in a district with several
bidirected edges carries one latent per incident edge, the faithful representation
the three-node pressure-test used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..graph._primitives import districts

if TYPE_CHECKING:
    from ..graph import MixedGraph


def latent_name(u: str, v: str) -> str:
    """The latent RV name for the bidirected edge between ``u`` and ``v``."""
    a, b = sorted((u, v))
    return f"L__{a}__{b}"


@dataclass(frozen=True)
class LatentStructure:
    """The explicit-latent realization of a projected ADMG.

    - ``latents``: every latent RV name (one per bidirected edge), sorted.
    - ``incident``: node -> the latents incident to it (its bidirected edges).
    - ``gauged``: node -> the latents this node is the sign-gauge anchor for (it is
      the first endpoint of those edges), so its loading on them is pinned positive.
    - ``confounded_districts``: districts carrying a bidirected edge (fit jointly),
      each a sorted tuple of nodes.
    - ``singletons``: nodes whose district is a lone node (fit per node, MVP path).
    - ``node_district``: node -> its district as a sorted tuple.
    """

    latents: tuple[str, ...]
    incident: dict[str, tuple[str, ...]]
    gauged: dict[str, tuple[str, ...]]
    confounded_districts: tuple[tuple[str, ...], ...]
    singletons: tuple[str, ...]
    node_district: dict[str, tuple[str, ...]]

    def latents_of(self, node: str) -> tuple[str, ...]:
        return self.incident.get(node, ())

    def gauged_of(self, node: str) -> tuple[str, ...]:
        return self.gauged.get(node, ())

    @property
    def has_latents(self) -> bool:
        return bool(self.latents)


def latent_structure(projection: MixedGraph) -> LatentStructure:
    """Analyze the projected ADMG into its explicit-latent bookkeeping.

    ``projection`` is the observed-margin ADMG (from :func:`latent_projection`); its
    ``<->`` edges become the latents. Every node lands in exactly one district; a
    district with no bidirected edge is a singleton (fit per node), one with a
    bidirected edge is confounded (fit jointly).
    """
    incident: dict[str, list[str]] = {}
    gauged: dict[str, list[str]] = {}
    latents: set[str] = set()
    for edge in projection.edges:
        if not edge.is_bidirected:
            continue
        u, v = edge.u, edge.v
        name = latent_name(u, v)
        latents.add(name)
        incident.setdefault(u, []).append(name)
        incident.setdefault(v, []).append(name)
        # the sign gauge: the first endpoint anchors the latent positive.
        anchor = min(u, v)
        gauged.setdefault(anchor, []).append(name)

    comps = districts(projection)
    confounded: list[tuple[str, ...]] = []
    singletons: list[str] = []
    node_district: dict[str, tuple[str, ...]] = {}
    for comp in comps:
        nodes = tuple(sorted(comp))
        for node in nodes:
            node_district[node] = nodes
        if len(nodes) == 1 and not incident.get(nodes[0]):
            singletons.append(nodes[0])
        else:
            confounded.append(nodes)

    return LatentStructure(
        latents=tuple(sorted(latents)),
        incident={k: tuple(sorted(v)) for k, v in incident.items()},
        gauged={k: tuple(sorted(v)) for k, v in gauged.items()},
        confounded_districts=tuple(sorted(confounded)),
        singletons=tuple(sorted(singletons)),
        node_district=node_district,
    )
