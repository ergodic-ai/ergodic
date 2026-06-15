# SPDX-License-Identifier: Apache-2.0
"""Causal discovery: recover the Asia network from sampled data.

The data are drawn from a known network, so we can score what discovery finds.
PC recovers the strong edges with no false positives and misses the ones driven by
rare events (a visit to Asia is rare) or by the deterministic "either" node. Raise
the sample size with load_asia(n=8000) and the rare-event edge comes back.

Run with: uv run python examples/discovery_asia.py
"""

from ergodic import dag, discover
from ergodic.datasets import load_asia
from ergodic.graph import MixedGraph


def skeleton(graph: MixedGraph) -> set[frozenset[str]]:
    """The undirected adjacencies of a graph."""
    return {frozenset((x, y)) for x in graph.nodes for y in graph.neighbors(x)}


def main() -> None:
    ds = load_asia()
    truth = skeleton(dag(list(ds.target["edges"])))
    result = discover(ds.data, method="pc")
    found = skeleton(result.graph)

    recall = len(truth & found) / len(truth)
    precision = len(truth & found) / max(1, len(found))

    print(ds.DESCR)
    print(f"rows: {ds.data.n_rows}, true edges: {ds.target['n_edges']}")
    print(f"recovered:          {sorted('-'.join(sorted(p)) for p in found)}")
    print(f"missed:             {sorted('-'.join(sorted(p)) for p in truth - found)}")
    print(f"skeleton recall:    {recall:.2f}")
    print(f"skeleton precision: {precision:.2f}")


if __name__ == "__main__":
    main()
