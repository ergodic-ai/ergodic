# Ergodic

Causal AI: causal discovery, causal inference, and process intelligence.

!!! warning "Pre-alpha"
    This project is early scaffolding. The public API is not settled and will change.

New to causal inference itself? Start with the [learn series](learn/index.md): seven notebook guides that teach the ideas on commercial examples, no background assumed. Know the field and want the API? Walk through the [tutorial](tutorial.md), then read the [concepts](concepts.md).

The work is organized around three pillars:

- [Causal discovery](discovery.md): learn causal structure from data, including time series.
- [Causal inference](inference.md): estimate causal effects from data and a causal model, or from a [panel design](panel.md) when the rollout is the experiment.
- [Process intelligence](process.md): discover and analyze processes from event logs.

All three sit on shared objects, which are built first:

- [Causal graphs](graph.md): one mixed-graph object covering DAG, ADMG, CPDAG, MAG, and PAG.
- [Domain knowledge](knowledge.md): prior constraints that discovery and inference read.
- [Data](data.md): typed datasets (tabular, time series, panel, hierarchical, event log).
- [Identification](identification.md): graph to estimand, on a single graph or a whole equivalence class.

## Installation

```bash
pip install ergodic
# or
uv add ergodic
```

## Quick start

```python
from ergodic import dag, DomainKnowledge, identify_effect

g = dag(["Smoking -> Tar", "Tar -> Cancer", "Smoking -> Cancer"])
g.d_separated("Smoking", "Cancer", "Tar")   # False: the direct edge remains

dk = DomainKnowledge().with_tiers([["Smoking"], ["Tar"], ["Cancer"]])
dk.is_consistent(g)   # True

identify_effect(g, "Smoking", "Cancer").estimand
# 'P(Cancer | do(Smoking)) = P(Cancer | Smoking)'
```

And the pillars meet in two calls: discover a graph from data, then estimate straight off it.

```python
from ergodic import discover, estimate_effect, tabular

data = tabular(frame)
result = discover(data)
estimate_effect(result.graph, data, "X", "Y", strategy="dml")
```

See the [API reference](reference.md) for the full surface.
