# Ergodic

Causal AI: causal discovery, causal inference, and process intelligence.

!!! warning "Pre-alpha"
    This project is early scaffolding. The public API is not settled and will change.

New here? Walk through the [tutorial](tutorial.md), then read the [concepts](concepts.md).

The work is organized around three pillars:

- [Causal discovery](discovery.md): learn causal structure from data.
- [Causal inference](inference.md): estimate causal effects from data and a causal model.
- [Process intelligence](process.md): discover and analyze processes from event logs.

All three sit on two shared objects, which are built first:

- [Causal graphs](graph.md): one mixed-graph object covering DAG, ADMG, CPDAG, MAG, and PAG.
- [Domain knowledge](knowledge.md): prior constraints that discovery and inference read.

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

See the [API reference](reference.md) for the full surface.
