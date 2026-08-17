# Ergodic

Causal AI: causal discovery, causal inference, and process intelligence.

!!! warning "Pre-alpha"
    This project is early scaffolding. The public API is not settled and will change.

New to causal inference itself? Start with the [learn series](learn/index.md): ten notebook
guides that teach the ideas on commercial examples, no background assumed. Already know the
field and want the API? Walk through [getting started](tutorial.md), then read
[concepts](concepts.md).

## Three pillars

<div class="grid cards" markdown>

-   :material-sitemap-outline:{ .lg .middle } __Causal discovery__

    ---

    Learn causal structure from data, including time series.

    [:octicons-arrow-right-24: Discovery](discovery.md)

-   :material-scale-balance:{ .lg .middle } __Causal inference__

    ---

    Estimate causal effects from data and a causal model, or from a [panel design](panel.md)
    when the rollout is the experiment.

    [:octicons-arrow-right-24: Inference](inference.md)

-   :material-cog-outline:{ .lg .middle } __Process intelligence__

    ---

    Discover and analyze processes from event logs.

    [:octicons-arrow-right-24: Process](process.md)

</div>

## Built on shared foundations

<div class="grid cards" markdown>

-   __Causal graphs__

    ---

    One mixed-graph object covering DAG, ADMG, CPDAG, MAG, and PAG.

    [:octicons-arrow-right-24: Graphs](graph.md)

-   __Domain knowledge__

    ---

    Prior constraints that discovery and inference read.

    [:octicons-arrow-right-24: Knowledge](knowledge.md)

-   __Data__

    ---

    Typed datasets: tabular, time series, panel, hierarchical, event log.

    [:octicons-arrow-right-24: Data](data.md)

-   __Identification__

    ---

    Graph to estimand, on a single graph or a whole equivalence class.

    [:octicons-arrow-right-24: Identification](identification.md)

</div>

## Installation

=== "pip"

    ```bash
    pip install ergodic
    ```

=== "uv"

    ```bash
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

The pillars meet in two calls: discover a graph from data, then estimate straight off it.

```python
from ergodic import discover, estimate_effect, tabular

data = tabular(frame)
result = discover(data)
estimate_effect(result.graph, data, "X", "Y", strategy="dml")
```

See the [API reference](reference.md) for the full surface.
