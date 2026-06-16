# Foundations

Every pillar in ergodic reads the same two things: a causal graph and the data. Two more
objects sit between them, the domain knowledge that constrains the graph and the identification
step that turns a graph into an estimand. These four are built first, and the rest of the
library is a consumer of them.

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } __Causal graphs__

    ---

    One mixed-graph object covers DAG, ADMG, CPDAG, MAG, and PAG. Edges read as glyphs like
    `X -> Y`, and graphs are immutable values: every edit returns a new graph.

    [:octicons-arrow-right-24: Causal graphs](graph.md)

-   :material-check-decagram-outline:{ .lg .middle } __Domain knowledge__

    ---

    Prior constraints that discovery and inference read: required and forbidden edges, tiers,
    roots, sinks, confounded pairs. Immutable, and composable with `&`.

    [:octicons-arrow-right-24: Domain knowledge](knowledge.md)

-   :material-table:{ .lg .middle } __Data__

    ---

    Typed datasets across five shapes: tabular, time series, panel, hierarchical, and event
    log. A dataframe builds one and stays at the boundary; the math runs in numpy.

    [:octicons-arrow-right-24: Data](data.md)

-   :material-function-variant:{ .lg .middle } __Identification__

    ---

    From a graph to an estimand, on a single graph or a whole equivalence class. Adjustment,
    front-door, instruments, and the general do-calculus ID algorithm.

    [:octicons-arrow-right-24: Identification](identification.md)

</div>

New to the ideas behind these objects? The [concepts](concepts.md) page covers the model behind
the graph and the knowledge set in plain words, so the method-by-method guides read easily.
