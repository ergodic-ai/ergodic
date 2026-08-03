# Pillars

ergodic is organized around three pillars. Each one reads the shared
[foundations](foundations.md), a graph and a typed dataset, and answers a different question.

<div class="grid cards" markdown>

-   :material-sitemap-outline:{ .lg .middle } __Causal discovery__

    ---

    Learn causal structure from data. Constraint-based searches (PC, FCI, RFCI), score-based
    ones (GES, BOSS, GRaSP), and the functional DirectLiNGAM, with time series through PCMCI.

    [:octicons-arrow-right-24: Causal discovery](discovery.md)

-   :material-scale-balance:{ .lg .middle } __Causal inference__

    ---

    Estimate causal effects from a graph and data. A library of estimators from plain
    adjustment to double machine learning, with per-unit effects and an optional Bayesian
    posterior.

    [:octicons-arrow-right-24: Causal inference](inference.md)

-   :material-cog-outline:{ .lg .middle } __Process intelligence__

    ---

    Discover and analyze processes from event logs: a process map, a causal case table, a mined
    process model, and decision-point analysis.

    [:octicons-arrow-right-24: Process intelligence](process.md)

</div>

When the rollout itself is the experiment and no graph carries it, the [panel designs](panel.md)
sit beside the inference pillar: difference-in-differences, synthetic control, and interrupted
time series.
