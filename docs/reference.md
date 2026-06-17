# API reference

The shared objects live at the top level of `ergodic`, and the pillars under their own modules. Most people arrive here from search, so the table below maps a task to its entry point and the guide that teaches it.

| To do this | Start with | Guide |
| --- | --- | --- |
| Build a causal graph | `dag`, `admg`, `cpdag`, `pag` | [Causal graphs](graph.md) |
| Turn a graph into an estimand | `identify_effect` | [Identification](identification.md) |
| Estimate an effect from data | `estimate_effect` | [Causal inference](inference.md) |
| Stress-test an estimate | `refute_effect` | [Checking an estimate](inference.md#checking-an-estimate) |
| Learn a graph from data | `discover`, `bootstrap` | [Causal discovery](discovery.md) |
| Measure a rollout with no graph | `did`, `synthetic_control`, `interrupted_time_series` | [Panel designs](panel.md) |
| Analyze a process from its log | `process_map`, `case_table`, `discover_model` | [Process intelligence](process.md) |
| Simulate or ask counterfactuals | `SCM` | [Structural causal models](scm.md) |

## Causal graphs

The mixed-graph object and its builders, separation, do-surgery, and interop.

::: ergodic.graph

## Domain knowledge

Hard prior constraints (required and forbidden edges, tiers, roots, sinks) that discovery and inference read.

::: ergodic.knowledge

## Data

The typed dataset family across five shapes, and its accessors.

::: ergodic.data

## Identification

Graph to estimand: adjustment, front-door, instruments, the general ID algorithm, and mediation.

::: ergodic.identification

## Causal discovery

Structure learning: the algorithms, the CI tests and scores they consume, and the `discover` facade.

::: ergodic.discovery

## Causal inference

Effect estimation: the estimators, the learners that fill their slots, refutation, and the quasi-experimental designs.

::: ergodic.inference

## Process intelligence

Event-log mining: the map, the causal case table, the model layer, and decision-point analysis.

::: ergodic.process

## Structural causal models

The Bayesian SCM: mechanisms, interventions, counterfactuals, and the identification-gated `ate`.

::: ergodic.scm.SCM

::: ergodic.scm.SCMResult

::: ergodic.scm.CategoricalResult

::: ergodic.scm.Sensitivity

::: ergodic.scm.GaussianAdditive

::: ergodic.scm.BinaryGumbelMax

::: ergodic.scm.CategoricalGumbelMax

::: ergodic.scm.Mechanism
