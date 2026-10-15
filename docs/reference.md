# API reference

The shared objects live at the top level of `ergodic`, and the pillars under their own modules. Most people arrive here from search, so the table below maps a task to its entry point and the guide that teaches it.

| To do this | Start with | Guide |
| --- | --- | --- |
| Build a causal graph | `dag`, `admg`, `cpdag`, `pag` | [Causal graphs](graph.md) |
| Turn a graph into an estimand | `identify_effect` | [Identification](identification.md) |
| Estimate an effect from data | `estimate_effect` | [Causal inference](inference.md) |
| Estimate per-unit effects (CATE) | `cate` | [Causal inference](inference.md) |
| Stress-test an estimate | `refute_effect` | [Checking an estimate](inference.md#checking-an-estimate) |
| Learn a graph from data | `discover`, `bootstrap` | [Causal discovery](discovery.md) |
| Decode edge probabilities into a DAG | `EdgeProbabilities`, `decode` | [Causal discovery](discovery.md) |
| Measure a rollout with no graph | `did`, `synthetic_control`, `interrupted_time_series` | [Panel designs](panel.md) |
| Forecast a series, or a whole hierarchy | `forecast`, `backtest` | [Forecasting](forecast.md) |
| Recover a hidden regime from one series | `recover`, `certify`, `calibrate` | [Hidden regimes](latent.md) |
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

Structure learning: the algorithms, the CI tests and scores they consume, the `discover` facade, and the edge-probability decoders (`EdgeProbabilities`, `decode`).

::: ergodic.discovery

## Causal inference

Effect estimation: the estimators, the learners that fill their slots, refutation, and the quasi-experimental designs.

::: ergodic.inference

### Per-unit effects (CATE/ITE) and explainability

The `cate` facade and the uniform per-unit result. `cate(graph, data, X, Y)` is identification-gated like `estimate_effect` and returns a `CATEResult`; `SCM.ite` returns an `ITEResult`. Both share the `PerUnitEffect` surface (`.values`, `.interval`, `.ate`, `__array__`). `CATEResult.explain` returns a `Heterogeneity` record for Tier-B characterization (BLP, GATES, CLAN, the calibration test, RATE).

::: ergodic.cate

::: ergodic.CATEResult

::: ergodic.ITEResult

::: ergodic.PerUnitEffect

::: ergodic.inference.Heterogeneity

### Causal forest, policy trees, and CATE model selection

The honest causal forest (GRF) is its own CATE learner, reachable through the `cate` facade (`method="causal_forest"` or `"grf"`) or directly as `CausalForest`; it returns a `CATEResult` with native per-unit infinitesimal-jackknife intervals, and `CausalForest(...).fit(...).variable_importance()` returns a `VariableImportance` driver signal. `policy_tree` (also `CATEResult.explain().policy_tree(...)`) learns an interpretable, depth-limited, doubly-robust-optimal treat/don't `PolicyTree`. `select_cate` ranks candidate CATE learners by the doubly-robust tau-risk and returns a `SelectionReport` with the selected result and a Q-aggregation ensemble.

::: ergodic.CausalForest

::: ergodic.inference.VariableImportance

::: ergodic.policy_tree

::: ergodic.PolicyTree

::: ergodic.select_cate

::: ergodic.SelectionReport

## Forecasting

Bayesian trajectories with causal structure in them: the `forecast` facade and its result, the mechanism contract and its zoo, the reconciliation menu, staged fitting and the training schedule, the rolling-origin backtester with its baselines and metrics, and the attribution waterfall.

Note the one shadowed name in the package. `ergodic.forecast` at the top level is the *function*, so `from ergodic import forecast` gives something callable; the module is reached with `from ergodic.forecast import backtest`, and only `import ergodic.forecast as mod` does the surprising thing of binding the function.

::: ergodic.forecast

## Welt

Welt's hosted foundation models, behind `pip install "ergodic[welt]"` (Python 3.11 to 3.13): `discover` runs a Welt causal discovery model and returns a `DiscoveryResult` (also `ergodic.discover(data, method="welt:arrow")`), `WeltLearner` fills a nuisance slot with a Welt tabular model under a shared fit budget, and `upload_dataset` and `load_research_dataset` move tables between ergodic and Welt. Every call is a Welt request; most are paid.

::: ergodic.welt

## Hidden regimes

Hidden-regime inference over one series: `recover` for the posterior state path and mechanism deltas, `certify` for the simulated-null existence test, `calibrate` for the measured false-fire rate, and `consensus` for laying several likelihoods' verdicts side by side. The built-in likelihoods live in `ergodic.latent.likelihoods` (`spectral_whittle`, `switching_ar`).

::: ergodic.latent

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
