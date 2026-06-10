# CLAUDE.md

Guidance for working in this repo.

## What this is

`ergodic` is a Causal AI library (from ergodic.ai), published on PyPI as `ergodic`. Three pillars: causal discovery, causal inference, and process intelligence. They all sit on two shared objects, which are built first.

## Core objects (built)

- `ergodic.graph`: a mixed-graph family. One `MixedGraph` core with typed endpoint marks (tail, arrowhead, circle) covers DAG, ADMG, MAG, CPDAG, and PAG. Build with `dag`, `admg`, `cpdag`, `pag`, or `mixed`. Edges read as glyph strings, e.g. `dag(["X -> Y", "Y -> Z"])`. Graphs are immutable value objects: every edit returns a new graph. Separation (`m_separated`, `d_separated`), surgery (`do`, subgraphs), and interop (networkx, adjacency, dict) live here.
- `ergodic.knowledge`: `DomainKnowledge`, prior constraints (required/forbidden edges, tiers, roots, sinks, confounded pairs). Immutable, composable with `&`, self-checking via `conflicts`/`validate`, with projections (`permits_edge`, `is_consistent`, `apply_to`, `edge_mask`) for discovery to read.
- `ergodic.identification`: graph to estimand. `identify_effect(graph, X, Y)` tries adjustment (the complete criterion), front-door, then instruments, and returns an `Identification` record with the estimand. Also `adjustment_set`, `minimal_adjustment_set`, `frontdoor_set`, `instruments`, and their `is_*` checks. Graph-only and deterministic.
- `ergodic.data`: the typed datasets the algorithms read. One `Dataset` base carries a column `Schema` (continuous, categorical, ordinal, datetime, excluded; binary is derived) across five shapes: `TabularData`, `TimeSeriesData`, `PanelData`, `HierarchicalTSData` (with a `Hierarchy` aggregation tree), and `EventLog`. Build with `tabular`, `timeseries`, `panel`, `hts`, `event_log`. Immutable; a dataframe builds it and stays at the boundary (`data/_frame.py`), the math runs in numpy. Accessors: `variables`, `codes`, `matrix`, `dropna`, `select`, `aligns_with(graph)`. Containers only; the per-shape algorithms ride with their pillars.
- `ergodic.inference`: estimation (E1 + E2 built, plus Bayesian learners). `estimate_effect(graph, data, X, Y)` identifies the effect then estimates it, returning a distribution-shaped `EffectEstimate` (a point with an analytic or bootstrap standard error, or draws). A `CausalEstimator` is the strategy and a `Learner` is the model in each nuisance slot; the `Learner` is structurally a sklearn estimator, so any sklearn or xgboost model drops in. Adjustment: `OutcomeRegression` (S/T-learner), `IPW`, `Stratification`, `AIPW`, and the cross-fit `DML`, `DRLearner`, `RLearner`, `XLearner`. Front-door: `TwoStageRegression`. Instruments: `TwoStageLeastSquares`, `Wald`, `PLIV`. Heterogeneous effects through `cate(data)` on the CATE estimators. Binary treatment for most, binary or continuous for `DML` and `PLIV`. Native `OLS`, `LogisticRegression`, and `KNN` learners keep it dependency-free. A `BayesianLearner` carries a posterior over its mean (detected by `predict_draws`); a `PyMCLearner` wraps a user-written PyMC model (pymc optional, imported inside the methods). `OutcomeRegression` (g-formula) and `AIPW` (doubly robust, with a Dirichlet Bayesian bootstrap on by default) then return a posterior `EffectEstimate` (`kind="posterior"`) through the same accessors, and the T-learner gives per-unit CATE credible intervals via `cate_draws`/`cate_interval`. A Bayesian slot never bootstraps (that would refit the sampler); with no posterior path it falls back to the analytic or point result.

These re-export from the top level: `from ergodic import dag, admg, DomainKnowledge, identify_effect, tabular, estimate_effect`.

## Design decisions to keep

- Full graph surface from day one (marks model), algorithms added in phases.
- Immutable value objects everywhere (graphs and knowledge).
- Hard domain knowledge now. `energy`/`log_prior` are stubbed and raise; soft/Bayesian knowledge attaches there later.
- networkx only at the interop boundary in `graph/_io.py`. The algorithms are native.
- No graph posteriors yet; revisit at the Bayesian phase.
- Data enters as a typed `Dataset`. The dataframe library lives only at `data/_frame.py` (pandas native, polars and arrow via `to_pandas`; narwhals deferred). The two axes, row shape and column schema, stay separate.
- Estimation factors into a `CausalEstimator` (the strategy) and a `Learner` (the sklearn estimator surface, native by default, sklearn/xgboost optional). The result is distribution-shaped: a point now, posterior draws later through the same `EffectEstimate`.

## Layout

- `src/ergodic/graph/`: `_marks`, `_graph`, `_separation`, `_surgery`, `_io`.
- `src/ergodic/data/`: `_schema`, `_frame`, `_dataset`, `_tabular`, `_timeseries`, `_panel`, `_hierarchy`, `_hts`, `_eventlog`.
- `src/ergodic/inference/`: `_target`, `_result`, `_learner` (Learner + BayesianLearner), `_estimator`, `_crossfit`, `_estimate`, `_registry`, `learners/` (OLS, logistic, KNN, PyMC), `estimators/` (twelve, grouped by family).
- `src/ergodic/knowledge.py`, `src/ergodic/identification.py`.
- `tests/`: `test_graph.py`, `test_knowledge.py`, `test_identification.py`, `test_data.py`, `test_inference.py`, `test_estimators_e2.py`, `test_bayes.py`, `test_smoke.py`.
- `plans/`: dated implementation journal, one entry per phase (`*-causal-graph-and-domain-knowledge.md`, `*-identification.md`, `*-data-objects.md`, `*-effect-estimation.md`, `*-bayesian-learners.md`).
- `docs/`: user guides (tutorial, concepts, graph, knowledge, data, identification, inference) and the API reference, built with MkDocs.

## Working here

- uv for everything: `uv sync --group docs` to set up.
- `make check` runs ruff, mypy strict, and pytest. `make docs-serve` previews the docs.
- mypy is strict and ruff is the linter and formatter. Keep both green.
- Prose in `plans/` and `docs/` follows `plans/guides/writing_guide.md` (no em-dashes, plain words, contractions).

## What's next

Built object-first: graph, knowledge, identification, the typed data family, estimation through E2 (twelve estimators with cross-fitting and CATE over `TabularData`), and the Bayesian learner route (a `PyMCLearner` posterior through the same `EffectEstimate`, with per-unit CATE intervals). Next on the estimation track is the rest of E3: TMLE, the panel-shape estimators (difference-in-differences, fixed effects, synthetic control), instrument and front-door posteriors, cross-fit Bayesian nuisances, and the joint PyMC estimators (with a JAX or NumPyro backend behind the same `predict_draws`). Also in the journal: the full do-calculus ID algorithm, equivalence classes (skeleton to CPDAG) with a discovery algorithm, then PAG separation. Plan a phase before writing it, and update the journal as you go.
