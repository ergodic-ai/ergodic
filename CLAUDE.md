# CLAUDE.md

Guidance for working in this repo.

## What this is

`ergodic` is a Causal AI library (from ergodic.ai), published on PyPI as `ergodic`. Three pillars: causal discovery, causal inference, and process intelligence. They all sit on two shared objects, which are built first.

## Core objects (built)

- `ergodic.graph`: a mixed-graph family. One `MixedGraph` core with typed endpoint marks (tail, arrowhead, circle) covers DAG, ADMG, MAG, CPDAG, and PAG. Build with `dag`, `admg`, `cpdag`, `pag`, or `mixed`. Edges read as glyph strings, e.g. `dag(["X -> Y", "Y -> Z"])`. Graphs are immutable value objects: every edit returns a new graph. One edge per node pair: conflicting duplicates raise, so a bow is written as an explicit latent confounder (`latent=`). Separation (`m_separated`, `d_separated`) validates node names, surgery (`do`, subgraphs), and interop (networkx, adjacency, dict) live here.
- `ergodic.knowledge`: `DomainKnowledge`, prior constraints (required/forbidden edges, tiers, roots, sinks, confounded pairs). Immutable, composable with `&`, self-checking via `conflicts`/`validate`, with projections (`permits_edge`, `is_consistent`, `apply_to`, `edge_mask`) for discovery to read. Adjacency bans cover every edge type and precedence is transitively closed.
- `ergodic.identification`: graph to estimand. `identify_effect(graph, X, Y)` tries adjustment (the complete criterion), front-door, then instruments, and returns an `Identification` record with the estimand. Also `adjustment_set`, `minimal_adjustment_set`, `frontdoor_set`, `instruments`, and their `is_*` checks. Graph-only and deterministic. Latent nodes are excluded from every returned set, inputs are validated, instruments are never descendants of the treatment (IV identification is linear/parametric and says so), and a graph with no directed path from treatment to outcome identifies as the zero-effect `"null"` strategy.
- `ergodic.data`: the typed datasets the algorithms read. One `Dataset` base carries a column `Schema` (continuous, categorical, ordinal, datetime, excluded; binary is derived) across five shapes: `TabularData`, `TimeSeriesData`, `PanelData`, `HierarchicalTSData` (with a `Hierarchy` aggregation tree), and `EventLog`. Build with `tabular`, `timeseries`, `panel`, `hts`, `event_log`. Immutable; a dataframe builds it and stays at the boundary (`data/_frame.py`), the math runs in numpy. Accessors: `variables`, `codes`, `matrix`, `dropna`, `select`, `aligns_with(graph)`. Containers only; the per-shape algorithms ride with their pillars.
- `ergodic.inference`: estimation (E1 + E2 built, plus Bayesian learners). `estimate_effect(graph, data, X, Y)` identifies the effect then estimates it, returning a distribution-shaped `EffectEstimate` (a point with an analytic or bootstrap standard error, or draws). A `CausalEstimator` is the strategy and a `Learner` is the model in each nuisance slot; the `Learner` is structurally a sklearn estimator, so any sklearn or xgboost model drops in. Adjustment: `OutcomeRegression` (S/T-learner), `IPW`, `Stratification`, `AIPW`, and the cross-fit `DML`, `DRLearner`, `RLearner`, `XLearner`. Front-door: `TwoStageRegression`. Instruments: `TwoStageLeastSquares`, `Wald`, `PLIV`. Heterogeneous effects through `cate(data)` on the CATE estimators. Binary treatment for most, binary or continuous for `DML` and `PLIV`; `treated=` picks the treated level of a non-numeric treatment (implicit coding warns), and the zero-effect `"null"` family lands in `NullEffect`. Native `OLS`, `LogisticRegression`, and `KNN` learners keep it dependency-free. A `BayesianLearner` carries a posterior over its mean (detected by `predict_draws`); a `PyMCLearner` wraps a user-written PyMC model (pymc optional, imported inside the methods). `OutcomeRegression` (g-formula) and `AIPW` (doubly robust, with a Dirichlet Bayesian bootstrap on by default) then return a posterior `EffectEstimate` (`kind="posterior"`) through the same accessors, and the T-learner gives per-unit CATE credible intervals via `cate_draws`/`cate_interval`. A Bayesian slot never bootstraps (that would refit the sampler); with no posterior path it falls back to the analytic or point result.

- `ergodic.discovery`: structure learning (D1 built). Two ingredient objects: a `CITest` is a class constructed with data (the stateful oracle: encoding per the test's declared form, cached alpha-independent queries; `Linear`, `Discrete`, `Mixed`, plus `DSeparationOracle` as ground truth) and a decomposable `Score` (`BIC`, Gaussian) answering `local(node, parents)`. Algorithms are thin consumers: `PC` (stable skeleton, v-structures, Meek closure, CPDAG with sepsets) wraps a test; `AStar` (exact, order lattice, node cap) wraps a score. `discover(data, method=...)` is the facade. Knowledge enters in four modes (prune, protect, orient, validate); `DomainKnowledge.restrict_to(skeleton)` confines any search to an adjacency support. Meek rules, `cpdag_of_dag`, and Dor-Tarsi extendability live in `ergodic.graph` (`_meek`), so `is_cpdag` really validates.

These re-export from the top level: `from ergodic import dag, admg, DomainKnowledge, identify_effect, tabular, estimate_effect, discover`.

## Design decisions to keep

- Full graph surface from day one (marks model), algorithms added in phases.
- Immutable value objects everywhere (graphs and knowledge).
- Hard domain knowledge now. `energy`/`log_prior` are stubbed and raise; soft/Bayesian knowledge attaches there later.
- networkx only at the interop boundary in `graph/_io.py`. The algorithms are native.
- No graph posteriors yet; revisit at the Bayesian phase.
- Data enters as a typed `Dataset`. The dataframe library lives only at `data/_frame.py` (pandas native, polars and arrow via `to_pandas`; narwhals deferred). The two axes, row shape and column schema, stay separate.
- Estimation factors into a `CausalEstimator` (the strategy) and a `Learner` (the sklearn estimator surface, native by default, sklearn/xgboost optional). The result is distribution-shaped: a point now, posterior draws later through the same `EffectEstimate`.
- Categorical encoding belongs to the algorithm wrapper, not the data layer. `data.matrix()` stays integer codes; estimation one-hots in its own design builder (`_covariate_matrix`); a discovery conditional-independence test owns its strategy, one-hot or natively categorical.

## Layout

- `src/ergodic/graph/`: `_marks`, `_graph`, `_separation`, `_surgery`, `_io`.
- `src/ergodic/data/`: `_schema`, `_frame`, `_dataset`, `_tabular`, `_timeseries`, `_panel`, `_hierarchy`, `_hts`, `_eventlog`.
- `src/ergodic/inference/`: `_target`, `_result`, `_learner` (Learner + BayesianLearner), `_estimator`, `_crossfit`, `_estimate`, `_registry`, `learners/` (OLS, logistic, KNN, PyMC), `estimators/` (thirteen, grouped by family, plus `_null`).
- `src/ergodic/discovery/`: `_result`, `_ci` (CITest base + DSeparationOracle), `_score`, `_encode`, `_stats`, `_registry`, `ci/` (linear, discrete, mixed), `scores/` (bic), `algorithms/` (pc, astar). Meek lives in `graph/_meek`.
- `src/ergodic/knowledge.py`, `src/ergodic/identification.py`.
- `tests/`: `test_graph.py`, `test_knowledge.py`, `test_identification.py`, `test_data.py`, `test_inference.py`, `test_estimators_e2.py`, `test_bayes.py`, `test_sklearn_interop.py`, `test_discovery_ci.py`, `test_discovery_pc.py`, `test_discovery_score.py`, `test_discovery.py`, `test_smoke.py`.
- `plans/`: dated implementation journal, one entry per phase (`*-causal-graph-and-domain-knowledge.md`, `*-identification.md`, `*-data-objects.md`, `*-effect-estimation.md`, `*-bayesian-learners.md`, `*-audit-and-hardening.md`, `*-discovery-taxonomy-and-design.md`, `*-discovery-d1.md`). Gitignored on purpose: an internal tracking device, not part of the repo.
- `docs/`: user guides (tutorial, concepts, graph, knowledge, data, identification, inference) and the API reference, built with MkDocs.

## Working here

- uv for everything: `uv sync --group docs` to set up.
- `make check` runs ruff (lint and format check), mypy strict, and pytest. `make docs-serve` previews the docs; `make docs` builds strict.
- mypy is strict and ruff is the linter and formatter. Keep both green.
- Prose in `plans/` and `docs/` follows `plans/guides/writing_guide.md` (no em-dashes, plain words, contractions).

## What's next

Built object-first: graph, knowledge, identification, the typed data family, estimation through E2 plus Bayesian learners, and discovery D1 (the CI-test and score ingredient contracts, PC to a CPDAG, A* exact search). On the discovery track, D2 is next: GES on the existing score contract, the mixed and discrete scores (degenerate-Gaussian BIC, BDeu), the GCM test that reuses `Learner`, then FCI (mostly PC machinery plus the journaled PAG-separation work), skeleton searches, and the bootstrap wrapper; the design doc (`*-discovery-taxonomy-and-design.md`) holds the full map. On the estimation track, the rest of E3 still waits: TMLE, the panel-shape estimators, instrument and front-door posteriors, cross-fit Bayesian nuisances, and the joint PyMC estimators. Also in the journal: the full do-calculus ID algorithm (decide bow multi-edges then) and PAG separation. Plan a phase before writing it, and update the journal as you go.
