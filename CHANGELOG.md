# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding: the `ergodic` package with `discovery`, `inference`,
  and `process` subpackages (placeholders).
- Tooling: uv project layout, ruff (lint + format), mypy (strict), pytest.
- MkDocs Material documentation skeleton, pre-commit hooks, and a Makefile.
- Causal graph objects (`ergodic.graph`): a mixed-graph core with typed endpoint
  marks covering DAG, ADMG, MAG, CPDAG, and PAG; immutable value semantics; d- and
  m-separation; the do-operator and subgraphs; networkx, adjacency, and dict interop.
- Domain knowledge (`ergodic.knowledge`): hard prior constraints (required and
  forbidden edges, tiers, roots, sinks, confounded pairs) with composition, conflict
  detection, and projections (`permits_edge`, `is_consistent`, `apply_to`, `edge_mask`).
- Identification (`ergodic.identification`): `identify_effect` plus the complete
  adjustment criterion, the front-door criterion, and conditional instruments, turning a
  graph and a treatment/outcome into an estimand.
- Typed data objects (`ergodic.data`): a `Dataset` base with a column `Schema`
  (continuous, categorical, ordinal, datetime, excluded) across five shapes,
  `tabular`, `timeseries`, `panel`, `hts` (with a `Hierarchy` aggregation tree),
  and `event_log`. Immutable, with type inference and overrides, codes and a float
  matrix, missing-value handling, and `aligns_with(graph)`. A dataframe builds it
  and stays at the boundary; pandas native, polars and arrow through `to_pandas`.
- Effect estimation (`ergodic.inference`): `estimate_effect` identifies an effect
  and estimates it over `TabularData`, returning a distribution-shaped
  `EffectEstimate` (a point with an analytic or bootstrap standard error, or
  draws). A `CausalEstimator` is the strategy and a `Learner` is the model in each
  nuisance slot; the learner is structurally a sklearn estimator. Adjustment:
  `OutcomeRegression` (S- and T-learner), `IPW`, `Stratification`, `AIPW`, and the
  cross-fit `DML`, `DRLearner`, `RLearner`, and `XLearner`; front-door:
  `TwoStageRegression`; instruments: `TwoStageLeastSquares`, `Wald`, and `PLIV`.
  Heterogeneous effects (CATE) through `cate`, binary or continuous treatment for
  `DML` and `PLIV`, and native `OLS`, `LogisticRegression`, and `KNN` learners.
- Bayesian learners (`ergodic.inference`): a `BayesianLearner` carries a posterior
  over its mean function (detected by `predict_draws`), and a `PyMCLearner` /
  `PyMCClassifier` wraps a user-written PyMC model (pymc optional, `ergodic[bayes]`).
  `OutcomeRegression` (the Bayesian g-formula) and `AIPW` (doubly robust, with a
  Dirichlet Bayesian bootstrap) return a posterior `EffectEstimate` through the same
  accessors, and the T-learner reports per-unit CATE credible intervals via
  `cate_draws` and `cate_interval`.
- Explicit treatment coding: `estimate_effect(..., treated=...)` picks which level
  counts as treated, and an implicit non-numeric coding now warns with the chosen
  levels (a string treatment used to be coded by alphabetical order, silently).
- The `"null"` identification strategy and `NullEffect` estimator: a graph with no
  directed path from treatment to outcome identifies trivially as a zero effect.
- `as_kind` accepts the kind name as a string (`g.as_kind("admg")`).
- Stratified cross-fitting folds for binary treatments, bootstrap shortfall
  diagnostics, and a clear error when a treatment arm is empty.
- Causal discovery (`ergodic.discovery`), the first build: a conditional
  independence test is a class constructed with data (the stateful oracle:
  encoding once, sufficient statistics, cached alpha-independent queries),
  with `Linear` (partial correlation), `Discrete` (G squared, natively
  categorical), `Mixed` (degenerate Gaussian for mixed columns), and a
  `DSeparationOracle` ground truth; a decomposable `Score` answers
  `local(node, parents)` with `BIC` (Gaussian); PC-stable orients through
  v-structures and the Meek rules and returns a CPDAG with separating sets;
  `AStar` is exact search over the order lattice; `discover(data, method=...)`
  is the facade and `discover` re-exports at the top level. Background
  knowledge enters in four modes (prune, protect, orient, validate), and
  `DomainKnowledge.restrict_to(skeleton)` confines any search to an adjacency
  support. The Meek rules, `cpdag_of_dag`, and the Dor-Tarsi extension check
  live in `ergodic.graph`, which makes `is_cpdag` a real validator.

### Fixed
- Instruments: descendants of the treatment are no longer certified as instruments
  (the exclusion test was unsound); latent nodes are excluded from adjustment sets,
  front-door sets, instruments, and conditioning sets; identification validates its
  inputs (unknown names and overlapping treatment/outcome raise); multi-treatment
  adjustment validity follows proper causal paths exactly.
- Graph construction: conflicting duplicate edges on one node pair raise instead of
  silently keeping the last one (also through `add_edge`, `from_networkx`, and
  `from_dict`); write a bow as an explicit latent confounder with `latent=`.
  Separation queries reject unknown node names instead of returning a wrong answer.
- Domain knowledge: `forbid_adjacency` covers bidirected and undirected edges;
  `conflicts` catches cycles spanning required edges and tier order together;
  precedence is transitively closed; `confound(a, a)` raises; the cycle checks no
  longer recurse (deep graphs build fine).
- Data: datasets copy their columns at the boundary, so mutating the source
  dataframe or array no longer mutates the dataset; tz-aware datetimes infer as
  datetime; missing categoricals and NaT reach `matrix()` as NaN instead of
  sentinel numbers; values outside declared categories raise; HTS mean coherence
  and `aggregate` share one definition.
- Inference: learner cloning works for sklearn pipelines and meta-estimators; the
  KNN zero-feature fallback averages the whole training set; the AIPW ATT posterior
  reweights the treated share consistently with the Bayesian bootstrap; the
  R-learner names its final slot when a learner lacks `sample_weight`.
- Docs: repaired the broken examples (`as_kind("admg")`, a single learner alias on
  AIPW, the phantom `"gbm"` alias) and refreshed stale passages.

### Changed
- `effect()` lost its unused `level` parameter; pick the interval level on the
  result (`estimate.interval(0.99)`).
- IV estimands now state the linear (parametric) modeling assumption.
