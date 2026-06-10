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
- Causal discovery, the score stack: `GES` (Chickering's greedy equivalence
  search, held to the A* optimum in the tests) and `HillClimb` (steepest
  ascent with a tabu list) consume the same `Score` contract as `AStar`;
  `DiscreteBIC` (multinomial), `BDeu`, and `MixedBIC` (degenerate Gaussian,
  no edge restriction) join the Gaussian `BIC`, with executable
  score-equivalence tests; the `Score` base gains a native-categorical codes
  form (`local_from_codes`); `score="auto"` picks the score from the column
  schema the way `test="auto"` picks the test; and `GCM` (Shah and Peters)
  turns any regression `Learner`, native or sklearn, into a conditional
  independence test that catches nonlinear dependence the partial correlation
  misses. `DSeparationOracle` now answers about the observed margin (latent
  nodes are not query-able variables).
- Causal discovery with latent confounders: `FCI` returns a PAG (stable
  skeleton, the possible-d-sep phase, Zhang's complete rule closure, no
  selection bias), held in the tests to the canonical PAG of the truth under
  a perfect oracle. The graph layer grew the toolkit underneath:
  m-separation now reads circle marks through definite-status paths (the
  class-invariant answer on a complete PAG), `mag_of_dag` is the latent
  projection, `pag_of_mag`/`mag_of_pag` convert between a MAG and its
  equivalence class, `fci_closure` joins `meek_closure`, `set_mark` edits one
  endpoint, and `is_pag` became a real complete-PAG validator by round trip,
  exactly as `is_cpdag` did. Domain knowledge translates to PAGs through
  ancestry: tiers pin an arrowhead at the later end, `confound(a, b)` orients
  a bidirected edge, directional forbids and roots/sinks validate only.
- Skeleton screening and stability: `MMPC` (max-min parents-children, AND
  rule) returns an undirected support for `restrict_to`, recovering the MMHC
  hybrid as a composition; `bootstrap` reruns any method on row resamples and
  returns adjacency and orientation frequencies with a thresholded
  `summary()` graph, the first soft output. `Dataset.take(rows)` resamples
  rows, and `discover` dispatches on each algorithm's declared ingredient.
- Identification on equivalence classes: `identify_effect`, `adjustment_set`,
  `minimal_adjustment_set`, and `is_valid_adjustment_set` accept a CPDAG,
  MAG, or PAG and answer through the generalized adjustment criterion
  (Perkovic et al., 2018), so a success holds in every graph the class
  represents and `estimate_effect` runs straight off a discovery result.
  Held executably to the all-members definition: exhaustive over candidate
  sets against the enumerated members of small random classes. The graph
  layer grew the toolkit underneath: `possible_descendants` and
  `possible_ancestors` (possibly directed reachability), `visible_edges`
  (Zhang's no-hidden-confounding certificate), and the member enumerators
  `dags_of_cpdag` and `mags_of_pag` (deterministic, capped without silent
  truncation). The front-door and instruments raise on a class and point at
  the enumerators (the per-member IDA pattern); mixed graphs raise.

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
- `m_separated` on a CPDAG-kind graph now answers for the equivalence class:
  undirected edges read as circle marks and paths need definite status,
  matching the PAG reading (a literal tail-tail walk could claim connection
  where every member of the class is separated). Spell marks with `mixed(...)`
  to walk them literally. `identify_effect` on a CPDAG or MAG used to run
  silently with single-graph semantics; it now dispatches on the face, and
  mixed graphs raise instead of being misread.
