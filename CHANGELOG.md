# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - Unreleased

To be released together with Welt.

### Added
- A versioned graph format. `MixedGraph.to_dict()` writes
  `"format": "ergodic.graph/1"`, and `ergodic.GRAPH_FORMAT` (also
  `ergodic.graph.GRAPH_FORMAT`) names the format this version reads.
  `MixedGraph.from_dict()` reads a dict without the field as version 1, so older
  output and hand-written payloads (`kind`, `nodes`, and edges of
  `{u, v, u_mark, v_mark}`) still load, and it raises `ValueError` naming the
  supported format on any other value. The rest of the `to_dict()` output is
  unchanged. Round trips are tested for every graph kind, latent and selection
  nodes included.
- `ergodic.welt`, an optional Welt integration installed with
  `pip install "ergodic[welt]"` (welt-client>=0.13, Python 3.11 to 3.13; the
  rest of ergodic never imports it, enforced by a test). `ergodic.welt.discover`
  and `discover(data, method="welt:<model>")` run a Welt causal discovery model
  and return a `DiscoveryResult` with the native DAG and Welt provenance in
  `diagnostics` (ids, model and configuration versions, the edge-score
  dataframe). A model known to return non-DAG graphs (`cdfm`, `avici`, or any the
  catalogue marks so) raises `WeltGraphNotDAGError` before anything is uploaded or
  paid for, and a result Welt does not declare a DAG raises it with the result
  attached; nothing is projected. `WeltLearner` puts any Welt classification or
  regression model in a nuisance slot, refuses `sample_weight`, and shares one
  client and a `max_fits` budget (default 10) with its clones.
  `upload_dataset` and `load_research_dataset` move tabular data between ergodic
  and Welt. `bootstrap` refuses `welt:` methods (each resample would be a paid
  job). (API-141)

### Changed (API-141)
- `DiscoveryResult.diagnostics` is typed `Mapping[str, Any]` so a Welt run can
  carry its provenance. Learner cloning honors a learner's own
  `__sklearn_clone__` override; scikit-learn estimators keep the `get_params`
  path.
- A weighted nuisance slot (the R-learner's `final`) refuses, before any fit, a
  learner that declares `supports_sample_weight = False`, as `WeltLearner` does.
  `discover(..., method="welt:<model>")` refuses `alpha` and local-only keyword
  arguments before any request.

### Changed
- Identification's refusal of a `mixed` graph now says how to fix it, reading
  the graph's marks. A fully directed acyclic graph gets two readings with their
  conditions: `graph.as_kind("dag")` for a search that assumes no hidden
  confounding (PC or GES with knowledge orienting every edge), and
  `graph.as_kind("mag")` for an FCI or RFCI result, whose directed edges mean
  ancestry and can hide confounding. A directed-plus-bidirected graph gets the
  same pair with `"admg"`. A partially directed graph lists the options (orient
  the rest with more knowledge, take one member with `consistent_extension`, or
  rediscover without the orienting knowledge); a graph that fits the CPDAG, MAG,
  or PAG rules names that face; a graph that fits none names seeded domain
  knowledge as the likely cause. Which graphs identification accepts is
  unchanged, and `estimate_effect` and `cate` raise the same message.
- A bound conditional-independence test or score handed data it was not bound
  to (for example `FCI(test=GCM(data)).fit(data)`) still raises, and the message
  now names the algorithm and both correct usages: fit the algorithm with no
  data, or build the test or score unbound and pass the data to `fit` or
  `discover`. A `DSeparationOracle` handed data says it takes none. The
  `discover` docstring now says it binds an unbound test.
- Install hints in error messages and module docs quote their extras
  (`pip install 'ergodic[scm]'`), which zsh would otherwise read as a glob.

### Fixed
- Docs: the GCM example in the discovery guide passed data twice and raised;
  the end-to-end discover-then-estimate example passed a `mixed` graph to
  `estimate_effect` and now re-faces it with `as_kind("dag")`, and says an FCI
  result goes in as it is or, when knowledge leaves it `mixed`, as a MAG and
  never a DAG; the graph guide's
  `as_kind` example imports `mixed`. The graph guide documents the format field.
- The README quotes the install extras (`pip install 'ergodic[bayes]'`), which
  zsh would otherwise read as a glob.
- `estimate_mediation`: the total effect is now computed directly as
  `E[Y(1, M(1))] - E[Y(0, M(0))]`. It used to be the pure NDE plus the pure NIE,
  which drops the mediated interaction whenever the treatment interacts with the
  mediator (always possible, since the outcome model is saturated in treatment and
  mediator). On `load_berkeley` (gender, department, admission) the total was
  -0.072 with a proportion mediated of 1.99; it is now the -0.142 raw gap and the
  proportion 1.01. `nde` and `nie` are unchanged (the pure effects), so under an
  interaction they no longer sum to the total; `proportion_mediated` stays
  `nie / total`. The `Design.estimand` no longer repeats the `NDE =`/`NIE =`
  prefix, and a treatment with more than two levels is told to recode to two
  levels instead of being pointed at `treated=` (which needs two levels too).
- `CATEResult.ate()` no longer treats per-unit analytic standard errors as
  independent (`sqrt(sum se^2) / n`); the units share one fitted model, so that
  was about `sqrt(n)` too narrow (0.0009 against a sampling spread of 0.038 on a
  4,000-row world). It now reports the AIPW average off the result's cached
  cross-fit scores (standard error `std(scores) / sqrt(n)`, 0.039 there, the
  same as `estimate_effect(strategy="aipw")`), the result's own standard error
  for a homogeneous result such as `DML.cate()`, and otherwise the mean as a
  point with no standard error. For the DR-learner's linear final stage the point
  is unchanged; for the causal forest it is the score average, which can differ
  slightly from the plain mean of the per-unit effects.
- `PolicyTree.value` is now the policy's mean outcome gain over treating no one,
  `mean[pi(X) * Gamma]`, in outcome units, as the docstrings described. It was
  `mean[(2 pi(X) - 1) * Gamma]`, so `regret()` and the gap to `treat_all_value`
  came out twice the outcome gain (example 11 priced its rule at ~$324k on a
  200k book; the true figure is ~$162k, value 0.82 against an oracle 0.80).
  `treat_none_value` is now 0 and `treat_all_value` (the ATE) is unchanged;
  `std_error` and `interval()` are on the new scale. The learned rule is the same
  (the two objectives share their argmax).
- Identification on a DAG or ADMG with `selection=` nodes now raises, as the
  class faces already did. It used to ignore the selection mark, so a selected
  collider `X -> S <- Y` identified as plain `P(Y | X)` although selecting on `S`
  makes the two dependent in the sample. This covers `identify_effect`,
  `identify_distribution`, `identify_mediation`, the adjustment, front-door and
  instrument helpers, and so `estimate_effect` and `cate`.
- `AStar` raises `ValueError` naming the violations when its searched DAG cannot
  satisfy the domain knowledge (for example a confounded pair), instead of an
  `AssertionError` (which `python -O` would also have skipped).
- Error messages and docstrings state limits plainly instead of internal planning
  words ("reserved seam", "not built yet", "phase 1", "MVP", "post-MVP", "when the
  causal layer lands", "E1", "v1", "lands in D2"): the neural and foundation CATE method names, `ITEResult.explain`,
  multi-unit synthetic control, a continuous mediator, per-member ID on a PAG,
  count and categorical forecast columns, scenario pins on endogenous nodes, the
  SCM's custom-mechanism and `engine="dml"` refusals, soft knowledge, and the
  `ergodic.latent` refusals. The neural and foundation names no longer suggest
  installing `ergodic[neural]`/`ergodic[foundation]`, which do not provide those
  learners in this version. A test reads
  every string constant in the package and refuses those words.
- `load_birthweight`: the `target` said -225 g with a tolerance of 90 g, the
  description and the case study said about -203 g, and neither figure traced to
  a published estimate on this extract. The target is now -231 g (tolerance
  30 g, naive -275.3 g), the AIPW estimate on this same 4,642-row Cattaneo (2010)
  extract in the Stata treatment-effects manual, and the description, the case
  study, and the example quote it with that source. The `ergodic.datasets`
  docstrings no longer call every bundled file public-domain or permissively
  licensed; they say which files derive from public-domain US government data
  and which are authors' replication data without an explicit licence, as the
  NOTICE does.
- Reprs: `CATEResult`, `ITEResult`, `DiDEffect`, `SyntheticControlEffect`, and
  `ITSEffect` print the short one-line summary again (with their own class name)
  instead of a field-by-field dump of every array, and `DiscoveryResult` prints
  the graph kind as `cpdag` rather than `GraphKind.CPDAG`.
- `SyntheticControlEffect.kind` is `"placebo"` instead of `"analytic"`: there is
  no standard error, and the interval is a placebo band.
- `HillClimb` breaks ties between score-equivalent moves by its documented
  enumeration order instead of by floating-point rounding. Adding `x -> y` and
  `y -> x` (or reversing a covered edge) score the same, but the computed deltas
  differ in the last bits depending on the BLAS build, so the climb took
  machine-dependent paths; on Apple Accelerate the default run on a five-node
  world stopped at a lower-scoring graph (BIC -127.0 against the truth's -123.2)
  with an extra edge. Moves within a relative 1e-9 are now ties, and a tie keeps
  the first move.
- The policy tree's docstrings no longer call its `value` and interval honest:
  they are measured on the scores the tree was chosen to maximize, so they are
  optimistic and the interval ignores the selection. `PolicyTree.evaluate`
  (new) scores the fixed rule on held-out doubly-robust scores and returns a
  `PolicyValue` with a valid interval and regret.
- The best-linear-projection docstrings no longer say the intercept is the ATE:
  the covariates enter uncentred, so the intercept is the projected effect at
  covariates all zero (the ATE only when they are centred). Outputs unchanged.
- `KNNRegressor` and `KNNClassifier` honour `sample_weight` (a weighted mean
  over the neighbours). They accepted it and dropped it, so the R-learner's
  weighted final stage ran unweighted with a KNN learner. `PyMCLearner` now
  declares `supports_sample_weight = False`, so the R-learner refuses it before
  any fit, and its `fit` raises on weights instead of ignoring them.
- KNN used before `fit` raises `RuntimeError` naming the learner instead of an
  `AssertionError`. `set_params` on the native KNN, OLS and logistic learners
  refuses unknown names, as sklearn does. `LogisticRegression` warns with the
  new `ergodic.inference.learners.ConvergenceWarning` when it stops at
  `max_iter` before converging.
- `MixedGraph.from_networkx` takes `kind=` like the module-level function (the
  classmethod had no such parameter). `MixedGraph` neighbourhood lookups
  (`parents`, `children`, `neighbors`, `spouses`, `ancestors`, `descendants`,
  `markov_blanket`) and `Hierarchy` lookups (`children_of`, `parent_of`,
  `is_leaf`, `leaf_descendants`) raise `ValueError` on a name that is not a node
  instead of answering with an empty set (or, for `is_leaf`, `True`), as
  identification and separation already did. `get_edge`/`has_edge` still answer
  `None`/`False` for an absent edge. `HierarchicalTSData.aggregate` on an unknown node now
  raises this `ValueError` rather than a `KeyError`.
- `SCM` refuses a graph that is not a DAG or an ADMG. A MAG passed the
  acyclic-directed-plus-bidirected check and was fitted as an ADMG, reading
  ancestral edges as direct mechanisms; it now raises with the alternatives
  (pick a member, or re-face a graph you read as one world with `as_kind`).
- `SCM.interventional`, `sample`, and `marginal` return posterior-predictive
  draws of the outcome, not a posterior of `E[Y | do(X)]`: the mean estimates the
  expectation, but `std_error` and `interval()` are the outcome's spread (sd 1.4
  in the guide's example, where the effect's own posterior sd is 0.023). The docstrings
  and the guide said otherwise, and the repr printed a "95% CrI". These results
  now carry `kind="predictive"` and print "95% predictive interval"; `ate` and
  `counterfactual` keep `kind="posterior"` and the credible interval.
- `SCM(cores=)` (new, default 1) caps the processes PyMC may use for the chains
  in every SCM fit, including the latent-district and sensitivity refits. PyMC
  otherwise ran up to four chains in parallel processes. The draws do not depend
  on it.
- `LatentResult.rhat_max` is the split R-hat after label alignment, over the
  sign-corrected delta draws grouped by chain. It was the raw R-hat, so chains
  sitting in mirror modes of a good fit reported 1.83 (the spread arm of the
  golden worlds; aligned, 1.01). The raw number moves to the new
  `rhat_max_unaligned`, and the R-hat warning now reads the aligned value.
- `CausalForest.cate()` records the share of per-unit standard errors clipped to
  zero (`diagnostics["ij_zero_se_share"]`) and warns when it exceeds 5%. With too
  few trees for the rows, the infinitesimal-jackknife Monte-Carlo correction
  exceeds the raw variance and the band has zero width; on a 4,000-row world 400
  trees gave 25% zero-width bands and 68% coverage of a nominal 90%, 1,600 trees
  1% and 87%. The estimator is unchanged; its docstring states these
  measurements and recommends `n_trees` of the order of half the rows or more.

## [0.1.1] - 2026-08-30

Tagged as `v0.1.1`; not uploaded to PyPI.

### Added
- Hidden-regime inference (`ergodic.latent`): one observed series, a hidden
  two-state switch, and three questions kept apart. `recover` returns the
  posterior path of the switch and a credible interval per named mechanism
  block (level, spread, rhythm, amplitude), `certify` decides whether a regime
  exists at all against control worlds simulated from a fitted no-switch model,
  and `calibrate` measures a configured detector's false-fire rate with a
  Wilson interval. `consensus` lays several likelihoods' verdicts side by side.
  The hidden chain (`MarkovChain`, `semi_markov`) is summed out by the forward
  algorithm, so NUTS samples only continuous parameters. Emissions plug in as
  a `RegimeLikelihood`; two ship, `spectral_whittle` and `switching_ar` (blind
  to amplitude, and documented as such). pymc is optional,
  `pip install 'ergodic[latent]'`. A guide page (`docs/latent.md`) and two
  example scripts come with it.

## [0.1.0] - 2026-08-17

The first release on PyPI: the graph, knowledge, identification, data,
inference, discovery, process, SCM, and forecast pillars.

### Added
- Tooling: uv project layout, ruff (lint and format), mypy (strict), pytest,
  MkDocs Material documentation, pre-commit hooks, and a Makefile.
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
- Quasi-experimental designs (`ergodic.inference`): estimators that read the
  panel structure instead of a graph, each stating its identification in a
  `Design` record and returning the same `EffectEstimate` surface. `did` over
  a `PanelData` runs the canonical two-period collapse for a single adoption
  date and Callaway-Sant'Anna group-time ATTs for staggered adoption (with
  `comparison="never_treated"` or `"not_yet_treated"` and unit-bootstrap
  inference); static two-way fixed effects ships as an explicit strategy with
  its staggered-adoption caveat and is never auto-picked. `event_study`
  reports ATT by event time with leads as the visible pre-trend check.
  `synthetic_control` fits simplex weights on the pre-period by projected
  gradient and reports per-period gaps with Abadie's in-space placebo
  inference (the RMSPE-ratio rank p-value; intervals come from the placebo
  band on purpose). `interrupted_time_series` over a `TimeSeriesData` is
  segmented regression with level and slope changes, optional seasonal terms,
  and Newey-West standard errors. All four re-export at the top level.
- Temporal causal discovery (`ergodic.discovery`): `discover(data,
  method="pcmci", max_lag=...)` accepts a `TimeSeriesData` or `PanelData`
  (panels pool windows within entities) and returns a
  `TimeSeriesDiscoveryResult` holding a window graph over lagged nodes like
  `X[t-1]`, per-link p-values, and a variable-level `summary_edges()`. The
  algorithm is PCMCI-shaped: a PC1 lagged-parent screen, momentary
  conditional independence tests for lagged links (conditioning sets drawn
  from a deeper lagged frame), a PC-stable contemporaneous phase conditioned
  on lagged parents, and collider plus within-slice Meek orientation. It
  reuses the existing `CITest` stack through `lagged_tabular`, so
  `test="auto"` and the cache come free. `granger` is the predictive
  baseline and returns a pair-level table rather than a graph (feedback
  across lags is normal; the docstring says what Granger does not claim).
  Shape guards keep the methods honest: iid methods reject temporal data
  (rows are not exchangeable) and `pcmci` rejects tabular data.
- The learn series: six beginner notebook guides (`docs/learn/`) that teach
  causal inference from zero on commercial examples, confounding and Simpson's
  paradox, good and bad controls, estimation with doubly robust and cross-fit
  estimators, heterogeneous effects and uplift targeting, instruments and the
  front door, and structure discovery ending at the discover-then-estimate
  bridge. Each guide is a py:percent script that mkdocs-jupyter executes at
  build time with fixed seeds, so the outputs and figures on the site are real
  and a broken example fails `make docs`. Docs review alongside: the API
  reference gains the discovery section (guide pages no longer inline `:::`
  blocks), stale forward references brought current, and the index routes
  beginners to the series.
- Learn guide 7, quasi-experiments: a staggered promo rollout recovered with
  difference-in-differences and an event study, and a flagship market's price
  change measured against its synthetic control, with the naive comparisons
  failing first. A new "Panel and time-series designs" guide page documents
  the design family, and the discovery guide gains its time-series section.
- Process intelligence (`ergodic.process`), the first build. The descriptive
  layer recovers the daily workflow over an `EventLog`: `process_map` (the
  directly-follows map as a value object with frequency and gap statistics
  and a `simplify` that records exactly what it dropped), `performance`
  (cycle times, and a waiting/service split when a lifecycle column carries
  start/complete pairs; with one timestamp per activity the split is honestly
  impossible and the result says so), `variant_table`, and six composable
  filters (time, variants, endpoints, activities, follows, performance), each
  declaring whether it drops events or whole cases and refusing to return an
  empty log. The causal layer is the point: `case_table` encodes a log into
  one row per case with timestamp-gated features (case-start features in
  `when="anytime"`; landmark modes `("after_events", k)` and
  `("after_elapsed", x)` measure prefix features at the landmark, drop
  already-treated cases with counts, and state their population), the named
  process confounders (open-case workload, calendar features), and an
  auto-built `DomainKnowledge` whose tiers forbid post-treatment adjustment
  mechanically, feeding `discover`, `identify_effect`, and `estimate_effect`
  unchanged; `kpi_panel` aggregates a log into a `PanelData` (or a single
  `TimeSeriesData`) so a process change becomes a natural experiment for the
  quasi-experimental designs. Learn guide 8 walks the whole arc: the process
  map convicts a review step of double its true cost, the case table acquits
  it, and a routing rollout is measured with `kpi_panel` into `did`.
- The process model layer (`ergodic.process`, second build). `discover_model`
  is the Inductive Miner implemented from the papers (pm4py is AGPL and is not
  used): recursive cuts of the directly-follows graph (exclusive choice,
  sequence, parallel, loop) into a block-structured `ProcessTree` (build one
  by hand with `seq`, `xor`, `par`, `loop`, and `TAU`; structural equality,
  canonical child order), with the infrequent variant behind `noise=` and
  provenance on the returned `ProcessModel` (flower fall-through paths,
  filtered arcs, dropped events). `ProcessTree.to_net()` converts to a
  `WorkflowNet` that is sound by construction, and `is_sound` verifies any
  net for real by reachability (option to complete, proper completion, no
  dead transitions; capped, raising past the cap rather than guessing).
  `conformance` checks a log against a model by token replay (the standard
  missing/remaining/consumed/produced fitness) and by optimal alignments
  (Dijkstra over the synchronous product with deterministic tie-breaking;
  per-variant costs and sync/log/model moves). `decision_points` and
  `decision_table` read an exclusive split causally: the branch taken is the
  treatment, the landmark is the moment the case reveals its branch, every
  covariate is measured strictly before it (workload, calendar, prefix
  features, case attributes), the auto-built `DomainKnowledge` tiers order
  covariates before branch before outcome, the default outcome is the
  remaining time from the decision, and a first-class `PositivityReport`
  (per-branch counts and shares, propensity overlap for two-way splits)
  warns when the branches barely overlap. Splits with a silent branch are
  refused with the reason. Learn guide 9 walks the arc: the miner finds the
  routing split, conformance separates deviation from structure, the naive
  lane comparison overstates the cost of review by a factor of three, and
  the decision table recovers the planted effect.
- Learn guide 8, process mining: the descriptive workflow on its own, in
  three steps on a simulated fulfillment flow, with the process maps drawn
  as real graphs (traffic-scaled arrows, counts and median gaps on every
  arc): see the process, measure it (the waiting-versus-service split finds
  the planted pick queue, eight hours of waiting in front of 26 minutes of
  work), and drill into the slice that hurts, with `simplify` declaring what
  it dropped. The process guides renumber to make room: process intelligence
  is guide 9 and decision points is guide 10.
- Selection bias end to end in the graph layer: `is_mag` reads the full
  Richardson-Spirtes ancestral conditions (undirected endpoints carry no
  arrowheads), `mag_of_dag` projects latent and selection nodes together
  through the one-witness-set theorem, `fci_closure` runs Zhang's complete
  ten rules (R5 to R7 are the selection rules), `mag_of_pag` resolves
  circle-tail edges by arrowhead augmentation, and `mags_of_pag` enumerates
  the selection members behind an explicit `selection=` switch. The
  d-separation oracle conditions every query on `graph.selection`, so FCI
  needs no algorithmic change and lands exactly on the canonical PAG of
  random latent-plus-selection worlds, held executably (Richardson-Spirtes
  Theorem 4.18, round trips, and the chordless-cycle world only R5 can
  orient). Identification raises on selection marks instead of silently
  misreading them.
- Discovery D4, the iid completion. PC gains `colliders="conservative"`
  and `"majority"` (Ramsey 2006; Colombo-Maathuis 2014): unshielded triples
  are voted across both neighborhoods' subsets, ambiguous triples are never
  oriented and gate Meek R1/R3 through `meek_closure`'s new `ambiguous=`
  parameter, and the face stays honest. RFCI (`method="rfci"`) skips FCI's
  possible-d-sep phase, re-tests triples and discriminating paths instead,
  and returns an RFCI-PAG whose marks stay sound (held executably against
  random latent, confounded, and selection worlds, with exact FCI agreement
  wherever possible-d-sep has nothing to do). BOSS and GRaSP (`method=
  "boss"`, `"grasp"`) search permutations through a grow-shrink projection,
  BOSS with a BES pass reusing GES's backward phase, both held to the A*
  optimum. DirectLiNGAM (`method="lingam"`) fully orients a DAG from
  non-Gaussian data with the Hyvarinen-Smith pairwise likelihood ratio and
  native backward-BIC pruning. A graphical-lasso screen (`method="glasso"`)
  estimates a sparse precision by block coordinate descent (matching
  sklearn entry for entry), picks the penalty by EBIC at the
  support-restricted refit MLE, and feeds `restrict_to` like MMPC. The
  facade learns methods that consume neither a test nor a score.
- Temporal unblockers: `bootstrap` accepts `method="pcmci"` through a
  moving-block design (a series resamples blocks that become entities of a
  panel, so no lag window ever crosses a block seam; a panel resamples
  whole entities; `block_length` overrides the default), and PCMCI reads
  `DomainKnowledge` with variables as processes: forbids, tiers, roots, and
  sinks prune at every lag, tiers orient contemporaneous edges, self-memory
  survives, `require` validates after the run, and `confound` raises.
- Real-world datasets (`ergodic.datasets`), the first set: sklearn-style loaders
  that return a `Bunch` carrying the typed dataset under `data` plus its causal
  question (treatment, outcome, confounders or instrument, and the `target` the
  literature already measured). `load_birthweight` (maternal smoking, backdoor
  adjustment), `load_berkeley` (the 1973 admissions table, Simpson's paradox),
  `load_pension401k` (401(k) eligibility, double machine learning), `load_smoking`
  (Proposition 99, synthetic control) and `load_germany` (reunification), and
  `load_asia` / `load_earthquake` (small networks written from their papers and
  sampled, for discovery). The bundled data ships in the wheel; provenance and
  citations are in the NOTICE. Example scripts in `examples/` run each analysis
  next to its known answer, and the tests check the recovery (smoking lands within
  the placebo band at -21 packs, double ML moves the pension gap from $19.5k to
  about $9k, the Berkeley gap reverses under department).
- Refutation (`ergodic.inference`): `refute_effect(graph, data, treatment, outcome,
  method=...)` stress-tests an estimate by re-running it under a perturbation and
  returns a `Refutation`. The model-agnostic checks: `"placebo_treatment"` scrambles
  the treatment and the effect should vanish, `"random_common_cause"` adds an
  irrelevant covariate and the effect should hold, `"data_subset"` re-estimates on a
  random slice. None proves an estimate right; a failure is a real warning.
  `refute_effect` and `Refutation` re-export at the top level.
- Case studies (`docs/case-studies/`): five narrative pages that apply the tools to
  a famous dataset and recover the number the original study published. Smoking and
  birthweight (backdoor adjustment), the 1973 Berkeley admissions paradox (Simpson's
  paradox), 401(k) eligibility (double machine learning), Proposition 99 (synthetic
  control), and structure discovery on a known network. Each opens with a causal
  balance table, shows the naive answer failing, recovers the published figure, and
  ends with a refutation check. They execute on every docs build, so the prose and
  the code stay in step.
- The general do-calculus ID algorithm (`ergodic.identification`):
  `identify_distribution(graph, outcome, do, given=...)` runs Shpitser-Pearl ID
  (and IDC when `given` is set) on a single DAG or ADMG, deciding whether
  `P(outcome | do(do), given)` is identifiable and returning the estimand as a
  typed `Functional` (the `Prob` / `Product` / `Marginal` / `Ratio` algebra, with
  `render`, `free_vars`, and `simplify`; public but marked provisional) or the
  `Hedge` witness when it is not. An empty `do` is the observational query.
  `identify_effect` now falls through to general ID after its named ladder misses
  on a single graph, so a graph that is do-calculus identifiable but outside
  adjustment, the front-door, and instruments (the Verma graph and its kin)
  returns `strategy="id"` with the functional set; a non-identifiable graph
  returns `strategy="none"` with the hedge. The `Identification` record gained
  `functional` and `hedge` fields. `ida(graph, treatment, outcome)` identifies
  across an equivalence class: it enumerates the members of a CPDAG and runs
  `identify_effect` per member, returning an `IDA` record (`members`,
  `n_identifiable`, `all_identifiable`, `agree`, `strategies`, `estimands`). On a
  PAG or MAG it returns the class-level adjustment answer where one applies and
  raises otherwise, since general per-member ID needs a DAG realization of each
  member. Held executably to the soundness-against-truth anchor (the
  functional evaluated on random discrete worlds matches the truth) and the
  identifiability catalog.
- The discrete plug-in estimator (`ergodic.inference`): `estimate_effect` on a
  discrete `strategy="id"` graph estimates the general functional directly,
  evaluating the empirical sum-product with a bootstrap standard error. Discrete-
  only and binary-treatment guarded, with a positivity check that raises on an
  empty stratum the formula divides by (the discrete analog of the overlap check);
  a continuous variable raises and points at the SCM pillar.
- Mediation (`ergodic.identification`, `ergodic.inference`):
  `identify_mediation(graph, treatment, outcome, mediators)` decides the natural
  direct and indirect effects (NDE, NIE) via a covariate set that blocks the X-Y,
  X-M, and M-Y back-doors (the three adjustment conditions), stating the cross-
  world assumption (no mediator-outcome confounder affected by treatment) in words
  since no data can test it, and returns a `MediationID`. The controlled direct
  effect (CDE) routes through the general ID engine as a plain interventional
  contrast and can identify even when NDE and NIE do not.
  `estimate_mediation(graph, data, treatment, outcome, mediators, ...)` is the
  g-computation estimator: NDE, NIE, and the total effect (each a distribution-
  shaped `EffectEstimate`), the per-level CDE, a `Design` record naming the
  assumptions, and `proportion_mediated` (flagged when the total effect is near
  zero). Binary treatment and a discrete mediator; held to a planted linear world
  where NDE and NIE recover the structural coefficients.
- Forecasting (`ergodic.forecast`), v1: `forecast(data, horizon)` fits a Bayesian
  `TemporalMechanism` per series and returns a distribution-shaped `ForecastResult`
  (per-node draws, `median`, `interval`, `to_frame`, provenance, and per-node
  sampler diagnostics). The mechanism zoo is `StructuralMechanism` (a damped local
  linear trend times a season plus a regression on graph parents, composed from
  those pieces rather than fused), `CausalOU` (drivers move the long-run attractor,
  so the transient answers "when" and not just "how much"), `Bass` (the diffusion
  S-curve), and `ScenarioMechanism`; mechanisms are auto-assigned from the schema
  with the seasonal period read off the time index, and count or categorical
  columns raise at a named seam. Hierarchies reconcile without refitting through
  `reconcile(result, hierarchy, method=...)`: `bottom_up`, `top_down` (historical or
  per-draw forecast proportions), `middle_out`, and the `mint` projection (diagonal
  posterior weights or `mint_shrink`, with `trust={node: w}`), each coherent to
  machine epsilon read relative to the size of each parent. Drivers ride a window
  graph over lagged names, exactly what `discover(..., method="pcmci")` returns:
  staged fitting walks the `TrainingSchedule` and hands each node's whole forecast
  posterior down to its children by injecting it into the child's trace, joint
  fitting puts every node in one model and keeps the cross-node correlation, and
  `fit="auto"` picks between them. `scenario=` pins a known future (an array, a
  scalar, or `"hold"`) and states that it makes no causal claim; pinning an
  endogenous node is refused by name and points at the `do=` that waits for
  identification on window graphs. `backtest(model, data, ...)` scores anything that
  fits a series and forecasts a horizon at rolling origins, with `Naive`,
  `SeasonalNaive`, and `Drift` baselines, fifteen metrics including MASE and CRPS,
  per-fold seeds, and the sampler's own report aggregated per fold;
  `forecast_model(**options)` wraps the facade as a backtestable model.
  `ForecastResult.explain()` returns the attribution waterfall: a baseline bar, a
  seasonality bar, and one bar per driver over any `Window` of the horizon, each
  carrying posterior draws and a credible interval, summing to the total with a
  reported `closure_error`. The structural engine recomposes the mechanism's own
  recorded pieces and costs no sampling; the replay engine (for a mechanism that
  does not expose them) zeroes the factor's sites in a copy of the trace and
  forecasts again on the stored seed. pymc is optional, `pip install
  'ergodic[forecast]'`, and the bare install never imports it.
- The structural causal model (`ergodic.scm`): a Bayesian `SCM` on PyMC with
  explicit-noise mechanisms (`GaussianAdditive`, and `BinaryGumbelMax` /
  `CategoricalGumbelMax` for discrete nodes), `pm.do` interventions, and batched
  twin-network counterfactuals with per-draw uncertainty. `ate` routes through
  `identify_effect` first and the return type names the world: a back-door
  effect returns a posterior `SCMResult`, a non-identified one returns a
  `Sensitivity` curve with no point accessor. A latent-confounded ADMG fits its
  confounded districts jointly, so a front-door effect computes from the fitted
  model; instruments and general ID still raise and point at `estimate_effect`.
  `SCM.ite` returns per-unit individual treatment effects off the twin network,
  as a contrast between two levels or a shift from each unit's own baseline.
  pymc is optional, `pip install 'ergodic[scm]'`.
- Per-unit effects and the CATE toolkit (`ergodic.inference`): `CATEResult` and
  `ITEResult` share one array-like `PerUnitEffect` surface (`values`,
  `interval`, `ate`, `to_frame`), and every meta-learner's `cate` returns one.
  The top-level `cate(graph, data, X, Y, method=...)` is identification-gated
  like `estimate_effect`; `uncertainty=False` skips the per-unit uncertainty
  pass for a fast point-only result. `CATEResult.explain()` gives honest
  characterization off out-of-fold scores (best linear projection, sorted group
  effects, CLAN, the calibration test, RATE). `CausalForest` is a native honest
  generalized random forest with infinitesimal-jackknife intervals,
  `policy_tree` learns an interpretable treat-or-not rule with its policy value
  and regret, and `select_cate` ranks candidate learners by doubly robust
  tau-risk and returns a Q-aggregated ensemble.

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

[Unreleased]: https://github.com/ergodic-ai/ergodic/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ergodic-ai/ergodic/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ergodic-ai/ergodic/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ergodic-ai/ergodic/releases/tag/v0.1.0
