# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2026-10-24

### Fixed
- Linear and GCM use `erfc` for normal tails: e.g. z=9 now gives approximately
  2.26e-19 instead of zero. Values below binary64 range still underflow (z=40);
  they are not floored or presented as accurately representable probabilities.
- `GraphSamples.top(k)` groups numerical weight ties at relative tolerance 1e-12,
  then log-target ties at absolute tolerance 1e-9, anchored at each group's largest
  value, and sorts final ties by named directed-edge tuples. Weights are unchanged.
- MMPC forward associations within absolute p-value distance 1e-15 now tie by
  variable name; alpha decisions keep unrounded p-values. This stabilizes greedy
  admission when tiny tails split, without claiming invariance to variable renaming.
  On 150 registered seven-node worlds, the tie change alone changed no skeletons
  (137 total skeleton errors before and after); PC/FCI/RFCI were also unchanged.

### Documentation
- GCM's default KNN remains uncalibrated. A registered direct-CI study (200 paired
  seeds per family/sample size) found 7.5–12% null rejection at nominal 5%.
  OLS failed on nonlinear conditional means, so neither qualified as a general
  default. Discover and graph-testing guides now disclose the measured limits.

## [0.3.1] - 2026-10-20

Correctness and inference safeguards from the pre-release audit (API-179 through
API-182). No new model families or hosted capabilities are added.

### Fixed
- Fitted CATE predictors reuse their training category vocabulary. Reordering
  categorical metadata no longer swaps predictions; unknown categories and
  incompatible prediction schemas fail explicitly.
- `estimate_effect(strategy="t_learner")` selects separate treatment-arm models,
  consistently with `cate(method="t_learner")`. Contradictory alias settings fail.
- GCM refuses constant endpoints, interpolating nuisance fits and degenerate
  residual products rather than returning a misleading independence p-value.
- Failed cross-fitted heterogeneity rankings no longer fall back to in-sample
  rankings for inferential summaries or CATE model selection.
  DML has no honest out-of-fold ranking proxy, so its GATES, CLAN, RATE and
  calibration summaries now refuse too; BLP and descriptive effects remain available.

### Changed
- Frequentist AIPW uses cross-fitting by default. Explicit in-sample mode remains
  available; Bayesian nuisance fits retain their documented in-sample posterior
  path. Results report the resolved mode and propensity diagnostics.
- Poor estimated overlap triggers diagnostics and warnings; severely degenerate
  propensity fits fail by default. An explicit override does not certify valid
  inference, and no rows are silently trimmed.
- Generic instrumental-variable estimates are labeled `linear_iv`. A binary
  complier `late` interpretation requires explicit assumptions and the supported
  treatment/instrument configuration. Scalar results retain identification.

### Migration

Point estimates and intervals can change because of corrected fitting and method
selection. Set seeds explicitly when comparing runs. To request the old parametric
AIPW recipe, use `mode="in_sample"`; this does not make arbitrary interpolating
learners valid. Inspect overlap diagnostics rather than suppressing failures.
Existing code that reads the IV target string must accept `linear_iv` for generic
IV coefficients. See the effect-estimation and heterogeneous-effect guides for
precise modes, assumptions and failure behavior.

## [0.3.0] - 2026-10-17

To be released together with welt-client 0.14 (API-164): decoding with domain knowledge
(API-151), graph evaluation (API-152) and the Welt Chronos-2 forecaster (API-163).

### Added
- Soft domain knowledge. `require`, `forbid`, `forbid_adjacency`, `forbid_between`,
  `order`, `with_tiers`, `root`, `sink` and `restrict_to` take `confidence=`. Left
  out (or 1) a statement is the hard constraint it always was. A value in `[0.5, 1)`
  makes it soft: it lands in the new `DomainKnowledge.soft` field as a
  `KnowledgeStatement` (checked on construction), and 0.5 changes nothing. Values
  below 0.5, above 1, NaN or a bool raise `ValueError` (state the opposite instead).
  The confidence applies to each pair a statement covers. `confound` stays hard
  only. `&` pools soft statements as a set, so a statement made twice counts once.
- `DomainKnowledge.log_odds(variables)`: the soft statements' summed log likelihood
  ratios on each directed edge, against "no edge" on its pair. `log_prior(graph)` and
  `energy(graph)`, which used to raise, now score a graph under the soft statements
  (minus infinity when it breaks a hard constraint). Soft orders are weighed pair by
  pair and are not closed transitively. `DomainKnowledge.statements()` lists every
  statement, read from the stored fields (hard) and the soft set, so equal knowledge
  lists equal statements. `DomainKnowledge.hard()` drops the soft statements.
- `decode(..., knowledge=)` weighs soft statements: each pair's directed states are
  multiplied by the statement ratios and renormalized before any decoder reads them,
  so all six decoders run unchanged and `exact_map` is exact on the combined score.
  A soft weight on what a hard constraint settles is dropped. Greedy with
  `threshold=` multiplies the input value's odds by the ratio on that direction.
  `diagnostics["knowledge_objective"]` is the score on the weighted states;
  `objective` keeps scoring the input.
- `decode` with `confound(a, b)` returns an ADMG: the directed part is decoded as
  before and `a <-> b` is added, to the graph and every candidate. A pair that also
  got a directed edge (a bow) carries its confounder as an explicit latent node
  `L__a__b`. `identify_effect` reads both. `diagnostics` add `graph_kind`,
  `confounded_pairs` and `latent_confounders`; the repr counts observed variables and
  names the latent confounders.
- A knowledge audit on every decode with knowledge: `diagnostics["knowledge_audit"]`
  has one entry per statement (the model's probability that it holds, from direct
  edges; the weakest of its pairs and the pairs below 0.1; whether it holds, an order
  read through directed paths, in the decoded graph and in a knowledge-free decode
  with the same settings and seed), `knowledge_cost` the score the knowledge cost,
  and `knowledge_conflicts` the statements with a pair the model gives under 0.1.
  `knowledge_violations_without_knowledge` lists what `violations()` finds on the
  knowledge-free graph, `confound` pairs left out, so an order broken only through
  the closure shows too.
  `n_hard_statements` and `n_soft_statements` count them. `decode(..., audit=False)`
  and `Decoder(..., audit=False)` skip it.
- `ergodic.KnowledgeConflictWarning` (a `UserWarning`), raised once by a decode whose
  knowledge strongly conflicts with the model, pointing at your call.
- `test_graph(candidate, data)` (also `ergodic.discovery.test_graph`) tests a
  candidate graph against data (API-152, DEC-082). It lists one implied
  independence per pair of observed variables without an edge, tests each with the
  conditional-independence test `discover` would pick from the columns (or `test=`),
  and combines the p-values with Fisher's C on `2k` degrees of freedom. A DAG uses
  Shipley's basis set (the parents of both); a CPDAG uses the same set in its
  consistent extension; an ADMG, a MAG, a DAG with `latent=` or `selection=` nodes
  and a PAG (through one member MAG) use m-separation, with the parents of both when
  they separate and otherwise a minimal set of the pair's observed ancestors. Every
  conditioning set is checked against the graph's separation routine, and a pair no
  observed set separates is reported in `untestable`. The result, a frozen
  `GraphTest`, carries the statistic, p-value, `k`, the test, `alpha`, notes, and the
  claims (`ImpliedIndependence`) ranked by p-value with Holm-adjusted p-values and
  decisions; `to_dict()` and `to_frame()` export it. The combined p-value is the
  level-`alpha` decision (`rejected`); the Holm decisions locate the failing claims.
  Claims whose test has zero degrees of freedom go to `vacuous` and are left out of
  `k`. An infinite value in an observed continuous column, or a non-finite p-value or
  statistic from the test, raises instead of reading as a pass; missing rows are
  dropped with a note. Only the default partial-correlation test is calibrated; the
  result notes any other test, and says that `GCM` with a nearest-neighbour learner
  and G squared with fewer than 1,000 rows over-reject true graphs. Claims whose
  p-values tie (for example, underflowed to 0) rank by the larger `|statistic|`. The
  p-value cannot separate Markov-equivalent graphs and a pass does not confirm the
  graph; the new guide "Test a graph against data" says so. Size with the default
  partial-correlation test on six-variable linear-Gaussian worlds at `alpha = 0.05`:
  the registered run (200 worlds per sample size) rejected the true graph in 9.5% of
  worlds with 200 rows, outside its [2%, 8%] band, and 4.5% with 1,000; a
  pre-registered replication on 2,000 fresh worlds per size gave 4.65% [3.77%, 5.67%]
  and 4.85% [3.95%, 5.88%], inside its [3.5%, 6.5%] band. Power against one missing
  edge of weight at least 0.5 is close to 1.
- `ergodic.welt.WeltForecaster` (also importable as `ergodic.welt.Chronos`) runs
  Welt's hosted Chronos-2 as an ergodic `ForecastModel` (API-163, DEC-087). `fit`
  checks the columns and keeps the window locally; each `forecast(horizon, *, seed,
  future=None)` is one paid `POST /v1/forecasts` job (DEC-085), sent with
  `welt.Client.submit_forecast` and read with the SDK's `Job.result`. Every target column
  is one series of a panel by default (`cross_learning=False`), or the targets of one
  series with `multivariate=True`. `past_covariates=` and `future_covariates=` map to
  Welt's covariate roles, categorical by the dataset's schema (cells sent as strings
  or integers, `True` as `1`; other cells as finite numbers, a boolean as 0.0/1.0);
  future covariates need `future=` (a dataframe or `TimeSeriesData`, only those
  columns are read). A
  `HierarchicalTSData` is forecast at every node (an aggregate without a column from
  its children's summed history, a node left out of `targets=` from its children's
  draws), and `reconcile=` reconciles inside `forecast`. The first forecast reads the
  catalogue entry (free) and refuses, before paying, a model without the `forecast`
  operation or a request outside every measured `forecast_envelopes` corner;
  `max_context=` sends only the most recent rows. Tables always go inline, as
  welt-client sends them, and a request over `welt.forecast.MAX_REQUEST_BYTES`
  (the API's 10 MiB limit) is refused before sending. The first job's model and
  configuration versions are required of every later job of the forecaster and its
  copies (and checked against each result), which also share one client and one
  `max_forecasts` budget (10) of jobs Welt accepted. Each node's
  `WeltForecastDiagnostics` records the job id, versions, quantile levels, Welt's own
  quantiles, `context_length` and `rearranged_steps`. A numeric time index, or a
  datetime index on a calendar Welt has no unit for (`B`, `C`, `BME`, `SME` and the
  like, declared or inferred by pandas; a declared one is trusted as given), is sent as
  consecutive days from 2000-01-01; any other irregular datetime index
  (daily data with missing days, say) is refused with a hint to reindex and leave
  the gaps as NaN. `NATIVE_QUANTILE_LEVELS` lists the 21 levels requested by default.
- `ergodic.forecast.quantile_draws(levels, quantiles, *, draws, seed)` turns
  per-step quantiles into `(S, H)` draws by inverse-CDF sampling: rows sorted
  (monotone rearrangement), linear between levels, exponential tails matched to the
  density at the outer levels, each step sampled independently. The draws reproduce
  the per-step marginals only, so sums over steps or series, and the intervals,
  coverage and CRPS of a `bottom_up` or `mint` reconciliation, treat the errors as
  independent.
- `ergodic.forecast.CovariateForecastModel`: a `ForecastModel` that names
  `future_covariates` and takes their values over the horizon as
  `forecast(..., future=...)`. `backtest` hands such a model, at each origin, those
  columns over the held-out window and nothing else, and records them in
  `BacktestReport.future_covariates`. It refuses, before any fit, a declared
  covariate that is not a column, a string in place of a sequence, and one the model
  names among its `targets`; a model that forecasts a declared covariate without
  naming it is refused on the first fold's result. Models without the attribute are called
  exactly as before.
- New guide "Forecast with Welt's Chronos-2": forecast with a known promotion
  calendar, backtest against baselines, and reconcile a hierarchy.

- A posterior over DAGs under a method (API-152, DEC-082). `graph_posterior(probabilities)`
  weighs each DAG by its pair log score (the pair states the decoders read) plus the
  knowledge's `log_prior`, over the DAGs the hard knowledge allows; order statements are
  read ancestrally, as `violations` reads them. `graph_posterior(data, score=...)` uses a
  decomposable score with a prior uniform over DAGs times the knowledge; BGe and BDeu are
  marginal likelihoods, the BIC family is labelled approximate. The result,
  `GraphSamples`, holds the distinct DAGs with weights, `edge_probabilities()` (pair
  states) with `edge_errors()`, `probability(feature)` with a Monte Carlo error,
  `log_target(graph)`, `best_graph` and `top(k)`. It is exact when it can be: every DAG
  enumerated up to 4 variables (5 for a score), and for edge probabilities without a
  path-level order constraint a subset recursion over sink sets (about `3^n` steps, up
  to `max_exact=15`) that gives the exact normalizer, exact edge marginals, exact draws
  and the exact best graph. Otherwise multi-chain structure MCMC (single-pair add,
  delete and reverse moves; each chain a four-rung parallel-tempering ladder; chains
  from the greedy decode and random DAGs) with split
  R-hat and bulk effective sample sizes, Rao-Blackwellized edge marginals for edge
  probabilities, and a `PosteriorConvergenceWarning`.
- `evaluate_graph(candidate, probabilities_or_samples)` reads a DAG, CPDAG, PAG, MAG or
  ADMG against that posterior (`GraphEvaluation`): the log-ratio to the best graph, the
  credible level (the mass ranked strictly above, ties half; a Bayesian tail
  probability, not a p-value), one `GraphClaim` per statement with its probability and
  Monte Carlo error, the class probability of a CPDAG, the expected SHD against the
  between-draw baseline (with the former's Monte Carlo error), and P(acyclic) under raw
  independent marginals. When every draw agrees, a Monte Carlo error is the rule-of-three
  bound `3 / draws` (`bound` on `FeatureProbability`, `GraphClaim` and
  `AdjustmentGroup`) rather than 0, and under the exact
  recursion the candidate's own share of its credible level is exact. A bidirected
  edge or latent node is listed as not assessable.
- Effects over a posterior over DAGs (DEC-082 decision 6). `adjustment_probability(samples,
  treatment, outcome, adjustment)` is the posterior probability that a set satisfies the
  adjustment criterion. `average_effect(samples, data, treatment, outcome)` groups the
  DAGs by the adjustment set identification picks, estimates once per set (the family
  default, seeded; `strategy=`, `learner=`, `rule="minimal"`) and returns an
  `AveragedEffect`: the `AdjustmentGroup`s (set, mass with its Monte Carlo error, DAG
  count, estimate, standard error), the mixture mean with its Monte Carlo error from the
  masses, and its variance by the law of
  total variance, and the zero-effect (no directed path), unidentified (a variable the
  data lacks makes every set invalid) and truncated (beyond `max_sets`) masses, the last
  two left out of the mixture and reported.
- The `BGe` score (`score="bge"`): the linear-Gaussian marginal likelihood of Geiger and
  Heckerman with the Kuipers, Moffa and Heckerman correction and an empirical-Bayes prior
  mean (the sample mean; BiDAG uses zero, which agrees on centred data), decomposable and
  score-equivalent, for `graph_posterior` and the score-based searches.
  `Score.marginal_likelihood` says whether a score is one (BDeu, BGe).

### Changed
- A decode with knowledge, hard-only knowledge included, now audits it: one extra
  knowledge-free decode, which can cost more than the constrained one, and a
  `KnowledgeConflictWarning` on strong conflicts. Code that turns warnings into
  errors will see it; pass `audit=False` or filter the category.
- The decoder aliases `"expected_count"` and `"soft_f1"` now resolve to the new
  `expected_count_v2` and `soft_f1_v2`, which take their edge budget from the pair
  states with the hard knowledge applied (a forbidden direction holds no mass, a
  required one all of it), so a soft statement near 1 gives the hard result's budget.
  `expected_count_v1` and `soft_f1_v1` keep their 0.2 results; without knowledge the
  two versions agree.
- `Glasso.fit` validates the knowledge first and raises on a contradiction (a
  confounded pair under an adjacency ban or outside a `restrict_to` support, for
  example) instead of dropping part of it.
- The discovery searches (`PC`, `FCI`, `RFCI`, `GES`, `BOSS`, `GRaSP`, `HillClimb`,
  `AStar`, `MMPC`, `DirectLiNGAM`, `Glasso`, `PCMCI`) and `bootstrap` raise
  `ValueError` on soft statements instead of dropping them; pass `knowledge.hard()`.
- `forbid_adjacency(a, b)` also records the pair in a new `nonadjacent` field: an
  adjacency ban rules out a bidirected edge too, while `forbid(a, b)` with
  `forbid(b, a)` no longer does. So `confound(a, b)` with the two directed bans means
  "confounded, and neither causes the other": it is not a conflict, the decoders
  return `a <-> b`, FCI and RFCI keep the pair adjacent instead of pruning it, MMPC
  keeps it in its skeleton and glasso no longer zeroes it. `forbid_adjacency(a, b)`
  and the two directed bans no longer compare equal. The repr is unchanged. Every
  `nonadjacent` pair also forbids both directed edges, however the object is built.
- A `DomainKnowledge` built directly with both directions of a pair in `forbidden`
  (the constructor or `dataclasses.replace`) no longer bans a bidirected edge on
  that pair; use `forbid_adjacency` or `nonadjacent=` for that.
- `DomainKnowledge.apply_to` keeps a bidirected edge on a pair forbidden in both
  directions; only an adjacency ban removes every mark.
- `DomainKnowledge.conflicts()` reports a confounded pair under an adjacency ban.
  `violations()` reads a node in `graph.latent` as a hidden cause: a latent node with
  two or more observed children joins each pair of them as a bidirected edge would
  (for `confound`, adjacency bans and a skeleton restriction), and edges at a latent
  node count against neither a root nor a sink.
- `ergodic.welt.discover(data, model, knowledge=k)` without `decoder=` decodes with
  greedy at the model's native threshold (greedy's default rule when the model
  records none), as `decoder="greedy"` does, instead of raising. Confounded knowledge
  is no longer refused before the paid run; knowledge naming unknown variables or
  contradicting itself still is. Without knowledge and without a decoder nothing
  changes.
- Pickles of `DomainKnowledge` from 0.2.x load, with no soft statements. A pair the
  pickle forbids in both directions becomes an adjacency ban (`nonadjacent`), its
  0.2 meaning, so it still rules out a bidirected edge.
- The `welt` extra requires `welt-client>=0.14`, the release that serves Chronos-2
  forecasting (DEC-085): `WeltForecaster` submits through
  `welt.Client.submit_forecast` and waits with the SDK's `Job.result`.

## [0.2.1] - 2026-10-15

### Fixed
- `WeltLearner` fits a slot with no feature columns locally. An empty adjustment set
  leaves nothing for a model to learn, so the nuisance is the training mean (or the
  class frequencies for classification): no upload, no paid fit, no charge to
  `max_fits`, and `predictor_id_` is `None`. In 0.2.0 it raised, so the "Discover in
  Welt, estimate in ergodic" guide stopped whenever Welt's Arrow found no confounders
  for the treatment, as it does on the guide's own data today. With no adjustment
  variables, DML is then the cross-fitted regression slope of the outcome on the
  treatment, identical to DML with ergodic's `ols` learner.
- `ArcStats` treats two NaN gaps (START and END arcs) as equal and hashes them alike,
  so `process_map(log) == process_map(log)` holds on Python 3.13 and 3.14, where
  dataclass equality compares fields one by one.

## [0.2.0] - 2026-10-15

Released together with welt-client 0.13.

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
  job).
- `ergodic.EdgeProbabilities` (also `ergodic.discovery.EdgeProbabilities`): an
  immutable matrix of directed-edge probabilities over named variables, source rows
  and target columns, with a zero diagonal and a read-only copy of the values. `kind`
  is `"independent"` (per-edge marginals, as AVICI and CDFM return), `"pair_state"`
  (three exclusive states per pair, `P[i, j] + P[j, i] <= 1` enforced) or
  `"frequency"` (bootstrap shares); `calibrated` defaults to `False` and `provenance`
  is a read-only mapping. `to_pair_states()` conditions independent marginals on "not
  both directions", `existence()` gives skeleton probabilities, `threshold(t)` a
  diagnostic `mixed` graph that may be cyclic, `to_frame()`/`from_frame()` a
  dataframe round trip, and `from_bootstrap()` reads a `BootstrapResult`. It is not
  a graph posterior.
- `ergodic.decode(probabilities, method="greedy", *, threshold=None, knowledge=None,
  samples=1, seed=None, max_nodes=20)`: explicit, versioned decoders from edge
  probabilities to a DAG, ported from Ergodic's research library: `greedy_v1`
  (default; optional strict `threshold`), `exact_map_v1` (A*, capped at `max_nodes`),
  `expected_count_v1`, `soft_f1_v1`, and the seeded samplers `pair_sampler_v1` and
  `perturb_and_map_v1` (candidate DAGs plus a medoid consensus; a seed is required).
  It returns a `DecodeResult` (a `DiscoveryResult`, `method="decode:<id>"`) labelled
  `decoded` with the decoder, parameters, seed, rejected cycles, objective and source
  provenance, and the note "not a graph posterior; orientation not certified".
  `DomainKnowledge` constrains decoding (forbidden edges, tiers, roots, sinks,
  `restrict_to`, required edges first); order statements and tiers are enforced as
  one time order consistent with every order statement and every edge, which is
  stricter than `DomainKnowledge.violations` for separate `order()` pairs.
  `confound` is refused.
  `ergodic.discovery.Decoder` holds a decoder and its settings.
- `ergodic.welt.discover(..., decoder=...)` and
  `discover(data, method="welt:<model>", decoder=...)`: opt-in local decoding. The
  run then accepts `avici` and `cdfm`, reads the native scores as
  `EdgeProbabilities(kind="independent")` with the Welt provenance, and returns a
  `DecodeResult` whose graph is the decoded DAG, with `edge_probabilities`,
  `welt_native_edges` and the decode labels in `diagnostics`. Greedy defaults to the
  model's native threshold (0.5 for AVICI, the recorded adaptive threshold for CDFM).
  With a decoder, `knowledge=` constrains the local decoding; it and every decoder
  setting, `max_nodes` against the variable count included, are checked before
  anything is uploaded. Any failure after the paid run raises `WeltDecodeError`
  with the result and its scores attached. Without `decoder=` nothing changes:
  non-DAG models are still refused before paying.
- A new documentation site at https://docs.ergodic.dev, one page per task. Each guide
  runs one complete program on bundled data and shows its recorded output; the
  docs check reruns every program and compares, and a program that needs a Welt
  key is tested against a fake Welt service. The site is versioned, searchable,
  and publishes each page as Markdown and an `llms.txt` index.
- Three learn notebooks: structural causal models (interventions,
  counterfactuals and sensitivity), hidden regimes (recover, certify and
  calibrate), and causal forests with policy trees.
- Every public function, class and member has a docstring, and the reference
  build fails when one is missing.
- `PolicyTree.evaluate(held_out)` scores a learned rule on doubly robust scores
  from units it was not learned on and returns a `PolicyValue` (`value`,
  `std_error`, `interval()`, `treat_all_value`, `regret()` and `regret_std_error`),
  the honest out-of-sample value; the tree's own `value` is measured on the scores
  it was chosen to maximize, so it is optimistic and its docstrings now say so.
- `SCM(cores=)` and `PyMCLearner(cores=)` (default 1) cap the processes PyMC may
  use for the chains; every SCM fit (singleton, latent district, sensitivity
  refit, process-pool worker) honours it. PyMC otherwise ran up to four chains in
  parallel processes. The draws do not depend on it.
- `MediationEstimate.mediated_interaction`, `total - nde - nie`.
- `LatentResult.rhat_max_unaligned`, the raw split R-hat (see Changed).
- `ergodic.inference.learners.ConvergenceWarning`, which `LogisticRegression` emits
  when it stops at `max_iter` before converging. It is ergodic's own class, so a
  filter on scikit-learn's `ConvergenceWarning` does not catch it.
- `MixedGraph.from_networkx(graph, kind=...)`, like the module-level function.
- `SCMResult.kind="predictive"` for posterior-predictive draws, and
  `CausalForest.cate()` records `diagnostics["ij_zero_se_share"]` (see Fixed).

### Changed
- `DiscoveryResult.diagnostics` is typed `Mapping[str, Any]` so a Welt run can
  carry its provenance. Learner cloning honors a learner's own
  `__sklearn_clone__` override; scikit-learn estimators keep the `get_params`
  path.
- A weighted nuisance slot (the R-learner's `final`) refuses, before any fit, a
  learner that declares `supports_sample_weight = False`, as `WeltLearner` does.
  `discover(..., method="welt:<model>")` refuses `alpha` and local-only keyword
  arguments before any request.
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
- `LatentResult.rhat_max` is now the split R-hat after label alignment, over the
  sign-corrected delta draws grouped by chain, and the R-hat warning reads it. It
  was the raw R-hat, so chains sitting in mirror modes of a good fit reported 1.83
  (the spread arm of the golden worlds; aligned, 1.01); that number is now
  `rhat_max_unaligned`.
- `MixedGraph` neighbourhood lookups (`parents`, `children`, `neighbors`,
  `spouses`, `ancestors`, `descendants`, `markov_blanket`) and `Hierarchy` lookups
  (`children_of`, `parent_of`, `is_leaf`, `leaf_descendants`) raise `ValueError` on
  a name that is not a node instead of answering with an empty set (or, for
  `is_leaf`, `True`), as identification and separation already did.
  `get_edge`/`has_edge` still answer `None`/`False` for an absent edge, and
  `HierarchicalTSData.aggregate` on an unknown node raises this `ValueError` rather
  than a `KeyError`. KNN used before `fit` raises `RuntimeError`, and `set_params`
  on the native KNN, OLS and logistic learners refuses unknown names, as sklearn
  does.
- The documentation moved to https://docs.ergodic.dev (Graphite builder); the MkDocs
  site is retired.

### Fixed
- `SCM.sample`, `SCM.interventional` and `SCM.ate` read the fitted posterior on
  every supported stack. With PyMC 5 and arviz 0.x (the
  versions Python 3.11 resolves), 0.1.0 handed `sample_posterior_predictive` a
  posterior it could not read, so every SCM parameter was silently drawn from its
  prior. The merged posterior is now a plain `xarray.Dataset`, which PyMC 5 and 6
  read the same way. If you used `ergodic.scm` on Python 3.11 (or with arviz below
  1.0), rerun those results.
- `estimate_mediation`: the total effect is now computed directly as
  `E[Y(1, M(1))] - E[Y(0, M(0))]`. It used to be the pure NDE plus the pure NIE,
  which drops the mediated interaction whenever the treatment interacts with the
  mediator (always possible, since the outcome model is saturated in treatment and
  mediator). On `load_berkeley` (gender, department, admission) the total was
  -0.072 with a proportion mediated of 1.99; it is now the -0.142 raw gap and the
  proportion 1.01. `nde` and `nie` are unchanged (the pure effects), so under an
  interaction they no longer sum to the total; `proportion_mediated` stays
  `nie / total`, Pearl's share explained by mediation, and the docs now name the
  other convention, `1 - nde / total` (VanderWeele's total NIE over TE, as CMAverse
  reports; 1.50 on Berkeley). The `Design.estimand` no longer repeats the `NDE =`/`NIE =`
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
  causal layer lands", "E1", "v1", "lands in D2", "in this phase", "Phase-2",
  "Track-D", "deferred"): the neural and foundation CATE method names, `ITEResult.explain`,
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
- The best-linear-projection docstrings no longer say the intercept is the ATE:
  the covariates enter uncentred, so the intercept is the projected effect at
  covariates all zero (the ATE only when they are centred). Outputs unchanged.
- `KNNRegressor` and `KNNClassifier` honour `sample_weight` (a weighted mean
  over the neighbours). They accepted it and dropped it, so the R-learner's
  weighted final stage ran unweighted with a KNN learner. `PyMCLearner` now
  declares `supports_sample_weight = False`, so the R-learner refuses it before
  any fit, and its `fit` raises on weights instead of ignoring them.
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
- Process trees splice a nested sequence into its parent, so
  `seq(a, seq(b, c))` is `seq(a, b, c)`. `discover_model(noise=...)` could return
  an unflattened `seq(..., seq(xor(TAU, 'standard'), xor(TAU, 'expedite')), ...)`
  that did not compare equal to the same model written by hand, although the
  tree docstring promised a canonical form. Nested `xor` and `par` nodes are
  kept as built (`decision_points` reads each `xor` as its own decision).

- `CausalForest.cate()` records the share of per-unit standard errors clipped to
  zero (`diagnostics["ij_zero_se_share"]`) and warns when it exceeds 5%. With too
  few trees for the rows, the infinitesimal-jackknife Monte-Carlo correction
  exceeds the raw variance and the band has zero width; on a 4,000-row world 400
  trees gave 25% zero-width bands and 68% coverage of a nominal 90%, 1,600 trees
  1% and 87%. The estimator is unchanged; its docstring states these
  measurements, names the variance correctly (the infinitesimal jackknife with the
  Wager-Hastie-Efron Monte-Carlo correction, not a bootstrap of little bags) and
  recommends at least as many trees as rows.
- The heterogeneity layer (`explain()`: BLP, CLAN; `policy_tree`;
  `PolicyTree.evaluate`) named the doubly robust design by variable while a
  categorical covariate with three or more levels has one column per non-first
  level, so names and columns drifted apart: a split reported on `Z` was really an
  indicator of the categorical, and BLP returned four coefficients for three names.
  The scores now name every column (`"C==level"` for such a categorical; a
  continuous or two-level covariate keeps its name), `policy_tree(result, "C")`
  selects the categorical's whole block, `DRScores` refuses a design whose width
  does not match its names, and `evaluate()` refuses held-out scores that encode
  the categorical with other levels. The forest's `variable_importance()` uses the
  same column names.

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

[Unreleased]: https://github.com/ergodic-ai/ergodic/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/ergodic-ai/ergodic/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/ergodic-ai/ergodic/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ergodic-ai/ergodic/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/ergodic-ai/ergodic/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ergodic-ai/ergodic/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ergodic-ai/ergodic/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ergodic-ai/ergodic/releases/tag/v0.1.0
