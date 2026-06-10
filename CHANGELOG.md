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
