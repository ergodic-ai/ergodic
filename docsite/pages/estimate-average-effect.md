# Estimate an average effect

Write down which variables confound a treatment and an outcome, then estimate the
average effect with an interval. The example measures what smoking during pregnancy
does to birthweight.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A list of confounders: variables that affect both the treatment and the outcome.

## Run the program

The bundled birthweight data is the extract of US natality records used by Cattaneo
(2010): 4,642 singleton births in Pennsylvania, with the mother's smoking, the baby's
weight in grams and 17 background variables.

```python file="average_effect.py"
from ergodic import dag, estimate_effect
from ergodic.datasets import load_birthweight

# 1. Load 4,642 births: the mother's smoking, the birthweight and 17 confounders.
births = load_birthweight()
smoking, weight = births.treatment, births.outcome
print(smoking, "->", weight, "with", len(births.confounders), "confounders")

# 2. Draw the graph: each confounder affects both smoking and birthweight.
edges = [f"{c} -> {smoking}" for c in births.confounders]
edges += [f"{c} -> {weight}" for c in births.confounders]
graph = dag([*edges, f"{smoking} -> {weight}"])

# 3. Compare the raw difference in mean birthweight.
rows = births.data.to_pandas()
means = rows.groupby(smoking, observed=True)[weight].mean()
print(f"raw difference: {means['smoker'] - means['nonsmoker']:.1f} g")

# 4. Estimate the average effect, adjusting for the confounders.
estimate = estimate_effect(
    graph, births.data, smoking, weight, treated=births.treated, random_state=0
)
print(estimate)
low, high = estimate.interval()
print(f"95% interval: {low:.0f} g to {high:.0f} g")
```

```output from="average_effect.py"
mbsmoke -> bweight with 17 confounders
raw difference: -275.3 g
EffectEstimate(aipw, analytic: ate=-219.5, se=26.3, 95% CI [-271, -168], n=4642)
95% interval: -271 g to -168 g
```

- Babies of smokers weigh 275 g less on average. Part of that gap comes from the
  confounders, not from smoking.
- Adjusted for the confounders, smoking lowers birthweight by about 220 g. The 95%
  interval runs from 271 g to 168 g lower. The Stata treatment-effects manual's
  doubly robust estimate on the same extract is −231 g (standard error 26 g), well
  inside that interval; `births.target` records it with its source.
- `aipw` is the augmented inverse-probability-weighted estimator. It combines an
  outcome model and a propensity model, and stays consistent if either one is right.

## Draw the graph

The graph is your assumption about the world. Each confounder points at both the
treatment and the outcome:

```python
edges = [f"{c} -> {smoking}" for c in births.confounders]
edges += [f"{c} -> {weight}" for c in births.confounders]
graph = dag([*edges, f"{smoking} -> {weight}"])
```

`estimate_effect` reads the adjustment set from this graph. You never pass the
confounders to it directly.

## Choose the treated level

`treated=` names the level that counts as treatment. Here the `mbsmoke` column holds
`"smoker"` and `"nonsmoker"`, and `births.treated` is `"smoker"`. The estimate is
the mean outcome if everyone were treated minus the mean if no one were.

!!! caution "The estimate is only as good as the graph"
    It assumes no confounder is missing from the list. A variable that affects both
    smoking and birthweight, but is not in the data, biases the result. No
    check on this page can detect such a variable.

By default the propensity model is logistic regression. The outcome model is
ordinary least squares for a continuous outcome, as here, and logistic regression
for a binary one. Pass `learner=` (an alias, a learner, or a mapping from slot to
learner) to use other models.

## Cross-fitting and overlap

AIPW now defaults to `mode="auto"`: frequentist nuisances use held-out predictions
from five shuffled, treatment-stratified folds. Set `n_folds=` and `random_state=`
explicitly for a recorded protocol. If no seed is supplied, AIPW uses seed 0 for
the split. Diagnostics report `n_folds_requested`, the number of nonempty folds
actually used as `n_folds`, and `crossfit_seed`. Both arms need at least two rows;
fold counts must be integers from 2 through the number of rows.

Cross-fitting prevents evaluating nuisance residuals on their own fitting rows.
It does not make every learner safe: valid ATE intervals also need overlap,
suitable nuisance convergence rates and independent observations. An unrestricted
decision-tree propensity model can predict only 0 and 1 even on held-out rows;
the default guard refuses that fit. This remains AIPW, with its counterfactual
outcome regressions and propensity residual corrections, rather than a partially
linear DML model. See [Chernozhukov et al.](https://arxiv.org/abs/1608.00060).

`mode="in_sample"` selects the old parametric fitting path explicitly. Its
influence-score interval needs appropriate parametric or empirical-process
conditions; unrestricted interpolating learners are not qualified. With any
Bayesian nuisance, `auto` preserves the in-sample posterior or posterior-mean
fallback without repeatedly fitting a sampler. `cross_fitted=0` and
`bayesian_in_sample=1` report that choice. Explicit `mode="cross_fitted"` refuses
Bayesian slots because cross-fitted posterior inference is not implemented.

Inspect `estimate.diagnostics`: `propensity_min`, `propensity_max`, and the
`propensity_q01`, `q05`, `median`, `q95` and `q99` entries use raw predictions,
before clipping. `propensity_clipped_fraction` records clipping;
`treated_effective_n`, `control_effective_n`, and each arm's `weight_max` and
`weight_q99` describe the clipped ATE inverse weights. `att_control_effective_n`
and `att_control_weight_max` describe ATT odds weights. For a Bayesian posterior,
these summaries use the posterior mean propensity; `posterior_draw_clipped_fraction`
and `posterior_draw_tail_fraction` separately summarize all propensity draw cells.

AIPW, IPW and stratification accept `overlap="warn"` (default), `"raise"`, or
`"ignore"`, plus `overlap_threshold=0.01` and `clip=0.001`. The default warns when
at least 10% of raw predictions lie outside `[0.01, 0.99]`, or an arm's effective
size falls below 10% of its observed count. It refuses when at least 50% require
clipping. `raise` also refuses moderate warnings; `ignore` explicitly overrides
these diagnostic guards, but invalid probabilities always fail. These thresholds
are diagnostic choices, not a test of population positivity. A result below the
thresholds is not a certificate of valid inference. Clipping cannot restore
missing support. No rows are trimmed; stratification refuses an occupied stratum
with a missing arm instead of dropping it and changing the estimand.

## Instrumental-variable targets

Instrument estimators now default to `target="linear_iv"`, including continuous
treatment and PLIV. This replaces the old automatic `late` label. The numerical
coefficient is unchanged; consumers checking `EffectTarget.LATE` should migrate to
`EffectTarget.LINEAR_IV` when they mean the generic linear IV coefficient.

A complier LATE requires additional assumptions. For one unconditional binary
instrument and a binary treatment, 2SLS or Wald accepts `target="late"` only with
`late_assumptions=True`. This asserts exogeneity, exclusion, relevance,
monotonicity, consistency and no interference; it does not verify them. Adjusted
or multivariate instruments and PLIV retain the generic linear IV target.
See [Angrist, Imbens and Rubin](https://doi.org/10.1080/01621459.1996.10476902).

Scalar results retain `identification`; IV results also state `assumptions`.
Their `first_stage_partial_r2`, `first_stage_f` and degrees of freedom describe
first-stage strength. PLIV computes them on its cross-fitted residuals, marked by
`first_stage_residualized=1`. The classical F statistic is descriptive, assumes a
homoskedastic linear first stage for its usual interpretation, and does not test
exclusion or exogeneity. Values below 10 warn; larger values do not establish
instrument validity. Ordinary normal IV intervals can be unreliable with weak
instruments; weak-IV robust intervals are not implemented here.

## Next steps

<div class="cards" markdown="1">

- [Refute an estimate](refute-an-estimate.md): stress-test this estimate with placebo and subset checks.
- [Average an effect over the graphs](evaluate-a-graph.md#average-an-effect-over-the-graphs): estimate across a posterior over DAGs.
- [Estimate heterogeneous effects](heterogeneous-effects.md): find who gains more or less from the treatment.
- [Python reference](reference-inference.md): every option of `estimate_effect`.

</div>
