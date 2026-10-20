# Estimate heterogeneous effects

Estimate the effect of a treatment for each unit, then test whether the effect
really varies and describe who gains most. The example uses a simulation, so you can
compare every estimate with the truth.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A binary treatment, and a graph in which the covariates that change the
      effect are causes of the outcome.

## Run the program

The simulated treatment helps patients with a high `need` score more. Older
patients are treated more often, so `age` confounds the comparison.

```python file="heterogeneous_effects.py"
import numpy as np
import pandas as pd
from ergodic import cate, dag, tabular

# 1. Simulate 2,000 patients. The true effect of the drug is 1 + 2 * need.
rng = np.random.default_rng(0)
n = 2000
age = rng.normal(0, 1, n)
need = rng.normal(0, 1, n)
drug = rng.binomial(1, 1 / (1 + np.exp(-age)))
true_effect = 1 + 2 * need
score = age + need + true_effect * drug + rng.normal(0, 1, n)
frame = pd.DataFrame({"age": age, "need": need, "drug": drug, "score": score})
data = tabular(frame, types={"drug": "categorical"})

# 2. Draw the graph: age confounds; need only changes the outcome.
graph = dag(["age -> drug", "age -> score", "need -> score", "drug -> score"])

# 3. Estimate one effect per patient, with an interval for each.
effects = cate(graph, data, "drug", "score", random_state=0)
print(effects.source, effects.kind, effects.covariates)
print(effects.to_frame().head(3).round(2))
error = np.sqrt(np.mean((effects.values - true_effect) ** 2))
print(f"mean effect {effects.values.mean():.2f}, error against the truth {error:.2f}")

# 4. Test for heterogeneity, then describe it.
summary = effects.explain()
print(f"heterogeneity p-value: {summary.calibration_test().heterogeneity_pvalue:.1e}")
for name, row in summary.blp().frame().items():
    print(f"{name:12} {row['coef']:5.2f}  [{row['ci_low']:.2f}, {row['ci_high']:.2f}]")
print("effect by predicted fifth:", summary.gates().group_effect.round(2))
```

```output from="heterogeneous_effects.py"
dr_learner analytic ('age', 'need')
      effect  std_error  ci_low  ci_high
unit
0       1.91       0.07    1.80     2.02
1       0.06       0.07   -0.06     0.17
2      -0.57       0.08   -0.71    -0.43
mean effect 1.06, error against the truth 0.12
heterogeneity p-value: 2.8e-193
(intercept)   1.06  [0.97, 1.16]
age           0.10  [-0.03, 0.23]
need          1.98  [1.87, 2.09]
effect by predicted fifth: [-1.63 -0.07  0.95  2.12  3.92]
```

- **Step 3.** `cate` fits the doubly robust DR-learner by default (Kennedy, 2023). Its covariates
  are the adjustment set the graph implies: `age` and `need`. Each row of
  `to_frame()` has the effect, its standard error and a 90% interval.
- The estimated effects are close to the truth: the root-mean-square error is
  about 0.12 on true effects that range from −5.0 to 7.5.
- **Calibration test.** A tiny p-value rejects "no heterogeneity".
- **Best linear projection.** The effect rises by about 2 per unit of `need`, the
  true slope. The `age` slope is small, and its interval includes 0. The
  intercept, 1.06, is the effect where both covariates are 0.
- **GATES.** Sorted into fifths by predicted effect, the measured effect climbs
  from below 0 to almost 4. The lowest fifth is harmed.

## Read the summaries

`effects.explain()` returns a `Heterogeneity` record. Its methods summarize the
doubly robust scores, not the per-unit points, so their intervals are valid even
when the per-unit model is wrong (Semenova and Chernozhukov, 2021; Chernozhukov,
Demirer, Duflo and Fernández-Val, 2025; Yadlowsky and others, 2024).

| Method | Question it answers |
| --- | --- |
| `calibration_test()` | Is there heterogeneity, and is it calibrated? `heterogeneity_pvalue` is one-sided. |
| `blp(level=0.9)` | Which covariates go with a bigger effect? The intercept is the effect where every covariate is 0; it equals the average effect only when the covariates are centred. |
| `gates(n_groups=5)` | Does the measured effect rise across groups of predicted effect? `difference` is top minus bottom. |
| `clan(n_groups=5)` | How do the most and least affected groups differ in their covariates? |
| `rate("autoc")` | Does ranking by predicted effect put the biggest gains first? Also `"qini"`. |
| `policy_tree(depth=2)` | Whom should you treat? See [Target a policy](target-a-policy.md). |

!!! caution "Covariates come from the graph"
    `cate` uses the adjustment set of the identified effect as its covariates. A
    column that changes the effect but is missing from the graph is never used.
    Draw it as a cause of the outcome, as `need` is here.

## Choose a method

`method=` picks the learner. The default `"dr_learner"` is doubly robust. Other
names: `"r_learner"`, `"x_learner"`, `"t_learner"`, `"s_learner"`, `"dml"` and
`"causal_forest"`. `learner=` fills the nuisance models, as in `estimate_effect`.
The same S/T names work with `estimate_effect(strategy=...)`: `t_learner` fits
separate outcome models for each arm, and `s_learner` fits one pooled model.
Passing a contradictory `per_arm=` flag raises an error in either facade.

- The uncertainty depends on the method. The DR-learner with its default linear
  final model has analytic standard errors. `"causal_forest"` (Athey, Tibshirani
  and Wager, 2019) uses the infinitesimal jackknife, which needs many trees: with
  too few, the variance is clipped to zero and some bands have zero width. On a
  4,000-row simulation, 400 trees gave 25% zero-width bands and covered the true
  effect for 68% of units at a nominal 90%; 1,600 trees gave 1% and 87%. Set
  `n_trees` to at least the number of rows when the per-unit intervals
  matter; `cate` warns when more than 5% of the bands have zero width, and
  `effects.diagnostics["ij_zero_se_share"]` records the share. The other learners
  bootstrap: `n_boot=` refits (default 1,000), which can be slow. Pass
  `uncertainty=False` to skip that step.
- With the default linear models, `"s_learner"` and `"dml"` return one constant
  effect for every unit. Give `"s_learner"` a flexible `learner=` to see variation.
- `select_cate(graph, data, treatment, outcome)` fits several learners, ranks them
  by doubly robust risk and returns the best fit and a weighted ensemble.

<details markdown="1">
<summary>Limits</summary>

- `cate` needs a binary treatment and an effect identified by adjustment. Front-door,
  instrumental-variable and general-formula identifications raise an error.
- `explain()` describes the units the model was fit on. To score new rows, fit an
  estimator such as `DRLearner` with `fit(identification, data)`, then call its
  `cate(new_rows)`. That result has no scores, so its `explain()` raises.
- New rows use the covariate types and category vocabulary saved during `fit`.
  Column order and categorical metadata order can change, and a batch can omit
  known category levels. Unknown values, missing required columns, changed
  covariate types and missing covariate values raise an error. Treatment and
  outcome columns are unnecessary when scoring new rows.
- If a fold fails while building the out-of-fold effect proxy, `explain().blp()`
  still projects the cross-fit scores on covariates. GATES, CLAN, RATE and calibration
  refuse to return intervals or p-values and report the ranking failure. They
  never substitute the fitted in-sample effects. `select_cate` also refuses a
  candidate without an out-of-fold proxy. DML supplies constant effects with no
  out-of-fold CATE ranking, so it also refuses these summaries while retaining BLP.
- `effects.ate()` returns the average effect. With the DR-learner it is the mean of
  the doubly robust scores, whose standard error is valid (the per-unit standard
  errors share one fitted model, so they are never combined as if independent); for
  the default linear final model its point equals the mean per-unit effect. A result
  without scores (one scored on new rows) returns the mean as a point, with no
  interval; use `estimate_effect` then.

</details>

## Next steps

<div class="cards" markdown="1">

- [Target a policy](target-a-policy.md): turn per-unit effects into a treatment rule.
- [Estimate an average effect](estimate-average-effect.md): one number, on real data.
- [Python reference](reference-inference.md): every option of `cate` and `select_cate`.

</div>
