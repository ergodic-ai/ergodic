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
EffectEstimate(aipw, analytic: ate=-222.3, se=25.8, 95% CI [-273, -172], n=4642)
95% interval: -273 g to -172 g
```

- Babies of smokers weigh 275 g less on average. Part of that gap comes from the
  confounders, not from smoking.
- Adjusted for the confounders, smoking lowers birthweight by about 222 g. The 95%
  interval runs from 273 g to 172 g lower.
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

## Next steps

<div class="cards" markdown="1">

- [Refute an estimate](refute-an-estimate.md): stress-test this estimate with placebo and subset checks.
- [Estimate heterogeneous effects](heterogeneous-effects.md): find who gains more or less from the treatment.
- [Python reference](reference-inference.md): every option of `estimate_effect`.

</div>
