# Refute an estimate

Re-run an effect estimate under three perturbations and check that it behaves as a
real effect should. The example stress-tests the effect of smoking on birthweight.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A graph, data and an estimate, as in
      [Estimate an average effect](estimate-average-effect.md).

## Run the program

The program repeats the birthweight estimate, then runs each refutation 100 times.
Each run is a full re-estimate, so the whole program takes a few seconds.

```python file="refute_effect.py"
from ergodic import dag, estimate_effect, refute_effect
from ergodic.datasets import load_birthweight

# 1. Load the births and draw the graph: every confounder affects both columns.
births = load_birthweight()
smoking, weight = births.treatment, births.outcome
edges = [f"{c} -> {smoking}" for c in births.confounders]
edges += [f"{c} -> {weight}" for c in births.confounders]
graph = dag([*edges, f"{smoking} -> {weight}"])

# 2. Estimate the effect once.
estimate = estimate_effect(
    graph, births.data, smoking, weight, treated=births.treated, random_state=0
)
print(estimate)

# 3. Run the three refutations. Pass the same options you used for the estimate.
for method in ("placebo_treatment", "random_common_cause", "data_subset"):
    check = refute_effect(
        graph,
        births.data,
        smoking,
        weight,
        method=method,
        treated=births.treated,
        estimand_value=estimate.value,
        random_state=0,
    )
    print(check)
    print(f"  spread of the perturbed estimates: {check.refuted_std:.1f} g")

# 4. Refute an estimate that leaves out every confounder.
naive = dag([f"{smoking} -> {weight}"])
check = refute_effect(
    naive, births.data, smoking, weight, treated=births.treated, random_state=0
)
print(check)
```

```output from="refute_effect.py"
EffectEstimate(aipw, analytic: ate=-219.5, se=26.3, 95% CI [-271, -168], n=4642)
Refutation(placebo_treatment: estimate=-219.5, under perturbation=4.747, p=0.0099, passed, n=100)
  spread of the perturbed estimates: 21.5 g
Refutation(random_common_cause: estimate=-219.5, under perturbation=-219.3, passed, n=100)
  spread of the perturbed estimates: 1.0 g
Refutation(data_subset: estimate=-219.5, under perturbation=-218.7, passed, n=100)
  spread of the perturbed estimates: 14.2 g
Refutation(placebo_treatment: estimate=-275.3, under perturbation=4.526, p=0.0099, passed, n=100)
```

- **Placebo treatment.** With the smoking column shuffled, the effect drops from
  −220 g to a few grams. In none of the 100 shuffles was the placebo effect as large
  in size as the real one, so `p` is the smallest value 100 runs can give: 1/101.
- **Random common cause.** Adding an independent random confounder to the graph
  leaves the estimate where it was, within a gram.
- **Data subset.** Re-estimating on random 80% subsets gives about the same effect,
  with a spread of about 14 g.
- **Step 4.** The naive graph ignores all 17 confounders, so its estimate is the raw
  difference of −275 g. It is confounded, yet it passes the placebo check too.

## What each check tests

`method=` picks the check. Each one re-runs the same identify-then-estimate path
as `estimate_effect`, so pass the same `strategy=`, `learner=` and `treated=` you
used for the estimate.

| `method` | Perturbation | `passed` when |
| --- | --- | --- |
| `"placebo_treatment"` (default) | Shuffle the treatment column. | The mean placebo effect is at most 20% of the real effect's size. |
| `"random_common_cause"` | Add an independent random column as a parent of both treatment and outcome. | The mean effect stays within 10% of the original. |
| `"data_subset"` | Re-estimate on a random `subset_fraction` of the rows (default 0.8). | The original is within two standard deviations of the subset estimates. |

The result is a `Refutation`. `estimand_value` is the effect on the real data;
`refuted_value` and `refuted_std` are the mean and spread of the perturbed runs.
`p_value` is set for the placebo check only: one plus the number of runs whose
effect is at least as large in size as the real one, over one plus the number of
runs. `diagnostics` holds the smallest and largest perturbed effect.

!!! caution "Passing is not proof"
    These checks catch a broken pipeline: an estimator that finds an effect in
    noise, or one that a single irrelevant column or a few rows can move. They
    cannot detect a confounder missing from the graph. Step 4 shows a biased
    estimate passing.

`passed` is a convenience flag with fixed thresholds. Read the numbers too.

<details markdown="1">
<summary>Run time and repeatability</summary>

- `n_simulations=` (default 100, at least 2) sets the number of runs. A flexible or
  cross-fit estimator refits on every run, so lower it when one estimate is slow.
- `estimand_value=` skips re-estimating on the real data when you already have the
  estimate.
- `random_state=` seeds both the perturbations and the estimator, so the same call
  gives the same numbers.
- `random_common_cause` adds a column named `_random_common_cause`, and raises if
  your data already has one.

</details>

## Next steps

<div class="cards" markdown="1">

- [Estimate an average effect](estimate-average-effect.md): the estimate this page refutes.
- [Estimate heterogeneous effects](heterogeneous-effects.md): find who gains more or less from the treatment.
- [Python reference](reference-inference.md): every option of `refute_effect`.

</div>
