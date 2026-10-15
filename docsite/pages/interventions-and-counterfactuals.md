# Simulate interventions and counterfactuals

Use a fitted SCM to answer three kinds of question: what the data shows, what happens
if you set a variable, and what would have happened to one unit under a different
choice.

!!! need "You'll need"
    - ergodic with the SCM extra: `pip install "ergodic[scm]"`.
    - A fitted `SCM`. [Fit a structural causal model](fit-an-scm.md) shows how.

## Run the program

The data is 500 simulated weeks. Demand raises both ad spend and sales, and each extra
unit of ads adds 2.0 to sales. Demand is observed, so the SCM can separate the two.

```python file="what_if.py"
import numpy as np
import pandas as pd
from ergodic import SCM, dag, tabular

# 1. Simulate 500 weeks. Demand raises both ad spend and sales.
rng = np.random.default_rng(1)
n = 500
demand = rng.normal(size=n)
ads = 0.5 * demand + rng.normal(size=n)
sales = 2.0 * ads + 3.0 * demand + rng.normal(size=n)
weeks = pd.DataFrame({"demand": demand, "ads": ads, "sales": sales})

# 2. Fit the SCM over the true graph.
graph = dag(["demand -> ads", "demand -> sales", "ads -> sales"])
scm = SCM(graph, tabular(weeks), draws=500, tune=500, chains=1, random_state=0).fit()

# 3. Observe: the distribution of sales as the data shows it.
print(scm.marginal("sales"))

# 4. Intervene: set ad spend to 1 for every week, then to 0.
high = scm.interventional("sales", do={"ads": 1.0})
low = scm.do({"ads": 0.0}).interventional("sales")
print(f"E[sales | do(ads=1)] - E[sales | do(ads=0)] = {high.value - low.value:.3f}")

# 5. Ask a counterfactual: what would week 0 have sold with ads at 1?
week0 = weeks.head(1)
print(week0.round(3).to_string(index=False))
what_if = scm.counterfactual(tabular(week0), do={"ads": 1.0}, target="sales")
print(what_if)

# 6. Individual effects: one more unit of ads, from each week's own level.
ite = scm.ite(tabular(weeks.head(3)), "ads", "sales", shift=1.0)
print(ite.framing, ite.values.round(3))
```

```output from="what_if.py" tol="0.05" rtol="0.05"
SCMResult(sample, identified: sales=-0.3091, sd=4.45, 95% CrI [-9.02, 8.42])
E[sales | do(ads=1)] - E[sales | do(ads=0)] = 2.088
 demand    ads  sales
  0.346 -1.198 -1.164
SCMResult(counterfactual, identified: sales=3.425, sd=0.142, 95% CrI [3.15, 3.69])
shift=1.0 [2.088 2.088 2.088]
```

- **Step 3.** `marginal` draws sales from the fitted model with nothing changed. The
  interval is the spread of weekly sales.
- **Step 4.** Setting ads cuts the link from demand to ads. The difference in mean
  sales is about 2.09, close to the true 2.0.
- **Step 5.** Week 0 had ads at −1.198 and sold −1.164. With ads at 1, it would have
  sold about 3.43: its own sales plus 2.09 × (1 − (−1.198)).
- **Step 6.** One more unit of ads adds the same 2.09 to every week, because each
  equation is linear.

## Observe, intervene, imagine

The three questions use the same fitted model in different ways:

| Question | Method | What it does |
|---|---|---|
| What does the data show? | `marginal(node)`, `sample(n)` | Draws from the model as fitted. |
| What if I set X? | `interventional(target, do=...)`, `do(...)` | Replaces X's equation with a constant, then draws. |
| What would this unit have done? | `counterfactual(units, do, target)`, `ite(...)` | Recovers each unit's own noise, then replays it with X changed. |

`do` returns a view you can chain, so these two lines ask the same question:

```python
high = scm.interventional("sales", do={"ads": 1.0})
low = scm.do({"ads": 0.0}).interventional("sales")
```

An intervention is not the same as filtering the data. Weeks with high ad spend also
had high demand, so their sales overstate what ads alone do. `do` removes that link.

`interventional` returns draws of individual outcomes under the intervention. Its
`.value` is their mean, but its interval and `std_error` describe the spread of
single weeks' sales, not your uncertainty about the mean. For the mean contrast with a
credible interval, use `scm.ate(treatment, outcome)`, as in
[Fit a structural causal model](fit-an-scm.md).

## Read a counterfactual

`counterfactual` needs every observed node's value for each unit: pass a dataset or a
mapping from node name to a column. It works backwards from those values to each
unit's noise terms, then recomputes the target with `do` applied.

The result is an `SCMResult`. By default its draws carry the uncertainty in the
fitted equations. Pass `use_draws=False` to get the point counterfactual for each
unit instead.

## Compute individual effects

`ite(units, treatment, outcome)` returns one effect per unit, as an `ITEResult`. Read
`.values` for the array and `.factual` for each unit's observed outcome. Choose one of
two contrasts:

- `levels=(low, high)`: outcome with the treatment set to `high`, minus outcome with
  it set to `low`. The default is `(0.0, 1.0)`.
- `shift=delta`: outcome with the treatment moved `delta` from each unit's own value,
  minus the observed outcome. It needs a continuous treatment and outcome.

With linear mechanisms, every unit gets the same effect: the fitted coefficient.
Effects differ between units when the outcome is binary. Each unit's effect is then
−1, 0 or 1: whether the change flips that unit's outcome.

!!! note "Individual effects are points"
    `ite` returns point effects without intervals. For one unit's interval, call
    `counterfactual` once per treatment level on that unit and subtract the draws.

<details markdown="1"><summary>What can each query take?</summary>

- A binary or categorical do-target takes its integer category code, for example
  `do={"churned": 1}`. A fractional or out-of-range code raises.
- A binary or categorical target returns a `CategoricalResult`. Read
  `.probabilities`, `.categories` and `.mode()`; for a counterfactual,
  `.per_unit()` gives each unit's category code.
- `ite` refuses an outcome with more than two categories. Use `counterfactual` to read
  its per-unit distribution.
- A latent node can't be a do-target or a query target.

</details>

## Next steps

<div class="cards" markdown="1">

- [Fit a structural causal model](fit-an-scm.md): mechanisms and the effect checks behind `ate`.
- [Estimate heterogeneous effects](heterogeneous-effects.md): conditional average effects with honest intervals.
- [Python reference](reference-scm.md): every query method of `SCM`.

</div>
