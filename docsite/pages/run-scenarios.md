# Run scenarios

Pin the future of a driver you control or already know, such as a planned price, and
compare the forecasts it leads to.

!!! need "You'll need"
    - ergodic with the forecasting extra: `pip install "ergodic[forecast]"`. It adds
      PyMC only. Optionally, `pip install jax numpyro` switches sampling to the
      faster NumPyro backend, which recorded the output on this page.
    - A window graph from the driver to the series you forecast.
      [Forecast with causal drivers](forecast-with-drivers.md) shows how to write one.

## Run the program

The data is 48 simulated months. Sales grow by 2 a month and fall by 12 for each unit
of price. You set the price, so its future is a plan, not a forecast.

```python file="price_scenarios.py"
import numpy as np
import pandas as pd
from ergodic import forecast, timeseries
from ergodic.forecast import schedule, window_graph

# 1. Simulate 48 months. Sales grow, and fall by 12 for each unit of price.
rng = np.random.default_rng(0)
T = 48
month = pd.date_range("2022-01-31", periods=T, freq="ME")
price = 10 + 0.05 * np.arange(T) + rng.normal(0, 0.3, T)
sales = 300 + 2.0 * np.arange(T) - 12.0 * price + rng.normal(0, 3, T)
frame = pd.DataFrame({"month": month, "price": price, "sales": sales})
data = timeseries(frame, time="month", freq="ME")

# 2. Price acts on sales in the same month. You set the price, so you pin its future.
graph = window_graph([("price", "sales", 0)])
print(schedule(data, graph, scenario={"price": "hold"}))

# 3. Forecast under two price plans: hold the last price, or raise it by 10%.
plans = {"hold": "hold", "raise 10%": round(1.1 * price[-1], 2)}
results = {}
for name, plan in plans.items():
    results[name] = forecast(
        data, 6, graph=graph, scenario={"price": plan}, draws=400, warmup=400, seed=0
    )
    print(f"{name:10} price {results[name].median('price')[0]:.2f}", end="  ")
    print("sales", results[name].median("sales").round(1))

# 4. Compare the plans draw by draw: the fit is the same, only the price differs.
change = results["raise 10%"].draws("sales") - results["hold"].draws("sales")
low, high = np.percentile(change, [5, 95], axis=0)
print(f"change in month 1: {change.mean(axis=0)[0]:.1f} (90% band {low[0]:.1f} to {high[0]:.1f})")
```

```output from="price_scenarios.py" tol="0.5" rtol="0.05"
TrainingSchedule: 2 stages, 1 by mcmc
  Stage 0 [resolved ]: price
  Stage 1 [mcmc     ]: sales
hold       price 12.94  sales [242.3 243.1 248.4 245.2 248.2 252.6]
raise 10%  price 14.23  sales [227.7 228.9 233.9 231.1 233.9 238.5]
change in month 1: -14.3 (90% band -19.6 to -7.4)
```

- **Step 2.** The schedule marks price as `resolved`: it is read from the plan, not
  sampled. Only sales is fitted.
- **Step 3.** Holding the price keeps it at 12.94 for six months. The raise sets it
  to 14.23, and every month's sales median drops by about 14.
- **Step 4.** The two runs use the same history and seed, so they share one posterior.
  The difference is the fitted price response alone: about −14 in month 1, with a
  90% band of −19.6 to −7.4. The true change is 1.29 × −12 ≈ −15.5.

## Write a plan

`scenario=` maps a driver to its future. Each value can be:

- `"hold"`: repeat the last observed value.
- a number: hold that value for the whole horizon.
- an array of length `horizon`: one value per step, for example a planned ramp.

A driver with a plan is not modelled; its path in the result is the plan itself. To
fix a plan once, give the driver `ScenarioMechanism(values=...)` through
`mechanisms=` instead.

Without `scenario=`, a driver with its own column is forecast like any other series,
and its uncertainty flows into its children.

## Compare plans fairly

Use the same `seed`, data and sampler settings for every plan. Then the fitted
posterior is identical, and subtracting draws, as step 4 does, isolates the effect of
the plan:

```python
change = results["raise 10%"].draws("sales") - results["hold"].draws("sales")
low, high = np.percentile(change, [5, 95], axis=0)
```

Each plan refits the model, so a run costs one sampler pass per plan.

!!! caution "A scenario is not an intervention"
    A scenario trusts the fitted regression of sales on price. The difference
    between plans is the causal effect of price only if the graph is right and
    nothing else drives both. If you set prices in response to demand, the fitted
    coefficient mixes the two, and so does every scenario built on it.

<details markdown="1"><summary>What does a scenario refuse?</summary>

- A key that isn't a column of the data.
- A node the graph says is driven by another series. Pinning it would be an
  intervention, which `forecast` doesn't support. Forecast it from its parents, or
  drop the edges into it if it really is set from outside.
- An array whose length isn't the horizon, or a string other than `"hold"`.
- A `ScenarioMechanism` without values and no `scenario=` entry.
- `fit="independent"` with a scenario: nothing would read the pinned path.

</details>

## Next steps

<div class="cards" markdown="1">

- [Forecast with causal drivers](forecast-with-drivers.md): forecast the driver instead of pinning it.
- [Simulate interventions and counterfactuals](interventions-and-counterfactuals.md): interventions in a structural causal model.
- [Python reference](reference-forecast.md): `forecast`, `ScenarioMechanism` and every option.

</div>
