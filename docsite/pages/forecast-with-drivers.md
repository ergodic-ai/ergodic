# Forecast with causal drivers

Forecast a series from the series that drive it, with a posterior band at every
step. Then take the forecast apart to see how much each driver contributes.

!!! need "You'll need"
    - ergodic with the forecasting extra: `pip install "ergodic[forecast]"`. It adds
      PyMC only.
    - Optionally, JAX and NumPyro: `pip install jax numpyro`. Sampling then runs on
      the faster NumPyro backend, which recorded the output on this page. Without
      them, PyMC's default backend is slower and its draws differ.
    - Your series in one table with a time column, read with `ergodic.timeseries`.

## Run the program

The data is 60 simulated months. Web visits grow by about 1.5 a month, and each
month's sales are 20 plus 0.8 times last month's visits, plus noise.

```python file="drivers.py"
import numpy as np
import pandas as pd
from ergodic import forecast, timeseries
from ergodic.forecast import schedule, window_graph

# 1. Simulate 60 months. Sales follow last month's web visits.
rng = np.random.default_rng(0)
T = 60
month = pd.date_range("2021-01-31", periods=T, freq="ME")
visits = 100 + 1.5 * np.arange(T) + rng.normal(0, 4, T)
sales = 20 + 0.8 * np.r_[visits[0], visits[:-1]] + rng.normal(0, 2, T)
frame = pd.DataFrame({"month": month, "visits": visits, "sales": sales})
data = timeseries(frame, time="month", freq="ME")

# 2. Say which series drives which: visits one month earlier drive sales.
graph = window_graph([("visits", "sales", 1)])
print(graph)
print(schedule(data, graph, fit="staged"))

# 3. Forecast six months. Visits are forecast too, and their uncertainty reaches sales.
result = forecast(data, 6, graph=graph, fit="staged", draws=500, warmup=500, seed=0)
print(result)
print(result.to_frame().round(1).to_string(index=False))

# 4. Check the sampler before you trust the bands.
for record in result.diagnostics.values():
    print(record)

# 5. Take the sales forecast apart: baseline, season and driver.
waterfall = result.explain(nodes="sales")["sales"]
bars = waterfall.to_frame()[["factor", "value", "lo", "hi"]]
print(bars.round(1).to_string(index=False))
print(f"total {waterfall.total:.1f}; bars add up: {waterfall.relative_closure_error < 1e-9}")
```

```output from="drivers.py" tol="0.5" rtol="0.05" vary="19-20,22"
MixedGraph(kind=dag, nodes=2, edges=[sales[t] <-- visits[t-1]])
TrainingSchedule: 2 stages, 2 by mcmc
  Stage 0 [mcmc     ]: visits
  Stage 1 [mcmc     ]: sales
ForecastResult(draws: 2 series [sales, visits], horizon=6, S=500)
  node  step  median    lo    hi
 sales     1   167.8 163.3 172.2
 sales     2   167.6 159.6 175.8
 sales     3   168.9 159.7 178.9
 sales     4   167.4 156.8 178.8
 sales     5   167.9 156.3 179.0
 sales     6   168.8 155.7 181.2
visits     1   184.3 176.0 193.6
visits     2   184.9 175.7 194.8
visits     3   184.6 172.9 195.2
visits     4   183.9 171.3 195.9
visits     5   182.6 167.7 195.7
visits     6   188.5 172.3 205.2
NodeDiagnostics(visits: 0/500 divergent (0.00%), min_ess=48)
NodeDiagnostics(sales: 0/500 divergent (0.00%), min_ess=62)
         factor  value    lo     hi
Level and trend   99.9  -1.2  169.4
         season   -0.2  -0.6    0.2
         visits  908.6 821.9 1021.5
total 1008.4; bars add up: True
```

- **Step 2.** The window graph has one edge, `visits[t-1] -> sales[t]`. The schedule
  fits visits first, then sales.
- **Step 3.** Each row is a forecast step with its median and a 90% band. The sales
  band widens partly because, from step 2 on, it carries the uncertainty of the
  visits forecast. Sales' own level and trend add the rest.
- **Step 4.** No divergent transitions, and tens of effective draws for the worst
  parameter (the exact count moves from machine to machine at this budget). That is
  enough for a page example; use a larger budget for decisions.
- **Step 5.** Over the six months, the model expects about 1,008 in sales. The
  visits bar is about 909, measured against a world with zero visits. The season bar
  is near zero, as it should be: the simulated data has no season.

## Write the window graph

A window graph is an ordinary DAG over lagged names such as `visits[t-1]`.
`window_graph` builds one from `(source, target, lag)` triples:

```python
graph = window_graph([("visits", "sales", 1)])
```

Lag 0 is a same-month effect. A series can feed itself at lag 1 or more. An edge that
points back in time, or a cycle, raises a `ValueError`. Every variable in the graph
must be a column of the data.

You can also pass the `.graph` of a [time-series discovery](time-series-discovery.md)
result, unchanged. It uses the same naming.

## Choose how the series are fitted

`fit=` decides how the nodes share information:

- `"staged"` fits one node at a time, in the order `schedule` shows. Each child
  receives its parents' whole forecast posterior, draw for draw.
- `"joint"` fits every node in one model, so their posteriors stay correlated.
- `"independent"` fits each series alone and ignores any graph; it refuses a
  `graph=` or a `scenario=`.
- `"auto"`, the default, is joint for up to 4 sampled nodes and staged above that.
  Without a graph, each series is fitted alone. `result.strategy` records the choice.

!!! caution "Read the diagnostics"
    With `fit=` left out, `"auto"` picks the joint fit for this two-node graph. On
    this data and budget, that gave single-digit effective draws, so the page uses
    `fit="staged"`. Even staged, the smallest effective sample sizes are usually
    below the 100 at which `forecast` warns. The trend model mixes slowly on series
    that wander like a random walk, and its starting level and slope are often the
    worst parameters, whichever way you fit. Check `result.diagnostics` for
    divergences and a small `min_ess` before you trust a band, and raise `draws` and
    `warmup`, or `chains`, until `min_ess` is in the hundreds.

## Read the forecast

`forecast` returns a `ForecastResult`. Each series holds a `(draws, horizon)` array
of sample paths, in the units of your data:

- `median(node)` and `mean(node)`: the point path.
- `interval(node, level=0.9)`: the lower and upper band.
- `draws(node)`: the raw paths.
- `to_frame(level=0.9)`: one row per node and step.
- `index`: the date of each step, here the month-ends from January 2026.

## Read the waterfall

`explain()` returns one bar for the series' own level and trend, one for the season,
and one per driver. Each bar carries posterior draws, so `to_frame()` shows a
90% band for it. The bars add up to `total`: the model's mean path, summed over the
window with the default `reduce="sum"`.

Each driver bar is measured against the driver at zero, which may be far outside your
data. When a driver trends, the level bar and the driver bar trade off: read them
together, with their bands, not as a firm split. On this data, the joint fit splits
the same total differently, with a smaller level bar and a larger visits bar.

Pass `windows=[Window("Q1", 0, 3), Window("Q2", 3, 6)]` to split the horizon, by
step or by date. Pass `reduce="mean"` for a rate rather than a flow. Pass
`method="loo"` to switch each factor off in turn instead of on; what no single
factor owns then appears as an `Interaction` bar.

<details markdown="1"><summary>What model sits behind each series?</summary>

Each continuous column gets a `StructuralMechanism`: a damped local linear trend,
a season, and one coefficient per driver. The season comes from the time index:
monthly data gets 12 monthly factors, quarterly data a period of 4, weekly data 52,
and anything else no season. Override per column with
`mechanisms={"sales": StructuralMechanism(seasonality=None)}`. `robust=True` swaps
the Normal likelihood for a Student-t, so a few outliers don't drag the level.
`CausalOU` and `Bass` are the other built-in mechanisms.

The damped trend flattens over a long horizon. Here, visits grew 1.5 a month in the
data, and the forecast median barely rises. Count and categorical columns raise.

</details>

## Next steps

<div class="cards" markdown="1">

- [Backtest a forecast](backtest-a-forecast.md): check the bands on held-out months before you trust them.
- [Run scenarios](run-scenarios.md): pin a driver's future and compare forecasts.
- [Python reference](reference-forecast.md): every option of `forecast`.

</div>
