# Backtest a forecast

Score a forecaster on months it has not seen, next to simple baselines, before you
trust its numbers or its bands.

!!! need "You'll need"
    - ergodic with the forecasting extra: `pip install "ergodic[forecast]"`. It adds
      PyMC only. Optionally, `pip install jax numpyro` switches sampling to the
      faster NumPyro backend, which recorded the output on this page.
      The baselines alone need no extra.
    - A series long enough to hold out a few horizons: by default, each fold trains
      on at least twice the horizon.

## Run the program

The data is 60 simulated months of sales with a steady trend and a yearly season.
The backtest cuts the history at two points, refits each model on everything before
the cut, and scores the six months after it.

```python file="backtest_sales.py"
import numpy as np
import pandas as pd
from ergodic import backtest, timeseries
from ergodic.forecast import Drift, Naive, SeasonalNaive, forecast_model

# 1. Simulate 60 months of sales with a trend and a yearly season.
rng = np.random.default_rng(0)
T = 60
month = pd.date_range("2021-01-31", periods=T, freq="ME")
season = 10 * np.sin(2 * np.pi * np.arange(T) / 12)
sales = 100 + 1.5 * np.arange(T) + season + rng.normal(0, 3, T)
data = timeseries(pd.DataFrame({"month": month, "sales": sales}), time="month", freq="ME")

# 2. Line up three baselines and the Bayesian forecaster.
models = [
    Naive(),
    SeasonalNaive(period=12),
    Drift(),
    forecast_model(name="Bayesian", draws=500, warmup=500),
]

# 3. Refit each model at two origins and score the next six months.
reports = [backtest(model, data, test_size=6, n_splits=2, m=12) for model in models]
print(reports[-1])

# 4. Compare the mean scores across folds.
columns = ["mae", "rmse", "mase", "crps", "coverage"]
table = pd.DataFrame({r.model: r.summary().loc["sales", columns] for r in reports}).T
print(table.round(2).to_string())

# 5. Check how the sampler behaved in each fold.
diagnostics = reports[-1].diagnostics()
print(diagnostics[["fold", "cutoff", "divergences", "min_ess"]].round(1).to_string(index=False))
```

```output from="backtest_sales.py" tol="0.5" rtol="0.1" vary="8-9"
BacktestReport(Bayesian, expanding: 2 folds at [48, 54], horizon=6, 1 nodes)
                 mae   rmse  mase   crps  coverage
Naive          10.53  10.87  0.57   6.79      0.92
SeasonalNaive  16.55  17.21  0.89  14.31      0.00
Drift          10.73  10.96  0.58   7.09      0.67
Bayesian        5.30   6.00  0.29   3.46      1.00
 fold  cutoff  divergences  min_ess
    0      48            1     25.8
    1      54            0     26.8
```

- **Step 3.** The folds train on the first 48 and the first 54 months. Each one
  forecasts the 6 months that follow.
- **Step 4.** The Bayesian forecaster halves the error of the best baseline on every
  error measure. Its 90% bands held all 12 held-out months.
- **Step 4, continued.** `SeasonalNaive` repeats last year, so it misses a year of trend: its
  bands never contain the actuals.
- **Step 5.** At most a divergent transition or two across both folds' 1,000 draws,
  and a few to a few tens of effective draws for the worst parameter in each fold,
  well below the 100 at which the fit warns. These diagnostics move from machine to
  machine at this budget. Raise `draws` and `warmup`, or `chains`, for decisions.

## Read the scores

Lower is better for every error measure here:

- **mae** and **rmse**: the error of the median path, in the units of the series.
- **mase**: the MAE divided by the in-sample error of a seasonal naive forecast at
  period `m`. Below 1 beats that floor. Pass `m=12` for monthly data with a yearly
  season; the default `m=1` scales by the plain naive.
- **crps**: the continuous ranked probability score. It scores the whole forecast
  distribution against each actual, and equals the absolute error for a point
  forecast.
- **coverage**: the share of actuals inside the central band at `level`, 0.9 by
  default. Judge it by its distance from 0.9, not by its size.

The report holds 14 metrics in all, including MAPE, sMAPE and rolling accuracy over
3- and 6-month windows. `report.metrics` lists them.

## Read the report

`backtest` returns a `BacktestReport`:

- `summary()`: the mean of each metric across folds, one row per series.
- `values(node, metric)`: one metric, fold by fold.
- `to_frame()`: one row per fold, series and metric.
- `diagnostics()`: one row per fold with the sampler's divergences and smallest
  effective sample size. It is empty for a model that doesn't sample.
- `compare(other)`: this report against another, metric by metric, with an
  `improved` column that knows which direction is better.

!!! caution "Two folds is a small test"
    Twelve held-out months can't tell a 90% band from a 95% one. Use more folds when
    the history allows, and treat a coverage far from the nominal level as a reason
    to widen your doubt, not to tune the level.

## Choose the folds

- `test_size` is both the held-out window and the forecast horizon.
- `mode="expanding"`, the default, makes `n_splits` folds. Each trains on the whole
  history before its cut. The cuts count back from the end, so the last months are
  always scored.
- `mode="single"` keeps only the last cut.
- `min_train_size` is the shortest training prefix allowed, `2 * test_size` by
  default.
- `seed` sets the randomness: fold `i` fits with `seed + i` and forecasts with
  `seed + 1000 + i`.

Each fold refits from scratch on a copy of the model, so no fold sees the future,
and the model you pass in is never changed.

<details markdown="1"><summary>Which models can I backtest?</summary>

Anything with `fit(data, *, seed=0)`, returning a fitted copy, and
`forecast(horizon, *, seed=0)`, returning a `ForecastResult`. The baselines and
`forecast_model(...)` both qualify, and so can your own model.

`forecast_model(name=..., **options)` takes every keyword of `forecast` except
`data` and `horizon`. A keyword `forecast` doesn't take raises at once, not at the
first fold. `name` labels the model in reports, so two configurations can be told
apart.

A series in the forecast that isn't a column of the data can't be scored. The report
lists it in `nodes_unscored`.

</details>

## Next steps

<div class="cards" markdown="1">

- [Forecast with causal drivers](forecast-with-drivers.md): add drivers to the forecaster you just tested.
- [Reconcile a hierarchy](reconcile-a-hierarchy.md): forecasts that add up across levels.
- [Python reference](reference-forecast.md): every metric and baseline.

</div>
