# Forecast with Welt's Chronos-2

Forecast a series with Chronos-2, hosted on Welt, then backtest it against
baselines and reconcile a hierarchy in ergodic. Covariates you know ahead of time,
such as a promotion calendar, go in with the history.

!!! need "You'll need"
    - Python 3.11, 3.12 or 3.13.
    - ergodic with the Welt extra: `pip install "ergodic[welt]"`.
    - A [Welt](https://welt.ergodic.dev) workspace with credits. Every forecast in
      this guide is a paid request, and a backtest pays once per fold.

## Connect to Welt

Run `welt login` once, or set `WELT_API_KEY`. `WeltForecaster` opens a short-lived
Welt client per forecast unless you pass your own `client=`. See
[Discover in Welt, estimate in ergodic](welt.md#connect-to-welt).

## Run the program

The data is 120 simulated days of sales for one store. Every Thursday is a
promotion day that lifts sales by about 30, and the promotion calendar is known in
advance. The program forecasts the next 14 days, then backtests Chronos-2 against two
free baselines.

```python file="welt_forecast.py" requires="a Welt API key"
import numpy as np
import pandas as pd
from ergodic import backtest, timeseries
from ergodic.forecast import Naive, SeasonalNaive
from ergodic.welt import WeltForecaster

# 1. Simulate 120 days of history and the 14 after, with a promotion every Thursday.
rng = np.random.default_rng(0)
days = pd.date_range("2026-01-01", periods=134, freq="D")
promo = (days.dayofweek == 3).astype(float)
sales = 200 + 0.5 * np.arange(134) + 30 * promo + rng.normal(0, 5, 134)
frame = pd.DataFrame({"day": days, "sales": sales, "promo": promo})
history, ahead = frame.iloc[:120], frame.iloc[120:]
data = timeseries(history, time="day", freq="D", types={"promo": "continuous"})

# 2. Forecast 14 days, telling Chronos-2 the promotion calendar: one paid job.
model = WeltForecaster(future_covariates=["promo"])
result = model.fit(data).forecast(14, future=ahead[["day", "promo"]], seed=0)
print(result)
print(result.median().round(0))
record = result.diagnostics["sales"]
print(record.model_version, record.configuration, len(record.quantile_levels))

# 3. Backtest it at three origins against two free baselines: three paid jobs.
models = [WeltForecaster(future_covariates=["promo"]), Naive(), SeasonalNaive(period=7)]
reports = [backtest(m, data, test_size=14, n_splits=3) for m in models]
columns = ["mae", "crps", "coverage"]
table = pd.DataFrame({r.model: r.summary().loc["sales", columns] for r in reports}).T
print(table.round(2).to_string())
print("paid forecasts:", models[0].forecasts_used)
```

The program needs your Welt key, so this page shows no recorded output. Each step
prints:

- **Step 2.** `ForecastResult(draws: 1 series [sales], horizon=14, S=1000)`, then the
  14 daily medians. The Thursdays stand out, because Chronos-2 was told when the
  promotions fall. Then the exact model and configuration versions Welt ran, and
  `21`: the quantile levels behind the draws.
- **Step 3.** One row per model (`welt:chronos-2`, `Naive`, `SeasonalNaive`) with
  its MAE, CRPS and 90% coverage, averaged over the three folds. `Naive` and
  `SeasonalNaive` run locally and cost nothing. Then `paid forecasts: 3`, one per
  fold.

## Read the result

`forecast` returns the same `ForecastResult` as every ergodic forecaster: per-series
draws with `median`, `interval`, `draws` and `to_frame`. Welt answers each step with
quantiles, and ergodic turns them into 1,000 draws (`draws=`), seeded by `seed=`:

```python
record = result.diagnostics["sales"]
print(record.model_version, record.configuration, len(record.quantile_levels))
```

Each series has a `WeltForecastDiagnostics` in `diagnostics`: the Welt `job_id`,
`model_version` and `configuration`, the `quantile_levels`, Welt's own `quantiles`
(one row per step), `context_length`, and `rearranged_steps` (the steps whose
native quantiles crossed and that Welt sorted).

!!! caution "The draws are per-step marginals"
    Each step is sampled on its own, so a draw is not a path: it has no dependence
    from one step to the next, and series have none between them. For one series
    as forecast, the median, intervals, coverage and CRPS read one step at a time,
    so they follow Welt's quantiles. A total over several steps, a bottom-up
    aggregate, and the intervals, coverage and CRPS after `bottom_up` or `mint`
    reconciliation all treat the errors as independent, which understates the
    spread when they move together. `top_down` scales the root's draws, so it
    doesn't depend on this. Chronos-2's intervals are not calibrated either;
    backtest coverage before you trust them.

The draws come from the quantiles by inverse-CDF sampling: linear between levels,
with exponential tails beyond the 1% and 99% levels.
`ergodic.forecast.quantile_draws` does the same for any quantile model.

## Give it covariates

| Argument | What it holds | Needs |
|---|---|---|
| `future_covariates=` | Columns whose future you know: promotions, holidays, prices. | Their values over the horizon, as `forecast(..., future=)`. |
| `past_covariates=` | Columns observed only up to now: temperature, traffic. | Nothing more. |

A categorical column, by the dataset's schema, goes to Welt as a categorical
covariate. Future covariates can't have missing values. Every column that is not a
covariate is forecast, unless you choose with `targets=`.

```python
model = WeltForecaster(future_covariates=["promo"])
result = model.fit(data).forecast(14, future=ahead[["day", "promo"]], seed=0)
```

`future=` is a dataframe or a `TimeSeriesData` with the time column and every
future covariate, one row per step. `forecast` reads only those columns, so a
frame that also holds the actuals leaks nothing.

`backtest` does this for you. At each origin it hands the model the true future
covariates of the window it forecasts, and nothing else from that window. The
targets never reach Welt.

## Reconcile a hierarchy

Forecast every level with Chronos-2, then make the levels add up with any
reconciliation strategy. ergodic reconciles the draws locally, without another job.

```python file="welt_reconcile.py" requires="a Welt API key"
import numpy as np
import pandas as pd
from ergodic import hts
from ergodic.forecast import reconcile
from ergodic.welt import WeltForecaster

# 1. Simulate 90 days of sales in two regions; the total is their sum.
rng = np.random.default_rng(0)
days = pd.date_range("2026-01-01", periods=90, freq="D")
north = 120 + 0.4 * np.arange(90) + rng.normal(0, 4, 90)
south = 80 + 0.2 * np.arange(90) + rng.normal(0, 4, 90)
frame = pd.DataFrame({"day": days, "total": north + south, "north": north, "south": south})
data = hts(frame, time="day", freq="D", hierarchy={"total": ["north", "south"]})
tree = data.hierarchy

# 2. Forecast all three levels in one paid job, and leave them as they are.
raw = WeltForecaster().fit(data).forecast(14, seed=0)
print(raw)
gap = raw.mean("total") - raw.mean("north") - raw.mean("south")
print("total minus regions:", gap.round(1))

# 3. Reconcile two ways. Both reuse the same draws.
for method in ["bottom_up", "mint"]:
    fixed = reconcile(raw, tree, method=method)
    print(f"{fixed.reconciliation:24} coherent={fixed.coherent}")
```

- **Step 2.** `ForecastResult(draws: 3 series [total, north, south], horizon=14,
  S=1000)`. Each level is its own series, forecast as if alone, so the total's
  forecast need not equal the regions' sum.
- **Step 3.** `bottom_up coherent=True` and `mint(weights=posterior) coherent=True`.

Pass `reconcile="mint"` (or `"bottom_up"`, `"top_down"`) to `WeltForecaster` to
reconcile inside `forecast`, which is how a backtest scores reconciled forecasts.
Read the reconciled coverage and CRPS of `bottom_up` and `mint` with the caution
above: they assume the series' errors are independent.
Pass `targets=tree.leaves` to forecast only the bottom level: each parent is then
the sum of its children's draws, and the job sends fewer series. MinT-shrink needs
in-sample residual variances, which Chronos-2 doesn't return: pass `residual_var=`
to `reconcile` yourself, or keep the posterior weights.

## Choose how series are sent

- By default each column is its own series, forecast as if it were sent alone
  (`cross_learning=False`). `cross_learning=True` lets the series of one job share
  information.
- `multivariate=True` sends the columns as the targets of one series, which
  Chronos-2 forecasts jointly.
- Welt prices a forecast by series length times variates (the targets plus
  covariates), so with covariates the multivariate job is the cheaper one: the
  covariates are sent once rather than with every series.

## Count what you pay for

Every `forecast` call is one paid job, and a backtest sends one per fold.
`max_forecasts=` caps the jobs this forecaster and every copy of it may send, 10
by default; one more raises `RuntimeError` before it pays. `forecasts_used` counts
the jobs Welt accepted. A request Welt refuses, such as a 422 for a bad cell, costs
nothing and isn't counted.

```python
print("paid forecasts:", models[0].forecasts_used)
```

The first forecast reads the model's catalogue entry, which is free, and checks the
request against its measured limits: history length, variates, series and
horizon. A request outside every measured limit raises before anything is sent.
`max_context=512` sends only the last 512 rows. Run `welt models list` to see the
limits Welt serves today.

The first job's model and configuration versions are then required of every later
job, so the folds of one backtest all run the same Chronos-2. Build a new
`WeltForecaster` to pick up a newer version.

The history and future tables always go inline in the request, as the Welt SDK
sends them; nothing is uploaded to your workspace. A request over 10 MiB of JSON,
the API's limit, raises before anything is sent: use `max_context=`, fewer series
per job, or `multivariate=True`, which sends the covariates once.

## Limits

- `fit` reads a `TimeSeriesData` or `HierarchicalTSData` with a regular datetime
  index, or an evenly spaced numeric one. A numeric index, or a datetime index on a
  business or custom calendar (`freq="B"` or `"C"`, declared or inferred), is sent
  as consecutive days from 1 January 2000. Chronos-2 reads the order of the rows,
  not the calendar, so the forecast is the same, but the job's records in Welt show
  those dates. `time_encoding` in the diagnostics says which was sent.
- Any other irregular datetime index, such as daily data with missing days, raises:
  sending it as steps would close up the gaps. Reindex to the regular frequency and
  leave the missing targets as NaN, which Chronos-2 handles, for example
  `frame.set_index("day").asfreq("D").reset_index()`.
- Targets must be continuous. Missing target values are sent as missing, and
  Chronos-2 handles them.
- Covariates are continuous or categorical. A categorical cell goes as a string or
  an integer (`True` as `1`, `2.0` as `2`); a continuous one as a number.
- `ergodic.forecast(data, ...)` stays the Bayesian forecaster. Chronos-2 is reached
  through `WeltForecaster`, which `backtest` and `reconcile` take like any other
  model.
- `Chronos` is another name for `WeltForecaster`.

<details markdown="1"><summary>What if a job times out?</summary>

`forecast` waits 600 seconds by default (`timeout=`). A timeout leaves the job
running in Welt, and the error names it. Read it with
`welt.Client().job(job_id).result()` rather than forecasting again. That returns
Welt's quantiles, one row per step; `ergodic.forecast.quantile_draws` turns them
into draws.

</details>

## Next steps

<div class="cards" markdown="1">

- [Backtest a forecast](backtest-a-forecast.md): folds, metrics and the report.
- [Reconcile a hierarchy](reconcile-a-hierarchy.md): every strategy and its options.
- [Python reference](reference-welt.md): `WeltForecaster` and every option.

</div>
