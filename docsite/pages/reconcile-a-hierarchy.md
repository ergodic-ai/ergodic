# Reconcile a hierarchy

Forecast every level of a hierarchy, such as regions and their total, then make the
levels add up. You choose which level to believe; ergodic does the arithmetic on
every posterior draw.

!!! need "You'll need"
    - ergodic with the forecasting extra: `pip install "ergodic[forecast]"`.
    - One column per series and a parent-to-children mapping, read with
      `ergodic.hts`. A parent must be the sum of its children.

## Run the program

The data is 48 simulated months of sales in two regions, `north` and `south`, plus
their `total`. Each of the three series gets its own model.

```python file="reconcile_regions.py"
import numpy as np
import pandas as pd
from ergodic import forecast, hts
from ergodic.forecast import reconcile

# 1. Simulate 48 months of sales in two regions; the total is their sum.
rng = np.random.default_rng(0)
T = 48
month = pd.date_range("2022-01-31", periods=T, freq="ME")
north = 60 + 1.0 * np.arange(T) + rng.normal(0, 2, T)
south = 40 + 0.5 * np.arange(T) + rng.normal(0, 2, T)
frame = pd.DataFrame({"month": month, "total": north + south, "north": north, "south": south})
data = hts(frame, time="month", freq="ME", hierarchy={"total": ["north", "south"]})
tree = data.hierarchy
print(tree)

# 2. Forecast every level with its own model, and leave the levels as they are.
raw = forecast(data, 6, reconcile="none", draws=400, warmup=400, seed=0)
print(raw)

# 3. The levels disagree: the total's forecast is not the sum of the regions'.
gap = raw.mean("total") - raw.mean("north") - raw.mean("south")
print("total minus regions:", gap.round(1))

# 4. Reconcile four ways. Each reuses the same draws; nothing is refitted.
recipes = [
    ("bottom_up", {}),
    ("top_down", {"data": data}),
    ("mint", {}),
    ("mint", {"weights": "mint_shrink"}),
]
for method, options in recipes:
    result = reconcile(raw, tree, method=method, **options)
    final = result.mean("total")[-1]
    print(f"{result.reconciliation:32} coherent={result.coherent}  total in month 6: {final:.1f}")
```

```output from="reconcile_regions.py" tol="0.5" rtol="0.05"
Hierarchy(nodes=3, leaves=2, agg='sum')
ForecastResult(draws: 3 series [total, north, south], horizon=6, S=400, coherent=False)
total minus regions: [1.1 1.5 1.3 0.7 1.1 1.2]
bottom_up                        coherent=True  total in month 6: 183.1
top_down(proportions=historical) coherent=True  total in month 6: 184.3
mint(weights=posterior)          coherent=True  total in month 6: 183.6
mint(weights=mint_shrink)        coherent=True  total in month 6: 183.8
```

- **Step 2.** `reconcile="none"` keeps each model's own answer, and the result says
  `coherent=False`.
- **Step 3.** On average, the total's model forecasts about 1 more per month than the
  two regional models add up to.
- **Step 4.** Every strategy makes the levels agree on every draw. They differ in
  where they land the total, because each believes a different level.

## Choose a strategy

| `method` | What it believes | Options |
|---|---|---|
| `"bottom_up"` | The leaves. Each parent becomes the sum of its children. | None |
| `"top_down"` | The root. Its draws are split among the children. | `data=` (required), `proportions="historical"` or `"forecast"` |
| `"middle_out"` | One level in between, summed up and split down from there. | `data=`, `level=` (the root is 0), `proportions=` |
| `"mint"` | Every level, weighted by how sure it is. | `weights="posterior"` or `"mint_shrink"`, `trust={node: weight}` |
| `"none"` | No level. Nothing changes. | None |

`historical` proportions split a parent by the children's shares in the training
data. `forecast` proportions use the shares in each draw of the children's own
forecasts. `mint` moves each forecast as little as possible, weighted by its
precision: the posterior variance by default, or the in-sample residuals with
`mint_shrink`. A node with `trust` weight 10 moves about a tenth as far.

An option a strategy doesn't read raises a `ValueError`, so give each method its own
options, as the program does.

## Reconcile inside the forecast

You don't have to call `reconcile` yourself. `forecast` reconciles a hierarchy with
`bottom_up` unless you pass another `reconcile=`, with the same options:

```python
raw = forecast(data, 6, reconcile="none", draws=400, warmup=400, seed=0)
```

Forecasting with `reconcile="none"` and reconciling afterwards lets you compare
strategies without sampling again.

!!! note "Check coherence yourself"
    `result.coherent` says whether the levels agree. `result.coherence_error(tree)`
    gives the largest gap between a parent and its children, relative to the
    parent's size. After reconciliation it is at the level of rounding error.

<details markdown="1"><summary>What shapes of hierarchy work?</summary>

- A strict tree from `{parent: [children]}`. Deeper trees work the same way.
- A node without a column is forecast as the sum of its children's forecasts.
  Every leaf needs a column.
- Reconciliation needs sum aggregation. A hierarchy built with
  `aggregation="mean"` raises.
- On flat data, any reconciliation option raises: there is nothing to reconcile.
- The result lists the total first, then each level down.

</details>

## Next steps

<div class="cards" markdown="1">

- [Backtest a forecast](backtest-a-forecast.md): score each level on held-out months.
- [Forecast with causal drivers](forecast-with-drivers.md): add drivers to the series.
- [Python reference](reference-forecast.md): `reconcile`, `hts` and every option.

</div>
