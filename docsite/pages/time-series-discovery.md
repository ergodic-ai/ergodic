# Discover from time series

Learn which variables drive which, and at what lag, from a time series. You'll run
PCMCI on a seeded simulation, read the lagged links and compare them with Granger
tests.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - One series per variable on a regular time index, or a panel of several units.

## Run the program

The simulation is 500 weeks of three series. Ad spend raises site visits a week later,
and visits raise sales two weeks later. Each series also carries its own past.

```python file="time_series.py"
import numpy as np
import pandas as pd

from ergodic import DomainKnowledge, discover, granger
from ergodic.data import timeseries

# 1. Simulate 500 weeks: ads -> visits at lag 1, visits -> sales at lag 2.
rng = np.random.default_rng(0)
weeks = 500
ads, visits, sales = np.zeros(weeks), np.zeros(weeks), np.zeros(weeks)
noise = rng.normal(size=(weeks, 3))
for t in range(2, weeks):
    ads[t] = 0.6 * ads[t - 1] + noise[t, 0]
    visits[t] = 0.5 * visits[t - 1] + 0.5 * ads[t - 1] + noise[t, 1]
    sales[t] = 0.5 * sales[t - 1] + 0.4 * visits[t - 2] + noise[t, 2]
frame = pd.DataFrame({"week": np.arange(weeks), "ads": ads, "visits": visits, "sales": sales})
data = timeseries(frame, time="week")

# 2. Run PCMCI with links up to two weeks back.
result = discover(data, method="pcmci", max_lag=2)
print(result)

# 3. Each link is (source, target, lag), with the p-value that kept it.
for (source, target, lag), p in sorted(result.links.items()):
    print(f"{source}[t-{lag}] -> {target}[t]  p={p:.2g}")

# 4. Granger tests answer a narrower, predictive question.
print("granger:", sorted(granger(data, max_lag=2).significant))

# 5. Add the order of the funnel: ads, then visits, then sales.
knowledge = DomainKnowledge().with_tiers([["ads"], ["visits"], ["sales"]])
informed = discover(data, method="pcmci", max_lag=2, knowledge=knowledge)
print(sorted(informed.links) == sorted(result.links))
pruned = int(informed.diagnostics["n_knowledge_pruned"])
tests = int(informed.diagnostics["n_ci_tests"]), int(result.diagnostics["n_ci_tests"])
print(f"{pruned} links pruned; {tests[0]} tests instead of {tests[1]}")
```

```output from="time_series.py"
TimeSeriesDiscoveryResult(pcmci/linear: dag on 9 window nodes, max_lag=2, 5 lagged links, 0 contemporaneous, alpha=0.05, n=496)
ads[t-1] -> ads[t]  p=0
ads[t-1] -> visits[t]  p=0
sales[t-1] -> sales[t]  p=0
visits[t-2] -> sales[t]  p=4.4e-15
visits[t-1] -> visits[t]  p=0
granger: [('ads', 'visits'), ('visits', 'sales')]
True
9 links pruned; 37 tests instead of 49
```

- **Step 2.** The result holds a window graph over nodes such as `ads[t-1]`. With
  `max_lag=2` it has 9 nodes: each variable at `t`, `t-1` and `t-2`. The tests
  condition on up to twice `max_lag` weeks of history, so the first four weeks start
  no window and `n=496`.
- **Step 3.** PCMCI finds exactly the five true links: the two cross effects at their
  true lags and each series' memory of its last week. A p-value smaller than
  floating-point precision prints as 0.
- **Step 4.** Granger finds the same two cross effects, as ordered pairs without lags.
- **Step 5.** The tiers forbid later variables from causing earlier ones at every
  lag. The links don't change, but the search skips the forbidden ones and runs fewer
  tests.

## Read the window graph

A `TimeSeriesDiscoveryResult` is a `DiscoveryResult` with two more fields: `max_lag`,
and `links`, which maps `(source, target, lag)` to a p-value. A lag of 0 is a
same-week edge. An oriented one is keyed `(source, target, 0)`; an undirected one once,
with the pair in sorted order.

`result.graph` lives on window nodes such as `visits[t-2]`. Its kind is `dag` when
every edge is oriented and `mixed` otherwise. `result.summary_edges()` flattens it to
variable pairs, with `(x, x)` for a series' own memory. There is no single
variable-level graph on purpose: `x` can drive `y` at one lag while `y` drives `x` at
another.

## Knowledge reads variables as processes

A constraint on `(x, y)` applies at every lag:

- `forbid`, tiers, `order`, `root` and `sink` remove links before testing. A series'
  memory, `x[t-1] -> x[t]`, survives roots and sinks. `forbid(x, x)` removes it.
- Tiers also orient a same-week edge from the earlier tier to the later one.
- `require(x, y)` names no lag, so it can't be forced. It is checked after the run,
  and PCMCI raises if no link from `x` to `y` survived.
- `confound` raises `NotImplementedError`: a window graph has no bidirected edges.

`n_knowledge_pruned` counts what the knowledge removed.

!!! caution "A same-week effect can add lagged links"
    PCMCI here tests each lagged link given lagged parents only. If `y[t]` causes
    `z[t]` in the same week, the past of `y` and of `y`'s causes can also show up as
    lagged links into `z`. Read lagged links into a variable with a same-week parent
    with care.

!!! note "Granger is a predictive test"
    `granger` asks whether the past of `x` improves the forecast of `y`, given every
    variable's past. A common cause you didn't measure makes `x` look causal, and it
    ignores same-week effects. That's why it returns a table of pairs
    (`GrangerResult`), not a graph. It needs continuous columns.

<details markdown="1"><summary>Can I use a panel of several units?</summary>

Yes. Build it with `ergodic.data.panel(frame, entity=..., time=...)` and pass it to
`discover(..., method="pcmci")`. Windows are built within each unit and pooled, so no
window crosses two units. `granger` takes a single `TimeSeriesData` only.

</details>

<details markdown="1"><summary>Can I bootstrap a time series?</summary>

Yes: `bootstrap(data, method="pcmci", max_lag=2)` resamples blocks of consecutive
weeks. See [Check stability with the bootstrap](bootstrap-stability.md).

</details>

## Next steps

<div class="cards" markdown="1">

- [Check stability with the bootstrap](bootstrap-stability.md): block resampling for series.
- [Forecast with causal drivers](forecast-with-drivers.md): use the drivers you found.
- [Python reference](reference-discovery.md): `PCMCI`, `granger` and their options.

</div>
