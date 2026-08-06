# Forecasting

Most forecasting tools hand you a number. This one hands you a number you can take apart.

`ergodic.forecast` fits a Bayesian model per series, so every step of the horizon arrives as a cloud of draws rather than a point with an error bar bolted on afterwards. The mechanism inside each series is *composed* rather than fused: a damped trend, a season, and one term per driver, put together in a stated order. That is what makes the rest of the pillar possible. Because the pieces are still there after fitting, the same object gives you the forecast, the uncertainty, the levels of a hierarchy agreeing with each other, an honest score against a naive baseline, and a waterfall saying what the number is made of.

Two things set it apart from the usual forecasting stack. The predictive structure is a graph, the same window graph temporal discovery already returns, so a discovered structure feeds a forecast with no translation. And the pillar is careful about the difference between an assumption and an intervention: pinning a driver's future says what you expect, not what you would cause, and the API says which one you are doing.

PyMC runs the sampling and it is optional: the bare install never imports it. `pip install ergodic[forecast]` adds it, and installing `jax` and `numpyro` alongside puts sampling on the fast path, which the engine spike measured at about four times the C backend on this model.

## Two graphs, and only one of them is causal

The pillar reads two structures and they mean different things.

The **hierarchy** is the arithmetic of the business. Retail plus wholesale is the total. It says nothing about what moves what; it says which numbers have to add up. It rides on the data, in a `HierarchicalTSData`, and it is what reconciliation reads.

The **window graph** is the causal one. It is a plain DAG over lagged node names, `promo[t] -> retail[t]`, `pmi[t-2] -> demand[t]`. Multiple lags between the same pair become edges between different node pairs, so one edge per pair still holds. This is exactly what `discover(data, method="pcmci")` returns, and `DomainKnowledge` applies to it unchanged.

Here is the running example, and it runs: a coffee roaster with two channels, four years of monthly history, and a marketing calendar. Every number quoted on this page comes out of this world, so it is written down rather than assumed, and the whole page can be typed in order.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_months = 48
month = np.arange(n_months)

# The marketing calendar: campaign bursts on their own rhythm, planned ahead.
promo = np.empty(n_months)
promo[0] = 10.0
for m in range(1, n_months):
    promo[m] = 10.0 + 0.5 * (promo[m - 1] - 10.0) + rng.normal(0.0, 2.5)

# Retail: a growing level, a Christmas-heavy season, 2.5 units per point of promo.
retail_season = 1.0 + 0.18 * np.sin(2.0 * np.pi * (month + 4) / 12.0)
retail = (120.0 + 0.9 * month + 2.5 * promo) * retail_season + rng.normal(0.0, 4.0, n_months)

# Wholesale: bigger, flatter, mildly seasonal, and nothing to do with the promo.
wholesale_season = 1.0 + 0.06 * np.sin(2.0 * np.pi * (month + 7) / 12.0)
wholesale = (240.0 + 1.6 * month) * wholesale_season + rng.normal(0.0, 6.0, n_months)

frame = pd.DataFrame(
    {
        "t": pd.date_range("2022-01-31", periods=n_months, freq="ME"),
        "retail": retail,
        "wholesale": wholesale,
        "total": retail + wholesale,
        "promo": promo,
    }
)
```

The hierarchy is the two channels and their total; the driver rides along as an ordinary column.

```python
from ergodic import forecast, hts, timeseries
from ergodic.data import Hierarchy

tree = Hierarchy.tree({"total": ["retail", "wholesale"]})
data = hts(frame[["t", "retail", "wholesale", "total"]], time="t", hierarchy=tree, freq="ME")
data
# HierarchicalTSData(schema=Schema(4 columns: 3 continuous, 1 datetime), n_rows=48,
#                    time='t', freq='ME', hierarchy=Hierarchy(nodes=3, leaves=2, agg='sum'))
```

Declaring `freq="ME"` is worth the keystrokes. It is what lets the result label its forecast steps on the calendar rather than by counting median day-gaps, which drifts on month ends.

## A first forecast

```python
result = forecast(data, 6, draws=300, warmup=300, seed=0)
result
# ForecastResult(draws: 3 series [total, retail, wholesale], horizon=6, S=300, coherent=True)
```

Every series comes back as `(S, H)` draws. The accessors read the cloud:

```python
result.median("total")
# array([526.8, 503. , 483. , 470.4, 461.2, 465.6])

result.interval("total", level=0.9)   # two arrays, lo and hi
result.draws("total").shape           # (300, 6)
result.to_frame().head(3)
#     node  step  median      lo      hi
#    total     1  526.80  504.13  543.17
#    total     2  502.97  480.06  528.30
#    total     3  483.03  455.88  507.76
```

The band widens with the horizon because the model replays the innovations it fitted and simulates fresh ones for the future. That split is the whole reason the band is honest rather than inherited from the training window's certainty.

The result also carries its provenance, which is how you find out what it did without reading the source:

```python
result.start                # numpy.datetime64('2026-01-31')
result.index                # the label of each forecast step
result.strategy             # FitStrategy(independent (requested 'auto'))
result.reconciliation       # 'bottom_up'
result.coherence_error(tree)  # 0.0
result.diagnostics["retail"]
# NodeDiagnostics(retail: 1/300 divergent (0.33%), min_ess=8)
```

Diagnostics ride along on every result and warn on breach, never raise. A backtest that refits over dozens of folds has to be able to finish, and a diagnostic is information about a fit rather than a verdict on it. Low effective sample size is common on these models at a few hundred draws, and the honest reason is in the model rather than in the sampler: a wandering level and an observation error explain much the same thing, so the two are nearly exchangeable and mix slowly.

## Mechanisms

A `TemporalMechanism` owns one series: the scale it works on, its states, its priors, and its likelihood. With `mechanisms=None` the schema picks, the same dispatch the rest of the package uses.

```python
from ergodic.forecast import assign_mechanisms, infer_period

infer_period(data)        # 12, read off the monthly index
assign_mechanisms(data)   # every continuous column gets StructuralMechanism(period=12)
```

Monthly spacing gives 12, quarterly 4, weekly 52. Anything else, including a plain integer index, gives no period and therefore no season, because guessing a period from data that does not carry one is how a forecast grows a season it never had. A count or categorical column raises rather than getting a Gaussian trend model fitted to it; count and intermittent mechanisms are a named seam, not a silent fallback.

Four mechanisms ship.

**`StructuralMechanism`** is the workhorse: a damped local linear trend, a season, and a regression on the node's graph parents, composed as `(trend + regression + season) * season_factor`. Damping is what stops a twelve-step extrapolation running away on a slope the last few points happened to suggest. The season is twelve monthly factors when the period is twelve and Fourier terms otherwise:

```python
from ergodic.forecast import StructuralMechanism

StructuralMechanism(period=12).seasonal_form()               # ('monthly', 12)
StructuralMechanism(period=7, n_harmonics=3).seasonal_form() # ('fourier', 7)
```

**`CausalOU`** puts the drivers in the long-run attractor instead of in the level. A price index, a margin, an inventory-days number: they sit near an equilibrium, and what a driver changes is the equilibrium, not the value. The value then walks there at its own speed. That difference is testable rather than decorative: hold a driver two units up and this mechanism's forecast walks to the new attractor over several periods where a level regression would step there at once. The transient is the answer to "when", not just "how much".

**`Bass`** is the diffusion S-curve for a launch with no history to extrapolate, with `p`, `q`, and a market size `M`.

**`ScenarioMechanism`** is the resolved role: nothing is fitted, because the path comes from outside.

Each mechanism owns its scaler, auto-picked (a positive series is divided by its median, a signed one is centered and standardized), so a series in dollars and a series in rates can share one set of priors. `robust=True` swaps the Normal likelihood for a Student-t, which lets a few wild points stay wild instead of dragging the level after them. It earns its keep when the shock lands near the forecast origin, where it bends the states that get extrapolated; a spike in the middle of the window is absorbed into the observation noise and the forecast never feels it either way.

## Making the levels agree

Fitted on their own, the total's forecast and the sum of the channels' forecasts are two different numbers.

```python
raw = forecast(data, 6, reconcile="none", draws=300, warmup=300, seed=0)
raw.coherent                                  # False
raw.coherence_error(tree, relative=False)     # 101.16
```

Reconciliation fixes that, and every strategy works on the draws already in hand, so switching between them costs no refitting:

```python
from ergodic.forecast import RECONCILE_METHODS, PROPORTION_RULES, WEIGHT_RULES, reconcile

RECONCILE_METHODS  # ('bottom_up', 'top_down', 'middle_out', 'mint', 'none')
PROPORTION_RULES   # ('historical', 'forecast')
WEIGHT_RULES       # ('posterior', 'mint_shrink')

bottom = reconcile(raw, tree, method="bottom_up")
mint = reconcile(raw, tree, method="mint", weights="mint_shrink")
top = reconcile(raw, tree, method="top_down", data=data, proportions="historical")

bottom.coherence_error(tree)  # 0.0
mint.coherence_error(tree)    # 2.05e-16
top.coherence_error(tree)     # 1.99e-16
```

Coherence is read relative to the size of the parent, not in absolute units, because the residual a projection leaves behind is rounding and rounding scales with the numbers being added. Adding four numbers near a billion cannot do better than about 1e-7 absolute, and calling that incoherent would be wrong.

**Which one to use** is a real question with a measurable answer, and the honest version of it is worth spelling out. A leaf's forecast error has two parts: churn that nothing can predict, and signal that a model estimates badly because the series is small and noisy. Top-down replaces the leaf's own forecast with a share of a cleanly estimated total, which helps with the second part and does nothing about the first. So top-down wins exactly when the leaves' estimation error is large next to their irreducible churn, and loses when the leaves are decent models of clean series.

The package's own test worlds show both directions, which is why neither is the default. On a world where the leaves are noisy and the total is strong, top-down beat bottom-up on leaf CRPS by around 12 percent across three seeds. On a world where the leaves cancel each other out, so the total is nearly flat while the leaves move, bottom-up beat top-down by a factor of five. Pooled over a twelve-world sweep, MinT came out ahead of both, which is the argument for it as a default when you have no strong view: it believes each level in proportion to how certain that level is, and finds the nearest coherent set.

`trust={node: weight}` multiplies a node's precision, which pins the ones you believe and lets the rest absorb the adjustment:

```python
plain = reconcile(raw, tree, method="mint")
trusted = reconcile(raw, tree, method="mint", trust={"total": 10.0})
# the total's own median moves 3.26 under plain mint, and 1.02 when it is trusted
```

The workflow that follows from all this is `forecast(..., reconcile="none")` once and then `reconcile(result, tree, method=...)` as often as you like. A reconciled result cannot be un-reconciled, because bottom-up overwrites the parents' draws with the sum, which is the point.

Note the difference between the string `"none"` and the default, which is `None`. The default means "whatever suits the data": bottom-up when the data is hierarchical, and nothing at all when it is flat. The string is a request, and a request only makes sense where there is a hierarchy to reconcile, so any explicit `reconcile=` on flat data raises rather than being ignored.

## Drivers, scenarios, and what is not an intervention

Give `forecast` a window graph and the series stop being independent.

```python
from ergodic.forecast import schedule, window_graph

graph = window_graph([("promo", "retail", 0), ("promo", "wholesale", 2)])
# edges: promo[t] --> retail[t], promo[t-2] --> wholesale[t]
```

A driver needs a future before a child can be forecast, and there are exactly two honest ways to have one. A driver that is a modelled column is simply forecast, and its whole posterior propagates into the child. A driver whose future is genuinely known, a published rate or a marketing calendar that is already signed off, arrives through `scenario=`. There is no third way: a driver with neither is refused before anything samples, rather than being silently held flat.

Propagation is not a figure of speech. Take the retail channel and its driver on their own, and forecast them twice on the same seeds, once with the driver pinned and once with it forecast by its own mechanism:

```python
from ergodic.forecast import window_graph

flat = timeseries(frame[["t", "retail", "promo"]], time="t", freq="ME")
g = window_graph([("promo", "retail", 0)])

pinned = forecast(flat, 6, graph=g, scenario={"promo": np.full(6, 12.0)},
                  fit="staged", draws=300, warmup=300, seed=0)
modelled = forecast(flat, 6, graph=g, fit="staged", draws=300, warmup=300, seed=0)

def band_width(result, node):
    lo, hi = result.interval(node, level=0.9)
    return float(np.mean(hi - lo))

band_width(pinned, "retail"), band_width(modelled, "retail")
# (19.87, 31.64)
```

That ratio, 1.59 here, is a property of the world rather than a constant: how much of the child's variance the driver carries, and how fast the driver's own band opens. Measured elsewhere in the package's tests it runs from about 1.06 to 1.73. What is invariant is the direction. Pinning a driver hides its uncertainty, and the narrower band is a claim you are making, not a discovery.

`scenario=` accepts an array of length `horizon`, a scalar for a flat path, or the string `"hold"`, which repeats the last observed value and is bitwise identical to passing that value explicitly. Naming it makes the assumption visible, which is the whole reason hold-flat is not the default.

**`scenario=` makes no causal claim.** It pins a path and trusts the fitted regressions downstream. That is exactly right for something exogenous or already decided, and quietly wrong for something the business sets in response to what it sees, like a price that moves with demand. The pillar refuses to blur the two:

```python
forecast(flat, 6, graph=g, scenario={"retail": np.full(6, 300.0)})
# ValueError: scenario={'retail': ...} pins a node the graph says is driven by promo[t].
# scenario= is for a node whose future is exogenous or already known, and it makes no
# causal claim [...] Pinning an endogenous node is an intervention, which needs the
# effect to be identified on the window graph and is what do= will be for when the
# causal layer lands.
```

`do=` is not a keyword yet. When it arrives it will route through identification on the window graph and refuse honestly when the effect is not identified, the same way `SCM.ate` does. The distinction lives in the API from day one so that no working call has to change shape later.

## Staged or joint

With a graph, fitting has two shapes, and `schedule()` shows you the plan before you spend a sampler on it.

```python
from ergodic.forecast import schedule

plan = np.full(6, 12.0)  # the promo calendar, already signed off

schedule(flat, g, scenario={"promo": plan})
# TrainingSchedule: 2 stages, 1 by mcmc
#   Stage 0 [resolved ]: promo
#   Stage 1 [mcmc     ]: retail
```

**Staged** fits one node at a time in wave order and hands each node's whole forecast posterior down to its children. The handoff is a posterior, not a path: the child's forecast model declares a fresh site per frozen parent, the parent's draws are written into a copy of the child's trace, and the predictive pass replays them by name. Every draw of the child sees the matching draw of its parent, so nothing collapses to a point estimate anywhere.

**Joint** puts every node in one model, so an upstream node's own latent mean path feeds the downstream regression directly and the nodes keep their joint posterior.

`fit="auto"` picks joint when the graph has no more than `AUTO_JOINT_MAX_NODES` (four) modelled nodes and stages anything larger. What staged costs is worth stating plainly: **it drops the posterior correlation between nodes**, everything except what flows through the parent handoff. If two series' uncertainties are related in a way the graph does not carry, staged will not know.

One more difference is worth knowing because it looks like a bug and is not. Staged regresses a child on its parent's *observed* history; joint regresses it on the parent's *latent mean path*. When the driver's own mechanism cannot represent the driver's movement, those are different estimands and the coefficients differ. A near-white-noise driver gets read as observation noise by its own trend model, so the mean path it hands the joint model is smoothed and the child's coefficient rises to compensate. Staged is stable across that whole range because it never smooths.

There is one more thing joint can do that staged cannot, and it is not a matter of taste. A hierarchy parent that carries its own column wants to be modelled *and* to be the sum of its children. Under staged fitting the aggregates are added up last, so a parent that models itself has no wave to run in and is refused, with the three ways out named: fit it jointly, drop the parent's column, or forecast without the graph and reconcile afterwards. One model over every node has no wave order to violate, so it fits a parent and its children together quite happily, and the reconciliation menu then applies to the result exactly as before.

Two shapes are refused whichever way you fit: a node that reads a hierarchy aggregate as a driver, and two nodes that drive each other. The second needs a joint multivariate mechanism, which is future work; the refusal says so.

## Backtesting

A forecast that has never been scored against a baseline is a story. `backtest` refits at rolling origins and scores what the model did not see, and it will drive anything that fits a series and forecasts a horizon, so a naive baseline and the Bayesian forecaster land in the same table.

```python
from ergodic import backtest
from ergodic.forecast import SeasonalNaive, forecast_model

flat_total = timeseries(frame[["t", "total"]], time="t", freq="ME")
model = forecast_model(name="structural", draws=300, warmup=300)
report = backtest(model, flat_total, test_size=6, n_splits=2, seed=4, m=12)
report
# BacktestReport(structural, expanding: 2 folds at [36, 42], horizon=6, 1 nodes)

report.summary()[["crps", "mase", "coverage"]]
#          crps   mase  coverage
# total  10.685  0.526       1.0
```

`forecast_model(**options)` is the facade wrapped as a model: configuration now, data later. It takes every keyword `forecast` takes and refuses one it does not, which matters when a mistake would otherwise surface at the first fold and a fold costs a NUTS run. `name=` is what the report calls it, so two configurations of the same model can sit in one table and be told apart.

Against the seasonal-naive baseline on the same splits:

```python
naive = backtest(SeasonalNaive(period=12), flat_total, test_size=6, n_splits=2, seed=4, m=12)
report.compare(naive)
#   node    metric   value   other    delta  improved
#  total  coverage   1.000   0.167    0.833      True
#  total      crps  10.685  31.540  -20.855      True
#  total      mase   0.526   1.420   -0.894      True
```

Three metrics carry most of the meaning. **MASE** scales the error by what the seasonal naive would have got in sample, so below one means the model earned its keep and above one means it did not. **CRPS** scores the whole distribution rather than the middle, charging a forecast both for being wrong and for being vague, which is what stops a confidently narrow band from looking good. **Coverage** is the share of held-out points inside the nominal band, and it is the one number that tells you whether to believe an interval.

The seeds are explicit and recorded. Fold `i` fits at `seed + i` and forecasts at `seed + 1000 + i`, and every fold records both, so a fold can be reproduced exactly.

```python
[(f.fit_seed, f.forecast_seed) for f in report.folds]   # [(4, 1004), (5, 1005)]
report.diagnostics()
#  fold  cutoff  n_nodes  divergences  max_divergence_share  min_ess  worst_node
#     0      36        1            0                   0.0   31.881       total
#     1      42        1            0                   0.0    6.409       total
```

That last table exists because a backtest is the one place a sampling problem can hide behind a decent score: an early fold that barely converged still contributes its metrics. Read it beside the summary.

**The cost is real and worth planning for.** Every fold refits everything from scratch, deliberately, because reusing a posterior across folds leaks the future into the past exactly the way reusing a nuisance model across cross-fitting folds does. The bill is folds times MCMC nodes NUTS runs: three folds of a five-node graph is fifteen fits. The default is `n_splits=3` with `test_size` as the horizon. A cheap approximate path may arrive later; it will never be the silent default.

## Explaining a forecast

`explain()` takes the forecast apart into a baseline, a seasonality bar, and one bar per driver, each with a posterior behind it.

```python
driven = pinned  # the staged fit above: retail, with the promo calendar pinned

attribution = driven.explain()
attribution["retail"]
# NodeAttribution(retail over horizon: total 1,112 = Level and trend +997.3,
#                 season -34.58, promo +148.8; closure +0.00e+00, structural)

attribution["retail"].to_frame()
#    node   window          factor      kind   value      lo       hi  share
#  retail  horizon  Level and trend  baseline  997.31  938.92  1058.48   0.90
#  retail  horizon           season    season  -34.58  -43.68   -26.32  -0.03
#  retail  horizon            promo    driver  148.79  115.73   182.34   0.13
```

Read a bar as what that piece contributes **at its own level**, not as what a change in it would do. A driver sitting at 12 with a coefficient of 2.5 owns what the bar says, season included. A bar's `value` is the posterior mean of its per-draw contribution, which is what makes the bars close: means add up even where medians do not.

Two engines produce that table and the result records which one it got. The **structural** engine is primary and costs nothing: the forecast model already recorded its own pieces per draw, so switching a factor off is arithmetic on arrays that exist. The identity elements are exact rather than approximate, which is why the monthly season is written as `12 * softmax(raw)`: it is exactly one when the raw parameters are zero. The **replay** engine is the fallback for a mechanism that does not expose its pieces. It zeroes the factor's sites in a copy of the trace and forecasts again *with the same seed*, so the fresh innovations come back identical and the difference between two runs is the factor and nothing else. It costs one predictive pass per bar, capped by `max_replays`.

`closure_error` is reported always, and it is not decoration. On the structural path it measures the recomposed bars against the model's own recorded mean path, so a composition rule that ever drifted from what the mechanism builds would show up as a number instead of as silence. On the replay path it measures whether the re-run reproduced the forecast it is explaining.

Windows slice the horizon, by position or by date on the result's own axis:

```python
from ergodic.forecast import Window

driven.explain(windows=[Window("Q1", 0, 3), Window("Q2", 3, 6)])
#  retail  Q1  Level and trend  baseline  497.29
#  retail  Q1           season    season   37.29
#  retail  Q1            promo    driver   82.80
#  retail  Q2  Level and trend  baseline  500.03
#  retail  Q2           season    season  -71.87
#  retail  Q2            promo    driver   65.99

driven.explain(windows=[Window("Q1", "2026-01-01", "2026-04-01")])["retail"]
# NodeAttribution(retail over Q1: total 617.4 = Level and trend +497.3,
#                 season +37.29, promo +82.8; closure +1.14e-13, structural)
```

`reduce="sum"` is right for a flow and `reduce="mean"` for a rate, applied per draw over the window. Getting it wrong is the difference between "the driver added 148 units this half year" and "148 units a month".

`method="loo"` asks the question the other way round. Instead of switching factors on one at a time, it takes each one out of the full model and books what no single factor owns as an `Interaction` bar:

```python
driven.explain(method="loo")["retail"].to_frame()
#  retail  horizon  Level and trend     baseline  997.31
#  retail  horizon           season       season  -39.90
#  retail  horizon            promo       driver  148.79
#  retail  horizon      Interaction  interaction    5.32
```

That bar is real structure rather than error. A multiplicative season scales the driver's contribution too, so removing pieces one at a time does not recover the whole. On an additive world it is exactly zero. Neither reading is more correct; they answer different questions.

A node with no model of its own is refused by name rather than explained:

```python
driven.explain(nodes="promo")
# ValueError: 'promo' has no model of its own to take apart: it is either an aggregate
# of its children, a node whose path was supplied through scenario=, or a series from a
# model that keeps no fit. Explainable here: ['retail'].
```

The result object holds data, not pictures. Plotting lives in your notebook, and the waterfall above is a dozen lines of matplotlib over `record.bars`; the [forecasting learn guide](learn/11-forecasting.py) draws one.

One more honesty note on attribution. The bars explain the model's **mean path**. The result's draws also carry observation noise, which averages to nothing, and on a reconciled result they carry whatever the projection moved. Both totals are reported, `total` and `forecast_total`, rather than assumed equal: a decomposition explains a model, and reconciliation happens after the model.

## What to watch

Five things worth knowing before any of this reaches a plan.

**Calibrate intervals against a backtest rather than trusting the posterior.** On a series where the level innovations swamp the slope, the trend state is close to unidentifiable and a long extrapolation under-covers: the engine spike measured 74 to 78 percent coverage against a nominal 90 on twelve seeded worlds, on all four sampler backends alike. That is a property of the model class and the world, not of the sampler. The backtest's `coverage` row is the honest reading and it is cheap.

**The damping parameter comes back low.** Across worlds with planted values from 0.75 to 0.90, the posterior median lands below the truth every time. A slope that decays faster than the truth flattens the forecast sooner than it should, so a long-horizon level extrapolation is biased toward under-extrapolating the trend. That is the conservative direction, which is why nothing fails, but a twelve-month plan built on a twelve-month extrapolation is understating the trend. The lever, if it matters for your series, is the prior gap between the level and slope scales rather than the prior on damping itself.

**Staged fitting drops cross-node correlation**, everything except what flows through the parent handoff. `fit="joint"` keeps it and costs one bigger sampler.

**`scenario=` is an assumption, not an intervention**, and the section above says what the difference costs.

**The pillar's name is shadowed, on purpose.** `ergodic.forecast` is both the module and the function on it, and the function wins: `from ergodic import forecast` gives you something callable, while `from ergodic.forecast import backtest` still reaches the module through the import system. The one thing that does not do what it looks like is `import ergodic.forecast as mod`, which binds the *function*, because Python prefers the attribute when one exists. This is the only shadowed name in the package.

## What is not here yet

Named so you can plan around it. `do=` and identification on window graphs, which is the causal layer. `structure="discover"` as a convenience keyword, though the manual path already works today: hand `discover(..., method="pcmci")`'s graph straight to `forecast(..., graph=...)`. The merge with `ergodic.scm` for counterfactual trajectories. The constrained Bayesian harmonizer, whose elasticity constraints will take DML posteriors rather than assumed priors. `select_forecaster`, the backtest-CRPS ranking and ensemble. Count and intermittent mechanisms, behind the seam that raises today. Full-covariance MinT-shrink, `PanelData` entry, and the foundation-model seam.

The [learn guide](learn/11-forecasting.py) runs the whole pillar end to end on one planted world, and the [API reference](reference.md#forecasting) has the surface.
