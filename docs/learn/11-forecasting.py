# %% [markdown]
# # Forecasting: a number for next quarter, and where it came from
#
# A specialty coffee roaster sells through two channels. Retail is the cafes and the
# webshop, small and seasonal and pushed around by whatever marketing is running.
# Wholesale is the supermarket contracts, larger and steadier. Finance wants one number
# for the next six months, and they want it three ways: for retail, for wholesale, and for
# the total, which had better be the two of them added up. Then somebody in the room asks
# the question that makes forecasting interesting. *Why* is that number what it is, and
# what happens to it if we cut the promo budget?
#
# A point forecast cannot answer that. A forecast that was built out of parts can, because
# the parts are still there afterwards. That is what this guide is about: forecasting with
# a model whose pieces stay separable, so the same object gives you the number, the
# uncertainty around it, the levels agreeing with each other, and an honest account of what
# is driving it.
#
# As always, the answer goes into the simulation first.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_months = 48
month = np.arange(n_months)

# The marketing calendar: campaign bursts, planned a couple of quarters ahead. It runs on
# its own rhythm rather than the calendar's, which is what lets its effect be told apart
# from the season later.
promo = np.empty(n_months)
promo[0] = 10.0
for m in range(1, n_months):
    promo[m] = 10.0 + 0.5 * (promo[m - 1] - 10.0) + rng.normal(0.0, 2.5)

# Retail: a growing level, a Christmas-heavy season, and 2.5 units per point of promo.
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
frame.tail(3)

# %% [markdown]
# Four series, monthly, four years of them. Three things are planted and worth keeping in
# mind, because we will go looking for each one later. Retail carries a strong season (plus
# or minus 18 percent) and wholesale carries a weak one (6 percent). Promo moves retail by
# **2.5 units per point** and moves wholesale not at all. And the total is exactly retail
# plus wholesale, which is a fact about the business, not about the model, and any forecast
# that breaks it is telling three inconsistent stories.
#
# We hold out the last six months, so there is something honest to score against.

# %%
from ergodic import forecast, hts, timeseries
from ergodic.data import Hierarchy

HOLDOUT = 6
train_frame = frame.iloc[:-HOLDOUT]
holdout = frame.iloc[-HOLDOUT:]

tree = Hierarchy.tree({"total": ["retail", "wholesale"]})
data = hts(train_frame[["t", "retail", "wholesale", "total"]], time="t", hierarchy=tree, freq="ME")
data

# %% [markdown]
# `hts` is the hierarchical time series: the same typed dataset as everything else in the
# package, plus a tree saying which series add up to which. This is the pillar's **first**
# graph, and it is not a causal one. It says nothing about what moves what; it says the
# arithmetic of the business.
#
# ## The first forecast
#
# `forecast(data, horizon)` reads the schema, gives every column a mechanism, fits each one,
# and continues it forward. The default mechanism is a damped local linear trend with a
# season, and it read `freq="ME"` off the index to know the season is twelve months long.
#
# `reconcile="none"` is deliberate. It gives us the raw per-series forecasts before anything
# is made to agree, which is what the next section needs. Sampling takes a few seconds per
# series.

# %%
raw = forecast(data, HOLDOUT, reconcile="none", draws=300, warmup=300, seed=0)
raw

# %% [markdown]
# The sampler complained, and the complaints are printed rather than swallowed: a few draws
# diverged and the effective sample size is small. That is the pillar warning and carrying
# on, deliberately, because a backtest that refits over dozens of folds has to be able to
# finish and a diagnostic is information about a fit rather than a verdict on it. The usual
# cause on this model is that a wandering level and observation noise explain much the same
# movement, so the two mix slowly; more draws is the usual fix, and 300 is a docs budget
# rather than a recommendation. The diagnostics ride on every result, and the backtest
# further down reads them fold by fold.

# %%
raw.diagnostics["retail"]

# %%
raw.to_frame().head(4)

# %% [markdown]
# Every series comes back as draws, not as a path: `median()` is the middle of the cloud and
# `interval()` is the band around it. The whole distribution is there, which is what lets a
# stocking decision be made against a percentile rather than against a point.

# %%
import matplotlib.pyplot as plt


def draw_fan(node, ax):
    history = train_frame[node].to_numpy()
    lo, hi = raw.interval(node, level=0.9)
    past = np.arange(len(history))
    future = np.arange(len(history), len(history) + HOLDOUT)
    ax.plot(past[-24:], history[-24:], color="#33415c", lw=1.4, label="history")
    ax.plot(future, raw.median(node), color="#d1495b", lw=1.6, label="forecast median")
    ax.fill_between(future, lo, hi, color="#d1495b", alpha=0.18, label="90% band")
    ax.plot(future, holdout[node].to_numpy(), color="#2a9d8f", lw=1.4, ls="--", label="held out")
    ax.set_title(node)
    ax.set_xlabel("month")


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
for node, ax in zip(("retail", "total"), axes, strict=True):
    draw_fan(node, ax)
axes[0].legend(fontsize=8, frameon=False)
fig.tight_layout()

# %% [markdown]
# Both bands widen with the horizon, which is the model simulating fresh innovations rather
# than replaying the ones it fitted. Retail's median climbs into Christmas because the
# season it learned says December is the month.
#
# ## Making the levels agree
#
# Each series was fitted on its own, so nothing forced the total's forecast to equal the sum
# of the other two. It does not.

# %%
print(f"coherence gap, unreconciled: {raw.coherence_error(tree, relative=False):.2f} units")

# %% [markdown]
# A hundred and twenty-seven units of coffee that nobody can account for, and that is the
# whole reason reconciliation exists: three separately sensible forecasts are one
# inconsistent plan. Three strategies fix it, all working on the draws that are already in
# hand, so switching between them costs no refitting at all:
#
# - **bottom_up** believes the leaves and adds them up. Coherent by construction, and it
#   throws away the total's own model.
# - **top_down** believes the total and splits it by each channel's historical share. The
#   opposite bet.
# - **mint** believes everybody a little, in proportion to how certain each one is, and
#   finds the nearest coherent set of forecasts.
#
# There is no universal winner, and the honest thing is to say why. A leaf forecast has two
# kinds of error in it: churn that no model can predict, and signal that a model estimates
# badly because the series is small and noisy. Top-down helps with the second and does
# nothing about the first, so it wins exactly when the leaves are noisy relative to the
# structure they share. Here the leaves are decent models of clean series, so we should not
# expect top-down to run away with it.

# %%
from ergodic.forecast import crps, reconcile

actuals = {node: holdout[node].to_numpy() for node in ("retail", "wholesale", "total")}
recipes = {
    "none": {},
    "bottom_up": {},
    "top_down": {"data": data, "proportions": "historical"},
    "mint": {"weights": "mint_shrink"},
}
rows = []
for name, options in recipes.items():
    result = reconcile(raw, tree, method=name, **options)
    rows.append(
        {
            "strategy": name,
            "coherence": result.coherence_error(tree, relative=False),
            **{node: crps(actuals[node], result.draws(node)) for node in actuals},
        }
    )
scores = pd.DataFrame(rows).set_index("strategy").round(3)
scores

# %% [markdown]
# CRPS is the probabilistic score to read here: it charges a forecast for being wrong *and*
# for being vague, so a narrow band in the wrong place is punished twice. Lower is better.
#
# Three things to take from the table. Every strategy except `none` closes the coherence gap
# to zero, which is the contract. Bottom-up and MinT both improve the total, because two
# clean channels added together are a better estimate of the total than the total's own
# model. And top-down is a disaster for the leaves here, doubling to tripling their error: it
# replaces each channel's own forecast with a share of the top, and these channels are worth
# more than that. On a business where the leaves are small and noisy, the same table would
# read the other way round. Reconciliation is a modeling choice with a cost attached, not a
# formality applied at the end.
#
# ## Is any of this any good?
#
# A forecast that has never been scored against a baseline is a story. The backtester refits
# the model at rolling origins and scores what it did not see, and it will run anything that
# fits a series and forecasts a horizon, so a naive baseline sits in the same table as the
# Bayesian model.
#
# The cost is worth stating plainly: every fold refits everything from scratch, on purpose,
# because reusing a posterior across folds leaks the future into the past. Two folds of a
# one-series model is two NUTS runs.

# %%
from ergodic import backtest
from ergodic.forecast import SeasonalNaive, forecast_model

flat = timeseries(train_frame[["t", "total"]], time="t", freq="ME")
models = {
    "structural": forecast_model(name="structural", draws=300, warmup=300),
    "seasonal naive": SeasonalNaive(period=12),
}
reports = {
    name: backtest(model, flat, test_size=HOLDOUT, n_splits=2, seed=4, m=12)
    for name, model in models.items()
}
pd.DataFrame({name: report.summary().loc["total"] for name, report in reports.items()}).T[
    ["crps", "mae", "mase", "coverage"]
].round(3)

# %% [markdown]
# MASE is the column to read first. It scales the error by what last year's same month would
# have got you, so **below one means the model earned its keep** and above one means it did
# not. At 0.62 against the naive baseline's 1.52 this model is worth running, and the CRPS
# gap says the same thing about the whole distribution rather than just the middle.
#
# Coverage is the second column: the share of held-out points that landed inside the 90
# percent band. At 1.000 over twelve points the band is, if anything, generous here, and the
# baseline's 0.083 is the opposite failure, a band far too tight to believe. Coverage is the
# number that tells you whether to trust an interval, and a backtest is the only place it
# can be measured.
#
# The backtest also carries the sampler's own report, fold by fold, so a good-looking score
# that came out of a fit that barely converged cannot hide.

# %%
reports["structural"].diagnostics().round(3)

# %% [markdown]
# ## What is driving it
#
# So far nothing knows that promo moves retail. Telling it takes the pillar's **second**
# graph, and this one is causal: a window graph over lagged names, `promo[t] -> retail[t]`,
# exactly the object temporal discovery returns, so a discovered structure can be handed
# over with no translation.
#
# The promo calendar is planned, which makes it a known-future driver: it arrives through
# `scenario=` rather than being forecast. `scenario=` pins a path and makes **no causal
# claim**. It is honest for something exogenous or already decided, and it is quietly wrong
# for something the business sets in response to what it sees, which is why intervening on
# an endogenous node is refused by name here and waits for the causal layer.

# %%
from ergodic.forecast import schedule, window_graph

planned_promo = np.full(HOLDOUT, 12.0)  # the plan: hold spend at 12 for the next half year
retail_data = timeseries(train_frame[["t", "retail", "promo"]], time="t", freq="ME")
graph = window_graph([("promo", "retail", 0)])

print(schedule(retail_data, graph, scenario={"promo": planned_promo}, fit="staged"))

# %%
driven = forecast(
    retail_data,
    HOLDOUT,
    graph=graph,
    scenario={"promo": planned_promo},
    fit="staged",
    draws=300,
    warmup=300,
    seed=0,
)
driven

# %% [markdown]
# The schedule ran the driver first (there was nothing to fit: its path was given) and then
# retail, which read it. The fitted coefficient lives on the scaled series the sampler works
# in, so it is not directly comparable to the planted 2.5. The place where it becomes
# readable is the attribution, which puts every piece back in the units the data arrived in.
#
# ## The waterfall
#
# `explain()` takes the forecast apart. The default mechanism was *composed* from a trend, a
# season and one term per driver, so those pieces are still recoverable per posterior draw,
# and taking them apart again costs no sampling at all. Each bar carries its own credible
# interval, and the bars add up to the total.

# %%
from ergodic.forecast import Window

attribution = driven.explain(windows=[Window("next half year", 0, HOLDOUT)])
record = attribution["retail"]
print(record)
record.to_frame().round(2)

# %% [markdown]
# Read a bar as what that piece contributes **at its own level**, not as what a change in it
# would do. Promo sitting at 12 for six months is worth what the promo bar says, season
# included; the level and trend carry the rest. `closure_error` is reported always, and it
# is the check that the pieces really do rebuild the model's own mean path rather than
# approximately so.
#
# We planted the answer, so we can check it. Promo held at 12 with a coefficient of 2.5 is
# worth 30 units a month, and each of those months is scaled by its own seasonal factor:

# %%
horizon_season = retail_season[-HOLDOUT:]
planted = 2.5 * 12.0 * horizon_season.sum()
promo_bar = next(bar for bar in record.bars if bar.name == "promo")
low, high = promo_bar.credible_interval(0.9)
print(f"planted promo contribution: {planted:8.1f}")
print(f"attributed:                 {promo_bar.value:8.1f}   90% band ({low:.1f}, {high:.1f})")

# %% [markdown]
# The planted 185 sits inside the band, and the point estimate lands under it, at 146. That
# gap is worth understanding rather than explaining away. A wandering level and a driver
# with its own slow movement are competing to explain the same low-frequency swings in
# retail, and the level wins some of what belongs to the driver, so the coefficient comes
# back attenuated. More history, or a driver that moves on a rhythm the level cannot mimic,
# closes it. The interval is doing its job: it says how much of this we know.

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = record.bars
running = 0.0
for index, bar in enumerate(bars):
    low, high = bar.credible_interval(0.9)
    color = "#33415c" if bar.kind == "baseline" else ("#2a9d8f" if bar.value > 0 else "#d1495b")
    ax.bar(index, bar.value, bottom=running, color=color, width=0.62)
    ax.errorbar(
        index,
        running + bar.value,
        yerr=[[bar.value - low], [high - bar.value]],
        fmt="none",
        ecolor="#495867",
        capsize=4,
        lw=1.1,
    )
    running += bar.value
ax.bar(len(bars), running, color="#9aa5b1", width=0.62)
ax.set_xticks(range(len(bars) + 1))
ax.set_xticklabels([bar.name for bar in bars] + ["total"], fontsize=9)
ax.set_ylabel("retail units, next six months")
ax.set_title("what the forecast is made of")
fig.tight_layout()

# %% [markdown]
# The bars are cut in composition order: seasonality first, then the drivers. Asking the
# question the other way round, `method="loo"`, takes each factor out of the full model
# instead and books what no single factor owns as an interaction bar. On a multiplicative
# season that bar is real, because the season scales the driver's contribution too, and
# neither reading is more correct than the other. They answer different questions.

# %%
driven.explain(windows=[Window("next half year", 0, HOLDOUT)], method="loo")[
    "retail"
].to_frame().round(2)

# %% [markdown]
# ## What to watch
#
# Four things worth knowing before this goes anywhere near a plan.
#
# **Calibrate the band against a backtest rather than trusting the posterior.** On a series
# where the level wanders enough to swamp the slope, a twelve-step band from this model
# family measures around 75 percent coverage against a nominal 90, and that is a property of
# the model class rather than of the sampler. The backtest's `coverage` row is the honest
# reading, and it is cheap.
#
# **The damping comes back low.** The parameter that decides how fast a slope decays is
# recovered systematically below the truth, so a long extrapolation flattens sooner than it
# should. That is the conservative direction, which is why nothing screams, but a
# twelve-month plan built off a twelve-month extrapolation is understating the trend.
#
# **Staged fitting drops the correlation between series.** Fitting one node at a time and
# handing each posterior down the graph scales, and what it costs is the joint posterior
# between nodes: everything except what flows through the handoff. `fit="joint"` keeps it
# and costs one bigger sampler, which is why small graphs get it by default.
#
# **`scenario=` is an assumption, not an intervention.** Pinning a driver's future says "we
# expect this path" and trusts the fitted regression to carry it through. Saying "we will
# *set* this path" about something the business decides in response to demand is a different
# claim, and this pillar refuses to pretend otherwise: `do=` waits for identification on the
# window graph, and until then the distinction lives in the name.
#
# One practical footnote. `ergodic.forecast` is both a module and the function on it, and the
# function wins: `from ergodic import forecast` gives you something callable, and
# `from ergodic.forecast import backtest` reaches the module. Only `import ergodic.forecast
# as mod` does the surprising thing, binding the function.
#
# ## Where this goes next
#
# Every number in this guide came out of one object. The forecast, the levels agreeing, the
# honest score against a naive baseline, and the account of what is driving it are the same
# fitted model read four ways, which is the argument for building a forecast out of parts
# instead of out of a black box. The [forecasting guide](../forecast.md) has the rest of the
# surface: the mechanism zoo, the reconciliation menu in full, and the staged-versus-joint
# story.
