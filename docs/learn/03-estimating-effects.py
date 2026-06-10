# %% [markdown]
# # Estimating effects: from a graph to a number
#
# A retention team flags customers at risk of churning and sends them a win-back voucher.
# A quarter later, finance compares spending and finds the voucher group spent far less than
# everyone else. The campaign looks like it burned money on people who shopped less anyway.
#
# By now you can see the trap without squinting: the voucher went to at-risk customers, and
# at-risk customers spend less. Risk confounds the comparison. Guide two told us what to do,
# close the back door, and in this world the back door runs through two measured variables,
# recency (days since the last order) and usage. This guide is about the step nobody teaches
# in the meeting: getting from "adjust for recency and usage" to a defensible number with an
# honest interval around it.
#
# Why is that a step at all? In guide one we stratified by hand: six tenure bands, compare
# inside each, done. With two confounders the same move needs a grid, six bands by six bands
# is 36 cells, and the data starts thinning out. With five confounders it's gone. Past the
# simplest cases, adjustment means fitting models, and the moment models enter, two new
# questions do too: which model, and what happens when it's wrong?

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
n = 12_000

recency = rng.uniform(0.0, 60.0, size=n)
usage = rng.normal(2.0, 1.0, size=n)

# The retention model scores risk from recency and usage, and risky customers get vouchers.
p_voucher = 1.0 / (1.0 + np.exp(-(0.08 * (recency - 30.0) - 0.7 * (usage - 2.0))))
voucher = (rng.uniform(size=n) < p_voucher).astype(float)

# Spending falls with recency, rises with usage, and the voucher adds exactly 25 dollars.
spend = 120.0 - 2.0 * recency + 30.0 * usage + 25.0 * voucher + rng.normal(0.0, 30.0, size=n)

df = pd.DataFrame({"recency": recency, "usage": usage, "voucher": voucher, "spend": spend})

naive = df[df.voucher == 1].spend.mean() - df[df.voucher == 0].spend.mean()
print(f"naive gap: {naive:+.1f}   truth: +25.0")

# %% [markdown]
# The naive gap is large and negative; the planted truth is plus 25. Same story as guide
# one, two confounders instead of one.
#
# ## Two models, and why the default uses both
#
# There are two different ways to use models to close a back door, and they fail in
# different places.
#
# The first models the outcome: fit spend as a function of recency, usage, and the voucher,
# then ask the fitted model two questions about every customer, what would you predict with
# the voucher, and without it, and average the difference. If the outcome model is right,
# this is exact. If it's wrong, say spending curves with recency and you fit a
# line, the answer inherits the error.
#
# The second models the treatment instead: fit the probability of getting a voucher from
# recency and usage (the propensity), then reweight customers so the voucher and no-voucher
# groups have the same mix of risk. Comparing the reweighted groups needs no model of
# spending at all. But if the propensity model is wrong, the reweighting is wrong.
#
# Each one alone is a single point of failure. The estimator ergodic picks by default, AIPW
# (augmented inverse propensity weighting), fits both and combines them so that the answer
# is right if either model is right. You only find out which model was wrong after the fact,
# if ever, so an estimator that gives you two chances instead of one is not a luxury.

# %%
from ergodic import dag, estimate_effect, tabular

g = dag(
    [
        "recency -> voucher",
        "usage -> voucher",
        "recency -> spend",
        "usage -> spend",
        "voucher -> spend",
    ]
)
data = tabular(df)

est = estimate_effect(g, data, "voucher", "spend")
est

# %% [markdown]
# The point estimate lands within a dollar of the truth, with an interval that covers it,
# and the rest of the line is just as important. The
# standard error measures how much the estimate would wobble across hypothetical reruns of
# the same world, and the 95% interval is the honest summary: a range of effects compatible
# with the data. If you reran this study many times, intervals built this way would cover
# the true effect in about 95% of runs. A point estimate without one is a guess wearing a
# suit.
#
# One call hides four steps worth knowing by name: the graph identified the effect (it found
# the adjustment set), the identification chose an estimator family, AIPW fit its two
# models, and the influence function gave the standard error analytically. Every step is
# inspectable, and every default is replaceable.
#
# ## Different roads, same place
#
# The single-model strategies are available by name, and on a world this polite (everything
# linear, both models correctly specified) they all land together:

# %%
rows = {}
for strategy in ["outcome_regression", "ipw", "aipw", "dml"]:
    e = estimate_effect(g, data, "voucher", "spend", strategy=strategy, random_state=0)
    lo, hi = e.interval()
    rows[strategy] = {"estimate": e.value, "low": lo, "high": hi}
pd.DataFrame(rows).T.round(2)

# %% [markdown]
# Two things to read off the table. The four roads agree, which is what a well specified
# world looks like; on real data, when the outcome-model answer and the propensity answer
# split, at least one model is misspecified, and you'd like to know that before acting. And
# pure weighting has the widest interval of the four. That's characteristic, not bad luck:
# IPW leans hardest on the customers with the most extreme treatment probabilities, where
# the weights blow up. The doubly robust estimators use the same propensity model without
# inheriting that fragility.
#
# ## When the world isn't linear
#
# Now break the politeness, in both models at once. Spending decays with recency along a
# curve, the way habits fade. And the retention team has gotten cleverer about
# targeting: vouchers go to the saveable middle, customers lapsed enough to need a nudge
# but not so lapsed they're gone, so treatment probability is a bump in recency, high in
# the middle and low at both ends. A straight line through a bump misses it everywhere.
# The voucher is still worth exactly 25.

# %%
rng2 = np.random.default_rng(31)
p_window = 0.05 + 0.80 * np.exp(-(((recency - 32.0) / 12.0) ** 2))
voucher2 = (rng2.uniform(size=n) < p_window).astype(float)
spend2 = (
    180.0 * np.exp(-recency / 20.0)
    + 15.0 * (usage - 2.0) ** 2
    + 25.0 * voucher2
    + rng2.normal(0.0, 20.0, size=n)
)
df2 = pd.DataFrame({"recency": recency, "usage": usage, "voucher": voucher2, "spend": spend2})
data2 = tabular(df2)

linear_fit = estimate_effect(g, data2, "voucher", "spend", strategy="aipw")
lo, hi = linear_fit.interval()
print(f"AIPW with linear learners: {linear_fit.value:+.1f}  CI [{lo:.1f}, {hi:.1f}]")

# %% [markdown]
# Off by a factor of nearly three, with a confident interval around the wrong answer. The
# default learners in the two slots are a linear regression and a logistic regression, both
# misspecified here, and being doubly robust doesn't help when both chances miss. The fix
# isn't a different estimator. It's better models in the slots, and the slots take any
# model with a sklearn surface:

# %%
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

flexible = estimate_effect(
    g,
    data2,
    "voucher",
    "spend",
    strategy="dml",
    learner={
        "outcome": GradientBoostingRegressor(random_state=0),
        "treatment": GradientBoostingClassifier(random_state=0),
    },
    random_state=0,
)
lo, hi = flexible.interval()
print(f"DML with gradient boosting: {flexible.value:+.1f}  CI [{lo:.1f}, {hi:.1f}]")

# %% [markdown]
# This is the honest reason machine learning belongs in causal inference. Not because a
# boosted tree knows what causes what (it doesn't; the graph carried all of that), but
# because once the graph says "adjust for recency and usage", something has to model those
# relationships, and flexible learners model curves that linear ones can't. DML (double
# machine learning) is built for exactly this pairing: it cross-fits, fitting each model on
# one part of the data and evaluating it on another, which is what keeps the interval valid
# with a flexible learner inside.
#
# ## Which average? ATE and ATT
#
# "The effect of the voucher" is several questions. The ATE (average treatment
# effect) asks: if we gave the voucher to everyone, what's the average gain? The ATT
# (average effect on the treated) asks: among the customers we treated, what did
# the voucher do? For the ROI of the campaign you ran, the ATT is the question.

# %%
att = estimate_effect(g, data, "voucher", "spend", target="att")
print(f"ATE {estimate_effect(g, data, 'voucher', 'spend').value:+.1f}   ATT {att.value:+.1f}")

# %% [markdown]
# Here they land together, because we built a world where the voucher is worth 25 to
# everyone. Real effects vary across people, the two averages split, and which one you want
# depends on the decision. That's the whole subject of the
# [next guide](04-heterogeneous-effects.py).
#
# ## The picture

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3.6))
labels = [
    "naive gap",
    "AIPW (linear world)",
    "AIPW linear\n(curved world)",
    "DML boosted\n(curved world)",
]
values = [naive, est.value, linear_fit.value, flexible.value]
intervals = [None, est.interval(), linear_fit.interval(), flexible.interval()]
for i, (value, ci) in enumerate(zip(values, intervals, strict=True)):
    ax.scatter([i], [value], color="#33415c", zorder=3)
    if ci is not None:
        ax.plot([i, i], list(ci), color="#33415c", lw=2)
ax.axhline(25.0, color="#d1495b", ls="--", lw=1.2, label="truth (+25)")
ax.set_xticks(range(len(labels)), labels, fontsize=9)
ax.set_ylabel("estimated effect (dollars)")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# ## What's still assumed
#
# Worth saying out loud, because no estimator on this page checks it for you. The graph has
# to be right: every estimate above trusted that recency and usage close the back door, and
# nothing in the data certifies that. The confounders have to be measured, or no model in
# any slot can adjust for them; that failure mode gets [its own guide](05-hidden-confounding.py).
# And the groups have to overlap: if customers with risk near one are always treated, there's
# nobody to compare them to, and the estimate for them rests on extrapolation. Estimation is
# the easy third of the problem. It's just the only third with error bars.
#
# Next: [who responds?](04-heterogeneous-effects.py) The 25 in this guide was the same for
# every customer, and that's the assumption real campaigns most enjoy violating.
