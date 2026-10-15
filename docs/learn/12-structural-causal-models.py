# %% [markdown]
# # Structural causal models: one fit, three kinds of question
#
# A subscription business gives some customers a discount at renewal time. The account
# team decides how big, and they give bigger discounts to the customers they think are
# most engaged. Three people now ask three different questions about the same data.
#
# The analyst asks what the data shows: customers with bigger discounts spend more, by how
# much? The pricing lead asks what a policy would do: if we *set* everyone's discount one
# dollar higher, how much more would they spend? And the account manager asks about one
# person: this customer got 5.25 dollars off and spent 56.85, what would they have spent with
# two dollars more?
#
# These are the three rungs of the causal ladder: seeing, doing, and imagining. Earlier
# guides answered the first two with an estimator per question. A structural causal model
# (SCM) answers all three from one fitted object, because it models every variable as an
# equation of its causes plus its own noise. This guide fits one, climbs the ladder, and
# then removes a column to show what the model does when the data can't answer.
#
# The answer key goes into the simulation first.

# %%
import logging

import numpy as np
import pandas as pd

# PyMC logs every sampler run; this keeps the notebook's output readable.
for name in ("pymc", "pymc.sampling", "pymc.stats.convergence"):
    logging.getLogger(name).setLevel(logging.ERROR)

rng = np.random.default_rng(11)
n = 1000

engagement = rng.normal(0.0, 1.0, size=n)
# The account team gives bigger discounts to engaged customers: the confounding.
discount = 5.0 + 2.0 * engagement + rng.normal(0.0, 1.5, size=n)
# The answer key: each extra dollar of discount adds 1.5 dollars of spend.
spend = 50.0 + 1.5 * discount + 8.0 * engagement + rng.normal(0.0, 4.0, size=n)
# Renewal is a yes or no, pushed up by discount and engagement.
logit = -0.5 + 0.35 * (discount - 5.0) + 1.2 * engagement
renewed = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

df = pd.DataFrame(
    {"engagement": engagement, "discount": discount, "spend": spend, "renewed": renewed}
)
df.head(3)

# %% [markdown]
# Engagement drives both the discount and the spend, so it's a confounder. The true effect
# of one more dollar of discount on spend is **1.5**. Renewal is a binary outcome with its
# own logistic response.
#
# ## Fitting the model
#
# An SCM is a graph plus one fitted equation per node. You give it the graph and the data,
# and it picks an equation for each node from the column's type: a Bayesian linear
# equation with Gaussian noise for a continuous column, and a Gumbel-max equation (a
# logistic model with explicit noise) for a binary or categorical one. Telling `tabular`
# that `renewed` is categorical says so explicitly instead of letting it guess.
#
# Fitting runs PyMC once per node. It needs the `scm` extra (`pip install "ergodic[scm]"`).

# %%
from ergodic import dag, tabular
from ergodic.scm import SCM

data = tabular(df, types={"renewed": "categorical"})
graph = dag(
    [
        "engagement -> discount",
        "engagement -> spend",
        "discount -> spend",
        "engagement -> renewed",
        "discount -> renewed",
    ]
)

scm = SCM(graph, data, draws=500, tune=500, chains=1, random_state=0).fit()
{node: type(mechanism).__name__ for node, mechanism in scm.mechanisms.items()}

# %% [markdown]
# ## Rung one and rung two: seeing and doing
#
# The data alone says something about discount and spend: regress one on the other and
# read the slope. That's the analyst's number. The pricing lead's number is different:
# `interventional` sets the discount for everyone with `do`, which cuts the arrow from
# engagement into discount, and reads the spend that follows.

# %%
seen = np.polyfit(df["discount"], df["spend"], 1)[0]

at_6 = scm.interventional("spend", {"discount": 6.0})
at_5 = scm.interventional("spend", {"discount": 5.0})

print(f"seeing: slope of spend on discount   {seen:.2f}")
print(f"doing:  spend at do(discount=6) minus do(discount=5)   {at_6.mean() - at_5.mean():.2f}")
print("truth:  1.50")

# %% [markdown]
# The observed slope is about 4. Engaged customers get bigger discounts *and* spend more,
# so the slope mixes the discount's effect with engagement's. Intervening severs that link
# and lands on 1.5.
#
# For the average effect you'd normally call `ate`. It does something the raw contrast
# doesn't: it asks the identification layer first, and only computes a number if the graph
# says the data can determine one. Here the confounder is measured, so it can.

# %%
ate = scm.ate("discount", "spend", levels=(5.0, 6.0))
lo, hi = ate.interval(0.95)
print(ate)
print(f"identified: {ate.identified}, 95% credible interval [{lo:.2f}, {hi:.2f}]")

# %% [markdown]
# The result is an `SCMResult`: posterior draws of the effect, with a credible interval
# that covers 1.5. Hold on to that type name, because it changes in the last section.
#
# ## Rung three: imagining
#
# The account manager's question is about one customer, not a population. What would
# *this* customer have spent with two more dollars of discount? Answering needs the
# customer's own noise: whatever made them spend what they spent, beyond what discount
# and engagement explain. The SCM recovers it from the customer's observed row (the
# step is called abduction), then replays the same customer with the discount changed.
# That's the twin network: two worlds that share one person's noise.

# %%
customer = data.take([0])
row = df.iloc[0]
print(f"factual: discount {row['discount']:.2f}, spend {row['spend']:.2f}")

what_if = scm.counterfactual(customer, {"discount": row["discount"] + 2.0}, "spend")
lo, hi = what_if.interval(0.95)
print(f"had the discount been {row['discount'] + 2.0:.2f}: spend {what_if.mean():.2f}")
print(f"95% credible interval [{lo:.2f}, {hi:.2f}]")
print(f"change for this customer: {what_if.mean() - row['spend']:.2f}")

# %% [markdown]
# Two extra dollars move this customer's spend by about 3, twice the per-dollar effect.
# With a linear equation that's exactly what you'd expect: the noise cancels, so every
# customer's counterfactual change is the slope times the change. The interval comes from
# uncertainty in the fitted slope. Pass `use_draws=False` for the single point.
#
# ## Per-customer effects with `ite`
#
# `ite` runs that twin-network contrast for every customer at once. With `levels=(5, 6)`
# it compares each customer's outcome at a discount of 5 and at 6, both worlds sharing
# that customer's noise.

# %%
ite_spend = scm.ite(data, "discount", "spend", levels=(5.0, 6.0))
values = np.asarray(ite_spend.values)
print(f"spend: per-customer effects run from {values.min():.3f} to {values.max():.3f}")

# %% [markdown]
# Every customer gets the same number. That isn't a bug: a linear equation with additive
# noise has one slope, so the individual effect is the slope for everyone. Per-customer
# spread appears when the equation is nonlinear, and renewal is. Compare a 3-dollar
# discount with an 8-dollar one:

# %%
renew_at_8 = scm.interventional("renewed", {"discount": 8.0})
renew_at_3 = scm.interventional("renewed", {"discount": 3.0})
print("population, do(discount=3):", renew_at_3)
print("population, do(discount=8):", renew_at_8)

ite_renew = scm.ite(data, "discount", "renewed", levels=(3.0, 8.0))
flips = np.asarray(ite_renew.values)
outcomes, counts = np.unique(flips, return_counts=True)
print("per-customer change in renewal:", dict(zip(outcomes.tolist(), counts.tolist(), strict=True)))
print(f"share of customers the bigger discount wins over: {np.mean(flips == 1):.1%}")

# %% [markdown]
# The population view says the renewal rate rises with the bigger discount. The individual
# view says *who*: each customer's effect is a flip, 1 if they renew only under the bigger
# discount, 0 if their answer doesn't change. About a quarter of customers flip, which
# matches the rise in the population rate from 30% to 55%, but now you can list them.
# That list is what the account manager can act on.
#
# A caution about what this per-customer answer rests on. It's exact *given the model*:
# the graph, the equation types, and the assumption that each customer's noise stays the
# same across the two worlds. None of that is testable from the data, so a counterfactual
# is only as good as the structural story behind it.
#
# ## When the data can't answer
#
# Now take engagement away. It's the most common situation in practice: the account team
# knew who was engaged, and nobody wrote it down. The graph still says engagement drives
# both discount and spend; mark it `latent` and fit on the two columns that remain.

# %%
hidden = dag(
    ["engagement -> discount", "engagement -> spend", "discount -> spend"],
    latent=["engagement"],
)
scm_hidden = SCM(
    hidden, tabular(df[["discount", "spend"]]), draws=500, tune=500, chains=1, random_state=0
).fit()

answer = scm_hidden.ate("discount", "spend")
print(type(answer).__name__)
print(answer)

# %% [markdown]
# The same `ate` call returns a `Sensitivity`, not an `SCMResult`. With engagement
# unmeasured, no adjustment set exists and the effect isn't a function of the data.
# Plenty of tools would still hand you a number with a tight interval. This object has no
# `.value` at all, so you can't read a point off it by accident.
#
# What it gives you instead is the effect as a function of how much confounding you
# assume. Here `c` is the share of the discount-spend covariance you attribute to the
# hidden confounder. With `c = 0` you're assuming no confounding, and you get the
# observed slope back.

# %%
cov_xy = float(np.cov(df["discount"], df["spend"])[0, 1])
for share in (0.0, 0.25, 0.5, 0.75, 1.0):
    c = share * cov_xy
    print(f"confounding explains {share:4.0%} of the covariance -> effect {answer.at(c):5.2f}")
low, high = answer.range()
print(f"swept range [{low:.2f}, {high:.2f}]")
print(f"effect if no confounding {answer.identified_at_zero:.2f}")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(answer.confounding, answer.effect, color="#33415c", lw=2, label="effect implied by c")
ax.axhline(1.5, color="#d1495b", ls="--", lw=1.2, label="true effect (from the code)")
ax.axhline(0.0, color="#9aa5b1", lw=1)
ax.set_xlabel("assumed confounding c (covariance attributed to engagement)")
ax.set_ylabel("effect of one dollar of discount")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# Read the curve as a decision tool. The truth sits on it, at the confounding strength the
# data can't reveal. The effect stays positive until hidden engagement explains *all* of
# the covariance between discount and spend. So "discounts raise spend" survives any
# confounding short of total, while "each dollar is worth 4" depends entirely on assuming
# none. If your team believes engagement explains at least half of the association, the
# curve tells you the effect is at most about 2.
#
# ## What to take away
#
# One fitted SCM answers population questions with `interventional` and `ate`, and
# individual ones with `counterfactual` and `ite`. Before it answers an average effect,
# it checks identification, and the return type tells you which world you're in: an
# `SCMResult` when the data determines the answer, a `Sensitivity` when it doesn't.
#
# The SCM guides go further: [fitting an SCM](../../docsite/pages/fit-an-scm.md) covers
# mechanisms and latent confounders, and
# [interventions and counterfactuals](../../docsite/pages/interventions-and-counterfactuals.md)
# covers the query surface. The [next guide](13-hidden-regimes.py) turns to a different
# kind of hidden variable: a switch that changes how a series behaves.
