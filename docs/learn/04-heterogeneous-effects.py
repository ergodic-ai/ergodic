# %% [markdown]
# # Who responds? Heterogeneous effects
#
# This time the team did everything right. The voucher went out as a proper A/B test, a coin
# flip per customer, so there's no confounding to untangle and nothing from guides one
# through three to fix. The test reads out at less than 9 dollars per customer. Serving the
# voucher costs 10. The campaign loses money, the experiment is airtight, case closed.
#
# Except that the average customer doesn't exist. A 9-dollar average is consistent with
# every customer gaining 9, and equally consistent with a third of customers gaining 25
# while the rest gain nothing. Same ATE, opposite decisions: in the second world you don't kill
# the campaign, you stop wasting it on people it can't move. Randomization settles who got
# the voucher. It says nothing about who responds to it.
#
# We'll build the second world on purpose, watch the A/B verdict condemn it, and then
# recover the per-customer effect that turns it profitable.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
n = 16_000

tenure = rng.uniform(0.0, 8.0, size=n)
activity = rng.normal(0.0, 1.0, size=n)

# A fair coin per customer: a real A/B test.
voucher = (rng.uniform(size=n) < 0.5).astype(float)

# The answer key, now a curve: new customers respond strongly, veterans barely at all.
tau = 28.0 * np.exp(-tenure / 2.5)

spend = 40.0 + 10.0 * tenure + 8.0 * activity + tau * voucher + rng.normal(0.0, 15.0, size=n)
df = pd.DataFrame({"tenure": tenure, "activity": activity, "voucher": voucher, "spend": spend})

print(f"true ATE: {tau.mean():.2f}")
print(f"true effect at tenure 0: {tau.max():.1f}, at tenure 8: {tau.min():.1f}")

# %% [markdown]
# The effect is 28 dollars for a brand-new customer and roughly nothing for an eight-year
# veteran, averaging out a bit above 8. First, the standard analysis, exactly as in guide
# three:

# %%
from ergodic import dag, estimate_effect, identify_effect, tabular

g = dag(["voucher -> spend", "tenure -> spend", "activity -> spend"])
data = tabular(df)

ate = estimate_effect(g, data, "voucher", "spend")
lo, hi = ate.interval()
print(f"estimated ATE: {ate.value:.2f}  CI [{lo:.2f}, {hi:.2f}]   serving cost: 10.00")

# %% [markdown]
# The interval sits well below the 10-dollar cost. As an average-versus-average decision,
# this campaign is dead, and no amount of better estimation changes that, because the
# estimate isn't wrong. The question is.
#
# ## From one effect to a function
#
# The conditional average treatment effect, CATE, is the effect as a function of who:
# CATE(tenure, activity) instead of one number. The estimator we'll use, the DR-learner,
# does something pleasingly direct. Cross-fitting the same two nuisance models AIPW uses,
# it builds a rough per-customer effect guess, noisy for any one person but centered on the
# right answer on average, and then fits a plain predictive model from each customer's
# features to that guess, like any supervised problem. The final regression is an ordinary
# model in an ordinary slot, so you choose how flexible the effect surface can be. The truth
# is a curve, so give the final slot a model that can bend:

# %%
from sklearn.ensemble import GradientBoostingRegressor

from ergodic.inference import DRLearner

model = DRLearner(
    {"final": GradientBoostingRegressor(random_state=0)},
    random_state=0,
).fit(identify_effect(g, "voucher", "spend"), data)

# .cate() returns a CATEResult (per-unit points + uncertainty + .explain()); this guide reads
# only the effect curve, so uncertainty=False skips the per-unit bootstrap, and .values is the
# raw per-customer effect array the arithmetic and plots below treat like any vector.
cate = model.cate(uncertainty=False).values
print(f"average of the per-customer effects: {cate.mean():.2f}")

# %% [markdown]
# The average of the per-customer effects recovers the ATE, as it must. The point of having
# them is the shape:

# %%
import matplotlib.pyplot as plt

order = np.argsort(tenure)
idx = rng.choice(n, size=2000, replace=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(tenure[idx], cate[idx], s=6, alpha=0.35, color="#9aa5b1", label="estimated CATE")
ax.plot(tenure[order], tau[order], color="#d1495b", lw=2, label="true effect (from the code)")
ax.axhline(10.0, color="#33415c", ls="--", lw=1.2, label="serving cost")
ax.set_xlabel("tenure (years)")
ax.set_ylabel("effect of the voucher (dollars)")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# The cloud tracks the curve. Don't read individual dots: a single customer's effect is not
# observable even in principle (you never see the same customer both treated and untreated),
# so per-customer estimates are noisy by nature and need far more data than an average does.
# Read the shape, and where it crosses the cost line. Everything to the left of the crossing
# is profit; everything to the right is spend on customers the voucher can't move.
#
# ## The targeting arithmetic
#
# Turn the curve into a policy: serve the voucher only where the predicted effect clears the
# serving cost. Because this is a simulation, we can grade any policy against the true tau,
# a luxury real campaigns don't get.

# %%
cost = 10.0
policies = {
    "treat nobody": np.zeros(n, dtype=bool),
    "treat everyone": np.ones(n, dtype=bool),
    "treat where predicted CATE > cost": cate > cost,
}
rows = {}
for name, chosen in policies.items():
    rows[name] = {
        "customers treated": int(chosen.sum()),
        "profit (true tau - cost, dollars)": float(((tau - cost) * chosen).sum()),
    }
pd.DataFrame(rows).T

# %% [markdown]
# Treating everyone loses about as much as the ATE said it would. Treating the predicted
# responders, roughly the newest third of the base, turns the same campaign into clear
# profit, with the same voucher, the same creative, and the same experiment behind it. The
# ATE compared two policies, all or nothing, and both were the wrong policy.
#
# The standard picture for this is the profit curve: rank customers by predicted effect,
# treat the top fraction, and plot the realized profit as the fraction grows.

# %%
ranked = np.argsort(-cate)
cumulative = np.cumsum(tau[ranked] - cost)
fraction = np.arange(1, n + 1) / n

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(fraction, cumulative, color="#33415c", lw=2)
ax.axhline(0.0, color="#9aa5b1", lw=1)
best = int(np.argmax(cumulative))
ax.scatter(
    [fraction[best]],
    [cumulative[best]],
    color="#d1495b",
    zorder=3,
    label=f"peak: treat top {fraction[best]:.0%}",
)
ax.set_xlabel("fraction of customers treated, best predicted first")
ax.set_ylabel("campaign profit (dollars)")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# The curve rises while the voucher reaches people it moves, peaks near the third of the
# base where the true effect crosses the cost, and then pays the serving cost for nothing
# all the way down to the treat-everyone loss at the right edge. Every targeting decision a
# marketing team debates is a point on this curve.
#
# ## What heterogeneity costs you
#
# Heterogeneity is a modeling claim, so the honesty bar is higher than for an ATE. The
# pseudo-effects are unbiased but loud, and a flexible final model can hallucinate structure in
# them; held-out validation and stability checks matter more here, not less. Per-customer
# uncertainty exists but is wide; the
# [heterogeneous effects guide](../../docsite/pages/heterogeneous-effects.md) shows how to get a
# per-customer interval when you need one. And the covariates you can target on are the
# covariates you fed the model: CATE(tenure) is silent about heterogeneity in anything you didn't
# measure.
#
# So far every confounder has been a column in the dataframe. The
# [next guide](05-hidden-confounding.py) takes that away.
