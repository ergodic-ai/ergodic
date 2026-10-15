# %% [markdown]
# # Causal forests and policy trees: from per-customer effects to a rule
#
# Guide four found a curve: a voucher that moves new customers and leaves veterans cold.
# This guide takes the next two steps a team actually needs. First, per-customer effects
# with error bars, from a model built for the job, so you know which estimates to lean on.
# Second, a rule simple enough to hand to the people who run the campaign: "offer it to
# customers with tenure above X", not a column of model scores.
#
# The world is a retention offer at a subscription business. It helps established
# customers and annoys new ones, so its effect changes sign. Customers who visit more
# respond more. And the offer wasn't randomized: the retention team sent it mostly to
# long-tenure customers, so tenure confounds the naive read. A third covariate, region,
# moves spend but has nothing to do with the effect. It's a decoy.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
n = 3000

tenure = rng.uniform(0.0, 6.0, size=n)  # years
visits = rng.normal(0.0, 1.0, size=n)  # standardized monthly visits
region = rng.normal(0.0, 1.0, size=n)  # moves spend, not the effect

# Historical targeting: long-tenure customers were more likely to get the offer.
p_offer = 1.0 / (1.0 + np.exp(-0.6 * (tenure - 3.0)))
offer = (rng.uniform(size=n) < p_offer).astype(float)

# The answer key: negative below about 2.5 years of tenure, positive above, plus visits.
tau = 4.0 * np.tanh(tenure - 2.5) + 1.0 * visits

spend = (
    30.0 + 5.0 * tenure + 3.0 * visits + 2.0 * region + tau * offer + rng.normal(0.0, 5.0, size=n)
)
df = pd.DataFrame(
    {"tenure": tenure, "visits": visits, "region": region, "offer": offer, "spend": spend}
)
print(f"true average effect: {tau.mean():.2f}")
print(f"share of customers the offer helps: {np.mean(tau > 0):.0%}")

# %% [markdown]
# The average effect is small and positive, and more than 40% of customers would be
# better off without the offer.
#
# ## An honest causal forest
#
# A causal forest is a random forest whose splits look for differences in the *effect*,
# not in the outcome. Before it grows, it removes what tenure, visits and region predict
# about spend and about who got the offer (cross-fitted nuisance models, as in guide
# three), so the confounding is handled by the same adjustment the graph asks for.
#
# It's also *honest*: each tree picks its splits on one half of its sample and estimates
# the effect in each leaf on the other half. A customer's outcome never both places a
# split and fills the leaf it created. That's what makes per-customer standard errors
# possible. The forest computes them with the infinitesimal jackknife, a formula over the
# trees, rather than by refitting on bootstrap samples.

# %%
from ergodic import CausalForest, dag, identify_effect, tabular

graph = dag(
    [
        "tenure -> offer",
        "tenure -> spend",
        "visits -> spend",
        "region -> spend",
        "offer -> spend",
    ]
)
data = tabular(df)
identification = identify_effect(graph, "offer", "spend")

forest = CausalForest(n_trees=400, min_leaf=15, random_state=0).fit(identification, data)
result = forest.cate()

effects = np.asarray(result.values)
print(f"band kind: {result.kind}")
print(f"correlation with the true effect: {np.corrcoef(effects, tau)[0, 1]:.3f}")
print(f"average of the per-customer effects: {effects.mean():.2f} (truth {tau.mean():.2f})")

# %% [markdown]
# The per-customer effects track the truth closely. If you only need the average, you
# don't need a forest: `estimate_effect` gives it with a standard interval.
# `result.ate()` gives one too: it averages the doubly robust scores cached on the
# result, never the per-customer standard errors, which share trees and so can't be
# combined as if independent.

# %%
from ergodic import estimate_effect

average = estimate_effect(graph, data, "offer", "spend")
lo, hi = average.interval()
print(f"estimate_effect: {average.value:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
from_scores = result.ate()
lo, hi = from_scores.interval()
print(f"result.ate():    {from_scores.value:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

# %% [markdown]
# Now the error bars. `result.interval(0.9)` returns a lower and an upper bound per
# customer. In a simulation you can check them against the truth, so do:

# %%
lower, upper = result.interval(0.9)
width = upper - lower
covered = (lower <= tau) & (tau <= upper)
print(f"90% bands that contain the true effect: {covered.mean():.0%}")
print(f"band width: median {np.median(width):.2f}, 95th percentile {np.quantile(width, 0.95):.2f}")
print(f"bands of zero width: {np.mean(width < 1e-9):.0%}")

# %% [markdown]
# Read this carefully. The bands cover the truth well under 90% of the time here, and some
# have zero width: with 400 trees for 3,000 customers the jackknife's correction for
# having finitely many trees outweighs the variance it corrects, and the estimate is
# clipped at zero (`forest.cate()` warns when more than 5% of the bands have zero width).
# More trees, at least as many as rows, shrink that share; the forest's
# smoothing can still leave the bands short where the effect changes steeply. So treat the
# forest's bands as a guide to *which* estimates are shaky (wider is shakier), not as a
# calibrated 90% promise. If you need calibrated per-customer intervals, check them on a
# simulation shaped like your data first, the way this cell does.
#
# The plot shows the estimates against the truth, by tenure, for a sample of customers:

# %%
import matplotlib.pyplot as plt

idx = rng.choice(n, size=300, replace=False)
order = np.argsort(tenure)

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    tenure[idx],
    effects[idx],
    yerr=[effects[idx] - lower[idx], upper[idx] - effects[idx]],
    fmt="o",
    ms=3,
    color="#9aa5b1",
    elinewidth=0.6,
    alpha=0.7,
    label="forest estimate, 90% band",
)
ax.plot(
    tenure[order],
    4.0 * np.tanh(tenure[order] - 2.5),
    color="#d1495b",
    lw=2,
    label="true effect at average visits",
)
ax.axhline(0.0, color="#33415c", lw=1)
ax.set_xlabel("tenure (years)")
ax.set_ylabel("effect of the offer on spend")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# The vertical scatter around the red curve is mostly real: the visits term moves each
# customer's effect up or down. Which covariates is the forest actually using?

# %%
importance = forest.variable_importance()
for name, value in importance.top():
    print(f"{name:<8} {value:.2f}")

# %% [markdown]
# Variable importance counts how often the forest splits on each covariate, with shallow
# splits weighted more. Tenure and visits lead. Region still gets a share, because a
# forest that searches for splits will find some in noise. Importance tells you where the
# forest looked, not proof that a covariate modifies the effect.
#
# ## From effects to a rule: the policy tree
#
# Three thousand per-customer numbers aren't a policy. A policy tree turns them into one:
# a shallow decision tree that says treat or don't treat, chosen to maximize the
# estimated value of the policy. It doesn't fit to the forest's point estimates. It reads
# the doubly robust scores cached on the result, per-customer effect estimates that are
# noisy one by one but unbiased on average, so the rule's value can be estimated honestly.
# At depth 2 or less it searches every split exhaustively, so the rule is the best one of
# its size, not a greedy approximation.

# %%
from ergodic import policy_tree

tree = policy_tree(result, ["tenure", "visits", "region"], depth=2)
print(tree.describe())

# %% [markdown]
# Read the bottom branch first: for nearly everyone it treats when tenure is above 2.63,
# close to the true flip point of 2.5 for a customer with average visits. Region never
# appears.
#
# The top split is less convincing. It peels off 81 customers with very low visits and
# treats 48 of them, the ones with the *very lowest* visits, which the true effect gives
# no reason to do. That's the exhaustive search fitting noise in a small corner: it finds
# the best rule for these scores, and with 81 customers the scores are noisy. A rule you
# hand to a team should survive dropping such a branch, so compare it with the best
# single split:

# %%
simple = policy_tree(result, ["tenure", "visits", "region"], depth=1)
print(simple.describe())
lo, hi = simple.interval(0.9)
print(f"depth 1 policy value {simple.value:.2f}   90% CI [{lo:.2f}, {hi:.2f}]")
print(f"depth 2 policy value {tree.value:.2f}")

# %% [markdown]
# The single split is the same tenure cut, and the extra branch adds about 0.06 to a value
# whose interval is about half a unit wide. Keep the one-line rule: treat customers
# with more than 2.63 years of tenure.
#
# ## What the rule is worth
#
# A policy tree reports its estimated value with a standard error, plus the two trivial
# policies for comparison. The value is the extra spend per customer the rule earns over
# treating no one, so treating no one scores 0 and treating everyone scores the average
# effect. These numbers come from the same customers the tree was chosen on, so they run
# a little high; `simple.evaluate(...)` scores the rule on customers it never saw.

# %%
lo, hi = simple.interval(0.9)
print(f"policy value          {simple.value:6.2f}   90% CI [{lo:.2f}, {hi:.2f}]")
print(f"treat everyone        {simple.treat_all_value:6.2f}")
print(f"treat no one          {simple.treat_none_value:6.2f}")
print(f"gain over the better trivial policy: {simple.regret('best_trivial'):.2f}")

chosen = simple.assign()
print(f"\nshare of customers the rule treats: {chosen.mean():.0%}")
print(f"estimated extra spend per customer vs treating no one: {simple.value:.2f}")

# GRADING ONLY: the same numbers from the true effects.
print(f"true extra spend per customer under the rule:          {np.mean(chosen * tau):.2f}")
print(f"true extra spend under the depth-2 tree:               {np.mean(tree.assign() * tau):.2f}")
print(f"true extra spend if everyone were treated:             {tau.mean():.2f}")
print(f"best possible (treat exactly those it helps):          {np.mean(np.maximum(tau, 0)):.2f}")

# %% [markdown]
# The estimated gain, 1.72 per customer, is close to the true 1.84, and the one-line rule
# gets nearly all of what a perfect oracle could (1.87). Graded against the truth, it
# even beats the depth-2 tree, which earns 1.80: the noise branch cost money.
# Blanket treatment earns a third as much, because it pays for every customer the offer
# annoys.
#
# ## What to take away
#
# A causal forest gives you per-customer effects that adjust for confounding, and bands
# that rank how sure each estimate is. Check those bands on a simulation before you read
# them as calibrated. A policy tree turns the result into a rule, scores it with the
# doubly robust value and a standard error, and compares it with the trivial policies, so
# the decision comes with its price attached.
#
# The guides go further: [heterogeneous effects](../../docsite/pages/heterogeneous-effects.md)
# compares the CATE learners, and [targeting a policy](../../docsite/pages/target-a-policy.md)
# covers RATE, the ranking test that tells you whether a targeting rule is worth building.
