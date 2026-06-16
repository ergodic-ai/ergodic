# %% [markdown]
# # Customer lifetime value: spend where it moves
#
# A growth team at a subscription brand has a lifetime-value model, and it's a good one.
# Feed it a customer's recency, frequency, tenure, and basket size and it predicts the two-year
# margin that customer will throw off, and it predicts it well. The team uses the model the
# obvious way. Rank everyone by predicted value, then pour the retention budget, a concierge
# onboarding perk that costs 25 dollars a head, into the top of the list. They also check the
# perk's record: customers who got it are worth about 120 dollars more over two years. Expand it.
#
# Two of those moves are wrong, and they're wrong in different ways. The first is a number
# everyone in causal inference has seen by now, the perk's record is confounded. The second is
# subtler and more expensive: the value a model can predict is not the value a perk can move,
# and ranking by the first is the wrong way to spend the second.
#
# The other case studies on this page run on real datasets, where a published number is the
# answer key. This one can't, because the question "what does the perk do" has no public dataset
# with a measured answer to grade against. So we borrow the move from the
# [learn series](../learn/01-why-causality.py): build the company in a simulation, write the
# true effect into the code, and check the methods against it.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n = 18_000

# Two customer traits, drawn independently on purpose.
engagement = rng.normal(0.0, 1.0, size=n)  # drives baseline value AND who the team perks
headroom = rng.normal(0.0, 1.0, size=n)  # drives how much the perk can move a customer
tenure = rng.uniform(0.0, 1.0, size=n)

# Account managers fight to give the perk to engaged, established customers.
p_perk = 1.0 / (1.0 + np.exp(-(1.7 * engagement + 0.8 * tenure - 0.4)))
perk = (rng.uniform(size=n) < p_perk).astype(float)

# The answer key. The perk's effect is keyed to headroom, which is independent of value:
# customers with room to grow respond, the already-maxed and the near-gone barely move.
tau = np.clip(10.0 + 45.0 * headroom, 0.0, None)

# Two-year margin: a baseline set by engagement and tenure, plus the perk's effect on the perked.
value = 120.0 + 85.0 * engagement + 35.0 * tenure + tau * perk + rng.normal(0.0, 30.0, size=n)
cost = 25.0

df = pd.DataFrame(
    {"engagement": engagement, "headroom": headroom, "tenure": tenure, "perk": perk, "value": value}
)
print(f"customers perked     : {perk.mean():.1%}")
print(f"true average effect  : ${tau.mean():.2f} per customer   (serving cost ${cost:.2f})")

# %% [markdown]
# Read the world off the code. Engagement does two jobs: it lifts a customer's baseline value,
# and it raises the odds the team hands them the perk. That double role is what makes it a
# confounder. The perk's true effect lives in a separate variable, headroom, which has nothing to
# do with how valuable a customer already is. Average that effect over everyone and it comes to
# about 23 dollars, below the 25 it costs to serve. On average, the perk loses money. Hold that
# thought, because it isn't the end of the story.
#
# ## The record looks great, and the record is confounded
#
# Compare perked customers to the rest, the way the team did:

# %%
gap = value[perk == 1].mean() - value[perk == 0].mean()
print(f"perked minus not-perked: ${gap:,.2f}")

# %% [markdown]
# About 120 dollars, more than five times the truth we just wrote down. Nothing is wrong with the
# arithmetic. What's wrong is the comparison. The perk went to engaged, established customers, and
# those customers were worth more before any perk touched them. So "perked vs not" is mostly
# "engaged vs not" in a costume, and the 120 is the perk's real effect plus the head start of the
# people who got it.
#
# The fix is the back door, the same one from [why causality](../learn/01-why-causality.py). Draw
# what drives what: engagement and tenure point at both the perk and value, and that's the path
# association leaks through.

# %%
from ergodic import dag, identify_effect

g = dag(
    [
        "engagement -> perk",
        "tenure -> perk",
        "engagement -> value",
        "tenure -> value",
        "headroom -> value",
        "perk -> value",
    ]
)
identify_effect(g, "perk", "value").adjustment_set

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

POS = {
    "engagement": (0.6, 1.4),
    "tenure": (1.8, 1.4),
    "headroom": (3.2, 1.4),
    "perk": (0.6, 0.0),
    "value": (3.2, 0.0),
}


def draw_dag(graph, pos, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for edge in graph.edges:
        u, _, v = edge.glyph().split()
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle="-|>",
                mutation_scale=16,
                shrinkA=26,
                shrinkB=26,
                lw=1.5,
                color="#33415c",
            )
        )
    for name, (x, y) in pos.items():
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#eef1f6", "ec": "#33415c"},
        )
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylim(-0.5, 1.9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()


draw_dag(g, POS, "What drives the perk, and what drives value")

# %% [markdown]
# The adjustment set comes back as engagement, tenure, and headroom. The first two close the back
# door. Headroom rides along because it predicts value, and a valid adjustment set is free to carry
# outcome predictors, which sharpens the estimate and, in a moment, lets the same fit read the
# effect per customer. With the set in hand, one call estimates the effect, the doubly robust
# estimator with flexible learners in both slots:

# %%
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ergodic import estimate_effect, tabular

data = tabular(df)
ate = estimate_effect(
    g,
    data,
    "perk",
    "value",
    strategy="aipw",
    learner={
        "outcome": GradientBoostingRegressor(random_state=0),
        "propensity": GradientBoostingClassifier(random_state=0),
    },
    random_state=0,
)
lo, hi = ate.interval()
print(f"adjusted effect: ${ate.value:.2f}   CI [{lo:.1f}, {hi:.1f}]   truth ${tau.mean():.2f}")

# %% [markdown]
# The adjusted effect lands at 23.65 dollars, interval 22.4 to 24.9, right on the planted 23.39. The
# confounding is gone, and the honest read of the average perk is now in front of us: it costs 25
# and returns 23, so rolled out to everyone it loses a little money on every customer. A team that
# stopped here would kill the program. That would be a mistake, and the reason is the second error.
#
# ## Predicted value is not movable value
#
# The team ranks customers by predicted lifetime value, because that's the model they have. Build
# it: a gradient-boosted model of value on the features a growth team actually has at decision time.

# %%
predicted_clv = (
    GradientBoostingRegressor(random_state=0)
    .fit(df[["engagement", "tenure"]], df["value"])
    .predict(df[["engagement", "tenure"]])
)
print(
    f"predicted-value model, in-sample correlation with true value: "
    f"{np.corrcoef(predicted_clv, value)[0, 1]:.2f}"
)

# %% [markdown]
# It predicts value well. Now the question that decides the budget: does a high predicted value mean
# a high response to the perk? The simulation knows each customer's true effect, so we can
# look directly.

# %%
print(f"corr(predicted value, true perk effect): {np.corrcoef(predicted_clv, tau)[0, 1]:+.3f}")

# %% [markdown]
# Essentially zero. The thing the team ranks on and the thing that should decide the budget are
# unrelated, and they're unrelated by construction, because value rides on engagement and the
# perk's effect rides on headroom. The whales the model loves are valuable and immovable. The
# customers the perk can actually grow are scattered through the middle of the value distribution,
# invisible to a ranking built on value.
#
# To target on the effect you have to estimate the effect per customer. That's the CATE, and the
# DR-learner gives it: it cross-fits the same two nuisance models the doubly robust estimator used,
# builds a per-customer pseudo-effect, and regresses it on the covariates. The estimated effect is a
# column, one number per customer.

# %%
from ergodic.inference import DRLearner

cate = (
    DRLearner(
        {
            "outcome": GradientBoostingRegressor(random_state=0),
            "propensity": GradientBoostingClassifier(random_state=0),
            "final": GradientBoostingRegressor(random_state=0),
        },
        random_state=0,
    )
    .fit(identify_effect(g, "perk", "value"), data)
    .cate()
)
print(f"average of per-customer effects: ${cate.mean():.2f}   (recovers the ATE)")
print(f"corr(estimated effect, true effect): {np.corrcoef(cate, tau)[0, 1]:.2f}")

# %% [markdown]
# The estimated effects average to the ATE, as they must, and they track the true per-customer
# effect at a correlation of 0.79. Now put the two rankings side by side. Each dot is a customer,
# predicted value on the x-axis, estimated perk effect on the y-axis.

# %%
plot_rng = np.random.default_rng(0)
idx = plot_rng.choice(n, size=2500, replace=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(predicted_clv[idx], cate[idx], s=7, alpha=0.35, color="#9aa5b1")
ax.axhline(cost, color="#d1495b", ls="--", lw=1.2, label=f"serving cost ${cost:.0f}")
ax.set_xlabel("predicted lifetime value (dollars)")
ax.set_ylabel("estimated perk effect (dollars)")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# The cloud is flat. Slide along the x-axis, from the customers the value model ranks lowest to the
# whales it ranks highest, and the estimated perk effect doesn't trend with it. Plenty of
# high-value customers sit below the cost line, where the perk is wasted, and plenty of mid-value
# customers sit well above it, where it pays. The vertical spread at every value level is the
# heterogeneity the value ranking can't see.
#
# ## The targeting arithmetic
#
# Turn the picture into money. Fix a budget, say enough to perk 30 percent of the base, and compare
# three ways to spend it. Treat everyone, which we already know loses. Treat the top 30 percent by
# predicted value, the current playbook. Treat the top 30 percent by estimated effect. Because this
# is a simulation, grade each policy against the true effect.

# %%
budget = int(0.30 * n)


def incremental(chosen):
    return float(((tau - cost) * chosen).sum())


def top_mask(score, k):
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(-score)[:k]] = True
    return mask


policies = {
    "treat everyone": np.ones(n, dtype=bool),
    "top 30% by predicted value": top_mask(predicted_clv, budget),
    "top 30% by estimated effect": top_mask(cate, budget),
}
pd.DataFrame(
    {
        name: {"customers treated": int(c.sum()), "incremental margin (USD)": incremental(c)}
        for name, c in policies.items()
    }
).T.round(0)

# %% [markdown]
# Targeting by predicted value still loses money, about 5,800 dollars, because the budget lands on
# valuable customers who would not have moved. Targeting by estimated effect, the same perk and the
# same 30 percent of the base, makes about 198,000 dollars.
# The program the average effect said to kill is a clear winner once the budget follows the effect
# instead of the value.
#
# The whole story is one figure. Rank customers best-first under each policy and plot the running
# margin as the budget grows.

# %%
fraction = np.arange(1, n + 1) / n
curve_effect = np.cumsum((tau - cost)[np.argsort(-cate)])
curve_value = np.cumsum((tau - cost)[np.argsort(-predicted_clv)])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(fraction, curve_effect, color="#33415c", lw=2, label="rank by estimated effect")
ax.plot(fraction, curve_value, color="#d1495b", lw=2, label="rank by predicted value")
ax.axhline(0.0, color="#9aa5b1", lw=1)
peak = int(np.argmax(curve_effect))
ax.scatter([fraction[peak]], [curve_effect[peak]], color="#33415c", zorder=3)
ax.set_xlabel("fraction of customers perked, best first")
ax.set_ylabel("cumulative margin (dollars)")
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# The effect-ranked curve climbs while the perk reaches customers it moves, peaks near the third of
# the base whose effect clears the cost, then declines as it starts paying to perk people it can't
# move. The value-ranked curve barely leaves zero and slides under, because predicted value carries
# no information about who responds. Every targeting decision the growth team argues over is a point
# on the dark curve, and the light one is the plan they walked in with.
#
# ## The placebo
#
# Every number here leans on one sentence: the perk is as good as random once engagement and tenure
# are held fixed. The data can't certify that. What it can check is whether the pipeline invents an
# effect where none exists. Scramble the perk so it points at nothing and re-estimate. A sound
# pipeline reports roughly zero.

# %%
from ergodic import refute_effect

ref = refute_effect(
    g,
    data,
    "perk",
    "value",
    method="placebo_treatment",
    strategy="aipw",
    estimand_value=ate.value,
    n_simulations=12,
    random_state=0,
)
print(f"real effect    : ${ref.estimand_value:.2f}")
print(f"placebo (mean) : ${ref.refuted_value:.2f}   p = {ref.p_value:.3f}   passed: {ref.passed}")

# %% [markdown]
# Scrambled, the effect collapses to roughly zero, far from the real 23.65, and the placebo runs
# almost never reach it. That rules out a leak firing on noise. It says nothing about an unmeasured
# trait, some loyalty the columns don't capture, that pushes a customer toward both the perk and a
# fat balance.
# If that exists, the back door is not fully closed and the adjusted effect drifts back toward the
# naive 120. The graph is an assumption, and the placebo tests the machinery, not the assumption.
#
# What survives is the lesson the two figures carry. The average effect decides whether a program
# can pay at all; here it said no. The per-customer effect decides who to spend on, and it turned
# the same program profitable. Predicting which customers are valuable is a different job from
# estimating which customers a perk can move, and a lifetime-value model, however good, only
# does the first. For the machinery underneath,
# [heterogeneous effects](../learn/04-heterogeneous-effects.py) walks through the DR-learner and
# the profit curve, and [estimating effects](../learn/03-estimating-effects.py) covers what the
# doubly robust estimator buys you.
