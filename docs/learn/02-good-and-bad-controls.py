# %% [markdown]
# # Good and bad controls
#
# Guide one ended with a rescue: stratifying by tenure turned a backwards answer into the
# right one. The natural conclusion is that controlling for things is safe, so you should
# control for everything you can measure. Throw every column into the regression and let the
# math sort it out.
#
# That advice is wrong, and it's wrong in an interesting way. There are variables you must
# control for, variables that erase the effect you're measuring when you control for them,
# and variables that manufacture bias out of nothing. The three cases look identical in a
# spreadsheet. They differ only in the causal structure around them, which is why the graph,
# and never a list of available columns, is the object that decides.
#
# We'll build one tiny world for each case, with the true effect written in the code, and
# watch what controlling does. The estimator stays deliberately dumb throughout: ordinary
# least squares, the workhorse regression. The point isn't the estimator. It's what you hand
# it.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n = 20_000


def regression_effect(df, treatment, outcome, controls=()):
    """The OLS coefficient on the treatment, with the given controls in the model."""
    cols = [df[treatment].to_numpy()] + [df[c].to_numpy() for c in controls]
    X = np.column_stack([*cols, np.ones(len(df))])
    beta, *_ = np.linalg.lstsq(X, df[outcome].to_numpy(), rcond=None)
    return beta[0]


# %% [markdown]
# ## The fork: control for it
#
# A retailer sends marketing emails. Engaged customers get more of them (they opted into
# more lists), and engaged customers buy more anyway. Each email is worth 5 dollars, says
# the answer key.

# %%
engagement = rng.normal(size=n)
emails = 4.0 + 2.0 * engagement + rng.normal(0.0, 1.0, size=n)
purchases = 50.0 + 5.0 * emails + 20.0 * engagement + rng.normal(0.0, 10.0, size=n)
fork = pd.DataFrame({"engagement": engagement, "emails": emails, "purchases": purchases})

adjusted = regression_effect(fork, "emails", "purchases", ["engagement"])
print(f"no controls:            {regression_effect(fork, 'emails', 'purchases'):.2f}")
print(f"controlling engagement: {adjusted:.2f}")
print("truth: 5.00")

# %% [markdown]
# This is guide one again, compressed. Engagement is a common cause of emails and purchases,
# a fork, and the uncontrolled regression reads the association that flows around the back:
# heavy email recipients are engaged people, and engaged people buy. Control for engagement
# and only the causal arrow is left. A confounder is a back door, and you close it by
# conditioning. So far the kitchen-sink advice looks good.
#
# ## The chain: leave it alone
#
# Now a different structure. The company runs ads in some regions and not others, assigned
# at random, so nothing confounds the ads. Ads bring people to the site, and visits turn
# into purchases. Ads work only through visits: a chain.

# %%
ads = rng.normal(size=n)
visits = 2.0 * ads + rng.normal(0.0, 1.0, size=n)
purchases = 3.0 * visits + rng.normal(0.0, 1.0, size=n)
chain = pd.DataFrame({"ads": ads, "visits": visits, "purchases": purchases})

print(f"no controls:        {regression_effect(chain, 'ads', 'purchases'):.2f}")
print(f"controlling visits: {regression_effect(chain, 'ads', 'purchases', ['visits']):.2f}")
print("truth: 6.00")

# %% [markdown]
# Each unit of ads produces 2 visits, each visit produces 3 dollars, so the effect of ads is
# 6. The plain regression finds it. Add visits as a control, the way the kitchen-sink rule
# says to, and the effect vanishes.
#
# Nothing malfunctioned. The regression answered the question you asked: "what do ads do,
# holding visits fixed?" That means comparing ad and no-ad regions among customers who
# visited equally often, and there the ads look identical, because the only thing ads do is
# bring visits and you just held that fixed. You amputated the mechanism and then measured
# the stump.
#
# A variable on the causal path from treatment to outcome is a mediator, and you control for
# it exactly when you mean to ask "how much of the effect does not pass through this
# channel?" That's a real question (it has its own field, mediation analysis), but it is not
# the question "do ads work". For total effects, mediators stay out.
#
# ## The collider: never, and your data may have done it for you
#
# The third structure is the strange one. Send product-update emails on a schedule that has
# nothing to do with the customer (say, by signup parity), and let product usage be whatever
# it is. In this world, emails and usage are completely independent, and neither causes the
# other. Both, independently, help retention.

# %%
emails = rng.normal(size=n)
usage = rng.normal(size=n)
score = emails + usage + rng.normal(0.0, 0.7, size=n)
retained = score > 0.0
world = pd.DataFrame({"emails": emails, "usage": usage, "retained": retained})

survivors = world[world["retained"]]
everyone_corr = world["emails"].corr(world["usage"])
survivor_corr = survivors["emails"].corr(survivors["usage"])
print(f"correlation of emails and usage, everyone:       {everyone_corr:+.3f}")
print(f"correlation of emails and usage, retained only:  {survivor_corr:+.3f}")

# %% [markdown]
# Among everyone, zero correlation, as built. Among retained customers, a solid negative
# one. An analyst who studies the retained customers, perhaps because churned customers
# aren't in the warehouse anymore, finds that email-heavy customers use the product less,
# and concludes the emails are hurting engagement. The true effect, from the code, is
# exactly nothing.
#
# Here's the logic of it. Retention is a shared consequence, a collider: two arrows point
# into it. Knowing a customer survived while getting few emails tells you something real
# about their usage: it was probably high, or they wouldn't have survived. Inside the
# survivors the two causes compete to explain the survival, so they anti-correlate.
# Conditioning on a common consequence creates association out of nothing; this is
# Berkson's paradox, the mirror image of the fork, where conditioning removes it.
#
# The nasty part is the second clause of the heading. Nobody types "control for retention"
# into a regression. They study a table that only contains retained customers, and the
# selection does the conditioning silently. Survivorship is a collider you've already
# conditioned on by the time the data reaches you.
#
# ## The graph knows which case you're in
#
# Three worlds, three answers: control, don't, and the data already did something you must
# undo. In a column list they're indistinguishable. As graphs:

# %%
from ergodic import dag

fork_g = dag(["engagement -> emails", "engagement -> purchases", "emails -> purchases"])
chain_g = dag(["ads -> visits", "visits -> purchases"])
collider_g = dag(["emails -> retained", "usage -> retained"])

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def draw_dag(graph, pos, ax, title):
    for edge in graph.edges:
        u, v = edge.directed_pair()
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle="-|>",
                mutation_scale=16,
                shrinkA=30,
                shrinkB=30,
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
            bbox={"boxstyle": "circle,pad=0.4", "fc": "#eef1f6", "ec": "#33415c"},
        )
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.4, 1.6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(11, 3))
draw_dag(
    fork_g,
    {"engagement": (1.0, 1.3), "emails": (0.0, 0.0), "purchases": (2.0, 0.0)},
    axes[0],
    "fork: control for it",
)
draw_dag(
    chain_g,
    {"ads": (0.0, 0.6), "visits": (1.0, 0.6), "purchases": (2.0, 0.6)},
    axes[1],
    "chain: leave it alone",
)
draw_dag(
    collider_g,
    {"emails": (0.0, 1.3), "usage": (2.0, 1.3), "retained": (1.0, 0.0)},
    axes[2],
    "collider: never condition",
)
fig.tight_layout()

# %% [markdown]
# The traffic rules are worth saying once, plainly. Association flows along a path unless
# something blocks it. A fork or a chain transmits, and conditioning on its middle node
# blocks it. A collider blocks, and conditioning on it (or on anything downstream of it)
# opens it. Every rule of thumb about good and bad controls is these two sentences applied
# to some path.
#
# You don't apply them by hand. Ask the graph:

# %%
from ergodic import adjustment_set
from ergodic.identification import is_valid_adjustment_set

print(adjustment_set(fork_g, "emails", "purchases"))
print(is_valid_adjustment_set(fork_g, "emails", "purchases", set()))

# %% [markdown]
# For the fork: adjust for engagement, and the empty set won't do. Both verdicts match what
# the regressions showed. For the chain:

# %%
print(adjustment_set(chain_g, "ads", "purchases"))
print(is_valid_adjustment_set(chain_g, "ads", "purchases", {"visits"}))

# %% [markdown]
# The right adjustment set is empty (the ads were as good as randomized), and the library
# refuses visits: a mediator is never a valid control for a total effect. And the collider
# world:

# %%
from ergodic import identify_effect

print(identify_effect(collider_g, "emails", "usage").strategy)
print(is_valid_adjustment_set(collider_g, "emails", "usage", {"retained"}))

# %% [markdown]
# The strategy comes back `null`: no causal path runs from emails to usage in this graph, so
# the true effect is zero before any data gets involved, and conditioning on retained is
# rejected. The analysis that found "emails hurt usage" was an answer to no well-posed
# causal question at all.
#
# ## Measure less, on purpose
#
# One more thing the graph buys you, and it points the other way: permission to ignore
# variables. Adjustment sets aren't unique, and the smallest one can be much smaller than
# the safest-looking one.

# %%
from ergodic.identification import minimal_adjustment_set

g = dag(
    [
        "geo -> competition",
        "competition -> price",
        "competition -> sales",
        "price -> sales",
        "season -> price",
        "season -> sales",
    ]
)
print(adjustment_set(g, "price", "sales"))
print(minimal_adjustment_set(g, "price", "sales"))

# %% [markdown]
# The canonical set takes everything safe; the minimal one says competition and season
# suffice, and geography can stay unmeasured: every back-door path it sits on runs through
# competition anyway, and competition is already held. When a variable is expensive to
# collect, that's not a style point, it's budget.
#
# ## Where this leaves us
#
# The rule was never "control for more". It's: close every back door, keep every causal
# path open, and don't condition on common consequences, including the conditioning your
# data collection did for you. A graph makes the rule mechanical, and `adjustment_set` and
# `is_valid_adjustment_set` apply it without you tracing a single path.
#
# Regressions were enough here because every relationship was linear and we wrote them that
# way. Real data is less polite. Next, [estimating effects](03-estimating-effects.py): what
# to do once you know what to adjust for, why one fitted model is a single point of failure,
# and what an interval around an effect means.
