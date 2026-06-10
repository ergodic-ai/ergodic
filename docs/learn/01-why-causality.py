# %% [markdown]
# # Why a correlation can't answer your question
#
# Here's a meeting that happens somewhere every week. A subscription retailer runs a loyalty
# program. An analyst pulls the numbers and finds that members spend less per quarter than
# non-members. The number is correct: clean data, no bugs, a real gap. Someone proposes
# killing the program.
#
# The number is right and the conclusion is wrong, and the gap between those two statements
# is what causal inference is about.
#
# To see it, we'll do something you can't do with real data: simulate the company.
# In a simulation we write the true effect of the program into the code, where everyone can
# read it. Then we can grade any analysis against the answer key. Keep that move in mind,
# because every guide in this series uses it. Real data never tells you whether your estimate
# was right; a simulated world does.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 20_000

# Years as a customer: people joined any time in the last ten years.
tenure = rng.uniform(0.0, 10.0, size=n)

# The program is pitched hard at signup, so the newer you are, the more likely you joined.
p_member = 1.0 / (1.0 + np.exp((tenure - 5.0) / 1.8))
member = (rng.uniform(size=n) < p_member).astype(float)

# Spending grows with tenure, and the program adds exactly 15 dollars a quarter.
# This line is the answer key.
spend = 25.0 + 12.0 * tenure + 15.0 * member + rng.normal(0.0, 18.0, size=n)

df = pd.DataFrame({"tenure": tenure, "member": member, "spend": spend})
df.head()

# %% [markdown]
# Three facts about this little world, all visible in the code. Spending rises with tenure.
# Membership falls with tenure, because marketing recruits at onboarding. And the program
# itself is worth 15 dollars a quarter to whoever is in it. That last number is the true
# effect. Everything below is an attempt to recover it from the data alone.
#
# ## The obvious analysis
#
# Compare members to non-members:

# %%
means = df.groupby("member")["spend"].mean()
naive = means[1.0] - means[0.0]
print(means.round(2))
print(f"\nnaive gap: {naive:+.2f} dollars per quarter")

# %% [markdown]
# The program looks like a disaster. Members spend about 27 dollars less per quarter, and
# the truth, which we wrote ourselves three cells ago, is that the program adds 15. The
# naive number doesn't have the wrong size. It has the wrong sign.
#
# Nothing is wrong with the data. What's wrong is the comparison. Members are mostly new
# customers, because that's who marketing signs up, and new customers spend less because
# spending grows with tenure. So "members vs non-members" is mostly "new customers vs old
# customers" wearing a costume. Tenure drives both who joins and how much they spend, and a
# variable that does that is called a confounder.
#
# ## Look inside the groups
#
# If tenure is the problem, compare like with like: split customers into tenure bands and
# compare members to non-members inside each band.

# %%
df["band"] = pd.qcut(df["tenure"], q=6)
by_band = df.groupby(["band", "member"], observed=True)["spend"].mean().unstack()
by_band["gap"] = by_band[1.0] - by_band[0.0]
by_band.round(2)

# %% [markdown]
# Inside every band the gap is positive, 13 or 14 dollars, near the true 15. (A notch low,
# because even inside a band members skew slightly newer. Banding coarsens; the estimator
# below adjusts for tenure as a number and removes the rest.) Pool the bands and the gap
# flips negative. An effect that holds in every subgroup and reverses in the total has a
# name, Simpson's paradox, and it isn't a curiosity. A 1986 study in the British Medical
# Journal compared two kidney-stone treatments: the better treatment lost in the overall
# table and won within both stone sizes, for exactly this reason. Doctors gave the better
# treatment to the harder cases.
#
# So stratifying fixed it. But notice how much that move smuggled in. We had to know that
# tenure was the variable to slice by. We had to know that slicing by it was safe, and in
# the next guide you'll see variables that make things worse when you slice by them. None of
# that knowledge is in the dataframe. It's knowledge about what causes what, and it needs
# somewhere to live.
#
# ## Draw what you believe
#
# A causal graph is that somewhere. One node per variable, one arrow per direct cause.

# %%
from ergodic import dag

g = dag(["tenure -> member", "tenure -> spend", "member -> spend"])

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def draw_dag(graph, pos, figsize=(5.5, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    for edge in graph.edges:
        u, v = edge.directed_pair()
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle="-|>",
                mutation_scale=18,
                shrinkA=32,
                shrinkB=32,
                lw=1.6,
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
            fontsize=11,
            bbox={"boxstyle": "circle,pad=0.45", "fc": "#eef1f6", "ec": "#33415c"},
        )
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 1.45)
    ax.axis("off")
    fig.tight_layout()


draw_dag(g, pos={"tenure": (1.0, 1.2), "member": (0.0, 0.0), "spend": (2.0, 0.0)})

# %% [markdown]
# Read the picture. Two routes connect `member` to `spend`. The direct arrow is the thing we
# want to measure: the program causing spending. The other route runs backwards through the
# fork at the top, `member <- tenure -> spend`. No causation flows along it, but association
# does: knowing someone is a member tells you they're probably new, and being new tells you
# they probably spend less. The naive comparison adds both routes together and reports the
# sum. That's all a confounder is: a back door that lets association through without
# causation.
#
# The question the business is asking isn't "what do members spend". It's "what would happen
# to spending if we enrolled someone". That's an intervention, written do(member), and on the
# graph an intervention is surgery: reach in and set membership yourself, and tenure no
# longer has any say in who joins.

# %%
print("before surgery, member's causes:", set(g.parents("member")))
print("after  surgery, member's causes:", set(g.do("member").parents("member")))

# %% [markdown]
# In the surgered graph the back door is gone, and only the direct arrow remains. The whole
# game of causal inference is computing what the surgered world would show, using only data
# from the unsurgered one. For this graph the recipe is exactly the stratification we did by
# hand: compare within tenure levels, then average over tenure. Asking the graph makes the
# recipe mechanical:

# %%
from ergodic import identify_effect

identify_effect(g, "member", "spend").estimand

# %% [markdown]
# That formula is called an estimand: the quantity to compute, derived from the graph alone,
# before any model touches the data. The library reads it and picks an estimator for you.

# %%
from ergodic import estimate_effect, tabular

data = tabular(df[["tenure", "member", "spend"]])
est = estimate_effect(g, data, "member", "spend")
est

# %%
summary = pd.DataFrame(
    {
        "estimate": [naive, est.value, 15.0],
        "low": [np.nan, est.interval()[0], np.nan],
        "high": [np.nan, est.interval()[1], np.nan],
    },
    index=["naive comparison", "adjusted for tenure", "truth (from the code)"],
)
summary.round(2)

# %% [markdown]
# The adjusted estimate lands on the truth, with an interval that covers it. One picture of
# the whole story: every dot is a customer, spending rises with tenure, and at any given
# tenure the members sit about 15 dollars higher. The naive comparison ignores the x-axis
# and compares the left cloud to the right cloud.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
idx = rng.choice(n, size=1500, replace=False)
for value, color, label in [(0.0, "#9aa5b1", "non-member"), (1.0, "#d1495b", "member")]:
    sel = idx[df["member"].to_numpy()[idx] == value]
    ax.scatter(df["tenure"].iloc[sel], df["spend"].iloc[sel], s=8, alpha=0.5, c=color, label=label)
ax.set_xlabel("tenure (years)")
ax.set_ylabel("spend (dollars / quarter)")
ax.set_xlim(0, 10)
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# ## What made this work
#
# Be clear about what the one-call estimate rested on, because none of it came from the
# data. We measured the confounder: tenure was a column, so the back door could be closed.
# And we drew the right graph: the arrows encoded who joins and why. The data then did the
# arithmetic.
#
# Both legs can fail. If the confounder isn't in the data at all, no adjustment on measured
# columns will save you, and you need the different machinery of guide five. If you can't
# draw the graph from knowledge, guide six is about learning it from the data, honestly,
# including the parts the data can't decide.
#
# Next: [good and bad controls](02-good-and-bad-controls.py). Stratifying on tenure rescued
# this analysis, so it's tempting to conclude you should stratify on everything you have.
# You shouldn't, and the reasons are the fastest way to understand why the graph, and not a
# variable list, is the object that matters.
