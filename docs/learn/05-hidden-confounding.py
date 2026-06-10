# %% [markdown]
# # Hidden confounding: when the back door has no handle
#
# A pricing analyst runs the cleanest regression of their life, two years of weekly data,
# sales on price, and presents the result: each dollar of price adds two and a half units of
# sales. Raise prices, sell more. The slide gets a laugh, then a worried silence, because
# the relationship is real and nobody can make it go away with more controls.
#
# Everyone in the room knows the missing word: demand. In good weeks, hot weather, paydays,
# events in town, customers want more, and the team also nudges prices up because they can.
# Demand pushes price and sales up together, and the upward push is stronger than the
# downward pull of price itself. The catch is that "demand" isn't a column. It's the
# weather, the calendar, the mood of a city. Every guide so far closed the back door by
# conditioning on the confounder, and for the first time the confounder isn't in the data.
#
# No adjustment can fix this. The honest options are two clever detours, and this guide
# walks both: the instrument and the front door.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 800  # two years of weekly data across eight regions

# Demand: real, strong, and about to be deleted from the dataset.
demand = rng.normal(0.0, 1.0, size=n)

# Wholesale cost moves around for supply-side reasons that have nothing to do with demand.
cost = rng.normal(0.0, 1.0, size=n)

# The team prices off cost, and shades the price up when demand is hot.
price = 20.0 + 1.5 * cost + 1.2 * demand + rng.normal(0.0, 0.5, size=n)

# The answer key: each dollar of price costs 2 units of sales. Demand lifts sales hard.
sales = 200.0 - 2.0 * price + 15.0 * demand + rng.normal(0.0, 3.0, size=n)

df = pd.DataFrame({"cost": cost, "price": price, "sales": sales})  # demand is NOT a column

naive_slope = np.polyfit(price, sales, 1)[0]
print(f"naive regression slope: {naive_slope:+.2f} units per dollar   truth: -2.00")

# %% [markdown]
# Wrong sign, again, and this time guide one's medicine fails. There's nothing to stratify
# by: the dataframe has cost, price, and sales, and the variable doing the damage is in none
# of them. Say it with the graph, marking demand as latent, present in the world and absent
# from the data:

# %%
from ergodic import adjustment_set, dag, identify_effect

g = dag(
    ["cost -> price", "demand -> price", "demand -> sales", "price -> sales"],
    latent=["demand"],
)
print(adjustment_set(g, "price", "sales"))

# %% [markdown]
# `None`: there is no set of measured variables that closes the back door, and the library
# won't pretend otherwise. But identification doesn't give up, because the graph holds one
# more thing, and it's the column nobody thinks about: cost.

# %%
result = identify_effect(g, "price", "sales")
print(result.strategy)
print(result.estimand)

# %% [markdown]
# ## The instrument
#
# Look at what makes cost special in that graph. Cost moves price (when wholesale goes up,
# retail follows). Cost doesn't care about demand (a supplier's contract doesn't read the
# local weather). And cost has no way to touch sales except through the price tag, because
# customers never see your costs. A variable like that is called an instrument, and it's a
# natural experiment hiding in the supply chain: the part of price variation that comes from
# cost is, by construction, clean of demand.
#
# The estimator, two-stage least squares, is exactly that sentence as algebra. First stage:
# regress price on cost, keeping only the cost-driven part of price. Second stage: regress
# sales on that cleaned price. Demand still moves sales, but it no longer correlates with
# the regressor, so the slope is the causal one.

# %%
from ergodic import estimate_effect, tabular

est = estimate_effect(g, tabular(df), "price", "sales")
lo, hi = est.interval()
print(f"2SLS: {est.value:+.2f} units per dollar  CI [{lo:.2f}, {hi:.2f}]   truth: -2.00")

# %% [markdown]
# From plus 2.5 to minus 2, using a column that never appears in the naive analysis at all.
#
# Now the bill, because instruments are not free. The estimand above says "under a linear
# (parametric) model", and it means it: unlike adjustment, which works for any functional
# form, IV identification here leans on linearity. More important, the exclusion
# restriction, cost touches sales only through price, is an assumption about the world, not
# a testable property of the data. If your supplier's costs rise in hot months and hot
# months drive demand, the instrument is broken and nothing in the output warns you. You
# argue an instrument; you don't compute it. And a weak instrument, one that barely moves
# price, trades the bias for variance and gives intervals too wide to act on. Relevance you
# can check (cost visibly moves price here); exclusion you defend in the meeting.
#
# ## The front door
#
# There's a second escape, stranger and more precious. The famous version is scientific:
# does smoking cause cancer? The tobacco industry's defense for decades was a hypothetical
# confounder, a genotype that both makes people smoke and raises cancer risk, and you can't
# adjust for a gene nobody measured. Suppose, though, that smoking works through a
# measurable middle step: tar deposits in the lungs. The genotype story has no arrow into
# tar except through smoking itself, and tar has no back door to cancer except through
# smoking, which is measured.
#
# Then each hop is clean even though the total is confounded. Smoking to tar: no open back
# door. Tar to cancer: the back door through smoking closes by conditioning on smoking. Two
# clean hops, chained, recover the dirty total.

# %%
rng = np.random.default_rng(19)
m = 6_000
genotype = rng.normal(0.0, 1.0, size=m)  # latent: never measured
smoking = (rng.uniform(size=m) < 1.0 / (1.0 + np.exp(-1.2 * genotype))).astype(float)
tar = (rng.uniform(size=m) < 0.05 + 0.85 * smoking).astype(float)
risk = 1.0 + 2.5 * tar + 1.5 * genotype + rng.normal(0.0, 1.0, size=m)
# The answer key: smoking shifts P(tar) by 0.85, tar adds 2.5, so smoking adds 2.125.

health = pd.DataFrame({"smoking": smoking, "tar": tar, "risk": risk})
naive_fd = risk[smoking == 1].mean() - risk[smoking == 0].mean()
print(f"naive gap: {naive_fd:+.2f}   truth: +2.12")

# %% [markdown]
# The naive gap overstates the effect by two thirds, exactly the inflation a smoking gene
# would produce. Write the graph with a bidirected edge, the notation for "these two share
# an unmeasured common cause", and identification finds the mediator route on its own:

# %%
from ergodic import admg

fd = admg(["smoking -> tar", "tar -> risk", "smoking <-> risk"])
ident = identify_effect(fd, "smoking", "risk")
print(ident.strategy, ident.mediators)

est_fd = estimate_effect(fd, tabular(health), "smoking", "risk")
lo, hi = est_fd.interval()
print(f"front-door: {est_fd.value:+.2f}  CI [{lo:.2f}, {hi:.2f}]   truth: +2.12")

# %% [markdown]
# Be honest about how rare this is. The front door needs a mediator that carries the entire
# effect (no side channel around it) and is itself unconfounded with the treatment. Most
# problems don't have one; when one exists, you get nonparametric identification under
# confounding you never measured, which is as close to magic as this subject offers. The
# commercial versions do exist: a recommendation's effect on purchases through the click,
# an ad's effect through the visit, a policy's effect through the feature it unlocks. The
# discipline is arguing the "entire effect" clause with a straight face.
#
# ## The two detours, drawn

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def draw_graph(graph, pos, ax, title):
    latent = set(graph.latent)
    for edge in graph.edges:
        u, mid, v = edge.glyph().split()
        if mid == "<--":
            u, v, mid = v, u, "-->"
        dashed = mid == "<->" or u in latent or v in latent
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle="<|-|>" if mid == "<->" else "-|>",
                connectionstyle="arc3,rad=0.25" if mid == "<->" else "arc3",
                mutation_scale=15,
                shrinkA=30,
                shrinkB=30,
                lw=1.5,
                ls="--" if dashed else "-",
                color="#b3434e" if mid == "<->" else "#33415c",
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
            style="italic" if name in latent else "normal",
            bbox={
                "boxstyle": "circle,pad=0.35",
                "fc": "white" if name in latent else "#eef1f6",
                "ec": "#b3434e" if name in latent else "#33415c",
                "ls": "--" if name in latent else "-",
            },
        )
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.4, 1.7)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
draw_graph(
    g,
    {"cost": (-0.2, 0.0), "price": (0.8, 0.0), "sales": (2.2, 0.0), "demand": (1.5, 1.4)},
    axes[0],
    "instrument: cost-driven price variation is clean of demand",
)
draw_graph(
    fd,
    {"smoking": (0.0, 0.0), "tar": (1.0, 0.0), "risk": (2.0, 0.0)},
    axes[1],
    "front door: two clean hops replace one dirty one",
)
fig.tight_layout()

# %% [markdown]
# Dashed means unmeasured: the latent demand node with its arrows, and the bidirected edge
# standing for a confounder nobody logged. Identification read both pictures and routed
# around the dashes.
#
# ## Where this leaves us
#
# When the confounder is measured, adjust. When it isn't, you need structure you can defend:
# a variable that moves the treatment for reasons unrelated to the outcome, or a mediator
# that carries the whole effect. Both rest on assumptions the data can't check, which is
# uncomfortable, and stating them as a graph at least makes the argument explicit enough to
# attack. That's not a weakness of the method. It's the method refusing to hide the part
# that was always going to be an argument.
#
# One assumption has run under this whole series so far: that you can draw the graph at
# all. The [last guide](06-discovering-structure.py) drops it, and lets the data draw the
# graph, including an honest mark for "a hidden confounder might live here".
