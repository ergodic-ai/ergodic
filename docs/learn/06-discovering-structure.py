# %% [markdown]
# # Discovering structure: when nobody can draw the graph
#
# One assumption has carried every guide so far: somebody could draw the graph. For a
# voucher campaign with three variables, fine. Now picture the growth team's weekly
# dashboard, six metrics with opinions attached. Everyone agrees ad spend matters. Nobody
# agrees on the arrows, and the meeting where you draw them is the meeting where causal
# projects die.
#
# Causal discovery is the alternative: let the data draw the arrows it can defend, and say
# plainly which arrows it can't. The second half of that sentence is the important one. A
# discovery method that always returns a fully directed graph is lying to you, because some
# arrows are not decidable from observational data, and an honest method returns an
# object with exactly the undecided parts marked.
#
# As always, we simulate the company so the true arrows are in the code, six weekly metrics
# with a real funnel inside:

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(6)
n = 3_000

# The answer key, this time a whole graph:
#   season -> ad_spend     (budgets chase the season)
#   season -> visits       (organic seasonal traffic)
#   quality -> visits      (site performance brings people back)
#   ad_spend -> visits -> signups -> revenue   (the funnel)
season = rng.normal(0.0, 1.0, size=n)
quality = rng.normal(0.0, 1.0, size=n)
ad_spend = 0.8 * season + rng.normal(0.0, 0.8, size=n)
visits = 0.7 * ad_spend + 0.5 * season + 0.7 * quality + rng.normal(0.0, 0.8, size=n)
signups = 0.8 * visits + rng.normal(0.0, 0.8, size=n)
revenue = 0.9 * signups + rng.normal(0.0, 0.8, size=n)

df = pd.DataFrame(
    {
        "season": season,
        "quality": quality,
        "ad_spend": ad_spend,
        "visits": visits,
        "signups": signups,
        "revenue": revenue,
    }
)
df.head()

# %% [markdown]
# ## One call, one honest graph

# %%
from ergodic import discover, tabular

data = tabular(df)
result = discover(data, method="pc", alpha=0.01)
print(result.graph.kind.value)
sorted(e.glyph() for e in result.graph.edges)

# %% [markdown]
# Six edges: the true skeleton exactly, five of them oriented, and one left as a plain dash,
# `ad_spend --- season`. (PC is the classic constraint-based algorithm; `alpha` is the
# significance level of its independence tests, stricter here to keep only well supported
# edges.)
#
# Where did the arrows come from? Not from time stamps or domain hints; the algorithm saw
# only the numbers. The information lives in independence patterns. Ad spend and site
# quality have nothing to do with each other, and the data says so, until you condition on
# visits, and then they become dependent. You've seen that signature before: it's the
# collider from guide two, seen from the other side. Only arrows colliding into visits
# produce it, so the data itself orients `ad_spend -> visits <- quality`. A few more steps
# of that logic (any other orientation would either create a collider the data rules out,
# or a cycle) carry the arrows down the funnel.
#
# And the dash is not indecision. Both orientations of season and ad spend produce exactly
# the same independence pattern, so no amount of this data could ever tell them apart. The
# object PC returns, a CPDAG, is the honest summary: arrows where the data decides,
# dashes where only an equivalence of explanations exists.
#
# ## Estimate straight off the discovered graph
#
# Here's the move this library was built around. You don't have to pick one graph out of
# the class to act. Identification accepts the CPDAG itself, and when it certifies an
# effect, the certificate holds in every graph the class contains, dashes and all:

# %%
from ergodic import estimate_effect, identify_effect

print(identify_effect(result.graph, "visits", "revenue").estimand)
est = estimate_effect(result.graph, data, "visits", "revenue", strategy="dml")
lo, hi = est.interval()
print(f"visits -> revenue: {est.value:.3f}  CI [{lo:.3f}, {hi:.3f}]   truth: 0.720")
# dml, because the treatment is continuous; the default estimator wants a binary one

# %% [markdown]
# Two calls from raw data to a causal effect with an interval: `discover`, then
# `estimate_effect` on what discovery returned. The planted truth is 0.8 times 0.9, and the
# estimate lands on it, with no human ever orienting an edge.
#
# Ask about the effect of ad spend, though, and the undecided dash suddenly matters:

# %%
verdict = identify_effect(result.graph, "ad_spend", "revenue")
print(verdict.identifiable)
print(verdict.estimand)

# %% [markdown]
# Refused, and the message names the exact edge at fault. If season drives budgets, season
# is a confounder you'd have to adjust for; if budgets drove the seasonal index, adjusting
# for it would cut the effect you're measuring. The two members of the class genuinely
# disagree about what ad spend does, so no single number is honest, and the library says so
# instead of picking a side. (When you want the per-member answers, `dags_of_cpdag`
# enumerates the class and you identify each member separately.)
#
# ## Knowledge breaks the tie
#
# You know something the data doesn't: seasons are not caused by marketing budgets. Say
# exactly that, as the tier ordering from the [knowledge guide](../knowledge.md), and
# discovery folds it in:

# %%
from ergodic import DomainKnowledge

dk = DomainKnowledge().with_tiers(
    [
        ["season", "quality"],
        ["ad_spend"],
        ["visits"],
        ["signups"],
        ["revenue"],
    ]
)
oriented = discover(data, method="pc", alpha=0.01, knowledge=dk)
print(oriented.graph.kind.value)
sorted(e.glyph() for e in oriented.graph.edges)

# %% [markdown]
# Every edge is directed now, and notice the bookkeeping: the result calls itself `mixed`,
# not `cpdag`. That's deliberate. The season arrow came from your knowledge, not from the
# data, so the graph is no longer a pure data-derived equivalence class, and the library
# won't quietly pretend it is. You re-face it yourself, an explicit act that documents the
# judgment call, and a fully oriented graph passes as a DAG:

# %%
funnel = oriented.graph.as_kind("dag")
est_ads = estimate_effect(funnel, data, "ad_spend", "revenue", strategy="dml")
lo, hi = est_ads.interval()
print(f"ad_spend -> revenue: {est_ads.value:.3f}  CI [{lo:.3f}, {hi:.3f}]   truth: 0.504")

# %% [markdown]
# Data plus one piece of domain knowledge answered what data alone could not, and the split
# of labor is exactly right: the data did the twelve-way work of finding the skeleton and
# the funnel arrows, and you supplied the single bit it couldn't.
#
# ## How sure are we about those edges?
#
# One run returns one graph and no sense of its fragility. Rerunning discovery on resampled
# rows shows which edges are load-bearing and which are luck:

# %%
from itertools import combinations

from ergodic.discovery import bootstrap

stability = bootstrap(data, method="pc", alpha=0.01, n_bootstrap=60, seed=0)
freq = {f"{u} - {v}": stability.frequency(u, v) for u, v in combinations(df.columns, 2)}
pairs = sorted(freq.items(), key=lambda kv: kv[1])

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([k for k, _ in pairs], [v for _, v in pairs], color="#33415c")
ax.axvline(0.8, color="#d1495b", ls="--", lw=1.2)
ax.set_xlabel("fraction of 60 bootstrap reruns containing the edge")
ax.set_xlim(0, 1)
fig.tight_layout()

# %% [markdown]
# The six true edges survive every resample and everything else stays near zero, which is
# what a trustworthy skeleton looks like. On harder data this plot is the difference
# between "the algorithm said so" and "the algorithm says so almost every time we shake the
# data". These frequencies are stability, not probability; a graph that's wrong for a
# systematic reason is stably wrong.
#
# ## The confounder you didn't measure
#
# Last act. Suppose nobody logged the seasonal index at all: same world, one column
# missing, and that column confounds ad spend and visits. PC's premise (every common cause
# is measured) is now false. The honest algorithm for this situation is FCI, and its output
# says less, on purpose:

# %%
data_unlogged = tabular(df.drop(columns="season"))
latent_result = discover(data_unlogged, method="fci", alpha=0.01)
print(latent_result.graph.kind.value)
sorted(e.glyph() for e in latent_result.graph.edges)

# %% [markdown]
# The funnel arrows survive (`visits --> signups --> revenue`), and the edges into visits
# now wear circles: `ad_spend o-> visits` reads "visits does not cause ad spend, and
# whether ad spend causes visits or merely shares a hidden cause with it, this data cannot
# tell". Which is precisely the truth we engineered. No DAG and no CPDAG can say that
# sentence; the circle exists because the question has no answer in the data.
#
# The estimates follow the same logic. The funnel effect never depended on season, and it
# still works, straight off the PAG:

# %%
est_pag = estimate_effect(latent_result.graph, data_unlogged, "visits", "revenue", strategy="dml")
lo, hi = est_pag.interval()
print(f"visits -> revenue: {est_pag.value:.3f}  CI [{lo:.3f}, {hi:.3f}]   truth: 0.720")

verdict = identify_effect(latent_result.graph, "ad_spend", "revenue")
print(f"ad_spend -> revenue: identifiable = {verdict.identifiable}")

# %% [markdown]
# The ad-spend question, which genuinely hinges on the unmeasured season, is refused again.
# The pattern is worth stating: hiding a confounder didn't poison everything. It poisoned
# the one question that depends on it, and the machinery billed exactly that question and
# no other.

# %%
from matplotlib.patches import FancyArrowPatch


def draw_graph(graph, pos, ax, title):
    for edge in graph.edges:
        u, mid, v = edge.glyph().split()
        if mid in ("<--", "<-o"):
            u, v, mid = v, u, {"<--": "-->", "<-o": "o->"}[mid]
        style = "-" if mid == "---" else "-|>"
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle=style,
                mutation_scale=15,
                shrinkA=34,
                shrinkB=34,
                lw=1.5,
                color="#33415c",
            )
        )
        if mid[0] == "o" or mid[-1] == "o":
            ends = [(u, v)] if mid[0] == "o" else []
            ends += [(v, u)] if mid[-1] == "o" else []
            for a, b in ends:
                x = pos[a][0] + 0.18 * (pos[b][0] - pos[a][0])
                y = pos[a][1] + 0.18 * (pos[b][1] - pos[a][1])
                ax.scatter([x], [y], s=42, facecolor="white", edgecolor="#33415c", zorder=3)
    for name, (x, y) in pos.items():
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#eef1f6", "ec": "#33415c"},
        )
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.5, 1.8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


layout = {
    "season": (0.0, 1.5),
    "quality": (1.6, 1.5),
    "ad_spend": (0.0, 0.2),
    "visits": (1.2, 0.2),
    "signups": (2.3, 0.2),
    "revenue": (3.3, 0.2),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
draw_graph(result.graph, layout, axes[0], "PC with season measured: a CPDAG")
draw_graph(
    latent_result.graph,
    {k: v for k, v in layout.items() if k != "season"},
    axes[1],
    "FCI with season hidden: a PAG",
)
fig.tight_layout()

# %% [markdown]
# Arrows where the data decides, a dash where it can't, circles where a hidden cause might
# hide. Three kinds of honesty, one picture each.
#
# ## Where the series leaves you
#
# Count the calls in this guide: `discover`, `identify_effect`, `estimate_effect`. That's
# the whole loop, from a dataframe of metrics to a causal effect with an interval, with
# every undecidable step surfaced rather than papered over: the dash the data couldn't
# orient, the mixed face that recorded your judgment call, the circle that priced in a
# confounder nobody logged, the refusal that named its reason.
#
# What you saw is a slice. Discovery has score-based searches and learner-driven tests for
# nonlinear worlds ([the discovery guide](../discovery.md)), estimation has the estimator
# library and the Bayesian path ([inference](../inference.md)), identification generalizes
# further than adjustment ([identification](../identification.md)), and the parts of the
# subject the library doesn't do yet, selection bias, time series, the do-calculus, are
# listed at the end of each guide rather than implied away. Simulation was the teacher
# throughout, and it's yours to keep: when you wonder whether a method survives a world,
# build the world and check. The answer key is one `rng` away.
