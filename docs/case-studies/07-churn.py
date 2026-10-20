# %% [markdown]
# # Customer churn: when the fix looks like the cause
#
# A subscription business runs a save program. A churn-risk model scores every customer each
# month, and the highest-risk ones get an offer: a discount and a call from a success manager.
# The quarter closes, someone pulls the numbers, and the result is alarming. Customers who got
# the offer churned at 34 percent. Customers who didn't churned at 12 percent. The save offer
# looks like it nearly triples churn. A VP asks why the company is paying to lose customers, and
# proposes killing the program.
#
# The offer works. Killing it would throw away real retention, and the reason the numbers say
# otherwise is the oldest trap in the book applied to churn: the offer went to the customers most
# likely to leave, so the comparison is rigged before any customer is saved.
#
# This is a companion to the [lifetime-value study](06-clv.py), and like that one it runs on a
# simulation rather than a real dataset, because the question "what does the offer do" has no
# public dataset with a measured answer. We write the true effect into the code and grade the
# methods against it.

# %%
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


rng = np.random.default_rng(7)
n = 20_000

tenure = rng.normal(0.0, 1.0, size=n)  # standardized; higher = longer-tenured = safer
usage_drop = rng.normal(0.0, 1.0, size=n)  # recent decline in use, the dissatisfaction signal
tickets = np.clip(0.9 * usage_drop + rng.normal(0.0, 1.0, size=n), 0.0, None)  # support contacts
tickets_z = (tickets - tickets.mean()) / tickets.std()

# The company's risk score drives who gets the offer; the score uses tickets, the true risk doesn't.
risk = 1.5 * usage_drop - 1.1 * tenure
score = risk + 0.7 * tickets_z
offer = (rng.uniform(size=n) < sigmoid(1.8 * (score - 0.7))).astype(int)

# The answer key. The offer cuts the churn logit, most for mid-risk customers who are on the fence;
# the safe have nothing to save and the near-certain leavers are too far gone.
save = 2.3 * np.exp(-((risk - 0.4) ** 2) / (2 * 1.1**2))
churn_p0 = sigmoid(-1.55 + risk)  # churn probability without the offer
churn_p1 = sigmoid(-1.55 + risk - save)  # with it
churned = (rng.uniform(size=n) < np.where(offer == 1, churn_p1, churn_p0)).astype(int)

true_effect = (churn_p1 - churn_p0).mean()
df = pd.DataFrame(
    {
        "tenure": tenure,
        "usage_drop": usage_drop,
        "tickets": tickets,
        "offer": offer,
        "churned": churned,
    }
)
print(f"customers offered : {offer.mean():.1%}")
print(f"overall churn     : {churned.mean():.1%}")
print(f"true average effect of the offer: {true_effect:+.3f}  (negative means it cuts churn)")

# %% [markdown]
# The true effect of the offer is to cut the churn probability by about 12 points, averaged over
# everyone. Read the world off the code. Two things drive churn, a recent drop in usage and short
# tenure, and both also drive who gets the offer, because the company aims the offer at risk.
# Support tickets are in the picture too, but look closely: tickets come out of `usage_drop`, the
# same dissatisfaction that drives churn, and tickets never enter the churn equation. They ride
# along with churn without causing it. Hold that, it matters later.
#
# ## The number that kills the program
#
# Compare the offered to the rest, the way the VP did:

# %%
p1 = churned[offer == 1].mean()
p0 = churned[offer == 0].mean()
print(f"churn among offered    : {p1:.1%}")
print(f"churn among not offered: {p0:.1%}")
print(f"naive difference       : {p1 - p0:+.1%}")

# %% [markdown]
# Offered customers churn at 34 percent, the rest at 12, a gap of plus 22 points. Taken at face
# value the offer is a catastrophe. But the offer was aimed at the riskiest customers, so the
# offered group was heading for the exit before anyone called them. The comparison puts the
# company's highest-risk customers, with the offer, against its lowest-risk customers, without it,
# and blames the offer for the difference. This is confounding by indication, the same shape as a
# drug that looks deadly because it's given to the sickest patients.
#
# The fix is to compare like with like. Draw what drives what:

# %%
from ergodic import dag, identify_effect

g = dag(
    [
        "tenure -> offer",
        "usage_drop -> offer",
        "tickets -> offer",
        "usage_drop -> tickets",
        "tenure -> churned",
        "usage_drop -> churned",
        "offer -> churned",
    ]
)
identify_effect(g, "offer", "churned").adjustment_set

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

POS = {
    "usage_drop": (1.5, 1.7),
    "tenure": (3.3, 1.7),
    "tickets": (0.0, 0.85),
    "offer": (0.9, 0.0),
    "churned": (3.3, 0.0),
}


def draw_graph(graph, pos, title):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for edge in graph.edges:
        u, mid, v = edge.glyph().split()
        if mid == "<--":
            u, v = v, u
        style = "-" if mid == "---" else "-|>"
        ax.add_patch(
            FancyArrowPatch(
                pos[u],
                pos[v],
                arrowstyle=style,
                mutation_scale=15,
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
    ax.set_xlim(-0.4, 4.0)
    ax.set_ylim(-0.5, 2.2)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()


draw_graph(g, POS, "What drives the offer, and what drives churn")

# %% [markdown]
# Usage and tenure point at both the offer and churn, so they're the back door, and the adjustment
# set names them (tickets rides along because the offer depends on it). Close that door and
# estimate the effect with the doubly robust estimator. The outcome is binary, so the estimate is
# a risk difference, the change in churn probability the offer causes.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

from ergodic import estimate_effect, tabular

data = tabular(df, types={"offer": "categorical", "churned": "categorical"})
est = estimate_effect(
    g,
    data,
    "offer",
    "churned",
    strategy="aipw",
    learner={
        "outcome": HistGradientBoostingClassifier(random_state=0),
        "propensity": HistGradientBoostingClassifier(random_state=0),
    },
    random_state=0,
)
lo, hi = est.interval()
print(f"naive difference : {p1 - p0:+.3f}")
print(f"adjusted effect  : {est.value:+.3f}   CI [{lo:.3f}, {hi:.3f}]")
print(f"truth            : {true_effect:+.3f}")

# %% [markdown]
# The sign flips. The naive plus 22 points becomes minus 13.1, interval minus 15.6 to minus 10.6,
# straddling the planted minus 12.2. Once the comparison holds risk fixed, the offer cuts churn by
# about 13 points, close to the truth. The program the quarterly numbers condemned is preventing
# real departures, not causing them. The naive table had it exactly backwards.
#
# The fit also warns about limited overlap: 28.4% of propensity estimates are in the
# tails outside [0.01, 0.99], and 1.9% require clipping. Treat the estimate cautiously;
# adjustment does not create comparisons where the data provides little support.
#
# ## Where the data runs out
#
# Adjustment worked, but it leans on having low-risk and high-risk customers in both groups to
# compare. Check that. Fit a model of who gets the offer and look at the predicted probability,
# the propensity, split by who actually got it.

# %%
prop = (
    HistGradientBoostingClassifier(random_state=0)
    .fit(df[["tenure", "usage_drop", "tickets"]], df["offer"])
    .predict_proba(df[["tenure", "usage_drop", "tickets"]])[:, 1]
)

fig, ax = plt.subplots(figsize=(7, 3.8))
bins = np.linspace(0, 1, 36)
ax.hist(prop[offer == 0], bins=bins, density=True, alpha=0.6, color="#9aa5b1", label="not offered")
ax.hist(prop[offer == 1], bins=bins, density=True, alpha=0.6, color="#d1495b", label="offered")
ax.set_xlabel("estimated probability of getting the offer")
ax.set_ylabel("density")
ax.legend(frameon=False)
fig.tight_layout()
print(f"customers with propensity above 0.95: {(prop > 0.95).mean():.1%}")

# %% [markdown]
# The two groups barely meet at the right edge. Among the highest-risk customers the offer was
# close to automatic, so there's almost no untreated customer up there to compare against. Sort
# customers into risk deciles and the offer rate climbs from about 70 percent at the median to 99.5
# percent in the top decile. Adjustment can hold risk fixed only where both kinds of customer
# exist, and in the top band they effectively don't. The estimate covers that region by
# extrapolating the model, not by comparing real customers, and no estimator can conjure a control
# group that isn't there. The honest move is to run an actual holdout in the saturated band, the
# one place the observational data can't speak. This is positivity, and risk-based targeting
# destroys it exactly where risk is highest.
#
# ## The predictor that isn't a lever
#
# Set the offer aside and ask a different question: what drives churn? A team building a churn
# model would notice support tickets predict it, and might propose cutting ticket volume to cut
# churn, by deflecting contacts or burying the support line. Let discovery weigh in. Hand it the
# churn variables and let it learn the structure.

# %%
from ergodic import discover

drivers = tabular(
    df[["tenure", "usage_drop", "tickets", "churned"]].iloc[:6000],
    types={"churned": "categorical"},
)
found = discover(drivers, method="pc")
print(f"corr(tickets, churned)  : {np.corrcoef(tickets, churned)[0, 1]:.3f}")
print("edges PC can defend     :", sorted(e.glyph() for e in found.graph.edges))

# %% [markdown]
# Tickets correlate with churn at 0.22, a signal a model would happily use. But discovery draws no
# edge between tickets and churn. It connects tickets to `usage_drop` and stops, and it points
# both `usage_drop` and `tenure` straight at churn. The reading is clean: once you know a customer's
# usage drop, their ticket count tells you nothing more about whether they'll churn. Tickets are a
# symptom of the same dissatisfaction that drives churn, sitting downstream, not a cause. Deflecting
# tickets would treat the thermometer and leave the fever. A churn model would rank tickets as
# important and be right that they predict; acting on them rests on a causal claim the data refuses.
#
# ## Savable is not the same as risky
#
# Back to the offer. It works on average, but the average hides who it works on. Estimate the effect
# per customer with the DR-learner, the same move as the [lifetime-value study](06-clv.py).

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

from ergodic.inference import DRLearner

# Only the point per-customer effect is read below (the decile table), so uncertainty=False
# skips the per-unit bootstrap; we never call .interval()/.std_error() here.
cate = (
    DRLearner(
        {
            "outcome": HistGradientBoostingClassifier(random_state=0),
            "propensity": HistGradientBoostingClassifier(random_state=0),
            "final": HistGradientBoostingRegressor(random_state=0),
        },
        random_state=0,
    )
    .fit(identify_effect(g, "offer", "churned"), data)
    .cate(uncertainty=False)
    .values  # the raw per-customer effect array (CATEResult.values) for the decile table below
)

baseline_risk = (
    HistGradientBoostingClassifier(random_state=0)
    .fit(df.loc[offer == 0, ["tenure", "usage_drop", "tickets"]], df.loc[offer == 0, "churned"])
    .predict_proba(df[["tenure", "usage_drop", "tickets"]])[:, 1]
)
decile = np.clip((baseline_risk.argsort().argsort() * 10 // n), 0, 9)
true_save = churn_p1 - churn_p0
by_decile = pd.DataFrame(
    {
        "estimated effect": [cate[decile == d].mean() for d in range(10)],
        "true effect": [true_save[decile == d].mean() for d in range(10)],
    },
    index=[f"d{d}" for d in range(10)],
)
by_decile.round(3)

# %% [markdown]
# The effect is a hump, not a slope. The lowest-risk customers have nothing to save, the
# highest-risk are leaving with or without the call, and the offer bites hardest in the middle,
# where a customer is genuinely on the fence. The DR-learner recovers the shape, a little flattened
# at the ends the way a noisy per-customer estimate is. The point is that the steepest savings
# don't sit at the top of the risk ranking, so targeting the riskiest customers, which is what the
# company does, spends the offer where it moves the needle least.

# %%
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(10)
ax.plot(x, -by_decile["true effect"], "o-", color="#33415c", lw=2, label="true churn reduction")
ax.plot(
    x, -by_decile["estimated effect"], "s--", color="#d1495b", lw=1.6, label="DR-learner estimate"
)
ax.set_xlabel("baseline churn-risk decile (low to high)")
ax.set_ylabel("churn reduction from the offer")
ax.legend(frameon=False)
fig.tight_layout()

# %%
budget = int(0.30 * n)
savable = np.argsort(cate)[:budget]  # most negative estimated effect
riskiest = np.argsort(-baseline_risk)[:budget]  # what the company targets
print(f"churn prevented, targeting the savable : {(-true_save[savable]).sum():,.0f}")
print(f"churn prevented, targeting the riskiest: {(-true_save[riskiest]).sum():,.0f}")

# %% [markdown]
# Spend the same budget, enough for 30 percent of the base, two ways. Target the customers the
# offer can move and it prevents about 1,380 departures. Target the highest-risk customers, the
# current playbook, and it prevents about 1,090. About a quarter more retention from the same
# spend, by aiming at savability instead of risk. Per-customer churn effects are noisy, more so
# than a lifetime-value number, so this is a ranking to validate on a holdout before betting the
# budget, not a guarantee.
#
# ## The placebo
#
# Every number after the naive one leans on the same assumption: the offer is as good as random
# once usage, tenure, and tickets are held fixed. The data can't prove it. What it can check is
# whether the pipeline finds an effect when there's nothing to find. Scramble the offer and
# re-estimate.

# %%
from ergodic import refute_effect

ref = refute_effect(
    g,
    data,
    "offer",
    "churned",
    method="placebo_treatment",
    strategy="aipw",
    estimand_value=est.value,
    n_simulations=8,
    random_state=0,
)
print(f"real effect    : {ref.estimand_value:+.3f}")
print(f"placebo (mean) : {ref.refuted_value:+.3f}   p = {ref.p_value:.3f}   passed: {ref.passed}")

# %% [markdown]
# Scrambled, the effect lands near zero and the placebo runs rarely reproduce the real one,
# so the machinery isn't inventing the save. That doesn't rule out an unmeasured driver, some
# intent to quit the columns never saw, that steered both the offer and the churn. If it exists the
# adjusted effect drifts back toward the naive plus 22. The graph is an assumption, and the placebo
# tests the pipeline, not the assumption.
#
# Two readings of the same quarter. The naive table said the save program triples churn and should
# die. Holding risk fixed, the program cuts churn by about 13 points and should grow, aimed at the
# customers it can move rather than the ones most likely to leave. The companion
# [lifetime-value study](06-clv.py) tells the mirror-image story on a continuous outcome, where the
# trap is targeting the valuable instead of the movable. For the methods, see
# [estimating effects](../learn/03-estimating-effects.py) and
# [discovering structure](../learn/06-discovering-structure.py).
