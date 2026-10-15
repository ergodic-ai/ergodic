# %% [markdown]
# # Does smoking lower birthweight? A back-door adjustment, start to finish
#
# Every guide before this one ran on a simulated world, where we wrote the true effect into
# the code and graded the analysis against it. Real data never hands you the answer key. So
# this page does the harder thing: a real dataset, a question with a known published answer,
# and the same three steps you'd run at work. Draw the graph, close the back door, estimate.
#
# The question is old and settled enough to check our work against. Does a mother smoking
# during pregnancy lower her baby's birthweight, and by how many grams? The data is 4,642
# singleton births in Pennsylvania, pulled from public-domain US natality records and made a
# textbook case by Almond, Chay and Lee (2005) and by Cattaneo (2010). The outcome is birth
# weight in grams. The published figure on this extract, after adjusting for who smokes, is
# around -231 g.
#
# We'll get there, but the interesting part is the gap we have to cross to get there. The raw
# difference in birthweight between smokers and nonsmokers is about -275 g. That number is a
# real measurement, and it is too large, for a reason you can read straight off the data
# before any model runs.

# %%
from ergodic.datasets import load_birthweight

ds = load_birthweight()
df = ds.data.to_pandas()
print(f"{ds.data.n_rows} births, {len(ds.confounders)} measured confounders")
print(f"treatment {ds.treatment!r} (treated = {ds.treated!r}), outcome {ds.outcome!r} in grams")
print(df["mbsmoke"].value_counts().to_dict())

# %% [markdown]
# 864 smokers, 3,778 nonsmokers. Before comparing their babies, compare the mothers.
#
# ## A causal EDA: who smokes?
#
# Generic histograms of the outcome won't tell you whether an estimate will be biased. The
# thing that biases it is imbalance in the confounders, the variables that move both who
# smokes and what the baby weighs. So the right first look is a balance table: for every
# confounder, its average among smokers next to its average among nonsmokers, and a single
# number for how far apart they are.
#
# That number is the standardized mean difference, the gap in means divided by the pooled
# standard deviation. Dividing by the spread puts age (measured in years) and marriage
# (measured as a rate) on the same ruler, so you can sort by it. As a rough convention an SMD
# past about 0.1 is a group difference worth worrying about, and past 0.25 it's large.
#
# Three of the confounders arrive as text (`mmarried`, `fbaby`, `prenatal1`), so we map them
# to 0/1 first to read them as rates.

# %%
import numpy as np
import pandas as pd

binary_text = {
    "mmarried": {"married": 1.0, "notmarried": 0.0},
    "fbaby": {"Yes": 1.0, "No": 0.0},
    "prenatal1": {"Yes": 1.0, "No": 0.0},
}
coded = df.copy()
for column, mapping in binary_text.items():
    coded[column] = coded[column].map(mapping)

smokers = coded[coded["mbsmoke"] == "smoker"]
nonsmokers = coded[coded["mbsmoke"] == "nonsmoker"]

rows = {}
for name in ds.confounders:
    a, b = smokers[name].astype(float), nonsmokers[name].astype(float)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    rows[name] = {
        "smoker": a.mean(),
        "nonsmoker": b.mean(),
        "SMD": (a.mean() - b.mean()) / pooled_sd,
    }
balance = pd.DataFrame(rows).T
balance = balance.reindex(balance["SMD"].abs().sort_values(ascending=False).index)
balance.round(2)

# %% [markdown]
# Read the top of the table. Smokers are married 47 percent of the time against 75 percent
# for nonsmokers, the largest gap in the set (SMD -0.60). They have fewer years of schooling,
# the mother near 11.6 years against 12.9 (SMD -0.55) and the father about the same gap. They
# drink during pregnancy about four times as often. They're a few years younger, attend fewer
# prenatal visits, and started prenatal care later. None of this is subtle. Smoking sorts
# along nearly every line that social science knows is tied to a smaller baby.
#
# That's the whole problem in one table. The two groups of mothers differ in ways that have
# their own effect on birthweight, so a plain smoker-versus-nonsmoker comparison can't be
# clean. Here is the single sharpest split, maternal education, drawn as two distributions.

# %%
import matplotlib.pyplot as plt


def draw_education(frame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(-0.5, 18.5, 1.0)
    for status, color, label in [
        ("nonsmoker", "#9aa5b1", "nonsmoker"),
        ("smoker", "#d1495b", "smoker"),
    ]:
        years = frame.loc[frame["mbsmoke"] == status, "medu"].astype(float)
        ax.hist(years, bins=bins, density=True, alpha=0.6, color=color, label=label)
    ax.set_xlabel("mother's years of education")
    ax.set_ylabel("share of mothers")
    ax.set_xlim(2, 18)
    ax.legend(frameon=False)
    fig.tight_layout()


draw_education(df)

# %% [markdown]
# The grey nonsmoker mass piles up at 12 years and above, the high-school-and-beyond block.
# The red smoker mass sits to the left, a real share of it below 12. Less schooling tracks
# lower income, less prenatal care, and a lighter baby through channels that have nothing to
# do with cigarettes. The picture is the bias, made visible. Every variable in the balance
# table tells a version of this story, and they stack.
#
# ## The naive answer, and why it overshoots
#
# Take the difference anyway. A two-node graph, smoking pointing at birthweight and nothing
# else, asks for exactly the raw gap with no adjustment.

# %%
from ergodic import dag, estimate_effect

naive = estimate_effect(
    dag([f"{ds.treatment} -> {ds.outcome}"]),
    ds.data,
    ds.treatment,
    ds.outcome,
    treated=ds.treated,
)
print(f"naive difference: {naive.value:+.1f} g")

# %% [markdown]
# About -275 g. Read it as a causal claim and it says smoking costs a baby more than half a
# pound. But we just saw why it can't be trusted as one. The smokers were also less often
# married, less educated, younger, and lighter on prenatal care, and each of those pushes
# birthweight down on its own. The -275 is smoking plus all of that, added together and
# reported as if it were smoking alone. The back door is open, and association is pouring
# through it alongside the real effect.
#
# How much of the -275 is the back door? Close it and see.
#
# ## Closing the back door
#
# The adjustment graph is honest bookkeeping. Every confounder points at the treatment,
# because it shifts who smokes, and at the outcome, because it shifts birthweight. Smoking
# points at birthweight, the one arrow we want to measure. No graph search, no cleverness,
# just the 17 names we already have written down.

# %%
edges = (
    [f"{c} -> {ds.treatment}" for c in ds.confounders]
    + [f"{c} -> {ds.outcome}" for c in ds.confounders]
    + [f"{ds.treatment} -> {ds.outcome}"]
)
g = dag(edges)

adjusted = estimate_effect(
    g, ds.data, ds.treatment, ds.outcome, treated=ds.treated, strategy="aipw"
)
adjusted

# %% [markdown]
# The estimator is AIPW, the doubly robust default that models both the outcome and who
# smokes and is right if either model is. The mechanics of that, and why you'd reach for it
# over plain regression or plain weighting, are the subject of
# [estimating effects](../learn/03-estimating-effects.py); this page trusts the default and
# reads the number.
#
# The number is about -222 g, with a 95 percent interval that runs roughly -273 to -172. So
# closing the back door moved the estimate by some 50 g, from -275 toward zero. That 50 g was
# the imbalance from the balance table: the part of the raw gap that was really the mothers
# differing, not the smoking. What's left, -222 g, is the effect of smoking with those
# differences held fixed.

# %%
summary = pd.DataFrame(
    {
        "effect (g)": [naive.value, adjusted.value, ds.target["value"]],
        "low": [np.nan, adjusted.interval()[0], np.nan],
        "high": [np.nan, adjusted.interval()[1], np.nan],
    },
    index=["naive difference", "adjusted (AIPW)", "published (Stata manual, AIPW)"],
)
summary.round(1)

# %% [markdown]
# The published adjusted figure on this same extract is about -231 g: the doubly robust (AIPW)
# estimate in the Stata treatment-effects manual, with a standard error of 26 g. Our
# adjustment lands at -222 g, well inside that interval. We started 44 g away from the
# published number and ended 9 g away, having moved in the right direction for the right
# reason. The remaining gap is down to which confounders enter and how each model is fit (the
# manual's own estimators range from about -210 to -240 g), and the published interval and
# ours overlap heavily. This is what a closed back door looks like on real data: not the
# exact textbook digit, but the textbook neighborhood, reached from a graph you could defend
# to a skeptic.
#
# ## What we had to assume
#
# Be exact about the claim. The -222 g holds if the 17 measured confounders close every back
# door between smoking and birthweight. That is an assumption about the world, and no amount
# of data inside this table can prove it. If something we didn't measure moves both smoking
# and birthweight, say a genetic factor, or diet, or stress, then a path is still open and
# the estimate still carries some of it. Adjustment cleans up the confounders you wrote down.
# It is silent about the ones you didn't.
#
# So the estimate is a conditional claim, and the honest move is to stress-test it. The
# cheapest test scrambles the treatment: shuffle the smoking labels at random, breaking any
# real link to birthweight, and re-run the whole pipeline. A method that's working should now
# report nothing, an effect near zero. If a scrambled treatment still produces a big effect,
# the pipeline is finding structure that isn't there, and you'd want to know before you trust
# the real run.

# %%
from ergodic import refute_effect

ref = refute_effect(
    g,
    ds.data,
    ds.treatment,
    ds.outcome,
    method="placebo_treatment",
    strategy="aipw",
    treated=ds.treated,
    estimand_value=adjusted.value,
    n_simulations=20,
    random_state=0,
)
print(f"real estimate:    {ref.estimand_value:+.1f} g")
print(f"placebo estimate: {ref.refuted_value:+.1f} g  (mean over 20 scrambles)")
print(f"verdict: {'passed' if ref.passed else 'FAILED'}")

# %% [markdown]
# The real estimate is -222 g; the placebo, averaged over 20 random shuffles, is about +3 g,
# a rounding error next to the real effect. The pipeline finds nothing when there's nothing
# to find. That's a green light, and it's worth being clear about what color of green. It
# rules out one failure mode, a bug or a quirk manufacturing an effect from noise. It says
# nothing about the unmeasured confounder, which by definition leaves no trace for a placebo
# to catch. Passing this check is necessary, not sufficient. It's the smoke detector, not the
# fire inspection.
#
# That's the case, end to end. A balance table that showed the bias before any model ran, a
# naive -275 g that the imbalance inflated, an adjusted -222 g that landed on the published
# neighborhood, and a placebo check that came back clean. The one input the data couldn't
# supply was the graph: the arrows that said which 17 variables to adjust for and which arrow
# to measure. Here we knew them from how the records are collected. When you don't, you can
# learn the structure from the data itself, the subject of
# [discovering structure](../learn/06-discovering-structure.py).
#
# For the estimator machinery underneath that one `estimate_effect` call, the doubly robust
# fit and the interval, see [estimating effects](../learn/03-estimating-effects.py).
