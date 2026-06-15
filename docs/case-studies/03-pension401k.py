# %% [markdown]
# # Does a 401(k) make you richer? The pension puzzle
#
# Here's a question a benefits team can act on. Does being eligible for a 401(k)
# plan raise a household's financial wealth, and by how much? Pull the SIPP survey and the
# raw answer is loud: households eligible for a plan hold about $19,500 more in net financial
# assets than households without one. Roll out eligibility everywhere and bank twenty grand a
# household, the slide deck writes itself.
#
# The catch is the one this whole series keeps circling. Eligibility isn't handed out by coin
# flip. It rides on your employer, and your employer rides on your income, and income drives
# wealth all on its own. So the $19,500 is the plan and the people tangled together, and the
# people are doing a lot of the lifting. Chernozhukov et al. (2018) made this dataset the
# standard demonstration of double machine learning for exactly that reason: the confounding
# is real, it's nonlinear, and there's a number to check yourself against.
#
# Unlike the simulated worlds in the [learn series](../learn/01-why-causality.py), this is
# real survey data, so nobody wrote the true effect into the code. What we have instead is a
# literature estimate, near $9,000, that careful work converges on. That's the target. The
# job of this page is to get a one-call estimate to land on it, and to be honest about what
# the landing rests on.

# %%
from ergodic.datasets import load_pension401k

ds = load_pension401k()
df = ds.data.to_pandas()

print(f"households: {len(df)}")
print(f"treatment:  {ds.treatment}  (1 = eligible for a 401(k))")
print(f"outcome:    {ds.outcome}  (net financial assets, dollars)")
print(f"target:     ${ds.target['value']:,.0f}  ({ds.target['source']})")

# %% [markdown]
# Nine confounders come with the data, the household features a plan's reach plausibly
# depends on: age, income (`inc`), education, family size, marital status, whether both
# partners earn, whether there's a defined-benefit pension (`db`), an IRA (`pira`), and home
# ownership (`hown`). Eligibility is taken as good as random once you hold these fixed, with
# income carrying most of the weight.
#
# ## Reading the imbalance before estimating anything
#
# Before any model, look at how different the two groups already are. For each confounder,
# take the eligible mean minus the not-eligible mean and divide by the pooled standard
# deviation. That's the standardized mean difference (SMD), and it's the quickest read on
# which variables are out of balance. Anything past about 0.1 is a gap worth worrying about;
# sort by size and the worst offenders rise to the top.

# %%
import numpy as np
import pandas as pd

elig = df[df[ds.treatment] == 1]
noelig = df[df[ds.treatment] == 0]

rows = []
for c in ds.confounders:
    m1, m0 = elig[c].mean(), noelig[c].mean()
    pooled = np.sqrt((elig[c].var(ddof=1) + noelig[c].var(ddof=1)) / 2.0)
    rows.append({"confounder": c, "not eligible": m0, "eligible": m1, "smd": (m1 - m0) / pooled})

balance = pd.DataFrame(rows).set_index("confounder")
balance = balance.reindex(balance["smd"].abs().sort_values(ascending=False).index)
balance.round(3)

# %% [markdown]
# Income is the standout, an SMD of 0.64. Eligible households earn about $46,900 a year
# against $31,500 for the rest, a gap of more than fifteen thousand dollars. Every other
# confounder leans the same way: eligible households more often hold a defined-benefit
# pension, own their home, and carry more education. None of that is the 401(k) doing its
# work. It's the kind of household that ends up with a plan in the first place, and every one
# of those features also pushes wealth up on its own.
#
# Income matters most, so look at it directly. The two distributions are shifted, the eligible
# group sitting clearly to the right, and that shift is the confounding made visible.

# %%
import matplotlib.pyplot as plt


def draw_income(frame, treatment):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 120_000, 40)
    for value, color, label in [(0, "#9aa5b1", "not eligible"), (1, "#d1495b", "eligible")]:
        ax.hist(
            frame.loc[frame[treatment] == value, "inc"].clip(upper=120_000),
            bins=bins,
            density=True,
            alpha=0.55,
            color=color,
            label=label,
        )
    ax.set_xlabel("household income (dollars, top-coded at 120k for the plot)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    return None


draw_income(df, ds.treatment)

# %% [markdown]
# Two things to read off it. The eligible mass is shifted right, which is the imbalance the
# SMD already flagged. And the two distributions still sit on top of each other across the
# whole range, so at almost every income level there are households both with and without a
# plan. That overlap is positivity, the condition that makes adjustment possible at all: you
# can only compare like with like where like-with-like exists. Income tops out far higher
# than this, so the plot is top-coded at 120k for readability; the overlap holds out in the
# tail too.
#
# ## The naive answer
#
# Now the number from the slide deck, computed honestly. A graph with a single arrow from
# eligibility to wealth claims no confounding at all, so estimating on it just contrasts the
# two group means.

# %%
from ergodic import dag, estimate_effect

naive = estimate_effect(dag([f"{ds.treatment} -> {ds.outcome}"]), ds.data, ds.treatment, ds.outcome)
print(f"naive gap: ${naive.value:,.0f}")

# %% [markdown]
# About $19,559, the headline. The balance table already told us why it's too big. Higher
# earners are both more often eligible and wealthier, so the contrast sweeps the income gap
# into the effect and reports the sum. The single arrow is a claim, that nothing else moves
# both eligibility and wealth, and the table is nine counts against it. Income alone, an SMD
# of 0.64 feeding straight into net worth, is enough to know this number is inflated.
#
# ## Closing the back door with double machine learning
#
# The fix is the one from [estimating effects](../learn/03-estimating-effects.py): draw the
# confounders in and adjust for them. Every confounder points at both the treatment and the
# outcome, and eligibility points at wealth. That's the back-door structure, and the graph
# carries it.

# %%
edges = (
    [f"{c} -> {ds.treatment}" for c in ds.confounders]
    + [f"{c} -> {ds.outcome}" for c in ds.confounders]
    + [f"{ds.treatment} -> {ds.outcome}"]
)
g = dag(edges)
print(f"{len(g.nodes)} nodes, {len(g.edges)} edges")

# %% [markdown]
# Adjustment here is not a grid you can split by hand. Nine confounders, several continuous,
# means models in the nuisance slots, and the relationships are not lines. Income's pull on
# wealth curves hard: the move from $30k to $45k buys far less wealth than the move from
# $150k to $165k. Double machine learning is built for this. It cross-fits a model of the
# outcome and a model of the treatment, partials both out, and reads the effect from what's
# left, and it stays honest as long as the learners in the slots are flexible enough to
# follow the curves. The `learner=` argument names those models, one for the outcome slot and
# one for the treatment slot, and it takes anything with the scikit-learn surface. So rather
# than commit to one, try a panel and watch what the choice does to the answer.

# %%
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def learners():
    # Each entry is (outcome regressor, treatment classifier). The treatment classifier's
    # predicted probability is the propensity. xgboost, catboost, and lightgbm would slot in
    # here unchanged; all the slot asks for is the scikit-learn fit/predict surface. (The
    # logistic model sees standardized inputs so its solver converges; the trees do not care.)
    return {
        "linear": (LinearRegression(), make_pipeline(StandardScaler(), LogisticRegression())),
        "k-NN": (KNeighborsRegressor(), KNeighborsClassifier()),
        "random forest": (
            RandomForestRegressor(n_estimators=100, random_state=0),
            RandomForestClassifier(n_estimators=100, random_state=0),
        ),
        "gradient boosting": (
            HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=0),
            HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=0),
        ),
    }


rows = []
for name, (outcome_model, treatment_model) in learners().items():
    est = estimate_effect(
        g,
        ds.data,
        ds.treatment,
        ds.outcome,
        strategy="dml",
        learner={"outcome": outcome_model, "treatment": treatment_model},
        random_state=0,
    )
    low, high = est.interval()
    rows.append({"learner": name, "effect (USD)": est.value, "ci low": low, "ci high": high})

zoo = pd.DataFrame(rows).set_index("learner")
flexible = zoo.loc[["k-NN", "random forest", "gradient boosting"], "effect (USD)"].mean()
zoo.round(0)

# %% [markdown]
# Four learners, one graph, one estimator, and the answer splits in two. The linear pair stops
# near $6,000. The three flexible learners, k-nearest-neighbors and the two tree ensembles, land
# together between about $8,900 and $9,100, right on the literature's $9,000. The intervals
# overlap, but the points separate cleanly. A straight line through the income effect
# under-adjusts; the flexible learners follow the curve. The gap between them is about three
# thousand dollars a household, real money riding on a modeling choice.
#
# Which raises the question a reader is right to ask. Here the literature hands us the answer, so
# we can see the flexible learners are the ones to trust. On your own data there is no answer key.
# How would you know which learner to believe?

# %% [markdown]
# ## Choosing a learner without the answer key
#
# Start with what double machine learning is actually made of. It never models the effect
# directly. It builds two ordinary predictions and combines them: the expected wealth given the
# confounders, and the probability of eligibility given the confounders. Both of those have
# answers you can see. You observe every household's wealth and every household's eligibility, so
# you can grade either prediction the usual way, by holding data out and scoring what the model
# never trained on. The effect is hidden. The two ingredients it is built from are not.
#
# So grade them. For each learner, fit the wealth model and the eligibility model with five-fold
# cross-validation and score the held-out predictions: an out-of-sample R-squared for wealth, an
# out-of-sample AUC for eligibility. Every number below is out-of-fold, each row scored by a model
# that never saw it, the same out-of-fold discipline double machine learning uses internally.

# %%
from sklearn.metrics import (
    d2_absolute_error_score,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

covariates = df[list(ds.confounders)].to_numpy(float)
wealth = df[ds.outcome].to_numpy(float)
eligible = df[ds.treatment].to_numpy(float)


def oof_wealth(model):
    return cross_val_predict(model, covariates, wealth, cv=KFold(5, shuffle=True, random_state=0))


scored = []
for name, (outcome_model, treatment_model) in learners().items():
    fitted = oof_wealth(outcome_model)
    elig_oof = cross_val_predict(
        treatment_model,
        covariates,
        eligible,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        method="predict_proba",
    )[:, 1]
    scored.append(
        {
            "learner": name,
            "wealth R2 (oof)": round(r2_score(wealth, fitted), 3),
            "wealth MedAE (oof)": round(median_absolute_error(wealth, fitted)),
            "wealth D2_abs (oof)": round(d2_absolute_error_score(wealth, fitted), 3),
            "eligibility AUC (oof)": round(roc_auc_score(eligible, elig_oof), 3),
            "effect (USD)": round(float(zoo.loc[name, "effect (USD)"])),
        }
    )

diagnostics = pd.DataFrame(scored).set_index("learner")
diagnostics[["wealth R2 (oof)", "eligibility AUC (oof)", "effect (USD)"]]

# %% [markdown]
# Read down the table and the tidy story falls apart. If predicting the nuisances better meant
# estimating the effect better, the rows would line up. They do not. The linear model predicts
# wealth better than the random forest and comes second only to boosting on eligibility, yet it is
# the one that misses by three thousand dollars. k-nearest-neighbors predicts the worst on both
# counts and still lands on the right answer. The best predictor of who holds a 401(k) is not the
# best estimator of what the 401(k) does.
#
# That decoupling is not a fluke, and it is not bad luck with these four learners. It points at
# three traps in choosing a learner, and past them, a short list of what to look for.

# %% [markdown]
# ## Choosing the right learners
#
# ### Simple is not always better
#
# The linear pair gave the worst estimate on the page, so the lesson reads like "simple loses, reach
# for the most flexible model you have." That is the wrong lesson. Take the two ends of the panel,
# the linear pair and the random forest, and cross them, so each nuisance slot holds either the
# simple model or the flexible one. Four combinations.

# %%
linear_outcome, linear_propensity = learners()["linear"]
forest_outcome, forest_propensity = learners()["random forest"]


def dml_effect(outcome_model, treatment_model):
    return estimate_effect(
        g,
        ds.data,
        ds.treatment,
        ds.outcome,
        strategy="dml",
        learner={"outcome": outcome_model, "treatment": treatment_model},
        random_state=0,
    ).value


grid = pd.DataFrame(
    {
        "propensity: logistic": {
            "outcome: linear": zoo.loc["linear", "effect (USD)"],
            "outcome: forest": dml_effect(forest_outcome, linear_propensity),
        },
        "propensity: forest": {
            "outcome: linear": dml_effect(linear_outcome, forest_propensity),
            "outcome: forest": zoo.loc["random forest", "effect (USD)"],
        },
    }
)
grid.round(0)

# %% [markdown]
# Only one cell is badly wrong, the corner where both models are linear, at $5,994. Make either half
# flexible and most of the bias is gone. A flexible outcome with the plain logistic propensity lands
# at $9,051, the whole gap closed; a flexible propensity with the plain linear outcome reaches
# $8,058, most of it. The both-flexible corner adds almost nothing on top of either single fix.
#
# This is how double machine learning removes bias, made visible. Its error is close to the product
# of the two nuisance errors, one from the outcome model and one from the propensity. A product is
# small when either factor is, so a single good model rescues a single bad one. Simple is not the
# problem. Simple in both slots at once is. The linear pair failed because both of its halves were
# wrong together, not because a line is forbidden, and a linear model paired with a flexible partner
# is a fine estimate here.

# %% [markdown]
# ### Metrics can be deceiving
#
# Two ways the scores can fool you. The first is the fit metric. R-squared is built from squared
# errors, so on a heavy-tailed outcome the rare giant values dominate it. Net financial assets are
# that kind of variable: a median near $1,500, a mean above $18,000, the richest one percent of
# households holding almost a quarter of all positive assets. A model that chases the tail scores
# well on R-squared while predicting the typical household poorly. Drop the squaring and the ranking
# moves.

# %%
diagnostics[["wealth R2 (oof)", "wealth MedAE (oof)", "wealth D2_abs (oof)"]]

# %% [markdown]
# By R-squared the linear model looks second best at predicting wealth. By median absolute error,
# the dollars it misses the typical household by, it is the worst of the four, off by about $13,000
# where gradient boosting is off by $7,000. And by D-squared on the absolute error, the
# outlier-robust cousin of R-squared that scores skill against guessing the median, the linear model
# is negative: in robust terms it predicts the typical household worse than a flat guess, while its
# R-squared claims a fifth of the variance. Only the two tree models clear zero on that scale.
# The variance R-squared rewarded was the tail.
#
# The second way is deeper, and the 2x2 already showed it. Even the right fit metric does not rank
# the effects, because a better fit does not buy a better estimate. The linear outcome model
# has the worst robust fit on the page, and paired with a flexible propensity it still gave $8,058,
# near the full answer. The product structure is why: once one nuisance is good, the other one's
# error gets multiplied by something near zero and stops mattering. So read any fit score, robust or
# not, as a floor rather than a ranking. It can tell you a learner is hopeless. Among the learners
# that clear the floor it cannot tell you which estimate to trust.

# %% [markdown]
# ### The devil is in the details
#
# A few details decide whether a learner helps. The first is preprocessing. k-nearest-neighbors
# looked like the weakest learner in the panel, lowest R-squared and negative robust skill. That was
# not k-nearest-neighbors. Its distance ran on the raw columns, where income in tens of thousands
# swamps everything else, so it was matching on income alone. Put the features on one scale and it
# changes.

# %%
scaled_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
scaled_propensity = make_pipeline(StandardScaler(), KNeighborsClassifier())
scaled_d2 = d2_absolute_error_score(wealth, oof_wealth(scaled_knn))
scaled_effect = dml_effect(scaled_knn, scaled_propensity)
print(
    f"k-NN unscaled : D2_abs {diagnostics.loc['k-NN', 'wealth D2_abs (oof)']:+.3f}   "
    f"effect ${zoo.loc['k-NN', 'effect (USD)']:,.0f}"
)
print(f"k-NN scaled   : D2_abs {scaled_d2:+.3f}   effect ${scaled_effect:,.0f}")

# %% [markdown]
# Scaled, k-nearest-neighbors crosses from negative robust skill to positive, and its estimate moves
# from $8,915 to $9,200, into the flexible cluster. The learner was never weak. The recipe around it
# was. A bad diagnostic is often a preprocessing bug wearing the learner's name.
#
# The second detail is where you spend effort. The 2x2 made the propensity model count as much as
# the outcome model, and in other panels more. The habit is to tune the outcome model and accept
# a default propensity, and the product structure says that backfires when the propensity is the
# misspecified half. The third detail is overlap. A flexible treatment model can push the predicted
# probability of eligibility to zero or one, which k-nearest-neighbors does here. For an estimator
# that divides by the propensity those values are poison and a positivity warning. The
# partialling-out in this DML tolerates them, part of why the panel still agrees, but the spread of
# the propensities is a check the AUC hides.

# %% [markdown]
# ### What to look for
#
# Put the three together and a checklist falls out, for any time you pick the learners behind an
# effect:
#
# - **Distrust a single fit score, and watch R-squared.** On a skewed outcome it rewards
#   tail-chasing. Read an outlier-robust metric beside it, the median absolute error or a rank
#   correlation, and treat any score as a floor rather than a ranking.
# - **Model both nuisances, not the outcome alone.** The bias is a product of two errors, so care on
#   the outcome model cannot save a careless propensity. When unsure, make both flexible.
# - **Mind the recipe.** Scale the features for distance and margin models, and leave the trees
#   alone. A weak-looking learner is often a preprocessing problem.
# - **Watch overlap.** Read the smallest and largest predicted propensity, not the AUC alone. Values
#   at zero or one are a positivity flag.
# - **Trust agreement over fit.** Run a varied panel and believe the number they converge on; the
#   lone outlier, usually the rigid model, is the one to doubt. For one principled score there is
#   the orthogonal, or R-, loss, which grades a learner by how well the two residuals predict each
#   other out of sample instead of by raw fit.
# - **Stress-test the survivor.** Scramble the treatment and the effect should vanish. That placebo
#   comes next, and it catches a pipeline that invents structure on noise.

# %% [markdown]
# ### The check you can run, and the one you cannot
#
# Every learner above leans on the same sentence: eligibility is as good as random once you hold
# income and the other eight fixed. Nothing in the data certifies it. If some unmeasured trait,
# call it a saver's temperament, pushes a household toward both a 401(k) employer and a fat
# balance, the back door is not fully closed and every estimate above drifts back up toward the
# naive $19,500. No learner can see that confounder, so no learner can fix it. The graph is an
# assumption, not a finding.
#
# What you can check is the placebo itself. Scramble the eligibility column so it points at nothing
# and re-estimate. A sound pipeline reports no effect; if it still finds one, the machinery is
# inventing structure. The run is kept small so the build stays quick.

# %%
from ergodic import refute_effect

ref = refute_effect(
    g,
    ds.data,
    ds.treatment,
    ds.outcome,
    method="placebo_treatment",
    strategy="dml",
    estimand_value=flexible,
    n_simulations=12,
    random_state=0,
)
print(f"flexible effect: ${ref.estimand_value:,.0f}")
print(f"placebo (mean):  ${ref.refuted_value:,.0f}   p = {ref.p_value:.3f}")
print(f"passed: {ref.passed}")

# %% [markdown]
# Scrambled eligibility gives about $469, a rounding error next to the roughly $9,000 the flexible
# estimate carries, and the placebo runs almost never match the real effect in size. The pipeline
# isn't manufacturing the answer. That's reassurance, not proof: it rules out one failure mode, a
# leak that would fire on noise, and says nothing about the unmeasured saver who would fool every
# column we do have.
#
# The same table holds a second reading we're leaving on the shelf. Eligibility is a clean
# instrument for actual participation (`p401`): it nudges who joins a plan without touching
# wealth except through joining, which is the textbook setup for an instrumental-variables
# estimate of the effect of participating rather than merely being eligible. The conditional
# IV estimator that reading needs is still future work in the library, so this page stays on
# the adjustment estimate and the eligibility effect it identifies.
#
# For more on the machinery underneath: [estimating effects](../learn/03-estimating-effects.py)
# walks through why double machine learning uses two models and what the interval means, and
# [heterogeneous effects](../learn/04-heterogeneous-effects.py) takes the next step, asking not
# what the average household gains but which households gain the most.
