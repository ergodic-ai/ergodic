# %% [markdown]
# # Quasi-experiments: when the rollout is the experiment
#
# A retail chain rolls a loyalty-app promo out to its stores in two waves. Headquarters
# picked the order, and headquarters did what headquarters does: the best regions went
# first. A quarter later you're asked what the promo earned, and nobody ran an experiment.
#
# Every guide so far closed back doors by measuring the confounder and adjusting for it.
# Here the confounders are awkward ones. Which stores adopted early depends on store
# quality, which nobody logged. How sales move from week to week depends on season and on
# the chain's growth, which touch every store at once. What you do have is a panel: the
# same 100 stores observed over the same 20 weeks. The designs in this guide use that
# structure directly. Each store is seen both before and after, so its own level can be
# differenced out without ever measuring it; stores without the promo live through the
# same weeks, so the calendar can be differenced out too. The rollout becomes the
# experiment, if you'll grant one assumption about trends, and most of this guide is
# about earning that assumption and checking it where it can be checked.
#
# As always, we write the answer into the simulation first.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n_stores, n_weeks = 100, 20
weeks = np.arange(1, n_weeks + 1)

# Store quality drives both the baseline and the rollout order. Nobody logged it.
quality = rng.normal(0.0, 1.0, size=n_stores)
rank = np.argsort(-quality)
adopt_week = np.zeros(n_stores, dtype=int)  # 0 means the store never adopts
adopt_week[rank[:30]] = 9  # the best stores go first
adopt_week[rank[30:60]] = 13  # the middle goes second


def tau(e):
    return 4.0 + 1.0 * e  # the answer key: +4 in the adoption week, +1 more each week on


store = np.repeat(np.arange(n_stores), n_weeks)
week = np.tile(weeks, n_stores)
calendar = 0.9 * (week - 1) + 3.5 * np.sin(2.0 * np.pi * week / 10.0)  # growth + season
on = (adopt_week[store] > 0) & (week >= adopt_week[store])
sales = (
    100.0
    + 12.0 * quality[store]  # the store's own level, set by quality
    + calendar  # the same calendar for everyone
    + np.where(on, tau(week - adopt_week[store]), 0.0)  # the planted effect
    + rng.normal(0.0, 2.0, size=n_stores * n_weeks)
)
df = pd.DataFrame(
    {
        "store": [f"s{i:03d}" for i in store],
        "week": week,
        "promo": on.astype(float),
        "sales": sales,
    }
)
df.head()

# %% [markdown]
# Weekly sales per store, in thousands of dollars. Three things are planted. Store quality
# sets each store's level and decides the rollout: the 30 best stores adopt in week 9, the
# next 30 in week 13, the bottom 40 never do. One calendar, growth plus a seasonal wave,
# moves every store in every week. And the promo pays tau(e) = 4 + e, where e counts weeks
# since adoption: 4 thousand in the adoption week, one thousand more each week after, an
# effect that grows the longer a store runs it.

# %%
n_early, n_late = int((adopt_week == 9).sum()), int((adopt_week == 13).sum())
n_never = int((adopt_week == 0).sum())
level_gap = 12.0 * (quality[adopt_week > 0].mean() - quality[adopt_week == 0].mean())
print(f"{n_early} stores adopt in week 9, {n_late} in week 13, {n_never} never adopt")
print("planted effect: tau(e) = 4 + e thousand per week, e = weeks since adoption")
print(f"planted level gap, adopters minus never-adopters: {level_gap:+.1f} (promo or not)")

# %% [markdown]
# ## Two obvious answers, both wrong
#
# The simulation can grade the two comparisons everyone reaches for: promo stores versus
# the rest, and after versus before.

# %%
df["adopter"] = df.groupby("store")["promo"].transform("max") == 1.0

wk20 = df[df.week == 20]
wk05 = df[df.week == 5]
xs_post = wk20[wk20.adopter].sales.mean() - wk20[~wk20.adopter].sales.mean()
xs_pre = wk05[wk05.adopter].sales.mean() - wk05[~wk05.adopter].sales.mean()

adopters = df[df.adopter]
on_mean = adopters[adopters.promo == 1.0].groupby("store").sales.mean()
off_mean = adopters[adopters.promo == 0.0].groupby("store").sales.mean()
before_after = (on_mean - off_mean).mean()

print(f"cross-section, week 20: promo stores minus the rest: {xs_post:+.1f}")
print(f"same comparison in week 5, before any store adopted: {xs_pre:+.1f}")
print(f"before-after, adopting stores only:                  {before_after:+.1f}")

# %% [markdown]
# The cross-section says the promo is worth almost 32 thousand a week. Run the identical
# comparison in week 5, when no store had the promo, and it already says 19: that's the
# level gap headquarters built into the rollout by sending winners first, and it accounts
# for most of the week-20 number. The promo didn't move sales toward those stores; the
# rollout moved toward high-sales stores.
#
# Before-after looks safer, same stores either side, and lands near 17 against a truth
# that never exceeds 15 for any single store-week. The chain was growing by 0.9 a week
# the whole time, and adoption sits in the second half of the panel, so the calendar gets
# billed to the promo.
#
# Each answer fails on one axis. Compare across stores and the store levels poison it;
# compare across time and the calendar poisons it. The panel has both axes, so use each
# one to cancel the other.
#
# ## Difference-in-differences
#
# For each adopting store, take its change in sales from before adoption to after. The
# store's level, however much headquarters liked it, sits in both halves and subtracts
# away. Then subtract the change that never-adopters showed over the same weeks, which is
# the calendar's contribution, since the calendar is all that moved them. What's left is
# the effect, resting on one assumption called parallel trends: had the adopters not
# adopted, their sales would have moved week to week the way the never-adopters' sales
# moved, levels apart but trajectories parallel. That's a claim about a world nobody
# observed, the adopters without their promo, so no test on this data can prove it; you
# argue it, and shortly we'll see how to probe it.
#
# One wrinkle: adoption is staggered across two waves, so a single before/after split
# doesn't exist. `did` reads adoption timing off the `promo` column, notices the two
# waves, and switches to its group-time flavor (Callaway and Sant'Anna): one small
# difference-in-differences per cohort and week, each against that cohort's last
# pre-adoption week and the never-adopters, aggregated with cohort-size weights. The
# matching truth is the same weighted average of tau over each cohort's observed post
# weeks, computed here from the answer key.

# %%
from ergodic import did, event_study, panel, synthetic_control

data = panel(df[["store", "week", "promo", "sales"]], entity="store", time="week")
est = did(data, "sales", "promo", n_boot=200, random_state=0)
lo, hi = est.interval()

groups = sorted(set(adopt_week[adopt_week > 0]))
sizes = np.array([(adopt_week == g).sum() for g in groups])
per_group = np.array([tau(np.arange(n_weeks - g + 1)).mean() for g in groups])
att_true = float((sizes * per_group).sum() / sizes.sum())

print(f"flavor picked: {est.flavor} (two adoption weeks, vs {est.comparison} stores)")
print(f"did estimate of the ATT: {est.value:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
print(f"planted ATT, same weighting: {att_true:.2f}")

# %% [markdown]
# The estimate is 8.66 against a planted 8.50, with an interval that covers it, from data
# where the naive answers said 17 and 32. The uncertainty comes from a unit bootstrap,
# resample stores with replacement and recompute, and the result carries its identifying
# assumptions along in `est.design`, so the claim travels with the number.
#
# ## The event study: seeing the assumption
#
# Parallel trends can't be verified, but it can be embarrassed. Each cohort has weeks
# before its adoption, and the same machinery runs there: difference the cohort against
# the never-adopters between two weeks that are both pre-adoption, where the promo can't
# be the explanation and the right answer is zero. `event_study` computes one estimate per
# event time, weeks relative to adoption: negative event times (leads) are those placebo
# checks, and nonnegative ones (lags) trace the effect as exposure accumulates. Event
# time -1 is the base week every difference is taken against, zero by construction, so
# it's omitted. We stop the lags at 7 because that's as far as both cohorts are observed;
# beyond 7 only the early wave remains and the line would change meaning mid-figure.

# %%
import matplotlib.pyplot as plt

es = event_study(data, "sales", "promo", leads=6, lags=7, n_boot=200, random_state=0)
worst_lead = max(abs(e.estimate) for e in es.by_event_time if e.event_time < 0)
print(f"largest lead estimate, absolute: {worst_lead:.2f}")


def draw_event_study(effects):
    fig, ax = plt.subplots(figsize=(7, 4))
    times = np.array([e.event_time for e in effects])
    mid = np.array([e.estimate for e in effects])
    low = np.array([e.interval[0] for e in effects])
    high = np.array([e.interval[1] for e in effects])
    ax.errorbar(
        times,
        mid,
        yerr=[mid - low, high - mid],
        fmt="o",
        ms=4.5,
        lw=1.4,
        capsize=3,
        color="#33415c",
        label="estimate, 95% band",
    )
    post = times[times >= 0]
    ax.plot(post, tau(post.astype(float)), ls="--", lw=1.2, color="#d1495b", label="planted tau(e)")
    ax.axhline(0.0, color="#9aa5b1", lw=1)
    ax.axvline(-0.5, color="#9aa5b1", lw=1, ls=":")
    ax.set_xlabel("event time (weeks since adoption)")
    ax.set_ylabel("effect on weekly sales (thousands)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()


draw_event_study(es.by_event_time)

# %% [markdown]
# This one picture is the whole argument. Left of the dotted line, the leads sit on zero,
# none larger than 0.45 in absolute value and every band straddling zero: in the six
# weeks before each wave adopted, adopters and never-adopters moved in lockstep. That is
# the parallel-trends check made visible, and a lead drifting away from zero here would
# be the design telling you not to trust its own headline. Right of the line, the lags
# climb the planted staircase, 4 in the adoption week, 11 by event week 7, and the dashed
# answer key runs through every band. A single ATT would have hidden that growth; the
# event study is usually worth reading before the headline number.
#
# ## Synthetic control: one market, no twin
#
# Later that quarter the pricing team cut prices in the chain's flagship market, Austin,
# and wants the revenue effect. Now the panel is thin in the other direction: one treated
# unit, once. There's no cohort to average, and a plain DiD against "all other markets"
# would assume Austin trends like the average market, which a flagship has no reason to
# do.
#
# Synthetic control takes the donor pool seriously instead of averaging it. Find
# nonnegative weights, summing to one, that make a blend of donor markets track Austin's
# revenue before the cut. Whatever blend tracks it is the best available stand-in for
# Austin without the cut, and the gap that opens after the cut is the effect. We build
# the world so a true blend exists: Austin is one half Omaha, three tenths Tucson, and a
# fifth Reno, plus noise, plus 6 thousand a week once the cut lands in week 21. Those
# three donors get distinctive profiles, their own levels, drift exposures, and seasonal
# swings, so that tracking Austin forces the fit to find them among the nineteen.

# %%
rng_sc = np.random.default_rng(30)
donor_names = [
    "akron",
    "boise",
    "dayton",
    "durham",
    "fresno",
    "lincoln",
    "lubbock",
    "norfolk",
    "omaha",
    "provo",
    "reno",
    "richmond",
    "savannah",
    "spokane",
    "tacoma",
    "tucson",
    "tulsa",
    "wichita",
    "yakima",
]
n_wk = 28
wk = np.arange(1, n_wk + 1)
national = np.cumsum(rng_sc.normal(0.25, 0.6, size=n_wk))  # shared drift
seasonal = 2.5 * np.sin(2.0 * np.pi * wk / 13.0)  # shared cycle
exposure = rng_sc.uniform(0.4, 1.0, size=len(donor_names))
swing = rng_sc.uniform(-1.0, 1.0, size=len(donor_names))
level = rng_sc.uniform(38.0, 46.0, size=len(donor_names))
profile = {"omaha": (52.0, 1.5, 0.9), "tucson": (48.0, 1.35, -0.6), "reno": (50.0, 1.2, 0.3)}
for name, (lvl, exp_, sw) in profile.items():  # the three donors austin is made of
    i = donor_names.index(name)
    level[i], exposure[i], swing[i] = lvl, exp_, sw
paths = (
    level[:, None]
    + exposure[:, None] * national[None, :]
    + swing[:, None] * seasonal[None, :]
    + rng_sc.normal(0.0, 0.5, size=(len(donor_names), n_wk))
)

mix = {"omaha": 0.5, "tucson": 0.3, "reno": 0.2}  # the answer key for the weights
austin = sum(w * paths[donor_names.index(name)] for name, w in mix.items())
austin = austin + rng_sc.normal(0.0, 0.4, size=n_wk)
cut_week, lift = 21, 6.0
austin[wk >= cut_week] += lift  # the answer key for the effect

mdf = pd.DataFrame(
    {
        "market": np.repeat([*donor_names, "austin"], n_wk),
        "week": np.tile(wk, len(donor_names) + 1),
        "revenue": np.concatenate([paths.reshape(-1), austin]),
    }
)
print(f"planted: austin = 0.5 omaha + 0.3 tucson + 0.2 reno, +{lift:.0f} from week {cut_week}")

# %%
markets = panel(mdf, entity="market", time="week")
sc = synthetic_control(markets, "revenue", treated="austin", at=21)

top = {m: round(w, 2) for m, w in sorted(sc.weights.items(), key=lambda kv: -kv[1]) if w > 0.01}
print(f"donor weights above 0.01: {top}")
print(f"estimated lift: {sc.value:.2f}   planted: {lift:.2f}")
print(f"pre-cut tracking error (rmspe): {sc.pre_rmspe:.2f}")
print(f"placebo p-value: {sc.p_value:.3f}")

# %% [markdown]
# Out of 19 donors, the three we planted carry 0.99 of the weight and the other sixteen
# sit below 0.01 each. Omaha anchors at 0.49, on its planted half; Tucson and Reno come
# out 0.22 and 0.28 rather than the planted 0.30 and 0.20, because over 20 noisy weeks a
# blend that trades some Tucson for some Reno tracks Austin almost as well, and weights
# between similar donors are weakly pinned even when the gap is solid. The estimated lift
# of 6.08 sits on the planted 6. And `pre_rmspe` is the diagnostic to read before any of
# that: 0.35 says the blend tracked Austin to within about a third of a thousand per week
# for 20 weeks, so the post-cut gap is measured against a stand-in that earned some
# trust. A synthetic control with a poor pre fit is a comparison against noise, whatever
# its post gap shows.
#
# With one treated unit there's no sampling story, so inference is a placebo test: rerun
# the entire fit on each donor as if it had been treated in week 21 and score every unit
# by the ratio of post-cut to pre-cut tracking error, which is how much its path diverges
# after the cut relative to how well it could be tracked before. The p-value is Austin's
# rank among those placebos, and Austin beats all 19, so p = 1/20 = 0.05, the smallest
# value a 19-donor pool can produce.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sc.periods, sc.observed, color="#d1495b", lw=1.6, label="austin, observed")
ax.plot(sc.periods, sc.synthetic, color="#33415c", lw=1.6, ls="--", label="synthetic austin")
ax.axvline(cut_week - 0.5, color="#9aa5b1", lw=1, ls=":")
ax.set_xlabel("week")
ax.set_ylabel("weekly revenue (thousands)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()

# %% [markdown]
# Two paths ride every bump together for 20 weeks, then split at the dotted line and stay
# about 6 apart. The picture is the estimator: the gap between the lines after the cut,
# averaged, is the number reported above.
#
# ## What these designs rest on
#
# Each design bought its answer with an assumption about a world nobody observed, and
# it's worth saying plainly what each one cannot check. Parallel trends is a claim about
# the adopters' counterfactual trajectory; the leads can embarrass it, and flat leads
# raise your confidence, but trends that ran parallel for six weeks can still split in
# week seven for a reason that arrives with the treatment, a competitor reacting to the
# rollout, say. Synthetic control needs Austin to be reachable from its donors: a flagship
# that's grown unlike any blend of other markets leaves a bad `pre_rmspe`, and a gap
# measured against a blend that never tracked means little. And both designs assume no
# anticipation. A store that stocks up the week before its promo, or customers who delay
# purchases waiting for a known price cut, contaminate the "before" that every difference
# in this guide rests on, and the cure is structural: date the treatment from its
# announcement rather than its launch.
#
# When there's no comparison pool at all, only a single series with a change date in it,
# `interrupted_time_series` fits the pre-period and extrapolates it; that and the flavors,
# pools, and failure modes of everything above are in the [panel guide](../panel.md).
