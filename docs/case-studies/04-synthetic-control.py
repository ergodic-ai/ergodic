# %% [markdown]
# # Proposition 99: building a California that never passed the law
#
# In November 1988, California voters passed Proposition 99. A 25-cent tax per pack, an
# anti-smoking media campaign, school programs, the lot, all live from 1989. Smoking fell in
# California afterward. It was also falling everywhere, for reasons that had nothing to do with
# Sacramento. So the question is the hard one: how much of the drop did the law buy, and how
# much would have happened anyway?
#
# Normally you answer that with a control group. Find places like California that didn't pass
# the law, watch them instead, and treat their path as California's path-not-taken. The trouble
# is that there's one California and no obvious twin. No single state is a stand-in for it, and
# averaging all the other states, as we're about to see, builds in a bias before the law even
# enters.
#
# Abadie, Diamond and Hainmueller (2010) turned this into a method. If no single state is
# California, maybe a weighted blend of several is. Find the blend that tracks California
# before 1989, call it the synthetic control, and read the gap that opens after. The library
# ships their panel, so you can rebuild the whole thing.

# %%
import numpy as np

from ergodic.datasets import load_smoking

ds = load_smoking()
print(f"treated unit: {ds.treated}, law passed at: {ds.at}")
print(f"outcome: {ds.outcome} (packs per capita)")
df = ds.data.to_pandas()
print(f"{df['state'].nunique()} states, {df['year'].min()}-{df['year'].max()}")
df.head()

# %% [markdown]
# One row per state per year: per-capita cigarette sales in packs, for 39 states from 1970 to
# 2000. California is the one that passed Proposition 99. The other 38 are the donor pool, the
# raw material a synthetic California gets built from.
#
# ## California against the average state
#
# The obvious thing is to plot California against the rest. Take the simple average of the
# other 38 states, year by year, and lay California on top of it.

# %%
ca = df[df.state == "California"].set_index("year")[ds.outcome]
others = df[df.state != "California"].groupby("year")[ds.outcome].mean()

pre_years = [y for y in ds.data.periods if y < ds.at]
post_years = [y for y in ds.data.periods if y >= ds.at]
pre_gap = ca.loc[pre_years].mean() - others.loc[pre_years].mean()
print(f"1970: California {ca.loc[1970]:.1f}, average state {others.loc[1970]:.1f}")
print(f"1988: California {ca.loc[1988]:.1f}, average state {others.loc[1988]:.1f}")
print(f"mean pre-1989 gap (California minus average): {pre_gap:+.1f} packs")

# %%
import matplotlib.pyplot as plt


def draw_raw(ca_path, avg_path, at):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ca_path.index, ca_path.to_numpy(), color="#d1495b", lw=1.8, label="California")
    ax.plot(
        avg_path.index,
        avg_path.to_numpy(),
        color="#33415c",
        lw=1.6,
        ls="--",
        label="average of other 38 states",
    )
    ax.axvline(at - 0.5, color="#9aa5b1", lw=1, ls=":")
    ax.set_xlabel("year")
    ax.set_ylabel("cigarette sales (packs per capita)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()


draw_raw(ca, others, ds.at)

# %% [markdown]
# Look at the left half, before the dotted line. California already sits below the average,
# and the two lines were never together: roughly level in 1970, then California pulls away and
# stays away, 24 packs lower by 1988. Across the whole pre-1989 stretch California runs about
# 14 packs under the average state, with no law in sight. California was a low-smoking state to
# begin with, and the average of 38 mostly higher states isn't its counterfactual. The gap
# between these two lines was there before Proposition 99, and most of it has nothing to do
# with the law.
#
# ## The naive answer
#
# Ignore that for a moment and do what the plot invites. Average California's sales after 1989,
# average the other states over the same years, and subtract.

# %%
naive_gap = ca.loc[post_years].mean() - others.loc[post_years].mean()
print(f"California post-1989 mean:    {ca.loc[post_years].mean():.1f} packs")
print(f"other states post-1989 mean:  {others.loc[post_years].mean():.1f} packs")
print(f"naive post-1989 gap:          {naive_gap:+.1f} packs")

# %% [markdown]
# Minus 42 packs. As an estimate of what the law did, it's almost double the published figure,
# and it's easy to see why it tempts: it's the number you can read straight off the chart, and
# it points the right way. The problem is what's inside it. About 14 of those 42 packs are the
# pre-existing level gap, the distance the two lines already kept in the 1970s, before any
# voter saw the proposition. The naive answer bundles that standing difference together with
# whatever the law did and reports the sum as if it were all law.
#
# ## Why it's wrong, and what to do instead
#
# California isn't the average state, so the average state isn't the comparison. What you want
# is a stand-in that already looked like California before 1989: same level of sales, same
# trajectory through the 1970s and 80s. No single state does that. A weighted combination of
# states can. Pick nonnegative weights that sum to one, one per donor state, so the blended
# path matches California across every pre-1989 year. Whatever blend tracks California before
# the law is the best guess at how California would have moved without it. The post-1989 gap to
# that blend is the effect, and the level difference that wrecked the naive answer is gone by
# construction, because the blend is fit to sit exactly where California sat.
#
# This is synthetic control, and the theory behind it, the donor pool, the weights, the
# placebo inference, is the subject of the
# [quasi-experiments guide](../learn/07-quasi-experiments.py). Here we point it at the data.

# %%
from ergodic import synthetic_control

sc = synthetic_control(ds.data, ds.outcome, treated=ds.treated, at=ds.at)
print(f"mean post-1989 gap (the ATT): {sc.value:.2f} packs")
print(f"pre-period tracking error (rmspe): {sc.pre_rmspe:.2f} packs")
print(f"placebo p-value: {sc.p_value:.3f}")

# %% [markdown]
# A mean gap of about minus 21 packs, against the naive minus 42. Half of what the raw
# comparison claimed was never the law; it was California being California. And the diagnostic
# to read before the headline is `pre_rmspe`, the tracking error over the 19 pre-1989 years:
# 2.34 packs. The synthetic California followed the real one to within a couple of packs for
# nearly two decades, so the post-1989 gap is measured against a stand-in that earned some
# trust. A synthetic control with a bad pre-period fit is a comparison against noise, whatever
# its gap shows.
#
# Which states make up synthetic California? Read the weights.

# %%
weights = {s: round(w, 3) for s, w in sorted(sc.weights.items(), key=lambda kv: -kv[1]) if w > 0.01}
top5 = dict(list(weights.items())[:5])
print("top donor weights:", top5)
print(f"these five carry {sum(top5.values()):.2f} of the weight")
print(f"{len(weights)} states carry weight above 0.01; the other donors sit below")

# %% [markdown]
# Utah, Nevada, New Mexico, Idaho and Montana do most of the work, together about 0.71 of the
# weight. Tobacco-control history aside, these are mountain-west and low-population states whose
# pre-1989 sales tracked California's, which is the only thing the fit was asked to match.
# California is rebuilt out of the states that moved like it. The heavy donors are its neighbors
# in cigarette sales, whatever the map says about geography.
#
# Now the picture the whole study rests on: observed California against synthetic California,
# with the law's arrival marked.


# %%
def draw_synthetic(effect, at):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(effect.periods, effect.observed, color="#d1495b", lw=1.8, label="California, observed")
    ax.plot(
        effect.periods,
        effect.synthetic,
        color="#33415c",
        lw=1.6,
        ls="--",
        label="synthetic California",
    )
    ax.axvline(at - 0.5, color="#9aa5b1", lw=1, ls=":")
    ax.set_xlabel("year")
    ax.set_ylabel("cigarette sales (packs per capita)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()


draw_synthetic(sc, ds.at)

# %% [markdown]
# Before the dotted line the two paths are nearly one line: synthetic California rides every
# bump real California takes, which is what the low pre-period rmspe meant. After 1989 they
# split, and the gap widens through the 1990s as the tax and the campaign compound.

# %%
gap = dict(zip(sc.periods, sc.gaps, strict=True))
print(f"gap in 1989 (year one): {gap[1989]:+.1f} packs")
print(f"gap by 2000:            {gap[2000]:+.1f} packs")
print(f"observed 2000 {sc.observed[-1]:.1f} vs synthetic 2000 {sc.synthetic[-1]:.1f}")

# %% [markdown]
# By 2000 the gap reaches about minus 28 packs: California sold 41.6 packs per capita, its
# synthetic twin would have sold near 70. That endpoint matches the published headline of
# roughly minus 26 packs per capita by 2000. Year one is small, minus 8, and the effect builds
# from there, the way a tax and a slow cultural campaign would.
#
# ## The honest caveat, and the check that ships with the method
#
# One treated unit means there's no sampling story to lean on, no standard error in the usual
# sense. So how do you know a 21-pack gap is real and not what any state might show by chance?
# Synthetic control answers this with its own built-in check, the placebo permutation. Pretend
# a donor state passed Proposition 99 in 1989, fit it a synthetic control from the remaining
# donors, and read its gap. Do that for every donor. If California's gap is just noise, plenty
# of placebo states should throw up gaps as large. If it stands out, the effect is real.

# %%
placebo_means = np.array([abs(np.mean(g)) for g in sc.placebo_gaps.values()])
n_bigger = int((placebo_means >= abs(sc.value)).sum())
print(f"{len(placebo_means)} placebo states (each donor treated in turn)")
print(f"California's mean post-1989 gap, absolute: {abs(sc.value):.1f} packs")
print(f"placebos with a raw gap at least as large: {n_bigger}")
print(f"placebo p-value (gap scaled by pre-period fit): {sc.p_value:.3f}")

# %% [markdown]
# Read the raw gaps first. Of the 38 placebo states, 2 show a mean gap as large as California's
# 21 packs. A raw gap on its own is a weak signal, though: a donor the synthetic control could
# never track well in the first place will drift far afterward for no interesting reason. So the
# p-value scales each gap by how tightly that state was tracked before 1989, the post-to-pre
# rmspe ratio, which rewards a state that was followed closely and then broke away. On that
# scale California's divergence is beaten by exactly one of the 38 donors, putting its rank at 2
# of 39 and the p-value at 2/39, about 0.05. This is the refutation for this design. The placebo
# asks what a placebo-treatment shuffle asks of a backdoor estimate, did a gap this clean show
# up where we know there was no law, and across the donor pool almost nothing does.
#
# What the number still rests on, said plainly. The convex-donor-span assumption:
# California's no-law path has to be reachable as a blend of donor states, and the low
# `pre_rmspe` is the evidence that it was. The no-anticipation assumption: nobody changed their
# smoking in 1988 because they saw the 1989 law coming, which would contaminate the pre-period
# the fit trusts. And one honest limitation of this fit specifically. The library fits the
# weights on the outcome pre-period alone, the cigarette-sales path, so the donor mix can come
# out a little different from the paper's, which also matches on income, beer sales, and the
# age distribution. The story is the same; the exact weights are not the paper's to the third
# decimal.
#
# ## A second case: German reunification
#
# The method travels. Abadie, Diamond and Hainmueller (2015) ran the same idea on West Germany.
# In 1990 reunification absorbed East Germany, a shock to the West German economy with no
# control group. Build a synthetic West Germany from other OECD countries, fit to its
# per-capita GDP before 1990, and read the gap.

# %%
from ergodic.datasets import load_germany

ds2 = load_germany()
sc2 = synthetic_control(ds2.data, ds2.outcome, treated=ds2.treated, at=ds2.at)
print(f"treated: {ds2.treated}, at: {ds2.at}, outcome: {ds2.outcome} (GDP per capita)")
print(f"mean post-1990 gap: {sc2.value:.0f} per year")
print(f"pre-period rmspe: {sc2.pre_rmspe:.0f}")


# %%
def draw_germany(effect, at):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(effect.periods, effect.observed, color="#d1495b", lw=1.8, label="West Germany")
    ax.plot(
        effect.periods,
        effect.synthetic,
        color="#33415c",
        lw=1.6,
        ls="--",
        label="synthetic West Germany",
    )
    ax.axvline(at - 0.5, color="#9aa5b1", lw=1, ls=":")
    ax.set_xlabel("year")
    ax.set_ylabel("GDP per capita")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()


draw_germany(sc2, ds2.at)

# %% [markdown]
# The gap opens slowly and then widens. The two lines hold together through the early 1990s,
# even cross briefly as reunification spending lifted output for a year or two, then West
# Germany falls below its synthetic counterpart and stays there, the gap reaching about minus
# 3,800 per capita by 2003. Averaged over the post-1990 years it runs near minus 1,600 per
# capita per year. Same machinery as California, a different outcome and a different shock, and
# the line you read is still observed against synthetic.
#
# For the method behind both cases, the donor pool, the weights on the simplex, and where
# synthetic control fits among difference-in-differences and the other panel designs, see
# [quasi-experiments](../learn/07-quasi-experiments.py).
