# %% [markdown]
# # Decision points: when the process picks a lane
#
# [Guide nine](09-process-intelligence.py) turned an event log into a map and a case
# table. The map is honest about traffic: who hands off to whom, how often, how slow. It
# says nothing about structure. Stare at one and you can't tell whether triage always
# runs, whether two lanes exclude each other, or whether a rare arc is the workflow or a
# typo. Arrows record what followed what; they don't commit to a model of the process.
#
# This guide mines the model and puts it to work three ways. The miner turns the log
# into a process tree, a statement about structure sharp enough to be wrong. Conformance
# checking replays the log against the tree and names every case that deviates, move by
# move. And the tree's exclusive split, the point where the process picks a lane for
# each case, is a decision, and a decision is a treatment: causal ground again, with the
# treatment found by a miner instead of named by hand.
#
# The operation is an insurance claims office. A claim gets filed and triaged, then
# routed down one of two lanes: `fast_track`, where a desk adjuster settles it from the
# file, or `full_review`, where a senior adjuster works it up. Management wants the
# thorough lane priced, because someone has proposed widening the fast lane. We simulate
# the office with the answer planted in the code, the move every guide makes. A
# complexity score lands on each claim at intake. Complexity drives the routing, since
# hard claims go to review, and it drives the settlement work on its own: the hardest
# claims carry about 24 hours more of it than the easiest. The review itself adds three
# hours, no more. Timestamps count hours, as in guide nine. And we plant a clerical
# wrinkle to give the miner something to chew on: in about two percent of claims the
# lane assignment was keyed into the system before the triage entry, so the log shows
# those two steps out of order.

# %%
import numpy as np
import pandas as pd

from ergodic import event_log


def claims_world(n, slope, seed):
    rng = np.random.default_rng(seed)
    complexity = rng.uniform(0.0, 1.0, size=n)
    p_review = 1.0 / (1.0 + np.exp(-slope * (complexity - 0.5)))  # hard claims get reviewed
    reviewed = rng.uniform(size=n) < p_review
    remaining = (  # hours from the lane assignment to settlement
        12.0
        + 24.0 * complexity  # complexity brings its own work
        + 3.0 * reviewed  # the answer key: full review adds three hours
        + rng.normal(0.0, 1.5, size=n)
    )
    start = rng.uniform(0.0, 720.0, size=n)  # filings spread over a 30-day month
    triage_gap = rng.uniform(0.5, 1.5, size=n)
    route_gap = rng.uniform(0.5, 2.0, size=n)
    swapped = rng.uniform(size=n) < 0.02  # the clerical slips

    rows = []
    for i in range(n):
        claim = f"C{i:04d}"
        lane = "full_review" if reviewed[i] else "fast_track"
        t_triage = start[i] + triage_gap[i]
        t_route = t_triage + route_gap[i]
        rows.append((claim, "file", start[i], complexity[i]))
        if swapped[i]:  # the lane entry landed before the triage entry
            rows.append((claim, lane, t_triage, complexity[i]))
            rows.append((claim, "triage", t_route, complexity[i]))
            t_lane = t_triage
        else:
            rows.append((claim, "triage", t_triage, complexity[i]))
            rows.append((claim, lane, t_route, complexity[i]))
            t_lane = t_route
        rows.append((claim, "settle", t_lane + remaining[i], complexity[i]))
    df = pd.DataFrame(rows, columns=["claim", "step", "t", "complexity"])
    log = event_log(df, case="claim", activity="step", timestamp="t")
    return log, reviewed, remaining, swapped


log, reviewed, remaining, swapped = claims_world(1500, slope=3.0, seed=9)
print(log)
print(f"claims routed to full_review: {int(reviewed.sum())} of 1500")
print(f"claims with the lane keyed in before triage: {int(swapped.sum())}")
print("planted effect of the full_review lane: +3.0 hours")

# %% [markdown]
# ## From map to model
#
# A process tree is built from four operators: `seq` runs its children in order, `xor`
# runs exactly one of them, `par` interleaves them, and `loop` repeats. Every activity
# sits in exactly one leaf. That's a stronger language than a map speaks, and
# `discover_model`, an inductive miner in the sense of Leemans, Fahland, and van der
# Aalst, recovers a tree from the log by cutting its directly-follows graph recursively:
# find a partition of the activities that one operator explains, split the log along
# it, recurse. Taken literally, our log mines to this:

# %%
from ergodic import discover_model

literal = discover_model(log)
print(literal.tree)

# %% [markdown]
# Read it aloud: a claim files; then the lane choice (one of `fast_track` or
# `full_review`) and triage run in parallel, in either order; then it settles. Parallel?
# The office swears triage comes first. But 32 claims carry the lane entry ahead of the
# triage entry, so the log holds both orders, and a miner that takes every arc at face
# value concludes the two steps are concurrent. The model isn't wrong about the log;
# it's wrong about the office. Thirty-two slips bought a weaker model.
#
# The `noise` knob exists for logs like this one. At `noise=0.1` the miner drops, at
# every level of the recursion, any directly-follows arc carrying less than a tenth of
# the heaviest arc leaving the same activity, and it reports what it filtered instead of
# hiding it:

# %%
from ergodic.process import is_sound

model = discover_model(log, noise=0.1)
print(model)
print(f"the workflow net is sound: {is_sound(model.net)}")

# %% [markdown]
# Five rare arcs gone (the slips' footprints), and the tree now says what the office
# says: file, then triage, then exactly one of the two lanes, then settle. The `xor`
# node is the routing decision, recovered from timestamps alone. The model is also
# executable, not a drawing: `model.net` is the tree's workflow net, and `is_sound`
# checks by walking its reachable states that every run can finish, no run strands
# tokens, and every step can fire on some path. Soundness is what makes the next move
# meaningful, because replaying a log against a model that can deadlock proves nothing.
#
# ## Does the log fit the model?
#
# A mined model fits its own log by construction, give or take what the filter dropped.
# The question that matters arrives when the model is a claim you intend to keep: which
# cases obey it, and what do the others do instead? `conformance` answers twice. Token
# replay pushes each trace through the net and counts the tokens it had to conjure or
# abandon, a fast and slightly forgiving score. Optimal alignments walk the trace and
# the model together and name every deviation: a log move is an event the model can't
# mirror, a model move is a step the model requires that the trace skipped.

# %%
from ergodic import conformance

conf = conformance(log, model)
print(conf)
print(f"share of claims fitting the model: {conf.share_fitting:.1%}")
for v in conf.variants:
    path = " -> ".join(v.sequence)
    print(f"{path:40} {v.n_cases:4d} cases   alignment cost {v.cost}")

# %% [markdown]
# Four distinct paths. The two the model accepts cover 1468 claims, and the two it
# rejects are exactly the 32 planted slips, each at alignment cost 2. Here is the more
# common deviant read move by move:

# %%
off = next(v for v in conf.variants if v.cost > 0)
print(" -> ".join(off.sequence), f"({off.n_cases} claims)")
print("   ".join(f"{kind}:{act}" for kind, act in off.alignment))

# %% [markdown]
# As a sentence: `file` matched; the model had to play `triage` on its own because the
# trace wasn't ready to (a model move); the lane event matched; and the trace's late
# triage entry found no model step left to mirror (a log move). These claims logged the
# lane before triage. Replay and alignments agree here, but they don't have to: replay
# can hand part credit to a trace that never forms a real run of the net, so alignments
# are the number to report and replay the cheap screen to run first on a huge log.
#
# One more check, because it teaches the trade. Score the log against the literal,
# unfiltered model from before:

# %%
print(conformance(log, literal))

# %% [markdown]
# A perfect fit, for the model that claims less. The parallel reading accepts both
# orders, so nothing can deviate. Weakening the model until the log fits is always
# available and is usually the wrong trade; the filtered model keeps the strong
# sequential claim and bills the 32 exceptions to the data-entry process, where they
# belong.
#
# ## The split as a decision
#
# So far the model is structure and trust, and nothing has needed the word "cause". The
# `xor` node changes that: an exclusive split is the process making a choice per case.
# `decision_points` locates every one and counts the traffic through it:

# %%
from ergodic import decision_points

(point,) = decision_points(log, model)
print(point)

# %% [markdown]
# One split, at path `(2,)` of the tree (the root's third child), lanes near balance:
# 745 claims took `fast_track` and 755 took `full_review`. Pricing it wants an outcome
# clocked from the decision itself, so we use remaining time, the hours from the lane
# assignment to settlement, because time a claim spent before the split can't be an
# effect of the split. The naive answer is one group-by away:

# %%
naive = float(remaining[reviewed].mean() - remaining[~reviewed].mean())
print(f"mean remaining time, fast_track:  {remaining[~reviewed].mean():.1f} hours")
print(f"mean remaining time, full_review: {remaining[reviewed].mean():.1f} hours")
print(f"the naive lane gap:               {naive:+.1f} hours   (planted: +3.0)")

# %%
import matplotlib.pyplot as plt


def draw_lanes(remaining, reviewed):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(remaining.min(), remaining.max(), 40)
    for mask, name, color in (
        (~reviewed, "fast_track", "#33415c"),
        (reviewed, "full_review", "#d1495b"),
    ):
        values = remaining[mask]
        ax.hist(values, bins=bins, alpha=0.55, color=color, label=name)
        ax.axvline(float(values.mean()), color=color, lw=1.6, ls="--")
    ax.set_xlabel("remaining time after the lane assignment (hours)")
    ax.set_ylabel("claims")
    ax.legend(frameon=False)
    fig.tight_layout()


draw_lanes(remaining, reviewed)

# %% [markdown]
# The thorough lane runs 8.8 hours behind, and the dashed means stand almost nine hours
# apart where we planted three. The other 5.8 hours never belonged to the lane. They
# belong to complexity, which decided the lane and brought the work along: full review
# collects the office's hardest claims and is then billed for the difficulty it
# attracted. It's the same arithmetic that inflated guide nine's review step, now read
# off a mined decision point instead of a hand-named activity.
#
# ## The decision table
#
# `decision_table` does for one split what `case_table` did for a whole log: one row per
# claim, with the timing discipline built in. The branch taken is the treatment (a
# column named `branch`). The landmark is the moment a claim reveals its lane, its first
# event inside the split's alphabet, and every covariate is measured strictly before
# that moment, so nothing downstream of the decision can leak into the adjustment set.
# We name three covariates: the intake complexity score, the time elapsed when the lane
# was assigned, and how many other claims were open at that moment (the default
# `covariates="auto"` would add the remaining prefix features).

# %%
from ergodic import decision_table

dt = decision_table(
    log,
    model,
    covariates=["complexity", "elapsed_at_landmark", "workload_at_landmark"],
)
print(dt)
print(f"covariates: {dt.covariates}")
print(dt.knowledge.permits_edge("complexity", "branch"))
print(dt.knowledge.permits_edge("remaining_time", "branch"))

# %% [markdown]
# The table ships its own `DomainKnowledge`, three tiers ordering covariates before
# branch before outcome, so discovery can't point an arrow backwards in time. It also
# ships a positivity report, and that's the part to read before any estimate:

# %%
pos = dt.positivity
print(f"branches:      {pos.branches}")
print(f"counts:        {pos.counts}   min_share: {pos.min_share:.2f}")
print(f"overlap_share: {pos.overlap_share:.2f}   flagged: {pos.flagged}")

# %% [markdown]
# Positivity is the requirement that both lanes were genuinely available to every kind
# of claim. The report checks it two ways: by share (each lane holds about half the
# claims) and by overlap (a propensity fit on the covariates stays inside [0.05, 0.95]
# for every claim, so `overlap_share` is 1.00). Our routing leans on complexity without
# obeying it; the easiest claim still has about a one-in-five chance of full review, and
# those exceptions are what the comparison feeds on. Nothing flagged, so the estimate is
# licensed. From here it's [guide six](06-discovering-structure.py) unchanged: discover
# on the table with its own knowledge, re-face the result as a DAG, estimate.

# %%
from ergodic import discover, estimate_effect

found = discover(dt.data, knowledge=dt.knowledge, alpha=0.01)
print(found.graph.kind.value, sorted(e.glyph() for e in found.graph.edges))

est = estimate_effect(
    found.graph.as_kind("dag"),
    dt.data,
    "branch",
    "remaining_time",
    strategy="dml",
    random_state=0,
    treated="full_review",
)
lo, hi = est.interval()
print(f"naive lane gap:  {naive:+.1f} hours")
print("planted effect:  +3.0 hours")
print(f"causal estimate: {est.value:+.2f} hours   95% CI [{lo:+.2f}, {hi:+.2f}]")

# %% [markdown]
# Discovery drew the confounding triangle, complexity into both the branch and the
# outcome, and left the clock covariates unconnected, which matches the world we built.
# The tiers did the orienting, so the result calls itself `mixed` and we re-face it as a
# DAG, the same explicit act as in guide six. Double machine learning then prices the
# lane at +3.02 hours against a planted +3.0, with the naive gap nearly three times the
# truth. So widening the fast lane saves about three hours per diverted claim. What full
# review catches, fraud or underpayment, this outcome doesn't see; the three hours is
# one side of the ledger, now priced correctly.
#
# ## When the routing leaves no overlap
#
# One rerun shows why positivity rides inside the table rather than in a footnote.
# Suppose the office adopts a strict protocol: complexity above a threshold goes to
# review, below it goes fast, with hesitation only near the line. Steepen the routing
# curve and rebuild everything, miner included:

# %%
log_sharp, _, _, _ = claims_world(900, slope=20.0, seed=11)
sharp = decision_table(
    log_sharp,
    discover_model(log_sharp, noise=0.1),
    covariates=["complexity", "elapsed_at_landmark", "workload_at_landmark"],
)
ps = sharp.positivity
print(f"counts:        {ps.counts}   min_share: {ps.min_share:.2f}")
print(f"overlap_share: {ps.overlap_share:.2f}   flagged: {ps.flagged}")

# %% [markdown]
# The branch shares still look healthy, near half and half. The overlap is what
# collapsed: `overlap_share` 0.25 says that for three quarters of the claims the
# covariates push the propensity outside [0.05, 0.95], because under the protocol an
# easy claim all but never sees review and a hard one all but never sees the fast lane.
# The two lanes now hold different kinds of claims with a thin sliver of common ground,
# so a branch comparison is extrapolation wearing a confidence interval, and the table
# says so up front with `flagged=True`, before any estimator produces a number that
# looks as solid as the healthy one. When the flag is up, the fix is data you don't
# have yet, a pilot or a spell of off-threshold routing, and no adjustment arithmetic
# substitutes for it.
#
# ## Three reads of one model
#
# The miner gives structure: operators where a map has arrows, a tree that commits to
# claims sharp enough to check. Conformance gives trust: the model can run (`is_sound`),
# 1468 of 1500 claims run through it, and the 32 that don't are named move by move.
# The decision table gives effects: the split's branch becomes a treatment, the landmark
# gates the covariates to what the office knew at decision time, the positivity report
# says whether the comparison is licensed at all, and the discover-then-estimate
# pipeline prices the lane at its planted three hours where the raw gap said nearly
# nine.
#
# The cautions of guide nine carry over and gain one. Adjustment closes the back doors
# the log records; a router acting on something unlogged, a gut read of the claimant,
# leaves one open, which is [guide five](05-hidden-confounding.py)'s problem with the
# same escape routes. Queues still entangle cases, so one claim's lane can touch its
# neighbor's clock. New to the model layer: a split with a silent branch, a lane that
# leaves no event, is reported by `decision_points` but refused by `decision_table`,
# because the decision moment of an eventless branch is unobservable without
# alignment-based replay, and the library raises rather than guesses.
#
# That's the series, for now: from one confounded loyalty program in guide one to a
# process model whose decision points carry price tags. The
# [process guide](../process.md) documents the model layer's full surface, mining cuts
# and replay conventions included, and the [API reference](../reference.md) covers
# every object these nine guides used.
