# %% [markdown]
# # Process intelligence: when the process is the question
#
# Every guide so far started from a table somebody had already prepared. Operational data
# doesn't arrive that way. What a support desk, a claims office, or an order pipeline logs
# is events: ticket T0412 was opened at 9:14, triaged at 9:44, sent to review at 10:02,
# closed the next afternoon. Case, activity, timestamp. Whatever the operation does all
# day is in those three columns, including a fair amount it doesn't know about itself.
#
# Process mining grew up around this data. Its tools draw the workflow as a map, rank the
# paths cases follow, and point at the slow ones; [guide eight](08-process-mining.py)
# walked that workflow end to end. It's honest work, and it's where
# every process tool stops, because the next question isn't descriptive. The map can say
# that reviewed tickets run ten hours slower. It can't say whether cutting the review
# would save ten hours or two, and the difference is an intervention question, the same
# gap this series opened on in [guide one](01-why-causality.py). So this guide walks both
# halves: the descriptive pass for what the process is, then the two bridges, `case_table`
# and `kpi_panel`, that carry a log into the causal machinery for what to do about it.
#
# The operation is a support desk where management suspects the second-review step, a
# quality gate some tickets pass through, of being the bottleneck. We simulate its month
# so the answer key sits in plain sight, the move every guide makes. Each ticket carries a
# complexity score recorded at intake. Complexity drives the review decision, because hard
# tickets get flagged to a second pair of eyes, and complexity drives cycle time on its
# own. The review itself adds five hours, no more. Timestamps are plain numbers counting
# hours, which keeps calendar noise out of the arithmetic; datetime stamps work the same
# everywhere below.

# %%
import numpy as np
import pandas as pd

from ergodic import event_log

rng = np.random.default_rng(3)
n = 1400

complexity = rng.normal(0.0, 1.0, size=n)
p_review = 1.0 / (1.0 + np.exp(-1.3 * complexity))  # hard tickets get flagged
review = rng.uniform(size=n) < p_review
cycle = (  # hours from open to close
    30.0
    + 6.0 * complexity  # complexity slows everything it touches
    + 5.0 * review  # the answer key: the review adds five hours
    + rng.normal(0.0, 2.0, size=n)
)
start = rng.uniform(0.0, 720.0, size=n)  # arrivals spread over a 30-day month

rows = []
for i in range(n):
    ticket = f"T{i:04d}"
    rows.append((ticket, "open", start[i], complexity[i]))
    rows.append((ticket, "triage", start[i] + 0.5, complexity[i]))
    if review[i]:
        rows.append((ticket, "review", start[i] + 1.0, complexity[i]))
    rows.append((ticket, "close", start[i] + cycle[i], complexity[i]))
df = pd.DataFrame(rows, columns=["ticket", "step", "t", "complexity"])

log = event_log(df, case="ticket", activity="step", timestamp="t")
print(log)
print(f"tickets sent to review: {int(review.sum())} of {n}")
print("planted effect of the review step: +5.0 hours")

# %% [markdown]
# ## The pass every tool does
#
# A directly-follows map is the first thing any process tool draws: who hands off to
# whom, how often, and how long the hop takes. `__start__` and `__end__` are synthetic
# bookends recording where cases enter and leave.

# %%
from ergodic import process_map

pm = process_map(log)
print(pm)
for (a, b), arc in pm.arcs.items():
    gap = "" if np.isnan(arc.mean_gap) else f"   mean gap {arc.mean_gap:6.1f} h"
    print(f"{a:>9} -> {b:<9} {arc.count:5d}{gap}")

# %% [markdown]
# Read the two arcs into `close`. The hop from `review` averages 36.7 hours; the hop
# straight from `triage`, 27.0. The map has named its suspect. The variant table sharpens
# the picture: a variant is one distinct path through the process, and this desk has two.

# %%
from ergodic.process import variant_table

vt = variant_table(log)
for v in vt:
    path = " -> ".join(v.sequence)
    print(f"{path:34} {v.n_cases:4d} cases ({v.share:.0%})   mean {v.mean_duration:.1f} h")

# %%
import matplotlib.pyplot as plt


def draw_variants(variants):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    y = np.arange(len(variants))
    colors = ["#d1495b" if "review" in v.sequence else "#33415c" for v in variants]
    ax.barh(y, [v.mean_duration for v in variants], height=0.5, color=colors)
    for i, v in enumerate(variants):
        ax.text(v.mean_duration + 0.5, i, f"{v.n_cases} tickets", va="center", fontsize=9)
    ax.set_yticks(y, [" -> ".join(v.sequence) for v in variants])
    ax.invert_yaxis()
    ax.set_xlim(0, 46)
    ax.set_xlabel("mean cycle time (hours)")
    fig.tight_layout()


draw_variants(vt)

# %% [markdown]
# ## The drilldown trap
#
# So far, honest description. The trouble starts when someone reads the gap between those
# two bars as the price of the review, which is exactly how this chart gets used:

# %%
reviewed = next(v for v in vt if "review" in v.sequence)
plain = next(v for v in vt if "review" not in v.sequence)
naive = reviewed.mean_duration - plain.mean_duration
print(f"reviewed minus not, mean cycle time: {naive:+.1f} hours")
print("planted effect of the review:        +5.0 hours")

# %% [markdown]
# The gap says 10.2 hours where we planted five: double the truth. The other half of that
# number never belonged to the review. Hard tickets get flagged into it, and hard tickets
# take longer wherever they go, so the review path collects the desk's hardest work and is
# then billed for the difficulty it attracted. "Are reviewed tickets slower" is what the
# table answers, and the answer is yes. The question management is asking is "would
# tickets close faster without the review", and that's a do-question, aimed at a process
# step instead of a loyalty program.
#
# ## The first bridge: one row per case
#
# Answering it needs the shape the rest of this series runs on: one row per unit with a
# treatment, an outcome, and covariates. `case_table` flattens the log into exactly that,
# and the flattening is where the causal care lives.

# %%
from ergodic import case_table

ct = case_table(log, "review", "cycle_time")
print(ct)
print(f"covariates: {ct.covariates}")

# %% [markdown]
# One row per ticket: did `review` happen, how long the case ran (`cycle_time` is derived
# from the stamps), and the covariates the log can vouch for. The rule behind that last
# group is timing. A covariate must be measurable no later than the moment the treatment
# question is asked, so nothing that happens after the review can leak into the adjustment
# set, the post-treatment trap of [guide two](02-good-and-bad-controls.py). Here that
# leaves `complexity`, the intake attribute, and one derived feature: `workload_at_start`,
# how many other tickets were open when this one arrived, the standard process confounder
# no spreadsheet carries. The table also ships a `DomainKnowledge` with three tiers,
# covariates before treatment before outcome, so backwards arrows are forbidden by
# machinery rather than by memory:

# %%
print(ct.knowledge.permits_edge("complexity", "review"))
print(ct.knowledge.permits_edge("cycle_time", "review"))

# %% [markdown]
# From here the playbook is guides one and six, unchanged. Draw the graph by hand and
# estimate; or hand the table's own knowledge to discovery and estimate off what comes
# back.

# %%
from ergodic import dag, estimate_effect

g = dag(
    [
        "complexity -> review",
        "complexity -> cycle_time",
        "review -> cycle_time",
        "workload_at_start -> cycle_time",
    ]
)
hand = estimate_effect(g, ct.data, "review", "cycle_time", random_state=0, n_boot=200)
lo, hi = hand.interval()
print(f"hand-drawn graph: {hand.value:+.2f} h   95% CI [{lo:+.2f}, {hi:+.2f}]")

# %%
from ergodic import discover

found = discover(ct.data, knowledge=ct.knowledge, alpha=0.01)
print(found.graph.kind.value, sorted(e.glyph() for e in found.graph.edges))
disc = estimate_effect(
    found.graph.as_kind("dag"), ct.data, "review", "cycle_time", random_state=0, n_boot=200
)
lo_d, hi_d = disc.interval()
print(f"discovered graph: {disc.value:+.2f} h   95% CI [{lo_d:+.2f}, {hi_d:+.2f}]")

# %% [markdown]
# Both routes print +4.87 with an interval covering the planted five, a long way from the
# variant table's 10.2. They agree to the printed digit because both graphs close the same
# back door, adjusting for complexity; the hand graph adjusts for workload too, which is
# unrelated to the review decision here and moves nothing visible. Discovery found the
# confounding triangle on its own and called its graph `mixed`, because the tiers,
# knowledge rather than data, oriented the edges; re-facing it as a DAG is the same
# explicit act it was in [guide six](06-discovering-structure.py). So the review costs
# five hours, half of what the bar chart implied, and that number can now be weighed
# against whatever defects the review catches.
#
# ## The second bridge: a rollout becomes a panel
#
# Half a year later the desk ships `new_routing`, a policy that assigns tickets to
# specialists instead of round-robin. It rolls out team by team, and the order is anything
# except random: the teams with the worst cycle times lobbied hardest and went first. Ten
# teams switch at week 10, ten more at week 18, ten never do. This is the staggered
# rollout of [guide seven](07-quasi-experiments.py), except nobody hands us a panel; there
# is only the log, now spanning 30 teams and 26 weeks, with stamps counting days.

# %%
rng2 = np.random.default_rng(8)
n_teams, n_weeks = 30, 26
adopt = np.array([10] * 10 + [18] * 10 + [n_weeks] * 10)  # n_weeks = never switches
base = 30.0 + 4.0 * (adopt < n_weeks) + rng2.normal(0.0, 1.0, size=n_teams)
shock = rng2.normal(0.0, 0.6, size=n_weeks)  # one calendar, shared by every team

rows2 = []
for t in range(n_teams):
    for w in range(n_weeks):
        routed = float(w >= adopt[t])
        m = int(rng2.integers(8, 13))
        starts = 7.0 * w + 6.9 * rng2.random(m)  # day stamps inside the week
        cycles = (  # hours, as before
            base[t]
            - 0.08 * w  # the desk was improving slowly anyway
            + shock[w]
            - 3.0 * routed  # the answer key: routing saves three hours
            + rng2.normal(0.0, 2.0, size=m)
        )
        for j in range(m):
            case_id = f"team{t:02d}-w{w:02d}-{j:02d}"
            rows2.append((case_id, "received", starts[j], f"team{t:02d}", routed))
            rows2.append(
                (case_id, "resolved", starts[j] + cycles[j] / 24.0, f"team{t:02d}", routed)
            )
df2 = pd.DataFrame(rows2, columns=["ticket", "step", "t", "team", "new_routing"])

log2 = event_log(df2, case="ticket", activity="step", timestamp="t")
print(log2)
print("planted: adopting teams run 4 hours slower at baseline, before any routing")
print("planted: new_routing takes 3.0 hours off every ticket from the adoption week on")

# %% [markdown]
# `kpi_panel` aggregates the log into the shape guide seven's designs eat: bucket each
# case by the week it starts (width 7.0 in the stamps' day units), split by team, average
# each cell. `cycle_time` comes out in days because the stamps count days, so we'll
# multiply by 24 whenever we want hours. The `new_routing` flag averages within each cell
# too, and since adoption lands on week boundaries the average is exactly 0 or 1: a
# ready-made treatment column.

# %%
from ergodic import kpi_panel

pnl = kpi_panel(log2, 7.0, unit="team", kpis=("cycle_time", "new_routing"))
frame = pnl.to_pandas()
print(f"panel: {pnl.n_rows} team-week rows, balanced: {pnl.is_balanced}")
print(f"new_routing values in the panel: {sorted(float(x) for x in frame['new_routing'].unique())}")

post = frame[frame["period"] >= 7.0 * 18]
naive2 = 24.0 * (
    post.loc[post["new_routing"] == 1.0, "cycle_time"].mean()
    - post.loc[post["new_routing"] == 0.0, "cycle_time"].mean()
)
print(f"naive, routed minus unrouted team-weeks after week 18: {naive2:+.2f} hours")

# %% [markdown]
# Routed team-weeks run about an hour slower than unrouted ones over the same late weeks,
# so at face value the policy failed. But the routed group is exactly the twenty teams
# that were slowest before the policy existed, and their four-hour handicap swallows the
# three-hour gain. Wrong sign, for the same reason as guide one: the treatment went where
# the problem was. The cure is also guide seven's, unchanged. `did` differences each
# adopting team against its own pre-adoption level, then against the never-adopters over
# the same weeks, removing the handicap and the calendar in one move; two adoption waves
# make it pick its group-time flavor automatically.

# %%
from ergodic import did

est = did(pnl, "cycle_time", "new_routing", n_boot=200, random_state=0)
lo2, hi2 = est.interval()
print(f"flavor: {est.flavor}, against {est.comparison} teams")
print(
    f"did estimate: {24.0 * est.value:+.2f} hours   95% CI [{24.0 * lo2:+.2f}, {24.0 * hi2:+.2f}]"
)
print("planted:      -3.00 hours")

# %% [markdown]
# Minus 3.02 against a planted minus 3.00, from data whose naive read had the wrong sign.
# The event study replays the design at every distance from adoption, leads as the placebo
# check, lags as the effect's path:

# %%
from ergodic import event_study

es = event_study(pnl, "cycle_time", "new_routing", leads=6, lags=7, n_boot=200, random_state=0)
worst_lead = 24.0 * max(abs(e.estimate) for e in es.by_event_time if e.event_time < 0)
print(f"largest lead estimate, absolute: {worst_lead:.2f} hours")


def draw_event_study(effects):
    fig, ax = plt.subplots(figsize=(7, 4))
    times = np.array([e.event_time for e in effects])
    mid = 24.0 * np.array([e.estimate for e in effects])
    low = 24.0 * np.array([e.interval[0] for e in effects])
    high = 24.0 * np.array([e.interval[1] for e in effects])
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
    after = times[times >= 0]
    ax.plot(after, np.full(after.size, -3.0), ls="--", lw=1.2, color="#d1495b", label="planted")
    ax.axhline(0.0, color="#9aa5b1", lw=1)
    ax.axvline(-0.5, color="#9aa5b1", lw=1, ls=":")
    ax.set_xlabel("event time (weeks since the team switched routing)")
    ax.set_ylabel("effect on mean cycle time (hours)")
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()


draw_event_study(es.by_event_time)

# %% [markdown]
# Left of the dotted line, the leads sit on zero, the largest of them 0.13 hours, every
# band straddling zero: in the weeks before each wave switched, adopters and
# never-adopters moved together, which is the parallel-trends assumption checked where it
# can be checked. Right of the line, the effect drops to minus three and stays flat,
# because we planted a constant effect; a growing one would climb here, the way guide
# seven's did. The lags stop at seven weeks because that's as far as both waves are
# observed.
#
# ## What the bridges rest on
#
# The case table closed the back door because complexity was in the log. If reviewers
# flag tickets on something the log never records, a gut read of the customer's mood,
# say, then a back door stands open and no adjustment on logged columns closes it; that
# is [guide five](05-hidden-confounding.py)'s problem, and its escape routes are the
# same here. The flattening also trusts every case-constant attribute to be known at
# case start. The timestamp gate protects against event leakage, but a column carries no
# event timing, so an attribute backfilled at closing time would walk straight through
# the gate wearing an intake badge.
#
# Interference is the assumption to lose sleep over in a process. Reviews consume
# reviewers: send one ticket to review and the next ticket waits a little longer, so one
# case's treatment can touch its neighbor's outcome through the queue. Nothing in this
# guide models that; our simulated reviews were queue-free, which is the kind of thing a
# simulation can decree and an operation can't. The workload features measure congestion,
# they don't undo it. And the panel design carries guide seven's burden unchanged:
# parallel trends is a claim about a world nobody observed, and flat leads support it
# without proving it.
#
# There's more process surface than one guide can walk: filters that slice a log while saying what
# they dropped, performance views, the landmark modes of `case_table` that ask the treatment
# question at a fixed point in each case's life, and, further out, conformance against the process
# you think you run. The [process mapping guide](../../docsite/pages/map-a-process.md) and the
# [conformance guide](../../docsite/pages/conformance-and-decisions.md) cover all of it, and the
# [difference-in-differences guide](../../docsite/pages/difference-in-differences.md) covers the
# designs the second bridge feeds.
