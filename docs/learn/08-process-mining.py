# %% [markdown]
# # Process mining: your operation in three steps
#
# Operational systems don't keep tables, they keep events: order O0214 was placed at 9:12,
# cleared payment at 9:13, was picked at four in the afternoon, shipped the next morning.
# Case, activity, timestamp. Whatever the operation does all day is in those three
# columns, and reading them is a discipline with its own product category, the workflow
# tools that Celonis and Disco sell. Their daily routine fits in three steps: see the
# process, measure it, drill into the slice that hurts. This guide walks the three steps
# on one simulated operation, each step a few lines of code, with the process graphs
# drawn for real.
#
# The operation is a web shop's fulfillment flow over a 30-day month, and as everywhere
# in this series the answer key is planted in the code. An order is placed and an
# automated gateway checks the payment, declining 2%. Expensive orders, about one in
# seven, detour through a manual fraud review, and one review in ten ends in a
# cancellation. Everything else flows to the warehouse: pick, pack, ship, except that
# packing catches a wrong or damaged item on 10% of passes and sends the order back to
# pick. Two queues are planted for the later steps to find: the pick queue holds about
# eight hours of waiting in front of roughly 25 minutes of picking, and packed parcels
# wait three and a half more at the dock for a carrier cutoff. Every step writes a start
# event when somebody takes the work up and a complete event when they finish, into a
# `lifecycle` column. Timestamps are plain numbers counting hours, which keeps calendar
# noise out of the arithmetic; datetime stamps work the same everywhere below.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1800

value = rng.lognormal(4.0, 0.8, size=n)  # order value in dollars
placed_at = rng.uniform(0.0, 720.0, size=n)  # orders arrive all month, stamps in hours
p_review = 1.0 / (1.0 + np.exp(-2.0 * (np.log(value) - 5.2)))  # pricey orders get flagged
takes_review = rng.uniform(size=n) < p_review  # the branch: ~15% of orders
declined = rng.uniform(size=n) < 0.02  # the gateway declines 2%
is_fraud = rng.uniform(size=n) < 0.10  # one review in ten ends in a cancellation

rows = []


def visit(order, act, t, wait, service):
    """One start/complete pair for a step; returns the completion time."""
    rows.append((order, act, t + wait, "start"))
    rows.append((order, act, t + wait + service, "complete"))
    return t + wait + service


for i in range(n):
    order = f"O{i:04d}"
    t = visit(order, "placed", placed_at[i], 0.0, 0.0)  # both stamps at once, service zero
    t = visit(order, "payment", t, rng.exponential(0.05), rng.uniform(0.01, 0.03))
    if declined[i]:
        visit(order, "cancel", t, rng.uniform(0.02, 0.2), 0.0)
        continue
    if takes_review[i]:
        t = visit(order, "review", t, rng.exponential(3.0), rng.uniform(0.3, 0.8))
        if is_fraud[i]:
            visit(order, "cancel", t, rng.uniform(0.05, 0.3), 0.0)
            continue
    while True:
        # the answer key: ~8 h of queue in front of ~25 min of picking
        t = visit(order, "pick", t, rng.exponential(8.0), rng.uniform(0.25, 0.6))
        t = visit(order, "pack", t, rng.exponential(1.2), rng.uniform(0.2, 0.5))
        if rng.uniform() >= 0.10:  # the answer key: pack rejects 10% of passes
            break
    # the dock holds packed parcels for carrier cutoffs, the second queue
    visit(order, "ship", t, rng.exponential(3.5), rng.uniform(0.02, 0.08))

df = pd.DataFrame(rows, columns=["order", "activity", "t", "lifecycle"])
print(f"{len(df)} event rows from {n} orders")
print("planted: ~8 hours of pick queue in front of ~25 minutes of picking")
print("planted: ~15% of orders detour through review; pack rejects 10% of passes")

# %% [markdown]
# ## Step 1: see the process
#
# An event log is the typed container for those three columns. `event_log` names them,
# groups the rows by case, and sorts each case in time; everything else in the dataframe
# rides along. We build two views of the same month. The working log keeps one row per
# step, the completions, which is what a map should count. The staged log keeps the start
# rows too; step two will need them.

# %%
from ergodic import event_log

done = df[df["lifecycle"] == "complete"].drop(columns="lifecycle")
log = event_log(done, case="order", activity="activity", timestamp="t")
staged = event_log(df, case="order", activity="activity", timestamp="t")
print(log)
print(staged)

# %% [markdown]
# `process_map` is the first thing any process tool computes: who hands off to whom, how
# often, with the mean and median gap on every arc. Synthetic `__start__` and `__end__`
# arcs record where cases enter and leave, so the boundaries are data too.

# %%
from ergodic import process_map

pm = process_map(log)
print(pm)
print(f"orders through the review detour: {pm.frequency('payment', 'review')}")
n_loop, n_pack = pm.frequency("pack", "pick"), pm.frequency("pick", "pack")
print(f"packs that sent an order back to pick: {n_loop} of {n_pack}")

# %% [markdown]
# Twelve arcs over seven activities, and two frequencies already sketch the structure:
# 271 orders took the review detour, and pack bounced an order back to pick 178 times,
# 9.3% of passes against the planted 10%. But nobody runs an operation off a table of
# arcs. The picture is what the commercial tools sell, and with a fixed activity set it
# takes one matplotlib helper: positions set by hand, arrow width following traffic,
# every arc labeled with its count and median gap in hours, and a bend or a label nudge
# wherever the geometry would collide. You can skip the next cell; the payoff is the
# picture it produces.

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ergodic.process import END, START

POSITIONS = {
    START: (0.0, 0.0),
    "placed": (1.0, 0.0),
    "payment": (2.0, 0.0),
    "review": (3.0, 0.9),
    "cancel": (3.0, -1.0),
    "pick": (4.0, 0.0),
    "pack": (5.0, 0.0),
    "ship": (6.0, 0.0),
    END: (7.0, 0.0),
}
BENDS = {  # arc3 curvature where straight arrows would overlap
    ("pick", "pack"): -0.18,
    ("pack", "pick"): -0.45,
    ("review", "cancel"): -0.22,
    ("cancel", END): 0.12,
}
NUDGES = {("review", "cancel"): (0.16, -0.42)}  # slide one label off the trunk


def draw_map(pm, positions):
    """Draw a ProcessMap at hand-set node positions; returns None."""
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    heaviest = max(stats.count for stats in pm.arcs.values())
    for (a, b), stats in pm.arcs.items():
        (xa, ya), (xb, yb) = positions[a], positions[b]
        boundary = a == START or b == END
        rad = BENDS.get((a, b), 0.0)
        scale = 0.6 if boundary else 1.0
        ax.add_patch(
            FancyArrowPatch(
                (xa, ya),
                (xb, yb),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=11,
                lw=scale * (0.6 + 5.0 * stats.count / heaviest),
                color="#9aa5b1" if boundary else "#33415c",
                shrinkA=17,
                shrinkB=17,
                zorder=1,
            )
        )
        # the label sits at the arc's midpoint, nudged off the line
        dx, dy, norm = xb - xa, yb - ya, ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        side = 1.0 if rad >= 0 else -1.0
        nx, ny = NUDGES.get((a, b), (0.0, 0.0))
        lx = (xa + xb) / 2 + 0.5 * rad * dy + side * 0.15 * dy / norm + nx
        ly = (ya + yb) / 2 - 0.5 * rad * dx - side * 0.15 * dx / norm + ny
        label = f"{stats.count}" if boundary else f"{stats.count}\n{stats.median_gap:.1f}h"
        ax.text(
            lx,
            ly,
            label,
            fontsize=7,
            ha="center",
            va="center",
            color="#52606d",
            linespacing=1.1,
            zorder=2,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        )
    for name in pm.activities:
        x, y = positions[name]
        ax.text(
            x,
            y,
            name,
            fontsize=9.5,
            ha="center",
            va="center",
            color="#1f2933",
            zorder=3,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#33415c", "lw": 1.2},
        )
    for name, tag in ((START, "start"), (END, "end")):
        x, y = positions[name]
        ax.plot([x], [y], "o", ms=7, color="#9aa5b1", zorder=3)
        ax.text(x, y - 0.16, tag, fontsize=8, ha="center", va="top", color="#9aa5b1")
    title = f"{pm.n_cases} orders, {len(pm.arcs)} arcs"
    if pm.dropped_arcs:
        title += f"; dropped {pm.dropped_arcs} arcs ({pm.dropped_share:.1%} of traversals)"
    ax.set_title(title, fontsize=9, loc="left", color="#52606d")
    ax.set_xlim(-0.4, 7.45)
    ax.set_ylim(-1.5, 1.35)
    ax.axis("off")
    fig.tight_layout()


draw_map(pm, POSITIONS)

# %% [markdown]
# There's the operation. The trunk runs left to right, the review detour rises above it,
# the two cancellation paths sink below, and the bent arc from pack back to pick is the
# rework loop. The heavy gaps sit where we planted them: the hop into pick carries a
# median of 6.1 hours against 1.2 into pack and 2.4 into ship. A full map is busy because
# it answers for every order, the 38 declines and the loop traffic included, and on a
# real log with hundreds of arcs this picture turns to spaghetti. The standard move is to
# thin it.

# %%
thin = pm.simplify(keep=0.9)
print(thin)
print(f"dropped {thin.dropped_arcs} arcs carrying {thin.dropped_share:.1%} of all traversals")
draw_map(thin, POSITIONS)

# %% [markdown]
# Same one-liner, new graph. `simplify(keep=0.9)` ranks the real arcs by count and keeps
# the smallest set covering 90% of the traversals, and what survives is the spine of the
# operation: placed, payment, pick, pack, ship. Clean has a price, and the price is
# written on the result instead of hidden: `dropped_arcs` says five arcs are gone,
# `dropped_share` says they carried 9.8% of the traffic, and both accumulate over
# repeated calls, so a twice-thinned map still knows its history. The figure title above
# reads those fields. The point matters because a thinned map that hid its filtering
# would lie about the process; on this picture the review desk and the rework loop never
# happened. One more rule is visible at the cancel node: every arc into it was dropped,
# but START and END arcs always survive, so the 60 cancelled orders still leave the map
# rather than evaporate.
#
# ## Step 2: measure it
#
# The map prices the hops between steps; it can't say whose time a hop is. Six hours
# between payment and pick could be a deep queue in front of fast pickers or slow picking
# with no queue at all, and the fix is different in each world. One stamp per step can't
# split those. Two can, and the staged log has two: `performance` pairs each start with
# its complete within a case and activity, reads service as complete minus start and
# waiting as start minus the previous complete in the same case, and counts any lifecycle
# row that never finds a partner in `n_unmatched` rather than dropping it in silence.

# %%
from ergodic.process import performance

perf = performance(staged, lifecycle="lifecycle")
print(
    f"cycle time over {perf.n_cases} orders: mean {perf.mean_duration:.1f} h, "
    f"median {perf.median_duration:.1f} h, p90 {perf.p90_duration:.1f} h"
)
print(f"lifecycle rows without a partner: {perf.n_unmatched}")
for s in perf.activities:
    wait = "     -" if s.mean_wait is None else f"{s.mean_wait:6.2f}"
    service = "     -" if s.mean_service is None else f"{s.mean_service:6.2f}"
    print(f"{s.activity:>8}   mean wait {wait} h   mean service {service} h")

# %% [markdown]
# The split is the finding. An order's median life is 12.7 hours, the mean is 14.9
# because queues have long tails, and one order in ten takes more than 28. The table says
# where the hours live: pick holds 8.15 hours of waiting against 0.43 hours of work.
# Pickers aren't slow; orders wait eight hours for under half an hour of picking. The
# dock is the same shape, a 3.5-hour wait for three minutes of scanning, and review and
# pack carry queues of 3.3 and 1.2 hours in front of half an hour and twenty minutes of
# work. This split is the chart every process-mining pitch leads with, because it
# redirects the fix: speeding up the pickers attacks 26 minutes of an 8.6-hour step,
# while the queue is the other 95%.

# %%
pick_flat = next(s for s in performance(log).activities if s.activity == "pick")
print(f"completions only: pick wait {pick_flat.mean_wait:.2f} h, service {pick_flat.mean_service}")


def draw_split(perf, steps):
    stats = {s.activity: s for s in perf.activities}
    fig, ax = plt.subplots(figsize=(7, 2.6))
    y = np.arange(len(steps))
    waits = np.array([stats[name].mean_wait for name in steps])
    services = np.array([stats[name].mean_service for name in steps])
    ax.barh(y, waits, height=0.55, color="#d1495b", label="waiting")
    ax.barh(y, services, left=waits, height=0.55, color="#33415c", label="service")
    ax.set_yticks(y, steps)
    ax.invert_yaxis()
    ax.set_xlabel("hours per visit (mean)")
    ax.legend(frameon=False)
    fig.tight_layout()


draw_split(perf, ["pick", "ship", "review", "pack", "payment"])

# %% [markdown]
# Red is waiting, navy is work, and the work is a sliver everywhere it matters. The first
# print is the honesty check that travels with this table: asked on the completions log,
# the same step reads 8.58 hours of wait and a service of None, because one stamp per
# activity can't split waiting from working, and `performance` says so rather than guess.
#
# The last descriptive table ranks the paths themselves. A variant is one distinct
# activity sequence, and `variant_table` orders them by how many orders follow each:

# %%
from ergodic.process import variant_table

vt = variant_table(log)
print(f"{len(vt)} distinct paths")
for v in vt[:5]:
    path = " -> ".join(v.sequence)
    print(f"{path:<66} {v.n_cases:4d} orders ({v.share:5.1%})  mean {v.mean_duration:5.1f} h")

# %% [markdown]
# Three orders in four take the five-step happy path. The reviewed path is the largest
# minority at 12.2%, running 4.1 hours over the happy path's mean. The single-rework path
# runs 8.4 hours over, about one extra trip through the 8.6-hour pick step. Below that
# the tail thins out fast, which is typical; what's atypical about this log is that seven
# steps produce only eight variants, where a real month produces hundreds, most followed
# by one case each.
#
# ## Step 3: drill in
#
# A mean over 1800 orders blurs the slice where the money is, so the third step cuts the
# log and reruns the first two steps on the piece. Six filters ship with the log: by time
# window, by variant, by endpoints, by which activities a case contains, by an
# a-then-b follows pattern, and by case duration or length. Each filter's docstring leads
# with its contract, "Drops cases:" or "Drops events:", so you always know whether whole
# orders or single rows are leaving. Activity names are validated against the log, and a
# filter that would leave nothing raises with "loosen it" instead of returning an empty
# log that breaks something three calls later. Filters compose by plain function
# application, and what they return is a full event log, so the map, the split, and the
# variants rerun on the slice unchanged.
#
# Say the finance team wants the fraud review priced. The honest slice is orders that
# went through review and still shipped; the cancelled ones leave the flow, and pricing
# them is a different question.

# %%
from ergodic.process import filter_activities, filter_endpoints

reviewed = filter_activities(log, mandatory={"review"})
cleared = filter_endpoints(reviewed, end_in={"ship"})
print(f"orders through review: {reviewed.n_cases}   reviewed and shipped: {cleared.n_cases}")

pm_review = process_map(cleared)
into_review = pm_review.arcs[("payment", "review")]
hop_slice = pm_review.arcs[("review", "pick")].median_gap
hop_full = pm.arcs[("payment", "pick")].median_gap
print(
    f"payment -> review:               median {into_review.median_gap:.1f} h, "
    f"mean {into_review.mean_gap:.1f} h"
)
print(f"review -> pick on the slice:     median {hop_slice:.1f} h")
print(f"payment -> pick on the full map: median {hop_full:.1f} h")
draw_map(pm_review, POSITIONS)

# %% [markdown]
# Same helper, third graph, and now the detour is the trunk: 249 orders, every one of
# them through review. The finding is in the medians. A reviewed order reaches review a
# median of 2.7 hours after payment, a mean of 3.8, the review queue's own tail at work.
# From review it reaches pick in a median of 6.0 hours, while the full map's hop
# from payment into pick read 6.1: the pick queue neither knows nor cares that fraud
# cleared the order first. Reviewed orders join the same line as everyone else, they just
# join it late. The 3.8-hour toll matches the variant table, where the reviewed path ran
# 4.1 hours over the happy one, within the month's noise, and it locates the cost: the
# screen taxes an order's entry into the warehouse and changes nothing behind it. If
# those hours hurt, the lever is the review queue's staffing, none of the machinery
# downstream.
#
# ## Where the map stops
#
# Three steps and a screenful of code recover the daily deliverables of a process-mining
# deployment: the drawn map, the thinned map with its losses declared, the
# waiting-versus-service split, the variant ranking, the drilled slice. All of it
# describes. The moment a finding suggests an action, the question changes kind: cutting
# the review buys back its 3.8-hour toll only if reviewed orders are otherwise like the
# rest, and the map can't certify that. An arc records precedence, not cause, and the
# detour here is routed by order value, so the reviewed slice is a selected slice. In
# this simulation the selection happens to be harmless; in
# [guide nine](09-process-intelligence.py) it isn't, and the same kind of naive gap
# bills a review step at double its true cost until the causal machinery corrects it.
# That guide carries logs like this one into case tables and KPI panels;
# [guide ten](10-decision-points.py) mines the process model itself and prices a routing
# decision. The full descriptive surface, including the four filters this guide didn't
# touch, is in the [process mapping guide](../../docsite/pages/map-a-process.md).
