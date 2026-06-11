# Process intelligence

An event log is the native record of an operational process: one row per event, with a case id (the ticket, the order), an activity name, and a timestamp. Whatever else the system wrote rides along as attributes. An attribute that is constant within a case (the team, the product line) describes the case; one that varies (the resource on each step) describes the event. `event_log(df, case=..., activity=..., timestamp=...)` builds the typed container, with rows grouped by case and sorted in time; the [data guide](data.md) covers its accessors.

`ergodic.process` reads that log on three layers. The descriptive layer recovers what process-mining tools do daily: the map, the timing, the variants, the filters. The model layer mines an executable process model from the log and checks the log against it. The causal layer asks the question those tools can't: what would happen to the cycle time if we changed the process.

## The log

Every snippet on this page runs on one simulated ticket desk, with the truth planted where we can check it. Tickets open, get triaged a second later, and close; harder tickets are more likely to take a detour through a review step. Review costs exactly 5.0 seconds of cycle time, and complexity drives both the review decision and the cycle time on its own.

```python
import numpy as np
import pandas as pd
from ergodic import event_log

rng = np.random.default_rng(11)
n = 1500
complexity = rng.normal(size=n)
review = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.3 * complexity))
cycle = 20.0 + 6.0 * complexity + 5.0 * review.astype(float) + rng.normal(0.0, 2.0, size=n)
start = rng.uniform(0.0, 5000.0, size=n)

rows = []
for i in range(n):
    case = f"c{i:04d}"
    rows.append((case, "open", start[i], complexity[i]))
    rows.append((case, "triage", start[i] + 1.0, complexity[i]))
    if review[i]:
        rows.append((case, "review", start[i] + 2.0, complexity[i]))
    rows.append((case, "close", start[i] + cycle[i], complexity[i]))

df = pd.DataFrame(rows, columns=["case", "activity", "time", "complexity"])
log = event_log(df, case="case", activity="activity", timestamp="time")
log   # EventLog(cases=1500, events=5264, activities=4)
```

The timestamps here are plain seconds. Datetime stamps work everywhere below, and a few features exist only for them.

## The map

`process_map` walks each case and counts how often one activity directly follows another, with the mean and median gap in seconds on every arc. Synthetic `START` and `END` arcs record where cases enter and leave, so the boundaries are data too.

```python
from ergodic import process_map

m = process_map(log)
m                                      # ProcessMap(activities=4, arcs=9, cases=1500)
m.frequency("triage", "review")        # 764: the reviewed share of 1500 cases
m.arcs[("review", "close")].mean_gap   # ~25.9 seconds from review to close
```

Nine arcs from a four-step process is the first finding. Three broken cases closed within a second of opening, before their own triage stamp, and the map refuses to tidy that away: `open -> close` and `close -> triage` sit there with count 3 until you decide what they mean.

Real maps have hundreds of arcs, and the standard move is to thin them:

```python
thin = m.simplify(keep=0.8)
thin   # ProcessMap(activities=4, arcs=6, cases=1500, dropped 3 arcs (19.6% of traversals))
```

`simplify(keep=0.8)` keeps the busiest arcs that cover 80% of the real traversals; `START` and `END` arcs always stay, and so does every activity. What it removed is written on the result: `dropped_arcs` and `dropped_share`, accumulating over repeated calls. The reason those fields exist is on display here. The thinned map kept `triage -> review` and dropped `triage -> close`, so it now shows every ticket routed through review, and the skip-review path that 733 tickets took survives only in the provenance. A filtered map that hides its filtering lies about paths and durations; this one can't.

A process map is deliberately not a `MixedGraph`, so nothing here flows into `identify_effect`. An arc is precedence, not cause. And rework is normal: `A -> B` and `B -> A` both happen in real logs, each direction with its own count and timing, while a causal graph holds one edge per node pair on purpose. So the map stays its own small value object, with `to_dict` for whatever renders it.

## Performance and variants

`performance` summarizes case cycle times (first event to last) and gives a per-activity timing table.

```python
from ergodic.process import performance, variant_table

perf = performance(log)
perf.mean_duration, perf.p90_duration   # (~22.6, ~32.3) seconds
perf.activities[2]   # ActivityStats(activity='review', count=764, mean_wait=1.0, mean_service=None)
```

`mean_service` is None for a reason: this log has one timestamp per activity, and one timestamp cannot split waiting from working. The 21 seconds that pass before a typical `close` are queue time plus handling time in unknown shares, and the summary won't guess. When the log carries a lifecycle column with `"start"` and `"complete"` rows, pass `performance(log, lifecycle="status")`: service is complete minus start, wait is start minus the previous complete in the same case, and lifecycle rows that never find a partner are counted in `n_unmatched` rather than hidden.

`variant_table` ranks the distinct activity sequences by how many cases follow them:

```python
for v in variant_table(log, top=2):
    print(v.n_cases, f"{v.share:.0%}", v.sequence)
# 764 51% ('open', 'triage', 'review', 'close')
# 733 49% ('open', 'triage', 'close')
```

The reviewed variant runs about 10.7 seconds slower on average (`mean_duration` ~27.9 against ~17.1). Hold that gap; the causal section is about how much of it review causes. The third variant, three cases of `('open', 'close', 'triage')`, is the broken stamps again: a variant table is where you notice such things. The ranking is shared with `filter_variants`, so "the top two variants" means the same thing everywhere.

## Filters

Six filters compose by plain function application, each returning a new immutable log. The distinction worth tracking is whether a filter drops whole cases or single events.

| Filter | Drops | Keeps |
| --- | --- | --- |
| `filter_time(log, start=, end=, mode=)` | cases; `mode="trim"` drops events | cases contained in or intersecting the window, or only the events inside it |
| `filter_variants(log, top=)` | cases | cases whose full sequence is a frequent variant (`min_cases=` is the other selector) |
| `filter_endpoints(log, start_in=, end_in=)` | cases | cases that start and end in the given sets, the standard way to drop incomplete cases |
| `filter_activities(log, mandatory=, forbidden=, keep=)` | cases; `keep=` then drops events | cases judged on their full trace, then only the kept activities' events |
| `filter_follows(log, a, b, directly=, within=)` | cases | cases where `b` follows `a`, directly or eventually, optionally within a gap in seconds |
| `filter_performance(log, min_duration=, max_duration=, min_events=, max_events=)` | cases | cases inside duration and event-count bounds |

```python
from ergodic.process import filter_activities, filter_performance

reviewed = filter_activities(log, mandatory=["review"])   # 764 cases survive
slow = filter_performance(reviewed, min_duration=30.0)    # 262 of them ran half a minute or more
```

An event-level drop can empty a case, and the case vanishes with its events. A filter that would leave nothing raises with "loosen it" instead of returning an empty log, so a window that matches no case fails at the filter, not three steps later. A misspelled activity name fails earlier still: the activity filters validate names against the log.

## From description to cause

Everything above describes. Here is the question it can't answer: reviewed tickets run about 10.8 seconds slower, so should we cut the review step? The drilldown every process-mining tool offers, ranking attributes by their correlation with slow cases, says nothing about that intervention. In this log complexity drives both the review decision and the cycle time, so review gets the hard tickets and looks slower than it is.

`case_table` is the bridge from the log to the causal stack. It flattens the log to one row per case, and it is strict about time:

```python
from ergodic import case_table

ct = case_table(log, "review", "cycle_time")
ct               # CaseTable(cases=1500, treatment='review', outcome='cycle_time', covariates=2, when='anytime')
ct.covariates    # ('complexity', 'workload_at_start')

a, y = ct.data.column("review"), ct.data.column("cycle_time")
float(y[a == 1].mean() - y[a == 0].mean())   # ~10.8: the naive gap, twice the planted 5.0
```

The treatment is an activity name (did it happen in the case) or a case-constant 0/1 column; the outcome is `"cycle_time"` (case duration in seconds), `"n_events"`, or a case-constant column. With `when="anytime"` the treatment reads "did review ever happen", so the only safe covariates are facts from the case's start: the case-constant attribute columns, `workload_at_start` (how many other cases were open at the first event, the queue confounder), and, on datetime stamps, the calendar features `start_weekday` and `start_hour`. Nothing measured after the treatment can land in the covariates, so the table can't leak the outcome back into the adjustment set.

The landmark modes ask the question at a fixed moment in each case's life instead:

```python
ct2 = case_table(log, "review", "cycle_time", when=("after_events", 2))
ct2.covariates
# ('complexity', 'elapsed_at_landmark', 'workload_at_landmark', 'workload_at_start')
```

`("after_events", k)` sits at the k-th event and `("after_elapsed", x)` at `x` seconds after the first. The treatment becomes "did review happen after the landmark", and prefix features measured at the landmark (elapsed time, events and distinct activities so far, workload then) join the case-start features. Cases that never reach the landmark are dropped and counted in `n_dropped_short`; cases already treated by it are dropped and counted in `n_dropped_treated` (both zero here, since review never lands in this log's first two events). The estimand is the effect among cases that reached the landmark untreated. That is the population the answer applies to, and for a question like "should we review this still-open ticket", it is the honest one. A derived covariate that is constant across the kept cases is dropped in silence, because a zero-variance column breaks discovery; under `after_elapsed`, `elapsed_at_landmark` always equals `x`, so it always drops there.

The table carries knowledge too. `ct.knowledge` is a `DomainKnowledge` with the tier order covariates, then treatment, then outcome, so adjusting for anything downstream of the treatment is forbidden mechanically rather than by convention:

```python
ct.knowledge.permits_edge("cycle_time", "review")   # False: the tiers forbid post-treatment arrows
```

From here the pillars compose. State the graph and estimate:

```python
from ergodic import dag, estimate_effect

g = dag([
    "complexity -> review",
    "complexity -> cycle_time",
    "review -> cycle_time",
    "workload_at_start -> cycle_time",
])
estimate_effect(g, ct.data, "review", "cycle_time", random_state=0).value   # ~5.1
```

Or discover the graph under the table's own knowledge, then estimate:

```python
from ergodic import discover

result = discover(ct.data, knowledge=ct.knowledge, alpha=0.01)
graph = result.graph.as_kind("dag")
est = estimate_effect(graph, ct.data, "review", "cycle_time", strategy="dml", random_state=0)
est.value   # ~5.05, against the planted 5.0
```

PC found `complexity -> review`, both arrows into `cycle_time`, and no workload edge at all, which is correct here: workload was a candidate confounder, and the data cleared it. The `as_kind("dag")` line is the one subtlety. Tier knowledge orients every edge, which puts the result outside the CPDAG class (the graph keeps the mixed face and `diagnostics["cpdag_valid"]` reads 0.0), so you re-face the fully oriented result as the DAG it is; the [discovery guide](discovery.md) covers knowledge-seeded orientations. Either route lands on the planted 5.0 within its standard error. Review costs 5 seconds, and the other 5.8 the drilldown blamed on it belong to complexity.

## Process changes as natural experiments

`case_table` treats per-case interventions. A process change (a new routing rule, a reorganized queue) arrives team by team and week by week, and the comparison that identifies its effect lives in time. `kpi_panel` aggregates the log into that shape. Here 30 teams run 26 weeks of tickets; ten adopt new routing at week 10, ten more at week 18, ten never do. The adopters were about 8 hours slower at baseline, and routing removes exactly 3 hours.

```python
rng = np.random.default_rng(0)
n_teams, n_weeks = 30, 26
adopt = np.array([10] * 10 + [18] * 10 + [n_weeks] * 10)
base = 30.0 + 8.0 * (adopt < n_weeks) + rng.normal(0.0, 1.0, n_teams)  # adopters started slower
shock = rng.normal(0.0, 0.6, n_weeks)                                  # one calendar for every team

rows = []
for team in range(n_teams):
    for week in range(n_weeks):
        routed = float(week >= adopt[team])
        for i in range(int(rng.integers(8, 13))):
            opened = pd.Timestamp("2026-01-05") + pd.Timedelta(days=7 * week + 6.9 * rng.random())
            hours = base[team] + shock[week] - 3.0 * routed + rng.normal(0.0, 1.0)
            case = f"t{team:02d}w{week:02d}c{i:02d}"
            rows.append((case, "received", opened, f"team{team:02d}", routed))
            rows.append((case, "resolved", opened + pd.Timedelta(hours=hours), f"team{team:02d}", routed))

df = pd.DataFrame(rows, columns=["case", "activity", "time", "team", "new_routing"])
log = event_log(df, case="case", activity="activity", timestamp="time")
```

```python
from ergodic import did, kpi_panel

pnl = kpi_panel(log, "week", unit="team", kpis=("cycle_time", "new_routing"))
pnl   # PanelData(..., n_rows=780, entity='team', time='period', ...)

est = did(pnl, "cycle_time", "new_routing", random_state=0)
est.flavor   # 'group_time': two adoption cohorts, so staggered difference-in-differences
est.value    # ~-10808 seconds, against the planted -10800 (3 hours)
```

Each case lands in the bucket of its first event and, when `unit=` is given, in its unit, read from a case-constant column. `"week"` buckets datetime stamps by their Monday (`"day"` and `"month"` also exist); numeric stamps take a positive width instead. A kpi is `"cycle_time"` (the mean case duration in the cell, in seconds on datetime stamps), `"n_cases"`, `"n_events"`, or the mean of any case-constant numeric column. That last rule is how the adoption flag above becomes an exact 0/1 treatment column: adoption aligns to week boundaries, so every cell mean is 0 or 1. The result is the `PanelData` that `did`, `event_study`, and `synthetic_control` consume; the [panel and time-series guide](panel.md) covers those designs and their assumptions. With `unit=None` the whole log is one series, a `TimeSeriesData` that `interrupted_time_series` reads for a single change with no comparison teams.

The naive comparison fails here for the same reason it failed per case: the adopting teams sat 8 hours slower at baseline, so routed cells look slower than never-routed ones even though routing helps. Differencing each team against its own past removes the baseline gap, and the staggered estimator recovers the planted 3 hours.

Two caveats travel with the aggregation. A case is attributed to the bucket where it starts, so a case that spans the rollout counts its whole duration where it began. And the aggregation assumes the log is complete: an export taken while cases are still open is missing exactly the long cases near its end, which biases late-bucket cycle times downward. Trim the tail with `filter_time`, or wait for the cases to close.

## The model

The map counts which step follows which. A model claims more: how the whole process runs, which steps are choices, which repeat, which interleave. `discover_model` mines one from the log with the inductive miner (Leemans, Fahland, and van der Aalst 2013). It builds the directly-follows graph of the traces, looks for a cut (an exclusive choice, a sequence, a parallel split, or a loop), partitions the activities, projects the traces into one sublog per part, and recurses.

A fresh log for this layer: a repair desk, 400 cases. Tickets open and get diagnosed, then cycle through fix and test until the test passes, with a rework step between rounds. Five cases carry the kind of broken stamps this page keeps meeting: fix logged before diagnose.

```python
rng = np.random.default_rng(3)
rows = []
for i in range(400):
    case = f"r{i:03d}"
    steps = ["open", "diagnose", "fix", "test"]
    while rng.uniform() < 0.3:
        steps += ["rework", "fix", "test"]   # a failed test goes back through rework
    steps.append("close")
    if i < 5:
        steps[1], steps[2] = steps[2], steps[1]   # five cases with swapped stamps
    t = rng.uniform(0.0, 5000.0)
    for act in steps:
        rows.append((case, act, t))
        t += rng.uniform(1.0, 3.0)

df = pd.DataFrame(rows, columns=["case", "activity", "time"])
log = event_log(df, case="case", activity="activity", timestamp="time")
```

```python
from ergodic import discover_model

discover_model(log).tree
# seq('open', loop(seq(par(xor(TAU, 'diagnose'), 'fix'), 'test'), 'rework'), 'close')
```

The result is a process tree, read through its operators: `seq` runs its children left to right, `xor` runs exactly one, `par` interleaves them, `loop(do, redo)` runs the do body and then any number of redo-then-do rounds, and `TAU` is the silent step that leaves no event. So this tree claims diagnose is optional and runs in parallel with fix. That's wrong, and the miner is right to say it, because it's what the log shows. On a map the swapped stamps would sit in a few thin arcs you could see and judge; a tree is global, so five bad cases out of 400 reshape the whole structure.

`noise=` is the infrequent-variant filter (the 2014 follow-up paper): at every recursion, before cut detection, a directly-follows edge is dropped when its count falls below `noise` times the heaviest edge leaving the same activity.

```python
model = discover_model(log, noise=0.1)
model.tree              # seq('open', 'diagnose', loop(seq('fix', 'test'), 'rework'), 'close')
model.n_filtered_arcs   # 3: the directly-follows edges the swapped stamps had created
model.fallthroughs      # (): no subtree fell back to the flower model
```

The planted process, exactly. And the rule that governed `simplify(keep=)` governs here too: whatever the miner removes is written on the result. `n_filtered_arcs` counts the filtered edges. `fallthroughs` lists the paths of every subtree where no cut was valid and the miner gave up into the flower model, `loop(TAU, xor(...))`, which accepts its activities in any order; an empty tuple means every subtree came from a real cut. `n_dropped_events` and `dropped_share` count the events discarded when an exclusive-choice cut sends each trace whole to one branch (zero here). A model that reported `noise=0.1` and nothing else would hide how much of the log it ignored; this one can't.

Trees are immutable values with structural equality, and the builders are public, so a mined tree compares directly against one written by hand:

```python
from ergodic.process import loop, seq

model.tree == seq("open", "diagnose", loop(seq("fix", "test"), "rework"), "close")   # True
```

`xor` and `par` keep their children in a canonical order, so equal models are equal objects however they were assembled.

A tree also executes. `model.net` is its workflow net (`tree.to_net()`), a Petri net with one source place, one sink place, and a silent transition wherever the control flow needs glue:

```python
from ergodic.process import is_sound

model.net             # WorkflowNet(places=8, transitions=8 (2 silent), arcs=16)
is_sound(model.net)   # True
```

`is_sound` decides van der Aalst's classical soundness by exploring the reachable markings: from every reachable marking the run can still finish (option to complete), a token on the sink means the run ended with nothing left behind (proper completion), and every transition fires on some path (no dead transitions). A net built from a tree is sound by construction, and the checker still verifies it for real, since nets can arrive from anywhere. A sound net is bounded, so the exploration terminates; past `max_states` distinct markings it raises rather than guess.

## Conformance

`conformance` holds the model fixed and asks where the log deviates from it. The model mined at `noise=0.1` says diagnose comes before fix; the five swapped cases are still in the log, so the check should find exactly those five.

```python
from ergodic import conformance

res = conformance(log, model)
res                 # ConformanceResult(fitness=0.999, alignment_fitness=0.998, fitting=395/400 cases)
res.share_fitting   # 0.9875
```

The fit question is answered twice. Token replay (Rozinat and van der Aalst 2008) pushes each trace through the net and counts the tokens it had to invent (`missing`) and the ones left behind (`remaining`); the standard half-and-half of the two ratios gives the `fitness` of 0.999. Optimal alignments (Adriansyah 2014) run a shortest-path search that walks the trace and the model together and names every deviation. Replay is the fast diagnostic, and it can part-credit a trace that never forms a real run of the net; the alignment is the optimal explanation and costs a search per variant. A case is fitting when its variant aligns at cost zero, which makes `share_fitting` the honest headline: 395 of 400 cases fit, and the five swapped ones are the gap.

Both checks run once per distinct variant and weight by case count, most frequent variant first. The deviating variants name what went wrong:

```python
worst = [v for v in res.variants if v.cost > 0][0]
worst.sequence, worst.n_cases   # (('open', 'fix', 'diagnose', 'test', 'close'), 3)
worst.alignment
# (('sync', 'open'), ('model', 'diagnose'), ('sync', 'fix'),
#  ('log', 'diagnose'), ('sync', 'test'), ('sync', 'close'))
```

A `sync` move is the trace and the model agreeing. A `model` move is a required visible step the trace skipped: diagnose was due after open and didn't come. A `log` move is an event the model can't mirror: the diagnose that showed up after fix. Silent model steps cost nothing and are omitted from the list. Two unit-cost moves against a `worst_cost` of 10 (delete the whole trace, run the model's cheapest visible path) put this variant's `alignment_fitness` at 0.8. The other two deviating variants are the same swap with rework rounds attached.

Every search inside (the silent closures in replay, the alignment itself) is capped by `max_states` and raises past the cap rather than return a partial answer, the same convention `is_sound` follows.

## Decision points

An xor node is where the process decides, and `decision_points` reads a mined model's splits causally: the branch taken is a treatment, the features known when a case reaches the split are the honest adjusters, and how thinly the data covers each branch is a first-class output rather than a surprise in the standard errors later.

One last world, with the truth planted. Tickets open, get triaged, then route through expedite or standard handling before resolving. Complexity drives the routing (hard tickets tend to be expedited) and drives the remaining time on its own; expediting saves exactly 5.0 seconds.

```python
rng = np.random.default_rng(7)
n = 1200
complexity = rng.uniform(0.0, 1.0, size=n)
expedite = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-3.0 * (complexity - 0.5)))
remaining = 12.0 + 8.0 * complexity - 5.0 * expedite + rng.normal(0.0, 1.0, size=n)
start = rng.uniform(0.0, 6000.0, size=n)

rows = []
for i in range(n):
    case = f"d{i:04d}"
    triaged = start[i] + rng.uniform(0.5, 1.5)
    routed = triaged + rng.uniform(0.5, 2.0)
    branch = "expedite" if expedite[i] else "standard"
    rows.append((case, "open", start[i], complexity[i]))
    rows.append((case, "triage", triaged, complexity[i]))
    rows.append((case, branch, routed, complexity[i]))
    rows.append((case, "resolve", routed + remaining[i], complexity[i]))

df = pd.DataFrame(rows, columns=["case", "activity", "time", "complexity"])
log = event_log(df, case="case", activity="activity", timestamp="time")
```

```python
from ergodic import decision_points

model = discover_model(log)
model.tree   # seq('open', 'triage', xor('expedite', 'standard'), 'resolve')
decision_points(log, model)   # (DecisionPoint(path=(2,), expedite=591, standard=609),)
```

`decision_points` locates every xor split and the branch each case took, read from which branch's activity alphabet its events hit; a mined tree carries each activity in exactly one leaf, so the read is unambiguous, and a tree that breaks the invariant raises. A case whose events hit two or more branches is ambiguous (a loop above a split can replay it) and is excluded from the counts; a case hitting none is skipped, or counted as the silent branch when the split has one and is always reached.

`decision_table` encodes one split for the causal stack, one row per unambiguously assigned case. The landmark is the case's first event inside the split's combined alphabet, the moment it reveals its branch, and every covariate is measured strictly before that moment, the same gate `case_table`'s landmark modes apply. The default outcome is `remaining_time`, seconds from the landmark to the case's last event, because time already spent before the decision isn't an effect of the branch. And the tier knowledge rides along as before: covariates, then `branch`, then the outcome.

```python
from ergodic import decision_table

dt = decision_table(
    log, model, covariates=["complexity", "elapsed_at_landmark", "workload_at_landmark"]
)
dt   # DecisionTable(cases=1200, branches=('expedite', 'standard'), outcome='remaining_time', covariates=3)

b, y = dt.data.column("branch"), dt.data.column("remaining_time")
float(y[b == "expedite"].mean() - y[b == "standard"].mean())   # ~-3.1, against the planted -5.0
```

The naive branch gap undersells expedite by nearly two seconds, the same trap as ever: expedite gets the complex tickets, and complex tickets run long. With `covariates="auto"` the table would also take `workload_at_start`; we name three because the two workload reads are the same queue a few seconds apart, and a near-duplicate pair leaves discovery an undirected covariate edge the tiers can't orient.

Before estimating, read the positivity report:

```python
p = dt.positivity
p.shares          # (0.4925, 0.5075)
p.overlap_share   # 1.0
p.flagged         # False
```

`overlap_share` is the share of cases whose in-sample propensity, a logistic regression of the branch on the covariates, lands inside [0.05, 0.95]. Here it's 1.0: by the covariates, every ticket had a real chance of going either way, so the branch comparison rests on data everywhere. `flagged` turns True when a branch holds under five percent of the cases or fewer than ten, or when `overlap_share` falls below 0.8, meaning the covariates nearly decide the branch for over a fifth of the cases and the comparison there rests on extrapolation. A flag is a warning that the branches barely overlap, not a verdict that an estimate is wrong, so it reports instead of raising.

Estimation is the same composition the case table used: discover under the table's knowledge, re-face the fully oriented result as the DAG it is, and estimate with `treated=` naming a branch; the [inference guide](inference.md) covers the estimators and learners.

```python
result = discover(dt.data, knowledge=dt.knowledge, alpha=0.01)
graph = result.graph.as_kind("dag")
est = estimate_effect(graph, dt.data, "branch", "remaining_time", strategy="dml",
                      random_state=0, treated="expedite")
est.value   # ~-5.01, against the planted -5.0
```

PC found `complexity -> branch`, both arrows into `remaining_time`, and no edge from the workload or elapsed-time features: they were candidate confounders, and the data cleared them. Expediting saves the planted 5 seconds, and the 1.9 the naive gap hid belonged to complexity.

A split with a silent branch is refused: taking `TAU` leaves no event, so the decision moment is unobservable without alignment-based replay (future work; `decision_points` still reports such splits). Classic decision mining fits a classifier at the split and predicts which branch a case will take. `decision_table` asks what taking the other branch would do, which is the number a routing change needs.

## Limits and what's next

The per-case estimators assume cases don't interfere, and in a real process they do: cases share resources, so reviewing one ticket can slow its neighbours through the queue. No method on this page pretends otherwise. When an intervention shifts a whole queue, prefer the panel route, where the unit is the team-week rather than the case. The other standing assumption is that constant attribute columns are known at case start: a label back-filled at close is post-treatment however constant it looks, and the table can't tell when a constant column was written.

The frontier comes next: off-policy evaluation of intervention policies, effect drift over time, and object-centric logs that drop the single-case assumption. The [API reference](reference.md) covers the full surface of what's here today.
