# Map a process from an event log

Turn an event log into a process map, cycle times and variants, then estimate what
one step does to the cycle time. The example uses simulated support tickets, where
the raw comparison overstates the effect by a factor of two.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - An event log: one row per event, with a case id, an activity name and a
      timestamp (datetime or numeric).

## Run the program

Each ticket is opened, triaged, sometimes reviewed, and closed. Complex tickets are
reviewed more often and take longer anyway. Review itself adds 5 hours.

```python file="process_map.py"
import numpy as np
import pandas as pd
from ergodic import case_table, discover, estimate_effect, event_log, kpi_panel, process_map
from ergodic.process import END, START, filter_activities, performance, variant_table

# 1. Simulate 1,500 tickets over 90 days; timestamps are in hours.
#    Complex tickets get reviewed more often, and review adds 5 hours.
rng = np.random.default_rng(11)
n = 1500
complexity = rng.normal(size=n)
review = rng.uniform(size=n) < 1 / (1 + np.exp(-1.3 * complexity))
hours = np.maximum(20 + 6 * complexity + 5 * review + rng.normal(0, 2, n), 3)
opened = rng.uniform(0, 90 * 24, n)
rows = []
for i in range(n):
    steps = [("open", 0.0), ("triage", 1.0)] + [("review", 2.0)] * int(review[i])
    for activity, at in [*steps, ("close", hours[i])]:
        rows.append((f"T{i:04d}", activity, opened[i] + at, complexity[i]))
frame = pd.DataFrame(rows, columns=["ticket", "activity", "hour", "complexity"])
log = event_log(frame, case="ticket", activity="activity", timestamp="hour")
print(log)

# 2. Map the process: how often each step follows another, and the mean gap.
pmap = process_map(log)
for (a, b), arc in pmap.arcs.items():
    if a != START and b != END:
        print(f"{a:>6} -> {b:<6} {arc.count:5d} times  {arc.mean_gap:5.1f} h")

# 3. Summarize cycle times and variants, and filter to reviewed tickets.
perf = performance(log)
print(f"cycle time: median {perf.median_duration:.1f} h, p90 {perf.p90_duration:.1f} h")
for variant in variant_table(log):
    print(" > ".join(variant.sequence), f"{variant.share:.0%}", f"{variant.mean_duration:.1f} h")
print(filter_activities(log, mandatory=["review"]))

# 4. One row per ticket, then discover the graph and estimate the effect of review.
table = case_table(log, "review", "cycle_time")
print(table)
graph = discover(table.data, knowledge=table.knowledge, alpha=0.01).graph
print(sorted(edge.glyph() for edge in graph.edges))
estimate = estimate_effect(
    graph.as_kind("dag"), table.data, "review", "cycle_time", strategy="dml", random_state=0
)
print(estimate)

# 5. Weekly KPIs for the panel designs.
panel = kpi_panel(log, 7 * 24, kpis=("n_cases", "cycle_time"))
print(panel.to_pandas().head(3).round(1))
```

```output from="process_map.py"
EventLog(cases=1500, events=5264, activities=4)
  open -> triage  1500 times    1.0 h
review -> close    764 times   25.9 h
triage -> close    736 times   16.1 h
triage -> review   764 times    1.0 h
cycle time: median 22.6 h, p90 32.3 h
open > triage > review > close 51% 27.9 h
open > triage > close 49% 17.1 h
EventLog(cases=764, events=3056, activities=4)
CaseTable(cases=1500, treatment='review', outcome='cycle_time', covariates=2, when='anytime')
['complexity --> cycle_time', 'complexity --> review', 'cycle_time <-- review']
EffectEstimate(dml, analytic: ate=5.053, se=0.115, 95% CI [4.83, 5.28], n=1500)
   period  n_cases  cycle_time
0     0.0    123.0        22.4
1   168.0    107.0        21.7
2   336.0    130.0        22.2
```

- **Step 1.** `event_log` groups the rows by case and orders them by time.
- **Step 2.** `process_map` counts directly-follows arcs: every triage is followed
  by a review or a close. It also holds `START` and `END` arcs for where cases enter
  and leave, which the loop skips because they have no gap.
- **Step 3.** Reviewed tickets take 27.9 hours on average and the others 17.1, a
  raw gap of 10.8 hours. `filter_activities` keeps the 764 tickets that contain a
  review.
- **Step 4.** `case_table` builds one row per ticket. `discover` finds that
  complexity drives both review and cycle time, and the estimate of review's effect
  is 5.05 hours, close to the planted 5. The raw gap was mostly complexity.
- **Step 5.** Each row is a 168-hour bucket of the tickets that started in it.

## Build the case table

`case_table(log, treatment, outcome="cycle_time", covariates="auto", when="anytime")`
is the bridge from a log to the causal tools. The treatment is an activity (did the
case contain it?) or a 0/1 column that is constant within each case. The outcome is
`"cycle_time"` (first event to last), `"n_events"`, or a case-constant column.

With `covariates="auto"` it takes every case-constant column plus derived features
known when the case starts. Here that's `complexity` and `workload_at_start`, the
number of cases open when the ticket arrived. With datetime timestamps it also adds
`start_weekday` and `start_hour`, and durations are in seconds instead of the
timestamps' own units.

`table.knowledge` orders the columns in tiers: covariates, then treatment, then
outcome. Discovery can't draw an arrow from the outcome or treatment back into a
covariate, so it can't adjust for something review caused.

!!! caution "Turn the result into a DAG before estimating"
    `discover` returns a mixed graph that stands for a class of DAGs. The tier
    knowledge orients many of its edges, but the graph keeps that kind. Call
    `graph.as_kind("dag")` before `estimate_effect`. It raises `InvalidGraphError`
    if an edge is still undirected, which can happen between two covariates in the
    same tier. If it does, pass the covariates you need with `covariates=[...]`.

## Choose when the treatment counts

`when="anytime"` derives its features from the start of the case. Case-constant
columns are different: `"auto"` takes all of them and assumes each was known when
the case started. A column filled in later, such as a resolution code, would be a
post-treatment covariate and could bias the estimate. Drop such columns, or name
the covariates you trust with `covariates=[...]`.

To ask about treatments that happen later, use a landmark:

- `when=("after_events", k)` measures each case after its first `k` events.
- `when=("after_elapsed", x)` measures it `x` seconds after it starts (in the
  timestamps' own units when they are numbers).

Cases treated before the landmark are dropped (`n_dropped_treated`), and so are
cases that end before it (`n_dropped_short`). The estimate then applies to cases
that reach the landmark untreated.

## Filter the log

Each filter returns a new log and raises instead of returning an empty one. Its
docstring says whether it drops whole cases or single events:

- `filter_time(log, start=, end=, mode="intersecting")`: cases in a time window;
  `"contained"` keeps only cases fully inside, `"trim"` drops events outside.
- `filter_variants(log, top=)` or `min_cases=`: frequent activity sequences.
- `filter_endpoints(log, start_in=, end_in=)`: cases by first and last activity.
- `filter_activities(log, keep=, mandatory=, forbidden=)`: cases by activities;
  `keep` drops the other events.
- `filter_follows(log, a, b, directly=True, within=)`: cases where `b` follows `a`.
- `filter_performance(log, min_duration=, max_duration=, min_events=, max_events=)`.

<details markdown="1"><summary>How do I get waiting and service times, or a simpler map?</summary>

- With one timestamp per event, `performance` can't split waiting from working:
  each activity's `mean_wait` is the gap from the previous event, and
  `mean_service` is `None`. Pass `lifecycle="column"` when a column marks `"start"`
  and `"complete"` rows. Unpaired rows are counted in `n_unmatched`.
- `pmap.simplify(keep=0.8)` keeps the busiest arcs that cover 80% of traversals.
  The result records `dropped_arcs` and `dropped_share`.
- `kpi_panel(log, "week", unit="team")` with datetime stamps returns a `PanelData`
  by team and week, the input for difference-in-differences. Without `unit` it
  returns one `TimeSeriesData`.

</details>

## Next steps

<div class="cards" markdown="1">

- [Check conformance and decision points](conformance-and-decisions.md): mine a model and estimate the effect of a branch.
- [Difference-in-differences and event studies](difference-in-differences.md): use the KPI panel.
- [Python reference](reference-process.md): every process function and its options.

</div>
