# Check conformance and decision points

Mine a process model from an event log, check how well the log fits it, and estimate
the effect of taking one branch of a decision. The example routes cases to an
expedited or a standard track.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - An event log; [Map a process from an event log](map-a-process.md) shows how to
      build one.

## Run the program

Each case is opened, triaged, sent to `expedite` or `standard`, and resolved.
Complex cases are expedited more often and take longer anyway. Expediting cuts the
time left by 5 hours. About 3% of standard cases escalate to expedite midway.

```python file="decision_points.py"
import numpy as np
import pandas as pd
from ergodic import conformance, decision_points, decision_table, discover, discover_model
from ergodic import estimate_effect, event_log
from ergodic.process import filter_variants, is_sound, seq, xor

# 1. Simulate 2,000 cases; timestamps are in hours. Complex cases are expedited
#    more often, expediting cuts 5 hours, and some standard cases escalate.
rng = np.random.default_rng(7)
n = 2000
complexity = rng.uniform(0, 1, n)
expedite = rng.uniform(size=n) < 1 / (1 + np.exp(-3 * (complexity - 0.5)))
remaining = 10 + 8 * complexity - 5 * expedite + rng.normal(0, 1, n)
escalate = ~expedite & (rng.uniform(size=n) < 0.03)
opened = rng.uniform(0, 90 * 24, n)
rows = []
for i in range(n):
    t = opened[i]
    decision = t + 1.0 + rng.uniform(0.5, 2.0)
    branch = "expedite" if expedite[i] else "standard"
    steps = [("open", t), ("triage", t + 1.0), (branch, decision)]
    steps += [("expedite", decision + 1.0)] * int(escalate[i])
    steps.append(("resolve", decision + remaining[i]))
    rows += [(f"C{i:04d}", activity, at, complexity[i]) for activity, at in steps]
frame = pd.DataFrame(rows, columns=["case", "activity", "hour", "complexity"])
log = event_log(frame, case="case", activity="activity", timestamp="hour")

# 2. Mine a model from the whole log, then from its two most frequent variants.
print(discover_model(log).tree)
model = discover_model(filter_variants(log, top=2))
print(model.tree)
print(model.tree == seq("open", "triage", xor("expedite", "standard"), "resolve"))
print(model.net, "sound:", is_sound(model.net))

# 3. Check the whole log against the model.
fit = conformance(log, model)
print(fit)
for variant in fit.variants:
    print(variant.n_cases, f"{variant.alignment_fitness:.3f}", variant.alignment)

# 4. Find the decision, and encode it as one row per case.
print(decision_points(log, model))
table = decision_table(
    log, model, covariates=["complexity", "elapsed_at_landmark", "workload_at_landmark"]
)
print(table)
positivity = table.positivity
print([round(share, 3) for share in positivity.shares], positivity.overlap_share, positivity.flagged)
by_case = table.data.to_pandas()
print(by_case.groupby("branch", observed=True)["remaining_time"].mean().round(2))

# 5. Discover the graph and estimate the effect of expediting.
graph = discover(table.data, knowledge=table.knowledge, alpha=0.01).graph
estimate = estimate_effect(
    graph.as_kind("dag"), table.data, "branch", "remaining_time",
    strategy="dml", treated="expedite", random_state=0,
)
print(estimate)
```

```output from="decision_points.py"
seq('open', 'triage', xor(TAU, 'standard'), xor(TAU, 'expedite'), 'resolve')
seq('open', 'triage', xor('expedite', 'standard'), 'resolve')
True
WorkflowNet(places=5, transitions=5 (0 silent), arcs=10) sound: True
ConformanceResult(fitness=0.998, alignment_fitness=0.999, fitting=1976/2000 cases)
1003 1.000 (('sync', 'open'), ('sync', 'triage'), ('sync', 'expedite'), ('sync', 'resolve'))
973 1.000 (('sync', 'open'), ('sync', 'triage'), ('sync', 'standard'), ('sync', 'resolve'))
24 0.889 (('sync', 'open'), ('sync', 'triage'), ('sync', 'standard'), ('log', 'expedite'), ('sync', 'resolve'))
(DecisionPoint(path=(2,), expedite=1003, standard=973, skipped=0, ambiguous=24),)
DecisionTable(cases=1976, branches=('expedite', 'standard'), outcome='remaining_time', covariates=3, dropped_ambiguous=24)
[0.508, 0.492] 1.0 False
branch
expedite     9.89
standard    13.10
Name: remaining_time, dtype: float64
EffectEstimate(dml, analytic: ate=-4.981, se=0.0489, 95% CI [-5.08, -4.88], n=1976)
```

- **Step 2.** Mined from the whole log, the model makes `standard` and `expedite`
  each optional (`xor(TAU, ...)`), because the escalations show them together.
  Mined from the two main variants, it is the intended process, equal to the tree
  built by hand. Its workflow net is sound.
- **Step 3.** 1,976 of 2,000 cases fit exactly. The 24 escalations have one
  `("log", "expedite")` move, an event the model can't mirror, so their alignment
  fitness is 0.889.
- **Step 4.** There is one decision, at path `(2,)`. The 24 escalated cases touch
  both branches, so they're dropped as ambiguous. The branches are close to 50/50
  and every case's estimated propensity is inside [0.05, 0.95], so positivity isn't
  flagged. Expedited cases have 3.21 hours less remaining time on average.
- **Step 5.** Adjusted for complexity, expediting cuts the remaining time by about
  5 hours. The raw gap understated it, because complex cases get expedited.

## Mine and check a model

`discover_model(log, noise=0.0)` is the Inductive Miner. It returns a
`ProcessModel` with a `tree` (a `ProcessTree`), its `net` (a `WorkflowNet`), and
what it removed on the way. `noise` drops each directly-follows arc lighter than
`noise` times the heaviest arc leaving the same activity; the count is in
`n_filtered_arcs`. The filter removes arcs, not traces, so an activity missing from
some traces still becomes optional at any `noise`. Mining from `filter_variants` is
the more direct way to get the main path.

A process tree is built from `seq`, `xor`, `par` and `loop(do, redo)`, with `TAU`
for a silent step. Strings become activity leaves. `tree.to_net()` converts it to a
sound workflow net. `is_sound(net)` checks any net for soundness and raises past
`max_states=100_000` reachable markings rather than guess.

`conformance(log, model)` takes a `ProcessModel`, a `ProcessTree` or a `WorkflowNet`
and answers twice:

- **Token replay** (`fit.fitness`) counts the tokens it had to add and the ones left
  behind. It's fast, but it can give part credit to a trace that never forms a real
  run.
- **Optimal alignments** (`fit.alignment_fitness`) find the cheapest way to walk the
  trace and the model together. Each `variant.alignment` lists moves: `sync` where
  they agree, `log` for an event the model can't mirror, and `model` for a visible
  step the trace skipped.

## Read a decision causally

At an `xor`, the branch a case takes is a treatment. `decision_table` measures each
case at the landmark, its first event inside the split. The covariates come from
before that moment, and the default outcome is `remaining_time`: from the landmark
to the last event. Time already spent before the decision can't be an effect of it.

`table.knowledge` orders covariates, then `branch`, then the outcome. Pass
`treated=` with a branch name when you estimate.

!!! caution "Read the positivity report"
    `table.positivity` is flagged when a branch has under 5% of the cases, any
    branch has fewer than 10 cases, or, for two branches, fewer than 80% of cases
    have a propensity in [0.05, 0.95]. A flag means the branches barely overlap, and
    the estimate there leans on extrapolation.

Here the covariates are listed by hand. With `covariates="auto"`, the table also
includes `workload_at_start`, and discovery leaves the edge between the two workload
columns undirected. `as_kind("dag")` then raises `InvalidGraphError`.

<details markdown="1"><summary>What can't a decision table handle?</summary>

- A tree with several `xor` nodes needs `point=`: a path from `decision_points`, or
  the `DecisionPoint` itself.
- A split with a silent branch (`xor(TAU, ...)`) raises: taking it leaves no event,
  so the decision moment can't be seen. `decision_points` still reports it.
- Every activity label must sit in exactly one leaf of the tree. A mined tree
  satisfies this; a hand-built tree that repeats a label raises.
- A log column named `branch` collides with the treatment column and raises.
- A replay or alignment search past `max_states` raises instead of returning a
  partial answer. So does a net that can't reach its final marking.

</details>

## Next steps

<div class="cards" markdown="1">

- [Map a process from an event log](map-a-process.md): maps, variants, filters and the case table.
- [Estimate heterogeneous effects](heterogeneous-effects.md): see which cases gain most from a branch.
- [Python reference](reference-process.md): every process function and its options.

</div>
