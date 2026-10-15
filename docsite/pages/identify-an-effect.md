# Check whether an effect is identifiable

Ask whether a causal graph lets you recover an effect from observational data, and get
the formula to estimate if it does. You'll check five graphs, including two from
discovery.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A graph: drawn by you with `dag`, or returned by
      [discovery](discover-a-graph.md).

## Run the program

Identification needs no data. It reads only the graph. `U` marks a common cause you
never measured: pass it in `latent=`.

```python file="identify_effect.py"
from ergodic import identify_effect
from ergodic.graph import cpdag, dag, mixed

# 1. Four single graphs in which X may cause Y.
worlds = {
    "measured confounder": dag(["Z -> X", "Z -> Y", "X -> Y"]),
    "hidden confounder, mediator": dag(["U -> X", "U -> Y", "X -> M", "M -> Y"], latent=["U"]),
    "hidden confounder, instrument": dag(["Z -> X", "U -> X", "U -> Y", "X -> Y"], latent=["U"]),
    "hidden confounder only": dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"]),
}
for name, graph in worlds.items():
    result = identify_effect(graph, "X", "Y")
    print(f"{name}: {result}")

# 2. The hedge explains why the last one fails.
print(result.hedge)

# 3. Two equivalence classes, as discovery returns them.
for graph in [cpdag(["Z --- X", "X --> Y", "W --> Y"]), cpdag(["Z --- X", "X --- Y"])]:
    result = identify_effect(graph, "X", "Y")
    print(bool(result), result.strategy, sorted(result.adjustment_set or []))

# 4. A graph without a verified face is refused, with advice.
try:
    identify_effect(mixed(["Z -> X", "Z -> Y", "X -> Y"]), "X", "Y")
except ValueError as error:
    print(error)
```

```output from="identify_effect.py"
measured confounder: Identification(adjustment: P(Y | do(X)) = sum_z P(Y | X, z) P(z), with z = {Z})
hidden confounder, mediator: Identification(frontdoor: P(Y | do(X)) = sum_m P(m | X) sum_x' P(Y | m, x') P(x'), with m = {M})
hidden confounder, instrument: Identification(iv: identified through instrument(s) Z under a linear (parametric) model; needs an IV estimator)
hidden confounder only: Identification(not identifiable, ['X'] -> ['Y'])
not identifiable: a hedge formed by ['X', 'Y'] and the c-component ['Y']
True adjustment ['W', 'Z']
False none []
identification needs a verified graph face (dag, admg, mag, cpdag, or pag), not 'mixed'; every edge is directed and there is no directed cycle. If it came from a search that assumes no hidden confounding (PC or GES with domain knowledge orienting every edge, say) and you read it as one causal world, re-face it with graph.as_kind('dag'); if it came from FCI or RFCI (or otherwise means ancestry, where a directed edge can hide confounding), re-face it with graph.as_kind('mag') instead, which identification reads as an equivalence class
```

- **Step 1.** Each world gets a different strategy. With `Z` measured, adjust for it.
  With `U` hidden but a mediator `M` measured, use the front-door formula. With an
  instrument `Z`, the effect is identified under a linear model only. With nothing
  else, it isn't identifiable.
- **Step 2.** The hedge names the structure that blocks identification: here `X` and
  `Y` share a hidden cause, and `X -> Y` is the only path.
- **Step 3.** On a CPDAG, a success holds in every graph of the class. The first class
  orients `X --> Y` and every member agrees on `{W, Z}`. In the second, `X --- Y` may
  point either way, so the members disagree and the answer is no.
- **Step 4.** A `mixed` graph has no verified meaning. The error says which kind it
  fits and how to re-face it.

## Read the result

`identify_effect(graph, treatment, outcome)` returns an `Identification`. It is truthy
when the effect is identifiable.

| Field | What it holds |
| --- | --- |
| `identifiable` | `True` or `False`. |
| `strategy` | `"null"`, `"adjustment"`, `"frontdoor"`, `"iv"`, `"id"`, or `"none"`. |
| `adjustment_set` | The set to adjust for, with `"adjustment"`. |
| `mediators` | The front-door set, with `"frontdoor"`. |
| `instruments` | `(instrument, conditioning set)` pairs, with `"iv"`. |
| `estimand` | The formula as text, or the reason it failed. |
| `functional` | The typed formula, with `"id"`. |
| `hedge` | The witness of failure, when the general algorithm proves none exists. |

The strategies are tried in order. First, if no directed path runs from treatment to
outcome, the effect is zero: `"null"`. Then adjustment, the front-door and
instruments. On a single graph, the general do-calculus algorithm (Shpitser and
Pearl, 2006) runs last. It returns a formula or a hedge. When it recognizes the
formula as adjustment or the front door, the strategy says so; otherwise it is
`"id"`.

## What each graph kind allows

| Kind | Where it comes from | What can identify |
| --- | --- | --- |
| `dag`, `admg` | Your drawing; LiNGAM | Every strategy, unless it marks `selection=` nodes |
| `cpdag` | PC, GES, BOSS, GRaSP | Null and adjustment, valid in every member |
| `pag`, `mag` | FCI, RFCI | Null and adjustment, valid in every member |
| `mixed` | Discovery with orienting knowledge, screens | Nothing until you re-face it |

A* and HillClimb also return a `dag`, but it is one member of an equivalence class,
picked by tie-breaking among DAGs with the same score. Identify on
`cpdag_of_dag(graph)` instead, so the answer doesn't rest on an arbitrary direction.

A graph that marks selection nodes (`selection=`) raises on every kind. The criteria
assume the data were not selected on any node, and an effect generally isn't
identifiable from a selected sample: with `X -> S <- Y` and `S` selected, `X` and `Y`
are dependent in the sample even though `X` has no effect on `Y`.

On a class, the front-door and instruments need a single graph. To see each member's
answer, use `ida`, described in
[Find adjustment sets, front-door sets and instruments](adjustment-sets.md).

!!! caution "An instrument identifies only under a linear model"
    `strategy="iv"` says so in its estimand. The estimate assumes the effect is
    linear and the same for everyone. Graphs where the instrument rule fires first,
    such as the napkin graph, can also be identified without that assumption: call
    `identify_distribution(graph, "Y", "X")` to get that formula.

<details markdown="1"><summary>What does an `"id"` result look like?</summary>

The Verma graph `A -> B -> C -> D`, with hidden causes shared by `A` and `C` and by `B`
and `D`, has no adjustment set, front-door set or instrument for `A -> D`. The general
algorithm still identifies it. `result.functional` holds the formula as a typed object,
and `result.estimand` is its text. `estimate_effect` can evaluate it on discrete data.

</details>

## Next steps

<div class="cards" markdown="1">

- [Find adjustment sets, front-door sets and instruments](adjustment-sets.md): the sets behind each strategy.
- [Estimate an average effect](estimate-average-effect.md): turn the formula into a number.
- [Python reference](reference-identification.md): `identify_effect` and `Identification`.

</div>
