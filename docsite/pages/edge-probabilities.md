# Work with edge probabilities

Hold a model's directed-edge probabilities over named variables, read them in the form
a decoder needs, and see where they disagree with any single graph. You'll also turn
bootstrap frequencies into the same object.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A square matrix of edge probabilities, rows the source and columns the target.
      AVICI and CDFM on Welt return one; so does any model that scores edges.

## Run the program

The matrix below is made up for four variables. Read `values[i, j]` as "how likely is
an edge from `names[i]` to `names[j]`".

```python file="edge_probabilities.py"
import numpy as np

from ergodic import EdgeProbabilities, tabular
from ergodic.discovery import bootstrap

# 1. Scores a model returned for four variables: rows the source, columns the target.
names = ("rain", "sprinkler", "wet", "slippery")
values = np.array(
    [
        [0.00, 0.45, 0.91, 0.06],
        [0.40, 0.00, 0.83, 0.05],
        [0.04, 0.07, 0.00, 0.88],
        [0.58, 0.02, 0.10, 0.00],
    ]
)
probs = EdgeProbabilities(names, values, kind="independent", provenance={"model": "example"})
print(probs)

# 2. Every edge above 0.5, as a diagnostic graph.
print(probs.threshold(0.5))

# 3. Three exclusive states per pair: no edge, one way, the other way.
pair = probs.to_pair_states()
print(pair.to_frame().round(3))

# 4. The chance each pair is adjacent, either way.
print(probs.existence().round(3))

# 5. Bootstrap frequencies of a discovery method, as edge probabilities.
rng = np.random.default_rng(0)
x = rng.normal(size=500)
y = 0.8 * x + rng.normal(size=500)
z = 0.8 * y + rng.normal(size=500)
data = tabular({"x": x, "y": y, "z": z})
freq = EdgeProbabilities.from_bootstrap(bootstrap(data, method="ges", n_bootstrap=20, seed=0))
print(freq)
print(freq.to_frame())
```

```output from="edge_probabilities.py"
EdgeProbabilities(independent, 4 variables, 4 directed cells above 0.5, uncalibrated)
MixedGraph(kind=mixed, nodes=4, edges=[rain <-- slippery, rain --> wet, slippery <-- wet, sprinkler --> wet])
target      rain  sprinkler    wet  slippery
source
rain       0.000      0.329  0.907     0.026
sprinkler  0.268      0.000  0.820     0.049
wet        0.004      0.013  0.000     0.868
slippery   0.565      0.019  0.013     0.000
[[0.    0.598 0.91  0.591]
 [0.598 0.    0.832 0.068]
 [0.91  0.832 0.    0.882]
 [0.591 0.068 0.882 0.   ]]
EdgeProbabilities(frequency, 3 variables, 0 directed cells above 0.5, uncalibrated)
target      x    y      z
source
x       0.000  0.5  0.025
y       0.500  0.0  0.500
z       0.025  0.5  0.000
```

- **Step 1.** `kind="independent"` says each number is a marginal on its own, so both
  directions of a pair may be high at once. The object stores a read-only copy, checks
  the diagonal is zero and every value is in `[0, 1]`, and keeps `provenance` as you
  gave it.
- **Step 2.** Four edges pass 0.5, and three of them, `rain --> wet --> slippery -->
  rain`, form a cycle. `threshold` shows that on purpose: it returns a `mixed` graph
  that may be cyclic, not a DAG.
- **Step 3.** `to_pair_states` conditions each pair on "not both directions": for
  `p = P[i, j]` and `q = P[j, i]` the states are `(1 - p)(1 - q)`, `p(1 - q)` and
  `q(1 - p)`, renormalized. Each pair's two cells now sum to at most 1; the rest is the
  no-edge mass. The ranking of directions is kept.
- **Step 4.** `existence` adds the two directions of each pair in that view. `rain` and
  `wet` are adjacent with probability 0.91, `sprinkler` and `slippery` with 0.068.
- **Step 5.** On a chain `x -> y -> z`, GES returns undirected edges, because the data
  can't tell the directions apart. `from_bootstrap` splits each undirected edge's
  frequency evenly between the two directions, so `x` and `y` get 0.5 each way.

## Read the object

| Field or method | What it gives |
| --- | --- |
| `variables` | The names, in row and column order. |
| `values` | The read-only `(d, d)` matrix, source rows and target columns. |
| `kind` | `"independent"`, `"pair_state"` or `"frequency"`. |
| `calibrated` | `False` unless you state the numbers are calibrated. ergodic never sets it. |
| `provenance` | Read-only metadata, such as the model, its version and the job. |
| `to_pair_states()` | The three-state view a decoder reads. |
| `existence()` | Skeleton probabilities, symmetric. |
| `threshold(t)` | Edges with value above `t`, as a diagnostic `mixed` graph. |
| `to_frame()`, `from_frame(frame, kind=...)` | A dataframe round trip, index `source` and columns `target`. |
| `from_bootstrap(result, undirected="split")` | Directed-edge frequencies from a `bootstrap` run. |

The three kinds read the numbers differently:

- **`independent`**: one marginal per directed edge. AVICI and CDFM return these.
- **`pair_state`**: three exclusive states per pair, so `P[i, j] + P[j, i] <= 1`.
  Building one that breaks this raises.
- **`frequency`**: the share of runs that drew each directed edge. One edge per pair per
  run, so the same bound holds.

!!! caution "Not a graph posterior"
    Edge probabilities are per-edge numbers. They ignore acyclicity and how edges
    depend on each other, and a model's scores are usually uncalibrated. Turning them
    into a DAG is a modelling choice you make with a decoder, not something the model
    claims. `graph_posterior` builds the posterior over DAGs they imply; see
    [Evaluate a candidate graph](evaluate-a-graph.md).

`threshold` draws a pair whose two directions both pass as an undirected `X --- Y`
edge, because a graph holds one edge per pair. Read it as "adjacent, direction
unresolved".

`from_bootstrap` drops bidirected and circle-marked glyphs, which say something other
than "this direction", and records the dropped share in
`provenance["dropped_frequency"]`. Pass `undirected="drop"` to drop undirected edges
too. It reads results over a table's variables; a window-graph bootstrap from PCMCI is
refused.

## Next steps

<div class="cards" markdown="1">

- [Decode a DAG from edge probabilities](decode-a-graph.md): turn the matrix into a DAG.
- [Evaluate a candidate graph](evaluate-a-graph.md): the posterior over DAGs, and a graph read against it.
- [Check stability with the bootstrap](bootstrap-stability.md): where frequencies come from.
- [Decode AVICI and CDFM](welt.md#decode-avici-and-cdfm): edge probabilities from Welt.

</div>
