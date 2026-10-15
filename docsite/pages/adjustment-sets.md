# Find adjustment sets, front-door sets and instruments

Find the variables to adjust for, check a set you already have in mind, and look for
front-door sets and instruments when adjustment fails. You'll also list the answers
across an equivalence class.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A graph you trust. [Check whether an effect is identifiable](identify-an-effect.md)
      shows which strategy applies.

## Run the program

The first graph is a known trap. `Z3` confounds `X` and `Y`, but it is also a
collider: both `Z1` and `Z2` cause it. `W` mediates the effect and `I` affects only
`X`.

```python file="adjustment_sets.py"
from ergodic.graph import cpdag, dag
from ergodic.identification import (
    adjustment_set,
    frontdoor_set,
    ida,
    instruments,
    is_instrument,
    is_valid_adjustment_set,
    minimal_adjustment_set,
)

# 1. Confounders Z1, Z2, Z3, a mediator W and an instrument I.
graph = dag([
    "Z1 -> X", "Z1 -> Z3", "Z2 -> Z3", "Z2 -> Y", "Z3 -> X", "Z3 -> Y",
    "X -> W", "W -> Y", "I -> X",
])

# 2. The canonical adjustment set, and a minimal one.
print("canonical:", sorted(adjustment_set(graph, "X", "Y")))
print("minimal:", sorted(minimal_adjustment_set(graph, "X", "Y")))

# 3. Check sets of your own.
for candidate in [[], ["Z3"], ["Z1", "Z3"], ["Z3", "W"]]:
    print(candidate, is_valid_adjustment_set(graph, "X", "Y", candidate))

# 4. Instruments, then a check with a larger conditioning set.
print("instruments:", instruments(graph, "X", "Y"))
print("Z1 given Z2, Z3:", is_instrument(graph, "Z1", "X", "Y", given=["Z2", "Z3"]))

# 5. With the confounder hidden, adjustment fails but the front door is open.
hidden = dag(["U -> X", "U -> Y", "X -> M", "M -> Y"], latent=["U"])
print("adjustment:", adjustment_set(hidden, "X", "Y"))
print("front-door:", sorted(frontdoor_set(hidden, "X", "Y")))

# 6. On a CPDAG, identify the effect in each member graph.
result = ida(cpdag(["Z --- X", "X --- Y", "Z --- Y"]), "X", "Y")
print(result)
print(sorted(member.strategy for member in result.members))
```

```output from="adjustment_sets.py"
canonical: ['I', 'Z1', 'Z2', 'Z3']
minimal: ['Z2', 'Z3']
[] False
['Z3'] False
['Z1', 'Z3'] True
['Z3', 'W'] False
instruments: [('I', frozenset())]
Z1 given Z2, Z3: True
adjustment: None
front-door: ['M']
IDA(['X'] -> ['Y'], cpdag: 6/6 identify, strategies=['adjustment', 'null'], agree=False)
['adjustment', 'adjustment', 'adjustment', 'null', 'null', 'null']
```

- **Step 2.** The canonical set holds every ancestor of `X` or `Y` that `X` doesn't
  affect. If any valid set exists, this one is valid. It includes `I`. That adds no
  bias when the graph is right, but it usually widens the interval, and if a
  confounder is missing from the graph it makes that bias larger. `{Z2, Z3}` is a smaller valid
  set.
- **Step 3.** The empty set leaves `X <- Z3 -> Y` open. `{Z3}` closes it but opens
  `X <- Z1 -> Z3 <- Z2 -> Y`, because conditioning on a collider connects its causes.
  Adding `Z1` closes that path again. `W` sits on the causal path, so any set holding
  it is invalid.
- **Step 4.** `I` is an instrument with no conditioning. `Z1` is one only given both
  `Z2` and `Z3`; the search tries just the empty set and single variables, so it
  doesn't list it.
- **Step 5.** `adjustment_set` returns `None` when no valid set exists. The mediator
  `M` satisfies the front-door criterion instead.
- **Step 6.** The class holds 6 DAGs. In 3 the effect is zero, because `Y` causes `X`;
  in 3 it is identified by adjustment. They don't agree, so the class has no single
  answer.

## The functions

| Function | Returns |
| --- | --- |
| `adjustment_set(graph, X, Y)` | The canonical valid set, or `None`. |
| `minimal_adjustment_set(graph, X, Y)` | One minimal valid set, or `None`. |
| `is_valid_adjustment_set(graph, X, Y, Z)` | Whether `Z` is a valid adjustment set. |
| `frontdoor_set(graph, X, Y)` | A front-door set, or `None`. |
| `is_frontdoor_set(graph, X, Y, M)` | Whether `M` satisfies the front-door criterion. |
| `instruments(graph, X, Y)` | `(instrument, conditioning set)` pairs. |
| `is_instrument(graph, I, X, Y, given=...)` | Whether `I` is an instrument given `given`. |
| `ida(graph, X, Y)` | An `IDA` with one `Identification` per member of a class. |

Every set excludes latent nodes, since you can't condition on what you didn't
measure.

Adjustment and its checks work on DAGs, ADMGs, CPDAGs, MAGs and PAGs. On a class, a
set is valid only when it is valid in every member (the generalized adjustment
criterion of Perković et al., 2018). The front-door and instrument functions need a
single graph.

!!! caution "These searches are not exhaustive"
    `frontdoor_set` tries every observed variable on a directed path from treatment
    to outcome together, then single mediators. `instruments` tries the empty set and single
    variables as conditioning sets. When they return nothing, a larger set may still
    work: check it with `is_frontdoor_set` or `is_instrument`.

## Read an IDA result

`ida` enumerates the DAGs in a CPDAG and identifies the effect in each. `IDA` reports
`n_members`, `n_identifiable`, the `strategies` and `estimands` seen, and `agree`:
`True` when every member identifies and all share one formula. `members` holds each
member's `Identification`.

<details markdown="1"><summary>Does `ida` work on a PAG?</summary>

When the generalized adjustment criterion settles the question for the whole class,
`ida` returns that single answer. Otherwise it raises, because listing a PAG's member
DAGs isn't supported. A class with more than `max_members=1000` members also raises.

</details>

## Next steps

<div class="cards" markdown="1">

- [Estimate an average effect](estimate-average-effect.md): estimate with the set you found.
- [Average an effect over the graphs](evaluate-a-graph.md#average-an-effect-over-the-graphs): how likely a set is valid when the graph is uncertain.
- [Estimate a mediated effect](mediation.md): split the effect through `W` into direct and indirect parts.
- [Python reference](reference-identification.md): every function on this page.

</div>
