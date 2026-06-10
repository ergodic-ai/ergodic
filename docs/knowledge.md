# Domain knowledge

`DomainKnowledge` records what you know about a problem before you look at the data: which edges must or must not exist, the order variables come in, which are roots or sinks, and which pairs share a hidden confounder. Discovery uses it to prune and orient. Inference uses it to assert the conditions an effect needs. It serves all three pillars.

Like a graph, a knowledge object is immutable. Each method returns a new object, so you build one up by chaining.

## Stating what you know

```python
from ergodic import DomainKnowledge

dk = (
    DomainKnowledge()
    .require("Treatment", "Recovery")     # this edge must be there
    .forbid("Recovery", "Treatment")      # this one must not
    .root("Age")                          # Age has no causes in the model
    .confound("Treatment", "Outcome")     # they share a hidden common cause
)
```

The constraints you can state:

- `require(cause, effect)` and `forbid(cause, effect)` for single directed edges.
- `forbid_adjacency(a, b)` to rule out any edge between two variables.
- `order(before, after)` and `with_tiers(...)` for temporal order.
- `root(*names)` and `sink(*names)` for variables with no causes or no effects.
- `confound(a, b)` for a known hidden common cause.
- `forbid_between(group_a, group_b)` to ban every edge from one set into another.

## Tiers are the natural form

Most background knowledge is an order: demographics come before behavior, behavior comes before the outcome. State it as tiers and an edge from a later tier into an earlier one becomes forbidden.

```python
dk = DomainKnowledge().with_tiers([
    ["age", "sex"],        # tier 0
    ["smoking", "diet"],   # tier 1
    ["lung_capacity"],     # tier 2
])

dk.permits_edge("smoking", "age")   # False: tier 1 cannot cause tier 0
dk.permits_edge("age", "smoking")   # True
```

## Combining and checking

Two knowledge objects combine with `&`, taking the union of their constraints. More knowledge admits fewer graphs.

```python
expert = DomainKnowledge().forbid("Z", "Y")
temporal = DomainKnowledge().with_tiers([["X"], ["Y"], ["Z"]])
both = expert & temporal
```

Knowledge can contradict itself. `conflicts()` lists the problems and `validate()` raises on any:

```python
bad = DomainKnowledge().require("X", "Y").forbid("X", "Y")
bad.conflicts()   # ['edge X->Y is both required and forbidden']
bad.validate()    # raises InconsistentKnowledgeError
```

It catches a required-edge cycle, a precedence cycle, a required edge into a root, and a few more.

## Using it against a graph

Check whether a candidate graph obeys the knowledge:

```python
from ergodic import dag

g = dag(["X -> Y", "Y -> Z"])
dk = DomainKnowledge().require("X", "Y").root("X")

dk.is_consistent(g)    # True
dk.violations(g)       # [] when consistent, else a list of reasons
```

## Feeding it to discovery

The object offers a few views so different discovery styles can read it:

```python
from ergodic import mixed

dk = DomainKnowledge().with_tiers([["A"], ["B"], ["C"]])

dk.permits_edge("A", "B")        # True: an allowed orientation
dk.permits_edge("B", "A")        # False: B is later than A

skeleton = mixed(["A -- B", "B -- C"])
dk.apply_to(skeleton).is_dag()   # True: the tiers orient both edges

mask, labels = dk.edge_mask()    # boolean matrix of allowed directed edges
```

`permits_edge` is the orientation rule a constraint-based search reads. `apply_to` drops forbidden edges, adds required ones, and orients an undirected edge when only one direction stays allowed. `edge_mask` is the boolean adjacency a score-based or continuous-optimization search wants.

## Soft knowledge is not built yet

`energy(graph)` and `log_prior(graph)` are where probabilistic knowledge will attach: a penalty or a prior over graphs rather than a hard yes or no. They raise for now. That work comes with Bayesian discovery, and it will carry provenance too (where a belief came from, and how much to trust it). For today, the hard constraints above are the surface.
