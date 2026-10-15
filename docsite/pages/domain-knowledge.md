# Add domain knowledge

Tell the search what you already know, such as which variables come first in time or
which edges must exist. You'll orient the edges the data left open and get a graph you
can use for identification.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - Facts you trust more than the data: a time order, or edges that must or must not
      exist.

## Run the program

The data is the Asia chest-clinic network from [Discover a graph](discover-a-graph.md).
There, GES found 7 of 8 edges but left two undirected and missed `asia -> tub`. Here
you add two kinds of knowledge: the order in which things happen, and one edge you
know from medicine.

```python file="domain_knowledge.py"
from ergodic import DomainKnowledge, discover, identify_effect
from ergodic.datasets import load_asia
from ergodic.graph import dag

asia = load_asia(n=5_000, seed=0)
truth = dag(asia.target["edges"])

# 1. Search without knowledge. Two edges stay undirected.
plain = discover(asia.data, method="ges").graph
print(plain.kind.value, sum(edge.is_undirected for edge in plain.edges), "undirected")

# 2. Write down what you know: exposures, then diseases, then symptoms.
knowledge = (
    DomainKnowledge()
    .with_tiers([["asia", "smoke"], ["tub", "lung", "bronc"], ["either"], ["xray", "dysp"]])
    .require("asia", "tub")
)
print(knowledge)

# 3. Search again with the knowledge.
result = discover(asia.data, method="ges", knowledge=knowledge)
print(result.graph)
print("same edges as the truth:", result.graph.edges == truth.edges)

# 4. Re-face the graph as a DAG before you identify an effect.
graph = result.graph.as_kind("dag")
print(identify_effect(graph, "smoke", "dysp"))
```

```output from="domain_knowledge.py"
cpdag 2 undirected
DomainKnowledge(vars=8, required=1, forbidden=0, precedes=23, roots=0, sinks=0, confounded=0)
MixedGraph(kind=mixed, nodes=8, edges=[asia --> tub, bronc --> dysp, bronc <-- smoke, dysp <-- either, either <-- lung, either <-- tub, either --> xray, lung <-- smoke])
same edges as the truth: True
Identification(adjustment: P(dysp | do(smoke)) = sum_z P(dysp | smoke, z) P(z), with z = {asia, tub})
```

- **Step 1.** Without knowledge, `bronc --- smoke` and `lung --- smoke` have no
  direction.
- **Step 2.** `with_tiers` says no variable in a later tier causes one in an earlier
  tier. `require` adds an edge the data is too thin to show. The summary counts 23
  ordered pairs, because the tiers are read transitively.
- **Step 3.** Every edge is now directed and the graph equals the true network.
  Its kind is `mixed`, not `cpdag`.
- **Step 4.** `as_kind("dag")` checks the graph is a DAG and labels it one. Then
  `identify_effect` accepts it. `{asia, tub}` is the canonical adjustment set. Smoking
  has no causes here, so the empty set would also be valid.

## Why the result is `mixed`

A CPDAG is the set of graphs the data alone can't tell apart. Once knowledge orients
an edge the data left open, the graph stands for a smaller set that is no longer an
equivalence class, so it fails the CPDAG rules. The search labels it `mixed` and sets
`diagnostics["cpdag_valid"]` to `0.0`. FCI does the same with
`diagnostics["pag_valid"]`.

Identification refuses a `mixed` graph, and its error says what to do. When every edge
is directed, it tells you to call `graph.as_kind("dag")`, as in step 4. When some edges
stay undirected, add knowledge until they are directed, or take one member with
`ergodic.graph.consistent_extension(graph)`. The answer then holds for that member
only.

## What each constraint does

| Method | Effect |
| --- | --- |
| `require(a, b)` | The edge `a -> b` must appear. |
| `forbid(a, b)` | No edge `a -> b`. The reverse stays allowed. |
| `forbid_adjacency(a, b)` | No edge of any kind between `a` and `b`. |
| `forbid_between(group_a, group_b)` | No edge from any variable in `group_a` into `group_b`. |
| `with_tiers([...])`, `order(a, b)` | A later variable never causes an earlier one. |
| `root(*names)`, `sink(*names)` | No incoming edges, or no outgoing edges. |
| `confound(a, b)` | `a` and `b` share a hidden common cause. |
| `restrict_to(skeleton)` | Only pairs adjacent in `skeleton` may be adjacent. |

Each method returns a new object, so you can chain them. Combine two objects with `&`.

The methods read knowledge in different ways:

- **PC** never tests a pair banned in both directions, never removes a required edge,
  orients required and tier-implied edges, and raises if its final graph breaks the
  knowledge.
- **GES, BOSS and GRaSP** start from the required edges, never insert a forbidden
  one, and orient what the constraints imply. GES's optimality guarantee assumes an
  unconstrained search, so with knowledge it is a heuristic.
- **FCI and RFCI** read knowledge as ancestry. A tier puts an arrowhead at the later
  variable, `confound` adds `a <-> b`, and `require` adds `a --> b`. One-way forbids,
  roots and sinks are only checked at the end, because in a world with hidden causes
  they don't fix a single mark.

!!! caution "Only FCI and RFCI can represent `confound`"
    A CPDAG or a DAG has no bidirected edge. With `confound`, PC, GES, BOSS, GRaSP,
    HillClimb and A* raise an error, because their graph can't satisfy the knowledge.
    LiNGAM raises too. MMPC and glasso accept it and only keep the pair adjacent. Use
    `method="fci"`.

<details markdown="1"><summary>What if my knowledge contradicts itself or the data?</summary>

`knowledge.conflicts()` lists contradictions, such as a required edge that a tier
forbids, and `knowledge.validate()` raises on them.

When the knowledge is consistent but the data disagrees, the search usually imposes
the knowledge without a word. Forbid a true edge and a score search routes the
dependence through other edges; require a false one and it appears in the result. A
search raises an error only when its final graph can't satisfy the knowledge, and
then `knowledge.violations(graph)` lists what broke. To check your knowledge against
the data, run the search with and without it and compare the edges that change.

</details>

<details markdown="1"><summary>How do I narrow a search to a screened skeleton?</summary>

Run a cheap screen first, then search inside it:
`DomainKnowledge().restrict_to(discover(data, method="mmpc").graph)`. Pairs outside
the screen are never tested or scored. PC orients less inside a screen, because a
pruned pair records no separating set.

</details>

## Next steps

<div class="cards" markdown="1">

- [Check whether an effect is identifiable](identify-an-effect.md): what the re-faced graph lets you estimate.
- [Choose a method and test](choose-a-method.md): the assumptions behind each search.
- [Python reference](reference-knowledge.md): every `DomainKnowledge` method.

</div>
