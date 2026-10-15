# Decode a DAG from edge probabilities

Turn a matrix of edge probabilities into a DAG with a named decoder, and see which
edges it dropped and why. You'll compare decoders, draw candidate DAGs with a seeded
sampler, constrain the result with domain knowledge, and weigh what you only believe
against what the model says.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - An `EdgeProbabilities` object. [Work with edge probabilities](edge-probabilities.md)
      shows how to build one.

## Run the program

The matrix is the one from [Work with edge probabilities](edge-probabilities.md): four
edges pass 0.5, and three of them form a cycle.

```python file="decode_a_graph.py"
import numpy as np

from ergodic import DomainKnowledge, EdgeProbabilities, decode

# 1. Independent edge probabilities; three edges above 0.5 form a cycle.
names = ("rain", "sprinkler", "wet", "slippery")
values = np.array(
    [
        [0.00, 0.45, 0.91, 0.06],
        [0.40, 0.00, 0.83, 0.05],
        [0.04, 0.07, 0.00, 0.88],
        [0.58, 0.02, 0.10, 0.00],
    ]
)
probs = EdgeProbabilities(names, values, kind="independent")

# 2. The default decoder: greedy pair-MAP.
result = decode(probs)
print(result)
print(result.graph)
print(result.diagnostics["rejected_cycles"], result.diagnostics["note"])

# 3. A lower threshold nominates more edges.
print(decode(probs, threshold=0.3).graph)

# 4. Exact MAP and the expected-count decoder.
print(decode(probs, "exact_map").graph)
print(decode(probs, "expected_count").graph)

# 5. Candidate DAGs from a seeded sampler, and their consensus.
sampled = decode(probs, "pair_sampler", samples=200, seed=0)
print(sampled)
print(sampled.graph)

# 6. Knowledge: nothing causes the sprinkler.
knowledge = DomainKnowledge().root("sprinkler")
print(decode(probs, threshold=0.3, knowledge=knowledge).graph)
```

```output from="decode_a_graph.py"
DecodeResult(decode:greedy_v1: decoded dag, 4 nodes, 3 edges)
MixedGraph(kind=dag, nodes=4, edges=[rain --> wet, slippery <-- wet, sprinkler --> wet])
(1,) not a graph posterior; orientation not certified
MixedGraph(kind=dag, nodes=4, edges=[rain --> sprinkler, rain --> wet, slippery <-- wet, sprinkler --> wet])
MixedGraph(kind=dag, nodes=4, edges=[rain --> wet, slippery <-- wet, sprinkler --> wet])
MixedGraph(kind=dag, nodes=4, edges=[rain --> sprinkler, rain --> wet, slippery <-- wet, sprinkler --> wet])
DecodeResult(decode:pair_sampler_v1: decoded dag, 4 nodes, 3 edges, consensus of 200 candidates)
MixedGraph(kind=dag, nodes=4, edges=[rain --> wet, slippery <-- wet, sprinkler --> wet])
MixedGraph(kind=dag, nodes=4, edges=[rain <-- sprinkler, rain --> wet, slippery <-- wet, sprinkler --> wet])
```

- **Step 2.** Greedy nominates each pair's likelier direction when it beats "no edge",
  which on independent marginals means above 0.5. It adds them strongest first and
  skips any that would close a cycle: `slippery --> rain`, the weakest of the three,
  is the one rejected cycle. The result is a `DecodeResult`, a `DiscoveryResult`
  labelled as decoded.
- **Step 3.** With `threshold=0.3` an edge is nominated when its own value is above 0.3,
  so `rain --> sprinkler` (0.45) joins. The cut is strict, as AVICI and CDFM apply
  theirs.
- **Step 4.** Exact MAP finds the DAG with the best total score and agrees with greedy
  here. The expected-count decoder keeps as many edges as the probabilities expect,
  rounded (four), so it adds `rain --> sprinkler` too.
- **Step 5.** The sampler draws 200 DAGs. `graph` is their medoid, the candidate closest
  to all the others (the earliest one on an exact tie); every candidate is in
  `candidates`.
- **Step 6.** `root("sprinkler")` forbids every edge into `sprinkler`, so the pair
  flips to `sprinkler --> rain`, its other direction at 0.40.

## Choose a decoder

Every decoder reads the pair-state view (see
[Work with edge probabilities](edge-probabilities.md)) and scores an edge by
`w(i -> j) = log P(i -> j) - log P(no edge)`.

| Decoder | What it does | Settings |
| --- | --- | --- |
| `"greedy"` (`greedy_v1`, default) | Each pair's better direction when `w > 0` (or its value is above `threshold`), strongest first, skipping cycles. | `threshold` |
| `"exact_map"` (`exact_map_v1`) | The DAG with the best total score, found exactly by A* search over orderings. | `max_nodes` |
| `"expected_count"` (`expected_count_v1`) | The best `K` edges compatible with the exact ordering, `K` the rounded expected edge count. | `max_nodes` |
| `"soft_f1"` (`soft_f1_v1`) | The same, with `K` from a label-free F1 estimate. | `max_nodes` |
| `"pair_sampler"` (`pair_sampler_v1`) | Draws each pair's state, adds edges in random order, drops those that close a cycle. | `samples`, `seed` |
| `"perturb_and_map"` (`perturb_and_map_v1`) | Exact MAP under randomly perturbed pair scores, once per sample. | `samples`, `seed`, `max_nodes` |

Greedy is the default. In the research it came from it matched exact MAP on 238 of 240
graphs. It isn't best everywhere: on one physical benchmark the expected-count decoder
scored better. The decoders that search orderings exactly refuse more than `max_nodes`
variables (20 by default), because that search can grow exponentially.

## Read the result

`decode` returns a `DecodeResult`. `graph` is a DAG over every variable (an ADMG when
your knowledge names hidden confounders), `method` is
`"decode:<id>"`, `candidates` holds every decoded DAG and `probabilities` the input.
`diagnostics` records:

| Key | What it holds |
| --- | --- |
| `decoded`, `native` | Always `True` and `False`. |
| `decoder`, `parameters` | The versioned id and its settings, such as `threshold`. |
| `threshold_source` | `"decoder"` when you passed a threshold, `None` otherwise. Through Welt it can be `"welt_native"`. |
| `seed`, `samples` | The sampler's seed and draw count (`None` and 1 otherwise). |
| `rejected_cycles` | Per candidate, how many nominated edges were skipped for closing a directed cycle. |
| `order_skipped` | Per candidate, how many were skipped only because they'd break the order knowledge. |
| `objective` | The total log score of `graph`'s directed edges on the input. |
| `knowledge_objective` | The same score on the pair states the soft knowledge reweighted. |
| `graph_kind`, `confounded_pairs` | `"dag"`, or `"admg"` with the `confound` pairs. |
| `knowledge_audit`, `knowledge_cost`, `knowledge_conflicts` | What the model makes of your knowledge; see [Decode with what you know](#decode-with-what-you-know). |
| `edge_budget` | `K`, for the cardinality decoders. |
| `source_kind`, `source_provenance` | Where the probabilities came from. |
| `note` | `"not a graph posterior; orientation not certified"`. |

!!! caution "A decoded graph is your choice, not the model's claim"
    A model that scores edges one at a time doesn't promise a DAG. The decoder picks
    one, and a different decoder, threshold or seed can pick another. Sampler
    candidates are not draws from a posterior over graphs, and nothing certifies the
    orientations.

Decoding is deterministic: ties go to the earlier variable, the same seed gives the
same candidates, and neither the input nor numpy's global random state changes.

## Use domain knowledge

`decode(..., knowledge=...)` applies [hard constraints](domain-knowledge.md):

- **Forbidden edges, tiers, roots, sinks and `restrict_to`** remove directed edges before
  decoding.
- **Order** (`order`, tiers) is enforced as one time order: the decoded DAG has a
  single topological order that agrees with every order statement and every edge at
  once. For tiers that's the same as "no later variable is an ancestor of an earlier
  one". For separate `order()` statements it's stricter than
  `DomainKnowledge.violations`, which checks each statement on its own: with
  `order("a", "b")` and `order("c", "d")`, the edges `b --> c` and `d --> a` pass
  `violations` together, but no single order fits them, so a decoder keeps only one.
- **Required edges** go in first, whatever their probability. A required set with a
  cycle raises.
- **`confound`** adds a bidirected edge after decoding, so the result is an ADMG. See
  [Decode with what you know](#decode-with-what-you-know).

Every variable the knowledge names must be in the probabilities. The decoded graph is
checked against the knowledge before it's returned.

## Decode with what you know

Hard constraints say "this is so". Often you only believe something: you're fairly
sure, not certain. Give any constraint a `confidence=` between 0.5 and 1 and the
decoder weighs it against the model instead of enforcing it. Every decode with
knowledge also tells you where your knowledge and the model disagree.

```python file="decode_with_knowledge.py"
import numpy as np

from ergodic import DomainKnowledge, EdgeProbabilities, decode, identify_effect

names = ("rain", "sprinkler", "wet", "slippery")
values = np.array(
    [
        [0.00, 0.45, 0.91, 0.06],
        [0.40, 0.00, 0.83, 0.05],
        [0.04, 0.07, 0.00, 0.88],
        [0.58, 0.02, 0.10, 0.00],
    ]
)
probs = EdgeProbabilities(names, values, kind="independent")

# 1. A soft statement: you're fairly sure nothing causes the sprinkler.
print(decode(probs, threshold=0.3).graph)
soft = DomainKnowledge().root("sprinkler", confidence=0.7)
print(decode(probs, threshold=0.3, knowledge=soft).graph)

# 2. A statement the model disagrees with: rain doesn't wet the ground.
doubtful = soft.forbid("rain", "wet", confidence=0.8)
result = decode(probs, threshold=0.3, knowledge=doubtful)
print(result.graph)

# 3. The audit: what the model makes of each statement.
for entry in result.diagnostics["knowledge_audit"]:
    print(
        f"{entry['statement']:<20} confidence {entry['confidence']}, "
        f"model {entry['probability']:.2f}, holds {entry['holds']}, "
        f"without knowledge {entry['holds_without_knowledge']}"
    )
print([entry["statement"] for entry in result.diagnostics["knowledge_conflicts"]])
print(round(result.diagnostics["knowledge_cost"], 3))

# 4. A hidden common cause of the sprinkler and slipperiness: an ADMG.
confounded = decode(probs, knowledge=DomainKnowledge().confound("sprinkler", "slippery"))
print(confounded)
print(confounded.graph)
print(identify_effect(confounded.graph, "sprinkler", "slippery"))
```

```output from="decode_with_knowledge.py"
MixedGraph(kind=dag, nodes=4, edges=[rain --> sprinkler, rain --> wet, slippery <-- wet, sprinkler --> wet])
MixedGraph(kind=dag, nodes=4, edges=[rain <-- sprinkler, rain --> wet, slippery <-- wet, sprinkler --> wet])
MixedGraph(kind=dag, nodes=4, edges=[rain <-- sprinkler, rain --> wet, slippery <-- wet, sprinkler --> wet])
no rain -> wet       confidence 0.8, model 0.09, holds False, without knowledge False
sprinkler is a root  confidence 0.7, model 0.65, holds True, without knowledge False
['no rain -> wet']
0.205
DecodeResult(decode:greedy_v1: decoded admg, 4 nodes, 4 edges)
MixedGraph(kind=admg, nodes=4, edges=[rain --> wet, slippery <-> sprinkler, slippery <-- wet, sprinkler --> wet])
Identification(frontdoor: P(slippery | do(sprinkler)) = sum_m P(m | sprinkler) sum_x' P(slippery | m, x') P(x'), with m = {wet})
```

- **Step 1.** At 0.7, `root("sprinkler")` is enough to flip `rain --> sprinkler`
  (0.45) to `sprinkler --> rain` (0.40): the model barely prefers one direction, so a
  moderate belief tips it.
- **Step 2.** The model gives `rain --> wet` 0.91. A 0.8 belief that it isn't there
  doesn't outweigh that, so the edge stays.
- **Step 3.** The audit has one entry per statement, as you wrote it. `model` is the
  model's probability that the statement holds; `no rain -> wet` gets 0.09. Anything
  below 0.1 is a strong conflict: it's listed in `knowledge_conflicts` and `decode`
  warns once. `knowledge_cost` is how much log score the knowledge cost against a
  decode without it.
- **Step 4.** `confound` doesn't change the directed edges. It adds
  `sprinkler <-> slippery`, so the result is an ADMG, and identification reads it as
  one: the effect is still identified, by the front door through `wet`.

### How a soft statement weighs

Every statement is about pairs of variables, and each pair has three states: no
edge, one direction, the other. A statement you hold with confidence `c` multiplies
the chance of each state by `c` where the state agrees with it and by `1 - c` where it
doesn't. So against "no edge", `require("a", "b", confidence=c)` multiplies the odds of
`a --> b` by `c / (1 - c)`, `forbid` divides them by it, and `root("x", confidence=c)`
divides the odds of every edge into `x`. Statements multiply. The decoders then run
unchanged on the reweighted states, and `"exact_map"` finds the best graph for the
combined score exactly. `knowledge.log_odds(variables)` shows the weights and
`knowledge.log_prior(graph)` scores a graph under them.

- **0.5 changes nothing.** Below 0.5 is refused: state the opposite instead.
- **Close to 1 acts like the hard constraint** whenever the hard result is reachable.
- **A soft order isn't transitive.** `order("a", "b", confidence=c)` weighs only the
  direct edge `b --> a`; hard order is read through chains.
- **`confound` is hard only.** A hidden cause isn't an edge state, so the model has
  no number for it, and its audit entry has no probability.
- **How far a belief outweighs the model depends on the model's calibration.** The
  scores of AVICI and CDFM aren't calibrated, so a 0.8 belief isn't worth the same
  against every model.

### Read the audit

| Key | What it holds |
| --- | --- |
| `knowledge_audit` | One entry per statement: `statement`, `kind`, `confidence` (`None` for hard), `probability`, `holds`, `holds_without_knowledge`. |
| `knowledge_conflicts` | The entries the model gives less than 0.1. |
| `knowledge_cost` | The knowledge-free decode's score minus this one's. Never negative for `"exact_map"`; the other decoders can come out negative. |

The model's probability treats each pair on its own, so a statement about many pairs
(tiers, a root, `restrict_to`) can get a low number even when no single pair is
surprising. `holds_without_knowledge` comes from a second decode with the same
settings and seed but no knowledge. Order and tier statements are read pair by pair
here, as the decoders weigh them.

### Confounders give an ADMG

With `confound("a", "b")` the directed part is decoded as before and `a <-> b` is
added, to `graph` and to every candidate. If the decoder also put a directed edge on
that pair, a graph can't hold both edges on one pair, so the confounder becomes an
explicit latent node, `L__a__b`, with an edge into each end. It's in `graph.latent`,
and `diagnostics["latent_confounders"]` maps it to its pair. Identification treats
both forms the same way.

## Decode a Welt model's scores

AVICI and CDFM return directed graphs that may hold cycles, so
`ergodic.welt.discover` refuses them before anything is paid for. Pass `decoder=` to
opt in: `discover(data, "cdfm", decoder="greedy")` runs the model once, reads its
scores as independent edge probabilities and decodes them. Greedy then uses the model's
own threshold (0.5 for AVICI, CDFM's adaptive one). The result keeps the scores in
`diagnostics["edge_probabilities"]`, so you can try another decoder locally without a
new run. The exact decoders' `max_nodes` cap (20 by default) is checked against your
data's variables before anything is uploaded. See
[Decode AVICI and CDFM](welt.md#decode-avici-and-cdfm).

Pass `knowledge=` without `decoder=` and ergodic decodes with greedy at the model's
own threshold, as `decoder="greedy"` would.

Arrow returns a DAG of its own, so ergodic uses that by default. Its scores are decoded
only if you ask, with `decoder=`, and are then read as independent marginals, like
AVICI's and CDFM's. That's a modelling choice: Welt doesn't state that Arrow's scores
are marginals.

## Next steps

<div class="cards" markdown="1">

- [Work with edge probabilities](edge-probabilities.md): build and read the input.
- [Add domain knowledge](domain-knowledge.md): every kind of statement.
- [Identify an effect](identify-an-effect.md): use the decoded DAG.
- [Python reference](reference-discovery.md): `decode`, `Decoder` and `DecodeResult`.

</div>
