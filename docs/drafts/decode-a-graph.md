# Decode a DAG from edge probabilities

Turn a matrix of edge probabilities into a DAG with a named decoder, and see which
edges it dropped and why. You'll compare decoders, draw candidate DAGs with a seeded
sampler, and constrain the result with domain knowledge.

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
  to all the others; every candidate is in `candidates`.
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

`decode` returns a `DecodeResult`. `graph` is a DAG over every variable, `method` is
`"decode:<id>"`, `candidates` holds every decoded DAG and `probabilities` the input.
`diagnostics` records:

| Key | What it holds |
| --- | --- |
| `decoded`, `native` | Always `True` and `False`. |
| `decoder`, `parameters` | The versioned id and its settings, such as `threshold`. |
| `seed`, `samples` | The sampler's seed and draw count (`None` and 1 otherwise). |
| `rejected_cycles` | Per candidate, how many nominated edges were skipped for closing a cycle. |
| `objective` | The total log score of `graph`. |
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

`decode(..., knowledge=...)` applies hard constraints:

- **Forbidden edges, tiers, roots, sinks and `restrict_to`** remove directed edges before
  decoding.
- **Order** (`order`, tiers) holds on ancestry: no later variable becomes an ancestor of
  an earlier one.
- **Required edges** go in first, whatever their probability. A required set with a
  cycle raises.
- **`confound`** raises: a DAG can't hold a latent confounder.

Every variable the knowledge names must be in the probabilities. The decoded graph is
checked against the knowledge before it's returned.

## Decode a Welt model's scores

AVICI and CDFM return directed graphs that may hold cycles, so
`ergodic.welt.discover` refuses them before anything is paid for. Pass `decoder=` to
opt in: `discover(data, "cdfm", decoder="greedy")` runs the model once, reads its
scores as independent edge probabilities and decodes them. Greedy then uses the model's
own threshold (0.5 for AVICI, CDFM's adaptive one). The result keeps the scores in
`diagnostics["edge_probabilities"]`, so you can try another decoder locally without a
new run. See [Run Welt models](welt.md).

## Next steps

<div class="cards" markdown="1">

- [Work with edge probabilities](edge-probabilities.md): build and read the input.
- [Identify an effect](identify-an-effect.md): use the decoded DAG.
- [Python reference](reference-discovery.md): `decode`, `Decoder` and `DecodeResult`.

</div>
