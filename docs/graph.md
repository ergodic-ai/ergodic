# Causal graphs

A causal graph in ergodic is a mixed graph: every edge carries a mark at each end. That one idea lets a single object represent the whole family you meet in causal work, from a plain DAG to the PAG an FCI search returns.

Graphs are immutable. Every edit returns a new graph and leaves the original alone, which keeps interventions and edge orientation easy to follow.

## Building a graph

Edges read as glyph strings, so the code looks like the picture:

```python
from ergodic import dag, admg, cpdag, pag

g = dag(["Smoking -> Tar", "Tar -> Cancer", "Smoking -> Cancer"])
confounded = admg(["X -> Y", "Y <-> Z"])      # <-> is a hidden common cause
partial = cpdag(["A -- B", "A -- C", "B -> D", "C -> D"])   # -- is undirected
discovered = pag(["X o-> Y", "Z o-> Y"])       # o is an unknown mark
```

The mark mini-language:

| Glyph | Meaning |
| --- | --- |
| `A -> B` or `A --> B` | A is a direct cause of B |
| `A <- B` | B is a direct cause of A |
| `A <-> B` | A and B share a hidden common cause |
| `A -- B` or `A --- B` | adjacent, orientation unspecified |
| `A o-> B` | arrow at B, unknown at A |
| `A o-o B` | unknown at both ends |

You can also pass `(u, v)` tuples (read as `u -> v`) or build an `Edge` directly. `mixed(...)` is the general builder when you want to set `latent`, `selection`, or a `kind` by hand.

### Edges, equality, and display

A graph holds one edge per pair of nodes. Two different specs for the same pair raise a `ValueError` naming both, so `dag(["X -> Y", "Y -> X"])` fails loudly instead of keeping one silently. `add_edge` follows the same rule: adding the edge already there is a no-op, and adding a different one raises (remove it first, or use `orient` to redirect). Graphs are values: two with the same nodes, edges, and kind are equal and hash the same, so you can use them as dict keys or set members.

Edges store the smaller node name first, so a printed edge can read reversed. `Smoking -> Cancer` shows as `Cancer <-- Smoking` when `Cancer` sorts first. It is the same edge, and `directed_pair()` always reports `('Smoking', 'Cancer')`.

## Kinds and validation

Each builder checks the graph against its kind and raises `InvalidGraphError` if it does not fit.

```python
dag(["X <-> Y"])              # raises: a DAG has no bidirected edges
dag(["X -> Y", "Y -> Z", "Z -> X"])  # raises: that is a cycle
admg(["X -> Y", "Y <-> Z"])   # fine
```

The kinds and what they allow:

- `DAG`: directed edges, acyclic.
- `ADMG`: directed and bidirected, acyclic. The object for latent confounding.
- `CPDAG`: directed and undirected. What PC and GES return.
- `MAG`: an ADMG with no almost-directed cycle.
- `PAG`: circles where the equivalence class disagrees. What FCI returns.

How deep do these checks go? `DAG`, `ADMG`, `CPDAG`, and `PAG` are checked in full. A CPDAG must stand for a real equivalence class: some DAG has exactly its skeleton and v-structures, and the graph equals that DAG's CPDAG, so every compelled edge is directed and every reversible one is left undirected. A PAG gets the same treatment through its own round trip: one member MAG must extract (`mag_of_pag`), and the canonical PAG of that member (`pag_of_mag`) must be exactly the graph, so every invariant mark is oriented and every varying mark is a circle. `MAG` keeps two honest gaps: undirected selection edges are rejected (no selection bias anywhere yet) and maximality is not tested. Use `mixed(...)` when you want marks without these guarantees.

Re-tag a graph with `as_kind`, which validates before it returns:

```python
m = mixed(["X <-> Y"])
m.as_kind("admg")   # or GraphKind.ADMG
```

## Asking questions

Family lookups read straight off the graph:

```python
g.parents("Cancer")        # frozenset({'Smoking', 'Tar'})
g.children("Smoking")      # frozenset({'Tar', 'Cancer'})
g.ancestors("Cancer")      # frozenset({'Smoking', 'Tar'})
g.markov_blanket("Tar")    # DAG only
```

### d-separation and m-separation

`d_separated` answers whether two variables are independent given a conditioning set, under the DAG. `m_separated` does the same for mixed graphs with bidirected edges.

```python
g.d_separated("Smoking", "Cancer", "Tar")   # False: the direct edge remains

chain = dag(["X -> Y", "Y -> Z"])
chain.d_separated("X", "Z")        # False
chain.d_separated("X", "Z", "Y")   # True: conditioning blocks the chain

collider = dag(["X -> Y", "Z -> Y"])
collider.d_separated("X", "Z")        # True: the collider is closed
collider.d_separated("X", "Z", "Y")   # False: conditioning opens it
```

The conditioning argument takes a single name or any iterable of names.

Circle marks get Zhang's definite-status reading: a path connects only when every vertex on it has definite status (a collider by its arrowheads, or a definite non-collider by a tail or by unshielded circles). On a valid complete PAG the answer is a class invariant, the same in every member MAG, so `m_separated` on the PAG that FCI returns means exactly what it means on a known graph:

```python
from ergodic import pag

p = pag(["X o-> Y", "Z o-> Y"])
p.m_separated("X", "Z")        # True: the collider is invariant across the class
p.m_separated("X", "Z", "Y")   # False
```

A CPDAG gets the same class reading. Its undirected edge says "some member orients it each way", which is exactly what a circle pair says in a PAG, so separation recodes undirected edges as circles and walks definite-status paths. The difference is not cosmetic: on `V0 --> V2 --- V4 <-- V1`, a literal tail-tail reading finds an open path from `V0` to `V1`, yet whichever way a member orients the middle edge, one end of it becomes a collider and blocks, and every member agrees they are separated. The CPDAG face answers for the class; spell the marks with `mixed(...)` when you want them walked literally.

## Intervening and slicing

`do` is graph surgery for an intervention: it cuts every arrowhead into the target, which removes incoming causes and hidden confounding while keeping the target's effects.

```python
g.do("Tar")                    # the mutilated graph after setting Tar
g.induced_subgraph({"Smoking", "Tar"})
g.ancestral_subgraph({"Cancer"})
```

## Talking to other tools

```python
g.to_networkx()                # a networkx MultiDiGraph carrying the marks
MixedGraph.from_networkx(nx_graph)
g.to_adjacency()               # (directed 0/1 matrix, node order)
data = g.to_dict()             # JSON-ready
MixedGraph.from_dict(data)
```

networkx is used only here, at the edge of the package. The graph algorithms are native.

## Moving between the faces

The equivalence-class machinery is public, because discovery is built on it and tests want it too:

```python
from ergodic.graph import cpdag_of_dag, consistent_extension, mag_of_dag, pag_of_mag, mag_of_pag

cpdag_of_dag(g)            # the CPDAG of a DAG's Markov equivalence class
consistent_extension(c)    # one DAG member of a PDAG, or None (Dor-Tarsi)

projected = mag_of_dag(dag(["L -> X", "L -> Y"], latent=["L"]))   # X <-> Y
canonical = pag_of_mag(projected)                                  # X o-o Y
member = mag_of_pag(canonical)                                     # one MAG back out
```

`mag_of_dag` is the latent projection: the MAG over the observed margin of a DAG or ADMG, preserving every m-separation among the observed. `pag_of_mag` computes the invariant marks of the Markov equivalence class (the canonical PAG, through the FCI rule closure), and `mag_of_pag` extracts one member. `meek_closure` and `fci_closure` (the orientation rule engines) are public too.

When one member is not enough, enumerate the class. `dags_of_cpdag` lists every consistent extension of a CPDAG and `mags_of_pag` every MAG of a PAG, deterministically, with a `max_members` cap that raises rather than truncating. These power the all-members tests behind class identification and the per-member (IDA-style) workflow when a class query comes back undecided.

Three reachability-and-certificate helpers ride along. `possible_descendants` and `possible_ancestors` follow possibly directed paths (no arrowhead back toward the start), so they answer "could X cause Y in some member?" on any face; on a DAG they are plain ancestry. `visible_edges` returns the directed MAG or PAG edges that Zhang's criterion certifies free of hidden confounding in every represented world, the certificate that class-level adjustment leans on:

```python
from ergodic.graph import dags_of_cpdag, mags_of_pag, possible_descendants, visible_edges

dags_of_cpdag(cpdag_of_dag(g))     # every DAG in the class
mags_of_pag(canonical)             # every MAG in the class
possible_descendants(canonical, "X")
visible_edges(canonical)           # {(tail, head), ...}
```

## Not built yet

Adjustment sets and identification queries are built now, including on CPDAGs, MAGs, and PAGs; see [Identification](identification.md). Discovery returns these faces; see [Discovery](discovery.md). Still missing: selection bias end to end (selection nodes in the projection, undirected PAG edges, rules R5 to R7) and possible m-separation (the existential cousin of the definite reading). The implementation journal in `plans/` has the order.
