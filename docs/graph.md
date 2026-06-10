# Causal graphs

A causal graph in ergodic is a mixed graph: every edge carries a mark at each end. That one idea lets a single object represent the whole family you meet in causal work, from a plain DAG to the PAG an FCI search returns.

Graphs are immutable. Every edit returns a new graph and leaves the original alone, which keeps interventions and edge orientation easy to follow.

## Building a graph

Edges read as glyph strings, so the code looks like the picture:

```python
from ergodic import dag, admg, cpdag, pag

g = dag(["Smoking -> Tar", "Tar -> Cancer", "Smoking -> Cancer"])
confounded = admg(["X -> Y", "Y <-> Z"])      # <-> is a hidden common cause
partial = cpdag(["A -- B", "B -> C"])          # -- is undirected
discovered = pag(["X o-> Y", "Y o-o Z"])       # o is an unknown mark
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
- `CPDAG`: directed and undirected. What constraint-based discovery returns.
- `MAG`: an ADMG with no almost-directed cycle.
- `PAG`: any marks, including circles. What FCI returns.

How deep do these checks go? `DAG` and `ADMG` are checked in full. The others are looser than their names suggest: `CPDAG` checks marks and acyclicity but not the Meek rules or that the graph stands for a real equivalence class, `MAG` rejects the undirected selection edges a MAG may legally carry and never tests maximality, and `PAG` checks acyclicity only. The full checks arrive with the discovery phase.

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

The conditioning argument takes a single name or any iterable of names. Circle marks (PAGs) need separate reasoning that is not built yet, so a graph with a circle raises a clear error.

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

## Not built yet

Adjustment sets and identification queries are built now; see [Identification](identification.md). Still planned: equivalence-class operations (skeleton to CPDAG, Markov equivalence), PAG separation, and latent projection. The implementation journal in `plans/` has the order.
