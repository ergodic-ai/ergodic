# Graphs and edge marks

Every causal graph in ergodic is one object, `MixedGraph`, whose edges carry a mark
at each end. Here you build a known network, question it, intervene on it, and see
how the marks change when the data cannot settle a direction.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.

## Read the marks

An edge has a mark at each end: a **tail**, an **arrowhead** or a **circle**. You
write edges as strings that look like the picture.

| You write | Marks | Reads as |
| --- | --- | --- |
| `A -> B` (or `A --> B`) | tail, arrowhead | A causes B directly |
| `A <-> B` | arrowhead, arrowhead | A and B share a hidden cause |
| `A -- B` (or `A --- B`) | tail, tail | Adjacent; the direction is not settled (CPDAG), or both cause a selection (MAG) |
| `A o-> B` | circle, arrowhead | B does not cause A; A may or may not cause B |
| `A o-o B` | circle, circle | Adjacent; no end is settled |

A circle means "this mark differs between graphs the data cannot tell apart".

## Run the program

The bundled burglar-alarm data (Pearl, 1988) records the network's edges in
`target["edges"]`. The program builds the network as a DAG and transforms it.

```python file="graph_marks.py"
from ergodic import dag
from ergodic.datasets import load_earthquake
from ergodic.graph import cpdag_of_dag, mag_of_dag, pag_of_mag


def show(title, graph):
    print(f"{title} ({graph.kind.value}):", ", ".join(sorted(e.glyph() for e in graph.edges)))


# 1. Build the burglar-alarm DAG from its published edges.
alarm = load_earthquake(n=100, seed=0)
graph = dag(alarm.target["edges"])
show("network", graph)

# 2. Ask separation questions.
print(graph.d_separated("Burglary", "Earthquake"))
print(graph.d_separated("Burglary", "Earthquake", "Alarm"))
print(graph.d_separated("JohnCalls", "MaryCalls", "Alarm"))

# 3. Intervene on the alarm: the edges into it are cut.
show("do(Alarm)", graph.do("Alarm"))

# 4. Read what data alone can tell apart: the equivalence class.
show("class", cpdag_of_dag(graph))
calls = graph.induced_subgraph({"Alarm", "JohnCalls", "MaryCalls"})
show("calls class", cpdag_of_dag(calls))

# 5. Hide the alarm, then read the class of what stays visible.
hidden = mag_of_dag(dag(alarm.target["edges"], latent=["Alarm"]))
show("alarm hidden", hidden)
show("its class", pag_of_mag(hidden))
```

```output from="graph_marks.py"
network (dag): Alarm --> JohnCalls, Alarm --> MaryCalls, Alarm <-- Burglary, Alarm <-- Earthquake
True
False
True
do(Alarm) (dag): Alarm --> JohnCalls, Alarm --> MaryCalls
class (cpdag): Alarm --> JohnCalls, Alarm --> MaryCalls, Alarm <-- Burglary, Alarm <-- Earthquake
calls class (cpdag): Alarm --- JohnCalls, Alarm --- MaryCalls
alarm hidden (mag): Burglary --> JohnCalls, Burglary --> MaryCalls, Earthquake --> JohnCalls, Earthquake --> MaryCalls, JohnCalls <-> MaryCalls
its class (pag): Burglary o-> JohnCalls, Burglary o-> MaryCalls, Earthquake o-> JohnCalls, Earthquake o-> MaryCalls, JohnCalls o-o MaryCalls
```

- **Step 1.** Four edges. ergodic prints an edge with the alphabetically smaller
  node first, so `Burglary -> Alarm` reads `Alarm <-- Burglary`. It is the same edge.
- **Step 2.** Burglary and Earthquake are independent, until you condition on the
  alarm they both cause. The two calls are independent given the alarm.
- **Step 3.** `do` removes every arrowhead into the target: setting the alarm cuts
  it off from its causes and keeps its effects.
- **Step 4.** The two causes collide at the alarm, so data fixes every direction
  here. Without the causes, the fork `JohnCalls <- Alarm -> MaryCalls` looks the
  same to data as a chain, so both edges stay undirected.
- **Step 5.** Hiding the alarm leaves `JohnCalls <-> MaryCalls`: a shared hidden
  cause. The PAG marks with a circle each end that data cannot settle.

## Know the kinds

Each builder checks its kind and raises `InvalidGraphError` when the edges break
its rules. A graph's `kind` says which reading applies.

| Kind | Builder | Edges allowed | Where it comes from |
| --- | --- | --- | --- |
| DAG | `dag` | `->`, no cycles | Your assumptions; `hc`, `exact` and `lingam` discovery; Welt's Arrow |
| ADMG | `admg` | `->` and `<->`, no directed cycles | Hidden confounders you name |
| CPDAG | `cpdag` | `->` and `--` | `pc`, `ges`, `boss` and `grasp` discovery |
| MAG | `mixed(..., kind=GraphKind.MAG)` | `->`, `<->`, `--` under ancestral rules | Projecting out latent and selection nodes |
| PAG | `pag` | Any marks, circles included | `fci` and `rfci` discovery |

`mixed` builds a graph with any marks and no kind rules; `mmpc` discovery returns
one, an undirected skeleton. `as_kind("dag")` re-tags a
graph after checking it.

!!! note "One edge per pair"
    A graph holds at most one edge between two nodes. Two different edges for the
    same pair raise a `ValueError`. To say "X causes Y and they also share a hidden
    cause", name the cause: `dag(["L -> X", "L -> Y", "X -> Y"], latent=["L"])`.

## Work with graphs as values

Graphs are immutable. `add_edge`, `remove_edge`, `orient`, `set_mark` and `do`
return a new graph and leave the original unchanged. Two graphs with the same
nodes, edges and kind are equal and hash the same.

`to_dict` writes a versioned dict (`"format": "ergodic.graph/1"`) and `from_dict`
reads it back; [Compatibility with Welt](compatibility.md) shows the layout.
`to_networkx`, `from_networkx` and `to_adjacency` connect to other tools.

<details markdown="1"><summary>How does separation read a circle?</summary>

`d_separated` takes only a DAG and raises `ValueError` on any other kind. Use
`m_separated` on the others. With circle marks, a path connects only when every
node on it has a settled role. A node is a collider when both path marks at it are
arrowheads. It is a definite non-collider when a path edge has a tail at it, or
when both path marks at it are circles and its two path neighbours are not
adjacent. On a complete PAG the answer is the same in every member graph.

A CPDAG reads its `--` edges as circle pairs, so it answers for the class too. A
`mixed` graph walks `--` edges literally as two tails. Circle marks get the
settled-role reading in every kind.

</details>

## Next steps

<div class="cards" markdown="1">

- [Assumptions behind each step](assumptions.md): what each kind of graph asks you to believe.
- [Discover a graph](discover-a-graph.md): learn a CPDAG or PAG from data.
- [Check whether an effect is identifiable](identify-an-effect.md): ask a graph for an estimand.

</div>
