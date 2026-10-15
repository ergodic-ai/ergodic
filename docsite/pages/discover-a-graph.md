# Discover a graph

Learn a causal graph from a table of observations, then read what the result can and
can't tell you. You'll run one search and check it against a network whose true graph
is known.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A table whose rows are independent observations. For time series, see
      [Discover from time series](time-series-discovery.md).

## Run the program

The Asia network (Lauritzen and Spiegelhalter, 1988) is a teaching model of a chest
clinic: a visit to Asia, smoking, tuberculosis, lung cancer, bronchitis, an x-ray and
shortness of breath (`dysp`). `either` is true when the patient has tuberculosis or
lung cancer. `load_asia` samples patients from the network's own probability tables,
so you know the answer in advance.

```python file="discover_graph.py"
from ergodic import discover
from ergodic.datasets import load_asia
from ergodic.graph import cpdag_of_dag, dag

# 1. Sample 5,000 patients from the Asia network.
asia = load_asia(n=5_000, seed=0)
print(asia.data.variables)

# 2. Learn the graph with greedy equivalence search (GES).
result = discover(asia.data, method="ges")
print(result.method, result.score, result.n)
print(result.graph)

# 3. Build the class of graphs the data could reveal: the truth's CPDAG.
truth = cpdag_of_dag(dag(asia.target["edges"]))
found = {edge.glyph() for edge in result.graph.edges}
expected = {edge.glyph() for edge in truth.edges}

# 4. Compare edge by edge, marks included.
print("right:", len(found & expected), "of", len(expected))
print("missed:", sorted(expected - found))
print("extra:", sorted(found - expected))
```

```output from="discover_graph.py"
('asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp')
ges bic_discrete 5000
MixedGraph(kind=cpdag, nodes=8, edges=[bronc --> dysp, bronc --- smoke, dysp <-- either, either <-- lung, either <-- tub, either --> xray, lung --- smoke])
right: 7 of 8
missed: ['asia --- tub']
extra: []
```

- **Step 2.** `discover` picked the discrete BIC score because every column is
  categorical. The result is a CPDAG: some edges have a direction, some don't.
- **Step 3.** No method can orient more than the data allows. `cpdag_of_dag` turns the
  true graph into the class of graphs that fit the same independences. That class is
  the fair target.
- **Step 4.** GES found 7 of the 8 edges with the right marks and added none. It missed
  `asia --- tub`: only 1 patient in 100 visited Asia, and a visit raises the chance of
  tuberculosis from 1% to 5%. In 5,000 rows that signal is too weak to see.

## Read the result

`discover` returns a `DiscoveryResult`. The graph is the headline; the other fields
say how it was found.

| Field | What it holds |
| --- | --- |
| `graph` | The learned graph. Its `kind` says how to read it: `cpdag`, `pag`, `dag` or `mixed`. |
| `method` | The method that ran, such as `"ges"`. |
| `test`, `score` | The independence test or the score it used. The other one is `None`; both are `None` for `"lingam"` and `"glasso"`. |
| `alpha` | The significance level, for methods that run tests. |
| `n` | The number of rows used. |
| `diagnostics` | Run counters: tests by size, score evaluations, validity flags. |
| `sepsets` | For test-based methods, the set that separated each removed pair. |

Each edge prints as a glyph. The marks at the two ends carry the meaning:

| Glyph | Meaning |
| --- | --- |
| `A --> B` | A causes B. |
| `A --- B` | In a CPDAG: adjacent, direction unknown. In a PAG: both cause the selection of the sample. |
| `A <-> B` | A hidden common cause, and neither causes the other. |
| `A o-> B` | B doesn't cause A. A may cause B, or a hidden cause may link them. |
| `A o-o B` | Adjacent, and the data can't say more. |

Circle marks come from FCI and RFCI. [Graphs and edge marks](graphs-and-marks.md)
covers each graph kind in full.

## Pick a method

`method=` takes a name. Every method except `"pcmci"` reads a `TabularData` and
refuses time series, whose rows aren't independent.

| `method=` | Returns | Use it when |
| --- | --- | --- |
| `"pc"` (default) | CPDAG | You assume no hidden common causes and want a fast, test-based search. |
| `"fci"` | PAG | Hidden common causes are possible. |
| `"rfci"` | PAG | As FCI, with many variables: faster, but it can keep extra edges. |
| `"ges"` | CPDAG | You prefer a score to many tests. |
| `"boss"`, `"grasp"` | CPDAG | You want a score-based search over variable orders. Prefer `"boss"` with a lot of domain knowledge. |
| `"astar"` | One DAG of the best class | You want a highest-scoring graph exactly. It refuses more than 20 variables by default. |
| `"hc"` | One DAG | You want a quick greedy baseline, with no optimality guarantee. |
| `"lingam"` | DAG | The data is continuous and linear, with non-Gaussian noise. |
| `"mmpc"`, `"glasso"` | Undirected graph | You want a cheap screen to narrow a later search. MMPC returns a skeleton; glasso returns the moral graph, which also joins parents of a common child. |
| `"pcmci"` | Window graph | Your data is a time series or a panel. |

[Choose a method and test](choose-a-method.md) lists the assumptions behind each one
and shows how to swap the independence test or the score.

The scores are score-equivalent: every DAG in an equivalence class gets the same
score. So A* and HillClimb return one member of a class, picked by tie-breaking, and
some of its directions are arbitrary. Before you identify an effect, turn the result
back into its class with `ergodic.graph.cpdag_of_dag(result.graph)`. To check a graph
you wrote or were given against the data, see
[Test a graph against data](test-a-graph.md).

!!! caution "A CPDAG stands for a set of graphs"
    An undirected edge such as `lung --- smoke` means the data can't tell the
    direction. The true graph has `smoke -> lung`, but the reverse fits the same
    independences. Add what you know as [domain knowledge](domain-knowledge.md) to
    orient it.

<details markdown="1"><summary>Why does another sample give another graph?</summary>

Each decision rests on a finite sample. With fewer rows, weak edges drop out and noise
can add a spurious one. [Check stability with the bootstrap](bootstrap-stability.md)
reruns the search on resamples and reports how often each edge appears.

</details>

<details markdown="1"><summary>What does `method="welt:..."` do?</summary>

`discover(data, method="welt:<model>")` sends the table to a hosted Welt model. It
needs `pip install "ergodic[welt]"`, runs one paid job, and takes no test or score.
Domain knowledge applies only when you also pass `decoder=`, because ergodic then
decodes the model's edge probabilities locally. See
[Discover in Welt, estimate in ergodic](welt.md) and [Decode a graph](decode-a-graph.md).

</details>

## Next steps

<div class="cards" markdown="1">

- [Add domain knowledge](domain-knowledge.md): orient edges the data leaves open.
- [Test a graph against data](test-a-graph.md): check the independences a candidate implies.
- [Check whether an effect is identifiable](identify-an-effect.md): use the graph to answer a causal question.
- [Python reference](reference-discovery.md): every method and its options.

</div>
