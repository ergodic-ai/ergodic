# Compatibility with Welt

Check which ergodic and welt-client versions work together, and what a Welt graph
needs to become an ergodic graph. The link between them is a versioned graph format.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - For Welt itself, Python 3.11 to 3.13 and `pip install "ergodic[welt]"`.

## Versions that work together

| ergodic | welt-client | Graph format | Python for the Welt link |
| --- | --- | --- | --- |
| 0.2.x | 0.13 or later | `ergodic.graph/1` | 3.11, 3.12, 3.13 |
| 0.1.x | none | none | none |

Both packages pin the other from their own side:

- `pip install "ergodic[welt]"` adds `welt-client>=0.13` on Python 3.11 to 3.13.
  On Python 3.14 the extra adds nothing, because welt-client does not support it.
- `pip install "welt-client[ergodic]"` adds `ergodic>=0.2,<0.3`.

welt-client also brings numpy 2 and scikit-learn into your environment.

## Read the graph format

Welt converts a graph by building a versioned dict and handing it to
`MixedGraph.from_dict`. It checks `ergodic.GRAPH_FORMAT`, not the package version.

```python file="graph_format.py"
import ergodic
from ergodic import MixedGraph, dag

# 1. Print the graph format this ergodic reads and writes.
print(ergodic.GRAPH_FORMAT)

# 2. Write a DAG as a dict: the payload Welt builds for a declared DAG.
graph = dag(["Z -> X", "Z -> Y", "X -> Y"])
payload = graph.to_dict()
print(payload["format"], payload["kind"], sorted(payload["nodes"]))
print(payload["edges"][0])

# 3. Read it back.
print(MixedGraph.from_dict(payload) == graph)

# 4. A payload with another format is refused, not guessed.
try:
    MixedGraph.from_dict({**payload, "format": "ergodic.graph/2"})
except ValueError as error:
    print("ValueError:", error)
```

```output from="graph_format.py"
ergodic.graph/1
ergodic.graph/1 dag ['X', 'Y', 'Z']
{'u': 'X', 'v': 'Y', 'u_mark': 'tail', 'v_mark': 'arrow'}
True
ValueError: unsupported graph format 'ergodic.graph/2'; this version of ergodic reads 'ergodic.graph/1' (a dict without a 'format' field is read as 'ergodic.graph/1')
```

- **Step 1.** ergodic 0.2 reads and writes `ergodic.graph/1`.
- **Step 2.** The dict holds the format, the kind, the nodes and one entry per edge,
  with a mark at each end: `tail` or `arrow` for a DAG.
- **Step 3.** The round trip gives back an equal graph.
- **Step 4.** Any other format raises `ValueError`. A dict without a `format` field
  reads as `ergodic.graph/1`, so older payloads still load.

ergodic 0.1 has no `GRAPH_FORMAT`, so welt-client refuses to convert into it and
raises an `OptionalDependencyError` that names the extra to install.

## What converts

Only a graph that Welt declares a native DAG converts. `CausalResult.to_ergodic()`
in welt-client keeps every variable, unconnected ones included, and the native
edge directions. It never projects, thresholds or relabels a graph into a DAG.

| Welt model | Native graph | In ergodic 0.2.0 |
| --- | --- | --- |
| `arrow` | DAG | Converts. `ergodic.welt.discover` returns it as a `DiscoveryResult`. |
| `cdfm` | Directed, may hold cycles | Refused before paying by default. With an explicit `decoder=`, its scores are decoded locally into a DAG labelled decoded. |
| `avici` | Directed, may hold cycles | Refused before paying by default. With an explicit `decoder=`, its scores are decoded locally into a DAG labelled decoded. |

By default, for `cdfm` and `avici`, `ergodic.welt.discover` raises
`WeltGraphNotDAGError` before it uploads anything. A model the catalogue marks as
returning another graph type is refused the same way. Pass `decoder=` (for example
`decoder="greedy"`) to run the model and decode its edge scores with
`ergodic.decode`: the graph is then the decoder's choice, not the model's, and the
result says so. See [Decode AVICI and CDFM](welt.md#decode-avici-and-cdfm). A model with no known graph type runs, and if its
result is not a declared DAG, the error carries the paid result in `.result` and
its edge scores in `.scores`.

!!! caution "A converted graph is still a model's guess"
    Conversion checks the format, not the science. Read the graph before you
    estimate on it.

## Next steps

<div class="cards" markdown="1">

- [Discover in Welt, estimate in ergodic](welt.md): the whole path, from discovery to an estimate.
- [Graphs and edge marks](graphs-and-marks.md): what the marks in the dict mean.
- [Changelog](changelog.md): what changed in each release.

</div>
