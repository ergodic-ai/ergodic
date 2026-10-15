<!--
Draft for docsite/pages/welt.md (API-148). Insert the section below before "## Limits",
and replace the Limits bullet that begins "`cdfm` and `avici` are refused" with:

- `cdfm` and `avici` return directed graphs that may hold cycles, so `discover`
  refuses them before it uploads anything unless you pass `decoder=` (see
  [Decode AVICI and CDFM](#decode-avici-and-cdfm)).

and the first Limits bullet's "and takes no domain knowledge" with "and takes no domain
knowledge (with `decoder=`, knowledge constrains the local decoding only)".
-->

## Decode AVICI and CDFM

AVICI and CDFM score every directed edge on its own and return a directed graph that
may hold cycles and two-way pairs. ergodic never turns that into a DAG silently, so
by default `discover` refuses them before anything is paid for. Pass `decoder=` to
decode their scores into a DAG yourself:

```python
from ergodic.welt import discover

result = discover(data, "cdfm", decoder="greedy")
```

- **One paid run.** The model runs once. Its scores become
  `result.diagnostics["edge_probabilities"]`, an `EdgeProbabilities` of kind
  `"independent"` with the Welt ids, versions and native decoder in its `provenance`.
- **A labelled, decoded graph.** `result.graph` is the DAG the decoder chose, not the
  model's own graph. `result.diagnostics["decoded"]` is `True`, and the diagnostics name
  the decoder, its settings and the edges it dropped for closing a cycle. The model's
  native edges stay in `welt_native_edges`.
- **The model's own threshold.** Greedy keeps an edge when its score is above the
  model's native threshold: 0.5 for AVICI and the adaptive threshold Welt records for
  CDFM. `diagnostics["threshold_source"]` says `"welt_native"`. Pass
  `decoder=Decoder("greedy", threshold=0.4)` to choose another.
- **Other decoders.** `decoder="exact_map"`, `"expected_count"` or a sampler such as
  `Decoder("pair_sampler", samples=100, seed=0)` work the same way. To compare
  decoders, decode `result.diagnostics["edge_probabilities"]` locally with
  `ergodic.decode`; that needs no new run.
- **Knowledge.** With `decoder=`, `knowledge=` constrains the decoding. Welt itself
  still runs without constraints. Knowledge that names an unknown variable, or a
  `confound` pair, raises before anything is uploaded.

`ergodic.discover(data, method="welt:avici", decoder="greedy")` does the same.

!!! caution "The decoded graph is your choice"
    The scores are uncalibrated and are not a posterior over graphs. A decoder picks one
    DAG from them, and its orientations are not certified. See
    [Decode a DAG from edge probabilities](decode-a-graph.md).
