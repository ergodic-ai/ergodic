# Discover in Welt, estimate in ergodic

Learn a causal graph with a hosted Welt model, then identify and estimate an effect
on it in ergodic. Welt also fits the nuisance models inside the estimate.

!!! need "You'll need"
    - Python 3.11, 3.12 or 3.13.
    - ergodic with the Welt extra: `pip install "ergodic[welt]"`.
    - A [Welt](https://welt.ergodic.dev) workspace with credits. Every discovery, fit
      and prediction in this guide is a paid request.

## Connect to Welt

Install the extra, then connect your workspace from a terminal:

```sh
pip install "ergodic[welt]"
welt login
```

`welt login` asks you to approve access in your browser, then saves the key for
later Python processes. Or set the `WELT_API_KEY` environment variable. Each
function in `ergodic.welt` opens a short-lived Welt client that finds the key this
way, unless you pass your own `client=`. The
[Welt SDK documentation](https://ergodic-ai.github.io/welt-python/connect.html)
covers keys and workspaces.

## Run the program

The data is 400 births from the bundled birthweight data (Almond, Chay and Lee,
2005). The program keeps five continuous columns: the parents' ages, the mother's
years of education, the number of prenatal visits and the birthweight in grams.

```python file="welt_effect.py" requires="a Welt API key"
from ergodic import estimate_effect, identify_effect
from ergodic.datasets import load_birthweight
from ergodic.welt import WeltLearner, discover

# 1. Take 400 births and five continuous columns.
births = load_birthweight()
columns = ["mage", "medu", "nprenatal", "bweight", "fage"]
data = births.data.select(columns).take(range(400))
print(data.shape)

# 2. Learn a DAG with Welt's Arrow model: one paid discovery job.
result = discover(data, model="arrow", seed=0)
for edge in sorted(edge.glyph() for edge in result.graph.edges):
    print(edge)
print(result.diagnostics["welt_model_version"])

# 3. Ask whether the graph identifies the effect of prenatal visits on birthweight.
identification = identify_effect(result.graph, "nprenatal", "bweight")
print(identification)

# 4. Estimate it with DML and a Welt tabular model in each nuisance slot.
learner = WeltLearner("tabicl-v2", max_fits=4)
estimate = estimate_effect(
    result.graph, data, "nprenatal", "bweight",
    strategy="dml", learner=learner, n_folds=2, random_state=0,
)
print(estimate)
print("paid fits:", learner.fits_used)
```

The program needs your Welt key, so this page shows no recorded output. Your graph
depends on the Arrow version Welt serves. Each step prints:

- **Step 1.** `(400, 5)`: 400 rows and five variables.
- **Step 2.** One line per edge, such as `medu --> nprenatal`, then the exact Arrow
  version Welt ran. Every variable stays in the graph, unconnected ones too.
- **Step 3.** An `Identification` record. The graph is a DAG with no hidden
  variables, so a directed path from `nprenatal` to `bweight` is identified by
  adjustment. The record names the adjustment set.
- **Step 4.** An `EffectEstimate(dml, analytic: …)` line: the average change in
  grams per extra prenatal visit, its standard error and 95% interval. Then
  `paid fits: 4`.

## Read the provenance

`discover` returns a `DiscoveryResult` with `method="welt:arrow"`, like ergodic's
own discovery methods. Its `diagnostics` record the Welt run: `welt_job_id`,
`welt_result_id`, `welt_dataset_id`, `welt_model`, `welt_model_version`,
`welt_configuration_version`, `welt_decoder`, `welt_seed` and `welt_scores`.

```python
print(result.diagnostics["welt_model_version"])
```

`welt_scores` is Arrow's native edge-score matrix as a dataframe, with rows for the
source and columns for the target. The scores are not calibrated confidence, and
ergodic never thresholds or reorients the graph with them unless you pass `decoder=`.
Then they're read as independent marginals, as for AVICI and CDFM below.
`ergodic.discover(data, method="welt:arrow")` runs the same job.

## Count what you pay for

The program makes 13 paid requests: one discovery job, four fits and eight
predictions. DML fits each of its two nuisance slots once per fold, so `n_folds=2`
means four fits. Each fold then predicts its
held-out rows in batches no larger than the model's prediction-row limit, which is
100 rows for `tabicl-v2` today.

```python
learner = WeltLearner("tabicl-v2", max_fits=4)
```

`max_fits` caps the paid fits of the learner and of every copy ergodic makes of it.
The default, 10, covers DML with five folds. A run that needs one more fit raises
`RuntimeError` before it pays. Estimators with a bootstrap standard error refit for
every resample, so prefer `strategy="dml"`, whose standard error is analytic.

Each fit uploads its rows as a new Welt dataset. Uploads are free and stay in your
workspace until you delete them.

## Choose a tabular model

`WeltLearner` passes any model id to Welt, and Welt's catalogue decides. These
tabular models ran classification and regression in the public catalogue
(`https://welt.ergodic.dev/v1/models`) on 15 October 2026:

| Model | Training rows | Columns | Rows per prediction |
| --- | --- | --- | --- |
| `tabicl-v2` (the default) | up to 8,000 | up to 20 | up to 100 |
| `kumo-medium` | up to 8,000 | up to 20 | up to 100 |
| `mitra-v2` | up to 4,000 | up to 20 | up to 100 |
| `tabdpt-1.3` | up to 500 | up to 20 | up to 100 |

Run `welt models list` to see what Welt serves today. A slot that needs
probabilities, such as a propensity, takes `WeltLearner(task="classification")`.
Pass a mapping such as `{"outcome": WeltLearner(), "propensity":
WeltLearner(task="classification")}` to `learner=`.

!!! caution "Discovered edges are not causal truth"
    Arrow learns the graph from 400 rows. Check it against what you know before you
    trust an effect estimated on it.

## Decode AVICI and CDFM

AVICI and CDFM score every directed edge on its own and return a directed graph that
may hold cycles and two-way pairs. ergodic never turns that into a DAG silently, so by
default `discover` refuses them before anything is paid for. Pass `decoder=` to decode
their scores into a DAG yourself.

```python file="welt_decode.py" requires="a Welt API key"
from ergodic import decode
from ergodic.datasets import load_birthweight
from ergodic.welt import discover

# 1. The same 400 births and five continuous columns.
births = load_birthweight()
columns = ["mage", "medu", "nprenatal", "bweight", "fage"]
data = births.data.select(columns).take(range(400))

# 2. Run CDFM once and decode its edge scores with greedy pair-MAP.
result = discover(data, model="cdfm", decoder="greedy", seed=0)
for edge in sorted(edge.glyph() for edge in result.graph.edges):
    print(edge)
print(result.diagnostics["decoded"], result.diagnostics["threshold_source"])
print(result.diagnostics["parameters"]["threshold"], result.diagnostics["rejected_cycles"])

# 3. The edges CDFM drew itself, cycles and two-way pairs included.
print(sorted(result.diagnostics["welt_native_edges"]))

# 4. Decode the same scores another way, locally: no new run.
probabilities = result.diagnostics["edge_probabilities"]
print(probabilities)
print(decode(probabilities, "expected_count").graph)
```

This program needs your Welt key too, so it shows no recorded output:

- **Step 2.** CDFM runs once, a paid discovery job. `result.graph` is the DAG the
  decoder chose, not CDFM's own graph, and `diagnostics["decoded"]` is `True`. Greedy
  keeps an edge when its score is above the model's native threshold: 0.5 for AVICI,
  and for CDFM the adaptive threshold Welt records, so `threshold_source` is
  `welt_native`. `rejected_cycles` counts the edges it skipped for closing a cycle.
- **Step 3.** CDFM's native edges stay in `welt_native_edges`, as source and target
  pairs.
- **Step 4.** The scores are an `EdgeProbabilities` of kind `independent`, with the
  Welt ids, versions and native decoder in its `provenance`. Decoding them again with
  `ergodic.decode` costs nothing.

Pass `decoder=Decoder("greedy", threshold=0.4)` (from `ergodic.discovery`) to choose
another threshold, or another decoder such as `"exact_map"` or
`Decoder("pair_sampler", samples=100, seed=0)`. With `decoder=`, `knowledge=`
constrains the decoding; Welt itself still runs without constraints, and knowledge that
names an unknown variable or a `confound` pair raises before anything is uploaded.
`ergodic.discover(data, method="welt:cdfm", decoder="greedy")` does the same.

!!! caution "The decoded graph is your choice"
    The scores are uncalibrated and are not a posterior over graphs. A decoder picks
    one DAG from them, and nothing certifies its orientations. See
    [Decode a DAG from edge probabilities](decode-a-graph.md).

## Limits

- Welt discovery reads only continuous variables with no missing values, and takes
  no domain knowledge (with `decoder=`, knowledge constrains the local decoding
  only). `bootstrap` refuses `welt:` methods, since each resample would be a paid job.
- `cdfm` and `avici` return directed graphs that may hold cycles, so without
  `decoder=` `discover` raises `WeltGraphNotDAGError` before it uploads anything. See
  [Decode AVICI and CDFM](#decode-avici-and-cdfm).
- `WeltLearner.fit` refuses `sample_weight`, because Welt fits take no row weights.
- On Python 3.14 the extra installs nothing, and `import ergodic.welt` raises an
  `ImportError` that names it. The rest of ergodic never imports Welt.

<details markdown="1"><summary>What if the graph has no path, or no adjustment set?</summary>

With no directed path from `nprenatal` to `bweight`, `identify_effect` returns the
`null` strategy: by the graph, the effect is zero. `strategy="dml"` then raises,
because DML estimates adjustment effects only.

With an empty adjustment set, DML's nuisance models have no features, and
`WeltLearner` raises before it fits. Leave those slots to the default learner.

</details>

<details markdown="1"><summary>What if a job times out?</summary>

`discover` and `WeltLearner` wait 600 seconds by default (`timeout=`). A timeout
leaves the job running in Welt. Reconnect with `welt.Client().job(job_id)` instead
of submitting again.

</details>

## Next steps

<div class="cards" markdown="1">

- [Compatibility with Welt](compatibility.md): versions, graph format and what converts.
- [Estimate an average effect](estimate-average-effect.md): the same kind of estimate with local learners.
- [Welt SDK documentation](https://ergodic-ai.github.io/welt-python/): models, keys and usage.

</div>
