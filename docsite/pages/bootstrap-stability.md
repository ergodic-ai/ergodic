# Check stability with the bootstrap

Rerun a discovery method on resamples of your data and count how often each edge comes
back. You'll find an edge that one run reported but the resamples don't support.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A discovery method that runs in seconds on your data. The bootstrap runs it once
      per resample.

## Run the program

The data is the Asia chest-clinic network from [Discover a graph](discover-a-graph.md),
with only 1,000 patients this time. Fewer rows make single runs less reliable.

```python file="bootstrap_stability.py"
from ergodic import discover
from ergodic.datasets import load_asia
from ergodic.discovery import bootstrap

# 1. Sample 1,000 patients and run GES once.
asia = load_asia(n=1_000, seed=0)
single = discover(asia.data, method="ges").graph
print(single)

# 2. Rerun GES on 100 resamples of the rows.
boot = bootstrap(asia.data, method="ges", n_bootstrap=100, seed=0)
print(boot)

# 3. How often was each pair adjacent?
for pair, share in sorted(boot.edges.items(), key=lambda item: -item[1])[:10]:
    print(f"{' - '.join(sorted(pair)):16} {share:.2f}")

# 4. Which marks did one pair wear?
for glyph in ["bronc --- smoke", "bronc --> smoke", "bronc <-- smoke"]:
    print(glyph, boot.marks.get(glyph, 0.0))

# 5. Keep the edges found in at least half the runs.
stable = boot.summary(threshold=0.5)
print(stable)
print("dropped:", sorted({e.glyph() for e in single.edges} - {e.glyph() for e in stable.edges}))
```

```output from="bootstrap_stability.py"
MixedGraph(kind=cpdag, nodes=8, edges=[asia <-- xray, bronc --> dysp, bronc --- smoke, dysp <-- either, either <-- lung, either <-- tub, either --> xray, lung --- smoke])
BootstrapResult(ges: 100 runs, 0 failed, 7/21 edges stable at 0.5)
bronc - dysp     1.00
bronc - smoke    1.00
either - lung    1.00
either - tub     1.00
either - xray    1.00
lung - smoke     0.95
dysp - either    0.73
asia - tub       0.34
asia - xray      0.27
dysp - lung      0.25
bronc --- smoke 0.82
bronc --> smoke 0.14
bronc <-- smoke 0.04
MixedGraph(kind=mixed, nodes=8, edges=[bronc --> dysp, bronc --- smoke, dysp <-- either, either <-- lung, either <-- tub, either --> xray, lung --- smoke])
dropped: ['asia <-- xray']
```

- **Step 1.** One run reports 8 edges, including `asia <-- xray`. The true network
  has no such edge.
- **Step 2.** 21 pairs were adjacent in at least one run. 7 of them were adjacent in
  at least half.
- **Step 3.** Six pairs appear in 95% of runs or more, and `dysp - either` in 73%.
  `asia - xray` appears in 27%: whether GES draws it depends on which patients are in
  the sample. The true edge `asia - tub` is fragile too, at 34%, because so few
  patients visited Asia.
- **Step 4.** `marks` splits a pair's share by orientation. The three glyphs for
  `bronc` and `smoke` add up to its adjacency share.
- **Step 5.** `summary` keeps each pair at or above the threshold, with its most
  frequent orientation. The spurious edge is gone.

## Read the frequencies

`bootstrap` returns a `BootstrapResult`:

| Field | What it holds |
| --- | --- |
| `edges` | Each node pair, as a `frozenset`, mapped to the share of runs in which it was adjacent. |
| `marks` | Each edge glyph, such as `"bronc --> dysp"`, mapped to the share of runs that drew it that way. |
| `n_runs`, `n_failed` | Runs that finished, and runs that raised and were skipped. |
| `frequency(a, b)` | The adjacency share of one pair, `0.0` if it never appeared. |
| `summary(threshold)` | A graph of the pairs at or above `threshold`, each with its most frequent glyph. |

Shares are over the runs that finished. A resample can fail, for example when a column
comes out constant, or when its graph can't satisfy `knowledge=`. It is counted in
`n_failed` and skipped, so with knowledge the shares describe only the runs that agreed
with it. Only a `ValueError` or `RuntimeError` is skipped; any other error stops the
bootstrap.

!!! caution "Frequencies are not probabilities"
    An edge at 0.9 came back in 90% of resamples. That measures how much the result
    moves with the sample. It is not the probability that the edge is real: a
    consistent bias, such as a missing confounder, shows up in every run.

The summary graph always has kind `mixed`. A vote over graphs gives no guarantee that
the result is a CPDAG, even when every run returned one. Re-face it with `as_kind` if
it fits a face, as in [Add domain knowledge](domain-knowledge.md).

## Options

`bootstrap(data, method=..., n_bootstrap=100, seed=0, knowledge=None, block_length=None, **kwargs)`
passes `knowledge` and any extra keyword, such as `alpha=` or `score=`, to every run.
The same data, seed and settings give the same result.

- **Tables** are resampled row by row, with replacement.
- **Time series** under `method="pcmci"` are resampled in blocks of consecutive rows.
  Each block becomes a separate unit, so no lag window crosses two blocks.
  `block_length` defaults to the larger of `ceil(sqrt(T))` and `2 * (max_lag + 1)`.
  Glyphs then name window nodes, such as `"x[t-1] --> y[t]"`.
- **Panels** are resampled by whole units, with replacement.

!!! note "Welt methods are refused"
    `bootstrap` raises for `method="welt:<model>"`, because each resample would be a
    paid Welt job. Run `discover` once instead.

<details markdown="1"><summary>How many resamples do I need?</summary>

A share from 100 runs has a standard error of at most 0.05, which is enough to tell a
stable edge from a fragile one. Add runs when you need to compare shares close to your
threshold. Each run costs one full search.

</details>

## Next steps

<div class="cards" markdown="1">

- [Add domain knowledge](domain-knowledge.md): pin down what resampling can't.
- [Discover from time series](time-series-discovery.md): the block bootstrap with PCMCI.
- [Work with edge probabilities](edge-probabilities.md): turn these frequencies into a DAG with a decoder.
- [Python reference](reference-discovery.md): `bootstrap` and `BootstrapResult`.

</div>
