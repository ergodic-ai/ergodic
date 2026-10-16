# Your first causal effect

Learn a causal graph from data, check that it answers your question, and estimate
the effect with an interval. The example runs in seconds on a bundled dataset.

!!! need "You'll need"
    - Python 3.11 or later.
    - ergodic, installed with `pip install "ergodic"`.

## Run the program

The data comes from the textbook burglar-alarm network (Pearl, 1988). A burglary or
an earthquake can set off an alarm, and the alarm makes John and Mary call. Because
the network is known, you can check the answer.

```python file="first_effect.py"
from ergodic import discover, estimate_effect, identify_effect
from ergodic.datasets import load_earthquake

# 1. Load 10,000 rows sampled from the burglar-alarm network.
alarm = load_earthquake(n=10_000, seed=0)
rows = alarm.data.to_pandas()
print(rows.shape)
print((rows == "True").mean().round(3))

# 2. Learn the graph from the data alone.
graph = discover(alarm.data, method="pc").graph
for edge in sorted(edge.glyph() for edge in graph.edges):
    print(edge)

# 3. Ask whether the data can answer the question.
identification = identify_effect(graph, "Burglary", "JohnCalls")
print(identification)

# 4. Estimate the effect of a burglary on John calling.
estimate = estimate_effect(graph, alarm.data, "Burglary", "JohnCalls", treated="True")
print(estimate)
```

```output from="first_effect.py"
(10000, 5)
Burglary      0.011
Earthquake    0.022
Alarm         0.017
JohnCalls     0.067
MaryCalls     0.022
dtype: float64
Alarm --> JohnCalls
Alarm --> MaryCalls
Alarm <-- Burglary
Alarm <-- Earthquake
Identification(adjustment: P(JohnCalls | do(Burglary)) = sum_z P(JohnCalls | Burglary, z) P(z), with z = {Earthquake})
EffectEstimate(aipw, analytic: ate=0.7982, se=0.0333, 95% CI [0.733, 0.863], n=10000)
```

- **Step 1.** Five yes/no columns. Burglaries are rare: about 1 row in 100.
- **Step 2.** `discover` runs the PC algorithm. It finds all four true edges and
  their directions.
- **Step 3.** The effect is identifiable, by adjusting for `Earthquake`.
- **Step 4.** A burglary raises the chance that John calls by about 0.80, with a 95%
  interval of 0.73 to 0.86.

## Check the answer

The network's own tables give the true effect: John calls with probability 0.849
after a burglary and 0.056 without one. The difference, 0.793, is inside the
interval.

## Read the identification

`identify_effect` answers a yes/no question before any estimation: can this graph
and this data recover the effect?

```python
identification = identify_effect(graph, "Burglary", "JohnCalls")
print(identification)
```

It returns the formula it will estimate. Here that is a sum over `Earthquake`: the
canonical adjustment set, every ancestor of the treatment or the outcome that the
treatment does not affect.

!!! caution "Discovery depends on the sample"
    With another sample, PC can leave an edge undirected. Then the
    effect may not be identifiable, and `estimate_effect` raises an error that
    names the ambiguous edge. That is the honest answer, not a bug.

`treated="True"` names the treated level, because the columns hold the strings
`"True"` and `"False"`.

## Next steps

<div class="cards" markdown="1">

- [Estimate an average effect](estimate-average-effect.md): adjust for known confounders on real data.
- [Notebooks](notebooks.md): the learn series and case studies.
- [Python reference](reference.md): every function and its parameters.
- [Example scripts](https://github.com/ergodic-ai/ergodic/tree/v0.3.0/examples): runnable analyses of the bundled datasets.

</div>
