# Test a graph against data

Check whether a causal graph you already have, written by hand or returned by
discovery, is compatible with a table of observations. You'll test a true graph, a
graph with a missing edge, and a graph that differs from the truth but implies the
same independences.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A candidate graph whose observed nodes are columns of your table.
    - A table whose rows are independent observations.

## Run the program

A missing edge is a claim. If the graph has no edge between `ads` and `sales`, it says
some set of other variables makes them independent. `test_graph` lists one such claim
for every pair without an edge, tests each on the data, and combines the p-values into
one. The data here come from a known graph, so you can see what a pass and a failure
look like.

```python file="test_a_graph.py"
import numpy as np

from ergodic import dag, tabular, test_graph

# 1. Simulate 1,000 weeks from a known linear graph.
rng = np.random.default_rng(0)
n = 1000
season = rng.normal(size=n)
ads = 0.8 * season + rng.normal(size=n)
visits = 0.9 * ads + 0.2 * season + rng.normal(size=n)
price = rng.normal(size=n)
sales = 0.7 * visits - 0.6 * price + rng.normal(size=n)
data = tabular(
    {"season": season, "ads": ads, "visits": visits, "price": price, "sales": sales}
)
truth = dag(
    [
        "season -> ads",
        "season -> visits",
        "ads -> visits",
        "visits -> sales",
        "price -> sales",
    ]
)

# 2. Test the true graph: one claim per pair without an edge.
result = test_graph(truth, data)
print(result)
print(result.to_frame()[["x", "y", "given", "p_value", "p_holm"]].round(3))

# 3. Drop an edge the data need.
missing = test_graph(truth.remove_edge("season", "visits"), data)
print(missing)
worst = missing.implications[0]
print(f"{worst.x} _||_ {worst.y} | {', '.join(worst.given)}: p_holm = {worst.p_holm:.2g}")

# 4. Reverse season -> ads: a different graph with the same independences.
twin = truth.remove_edge("season", "ads").add_edge("ads -> season").as_kind("dag")
print(test_graph(twin, data))
```

```output from="test_a_graph.py"
GraphTest(dag, linear test, k=5, C=4.537, p=0.9199, not rejected at alpha=0.05, 0 claim(s) rejected after Holm)
        x       y                  given  p_value  p_holm
0  season   sales          price, visits    0.280     1.0
1   price     ads                 season    0.655     1.0
2   price  visits            ads, season    0.769     1.0
3   price  season                           0.811     1.0
4     ads   sales  price, season, visits    0.905     1.0
GraphTest(dag, linear test, k=6, C=19.308, p=0.08136, not rejected at alpha=0.05, 1 claim(s) rejected after Holm)
season _||_ visits | ads: p_holm = 0.0038
GraphTest(dag, linear test, k=5, C=4.824, p=0.9026, not rejected at alpha=0.05, 0 claim(s) rejected after Holm)
```

- **Step 2.** The true graph has five pairs without an edge, so it makes five claims.
  For a DAG each claim conditions on the parents of both variables, as in Shipley's
  basis set. Fisher's C sums `-2 log p` over the claims; if every claim holds it follows
  a chi-square with 10 degrees of freedom, and the p-value is its tail. Each claim
  also gets a Holm-adjusted p-value.
- **Step 3.** Without `season -> visits` the graph claims that `season` and `visits` are
  independent given `ads`. They aren't: that claim's Holm-adjusted p-value is 0.0038, so
  Holm rejects it at 0.05. Fisher's C doesn't, with p = 0.081, because one false claim
  is averaged with five true ones. Read both: the combined p-value is the better
  detector when many claims fail a little, and the ranked claims are the better one
  when a single claim fails clearly. They come smallest p-value first, so the first one
  points at the problem.
- **Step 4.** The reversed graph is Markov equivalent to the truth: it has the same
  edges and the same colliders, so it implies the same independences. It passes too.
  No test on observational data can tell the two apart.

## What the p-value means

The p-value answers one question: if every independence the graph implies held, how
surprising would data like these be? A small value is evidence against the graph. The
claims with small Holm-adjusted p-values say which missing edges the data reject.

!!! caution "A pass doesn't confirm the graph"
    A large p-value means the data don't contradict the graph's claims. It doesn't
    mean the graph is right:

    - Markov-equivalent graphs imply the same independences, so they all get the same
      verdict. Edge directions inside an equivalence class are never tested.
    - Only missing edges are tested. An edge the graph has but the world lacks costs
      nothing, so a complete graph passes any data.
    - With few rows every test has little power, and a wrong graph can pass.
    - The independence test brings its own assumptions. The partial-correlation test
      sees only linear dependence.

There are two readings of the same tests. `p_value` combines all claims with Fisher's C,
which assumes the claims' tests are independent of each other. Shipley (2000) built the
DAG basis set so that they are when the model holds. `rejected` on a claim uses Holm's
procedure instead, which keeps the chance of any false rejection at `alpha` whatever
the dependence. On simulated six-variable linear graphs, the true graph was rejected at
`alpha = 0.05` in 5.5% of 3,000 seeded worlds with 200 rows, and 4.6% with 1,000 rows.

## Read the result

`test_graph` returns a `GraphTest`:

| Field | What it holds |
| --- | --- |
| `statistic`, `dof` | Fisher's C and its degrees of freedom, twice the number of claims. |
| `p_value` | The combined p-value; 1.0 when there is nothing to test. |
| `rejected` | Whether `p_value <= alpha`. |
| `k`, `n_rejected` | How many claims were tested, and how many Holm rejects. |
| `test` | The independence test that ran, such as `"linear"`. |
| `kind` | How the graph was read: `dag`, `cpdag`, `admg`, `mag` or `pag`. |
| `n` | Rows used, after dropping incomplete ones. |
| `implications` | The claims, smallest p-value first. |
| `untestable` | Pairs without an edge that no observed set separates. |
| `notes` | Caveats about this run, such as a small sample. |

Each claim is an `ImpliedIndependence` with `x`, `y`, `given`, `statistic`, `p_value`,
`p_holm`, `rejected` and `rule`. `to_frame()` lays them out as a table, and `to_dict()`
gives the whole result as plain data.

## Which claims each graph makes

| Graph | The claim for a pair without an edge |
| --- | --- |
| DAG | Independent given the parents of both. |
| CPDAG | The same, read in one member DAG. Every member implies it. |
| ADMG, or a DAG with `latent=` nodes | Independent given the parents of both when that set separates them in the graph. Otherwise a minimal set of the pair's observed ancestors. |
| PAG | The same, read in one member MAG. Every member implies it. |

Every conditioning set is checked against the graph's own separation rule before any
data are touched. With a hidden common cause, some pairs without an edge still can't be
separated by any set of observed variables. Those pairs carry no claim and appear in
`untestable`. Fisher's C is an approximation for these graphs, because the claims'
tests needn't be independent, so lean on the Holm decisions there.

## Choose the test

`test=None` picks the test from the columns of the graph's observed variables, as
`discover` does: partial correlation when all are continuous, G squared when all are
discrete, and the degenerate-Gaussian test when they mix. Pass a name such as
`test="gcm"` or an unbound test such as `test=GCM(learner="knn")` to choose another.
[Choose a method and test](choose-a-method.md) describes each one. A test already bound
to data is refused, so the rows come from one place.

<details markdown="1"><summary>What if my table has more columns than the graph?</summary>

The extra columns are ignored, and a note lists them. The graph's claims are about its
own variables, and they hold in the margin over those variables.

</details>

## Next steps

<div class="cards" markdown="1">

- [Discover a graph](discover-a-graph.md): learn a candidate from the data.
- [Assumptions behind each step](assumptions.md): what a graph assumes and what ergodic checks.
- [Python reference](reference-discovery.md): `test_graph` and `GraphTest` in full.

</div>
