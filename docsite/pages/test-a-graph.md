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
visits = 0.9 * ads + 0.3 * season + rng.normal(size=n)
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
GraphTest(dag, linear test, k=5, C=4.589, p=0.9169, not rejected at alpha=0.05, 0 claim(s) rejected after Holm)
        x       y                  given  p_value  p_holm
0  season   sales          price, visits    0.273     1.0
1   price     ads                 season    0.655     1.0
2   price  visits            ads, season    0.769     1.0
3   price  season                           0.811     1.0
4     ads   sales  price, season, visits    0.905     1.0
GraphTest(dag, linear test, k=6, C=42.518, p=2.724e-05, rejected at alpha=0.05, 1 claim(s) rejected after Holm)
season _||_ visits | ads: p_holm = 3.5e-08
GraphTest(dag, linear test, k=5, C=4.840, p=0.9016, not rejected at alpha=0.05, 0 claim(s) rejected after Holm)
```

- **Step 2.** The true graph has five pairs without an edge, so it makes five claims.
  For a DAG each claim conditions on the parents of both variables, as in Shipley's
  basis set. Fisher's C sums `-2 log p` over the claims; if every claim holds it follows
  a chi-square with 10 degrees of freedom, and the p-value is its tail. Each claim
  also gets a Holm-adjusted p-value.
- **Step 3.** Without `season -> visits` the graph claims that `season` and `visits` are
  independent given `ads`. They aren't, and the combined p-value of 0.000027 rejects the
  graph. The claims come ranked, smallest p-value first, and Holm rejects only the first
  one, so it points at the missing edge.
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

Decide how you'll read the result before you run the test. The combined p-value is the
decision: `rejected` is `True` when it is at most `alpha`, and that is the test whose
error rate is `alpha`. The Holm-adjusted p-values then locate the claims that fail.
Don't reject the graph when either one rejects. Taking the union spends more than
`alpha`: on simulated linear graphs it rejected true graphs about 8% of the time at
`alpha = 0.05`.

Fisher's C assumes the claims' tests are independent of each other. Shipley (2000)
argues that for the DAG basis set they are when the model holds, and with
large-sample tests such as partial correlation and G squared that holds
approximately. The partial-correlation test is the only one whose size is calibrated.
On simulated six-variable linear-Gaussian graphs, the registered run of 200 worlds per
sample size rejected the true graph at `alpha = 0.05` in 9.5% of worlds with 200 rows,
outside its band of 2% to 8%, and 4.5% with 1,000 rows. A pre-registered replication
on 2,000 fresh worlds per sample size gave 4.65% and 4.85%, inside its band of 3.5% to
6.5%:

| Rows | Registered run (200 worlds) | Replication (2,000 worlds, 95% interval) |
| --- | --- | --- |
| 200 | 9.5% | 4.65% (3.77% to 5.67%) |
| 1,000 | 4.5% | 4.85% (3.95% to 5.88%) |

Against the true graph minus one edge of weight at least 0.5, power was close to 1.

The basis set is also not every independence the graph implies. Together the claims
imply the rest when the distribution has the composition property, as Gaussian
distributions do. With discrete data a pass says less.

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
| `implications` | The claims, smallest p-value first. Ties, such as p-values too small to represent that print as 0, go to the larger statistic. |
| `untestable` | Pairs without an edge that no observed set separates. |
| `vacuous` | Claims whose test had zero degrees of freedom, such as one with a single-level column. They carry no information and are left out of `k`. |
| `notes` | Caveats about this run, such as a small sample or a test that isn't calibrated. |

Each claim is an `ImpliedIndependence` with `x`, `y`, `given`, `statistic`, `p_value`,
`p_holm`, `rejected`, `dof` and `rule`. `x` comes before `y` in a topological order of
the graph; for a CPDAG or PAG, of the member graph the claims were read in.
`to_frame()` lays them out as a table, and `to_dict()` gives the whole result as plain
data.

Rows with a missing value in the graph's columns are dropped, and a note counts them. An
infinite value raises an error instead: a test would read it as no evidence at all.

## Which claims each graph makes

| Graph | The claim for a pair without an edge |
| --- | --- |
| DAG | Independent given the parents of both. |
| CPDAG | The same, read in one member DAG. Every member implies it. |
| ADMG, or a DAG with `latent=` or `selection=` nodes | Independent given the parents of both when that set separates them in the graph. Otherwise a minimal set drawn from the observed ancestors of the pair and of the selection nodes. |
| MAG | The same, with undirected edges followed back as well. |
| PAG | The same, read in one member MAG. Every member implies it. |

With `selection=` nodes, every claim is read given the selection nodes too, because
the data were sampled given them. Every conditioning set is checked against the
graph's own separation rule before any data are touched. With a hidden common cause or
selection, some pairs without an edge still can't be separated by any set of observed
variables. Those pairs carry no claim and appear in `untestable`. Fisher's C is an
approximation for these graphs, because the claims' tests needn't be independent.

## Choose the test

`test=None` picks the test from the columns of the graph's observed variables, as
`discover` does: partial correlation when all are continuous, G squared when all are
discrete, and the degenerate-Gaussian test when they mix. Pass a name or an unbound
test to choose another; [Choose a method and test](choose-a-method.md) describes each
one. A test already bound to data is refused, so the rows come from one place.

Only the partial-correlation test, the default for continuous data, is calibrated. With
any other test the result carries a note saying so, and two are known to reject true
graphs too often:

- `GCM` with its default nearest-neighbour learner rejected true linear-Gaussian graphs
  about 30% of the time at `alpha = 0.05`. For near-linear data keep the default test.
  `GCM(learner="ols")` is closer to it but isn't calibrated either.
- G squared, the default for discrete data, rejected true binary graphs about 10% of
  the time with 200 rows (about 5% with 1,000).

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

!!! caution "GCM learner calibration"
    The default GCM learner remains KNN; it is not a generally calibrated CI test.
    A paired 200-seed direct-CI diagnostic at alpha 0.05 found null rejection rates
    of 9.5%/7.5% (linear), 12%/10% (quadratic), and 10%/9% (sine), at n=200/1000.
    OLS gave 7.5%/5%, 100%/100%, and 99%/100%, respectively: replacing KNN with
    OLS would fail badly for these nonlinear conditional means. Neither qualified
    across all cells of the registered 2–8% investigation band, so the default
    is unchanged. All cells used 200 trials with zero refusals; 95% Wilson intervals
    for the KNN rates span roughly 4.6–17.2%. These are direct-CI results, distinct
    from the earlier ~30% rejection of true graphs by Fisher's combined test.
    Learner choice must match the conditional mean; these diagnostics do not
    establish calibration for arbitrary data or guarantee detection of dependence.
    The reproducible protocol is in `scripts/audits/qualify_gcm.py`.

Linear and GCM compute two-sided normal tails with `erfc`, so strong statistics
such as z=9 retain their nonzero p-values (about 2.26e-19). Tails below binary64's
representable range still become zero, for example z=40. Graph-test aggregation
retains its existing safeguards for zero tails; no finite float can represent
those smaller probabilities accurately.
