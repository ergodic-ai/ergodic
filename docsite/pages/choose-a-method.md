# Choose a method and test

Pick a discovery method from the assumptions you can defend, then pick the
independence test or score it runs on. You'll see two cases where the right choice
changes the graph.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A first run from [Discover a graph](discover-a-graph.md), so you know how to read
      the result.

## Run the program

Both worlds are simulated with fixed seeds, so you know the true graph. In the first,
the noise is uniform rather than Gaussian. In the second, a drug dose raises a blood
level exponentially and the response saturates.

```python file="choose_method.py"
import numpy as np
import pandas as pd

from ergodic import discover, tabular
from ergodic.discovery import GCM, Linear

# 1. A linear world with uniform noise: a -> b -> c -> d and a -> d.
rng = np.random.default_rng(0)
n = 2_000
a = rng.uniform(-1, 1, n)
b = 0.8 * a + rng.uniform(-1, 1, n)
c = 0.7 * b + rng.uniform(-1, 1, n)
d = 0.5 * a + 0.6 * c + rng.uniform(-1, 1, n)
linear_world = tabular(pd.DataFrame({"a": a, "b": b, "c": c, "d": d}))

# 2. PC returns the equivalence class; LiNGAM uses the non-Gaussian noise.
print("pc:    ", discover(linear_world, method="pc").graph)
print("lingam:", discover(linear_world, method="lingam").graph)

# 3. A nonlinear chain: dose -> level -> response.
rng = np.random.default_rng(0)
dose = rng.uniform(0, 3, 1_000)
level = np.exp(dose) + rng.normal(0, 1, 1_000)
response = 10 * level / (level + 5) + rng.normal(0, 0.5, 1_000)
chain = tabular(pd.DataFrame({"dose": dose, "level": level, "response": response}))

# 4. Is dose independent of response given level? It is, in truth.
for test in [Linear(chain), GCM(chain, learner="knn")]:
    p = test.test("dose", "response", ["level"]).p_value
    print(f"{test.name}: p = {p:.3f}")

# 5. Run PC with each test.
print("linear:", discover(chain, method="pc", test="linear").graph)
print("gcm:   ", discover(chain, method="pc", test=GCM(learner="knn")).graph)
```

```output from="choose_method.py"
pc:     MixedGraph(kind=cpdag, nodes=4, edges=[a --- b, a --> d, b --- c, c --> d])
lingam: MixedGraph(kind=dag, nodes=4, edges=[a --> b, a --> d, b --> c, c --> d])
linear: p = 0.000
gcm: p = 0.730
linear: MixedGraph(kind=cpdag, nodes=3, edges=[dose --- level, dose --- response, level --- response])
gcm:    MixedGraph(kind=cpdag, nodes=3, edges=[dose --- level, level --- response])
```

- **Step 2.** PC can't orient `a - b` or `b - c`: Gaussian data with those edges
  reversed would look the same. LiNGAM assumes linear effects with non-Gaussian
  noise, and under that assumption it recovers every direction.
- **Step 4.** The partial-correlation test removes only the linear part of `level`.
  The curved part is left over, so it reports a dependence that isn't there. The GCM
  test regresses with nearest neighbours, follows the curve and finds none.
- **Step 5.** With the linear test, PC keeps a false `dose --- response` edge. With
  GCM it finds the chain. A chain has no collider, so its edges stay undirected.

## The methods and their assumptions

The test- and score-based searches assume the causal Markov condition and
faithfulness: the independences in the data are exactly those the graph implies.
Some need less: LiNGAM's directions come from its noise assumption, and BOSS, GRaSP
and exact search are consistent under weaker conditions than faithfulness. [Assumptions behind each
step](assumptions.md) explains both. The table lists what each method adds.

| `method=` | Runs on | Hidden common causes | Data | Returns |
| --- | --- | --- | --- | --- |
| `"pc"` | Test | Not allowed | Any, through the test | CPDAG |
| `"fci"`, `"rfci"` | Test | Allowed, and selection bias | Any, through the test | PAG |
| `"mmpc"` | Test | Not modelled | Any, through the test | Undirected skeleton |
| `"ges"`, `"boss"`, `"grasp"` | Score | Not allowed | Any, through the score | CPDAG |
| `"hc"` | Score | Not allowed | Any, through the score | One DAG, a local optimum |
| `"astar"` | Score | Not allowed | Any, through the score | One DAG of the best-scoring class |
| `"lingam"` | Itself | Not allowed | Continuous, linear, non-Gaussian noise | DAG |
| `"glasso"` | Itself | Not modelled | Continuous, Gaussian | Moral graph (skeleton plus parents of common children) |
| `"pcmci"` | Test | Not allowed | Time series or panel | Window graph |

On Gaussian data, LiNGAM still runs, but its directions mean nothing. A* refuses more
than `node_limit=20` variables, because exact search grows exponentially. Every DAG
in a class gets the same score, so the DAG that A* or HillClimb returns is one member
picked by tie-breaking. Identify effects on `cpdag_of_dag(result.graph)`, not on that
DAG.

## Tests and scores

`test=` and `score=` take a name or an object. The default `"auto"` reads the column
types: all continuous, all categorical, or a mix.

| `test=` | Class | Use it for |
| --- | --- | --- |
| `"linear"`, `"fisher_z"` | `Linear` | Continuous data with linear relations. Auto pick for all-continuous data. |
| `"discrete"`, `"g_sq"` | `Discrete` | Categorical data (G-squared). Auto pick for all-categorical data. |
| `"mixed"`, `"dg"` | `Mixed` | A mix of continuous and categorical columns. Auto pick for mixed data. |
| `"gcm"` | `GCM` | Continuous data with nonlinear relations. |

| `score=` | Class | Use it for |
| --- | --- | --- |
| `"bic"` | `BIC` | Continuous data, Gaussian linear model. |
| `"bic_discrete"`, `"discrete_bic"` | `DiscreteBIC` | Categorical data. |
| `"bdeu"` | `BDeu` | Categorical data, Bayesian score with `ess=10.0`. |
| `"bge"` | `BGe` | Continuous data, linear Gaussian model, Bayesian score (the marginal likelihood). |
| `"bic_mixed"`, `"mixed_bic"` | `MixedBIC` | A mix of column types. |

BDeu and BGe are marginal likelihoods, so a score search looks for their
highest-scoring graph, the posterior mode under a uniform prior. The searches are
heuristic except A*, and GES returns the mode's equivalence class as a CPDAG.
`graph_posterior(data, score="bge")` gives the whole posterior over DAGs. See
[Evaluate a candidate graph](evaluate-a-graph.md#get-a-posterior-from-data-with-a-score).

## Give GCM a learner

`GCM` turns any regression model into a test. `learner=` takes an alias (`"ols"`,
`"knn"`) or any scikit-learn regressor, such as
`RandomForestRegressor(min_samples_leaf=20)`. The default is the built-in
nearest-neighbours regressor.

Give the data in one place. Inside `discover`, build the test without data, as in
step 5:

```python
print("gcm:   ", discover(chain, method="pc", test=GCM(learner="knn")).graph)
```

To query a test yourself, bind it to the data, as in step 4. A bound test also works
with the class API, `PC(test=GCM(chain, learner="knn")).fit()`. Passing a bound test
to `discover` raises an error, because the data would then come from two places.

!!! caution "GCM sees only the conditional mean"
    GCM tests whether two residuals are correlated. Dependence that leaves the
    conditional covariance at zero, such as one variable changing only the spread of
    another, goes unseen. Categorical endpoints with three or more levels raise an error; use `"mixed"`. A
    binary endpoint is tested as a 0/1 column.

<details markdown="1"><summary>How do I make the search faster?</summary>

For test-based methods, `max_cond_size=` caps the size of conditioning sets. For
GES, `max_degree=` caps the edges at any node. For HillClimb and A*, `max_parents=`
caps the parents of each node. You can also screen first with `"mmpc"` or `"glasso"`
and restrict the search to the screen (see
[Add domain knowledge](domain-knowledge.md)).

</details>

## Next steps

<div class="cards" markdown="1">

- [Check stability with the bootstrap](bootstrap-stability.md): see how much of the graph survives resampling.
- [Evaluate a candidate graph](evaluate-a-graph.md): a posterior over DAGs from a score, and a graph read against it.
- [Add domain knowledge](domain-knowledge.md): orient what no test or score can.
- [Python reference](reference-discovery.md): every method, test and score.

</div>
