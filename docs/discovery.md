# Causal discovery

Discovery learns causal structure from data. Two ingredients drive every search in D1: a conditional independence test for the constraint algorithms, and a decomposable score for the exact search. Algorithms are thin consumers; the ingredients are objects you instantiate, configure, share, and replace.

## One call

```python
import numpy as np
import pandas as pd
from ergodic import tabular, discover

rng = np.random.default_rng(0)
n = 4000
x = rng.normal(size=n)
m = 0.8 * x + rng.normal(size=n)
y = 0.7 * m + rng.normal(size=n)
data = tabular(pd.DataFrame({"X": x, "M": m, "Y": y}))

result = discover(data, method="pc")
result.graph        # the CPDAG: X -- M -- Y (a chain is unoriented inside its class)
result.sepsets      # {frozenset({'X', 'Y'}): frozenset({'M'})}
```

The result is a record, not a bare graph: the algorithm, the ingredient, the significance level, the run diagnostics, and the separating sets ride along with it.

## The test is the oracle

A test is a class. Constructed with data, it is the stateful oracle: encoding done once, sufficient statistics precomputed, every query cached.

```python
from ergodic.discovery import PC, Linear

linear = Linear(data)                       # bound: the oracle
result = PC(test=linear, alpha=0.05).fit()  # PC queries it
```

Constructed bare, it is a reusable configuration that the algorithm binds at fit:

```python
PC(test=Linear()).fit(data)     # bound at fit
PC(test="linear").fit(data)     # the string alias
```

The data comes from exactly one place. An unbound test needs data at `fit`; a bound test can omit it; a bound test plus a different dataset raises; neither raises.

Alpha lives in the algorithm, not the test, so the p-value cache is alpha-independent. One warmed oracle serves many runs:

```python
linear = Linear(data)
strict = PC(test=linear, alpha=0.01).fit()
loose = PC(test=linear, alpha=0.10).fit()   # every p-value comes from the cache
```

## The tests

| Test | Alias | Data | Statistic |
| --- | --- | --- | --- |
| `Linear` | `"linear"`, `"fisher_z"` | continuous | partial correlation, Fisher z |
| `Discrete` | `"discrete"`, `"g_sq"` | categorical | G squared, natively on the codes |
| `Mixed` | `"mixed"`, `"dg"` | anything | degenerate Gaussian: one-hot blocks, partial canonical correlations, Bartlett chi-square |

`test="auto"` (the default) reads the column schema: all continuous picks `Linear`, all discrete picks `Discrete`, anything else picks `Mixed`. `Mixed` agrees with `Linear` on continuous data, so it is the safe general choice; `Discrete` never encodes, so it is exact for categorical levels.

## Write your own test

Subclass, declare the input form, implement the statistic. The base owns binding, encoding, caching, and validation:

```python
from ergodic.discovery import CITest

class MyTest(CITest):
    name = "my_test"
    needs = "numeric"                       # 2-D float blocks; "codes" for integer categories

    def statistic(self, x, y, z):           # arrays the oracle prepared
        ...
        return stat, p_value, dof
```

`MyTest(data)` now drops into PC like any shipped test.

## The ground-truth oracle

`DSeparationOracle` answers queries from a known graph instead of data. It quacks like any bound test, which makes algorithm behavior exact and inspectable:

```python
from ergodic import dag
from ergodic.discovery import DSeparationOracle

truth = dag(["X -> M", "M -> Y"])
PC(test=DSeparationOracle(truth)).fit().graph    # exactly the CPDAG of the truth
```

This is the recovery guarantee made executable: under a perfect oracle, PC returns the equivalence class, no sampling noise involved.

## Knowledge

Discovery consumes `DomainKnowledge` in four modes: prune (forbidden adjacencies are never tested), protect (required edges are never removed), orient (tiers and required edges fix arrowheads before Meek propagation), and validate (the returned graph passes `is_consistent`).

```python
from ergodic import DomainKnowledge

dk = DomainKnowledge().with_tiers([["X"], ["M"], ["Y"]])
discover(data, method="pc", knowledge=dk)    # tier order orients the chain
```

A skeleton is a constraint like any other: `restrict_to` confines the search to an adjacency support, which is how an undirected screen (or a previous run) feeds the next algorithm.

```python
dk = DomainKnowledge().restrict_to(skeleton_graph)
discover(data, method="astar", knowledge=dk)   # exact search inside the support
```

## The score and exact search

Score-based search runs on one queryable object: the local score of a node given its parents.

```python
from ergodic.discovery import AStar, BIC

bic = BIC(data)                  # covariance computed once
bic.local("Y", ["M"])            # cached local score
result = AStar(score=bic).fit()  # the globally optimal DAG under the score
```

`AStar` is the Yuan and Malone shortest path over the order lattice: exact, not greedy. It is honest about scale (a node limit, raised with a pointer at `max_parents` and `restrict_to`), and it consumes knowledge the same four ways. BIC is score-equivalent, so the returned DAG is one member of the optimal equivalence class; take `cpdag_of_dag` of it to compare with PC.

A custom score is one method, mirroring the tests:

```python
from ergodic.discovery import Score

class MyScore(Score):
    name = "my_score"

    def local_from_arrays(self, y, X):
        ...
        return value
```

## Putting the pieces together

The pillars compose end to end: discover the structure, take a DAG member of the class, then identify and estimate on it.

```python
from ergodic import estimate_effect
from ergodic.graph import consistent_extension

result = discover(data, method="pc", knowledge=dk)
member = consistent_extension(result.graph)        # one DAG from the equivalence class
est = estimate_effect(member, data, "X", "Y", strategy="dml")
```

And the ingredients compose sideways: a warmed oracle across algorithms, a skeleton from one method restricting the next, a d-separation oracle standing in for data, your test inside unchanged PC.

## Not built yet

GES and the order-based searches (the score contract is ready for them), the mixed and discrete scores (BIC is Gaussian in D1), the GCM test that turns any `Learner` into a CI test, FCI and the PAG output (the PC skeleton machinery is most of it), permutation significance behind the same tests, per-pair test dispatch from the schema, the skeleton searches (graphical lasso, Markov blankets), the bootstrap stability wrapper, and the time-series shapes.

::: ergodic.discovery
