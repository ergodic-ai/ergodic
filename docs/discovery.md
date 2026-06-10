# Causal discovery

Discovery learns causal structure from data. Two ingredients drive every search: a conditional independence test for the constraint algorithms (PC, FCI, MMPC), and a decomposable score for the score-based ones (GES, hill climbing, exact A*). Algorithms are thin consumers; the ingredients are objects you instantiate, configure, share, and replace.

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
| `GCM` | `"gcm"` | continuous | generalized covariance measure: a regression learner per endpoint, residual product |

`test="auto"` (the default) reads the column schema: all continuous picks `Linear`, all discrete picks `Discrete`, anything else picks `Mixed`. `Mixed` agrees with `Linear` on continuous data, so it is the safe general choice; `Discrete` never encodes, so it is exact for categorical levels; `GCM` is the one to reach for when the relationships are nonlinear, and you choose it explicitly.

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

## Any learner is a test

`GCM` (Shah and Peters 2020) turns a conditional-mean regressor into a conditional independence test: regress each endpoint on the conditioning set, and check that the product of the two residuals has mean zero. The regression slot is the same `Learner` surface estimation uses, so the native learners, any sklearn regressor, or xgboost drop in unchanged.

```python
from ergodic.discovery import GCM, PC

gcm = GCM(data)                                  # the native KNN regressor by default
PC(test=gcm).fit()

from sklearn.ensemble import RandomForestRegressor
PC(test=GCM(data, learner=RandomForestRegressor())).fit(data)
```

This is the test that catches what the partial correlation misses. A quadratic confounder (`X = Z**2 + noise`, `Y = Z**2 + noise`) leaves structure in both linear regressions, so `Linear` declares `X` and `Y` dependent given `Z` and PC keeps a spurious edge; `GCM` with a flexible learner removes it. The honest limit: dependence that leaves the conditional covariance at zero is invisible to it.

## The ground-truth oracle

`DSeparationOracle` answers queries from a known graph instead of data. It quacks like any bound test, which makes algorithm behavior exact and inspectable:

```python
from ergodic import dag
from ergodic.discovery import DSeparationOracle

truth = dag(["X -> M", "M -> Y"])
PC(test=DSeparationOracle(truth)).fit().graph    # exactly the CPDAG of the truth
```

This is the recovery guarantee made executable: under a perfect oracle, PC returns the equivalence class, no sampling noise involved.

## Latent confounders: FCI and the PAG

PC assumes every common cause is measured. Drop that assumption and the honest output changes type: FCI returns a partial ancestral graph (PAG), whose marks survive hidden confounding. An arrowhead at `V` says `V` does not cause the other endpoint in any graph compatible with the data; a tail says it does in all of them; a circle says the data cannot tell. A pair that looks adjacent may share only a hidden common cause, and the PAG says exactly that with `o-o` or `<->` instead of pretending to know.

```python
import numpy as np
import pandas as pd
from ergodic import tabular, discover

rng = np.random.default_rng(0)
n = 4000
l1, l2 = rng.normal(size=n), rng.normal(size=n)   # two hidden confounders
a = l1 + 0.5 * rng.normal(size=n)
b = l2 + 0.5 * rng.normal(size=n)
c = l1 + l2 + 0.5 * rng.normal(size=n)
data = tabular(pd.DataFrame({"A": a, "B": b, "C": c}))

result = discover(data, method="fci", alpha=0.01)
result.graph        # the PAG: A o-> C <-o B
```

Read that result the only way it can be read: C does not cause A and does not cause B (the arrowheads at C), and whether A and B cause C or merely share hidden causes with it stays open (the circles). No CPDAG can say this.

The run is PC's stable skeleton, then the possible-d-sep phase (with latents, a separating set can live outside both neighbourhoods, so candidate sets widen), then re-orientation and Zhang's complete rule closure, assuming no selection bias. With a perfect oracle the output is exactly the canonical PAG of the truth, `pag_of_mag(mag_of_dag(truth))`, and the test suite holds it to that identity across random graphs. `max_cond_size` caps both testing phases, and the same warmed oracle that served PC serves FCI from cache.

Knowledge translates through ancestry, not edge direction. `require(u, v)` orients `u -> v`; `confound(a, b)` orients `a <-> b` (a confounded pair is read as a pure hidden common cause); a tier pins only an arrowhead at the later end, because a later variable cannot cause an earlier one while confounding stays possible. Directional forbids, roots, and sinks fix no single mark in a latent world, so they only validate. Knowledge-seeded marks are not class invariants: the result then keeps the `mixed` face and `diagnostics["pag_valid"]` is `0.0`.

## The skeleton screen

MMPC (max-min parents and children) is a constraint search that stops at the skeleton: for each target it keeps the variables no subset of the candidate set can separate from it, then keeps an adjacency only when it holds from both sides. It is the cheap screen you run before an expensive search, and its output is built to feed `restrict_to`:

```python
screen = discover(data, method="mmpc")
support = DomainKnowledge().restrict_to(screen.graph)
discover(data, method="astar", knowledge=support)   # exact search inside the screen
```

That composition is the classic MMHC hybrid, recovered here as two objects instead of one algorithm. A score search inside the screen loses nothing (scores need no separating sets); PC inside the screen keeps the skeleton but can orient less, because a pruned pair is never tested and so never witnesses a v-structure.

## How stable is that edge?

One run returns one graph and no sense of how fragile it is. The bootstrap wrapper reruns any method on row resamples and counts what survives:

```python
from ergodic.discovery import bootstrap

stability = bootstrap(data, method="pc", n_bootstrap=100, seed=0)
stability.frequency("A", "C")      # adjacency frequency in [0, 1]
stability.marks                    # frequency per oriented glyph
stability.summary(threshold=0.8)   # the graph of edges that survive 80% of resamples
```

It wraps constraint and score methods alike (`bootstrap(data, method="ges")` works unchanged), and the frequencies are resampling stability, not posterior probabilities; graph posteriors are a later, Bayesian phase.

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

## The scores and the score-based searches

Score-based search runs on one queryable object: the local score of a node given its parents.

```python
from ergodic.discovery import AStar, BIC

bic = BIC(data)                  # covariance computed once
bic.local("Y", ["M"])            # cached local score
result = AStar(score=bic).fit()  # the globally optimal DAG under the score
```

| Score | Alias | Data | Form |
| --- | --- | --- | --- |
| `BIC` | `"bic"` | continuous | Gaussian BIC, one covariance pass, Schur complement locals |
| `DiscreteBIC` | `"bic_discrete"` | categorical | multinomial BIC, natively on the codes |
| `BDeu` | `"bdeu"` | categorical | Bayesian Dirichlet equivalent uniform, `ess` prior weight |
| `MixedBIC` | `"bic_mixed"` | anything | degenerate Gaussian BIC: one-hot blocks as a joint Gaussian |

`score="auto"` (the default) mirrors the tests: all continuous picks `BIC`, all discrete picks `DiscreteBIC`, anything else picks `MixedBIC`. All four are decomposable and score-equivalent, so equal totals across a Markov equivalence class are a tested property, not a hope; `BDeu` you ask for by name.

Three searches consume the same object:

```python
from ergodic.discovery import GES, HillClimb

GES(score=bic).fit()         # greedy over equivalence classes, returns a CPDAG
HillClimb(score=bic).fit()   # add/remove/reverse over DAGs, with a tabu list
AStar(score=bic).fit()       # the exact optimum, honest about scale
```

`GES` (Chickering 2002) is the standard: a forward pass inserts the best edge into the equivalence class until nothing improves, a backward pass deletes, and the result is a CPDAG directly. In the large-sample limit, with a score-equivalent score, it finds the optimal class; on real data the exact search is its referee, and the test suite holds GES to the A* optimum. `HillClimb` is the fast baseline: steepest ascent over DAGs with a tabu ring to escape plateaus, no optimality story. `AStar` is the Yuan and Malone shortest path over the order lattice: exact, not greedy, with a node limit raised with a pointer at `max_parents` and `restrict_to`. All three consume knowledge the same four ways, and a DAG result compares with PC through `cpdag_of_dag`.

A custom score is one method, mirroring the tests: implement `local_from_arrays(y, X)` for numeric data, or declare `needs = "codes"` and implement `local_from_codes(y, y_card, zs, z_cards)` for native categorical scoring.

```python
from ergodic.discovery import Score

class MyScore(Score):
    name = "my_score"

    def local_from_arrays(self, y, X):
        ...
        return value
```

## Putting the pieces together

The pillars compose end to end, and you do not have to pick a member of the class to do it. Identification accepts the CPDAG or PAG directly and answers with the generalized adjustment criterion, so a success holds in every graph the class represents, and `estimate_effect` rides on that:

```python
from ergodic import estimate_effect

result = discover(data, method="pc", knowledge=dk)
est = estimate_effect(result.graph, data, "X", "Y", strategy="dml")
```

The same line works on an FCI result: the PAG may keep circles where confounding cannot be ruled out, and the effect still estimates whenever the treatment's outgoing edge is visible (certified clean of hidden confounding in every member). When the class genuinely disagrees about the effect, identification says so and names why, and the per-member route is still there: enumerate with `dags_of_cpdag` or `mags_of_pag` (or take one member with `consistent_extension` / `mag_of_pag`) and identify each, the IDA pattern. The [identification guide](identification.md) walks through both outcomes.

And the ingredients compose sideways: a warmed oracle across algorithms, a skeleton from one method restricting the next, a d-separation oracle standing in for data, your test inside unchanged PC.

## Not built yet

Selection bias end to end (FCI's rules R5 to R7 and undirected PAG edges), RFCI and the conservative and majority variants, the graphical lasso screen, the order-based searches (GRaSP, BOSS; the score contract is ready for them), permutation significance behind the same tests, per-pair test dispatch from the schema, and the time-series shapes.
