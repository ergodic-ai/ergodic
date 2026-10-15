# Causal discovery

You have data, and a question the data alone won't answer: which variables cause which? Discovery learns that structure, the graph itself, from the data.

Two ingredients drive every search: a conditional independence test for the constraint-based algorithms (PC, FCI, MMPC), and a decomposable score for the score-based ones (GES, hill climbing, exact A*). The algorithms are thin consumers; the ingredients are objects you instantiate, configure, share, and replace.

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

Data is supplied in exactly one place. An unbound test gets its data at `fit`. A bound test already has data, so `fit` can omit it. Passing a bound test a different dataset raises; passing it the same data is fine.

Alpha lives in the algorithm, not the test, so the p-value cache is alpha-independent. One warmed oracle serves many runs:

```python
linear = Linear(data)
strict = PC(test=linear, alpha=0.01).fit()
loose = PC(test=linear, alpha=0.10).fit()   # every p-value comes from the cache
```

## When triples disagree: conservative and majority colliders

This is a refinement for messy real data; on a first read you can skip to [The tests](#the-tests). Standard PC decides each v-structure from one separating set, the first one the skeleton phase happened to record. On real data different sets can disagree, and a collider decided by a coin flip of search order is how PC earns its reputation for flakiness. The conservative and majority rules re-test each unshielded triple against every subset of both endpoints' neighborhoods and vote:

```python
result = discover(data, method="pc", colliders="majority")
result.diagnostics["n_ambiguous_triples"]   # 0.0 on this clean chain
```

`"conservative"` (Ramsey, Zhang, and Spirtes 2006) orients a collider only when the middle node sits in no separating set at all, and calls the triple ambiguous on any disagreement. `"majority"` (Colombo and Maathuis 2014) lets the fraction decide, with an exact half ambiguous. Ambiguous triples are never oriented as colliders, and the Meek rules that would read them as definite non-colliders skip them too, so the ambiguity propagates honestly instead of hardening into an arrow. When any triple stays ambiguous the result may not be a valid CPDAG; it then keeps the `mixed` face with `cpdag_valid` 0.0, the same convention knowledge conflicts use. Under a perfect oracle on a faithful world no triple is ever ambiguous and all three settings agree, which the test suite holds executably.

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
PC(test=GCM(data, learner=RandomForestRegressor())).fit()
```

The data goes in one place: a test built with data is bound, so `fit()` takes none. To pass the data at fit (or through `discover`), build the test unbound, as in `PC(test=GCM(learner=RandomForestRegressor())).fit(data)`.

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

The run is PC's stable skeleton, then the possible-d-sep phase (with latents, a separating set can live outside both neighbourhoods, so candidate sets widen), then re-orientation and Zhang's complete rule closure, all ten rules. With a perfect oracle the output is exactly the canonical PAG of the truth, `pag_of_mag(mag_of_dag(truth))`, and the test suite holds it to that identity across random graphs, with and without selection. `max_cond_size` caps both testing phases, and the same warmed oracle that served PC serves FCI from cache.

Selection bias needs nothing extra from you, because it lives in the world, not the algorithm: data sampled under selection simply answers every independence test conditional on it. What changes is what the marks can say. An undirected edge in the output states that both endpoints cause the selection in every member of the class, and a `-o` edge that one end is settled while the other stays open. The classic worry, two independent causes of admission looking dependent in the admitted sample, lands as an `o-o` edge: real dependence, source honestly undecided. With richer structure the selection rules R5 to R7 pin the marks exactly; the chordless cycle below is the textbook case, here read off a ground-truth oracle:

```python
from ergodic import dag
from ergodic.discovery import DSeparationOracle, FCI

world = dag(
    ["A -> S1", "B -> S1", "B -> S2", "C -> S2",
     "C -> S3", "D -> S3", "D -> S4", "A -> S4"],
    selection=["S1", "S2", "S3", "S4"],
)
FCI(test=DSeparationOracle(world)).fit().graph
# the PAG: A --- B --- C --- D --- A, every mark an invariant tail
```

One consumer refuses on purpose: identification raises on a class whose marks say selection is there, because the generalized adjustment criterion assumes none, and a causal effect is generally not identifiable from a selection-biased margin.

Knowledge translates through ancestry, not edge direction. `require(u, v)` orients `u -> v`; `confound(a, b)` orients `a <-> b` (a confounded pair is read as a pure hidden common cause); a tier pins only an arrowhead at the later end, because a later variable cannot cause an earlier one while confounding stays possible. Directional forbids, roots, and sinks fix no single mark in a latent world, so they only validate. Knowledge-seeded marks are not class invariants: the result then keeps the `mixed` face and `diagnostics["pag_valid"]` is `0.0`.

### RFCI, the fast relaxation

FCI's cost center is the possible-d-sep phase: with latents, a separating set can hide anywhere, so FCI re-tests every adjacency against subsets of a widened candidate pool. RFCI (Colombo, Maathuis, Kalisch, and Richardson 2012) skips that phase entirely and only ever conditions on subsets of neighborhoods and recorded separating sets, paying with a weaker but still honest output:

```python
discover(data, method="rfci", alpha=0.01)   # same call shape as fci
```

The trade is precise. Extra independence tests on each unshielded triple and along each discriminating path catch most of what possible-d-sep would have caught (and can remove edges of their own); what survives is an RFCI-PAG whose skeleton may keep an edge the FCI PAG removes. Every arrowhead and tail it does show is a sound ancestral statement about the true MAG, and the test suite holds that soundness executably against random latent, confounded, and selection worlds, alongside exact agreement with FCI wherever possible-d-sep has nothing to do. One reading note: an RFCI-PAG can stand for more than one Markov equivalence class, so a `mixed` face with `pag_valid` 0.0 is an expected honest outcome on hard worlds, not an error.

## The skeleton screens

MMPC (max-min parents and children) is a constraint search that stops at the skeleton: for each target it keeps the variables no subset of the candidate set can separate from it, then keeps an adjacency only when it holds from both sides. It is the cheap screen you run before an expensive search, and its output is built to feed `restrict_to`:

```python
screen = discover(data, method="mmpc")
support = DomainKnowledge().restrict_to(screen.graph)
discover(data, method="astar", knowledge=support)   # exact search inside the screen
```

That composition is the classic MMHC hybrid, recovered here as two objects instead of one algorithm. A score search inside the screen loses nothing (scores need no separating sets); PC inside the screen keeps the skeleton but can orient less, because a pruned pair is never tested and so never witnesses a v-structure.

The graphical lasso is the second screen, for continuous data: a sparse precision matrix estimated by the Friedman, Hastie, and Tibshirani block coordinate descent, with the penalty picked by EBIC over a warm-started path. A precision zero is a conditional independence under Gaussianity, so the nonzeros are a skeleton source, not causal structure, and the result says so by coming back undirected on the `mixed` face:

```python
screen = discover(data, method="glasso")    # EBIC picks the penalty
screen.diagnostics["lam"]                   # the chosen penalty rides along
discover(data, method="ges", knowledge=DomainKnowledge().restrict_to(screen.graph))
```

A screen is judged on recall, and that is the honest framing for this one: on strongly correlated data the penalized precision keeps extra edges at every penalty (a known property of the estimator, not of this implementation, which matches sklearn entry for entry), and the downstream search prunes them. Adjacency knowledge translates directly: a banned pair gets an infinite penalty, a required pair an unpenalized one.

## How stable is that edge?

One run returns one graph and no sense of how fragile it is. The bootstrap wrapper reruns any method on row resamples and counts what survives:

```python
from ergodic.discovery import bootstrap

# `data` is any dataset from above; the FCI example's A, B, C is in scope here
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

Five searches consume the same object:

```python
from ergodic.discovery import BOSS, GES, GRaSP, HillClimb

GES(score=bic).fit()         # greedy over equivalence classes, returns a CPDAG
HillClimb(score=bic).fit()   # add/remove/reverse over DAGs, with a tabu list
AStar(score=bic).fit()       # the exact optimum, honest about scale
BOSS(score=bic).fit()        # best-order search over permutations, a CPDAG
GRaSP(score=bic).fit()       # permutation search by tucks, a CPDAG
```

`GES` (Chickering 2002) is the standard: a forward pass inserts the best edge into the equivalence class until nothing improves, a backward pass deletes, and the result is a CPDAG directly. In the large-sample limit, with a score-equivalent score, it finds the optimal class; on real data the exact search is its referee, and the test suite holds GES to the A* optimum. `HillClimb` is the fast baseline: steepest ascent over DAGs with a tabu ring to escape plateaus, no optimality story. `AStar` is the Yuan and Malone shortest path over the order lattice: exact, not greedy, with a node limit raised with a pointer at `max_parents` and `restrict_to`. All five consume knowledge the same four ways, and a DAG result compares with PC through `cpdag_of_dag`.

`BOSS` and `GRaSP` are the order-based pair, the current accuracy state of the art among greedy searches. Both read a permutation of the variables through the same projection: each node takes the parents that grow-shrink picks from its predecessors, and the permutation's score is the sum of those locals. `BOSS` (Andrews and colleagues 2023) moves one variable at a time to its best position and sweeps until nothing improves, then runs the backward phase of GES on the result (`use_bes=False` turns that off); `GRaSP` (Lam, Andrews, and Ramsey 2022) walks tuck operations depth-first. Both are held to the A* optimum in the test suite, the same referee GES answers to. Under heavy domain knowledge prefer `BOSS`: tucks only move existing edges, so a tier-gated variable can strand, which the GRaSP docstring documents.

A custom score is one method, mirroring the tests: implement `local_from_arrays(y, X)` for numeric data, or declare `needs = "codes"` and implement `local_from_codes(y, y_card, zs, z_cards)` for native categorical scoring.

```python
from ergodic.discovery import Score

class MyScore(Score):
    name = "my_score"

    def local_from_arrays(self, y, X):
        ...
        return value
```

## When the noise tells the direction: DirectLiNGAM

Everything above stops at the equivalence class, because observational data under Gaussian noise cannot tell `X -> Y` from `Y -> X`. Non-Gaussian noise can. In a linear model with independent non-Gaussian disturbances the direction is identifiable, and DirectLiNGAM (Shimizu and colleagues 2011) finds it directly: the most exogenous variable is the one whose regression residuals look most independent of it, measured by the Hyvarinen and Smith pairwise likelihood ratio; regress it out, repeat on the residuals, and the causal order falls out. A backward BIC pass then prunes the dense order down to a DAG.

```python
rng = np.random.default_rng(0)
n = 3000
x = rng.uniform(-1, 1, size=n)              # uniform noise: non-Gaussian
m = 0.8 * x + rng.uniform(-1, 1, size=n)
y = 0.7 * m + rng.uniform(-1, 1, size=n)
data = tabular(pd.DataFrame({"X": x, "M": m, "Y": y}))

discover(data, method="lingam").graph        # the DAG: X -> M -> Y, fully oriented
```

The output is a full DAG, the only family here that orients inside the equivalence class. The assumptions are the price: linearity, no hidden confounders, and genuinely non-Gaussian noise. On Gaussian data the model is not identifiable and the returned order is arbitrary, which the docstring says plainly. It consumes neither a test nor a score (the regression and residual machinery is the algorithm), takes continuous columns only, and reads knowledge through the order: tiers and declared roots gate the root choice, forbidden and required edges act at pruning, and a required edge that contradicts the data-driven order raises instead of silently bending.

## Time series and panels

Every method above assumes exchangeable rows: shuffle the data and nothing changes. In a time series the row order is the signal, so the temporal shapes get their own method, PCMCI+ (Runge 2020), reached through the same facade. It reads the series through a lagged window: row t of an internal frame holds every variable at lags 0 through `max_lag`, in columns named like `X[t-1]`, and the same CI tests run on that frame unchanged (`test="auto"` reads its schema, and `test="gcm"` works here too).

```python
import numpy as np
import pandas as pd
from ergodic import timeseries, discover

rng = np.random.default_rng(0)
n = 500
x, y = np.zeros(n), np.zeros(n)
for t in range(1, n):
    x[t] = 0.7 * x[t - 1] + rng.normal()
    y[t] = 0.8 * x[t - 1] + 0.5 * y[t - 1] + rng.normal()
ts = timeseries(pd.DataFrame({"t": np.arange(n), "X": x, "Y": y}), time="t")

result = discover(ts, method="pcmci", max_lag=2)
result        # TimeSeriesDiscoveryResult(pcmci/linear: dag on 6 window nodes, ...)
result.graph  # the window graph: X[t-1] -> X[t], Y[t-1] -> Y[t], X[t-1] -> Y[t]
```

The output lives on window nodes, one per variable per lag. Lagged edges point forward in time for free, since a cause precedes its effect; the work is in deciding which lagged links survive conditioning. Contemporaneous edges, lag zero at both ends, have no time order to lean on, so they orient the way PC orients: colliders decide what they can, two within-slice Meek rules propagate, and an edge nothing decides stays undirected, dropping the graph to the mixed face.

The result is a `TimeSeriesDiscoveryResult`, and `links` carries the evidence: a p-value per surviving link, keyed `(source, target, lag)`.

```python
sorted(result.links)         # [('X', 'X', 1), ('X', 'Y', 1), ('Y', 'Y', 1)]
result.links[("X", "Y", 1)]  # the p-value of the test that kept X[t-1] -> Y[t]
result.summary_edges()       # {('X', 'X'), ('X', 'Y'), ('Y', 'Y')}
```

There is deliberately no summary graph. A stationary process happily runs feedback across lags, X driving Y at lag 1 while Y drives X at lag 2, and a variable-level graph would need two opposed edges between one pair to say so; the one-edge-per-pair graph model isn't bent around that. The window graph is the primary object, and `summary_edges()` gives the flat variable-level pairs when that's all you need.

Panels run through the same call. `discover(panel_data, method="pcmci", max_lag=2)` builds windows within each entity and stacks them, pooling the entities under the assumption that they all follow one process; entities with fewer than `max_lag + 2` rows are skipped with a warning.

Choose `max_lag` from what you know about the process, because the window is the universe: a true effect at lag `max_lag + 1` is invisible, since no test ever sees the column that would carry it. And lags count rows, so irregular sampling earns a warning; a lag of one means the previous row, whatever the calendar gap.

`granger` is the predictive baseline, on a `TimeSeriesData` with continuous variables:

```python
from ergodic import granger

g = granger(ts, max_lag=2)
g.pvalues[("X", "Y")]   # tiny: X's lags improve the forecast of Y
g.pvalues[("Y", "X")]   # ~0.4: Y's lags add nothing to X
g.significant           # {('X', 'Y')}
```

Granger answers a forecasting question: do X's lags improve the prediction of Y once Y's own lags and everyone else's are in the regression? That's worth knowing, and it is on purpose that the result is a table of ordered pairs and never a graph. The test is blind to confounding (a common driver makes X "Granger-cause" Y with no causal link between them) and to contemporaneous effects, so drawing it as a graph would claim more than it measures. Treat the table as a screen and use PCMCI for structure.

The shapes refuse to cross, with the reason in the error. An iid method handed temporal data raises, and PCMCI rejects flat tables in the other direction:

```python
discover(ts, method="pc")
# TypeError: TimeSeriesData rows are not exchangeable, so an iid method would
# misread them; use method='pcmci' for time series and panels, ...
```

The stability `bootstrap` reads the temporal shapes through a design that respects the dependence. Plain row resampling would break exactly what PCMCI measures, so a time series resamples moving blocks instead, and the trick is that each drawn block becomes an entity of a panel: PCMCI pools lag windows within entities, so no window ever crosses a block seam, and glued blocks never manufacture a fake adjacency. A panel resamples whole entities (the same cluster bootstrap the panel designs use). `block_length` overrides the default (roughly the square root of the series length, floored to out-span the lag depth), and the frequencies come back over window-graph glyphs:

```python
stability = bootstrap(ts, method="pcmci", n_bootstrap=50, max_lag=2, seed=0)
stability.marks                    # {"X[t-1] --> Y[t]": 1.0, ...}
```

The honesty note rides along: block resampling treats the series as stationary at the block scale.

Background knowledge enters PCMCI with variables read as processes, so a constraint on a pair applies at every lag, contemporaneous included. A directional forbid kills the link at all lags and the lag-zero orientation; an adjacency ban removes the pair; tiers prune later-to-earlier at every lag and orient a real contemporaneous edge from the earlier tier to the later; roots and sinks forbid cross-variable links while self-memory (`X[t-1] -> X[t]`) survives. Two constraints stay honest about not translating: `require` names no lag, so it validates after the run (no surviving lag raises), and `confound` has no analogue in a window graph and raises. `diagnostics["n_knowledge_pruned"]` counts what knowledge removed.

## Putting the pieces together

The pillars compose end to end, and you do not have to pick a member of the class to do it. Identification accepts the CPDAG or PAG directly and answers with the generalized adjustment criterion, so a success holds in every graph the class represents, and `estimate_effect` rides on that:

```python
from ergodic import estimate_effect

result = discover(data, method="pc", knowledge=dk)
est = estimate_effect(result.graph.as_kind("dag"), data, "X", "Y", strategy="dml")
```

The `as_kind("dag")` is there because of the knowledge. The tiers `X < M < Y` orient every edge, including ones the data alone leaves open, which takes the result out of the CPDAG class: it keeps the `mixed` face, and identification refuses a `mixed` graph until you re-face it as the DAG it is. When some edges stay undirected, the error names the options.

The same line works on an FCI result: the PAG may keep circles where confounding cannot be ruled out, and the effect still estimates whenever the treatment's outgoing edge is visible (certified clean of hidden confounding in every member). When the class genuinely disagrees about the effect, identification says so and names why, and the per-member route is still there: enumerate with `dags_of_cpdag` or `mags_of_pag` (or take one member with `consistent_extension` / `mag_of_pag`) and identify each, the IDA pattern. The [identification guide](identification.md) walks through both outcomes.

And the ingredients compose sideways: a warmed oracle across algorithms, a skeleton from one method restricting the next, a d-separation oracle standing in for data, your test inside unchanged PC.

## Not built yet

The temporal cousins of the iid algorithms (VAR-LiNGAM, tsFCI, LPCMCI), permutation significance behind the same tests, and per-pair test dispatch from the schema.
