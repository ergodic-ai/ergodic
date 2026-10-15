# Evaluate a candidate graph

Turn a model's edge probabilities, or your data and a score, into a posterior over
DAGs, and read any graph you care about against it: how far it is from the best graph,
how much of the posterior the method puts on graphs it prefers, and how likely each of
the graph's claims is. You'll evaluate a decoded DAG, a CPDAG, a PAG and an ADMG, run
the sampler where the exact computation stops, and get a posterior from data with the
BGe score.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - An `EdgeProbabilities` object ([Work with edge probabilities](edge-probabilities.md)),
      or a table of data.

## Run the program

The matrix is the one from [Decode a DAG from edge probabilities](decode-a-graph.md):
four variables, independent marginals, and three edges above 0.5 that close a cycle.

```python file="evaluate_a_graph.py"
import numpy as np

from ergodic import EdgeProbabilities, dag, decode, evaluate_graph, graph_posterior

# 1. The same four variables as the decoder guide, read as independent marginals.
names = ("rain", "sprinkler", "wet", "slippery")
values = np.array(
    [
        [0.00, 0.45, 0.91, 0.06],
        [0.40, 0.00, 0.83, 0.05],
        [0.04, 0.07, 0.00, 0.88],
        [0.58, 0.02, 0.10, 0.00],
    ]
)
probs = EdgeProbabilities(names, values, kind="independent")

# 2. The posterior over DAGs under this model: exact, every DAG enumerated.
posterior = graph_posterior(probs)
print(posterior)
print(posterior.edge_probabilities().to_frame().round(3))

# 3. Evaluate the greedy decode.
decoded = decode(probs).graph
result = evaluate_graph(decoded, posterior)
print(result)
print(round(result.credible_level, 3), round(result.log_ratio_to_best, 3))
print(round(result.expected_shd, 2), round(result.expected_shd_between_draws, 2))
print(round(result.p_acyclic, 3))
print(result.claims_frame()[["claim", "probability"]].round(3))

# 4. A graph you might have drawn yourself, with slipperiness causing rain.
mine = dag(["rain -> wet", "sprinkler -> wet", "slippery -> rain"], nodes=names)
other = evaluate_graph(mine, posterior)
print(round(other.credible_level, 3), round(other.log_ratio_to_best, 2))
```

```output from="evaluate_a_graph.py"
GraphSamples(exact, pair_state, 4 variables, enumerated, 543 distinct DAGs)
target      rain  sprinkler    wet  slippery
source
rain       0.000      0.313  0.851     0.048
sprinkler  0.273      0.000  0.816     0.047
wet        0.005      0.010  0.000     0.756
slippery   0.198      0.007  0.024     0.000
GraphEvaluation(dag vs exact posterior, log-ratio to best 0, credible level 0.0929, 6 claims)
0.093 0.0
1.46 2.09
0.349
                                    claim  probability
0      no edge between rain and sprinkler        0.414
1       no edge between rain and slippery        0.754
2                        wet --> slippery        0.756
3                       sprinkler --> wet        0.816
4                            rain --> wet        0.851
5  no edge between sprinkler and slippery        0.946
0.517 -1.67
```

- **Step 2.** `graph_posterior` builds the posterior. With four variables it lists all
  543 DAGs with their exact probabilities. The edge marginals are posterior
  probabilities that each edge is in the DAG. They differ from the input: `slippery -->
  rain` drops from 0.58 to 0.20, because with `rain --> wet` and `wet --> slippery` it
  closes a cycle.
- **Step 3.** The greedy decode is the best graph here, so its log-ratio to the best
  is 0. Its credible level is 0.093: the posterior puts no mass on graphs it ranks
  higher, and the 0.093 is half the decode's own probability (ties count half). A
  draw from the posterior differs from it on 1.46 pairs on average, against 2.09
  between two draws, so the decode sits near the middle of the posterior. Under the
  raw matrix, independent coin flips form a DAG only 35% of the time.
- **Step 4.** Swapping `wet --> slippery` for `slippery --> rain` costs a factor of
  `exp(1.67)`, about 5, in posterior probability, and about half of the posterior
  mass sits on graphs the method ranks higher.

## What the numbers are

`graph_posterior` builds a distribution over DAGs from a method's output. From edge
probabilities, a DAG's weight is the product, over pairs of variables, of the
probability the model gives the state the DAG picks for that pair (no edge, one
direction or the other), times what your knowledge says. Independent marginals are
read as pair states first, as the decoders read them. For independent marginals this
is exactly the model's own product of coin flips, restricted to DAGs.

!!! note "Posterior probabilities under the method"
    Every probability on this page is a posterior probability under the method: it
    says how much the method's own numbers, with your knowledge, favor a graph or a
    claim. It's as good as the method's calibration, and a model's edge scores are
    usually not calibrated. A credible level of 0.05 says the method puts 5% of its
    mass on graphs it prefers; on its own it says nothing about how often such graphs
    are wrong.

    We checked this on 240 research graphs with known true DAGs, using pair-state
    matrices from Ergodic's research. The posterior's edge probabilities were well
    calibrated (expected calibration error 0.0037). The true DAG fell inside
    the 80%, 90% and 95% credible sets in 79%, 87% and 92% of the graphs, a little
    below nominal at the top. On the 12 twenty-variable graphs it fell inside them in
    only 50%, 58% and 67%. The sampler agreed with the exact computation (median
    largest pair-state difference 0.0015). These results cover those research
    matrices only. They don't cover AVICI or CDFM on the Welt API, or any other model.

**The normalizer cancels.** The posterior probability of a DAG is its weight divided
by the sum of every DAG's weight, `Z`. Every comparison here is a ratio of two weights
or a sum of probabilities, so `Z` drops out: the log-ratio to the best graph is the
difference of two log weights, and nothing on this page needs `Z` to be right.

**The credible level is a Bayesian tail probability.** It's the posterior mass of the
DAGs the method ranks strictly above the candidate, plus half the mass of DAGs it ranks
exactly level with it, the candidate included. That's the mid-point between the mass
above the candidate and the mass of the smallest highest-posterior set that contains
it. A level near 0 means the candidate is among the
method's favorite graphs; a level near 1 means almost all the mass sits on graphs it
prefers. It isn't a p-value. It tests nothing about the data, and a level near 0 doesn't
confirm the graph. The absolute probability of one DAG is tiny on any realistic number
of variables, so it isn't a headline number; for an exact posterior it's in
`diagnostics["candidate_probability"]`.

## Evaluate other kinds of graph

`evaluate_graph` reads a CPDAG, a PAG or an ADMG too. What each claims:

| Candidate | Claims | Also |
| --- | --- | --- |
| DAG | Every pair's state: `a --> b`, or no edge. | Log-ratio, credible level, expected SHD. |
| CPDAG | Adjacency for an undirected edge, the direction of a directed (compelled) edge, no edge for an absent pair. | The probability of the equivalence class. |
| PAG, MAG | Adjacency for each edge, no edge for each absent pair, and the marks read as ancestry: a tail at `a` on an edge to `b` claims `a` is an ancestor of `b`; an arrowhead at `b` claims `b` is not an ancestor of `a`; a circle claims nothing. | |
| ADMG | Each directed edge between observed variables. | `<->` and latent nodes are listed as not assessable. |

```python file="evaluate_kinds.py"
import numpy as np

from ergodic import EdgeProbabilities, admg, dag, evaluate_graph, graph_posterior
from ergodic.graph import cpdag_of_dag, mag_of_dag, pag_of_mag

names = ("rain", "sprinkler", "wet", "slippery")
values = np.array(
    [
        [0.00, 0.45, 0.91, 0.06],
        [0.40, 0.00, 0.83, 0.05],
        [0.04, 0.07, 0.00, 0.88],
        [0.58, 0.02, 0.10, 0.00],
    ]
)
posterior = graph_posterior(EdgeProbabilities(names, values, kind="independent"))
graph = dag(["rain -> wet", "sprinkler -> wet", "wet -> slippery"], nodes=names)

# 1. A CPDAG: adjacencies, compelled directions and the equivalence class.
cpdag = evaluate_graph(cpdag_of_dag(graph), posterior)
print(cpdag.claims_frame()[["claim", "kind", "probability"]].round(3))

# 2. A PAG: marks read as ancestry; circles claim nothing.
pag = pag_of_mag(mag_of_dag(graph))
print(pag)
for claim in evaluate_graph(pag, posterior).claims:
    if claim.kind in ("ancestor", "not_ancestor"):
        print(f"{claim.claim:<35} {claim.probability:.3f}")

# 3. An ADMG: directed edges are claims; a hidden common cause is not assessable.
confounded = admg(["rain -> wet", "sprinkler -> wet", "wet -> slippery", "rain <-> slippery"])
result = evaluate_graph(confounded, posterior)
for claim in result.claims:
    shown = "not assessable" if claim.probability is None else f"{claim.probability:.3f}"
    print(f"{claim.claim:<20} {shown}")
```

```output from="evaluate_kinds.py"
                                    claim               kind  probability
0    the graph's Markov equivalence class  equivalence_class        0.186
1      no edge between rain and sprinkler            no_edge        0.414
2       no edge between rain and slippery            no_edge        0.754
3                        wet --> slippery               edge        0.756
4                       sprinkler --> wet               edge        0.816
5                            rain --> wet               edge        0.851
6  no edge between sprinkler and slippery            no_edge        0.946
MixedGraph(kind=pag, nodes=4, edges=[rain o-> wet, slippery <-- wet, sprinkler o-> wet])
wet is not an ancestor of rain      0.935
slippery is not an ancestor of wet  0.860
wet is an ancestor of slippery      0.756
wet is not an ancestor of sprinkler 0.983
rain <-> slippery    not assessable
rain --> wet         0.851
wet --> slippery     0.756
sprinkler --> wet    0.816
```

- **Step 1.** The collider at `wet` makes every edge compelled, so the CPDAG claims the
  same edges as the DAG, plus its class, which holds 19% of the posterior.
- **Step 2.** The PAG of this DAG leaves circles at `rain` and `sprinkler`, so it
  doesn't claim either causes `wet`, only that `wet` causes neither of them. The tail
  at `wet` on `wet --> slippery` claims `wet` is an ancestor of `slippery`.
- **Step 3.** A posterior over DAGs of the observed variables has no state for a hidden
  common cause, so `rain <-> slippery` gets no number.

A candidate's variables must be exactly the posterior's (an ADMG may add latent
nodes). A `mixed`-kind graph is read only when it's a DAG; tag anything else with
`as_kind`.

## How it's computed

`graph_posterior` is exact when it can be:

- **Every DAG enumerated** up to four variables (five with a score).
- **The exact subset recursion** when the weight factorizes by pair, which holds for
  edge probabilities unless an order statement could be broken by a path (see below).
  A dynamic program over subsets of the variables, about `3^n` steps, gives the exact
  normalizer, the exact edge marginals, exact draws (`samples`, 10,000 by default) and
  the exact best graph. It runs up to `max_exact` variables, 15 by default. On a
  laptop that took about 2 s at 10 variables, 2.5 s at 12 and 7 s at 15, mostly the
  draws.
- **Structure MCMC** otherwise: four chains, each proposing to change one pair's state
  at a time (add, delete or reverse an edge) and rejecting any change that closes a
  cycle or breaks the knowledge. One chain starts from the greedy decode, the others
  from random DAGs. Each chain also runs three flattened copies of the posterior (its
  powers in `temperatures`, 0.6, 0.36 and 0.22 at 10 variables and closer to 1 on more)
  that trade graphs with it, so it can cross between modes that differ in several edges
  at once; only the unflattened copy's draws are kept. On 20 variables the defaults took
  about 14 s, and about 3 s on 10.

Pass `method="exact"` or `method="mcmc"` to choose. Exact pair-state claims carry no
Monte Carlo error. Claims that aren't about one pair, the credible level and
`samples.probability(feature)` come from the draws, with a Monte Carlo error.

MCMC reports the rank-normalized split R-hat and the bulk effective sample size of the
log weight and of every edge indicator in `diagnostics`, and warns with
`PosteriorConvergenceWarning` when the log weight's R-hat is above 1.01, its effective
sample size is below 100 per chain, or an edge with a posterior probability between
0.01 and 0.99 has an R-hat above 1.05. Raise `samples`, `warmup` or `thin` when it
does.

```python file="posterior_mcmc.py"
import numpy as np

from ergodic import DomainKnowledge, EdgeProbabilities, evaluate_graph, graph_posterior

# 1. Edge probabilities over six variables, as a model might return them.
rng = np.random.default_rng(1)
names = ("a", "b", "c", "d", "e", "f")
values = np.zeros((6, 6))
for i in range(6):
    for j in range(i + 1, 6):
        _, forward, backward = rng.dirichlet([2.0, 1.0, 0.5])
        values[i, j], values[j, i] = forward, backward
probs = EdgeProbabilities(names, values, kind="pair_state")

# 2. Without knowledge the target factorizes by pair: exact draws.
exact = graph_posterior(probs, seed=0)
print(exact.method, exact.diagnostics["engine"], exact.n_draws)

# 3. "a before f" with e and b free to sit anywhere is an ancestral constraint: MCMC.
knowledge = DomainKnowledge().order("a", "f").forbid("b", "c", confidence=0.9)
chained = graph_posterior(probs, knowledge=knowledge, seed=0)
diagnostics = chained.diagnostics
print(chained.method, diagnostics["engine"], diagnostics["converged"])
print(round(diagnostics["rhat_log_target"], 2), diagnostics["ess_log_target"] > 2000)
print(diagnostics["target"]["ancestral_constraints"])

# 4. A feature that is not one edge: is a an ancestor of f?
feature = chained.probability(lambda graph: "a" in graph.ancestors("f"))
print(round(feature.probability, 2), feature.mc_error < 0.02)

# 5. The best graph found, and its claims.
result = evaluate_graph(chained.best_graph, chained)
print(result.best_source, round(result.log_ratio_to_best, 3), round(result.credible_level, 3))
```

```output from="posterior_mcmc.py" tol="0.02"
exact subset_recursion 10000
mcmc structure_mcmc True
1.0 True
[('a', 'f')]
0.32 True
greedy_decode 0.0 0.002
```

## Use domain knowledge

`graph_posterior(..., knowledge=...)` reads the same statements as
[the decoders](decode-a-graph.md#use-domain-knowledge), with one difference in how
order is read.

- **Hard statements** restrict the DAGs: forbidden and required edges, roots, sinks,
  tiers and `restrict_to`.
- **Order is ancestral.** `order("a", "b")` means `b` is not an ancestor of `a`, as
  `DomainKnowledge.violations` reads it. Edges against the order are removed pair by
  pair. When some path of allowed edges could still run from `b` back to `a`, the
  statement stays a constraint on paths, which no pair-by-pair weight can express, so
  the exact recursion doesn't apply and MCMC runs. Tiers that place every variable
  never need that. The decoders are stricter: they ask for one time order that fits
  every statement at once.
- **Soft statements** weigh each pair's states as they do for the decoders, so a DAG's
  log weight is the pair log score plus `knowledge.log_prior(graph)` (of the knowledge
  without its `confound` statements; see below).
- **`confound`** speaks about no directed edge, so a posterior over DAGs ignores it and
  lists it in `diagnostics["ignored_confounders"]`.

## Get a posterior from data with a score

`graph_posterior(data, score=...)` uses a decomposable score instead: a DAG's weight is
`exp(score(DAG))`, with a prior that is uniform over DAGs times your knowledge. BDeu
(discrete data) and BGe (linear-Gaussian data) are log marginal likelihoods, so the
result is a Bayesian posterior under their priors. BIC only approximates one, and the
result says so in `diagnostics["approximate"]`. Without `score=`, all-continuous data
gets `"bge"` and all-discrete data `"bdeu"`.

```python file="score_posterior.py"
import numpy as np

from ergodic import dag, evaluate_graph, graph_posterior, tabular
from ergodic.graph import cpdag_of_dag

# 1. Linear-Gaussian data from a known DAG: x -> y -> w <- z.
rng = np.random.default_rng(0)
x = rng.normal(size=300)
y = 0.8 * x + rng.normal(size=300)
z = rng.normal(size=300)
w = 0.7 * y - 0.6 * z + rng.normal(size=300)
data = tabular({"x": x, "y": y, "z": z, "w": w})
truth = dag(["x -> y", "y -> w", "z -> w"])

# 2. The BGe posterior: exact, every DAG enumerated.
posterior = graph_posterior(data, score="bge")
print(posterior)
for graph, weight in posterior.top(3):
    print(round(weight, 3), graph)

# 3. The true DAG ties with its Markov-equivalent twin; the class holds most mass.
result = evaluate_graph(truth, posterior)
print(round(result.credible_level, 3), round(result.mass_above, 3))
print(round(evaluate_graph(cpdag_of_dag(truth), posterior).class_probability.probability, 3))

# 4. BIC only approximates the marginal likelihood, and the result says so.
approximate = graph_posterior(data, score="bic")
print(approximate.diagnostics["approximate"], approximate.diagnostics["note"])
```

```output from="score_posterior.py" tol="0.002"
GraphSamples(exact, score:bge, 4 variables, enumerated, 543 distinct DAGs)
0.401 MixedGraph(kind=dag, nodes=4, edges=[w <-- y, w <-- z, x --> y])
0.401 MixedGraph(kind=dag, nodes=4, edges=[w <-- y, w <-- z, x <-- y])
0.014 MixedGraph(kind=dag, nodes=4, edges=[w <-- y, w <-- z, x --> y, y --> z])
0.401 0.0
0.803
True the score is a penalized likelihood, not a marginal likelihood, so this posterior is an approximation
```

- **Step 2.** The two best DAGs differ only in the direction of `x --- y`. They're
  Markov equivalent, so BGe scores them the same, and the data can't tell them apart.
- **Step 3.** Nothing ranks above the true DAG, and its twin ties with it, so its
  credible level is half their mass. With a score that can't tell equivalent DAGs apart,
  read the CPDAG's class probability instead: 0.80 here.

BGe (Geiger and Heckerman, with the correction of Kuipers, Moffa and Heckerman) uses
`alpha_mu = 1`, `alpha_w = n + 2` for `n` variables and a scale matrix `t I`,
`t = alpha_mu (alpha_w - n - 1) / (alpha_mu + 1)`, as BiDAG does. The prior's mean is
the data's own mean, an empirical-Bayes choice: the score is the exact marginal
likelihood under a prior that depends on the data. Equivalent DAGs still score the
same. BiDAG centres the prior at zero, which agrees on centred data. `t` is in your
data's units, so standardize columns whose scales differ by orders
of magnitude. BGe is also a score for the searches: `discover(data, "ges",
score="bge")`. A score posterior is exact by enumeration up to five variables and runs
MCMC above that.

## Average an effect over the graphs

A posterior over DAGs carries straight through to effects. `adjustment_probability`
is the posterior probability that a set is a valid adjustment set, by ergodic's
adjustment criterion. `average_effect` asks each DAG which adjustment set identifies
the effect (the one `identify_effect` would use), groups the DAGs by set, estimates the
effect once per set with ergodic's estimators, and mixes the estimates by posterior
mass.

```python file="average_over_graphs.py"
import numpy as np

from ergodic import (
    EdgeProbabilities,
    adjustment_probability,
    average_effect,
    graph_posterior,
    tabular,
)

# 1. Data where c confounds a binary treatment t and the outcome y; the effect is 2.
rng = np.random.default_rng(0)
c = rng.normal(size=2000)
t = (0.9 * c + rng.normal(size=2000) > 0).astype(float)
y = 2.0 * t + 1.5 * c + rng.normal(size=2000)
data = tabular({"c": c, "t": t, "y": y})

# 2. A model's pair-state probabilities: sure about t -> y and c -> y, unsure about c, t.
names = ("c", "t", "y")
values = np.array(
    [
        [0.00, 0.60, 0.95],
        [0.15, 0.00, 0.90],
        [0.02, 0.05, 0.00],
    ]
)
posterior = graph_posterior(EdgeProbabilities(names, values, kind="pair_state"))

# 3. How likely is each adjustment set to be valid?
print(round(adjustment_probability(posterior, "t", "y", ["c"]).probability, 3))
print(round(adjustment_probability(posterior, "t", "y", []).probability, 3))

# 4. The effect, estimated once per adjustment set and averaged by posterior mass.
result = average_effect(posterior, data, "t", "y")
print(result.to_frame()[["adjustment_set", "mass", "n_dags", "estimate", "std_error"]].round(3))
print(round(result.mean, 3), round(result.std, 3))
print(round(result.zero_effect_mass, 3), result.unidentified_mass, result.truncated_mass)
```

```output from="average_over_graphs.py"
0.811
0.404
       adjustment_set   mass  n_dags  estimate  std_error
0                   c  0.757       3     1.969      0.066
1             (empty)  0.156       6     3.563      0.072
2  (no directed path)  0.087      16     0.000      0.000
2.046 0.857
0.087 0.0 0.0
```

- **Step 3.** `{c}` is valid unless `t --> c`, which makes `c` a mediator. Adjusting
  for nothing is valid when no back-door path runs through `c`; here `c --> y` is
  nearly certain, so that means no edge from `c` into `t`.
- **Step 4.** Three DAGs adjust for `c` and estimate about 2. Six adjust for nothing
  and pick up the confounding. In the rest no directed path runs from `t` to `y`,
  so the effect is 0 there. The mixture mean weighs the three by mass. Its standard
  deviation adds the spread between the groups to their own standard errors (the law
  of total variance), so it says how much the answer depends on the graph.

Mass is kept apart in three ways. DAGs with no directed path enter the mixture at 0
and are reported as `zero_effect_mass`. If your data lacks a variable of the posterior,
it's treated as hidden, and DAGs where no adjustment set over your columns is valid are
`unidentified_mass`. Only the `max_sets` heaviest sets (20 by default) are estimated;
the rest are `truncated_mass`. The mixture covers `identified_mass` and leaves the
other two out, never averaging them in. The default estimator is the doubly-robust
AIPW, which needs a binary treatment and raises otherwise; pass `strategy="dml"` for a
continuous one, and `rule="minimal"` to adjust for a minimal set instead.

`average_effect` also reports `mean_mc_error`, the mean's Monte Carlo error from the
posterior masses (0 when every DAG was enumerated), and `evaluate_graph` reports
`expected_shd_error`. When every draw agrees, for example nothing ranks above the
candidate, an error of 0 would claim too much, so the error is the rule-of-three bound
`3 / draws` instead and `bound` is `True` on that `FeatureProbability`. Under the exact
recursion the candidate's own share of the credible level is exact. The between-draws
SHD baseline is a plug-in without an error.

## Read the result

`graph_posterior` returns `GraphSamples`:

| Attribute or method | What it holds |
| --- | --- |
| `method` | `"exact"` or `"mcmc"`. |
| `adjacency`, `weights` | The distinct DAGs (boolean matrices) and their weights: exact probabilities when enumerated, else their share of the draws. |
| `log_targets`, `log_target(graph)` | A DAG's log weight. Only differences matter. |
| `exact_probabilities`, `log_normalizer` | Exact probabilities and `log Z`, when exact. |
| `edge_probabilities()`, `edge_errors()` | Posterior edge marginals, as pair states, and their Monte Carlo errors. |
| `probability(feature)` | The posterior probability that `feature(graph)` holds, with its Monte Carlo error. |
| `best_graph`, `top(k)` | The best DAG found, and the heaviest DAGs. |
| `diagnostics` | `engine`, `edge_estimator`, `best_source`, and for MCMC `rhat_log_target`, `ess_log_target`, `max_edge_rhat`, `min_edge_ess`, `acceptance`, `swap_acceptance`, `converged`. |

Under MCMC on edge probabilities, the edge marginals average each draw's chance of
each pair state given the rest of the graph (Rao-Blackwellized), which is less noisy
than counting.

`evaluate_graph` returns `GraphEvaluation`:

| Attribute | What it holds |
| --- | --- |
| `log_ratio_to_best`, `best_graph`, `best_source` | DAGs only. The best graph is the exact best when exact; otherwise the better of the greedy decode and the best draw, then improved one pair at a time while that helps. |
| `credible_level`, `credible_level_error`, `mass_above` | DAGs only. Ties count half; graphs whose log weights differ by under `1e-9` (relative) tie. |
| `claims`, `claims_frame()` | One `GraphClaim` per statement: text, kind, probability, Monte Carlo error. |
| `class_probability` | CPDAGs only. |
| `expected_shd`, `expected_shd_between_draws` | Expected count of pairs whose state differs from a draw (DAGs only), and between two draws. |
| `p_acyclic`, `p_acyclic_error` | Independent marginals only: exact up to `max_exact` variables, else from 20,000 draws. |
| `not_assessable` | Hidden common causes the DAG posterior can't price. |

Nothing touches numpy's global random state, and the same `seed` gives the same draws.

## Next steps

<div class="cards" markdown="1">

- [Decode a DAG from edge probabilities](decode-a-graph.md): pick one DAG instead.
- [Add domain knowledge](domain-knowledge.md): every kind of statement.
- [Choose a method and test](choose-a-method.md): the scores, BGe among them.
- [Estimate an average effect](estimate-average-effect.md): one graph, one estimate.
- [Python reference](reference-discovery.md): `graph_posterior`, `evaluate_graph`, `GraphSamples`.

</div>
