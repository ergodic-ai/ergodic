# Structural causal models

A structural causal model writes every variable as an equation: its value is a function of its causes plus its own noise. `score = 2 * hours + 1 * sleep + U`. Fit those equations and you can do three things a regression cannot. Sample new worlds. Intervene and read off what changes, severing the variable from its causes rather than conditioning on it. And ask a counterfactual: this student scored 3.5; what would they have scored had they studied twice as long, holding everything else about them fixed.

`ergodic.scm` fits that model and answers those queries. Two choices set it apart from the usual structural-equation tooling. It is Bayesian: every mechanism carries a posterior, so every answer comes with calibrated uncertainty by default. And it is honest about identification. Before it returns an effect, it asks whether the data determine that effect. When they do, you get a number with a credible interval. When they don't, you get the effect as a function of the assumption you would have to make, and there is no way to read a naked point off it. That refusal is the point of the pillar.

PyMC runs the sampling. It is an optional dependency: the bare install never imports it, and `import ergodic.scm` stays cheap until you fit. Install the extra with `pip install ergodic[scm]`.

## Building one

An `SCM` is a graph plus a fitted mechanism per node. Hand it a `MixedGraph` (a DAG for the MVP) and a `Dataset`, and the mechanisms are assigned by the schema, the same dispatch the rest of the package uses to read a column. A continuous node gets a Bayesian-linear Gaussian-additive equation: a linear mean over its parents with a prior on the coefficients and a half-normal prior on the noise scale. A binary or categorical node gets a Gumbel-max mechanism, the structural form of a logistic or softmax. It samples a category in the generative path and inverts to recover the noise in a counterfactual, so a binary or categorical node is a full citizen of every query.

Here is a world where the truth is planted. Study hours and sleep both raise the exam score, and hours raises it by 2.0 per standard deviation.

```python
import numpy as np
import pandas as pd
from ergodic import SCM, dag, tabular

rng = np.random.default_rng(0)
n = 2000
hours = rng.normal(0.0, 1.0, size=n)
sleep = rng.normal(0.0, 1.0, size=n)
score = 2.0 * hours + 1.0 * sleep + rng.normal(0.0, 1.0, size=n)
data = tabular(pd.DataFrame({"hours": hours, "sleep": sleep, "score": score}))

g = dag(["hours -> score", "sleep -> score"])
scm = SCM(g, data, draws=300, tune=300, chains=2, random_state=0)
scm   # SCM(3 nodes, unfitted, n=2000)
```

`draws`, `tune`, and `chains` set the per-node sampler; `random_state` makes a seeded run reproducible. Nothing has sampled yet. The mechanisms are chosen and waiting:

```python
{k: type(v).__name__ for k, v in scm.mechanisms.items()}
# {'hours': 'GaussianAdditive', 'sleep': 'GaussianAdditive', 'score': 'GaussianAdditive'}
```

To override one, pass `mechanisms={node: spec}`. In the MVP a spec is a built-in name (`"linear"`, `"gaussian"`, or `"ols"` for the additive Gaussian; `"binary"` or `"logistic"`; `"categorical"` or `"softmax"`) or a `Mechanism` instance with its priors set the way you want:

```python
from ergodic.scm import GaussianAdditive

SCM(g, data, mechanisms={"score": GaussianAdditive(name="", parents=(), coef_prior_sd=2.0)})
```

A mechanism is a frozen value object, so the name and parents fill in from the graph and your instance is never mutated; reuse one across models freely. Richer overrides (a sklearn `Learner` as the mean, GP and BART built-ins, a `PyMCLearner` subclass, a raw PyMC callback) are planned and not in this release.

The MVP fits a Markovian DAG. A graph with bidirected edges is rejected at construction, because latent-district fitting is post-MVP. A latent confounder still has a place: mark it `latent=` on the DAG and the SCM will answer its effect honestly through `ate`, covered below. It is not fitted, since it carries no data.

## Fitting

`fit` runs the sampler. The work factorizes. For a Markovian DAG with complete data the posterior factors over nodes, so each node is fitted on its own likelihood given its observed parents, and the per-node fits are independent. The default is sequential, which is robust and safe under continuous integration.

```python
scm.fit()
scm   # SCM(3 nodes, fitted, n=2000)
```

`fit(parallel=True)` spreads the nodes over a process pool. That path pins the BLAS and OpenMP thread counts to one inside each worker, because multi-threaded linear algebra under multiprocess NUTS oversubscribes the cores and the operating system starts killing sampler workers. If you run the SCM from your own scripts, set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` for the same reason.

A fitted `SCM` is stateful: it holds the per-node posteriors, and the queries are methods that read them.

## The three queries

### Sample: a new world

`sample` draws from the generative model: every node, in order, parents through its equation plus fresh noise. It returns the draws of a sink node as a summary; read any node with `marginal`.

```python
scm.sample(n=2000)
# SCMResult(sample, identified: score=-0.03779, sd=2.43, 95% CrI [-4.8, 4.72])
```

Every query returns an `SCMResult`: a posterior over the answer, with `value` (the mean), `std_error`, `draws()`, and `interval()`. The point is the degenerate one-draw case, the same distribution shape the Bayesian estimators in [`ergodic.inference`](inference.md) emit. The `identified` flag in the repr is the honesty marker; for a query that is a function of the data alone it reads `identified`.

### Interventional: do, not condition

`interventional` answers `E[Y | do(X = x)]`. This is surgery, not conditioning. PyMC's `pm.do` severs the intervened node from its causes and pins it to the value, so the back door through any confounder is cut and what flows downstream is the causal effect. The query carries the parameter posterior, so the answer has a credible interval that reflects how well the mechanisms are pinned.

```python
high = scm.interventional("score", do={"hours": 1.0}, n=2000)
high   # SCMResult(interventional, identified: score=2.013, sd=1.4, 95% CrI [-0.722, 4.75])

low = scm.interventional("score", do={"hours": 0.0}, n=2000)
high.value - low.value   # ~1.99, against the planted 2.0
```

`do` returns a lightweight intervened view for chaining, sharing the fit without refitting:

```python
scm.do({"hours": 1.0}).interventional("score").value   # ~2.01
```

Binary and categorical nodes ride this path too. The generative model samples a Gumbel-max node the same way it samples a Gaussian one, so `sample`, `interventional`, `do`, and `marginal` all run on a graph with binary or categorical nodes. A categorical do-target is pinned by its integer category code, the same code the data layer reads, so `do={"region": 2}` fixes the region to category 2; a fractional or out-of-range code raises rather than snap to the nearest. The one query that still refuses the categorical case is `ate` on a categorical outcome, since a category has no scalar average effect to contrast; that is covered below.

### Counterfactual: the unit's own noise

A counterfactual is about one specific unit, not the average. The student scored 3.5 with their actual study hours. What would they have scored with different hours, keeping everything else about them as it was, including the parts the model never sees?

The answer is a twin network. Take the unit's factual row, run the equations backward to recover the noise that produced it (for an additive mechanism this is the exact residual, `U = score - (b0 + w . parents)`), then run the equations forward on the action branch with the intervened value and that same noise reused. The unit keeps its own draw of every unmodeled influence, which is what makes the answer specific to them.

```python
unit = {"hours": np.array([0.0]), "sleep": np.array([1.0]), "score": np.array([3.5])}
cf = scm.counterfactual(unit, do={"hours": 2.0}, target="score")
cf   # SCMResult(counterfactual, identified: score=7.484, sd=0.0606, 95% CrI [7.37, 7.6])
```

The factual score was 3.5; moving hours from 0 to 2 at a coefficient near 2 adds about 4, and the counterfactual lands at 7.48. The interval is narrow because the only uncertainty left is in the parameters: for an invertible mechanism the noise is exact, not resampled. Pass `use_draws=False` to collapse to the deterministic point at the posterior mean. `units` can be a `Dataset` or a mapping of node name to a column, and the query batches over all the units at once.

A categorical target works here where it did not for the interventional query, because the Gumbel-max mechanism is invertible: the observed category pins the noise through a truncated-Gumbel abduction, and the action branch reuses it. Take a world where tenure drives a binary promotion flag. The result is a `CategoricalResult`, since the answer is a distribution over categories rather than a number:

```python
rng = np.random.default_rng(1)
tenure = rng.normal(0.0, 1.0, size=2000)
promo = (rng.uniform(size=2000) < 1.0 / (1.0 + np.exp(-tenure))).astype(int)
spend = 3.0 + 2.0 * promo + tenure + rng.normal(0.0, 1.0, size=2000)
pdata = tabular(pd.DataFrame({"tenure": tenure, "promo": promo, "spend": spend}))

pg = dag(["tenure -> promo", "promo -> spend", "tenure -> spend"])
pscm = SCM(pg, pdata, draws=300, tune=300, chains=2, random_state=0).fit()

# a unit that was not promoted at tenure 0.5: would it have been, at tenure -1.0?
unit = {"tenure": np.array([0.5]), "promo": np.array([0]), "spend": np.array([4.0])}
cf = pscm.counterfactual(unit, do={"tenure": -1.0}, target="promo")
cf            # CategoricalResult(counterfactual, identified: promo [0=1, 1=0])
cf.mode()     # 0: the most probable counterfactual category
cf.per_unit() # array([0]): the per-unit counterfactual category code
```

A counterfactual over a non-invertible mechanism (a probabilistic-only node, where the noise is not recoverable) is refused rather than faked. That answer needs the wider abduction posterior, which is post-MVP. Every built-in mechanism is invertible, so the refusal guards a future custom mechanism, not today's worlds.

## The honesty layer

`ate` is where the pillar earns its name. Ask for the average effect of a treatment on an outcome and the SCM routes the question through [`identify_effect`](identification.md) before it computes anything. What comes back depends on what the graph says the data can support, and there are three cases.

When the effect is identified by back-door adjustment, you get a posterior. The SCM intervenes at both levels and takes the contrast, `E[Y | do(X = high)] - E[Y | do(X = low)]`, carrying the parameter uncertainty through to a credible interval.

```python
ate = scm.ate("hours", "score", levels=(0.0, 1.0))
ate   # SCMResult(ate, identified: score=1.992, sd=0.023, 95% CrI [1.95, 2.04])
```

A binary treatment is the everyday causal query, and `ate` computes it natively. Take a confounded world: a binary confounder Z pushes a binary treatment T and the outcome Y, T raises Y by a planted 2.0, and Z raises both, so the raw treated-minus-untreated difference overstates the effect. The SCM samples Z, intervenes on T, and marginalizes Z back out, which is the back-door g-formula. A binary treatment's levels default to (0, 1), so you do not pass them.

```python
import numpy as np
import pandas as pd
from ergodic import SCM, dag, tabular, estimate_effect

rng = np.random.default_rng(4)
n = 3000
z = rng.normal(0.0, 1.0, size=n)                                # confounder
t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(int)  # binary treatment
y = 2.0 * t + 1.6 * z + rng.normal(0.0, 1.0, size=n)            # planted effect 2.0
data = tabular(pd.DataFrame({"Z": z, "T": t, "Y": y}))

g = dag(["Z -> T", "Z -> Y", "T -> Y"])
scm = SCM(g, data, draws=300, tune=300, chains=2, random_state=0).fit()

ate = scm.ate("T", "Y")
ate          # SCMResult(ate, identified: Y=2.012, sd=0.039, 95% CrI [1.94, 2.09])
```

The SCM lands at 2.01, on the planted 2.0. The naive treated-minus-untreated difference is 3.32, pulled up by the confounder the g-formula adjusts away, and `estimate_effect` on the same graph agrees at 2.00.

```python
df = data.to_pandas()
df[df["T"] == 1]["Y"].mean() - df[df["T"] == 0]["Y"].mean()   # 3.32, the confounded difference
estimate_effect(g, data, "T", "Y").value                      # ~2.00, the adjusted effect
```

A multi-category treatment has no canonical pair, so it asks for the two codes through `levels=(low_code, high_code)`, the same shape as the inference pillar's `treated=`. The one case `ate` still refuses is a categorical outcome, because a category has no scalar average to contrast; `ate` raises there and points you at `interventional`, which reads the whole post-intervention distribution over the categories.

The no-path case lands here too: a graph with no directed path from treatment to outcome is identified as the zero effect, and `ate` returns an `SCMResult` whose value is exactly 0.0.

When the effect is not identified, the answer is different in kind. Take a bow: a latent confounder U drives both the treatment and the outcome, and there is no valid adjustment set. The observational slope mixes the causal effect with the confounding, and no amount of data separates them. A confident point estimate here would be a lie with tight error bars, which is the failure mode the pillar exists to refuse. So `ate` returns a `Sensitivity` instead.

```python
rng = np.random.default_rng(1)
u = rng.normal(0.0, 1.0, size=3000)                     # the unobserved confounder
x = 0.8 * u + rng.normal(0.0, 1.0, size=3000)
y = 1.5 * x + 2.0 * u + rng.normal(0.0, 1.0, size=3000)  # true effect of x on y is 1.5
bdata = tabular(pd.DataFrame({"x": x, "y": y}))

bow = dag(["U -> x", "U -> y", "x -> y"], latent=["U"])
bscm = SCM(bow, bdata, draws=300, tune=300, chains=2, random_state=0).fit()
sens = bscm.ate("x", "y")
sens
# Sensitivity(x -> y: effect in [0, 2.47] over assumed confounding, b(0)=2.47;
#   not identified: ...a hedge formed by ['x', 'y'] and the c-component ['y'])
```

A `Sensitivity` is the effect as a function of the confounding you assume. For the linear-Gaussian bow the relationship is exact: the observed covariance fixes `cov(x, y) = b * var(x) + c`, where `c` is the confounding channel the data cannot see, so the causal effect is the line `b(c) = (cov - c) / var(x)`. Read it where you like.

```python
sens.identified_at_zero   # 2.47: b(0), the answer if you assume no confounding (the naive slope)
sens.at(0.0)              # 2.47: same point, read off the curve
sens.range()              # (0.0, 2.47): the effect over the swept confounding grid
```

`identified_at_zero` is the value under the assumption that none of the association is confounding, which is the observational slope and the number a naive analysis would report. The true effect in this world is 1.5, and it sits inside the band, at the confounding strength the data cannot pin. There is no `value` on a `Sensitivity` and no scalar accessor: a non-identified effect is a range, and the type makes a naked read impossible by construction. The prior is the answer, so the type forces you to state it.

The third case is where the latent machinery earns its place. An effect can be identified and still not be a plain back-door adjustment: front-door, instruments, or the general do-calculus functional. Hand the SCM a latent-confounded graph (a `<->` edge, or a `latent=` confounder) and it fits the confounded district jointly, with one explicit latent variable per bidirected edge standing in for the shared cause. The latent absorbs the confounding, so the mechanism on the causal path fits clean and `do` on the fitted model recovers the identified effect. A front-door world computes.

```python
rng = np.random.default_rng(3)
u = rng.normal(0.0, 1.0, size=3000)
a = u + rng.normal(0.0, 1.0, size=3000)
m = 1.2 * a + rng.normal(0.0, 1.0, size=3000)
y = 0.9 * m + 2.0 * u + rng.normal(0.0, 1.0, size=3000)   # the effect of A on Y is 1.2 * 0.9 = 1.08
fdata = tabular(pd.DataFrame({"A": a, "M": m, "Y": y}))

fd = dag(["A -> M", "M -> Y", "U -> A", "U -> Y"], latent=["U"])   # front-door, U latent
fdscm = SCM(fd, fdata, draws=200, tune=200, chains=2, random_state=0).fit()
fdscm.ate("A", "Y")
# SCMResult(ate, identified: Y=1.1, sd=0.039, 95% CrI [1.02, 1.18])
```

Without the latent, the SCM would have read a confounded `Y` on `M` slope and handed back a confident wrong number; the explicit latent is what makes the contrast right. Two criteria still raise. An instrument needs the exclusion restriction, which the latent fit does not impose, so an instrument effect points you at [`estimate_effect`](inference.md) and its two-stage estimator. The general do-calculus functional is the same story: identified, but not computed from the SCM this phase, so it raises with the pointer. The `engine="dml"` path, which would reuse the cross-fit DML estimator from a fitted SCM, is a stub for now.

So `ate` reads the graph and the return type tells you which world you are in. Identified, whether by back door or by a front-door path the latent district fits, you get a number with an interval. Not identified, you get the `Sensitivity` curve and have to name your assumption. Identified by an instrument or the general functional, you get sent to the tool that computes it. The honesty never bends: the latent machinery widens what the SCM can compute, never what it will claim.

## What is here, and what is next

The SCM fits a DAG or a latent-confounded ADMG: continuous nodes as Bayesian-linear Gaussian-additive equations, binary and categorical nodes as Gumbel-max, and one explicit latent per bidirected edge for a confounded district. It answers observational sampling, `pm.do` interventions, and the explicit-noise counterfactual on any node type, all carrying a posterior, with non-invertible-mechanism counterfactuals deferred. And it gates every effect on identification, returning a posterior, a `Sensitivity`, or a pointer.

The next pieces are sketched in the source. Policy search and off-policy evaluation on the fitted model come next, tying the SCM to the process pillar and turning a fitted causal model into a decision. Then the instrument and general-functional estimands computed from the SCM, the richer mechanism overrides, knowledge as priors, and the equivalence-class ensemble. The [API reference](reference.md) covers the surface that ships today.
