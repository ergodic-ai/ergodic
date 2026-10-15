# Fit a structural causal model

Fit a Bayesian structural causal model (SCM) over a graph and ask it for an average
effect. You'll also see what it returns when the graph says the data can't answer.

!!! need "You'll need"
    - ergodic with the SCM extra: `pip install "ergodic[scm]"`. It adds PyMC.
    - A causal graph you're willing to defend: a DAG, or an ADMG whose `<->` edges
      stand for hidden common causes.

## Run the program

The data is simulated, so you know the answer. A hidden variable `U` drives both `X`
and `Y`, and `X` affects `Y` only through `M`. The true effect of `X` on `Y` is
0.8 × 1.5 = 1.2.

```python file="fit_scm.py"
import numpy as np
import pandas as pd
from ergodic import SCM, admg, dag, tabular

# 1. Simulate 500 units. A hidden U drives X and Y; X acts on Y only through M.
rng = np.random.default_rng(0)
n = 500
u = rng.normal(size=n)
x = u + rng.normal(size=n)
m = 0.8 * x + rng.normal(size=n)
y = 1.5 * m + 2.0 * u + rng.normal(size=n)
data = tabular(pd.DataFrame({"X": x, "M": m, "Y": y}))
print(f"true effect of X on Y: {0.8 * 1.5:.2f}")
print(f"slope of Y on X:       {np.polyfit(x, y, 1)[0]:.2f}")

# 2. Draw the graph. X <-> Y says that a hidden common cause links X and Y.
graph = admg(["X -> M", "M -> Y", "X <-> Y"])

# 3. Fit one mechanism per node.
scm = SCM(graph, data, draws=500, tune=500, chains=1, random_state=0).fit()
for node, mechanism in scm.mechanisms.items():
    print(f"{node}: {type(mechanism).__name__}, parents {mechanism.parents}")

# 4. Ask for the average effect of a one-unit increase in X.
effect = scm.ate("X", "Y")
print(effect)

# 5. Drop the mediator from the graph. Now no formula identifies the effect.
bow = dag(["U -> X", "U -> Y", "X -> Y"], latent=["U"])
bounds = SCM(bow, data, draws=500, tune=500, chains=1, random_state=0).fit().ate("X", "Y")
low, high = bounds.range()
print(f"{type(bounds).__name__}: effect between {low:.2f} and {high:.2f}")
print(f"effect if the hidden path carries covariance 2.0: {bounds.at(2.0):.2f}")
```

```output from="fit_scm.py" tol="0.05" rtol="0.05"
true effect of X on Y: 1.20
slope of Y on X:       2.32
X: GaussianAdditive, parents ()
M: GaussianAdditive, parents ('X',)
Y: GaussianAdditive, parents ('M',)
SCMResult(ate, identified: Y=1.265, sd=0.0772, 95% CrI [1.12, 1.42])
Sensitivity: effect between 0.00 and 2.32
effect if the hidden path carries covariance 2.0: 1.28
```

- **Step 1.** The raw slope is almost twice the true effect. The hidden `U` inflates
  it.
- **Step 3.** Each continuous node gets a `GaussianAdditive` mechanism: a Bayesian
  linear equation plus its own noise term.
- **Step 4.** The graph allows the front-door formula, so `ate` returns an
  `SCMResult`: a posterior mean of about 1.27, with a 95% credible interval that
  contains the true 1.2.
- **Step 5.** Without `M`, the effect isn't identified. `ate` returns a `Sensitivity`
  instead of a number: the effect for each assumed strength of the hidden path.

## Read the two return types

`ate` asks [`identify_effect`](identify-an-effect.md) first, and the answer decides
the return type:

- **Identified** by adjustment, by the front-door formula, or because no directed
  path exists: an `SCMResult`. Read `.value`, `.std_error`, `.interval(level)` and
  `.draws()`.
- **Not identified**: a `Sensitivity`. It has no `.value`, so you can't read a single
  number by mistake.

For a continuous treatment and outcome, a `Sensitivity` holds the line
`b(c) = (cov(X, Y) − c) / var(X)`, where `c` is the covariance the hidden path adds.
Read it with `.range()`, `.at(c)` and the `.confounding` and `.effect` arrays:

```python
low, high = bounds.range()
print(f"{type(bounds).__name__}: effect between {low:.2f} and {high:.2f}")
print(f"effect if the hidden path carries covariance 2.0: {bounds.at(2.0):.2f}")
```

The range runs from no confounding (`c = 0`, the raw slope) to all of the covariance
being confounding (an effect of zero). Here `U` adds a covariance of 2.0, so
`.at(2.0)` lands near the truth. With real data you don't know `c`.

The range covers the truth only if the hidden path pushes in the same direction as
the observed association, and adds no more than all of it. For any other `c`,
`.at(c)` extrapolates exactly along the line.

For a binary or categorical treatment with a continuous outcome, the band comes from
refitting the model under a grid of confounding priors instead. Then `.confounding`
holds prior scales, not covariances.

## Choose the mechanisms

`SCM` assigns a mechanism to each node from its column type:

- continuous: `GaussianAdditive`;
- binary: `BinaryGumbelMax`, a logistic choice with explicit noise;
- categorical: `CategoricalGumbelMax`.

To override one, pass `mechanisms={"Y": "linear"}`. The names are `"linear"`,
`"binary"` and `"categorical"`. You can also pass a mechanism instance from
`ergodic.scm`, for example
`GaussianAdditive(name="Y", parents=("M",), coef_prior_sd=1.0)`.

`draws`, `tune` and `chains` set the sampler budget. The defaults are 1,000 draws,
1,000 tuning steps and 2 chains. This page uses a smaller budget so it runs in
seconds. `cores` (default 1) caps the processes PyMC may use for the chains, so the
chains run one after another unless you raise it; the draws are the same either way.

!!! caution "The graph carries the assumptions"
    The front-door answer holds only if `M` carries all of the effect of `X`,
    nothing hidden links `X` and `M`, and nothing hidden links `M` and `Y`. The data
    can't check these assumptions.

<details markdown="1"><summary>Which graphs and effects does the SCM accept?</summary>

- A DAG, an ADMG, or a DAG with `latent=` nodes. Each `<->` edge becomes one hidden
  variable per unit. Nodes connected through `<->` edges are fitted jointly, as one
  group. A latent node can't be a query target or an intervention target.
- Any other kind raises a `ValueError`: a CPDAG, PAG or MAG stands for a class of
  graphs, and a MAG's or PAG's `->` means ancestry, not a direct effect. A `mixed`
  graph raises too. Pick one member of the class first, or re-face a graph you read
  as one causal world with `graph.as_kind("dag")` or `graph.as_kind("admg")`.
- A custom mechanism can't sit on a node with a `<->` edge; only the built-in
  mechanisms carry the hidden variable.
- `ate` needs a numeric outcome. For a binary or categorical outcome, use
  `interventional(...)`, which returns a `CategoricalResult` with per-category
  probabilities.
- A categorical treatment with more than two levels needs explicit
  `levels=(low_code, high_code)`.
- An effect identified only by an instrument, or only by the general ID formula,
  raises `NotImplementedError`. Use `estimate_effect` for those. An instrument
  identifies the effect only under a linear, homogeneous model, an assumption the
  SCM's contrast doesn't make.
- A jointly fitted group of more than 8 nodes still fits, with a warning that it may be
  slow. Raise `district_size_cap=` to silence it.

</details>

## Next steps

<div class="cards" markdown="1">

- [Simulate interventions and counterfactuals](interventions-and-counterfactuals.md): set a variable, or replay one unit's history.
- [Check whether an effect is identifiable](identify-an-effect.md): the formulas `ate` checks first.
- [Python reference](reference-scm.md): every option of `SCM`.

</div>
