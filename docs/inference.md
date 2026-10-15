# Causal inference

How much does the treatment actually change the outcome? That is what estimation answers, as a number with a confidence interval around it.

Give `estimate_effect` a graph, a dataset, a treatment, and an outcome. It works out whether the effect can be identified, picks the estimator that fits, fills that estimator's helper models (its nuisance slots) with a learner, fits, and returns an `EffectEstimate`.

## One call

```python
import numpy as np
import pandas as pd
from ergodic import dag, tabular, estimate_effect

rng = np.random.default_rng(0)
n = 4000
z = rng.normal(size=n)
a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * z))).astype(float)
y = 2.0 * a + 1.5 * z + rng.normal(size=n)   # true effect of A on Y is 2.0
data = tabular(pd.DataFrame({"A": a, "Z": z, "Y": y}))

g = dag(["Z -> A", "Z -> Y", "A -> Y"])
est = estimate_effect(g, data, "A", "Y")
est            # EffectEstimate(aipw, analytic: ate=1.997, se=0.0342, 95% CI [1.93, 2.06], n=4000)
est.value      # ~2.0
est.interval() # (~1.93, ~2.06)
```

The graph chose the family (here adjustment, since `Z` closes the back door). The family default is AIPW, the doubly-robust estimator.

## Two axes: the estimator and the learner

An estimation method factors into a strategy and a model. The strategy is the `CausalEstimator`: how you turn the estimand into a number. The learner is the model in each nuisance slot.

Swap the strategy by name:

```python
estimate_effect(g, data, "A", "Y", strategy="outcome_regression")  # g-formula
estimate_effect(g, data, "A", "Y", strategy="ipw")                 # propensity weighting
estimate_effect(g, data, "A", "Y", strategy="aipw")                # doubly robust (default)
```

Swap the learner by alias, by a slot mapping, or with any object that has `fit` and `predict`:

```python
estimate_effect(g, data, "A", "Y", strategy="aipw", learner="knn")
estimate_effect(
    g, data, "A", "Y", strategy="aipw",
    learner={"outcome": "ols", "propensity": "logistic"},
)
```

A single alias fills every slot, so it has to suit them all. `"knn"` works here because the propensity slot gets its classifier variant; `"ols"` alone would be refused there, since that slot needs a classifier. When the slots need different models, pass a slot mapping.

The learner is structurally a sklearn estimator, so any sklearn or xgboost model drops into a slot with no adapter:

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

estimate_effect(
    g, data, "A", "Y", strategy="aipw",
    learner={"outcome": GradientBoostingRegressor(), "propensity": GradientBoostingClassifier()},
)
```

The native `OLS` and `LogisticRegression` cover the defaults, so a zero-dependency install still estimates. The default for a slot follows the column type: a regression slot gets `OLS`, a classification slot (propensity, mediator) gets `LogisticRegression`, and a binary outcome gets `LogisticRegression`.

## The result is distribution-shaped

`EffectEstimate` carries a point with a standard error, or a set of draws, and the accessors read whichever is present.

```python
est = estimate_effect(g, data, "A", "Y", strategy="aipw")
est.kind                 # 'analytic', from the efficient influence function
est.value, est.std_error
est.interval(0.99)       # a wider interval

boot = estimate_effect(g, data, "A", "Y", strategy="ipw", random_state=0)
boot.kind                # 'bootstrap'
boot.draws.shape         # (1000,)
boot.interval()          # the 2.5% and 97.5% draw quantiles
```

AIPW gets an exact (analytic) standard error with no resampling, because its math is built so small errors in the helper models cancel out (the property called Neyman-orthogonality). Outcome regression and IPW bootstrap, which fills the draws. A point estimate is the degenerate case of a distribution, so the Bayesian path below returns posterior draws through the same result and the same accessors.

## Targets

Ask for the average effect (the default for adjustment and front-door), the effect on the treated, or the local effect for instruments:

```python
estimate_effect(g, data, "A", "Y", strategy="aipw", target="ate")
estimate_effect(g, data, "A", "Y", strategy="aipw", target="att")
```

A two-level treatment that is not 0/1 (say `"drug"` and `"placebo"`) gets coded by level order, and that order may not be the contrast you mean. Pass `treated=` to name the level that codes to 1:

```python
estimate_effect(g, data, "A", "Y", treated="drug")   # the effect of drug over placebo
```

Without it the estimate still runs, with a warning naming the implicit coding.

## Front-door and instruments

Same call, a different graph. When a confounder is unobserved, a mediator or an instrument carries the identification, and the matching estimator is selected.

```python
from ergodic import admg, dag

# illustrative: data_fd and data_iv are datasets generated to fit each graph
# front-door: binary treatment A, binary mediator M, unobserved A-Y confounding
fd = admg(["A -> M", "M -> Y", "A <-> Y"])
estimate_effect(fd, data_fd, "A", "Y")   # two_stage_regression, target ATE

# instrument Z reaches Y only through A; U is the unobserved confounder
iv = dag(["Z -> A", "A -> Y", "U -> A", "U -> Y"], latent=["U"])
estimate_effect(iv, data_iv, "A", "Y")   # two_stage_least_squares, target LATE
```

A graph holds one edge per node pair, so a treatment with both a direct effect and unobserved confounding on the same pair (the bow) is written as above: the confounder is its own node, marked latent.

## Cross-fitting, orthogonal estimators, and heterogeneous effects

Cross-fitting fits each nuisance out of fold, so a flexible learner keeps a valid standard error. It powers double machine learning and the doubly-robust learner.

```python
estimate_effect(g, data, "A", "Y", strategy="dml")         # partialling-out, analytic SE
estimate_effect(g, data, "A", "Y", strategy="dr_learner")  # doubly robust, analytic SE
```

`DML` reads a binary or a continuous treatment, so it is the way past the binary-treatment limit. A continuous treatment reports the average partial effect.

For a heterogeneous effect, the CATE estimators (`DRLearner`, `RLearner`, `XLearner`, and the T-learner) expose `cate`:

```python
from ergodic import identify_effect
from ergodic.inference import DRLearner

model = DRLearner(random_state=0).fit(identify_effect(g, "A", "Y"), data)
model.cate()         # the per-unit effect, in-sample
model.cate(other)    # other: another aligned dataset (illustrative)
estimate_effect(g, data, "A", "Y", strategy="dr_learner").value  # the average, with its SE
```

The native `KNNRegressor` and `KNNClassifier` (alias `"knn"`) give cross-fitting a flexible nuisance with no dependency, and `PLIV` is the flexible-learner instrument estimator:

```python
estimate_effect(g, data, "A", "Y", strategy="dml", learner="knn")
estimate_effect(iv, data_iv, "A", "Y", strategy="pliv")    # orthogonal IV, LATE
```

## Mediation: the direct and indirect effects

A total effect splits into the part that flows through a mediator and the part that does not. `estimate_mediation` identifies the decomposition (through [`identify_mediation`](identification.md)), then estimates it by g-computation.

```python
from ergodic import dag, estimate_mediation

g = dag(["X -> M", "M -> Y", "X -> Y", "W -> X", "W -> Y"])
med = estimate_mediation(g, data, "X", "Y", "M", random_state=0)
med.nde.value     # the natural direct effect, with a bootstrap SE
med.nie.value     # the natural indirect effect
med.total.value   # the total effect, E[Y(1, M(1))] - E[Y(0, M(0))]
med.cde           # a dict: each mediator level to its controlled direct effect
med.proportion_mediated   # nie / total, the share carried by the mediator
```

NDE, NIE, and the total each come back as a distribution-shaped `EffectEstimate`, the same surface the rest of estimation returns, so `value`, `std_error`, and `interval()` all read the same way. The estimator fits an outcome model `E[Y | x, m, w]` and a mediator model `P(m | x, w)`, plugs them into the mediation formula, and bootstraps the rows for the standard error. It handles a binary treatment, a discrete mediator, and a continuous or binary outcome. A continuous mediator raises and points at the density path, which isn't available yet.

The NDE and NIE here are the pure effects: each holds the other path at the control level. When the treatment changes how the mediator acts on the outcome (an interaction), they don't add up to the total; the remainder, `total - nde - nie`, is the mediated interaction. The total is computed directly, so it's right either way.

The result carries a `Design` record naming the assumptions, the same role `Design` plays for the panel designs, because the natural effects rest on a cross-world condition no data can test (see the [identification guide](identification.md)). `proportion_mediated` is `nie / total`, flagged when the total effect is near zero, since a small denominator makes the ratio unstable.

## Discrete ID: the plug-in

When identification falls through to the general do-calculus algorithm (`strategy="id"`, the [identification guide](identification.md) covers when), the estimand is a typed functional with no named estimator. On discrete data it needs none: the functional is a sum-product of conditional frequencies, so the plug-in estimates each term by its empirical count and evaluates the tree. `estimate_effect` routes there automatically.

```python
from ergodic import admg, estimate_effect

# illustrative: discrete_data is a categorical dataset matching the Verma graph
g = admg(["A -> B", "B -> C", "C -> D", "A <-> C", "B <-> D"])
est = estimate_effect(g, discrete_data, "A", "D", random_state=0)
est.estimator   # 'plug_in'
est.value       # the ATE, with a bootstrap SE
```

The guards are honest. A continuous variable in the functional raises and points at the SCM path, which does not exist yet, so `strategy="id"` is not estimable on continuous data. A non-binary treatment raises (the ATE is a two-level contrast). A zero-count stratum the formula divides by raises with the offending assignment, the discrete analog of the overlap check, rather than smoothing in silence.

## Bayesian learners and the posterior

A learner can carry a posterior. A `PyMCLearner` wraps a PyMC model: you write the model, the base samples it and serves the posterior of its mean function. Drop one into a slot and the estimator returns a posterior, through the same `EffectEstimate` and the same accessors, now with `kind="posterior"`.

```python
import pymc as pm
from ergodic.inference import PyMCLearner

class StudentTOutcome(PyMCLearner):
    def build(self, X, y):
        with pm.Model() as model:
            x = pm.Data("X", X)                          # name the design "X"
            b = pm.Normal("b", 0.0, 1.0, shape=X.shape[1])
            pm.Deterministic("mu", x @ b)                # name the mean node "mu"
            pm.StudentT("y", nu=4, mu=model["mu"], sigma=pm.HalfNormal("s", 1.0), observed=y)
        return model

est = estimate_effect(g, data, "A", "Y", strategy="outcome_regression", learner=StudentTOutcome())
est.kind            # 'posterior'
est.value           # the posterior mean of the ATE
est.interval()      # the 2.5% and 97.5% posterior quantiles, a credible interval
est.draws.shape     # (S,)
```

You write `build` and follow one convention: put the design in a `pm.Data("X", X)` container and name the mean node `"mu"` (a regression mean) or `"p"` (a class-1 probability, for `PyMCClassifier`). The base owns the sampling, the posterior-predictive of that node, and cloning. PyMC is an optional dependency, `pip install ergodic[bayes]`, and importing `ergodic.inference` never imports pymc.

A Bayesian learner's `predict` is still the posterior mean, so it is also a plain point learner: hand it to any estimator and it acts as its mean. The posterior path turns on when the estimator knows how to propagate the draws. Two do today:

- `OutcomeRegression` is the Bayesian g-formula. Each outcome draw gives one average contrast, and their spread is the posterior of the ATE (or the ATT).
- `AIPW` is the doubly-robust posterior. It evaluates the efficient influence function at each nuisance draw and reweights the rows by a Dirichlet(1) Bayesian bootstrap, so the posterior carries the model and the sampling uncertainty together and keeps the doubly-robust protection. Pass `bayesian_bootstrap=False` to drop the reweighting.

With a Bayesian outcome learner, the T-learner gives every unit its own posterior, so the CATE comes with a credible interval:

```python
from ergodic import identify_effect
from ergodic.inference import OutcomeRegression

model = OutcomeRegression(StudentTOutcome(), per_arm=True).fit(identify_effect(g, "A", "Y"), data)
model.cate_draws()                        # (S, n): the per-unit posterior
lo, hi = model.cate_interval(level=0.9)   # the per-unit 90% credible band
```

A mix of one Bayesian and one point learner does not form a joint posterior, so the estimator falls back to its point or analytic result. An estimator with no posterior path returns a single point rather than refit the sampler in a bootstrap loop.

## Checking an estimate

A single number says nothing about how fragile it is. `refute_effect` reruns the whole identify-then-estimate path under a deliberate perturbation and checks the answer behaves the way a trustworthy estimate should. Pass the same `strategy`, `learner`, and `treated` you gave `estimate_effect`.

```python
from ergodic import refute_effect

ref = refute_effect(g, data, "A", "Y", strategy="aipw", random_state=0)
ref            # Refutation(placebo_treatment: estimate=1.997, under perturbation=-0.003, p=0.0099, passed, n=100)
ref.passed     # True: with the treatment scrambled, the effect collapses to about zero
ref.p_value    # ~0.01: almost no scrambled run reaches the real effect in size
```

The default is the placebo treatment: shuffle the treatment column so it points at nothing, re-estimate, and a real effect should fall to near zero. Two more checks come from `method=`:

- `"placebo_treatment"` (the default) breaks the treatment's link to everything. It passes when the mean placebo effect is under 20 percent of the real one, and `p_value` is the share of placebo runs whose effect matches or beats the real one in size.
- `"random_common_cause"` adds an irrelevant random parent of both treatment and outcome. The effect should not move; it passes when the mean stays within 10 percent of the original. `p_value` is `None` here.
- `"data_subset"` re-estimates on a random 80 percent of the rows (set by `subset_fraction`). The effect should hold; it passes when the original sits within two standard deviations of the subset spread.

None of these proves the estimate is right, since nothing on observational data can, but a failure is a real warning, and they cost only the refitting. Look at the actual numbers, since `passed` alone hides how close the call was. These are the cheap, model-agnostic checks; sensitivity to a specific named confounder is a sharper, separate tool that comes later. Every case study ends with this placebo check.

## The gates

Two checks guard a call, each with a readable error. The family gate: an estimator only consumes the family it is built for, so asking for two-stage least squares on an effect identified by adjustment is refused. The slot gate: a classification slot needs a learner with `predict_proba`, so a plain regressor there is refused.

```python
from ergodic.inference import OLS

estimate_effect(g, data, "A", "Y", strategy="2sls")
# ValueError: estimator '2sls' consumes ['iv'], but the effect was identified by 'adjustment'

estimate_effect(g, data, "A", "Y", strategy="ipw", learner=OLS())
# TypeError: slot 'propensity' needs a classifier with predict_proba; got OLS
```

## The estimator library

Adjustment: `OutcomeRegression` (the S- and T-learner), `IPW`, `Stratification`, `AIPW`, `DML`, `DRLearner`, `RLearner`, `XLearner`. Front-door: `TwoStageRegression`. Instruments: `TwoStageLeastSquares`, `Wald`, `PLIV`. The default per family is the doubly-robust or orthogonal one. Standard errors come from an influence function where one is known (AIPW, DML, DRLearner, two-stage least squares, PLIV), bootstrap otherwise. Treatment is binary for most, binary or continuous for `DML` and `PLIV`; the outcome is continuous or binary; the data is tabular. One more family is trivial by design: a graph with no directed path from treatment to outcome identifies the `null` strategy, and `NullEffect` reports an exact zero.

The panel and time-series designs have their own page. Difference-in-differences, event studies, synthetic control, and interrupted time series don't consume a graph: a comparison built from the data's time structure stands in for the counterfactual, so they read a `PanelData` or `TimeSeriesData` directly and carry a `Design` record in place of an `Identification`. The [panel and time-series guide](panel.md) covers them.

## Not built yet

TMLE and the orthogonal front-door. Multi-level and multi-valued treatment, and dose-response curves. Frequentist per-unit CATE intervals (the Bayesian path above gives them now; the analytic version needs per-unit standard errors). The Bayesian posterior covers the adjustment estimators, so the instrument and front-door posteriors, cross-fit Bayesian nuisances, and the joint PyMC estimators that sample the whole model come next, as does a JAX or NumPyro backend behind the same `predict_draws`. On the panel page, multi-unit synthetic control: today it runs one call per treated unit.

Mediation ships the g-computation estimator for a discrete mediator. The multiply-robust, cross-fit one-step estimator (valid inference under slower nuisance rates, the same orthogonality story as DML) and the continuous-mediator density form come next. The discrete plug-in covers `strategy="id"` on categorical data; the continuous general functional waits for the SCM and its fitted mechanisms, a separate pillar.
