# Estimate effects of numerical changes

Compare two values of a numerical treatment, such as a dose or a price. Choose a
partially linear model for a constant effect per unit, or continuous DR for a
nonlinear dose response. Both methods identify the adjustment set from your graph.

!!! need "You'll need"
    - A checkout containing this unreleased feature; it isn't in ergodic 0.3.2.
    - Independent rows with a numerical treatment, a numerical or binary outcome,
      and an adjustment-identifiable graph.
    - Enough observations near both treatment values, across the relevant covariates.

## Run the program

This simulation has a nonlinear response and an effect that varies with `z`.
Changing the dose from -0.5 to 1 has a true conditional effect of `3 + 0.75*z`.
The population effect is 3 because `z` is centered at zero in the generating model.

```python file="continuous_treatments.py"
import numpy as np
from ergodic import dag, estimate_contrast, identify_effect, tabular
from ergodic.inference import ContinuousDR

rng = np.random.default_rng(17)
n = 6000
z = rng.uniform(-1, 1, n)
dose = 0.3 * z + rng.normal(size=n)
response = 1.5 * dose + dose**2 + 0.5 * dose * z + z + rng.normal(0, 0.15, n)
context = tabular({"dose": dose, "z": z, "response": response})
graph = dag(["z -> dose", "z -> response", "dose -> response"])

linear = estimate_contrast(
    graph, context, "dose", "response",
    baseline=-0.5, value=1, method="partially_linear", random_state=0,
)
print(f"partially linear: {linear.value:.2f}")

identified = identify_effect(graph, "dose", "response")
fitted = ContinuousDR(random_state=0).fit(identified, context)
average = fitted.contrast(baseline=-0.5, value=1)
print(f"continuous DR: {average.value:.2f}")
print(f"bandwidth: {average.diagnostics['bandwidth']:.3f}")

# Query rows need only the adjustment covariates, not dose or response.
rows = tabular({"z": [-0.5, 0.0, 0.5]})
effects = fitted.effects(rows, baseline=-0.5, value=1)
print("conditional effects:", np.round(effects.values, 2))
print("true conditional effects:", np.array([2.625, 3.0, 3.375]))
print("query interval available:", effects.std_error is not None)
```

```output from="continuous_treatments.py"
partially linear: 2.24
continuous DR: 2.99
bandwidth: 0.116
conditional effects: [2.58 2.99 3.4 ]
true conditional effects: [2.625 3.    3.375]
query interval available: False
```

The partially linear fit summarizes a slope, so multiplying it by the dose change
doesn't recover this nonlinear contrast. Continuous DR estimates the two dose
responses separately. This example illustrates the distinction; it isn't a
benchmark showing one method always wins.

## Choose the method for each variable

`estimate_contrast(graph, context, treatment, outcome, baseline=..., value=...,
method=...)` handles one treatment variable per call. Choose independently for
another variable; there is no global method selection.

| Method | Assumption | Population result | Query-row result |
|---|---|---|---|
| `partially_linear` | Homogeneous linear treatment effect, flexible covariate response | Slope times the dose change, with scaled uncertainty | The same effect per row, with shared slope uncertainty |
| `continuous_dr` | Smooth dose response and conditional treatment density | Difference of paired kernel DR scores, with a pointwise asymptotic interval | Conditional expected-effect predictions; no per-row interval |

For repeated queries, use `DML(...).fit(identified, context)` or
`ContinuousDR(...).fit(identified, context)`. Both expose
`contrast(baseline=..., value=...)` and `effects(rows, baseline=..., value=...)`.
Continuous DR reuses the final model for each dose pair across query batches.
Refitting the context clears its caches.

The existing `DML.effect()` and `DML.cate()` still report a **unit slope**.
`DML.contrast()` scales that slope by the dose change; reversing the contrast
reverses its sign, not its standard error. Averaging its row effects preserves the
shared uncertainty. Duplicating query rows cannot make the slope more precise.

## Numerical and classification outcomes

For a numerical outcome, the effect is a change in its expected value. For a
binary outcome, encode the desired event as 1 and the other event as 0: the effect
is a probability difference. An effect of 0.08 means eight percentage points.
Explicit contrasts preserve that meaning even when categorical labels are stored
in reversed order. DML's legacy slope API retains its original category coding;
new contrasts use raw numerical treatment values and outcome event 1.

Multiclass outcomes, joint changes to several treatments, and categorical
multi-arm treatments aren't implemented by these methods. Query effects are
conditional averages, not individual structural counterfactuals. Heterogeneity
is modeled over the graph-selected adjustment covariates.

## Prediction and density models

Pass an outcome learner or an `outcome`/`final` learner map through
`ContinuousDR(learners=...)`. By default, the outcome learner is OLS for numerical
outcomes or logistic regression for binary outcomes. The final effect regressor
is OLS. More flexible learners can model richer responses; the final learner must
always predict numerical effect scores, even for a classification outcome.

The separate `density=` argument requires a `TreatmentDensity` with
`fit(X, treatment)` and `pdf(X, treatment)`. The latter returns one conditional
density per row in inverse treatment units. A regressor that only predicts the
mean treatment is not a density estimator. The default `GaussianTreatmentDensity`
assumes a linear conditional mean and homoskedastic Gaussian residuals. Replace
it when that assumption is inappropriate. Each fold fits fresh nuisance models.
No hosted model is called by default, and this guide doesn't qualify Kumo or any
other remote learner for causal estimation.

## Bandwidth, support and uncertainty

Continuous DR uses the cross-fitted kernel score from
[Colangelo and Lee, equation (1)](https://arxiv.org/html/2004.03036v7):

```text
m(a, X) + K_h(T-a) / f(a|X) * (Y-m(a,X))
```

Both nuisances are evaluated at the requested dose `a`, using fits that exclude
the scored row's fold. Contrast scores are differenced before averaging and
computing the standard error, retaining dependence between the dose estimates.

The Epanechnikov bandwidth defaults to `sd(T) * n**(-1/4)` using the context sample
standard deviation. An explicit positive `bandwidth` is in treatment units. This
is an undersmoothing heuristic, not a tuned optimum. Results record bandwidth,
local effective sample sizes, minimum estimated density and procedure version.

Continuous DR refuses doses outside the observed range, within a bandwidth of
its boundaries, with density below `min_density` (default `1e-6` in inverse
treatment units), or with effective local sample size below `min_effective_n`
(default 10). DML checks the observed range. These checks diagnose some support
failures; they don't prove positivity. Neither method silently clips effects or
weights. Numeric query covariates retain their values; categorical query
covariates use context encoding and reject unseen levels.

Double robustness for point doses is **asymptotic** as bandwidth shrinks. At a
finite bandwidth, smoothing bias remains. `average.interval()` is a pointwise
normal approximation without bias correction: it needs appropriate
undersmoothing and nuisance convergence rates, not merely one correct nuisance.
It doesn't account for graph selection or simultaneous comparisons. A large
user-selected bandwidth can invalidate the approximation.

`effects.interval()` raises because query-row uncertainty isn't available.
`effects.ate()` averages those predictions over the query rows without an
interval; it isn't the fitted population's DR estimate or a transportability
guarantee. Both methods rely on the graph's causal assumptions, which the data
alone cannot establish.

Continue with [average effects](estimate-average-effect.md) or
[heterogeneous effects](heterogeneous-effects.md).
