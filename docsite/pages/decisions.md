# Rank interventions

Use a graph and context observations to compare actions on one target. The
intervention map names what can change. Defaults choose the reference and
alternative values from context; query rows only request personalised rankings.
This API is unreleased and requires the decision-pipeline source revision.

## A three-category treatment

This synthetic example has three actions and a continuous target. The effect of
an action depends on the baseline variable `z`. The graph is supplied here;
Welt's hosted decision pipeline also runs discovery and records its analysis graph.

```python file="rank_decisions.py"
import numpy as np
import pandas as pd
from ergodic import dag
from ergodic.decision import decide

rng = np.random.default_rng(8)
n = 4000
z = rng.uniform(-1, 1, n)
a = rng.integers(0, 3, n)
labels = np.array(["none", "small", "large"])[a]
y = z + a * (1 + .5 * z) + rng.normal(0, .2, n)
context = pd.DataFrame({"z": z, "action": labels, "outcome": y})
result = decide(
    dag(["z -> outcome", "action -> outcome"]),
    context,
    target="outcome",
    interventions={"action": {"baseline": "none", "values": ("small", "large")}},
    rows=pd.DataFrame({"z": [-.5, .5]}),
    seed=8,
)
for action in result["population"]:
    print(action["value"], round(action["effect"], 2))
print("query best actions:", [row["interventions"][0]["value"] for row in result["individual"]])
print("skipped:", len(result["skipped"]))
```

```output from="rank_decisions.py"
large 2.0
small 1.01
query best actions: ['large', 'large']
skipped: 0
```

## Choose the target and treatment types

Binary and multicategory treatments use cross-fitted, multi-arm DR scores over
the full context population. Numerical treatments default to `partially_linear`,
a homogeneous slope assumption. Select `continuous_dr` per variable when a
nonlinear dose response is required; see [numerical contrasts](continuous-treatments.md).
Only the implemented method names are accepted.

Numerical targets report differences in their units. Binary and multicategory
targets report class-probability differences. Set `target_kind="categorical"`
and `target_class` to the event you want to optimise. Boolean and 0/1 binary
targets default to True/1. Integer-coded categories need an explicit categorical
kind: class identifiers do not define a numerical utility.

For categorical outcomes the true class contrasts sum to zero. We enforce this
identity by projecting joint row-aligned scores before calculating means and
standard errors, including their cross-class covariance. Query predictions use
the same linear projection. This is not clipping, calibration or a guarantee
that finite-sample effects lie in [-1, 1].

## Defaults and interpretation

- Boolean/0–1 treatment: false to true.
- Categorical treatment: most frequent label to every other label; ties use stable label order.
- Numerical treatment: median to the 25th and 75th percentiles.

Every returned action records the actual baseline, destination, method and
adjustment set. `objective="minimize"` reverses the ranking direction. Rankings
compare one variable at a time; they do not combine actions or account for costs.
A row compares the same baseline/destination as the population. It does not use
the row's observed treatment as an implicit baseline.

Identification is conditional on the graph. The engine accepts adjustment or
null effects and reports other identification strategies as skipped. A valid
pretreatment adjustment set supplies the query features; unused columns do not
become evidence about an individual's potential outcome. Numeric range and
query-propensity checks refuse obvious extrapolation, but do not certify joint
support. A scientific refusal is atomic for the variable, never a zero effect.

Population intervals are pointwise and asymptotic. They exclude graph discovery,
ranking selection, multiple comparisons and continuous-DR smoothing bias. Query
effects are conditional expectations with no individual intervals or structural
ITE claim. All encodings, folds, defaults and nuisance fits use context only.

Pass regression learners in the `outcome` and `final` slots and a classifier
with normalised arm probabilities in `propensity`. Each fold gets a fresh model.
Welt's server uses its admitted native models; local ergodic defaults are simple
working models, not a guarantee of correct nuisance specification.

## Candidate binary ranking protocol

Explicit `protocol="decision-contrasts_v2"` keeps population DR estimation and
uses S/treatment toggling as the default for binary-treatment, binary-target query rankings.
The original protocol remains the default when the argument is omitted. `method`
is still per intervention: `s`, `dr`, or, with independent randomized validation,
`auto`. All other treatment/target defaults are unchanged. S is a simple
predeclared baseline, not an empirically proven universal winner.

For discrete treatments and a binary target, v2 fits the selected event once and
derives the opposite class contrast by negation. An outcome classifier supplies
probability means, not labels; DR's final learner remains a regressor. Pass
`learners_by_variable={"action": {"outcome": classifier, "final": regressor}}`
to override the shared learner slots. Population `method="dr"` and query
`method="s"` intentionally identify different estimators. The population ATE is
not the average of the S model's query effects.

Optional `validation` is a separate labeled dataframe with the context schema.
Supply `unit_id` and `validation_design="independent_bernoulli"`; context and
validation identifiers must be unique and disjoint, and are excluded from all
learning. The declaration asserts independent noninterfering units, truthful IDs,
pretreatment covariates and no validation-driven tuning or reuse. Merely renaming
the same units does not satisfy it. No validation row fits a model or preprocessing.

The initial gate supports binary treatment/outcome only. Each intervention supplies
`validation_probability`, the known assignment probability of its requested value
in the validation experiment, strictly between .01 and .99. It is not the learned
context propensity. For a fixed top-20% targeting policy, fractional boundary ties
preserve exactly 20% selected mass. Hoeffding bounds target the finite validation
cohort, conditional on frozen models/features and binary potential outcomes.
Three comparisons per prespecified contrast receive Bonferroni correction.

Auto chooses DR only if both DR-minus-S and DR-minus-random targeting gains have
strictly positive lower bounds; otherwise it chooses S and reports S's own evidence.
`ranking_validation.status` is `supported`, `evidence_against`, `inconclusive`, or
`not_assessed`. No failed gate silently zeroes estimates. Supported targeting means
an advantage over random selection of the same fraction, not benefit versus treating
nobody, accurate individual causal effects, a correct discovered graph, or future
population performance. Repeated selection using the same validation cohort is
outside the guarantee. Other assignment designs are not silently approximated.
