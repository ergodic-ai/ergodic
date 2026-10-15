# Estimate a mediated effect

Split the effect of a treatment into the part that flows through a mediator and the
part that doesn't. You'll check that the split is identifiable, estimate it with
intervals and compare it with the values planted in a simulation.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - A binary treatment, a binary or low-cardinality categorical mediator, and a
      continuous or binary outcome.

## Run the program

The simulation is a job-training programme. Training raises wages directly, and it
also makes workers more likely to get certified, which raises wages too. Prior
experience affects all three.

```python file="mediation.py"
import numpy as np
import pandas as pd

from ergodic import dag, tabular
from ergodic.identification import identify_mediation
from ergodic.inference import estimate_mediation


def expit(z):
    return 1 / (1 + np.exp(-z))


# 1. Simulate 2,000 workers. The direct effect of training is 1.0.
rng = np.random.default_rng(0)
n = 2_000
experience = rng.normal(size=n)
training = rng.binomial(1, expit(0.5 * experience))
certified = rng.binomial(1, expit(-1 + 2 * training + 0.5 * experience))
wage = 20 + 1.0 * training + 3.0 * certified + 1.5 * experience + rng.normal(0, 2, n)
frame = pd.DataFrame(
    {"experience": experience, "training": training, "certified": certified, "wage": wage}
)
data = tabular(frame, types={"training": "categorical", "certified": "categorical"})

# 2. Draw the graph and check that the split is identifiable.
graph = dag([
    "experience -> training", "experience -> certified", "experience -> wage",
    "training -> certified", "training -> wage", "certified -> wage",
])
print(identify_mediation(graph, "training", "wage", "certified"))

# 3. Estimate the direct and indirect effects.
split = estimate_mediation(
    graph, data, "training", "wage", "certified", treated=1, random_state=0, n_boot=200
)
for name, effect in [("direct", split.nde), ("indirect", split.nie), ("total", split.total)]:
    low, high = effect.interval()
    print(f"{name:8} {effect.value:.2f}  95% interval [{low:.2f}, {high:.2f}]")
print(f"proportion mediated: {split.proportion_mediated:.2f}")

# 4. The planted indirect effect: 3.0 times the rise in certification.
planted = 3.0 * np.mean(expit(1 + 0.5 * experience) - expit(-1 + 0.5 * experience))
print(f"planted: direct 1.00, indirect {planted:.2f}")

# 5. A confounder of certification and wage that training itself affects.
confounded = dag([*graph.edge_list(), "training -> motivation",
                  "motivation -> certified", "motivation -> wage"])
print(identify_mediation(confounded, "training", "wage", "certified"))
```

```output from="mediation.py"
MediationID(identifiable, ['training'] -> [certified] -> ['wage'], W={experience}, cde=True)
direct   1.17  95% interval [0.98, 1.38]
indirect 1.21  95% interval [1.04, 1.37]
total    2.42  95% interval [2.19, 2.66]
proportion mediated: 0.50
planted: direct 1.00, indirect 1.32
MediationID(NDE/NIE not identifiable, ['training'] -> [certified] -> ['wage'], cde=True)
```

- **Step 2.** The split is identifiable when you adjust for `experience`, written
  `W={experience}`. `cde=True` says the controlled direct effect is identifiable too.
- **Step 3.** Training raises the wage by about 1.17 directly and 1.21 through
  certification. Each interval comes from 200 bootstrap resamples of the rows.
- **Step 4.** Both intervals contain the planted values, 1.00 and 1.32.
- **Step 5.** Motivation now confounds certification and wage, and training itself
  changes it. Adjusting for motivation would block part of the effect; leaving it out
  leaves the mediator confounded. No set of pre-treatment variables fixes both, so
  the natural effects are not identifiable. The controlled direct effect still is.

## Three effects

- The **natural direct effect** (NDE) is the change in the outcome when you switch the
  treatment on but keep the mediator where it would have been without treatment.
- The **natural indirect effect** (NIE) is the change when you keep the treatment off
  but move the mediator to where treatment would have put it.
- The **controlled direct effect** (CDE) fixes the mediator at one level for everyone.
  `split.cde` maps each mediator level to its estimate.

`identify_mediation` returns a `MediationID`. It is truthy when the natural effects
are identifiable. It carries the formulas (`nde_estimand`, `nie_estimand`,
`cde_estimand`), the adjustment set (`covariates`) and, in `assumptions`, the
assumptions in words.

!!! caution "The natural effects rest on an assumption no data can test"
    They compare outcomes across two worlds that never both happen: treated, with the
    mediator at its untreated value. The graph implies that assumption only if you
    read it as a structural model with independent errors, and no experiment can
    check it. The CDE needs no such assumption.

## Read the estimate

`estimate_mediation` returns a `MediationEstimate`. `nde`, `nie` and `total` are
`EffectEstimate` objects with `value`, `std_error` and `interval()`, from a bootstrap
of `n_boot=1000` resamples by default.

The method is g-computation: fit the outcome given treatment, mediator and covariates,
fit the mediator given treatment and covariates, then average the mediation formula
over your rows. The outcome model is ordinary least squares, or logistic regression
for a binary outcome. The mediator model is logistic regression. Pass
`learner={"outcome": ..., "mediator": ...}` to use others; the mediator needs a
classifier.

`nde` and `nie` are the pure effects: each holds the other path at the untreated
level. `total` is the total effect of switching treatment on, computed directly. When
treatment and mediator interact in their effect on the outcome, the pure effects don't
add up to it, and `total - nde - nie` is the mediated interaction. The outcome model
includes that interaction, so it never quite vanishes in a fit, which is why the
program's total is close to, but not exactly, the sum of its two parts.

`proportion_mediated` is `nie / total`, the pure indirect effect's share. Under an
interaction it and `nde / total` need not sum to one. It is unstable when the total
effect is near zero, and `proportion_mediated_flagged` is then `True`.

<details markdown="1"><summary>What if my mediator is continuous?</summary>

`estimate_mediation` raises: it supports discrete mediators only. A treatment with
more than two levels also raises. `identify_mediation` itself has no such limit, so
you can still check identifiability.

</details>

## Next steps

<div class="cards" markdown="1">

- [Find adjustment sets, front-door sets and instruments](adjustment-sets.md): the sets behind the mediation formula.
- [Estimate an average effect](estimate-average-effect.md): the total effect alone.
- [Python reference](reference-inference.md): `estimate_mediation` and `MediationEstimate`.

</div>
