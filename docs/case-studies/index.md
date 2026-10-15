# Case studies

The learn series teaches each idea on a small simulated world, where the true answer is written into the code and you can grade any analysis against it. Most of these pages do the opposite. Each takes a dataset you may know, asks the question the original study asked, and checks ergodic against the number that study published.

The last two are a deliberate exception. Customer lifetime value and churn are simulated business worlds, because the question they ask, what an intervention does to a customer, has no public dataset with a measured answer to check against. There the planted truth is the answer key, the way the learn series works, but the analysis is the full applied pipeline the rest of these pages run.

So this is the applied tier. The concepts live in [the learn series](../learn/index.md), the full surface is in [the reference](../reference.md), any unfamiliar term is in the [glossary](../glossary.md), and here they meet on a real problem.

| Case study | Method | The question | What it recovers |
| --- | --- | --- | --- |
| [Smoking and birthweight](01-birthweight.py) | backdoor adjustment | Does maternal smoking lower birthweight? | naive -275 g, adjusted -222 g, near the published -231 |
| [Berkeley admissions](02-berkeley.py) | confounding, Simpson's paradox | Did 1973 admissions favor men? | a pooled -14 point gap reverses to +4 once you condition on department |
| [401(k) and wealth](03-pension401k.py) | double machine learning | What does 401(k) eligibility do to savings? | naive $19,500, a panel of flexible learners near $9,000 |
| [Proposition 99](04-synthetic-control.py) | synthetic control | Did California's tobacco program cut smoking? | -21 packs per capita, placebo p = 0.05 |
| [Discovery on a known network](05-discovery.py) | structure learning | Where does the graph come from? | the strong edges recovered, the rare ones honestly missed |
| [Customer lifetime value](06-clv.py) | doubly robust ATE, CATE targeting | What does a retention perk do, and who should get it? | naive $120, true $23, and lift targeting that makes $198k where value targeting loses money |
| [Customer churn](07-churn.py) | confounding by indication, discovery, targeting | Does the save offer reduce churn, or cause it? | naive +22 points, adjusted -11, near the planted -12 |

Every number on these pages is computed when the docs build, so the prose and the code can't drift apart. The real-data pages load from `ergodic.datasets`, where each loader hands back the dataset and its causal question together; the two simulated pages write their world at the top of the page, in plain sight.

A pattern repeats across the inference pages. The naive comparison is wrong because the groups differ, that difference shows up in a balance table, and adjustment closes the gap. Each page ends with a refutation check: scramble the treatment, and the effect should vanish. None of that proves an estimate right, but a failure is a real warning.
