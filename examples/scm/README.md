# SCM cookbook

A structural causal model is the one object that answers all three rungs of the causal
ladder from a single fit: what the data shows (association), what an intervention would
do (`do`), and what would have happened to one individual (the counterfactual). These
ten scripts put that to work, from the textbook warm-up to uses that are not obvious at
all.

Every script is self-contained and runs on its own. It builds a small world with a
*known* planted truth, fits an `ergodic.scm.SCM`, and prints the naive answer next to
the causal one next to the truth, so the gap is impossible to miss. The data is
simulated on a fixed seed (the SCM needs PyMC, the `scm` extra), so the numbers below
reproduce.

Run any of them after `uv sync`:

```
uv run python examples/scm/01_three_rungs.py
```

## Foundations

| Script | The idea | What it shows |
| --- | --- | --- |
| `01_three_rungs.py` | See, do, imagine on one fitted model | The same fork reads 2.9 by association, 1.5 under `do`, and moves one unit by exactly 1.5 in counterfactual. Three questions, three numbers, one fit. |
| `02_honesty_and_tipping_point.py` | A non-identified effect returns a curve, not a number | `ate()` hands back a `Sensitivity` with no `.value` accessor at all. It doubles as a decision tool: the effect here stays positive unless hidden confounding explains the *entire* observed covariance. |

## Applied

| Script | The idea | What it shows |
| --- | --- | --- |
| `03_causal_price_optimization.py` | Price off the causal demand curve, not the observed one | Promotions move price and demand together, so the observed curve looks 2.8x as elastic as the truth. Pricing off it sets $8.50 and sacrifices 19% of profit; the `do`-price curve sets the true optimum, $13.50. |
| `04_counterfactual_recourse.py` | The minimal real change that flips a denial | For a denied applicant, the SCM ranks the levers by their *whole-graph* effect. Income looks weak by its own coefficient (0.4) but is the cheapest lever once its cascade into savings and debt is counted, about 2.4x less change than the naive read demands. |
| `05_counterfactual_fairness.py` | Would the decision change if only the group flipped? | A per-individual twin-network audit, splitting the group gap into a legitimate through-qualification part and an illegitimate direct part (about 0.5 of score here). The direct part is what a fair model must remove. |
| `06_causal_attribution.py` | The best-looking channel can be the worst | Ads reach high-intent buyers, so naive attribution crowns them. Adjusting for intent reverses it: email's causal lift is several times the ad's. Spending against the naive ranking backs the weaker channel. |
| `07_but_for_accounting.py` | A campaign's incremental revenue, customer by customer | The CFO's question: what would promoted customers have spent without the promotion? The naive read credits the promotion with $236k; the summed but-for world lands at $101k, against the planted $103k. The honest incremental, framed as a per-customer ledger. |

## Frontier

These are the uses where the per-individual counterfactual is irreplaceable: no
population average or single estimand answers them.

| Script | The idea | What it shows |
| --- | --- | --- |
| `08_off_policy_evaluation.py` | Score a targeting policy you never ran | The twin network labels each customer "saveable" or not (a per-unit query no average can give), so any policy can be valued off the fitted model. The targeted policy saves the same customers as blanket-offering at a fifth of the cost, beating both the historical and the blanket policy on net value. |
| `09_root_cause_attribution.py` | Which upstream step caused one bad outcome | A shock at B sinks B, C, and the Output together, so a correlation view blames all three. Cancelling each step's own noise in counterfactual cleanly fingers B: the cause recovers the Output, its downstream symptoms do not. |
| `10_frontdoor_unmeasured_confounder.py` | Identify through a confounder you can never measure | Smoking to cancer with a hidden genotype: adjustment is impossible, but the front door through tar identifies the effect at 1.2 where the naive regression reports 1.9. The SCM's latent-district fit is the engine that most causal tools lack. |

## Notes

- These are demonstrations, not part of the docs build, so they are free to lean on
  simulated worlds with planted truths. The numbers are golden under their seeds.
- They use the public SCM surface (`sample`, `interventional`/`do`, `counterfactual`,
  `ate`). Script 10 also peeks at `scm._latent` to show the confounded-district split
  that makes front-door identification work; that is internal and shown only to teach.
- The honesty layer is load-bearing throughout: `ate()` returns a posterior point only
  when the effect is identified, and a `Sensitivity` bound when it is not (script 02).
