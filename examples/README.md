# Examples

Short, runnable scripts. The top-level ones load a real dataset, run a causal
analysis, and print the estimate next to the answer the literature already measured.
The `cate/` and `scm/` folders are cookbooks on simulated worlds with a planted truth,
and the two hidden-regime scripts seal a switch into a simulation and recover it blind.

Run any of them after an editable install (`uv sync`):

```
uv run python examples/adjustment_birthweight.py
```

## Real datasets

Each script uses a loader from `ergodic.datasets`, so the data comes with its causal
question attached (the treatment, the outcome, the confounders, and the target).

| Script | Dataset | Question | What it shows |
| --- | --- | --- | --- |
| `adjustment_birthweight.py` | birthweight | Does smoking lower birthweight? | Backdoor adjustment, naive -275 g moves to -222 g (literature: -225 g) |
| `confounding_berkeley.py` | berkeley | Did admissions favor men? | Simpson's paradox: a pooled gap reverses under the confounder |
| `double_ml_pension.py` | pension401k | Effect of 401(k) eligibility on wealth | Double machine learning, $19.5k naive falls to about $9k |
| `synthetic_control_smoking.py` | smoking | Did Proposition 99 cut sales? | Synthetic control with placebo inference |
| `discovery_asia.py` | asia | Recover the network structure | Constraint-based discovery scored against a known graph |

The analyses behind these numbers are checked in `tests/test_datasets.py`, so they
stay honest as the estimators change.

## Hidden regimes

Neither script loads a dataset. Each generates a world with a sealed hidden switch,
recovers it blind with `ergodic.latent`, and unseals the switch only to grade the
answer. They need the latent extra (`uv sync --extra latent`) and a few minutes of
sampling.

| Script | What it shows |
| --- | --- |
| `latent_switch.py` | One level-shift world through all three verbs: `recover()` finds when the switch was on, `certify()` tests whether a regime exists against a simulated null, and `calibrate()` measures the false-fire rate on control worlds. |
| `hidden_regimes.py` | A notebook-style tour (`# %%` cells) over five sealed worlds (control, level, spread, rhythm, amplitude): which delta block fires on each and how well the recovered path matches the switch, then `certify()` and `calibrate()`. Add `--group docs` to draw the figures. |

## CATE cookbook (`cate/`)

Who responds, and how sure you can be, per unit. [`cate/README.md`](cate/README.md)
lists the numbers each script prints.

| Script | What it shows |
| --- | --- |
| `cate/01_cate_basics.py` | One `cate()` call returns a per-unit effect; every accessor on the result. |
| `cate/02_meta_learners_compared.py` | S, T, X, DR, R and DML learners on one world with a known effect gradient. |
| `cate/03_is_there_heterogeneity.py` | The omnibus calibration test, on a homogeneous world and on a real gradient. |
| `cate/04_where_is_the_effect_higher.py` | BLP, GATES and CLAN find the real modifier and clear a decoy. |
| `cate/05_targeting_with_rate.py` | RATE values a predicted-effect ranking, then prices a treat-or-not decision. |
| `cate/06_per_unit_uncertainty.py` | Three ways to a per-unit interval: bootstrap, analytic, and a Bayesian slot. |
| `cate/07_cate_vs_ite.py` | The conditional average next to the individual counterfactual from `SCM.ite()` (needs the `scm` extra). |
| `cate/08_identification_gating.py` | `cate()` answers on an identified graph and refuses on graphs that don't license it. |
| `cate/09_case_study_uplift_end_to_end.py` | A retention discount from a confounded read to a priced targeting rule. |
| `cate/10_causal_forest.py` | The honest causal forest, its per-unit jackknife band, and `variable_importance()`. |
| `cate/11_policy_trees.py` | `policy_tree` turns a ranking into a one-split rule and reports its policy value. |
| `cate/12_model_selection.py` | `select_cate` ranks learners by doubly robust tau-risk and builds an ensemble. |
| `cate/13_case_study_phase2.py` | A B2B win-back program through model selection, the forest and a depth-2 policy tree. |

## SCM cookbook (`scm/`)

One fitted structural causal model answers association, intervention and
counterfactual questions. These scripts need the `scm` extra (PyMC).
[`scm/README.md`](scm/README.md) lists the numbers each script prints.

| Script | What it shows |
| --- | --- |
| `scm/01_three_rungs.py` | See, do and imagine on one fit: three questions, three different numbers. |
| `scm/02_honesty_and_tipping_point.py` | A non-identified effect returns a `Sensitivity` curve, and you read where its sign tips. |
| `scm/03_causal_price_optimization.py` | Pricing off the interventional demand curve instead of the observed one. |
| `scm/04_counterfactual_recourse.py` | The smallest real change that flips a denied applicant, counting downstream effects. |
| `scm/05_counterfactual_fairness.py` | A per-person audit that splits a group gap into a direct and an indirect part. |
| `scm/06_causal_attribution.py` | Adjusting for purchase intent reverses a naive channel ranking. |
| `scm/07_but_for_accounting.py` | A campaign's incremental revenue as a per-customer but-for ledger. |
| `scm/08_off_policy_evaluation.py` | Value a targeting policy you never ran, from per-customer counterfactuals. |
| `scm/09_root_cause_attribution.py` | Cancel each step's own noise to find which upstream step caused one bad outcome. |
| `scm/10_frontdoor_unmeasured_confounder.py` | Identify an effect through a confounder you never measure, via the front door. |
