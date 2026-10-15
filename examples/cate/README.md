# CATE cookbook

An average effect tells you whether a treatment works. The conditional average
treatment effect (CATE) tells you *for whom*, and that is where the money and the
decisions are. These scripts put the meta-learner, the causal forest, and the
explainability and policy layers to work, from the result surface to a full priced
uplift case study.

Every script is self-contained and runs on its own. It builds a small world with a
*known* heterogeneous truth, runs `cate()` (or `SCM.ite()`), and prints the naive
read next to the recovered effect next to the planted truth, so the gap is impossible
to miss. The data is simulated on a fixed seed, so the numbers below reproduce.

Run any of them after `uv sync`:

```
uv run python examples/cate/01_cate_basics.py
```

Script 07 also fits an `ergodic.scm.SCM`, so it needs the `scm` extra (PyMC); pin the
BLAS threads when you run it:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python examples/cate/07_cate_vs_ite.py
```

## Foundations

| Script | The idea | What it shows |
| --- | --- | --- |
| `01_cate_basics.py` | From one number to a function | A single `cate()` call returns a per-unit vector: the average lift is 1.0, but the discount runs from -2.6 for new customers to +4.0 for loyal ones. `.values`, `.interval()`, `.ate()`, `.to_frame()`, and `np.mean(result)` all read off the same result. |
| `02_meta_learners_compared.py` | Five learners, one truth | S/T/X/DR/R/DML on the same `tau(z)=1+z` world. All recover the average ~1.0, but with a linear learner the S-learner and DML report a single effect for everyone (sd 0), while T/X/DR/R track the gradient (corr 1.00 with the truth). |
| `06_per_unit_uncertainty.py` | How sure are we, per person | Three roads to a per-unit band on one surface: the bootstrap, the DR analytic standard error, and a Bayesian-slot posterior. Each brackets `tau(z)=1+z` at close to its nominal 90% (coverage ~1.0 here) with widths 0.20-0.26. |
| `08_identification_gating.py` | cate() only answers when the graph licenses it | Runs on an adjustment-identified graph (ate 1.03), refuses on a bow ("not identifiable") and on the front-door family ("needs adjustment-identification"), and points neural/foundation `method=` names at their optional extras. The refusals are the point. |

## Explainability

| Script | The idea | What it shows |
| --- | --- | --- |
| `03_is_there_heterogeneity.py` | Prove it's there before you explain it | The omnibus calibration test on a homogeneous null vs a real gradient. On the null it does not reject (p 0.44, BLP slope CI covers 0) despite noise spread in the raw CATE; on the gradient it rejects hard (p 0.0000, slope 0.93). |
| `04_where_is_the_effect_higher.py` | Where, and for whom | BLP, GATES, and CLAN on a world with one real modifier (tenure) and one decoy (age). All three finger tenure (slope 1.01, GATES monotone with a 2.74 top-minus-bottom, CLAN diff 2.79) and clear age (slope -0.01, CI covers 0). |
| `05_targeting_with_rate.py` | Value a targeting rule, in dollars | RATE values the predicted-effect ranking: AUTOC 0.85 (rejects no-HTE), Qini 0.26. Turning the ranking into a treat/don't decision (cost 0.5) is worth about $17,500 more than blanket treatment on a 100,000-customer book. |

## Frontier

| Script | The idea | What it shows |
| --- | --- | --- |
| `07_cate_vs_ite.py` | The average for people like you, vs your own counterfactual | The conditional-average CATE (smooth, sd 0.13, corr 1.0 with z) next to the individual `SCM.ite()` counterfactual (a discrete flip in {-1,0,1}) on one binary-outcome world. Both average to the same ATE (~0.23) and share the `PerUnitEffect` surface; `ITEResult.explain()` raises with its pointer. |
| `09_case_study_uplift_end_to_end.py` | A retention-discount program, end to end | The whole pipeline in one script. A confounded 6.58 naive read becomes an honest 1.89 adjusted ATE; five learners recover `tau=2+3*tenure`; the calibration test confirms it; BLP/GATES/CLAN name tenure (slope 3.16) and clear the support decoy; and a RATE-validated targeting rule is worth ~$270k over blanket on a 250k book, landing within ~$2,900 of the true-`tau` oracle. |

## Advanced: the causal forest, policy trees, and model selection

| Script | The idea | What it shows |
| --- | --- | --- |
| `10_causal_forest.py` | Per-unit effects with honest, native error bars | The honest causal forest recovers `tau(tenure)=1+1.5*tenure` (corr 0.99, ATE 0.99) and carries its own per-unit band: the infinitesimal-jackknife interval is `kind="analytic"` with widths that vary per unit (0.00 to 2.49), unlike the flat ~0.17 bootstrap band a meta-learner reports. `variable_importance()` fingers tenure (0.67) over the region decoy (0.33). |
| `11_policy_trees.py` | From a ranking to a rule you can act on | On a sign-flipping `tau(tenure)=2*tenure` world, RATE confirms the ranking (AUTOC 1.73, rejects no-HTE), then `policy_tree` turns it into a rule: split tenure at -0.03 (the planted flip point), treat the positive side. Policy value 0.82 per customer (oracle 0.80) beats treat-all/none (regret 0.81), worth ~$162k over blanket on a 200k book. |
| `12_model_selection.py` | Which learner do I trust -- the data decides | On a pure-interaction `tau=3*tenure` world the pooled S-learner is misspecified; `select_cate`'s doubly-robust tau-risk ranks it last (12.4, ~3.6x the others) and picks the right one (t_learner, 3.46). The Q-aggregation ensemble matches the best single (3.46) and puts ~0 weight on the S-learner. `causal_forest` is scored among the candidates. |
| `13_case_study_phase2.py` | The advanced toolkit, end to end | A B2B win-back program with `tau(engagement)=2+4*engagement`. A confounded -0.30 naive read (sign-flipped) becomes an honest 2.04 ATE; `select_cate` ranks S/T/X/DR/R/forest and picks the winner (S-learner worst at 52.9); the calibration test rejects; BLP (engagement slope 4.0, decoys cleared), GATES (-3.66 to 7.54), and the forest's `variable_importance` (engagement 0.51) all name the driver; and a depth-2 `policy_tree` rule is worth ~$326k over blanket on a 300k book, within ~$4,200 of the oracle. |

## Notes

- These are demonstrations, not part of the docs build, so they are free to lean on
  simulated worlds with planted truths. The numbers above are golden under their
  seeds.
- They use the public CATE surface: the `cate()` facade (including
  `method="causal_forest"`), the `CATEResult` / `ITEResult` / `PerUnitEffect` result
  types, `CATEResult.explain()` and the `Heterogeneity` record (BLP, GATES, CLAN, the
  calibration test, RATE, `policy_tree`), and `SCM.ite()`. The advanced scripts also
  use `CausalForest` (with `variable_importance()`), the standalone `policy_tree` /
  `PolicyTree`, and `select_cate` / `SelectionReport`. Script 06 defines a small
  self-contained Bayesian-linear learner to show the posterior path without a heavy
  dependency.
- A note on the RATE interval: the two-sided CI is documented as anti-conservative,
  so script 05 uses `rejects()` (the one-sided test) for significance and reads the
  CI as a rough band.
- The case study (`09`) is a candidate to be promoted to a `docs/learn/` guide once
  the docs redesign settles; it already walks the full discover-identify-estimate-
  explain-act arc against a planted truth.
