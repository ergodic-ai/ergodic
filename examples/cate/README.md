# CATE cookbook

An average effect tells you whether a treatment works. The conditional average
treatment effect (CATE) tells you *for whom*, and that is where the money and the
decisions are. These nine scripts put the meta-learner and explainability layer to
work, from the result surface to a full priced uplift case study.

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

| Script | The idea | The headline |
| --- | --- | --- |
| `01_cate_basics.py` | From one number to a function | One `cate()` call returns a per-unit vector: the average lift is 1.0, but the discount runs from -2.6 for new customers to +4.0 for loyal ones. `.values`, `.interval()`, `.ate()`, `.to_frame()`, and `np.mean(result)` all read off the one result. |
| `02_meta_learners_compared.py` | Five learners, one truth | S/T/X/DR/R/DML on the same `tau(z)=1+z` world. All recover the average ~1.0, but with a linear learner the S-learner and DML report ONE effect for everyone (sd 0), while T/X/DR/R track the gradient (corr 1.00 with the truth). |
| `06_per_unit_uncertainty.py` | How sure are we, per person | Three roads to a per-unit band on one surface: the bootstrap, the DR analytic standard error, and a Bayesian-slot posterior. Each brackets `tau(z)=1+z` at ~its nominal 90% (coverage ~1.0 here) with widths 0.20-0.26. |
| `08_identification_gating.py` | cate() only answers when the graph licenses it | Runs on an adjustment-identified graph (ate 1.03), refuses on a bow ("not identifiable") and on the front-door family ("needs adjustment-identification"), and points neural/foundation `method=` names at their optional extras. The refusals are the feature. |

## Explainability (the Tier-B characterization)

| Script | The idea | The headline |
| --- | --- | --- |
| `03_is_there_heterogeneity.py` | Prove it's there before you explain it | The omnibus calibration test on a homogeneous null vs a real gradient. On the null it does NOT reject (p 0.44, BLP slope CI covers 0) despite noise spread in the raw CATE; on the gradient it rejects hard (p 0.0000, slope 0.93). |
| `04_where_is_the_effect_higher.py` | Where, and for whom | BLP, GATES, and CLAN on a world with one real modifier (tenure) and one decoy (age). All three finger tenure (slope 1.01, GATES monotone with a 2.74 top-minus-bottom, CLAN diff 2.79) and clear age (slope -0.01, CI covers 0). |
| `05_targeting_with_rate.py` | Value a targeting rule, in dollars | RATE values the predicted-effect ranking: AUTOC 0.85 (rejects no-HTE), Qini 0.26. Turning the ranking into a treat/don't decision (cost 0.5) is worth ~$17,500 more than blanket treatment on a 100,000-customer book. |

## Frontier

| Script | The idea | The headline |
| --- | --- | --- |
| `07_cate_vs_ite.py` | The average for people like you, vs your own counterfactual | The conditional-average CATE (smooth, sd 0.13, corr 1.0 with z) next to the individual `SCM.ite()` counterfactual (a discrete flip in {-1,0,1}) on one binary-outcome world. Both average to the same ATE (~0.23) and share the `PerUnitEffect` surface; `ITEResult.explain()` raises with its pointer. |
| `09_case_study_uplift_end_to_end.py` | A retention-discount program, end to end | The headline. A confounded 6.58 naive read becomes an honest 1.89 adjusted ATE; five learners recover `tau=2+3*tenure`; the calibration test confirms it; BLP/GATES/CLAN name tenure (slope 3.16) and clear the support decoy; and a RATE-validated targeting rule is worth ~$270k over blanket on a 250k book, landing within ~$2,900 of the true-`tau` oracle. |

## Notes

- These are demonstrations, not part of the docs build, so they are free to lean on
  simulated worlds with planted truths. The numbers above are golden under their
  seeds.
- They use the public CATE surface: the `cate()` facade, the `CATEResult` /
  `ITEResult` / `PerUnitEffect` result types, `CATEResult.explain()` and the
  `Heterogeneity` record (BLP, GATES, CLAN, the calibration test, RATE), and
  `SCM.ite()`. Script 06 defines a small self-contained Bayesian-linear learner to
  show the posterior path without a heavy dependency.
- A note on the RATE interval: the two-sided CI is documented as anti-conservative,
  so script 05 uses `rejects()` (the one-sided test) for significance and reads the
  CI as a rough band.
- The case study (`09`) is a candidate to be promoted to a `docs/learn/` guide once
  the docs redesign settles; it already walks the full discover-identify-estimate-
  explain-act arc against a planted truth.
