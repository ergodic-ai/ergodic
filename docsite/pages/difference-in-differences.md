# Difference-in-differences and event studies

Estimate the effect of a treatment that units adopt at different times, by comparing
their change in outcome with the change among units that have not adopted. Then
trace the effect year by year.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - Panel data: one row per unit and period, an outcome, and a treatment coded
      0/1 or `True`/`False` that stays on once it turns on.

## Run the program

The program simulates stores that start a promotion in 2014, in 2017 or never.
Because the data is simulated, you can check each number against the truth.

```python file="difference_in_differences.py"
import numpy as np
import pandas as pd
from ergodic import did, event_study, panel

# 1. Simulate 120 stores over 2010-2019. Some start a promotion in 2014, some in
#    2017, the rest never. The effect is 1.0 in the first year and grows 0.5 a year.
rng = np.random.default_rng(0)
rows = []
for store in range(120):
    start = int(rng.choice([2014, 2017, 0]))  # 0: never starts
    level = rng.normal(0, 1)
    for year in range(2010, 2020):
        promo = start > 0 and year >= start
        effect = 1.0 + 0.5 * (year - start) if promo else 0.0
        sales = level + 0.3 * (year - 2010) + effect + rng.normal(0, 0.5)
        rows.append({"store": store, "year": year, "promo": promo, "sales": sales})

# 2. Build the panel: one row per store and year.
data = panel(pd.DataFrame(rows), entity="store", time="year")
print(data.n_entities, "stores,", data.n_periods, "years, balanced:", data.is_balanced)

# 3. Estimate the average effect on the stores that ran the promotion.
result = did(data, "sales", "promo", random_state=0)
low, high = result.interval()
print(f"{result.flavor}: ATT {result.value:.2f}, 95% interval [{low:.2f}, {high:.2f}]")

# 4. Compare with the two-way fixed-effects regression.
twfe = did(data, "sales", "promo", strategy="twfe", random_state=0)
print(f"twfe: ATT {twfe.value:.2f}")

# 5. Trace the effect by years since the promotion started.
study = event_study(data, "sales", "promo", leads=3, random_state=0)
for row in study.by_event_time:
    low, high = row.interval
    print(f"{row.event_time:+d}  {row.estimate:5.2f}  [{low:5.2f}, {high:5.2f}]")
```

```output from="difference_in_differences.py"
120 stores, 10 years, balanced: True
group_time: ATT 1.97, 95% interval [1.78, 2.15]
twfe: ATT 1.50
-3   0.06  [-0.19,  0.33]
-2   0.05  [-0.17,  0.28]
+0   1.11  [ 0.88,  1.35]
+1   1.69  [ 1.46,  1.94]
+2   1.89  [ 1.71,  2.09]
+3   2.70  [ 2.44,  2.98]
+4   3.12  [ 2.84,  3.42]
+5   3.47  [ 3.20,  3.78]
```

- **Step 2.** `panel` keys the rows by `(store, year)` and sorts them. Each pair
  must be unique.
- **Step 3.** Adoption is staggered, so `did` picks the `group_time` estimator. It
  compares each cohort with the stores that never ran the promotion. The true
  average effect, computed the same way, is 1.86. The interval contains it.
- **Step 4.** The two-way fixed-effects regression gives 1.50, well below the truth.
  With staggered adoption and an effect that grows, it mixes comparisons that
  partly cancel the effect (Goodman-Bacon, 2021).
- **Step 5.** Event time 0 is the first promoted year. The leads, −3 and −2, are
  near zero, as parallel trends predict. The lags rise as the true effect does:
  1.0, 1.5, 2.0, 2.5, 3.0 and 3.5.

## Choose the estimator

`strategy=` picks how `did` estimates the average effect on the treated (ATT).

| `strategy` | Use it when | How it works |
| --- | --- | --- |
| `"auto"` (default) | Always a safe start. | `canonical` for one adoption period, else `group_time`. Never `twfe`. |
| `"canonical"` | Every treated unit adopts in the same period. | Each unit's change from its pre-period mean to its post-period mean, treated minus comparison. |
| `"group_time"` | Units adopt at different times. | One effect per cohort and period (Callaway and Sant'Anna, 2021). Per cohort, the post-period effects are averaged, then cohorts are weighted by size. |
| `"twfe"` | You need the classic regression for reference. | Two-way fixed effects. Biased under staggered adoption when effects change over time. |

`comparison="never_treated"` (default) compares with units that never adopt and
raises if there are none. `comparison="not_yet_treated"` also uses units that adopt
later. `controls=` adds covariates.

!!! caution "The design rests on two assumptions"
    Parallel trends: without the treatment, treated and comparison units would have
    moved together. No anticipation: outcomes do not react before adoption. Each
    result carries them in `result.design.assumptions`. Leads away from zero in an
    event study are evidence against parallel trends.

<details markdown="1">
<summary>Results, inference and errors</summary>

- The headline result works like any estimate: `value`, `std_error`, `interval()`.
  `flavor` names the estimator that ran and `comparison` the pool.
- `result.by_group_time` lists every cohort-period cell. `study.by_event_time` has
  one entry per event time, with a standard error and a 95% percentile interval.
- `event_study(leads=, lags=)` bounds the window. Event time −1 is the reference
  period, so it is never reported.
- Inference is a unit bootstrap: `n_boot=` resamples of whole units (default
  1,000). `random_state=` makes it repeatable.
- `did` raises for a treatment that switches back from 1 to 0, a treatment that is
  not 0/1 or boolean, a missing value, or an empty comparison pool. The canonical
  estimator drops, with a warning, units that lack a pre or a post period.

</details>

## Next steps

<div class="cards" markdown="1">

- [Synthetic control and interrupted time series](synthetic-control.md): one treated unit, no comparison cohort.
- [Estimate an average effect](estimate-average-effect.md): adjust for confounders instead.
- [Python reference](reference-inference.md): every option of `did` and `event_study`.

</div>
