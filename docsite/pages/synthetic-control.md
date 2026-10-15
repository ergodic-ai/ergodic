# Synthetic control and interrupted time series

Estimate the effect of a policy on one treated unit by building a weighted
combination of untreated units that tracks it before the policy. Then compare with
an interrupted time series, which uses the treated unit alone.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - Panel data with one treated unit, several untreated donor units, and a
      treatment date with at least 2 periods before it and 1 after.

## Run the program

The bundled smoking panel records cigarette sales per person in 39 US states from
1970 to 2000 (Abadie, Diamond and Hainmueller, 2010). California's Proposition 99,
a tobacco tax, took effect in 1989; the other 38 states are the donors.

```python file="synthetic_control.py"
from ergodic import interrupted_time_series, synthetic_control, timeseries
from ergodic.datasets import load_smoking

# 1. Load the panel: packs per person, by state and year.
smoking = load_smoking()
print(smoking.data.n_entities, "states,", smoking.data.periods[0], "to", smoking.data.periods[-1])

# 2. Fit a synthetic California from the other states, before 1989.
fit = synthetic_control(smoking.data, "cigsale", treated="California", at=1989)
for state, weight in sorted(fit.weights.items(), key=lambda item: -item[1])[:5]:
    print(f"  {state:12} {weight:.2f}")
print(f"pre-1989 tracking error: {fit.pre_rmspe:.1f} packs")

# 3. Read the effect: the mean gap after 1989, with placebo inference.
low, high = fit.interval()
print(f"effect {fit.value:.1f} packs, placebo band [{low:.1f}, {high:.1f}]")
print(f"placebo p-value {fit.p_value:.3f}, gap in {fit.periods[-1]}: {fit.gaps[-1]:.1f}")

# 4. Fit an interrupted time series on California alone.
rows = smoking.data.to_pandas()
california = rows[rows["state"] == "California"][["year", "cigsale"]]
its = interrupted_time_series(timeseries(california, time="year"), "cigsale", at=1989)
low, high = its.interval()
print(f"interrupted time series: {its.value:.1f} packs, 95% interval [{low:.1f}, {high:.1f}]")
```

```output from="synthetic_control.py"
39 states, 1970 to 2000
  Utah         0.21
  Nevada       0.14
  New Mexico   0.13
  Idaho        0.12
  Montana      0.11
pre-1989 tracking error: 2.3 packs
effect -21.0 packs, placebo band [-39.3, -2.7]
placebo p-value 0.051, gap in 2000: -27.9
interrupted time series: -28.3 packs, 95% interval [-40.8, -15.8]
```

- **Step 2.** Five states carry most of the weight. The weights are nonnegative and
  sum to one. Before 1989 the synthetic California misses the real one by about 2
  packs per person a year.
- **Step 3.** After 1989, California sold about 21 fewer packs per person a year
  than its synthetic copy. By 2000 the gap was about 28 packs.
- **Placebo inference.** The fit is rerun with each donor state as if it were
  treated. Only one placebo has a post-to-pre error ratio as large as
  California's, so the p-value is 2/39. The band adds and subtracts the 95th
  percentile of the placebo states' mean gaps, in absolute value.
- **Step 4.** The interrupted time series projects California's own pre-1989
  trend forward. Its estimate is larger, because it attributes all of the
  post-1989 change to the law.

## Read a synthetic control

`synthetic_control` returns a `SyntheticControlEffect`. `value` is the mean of
observed minus synthetic over the periods from `at` on. `observed`, `synthetic` and
`gaps` run over `periods`, ready to plot.

!!! caution "Check the pre-period fit first"
    The effect is only credible if the synthetic unit tracks the treated one before
    the policy. A large `pre_rmspe` relative to the outcome means the donors cannot
    reproduce the treated unit, and the gap after `at` is not an effect.

The weights fit the outcome path only. They do not match covariates such as income
or retail price, so they can differ from published synthetic controls that do. There is no sampling standard error
for one treated unit: `std_error` is `None`, and `interval()` is a placebo band,
not a confidence interval.

Name the treated unit in one of two ways: `treated=` and `at=`, or `treatment=`, a
0/1 column that marks exactly one unit. `donors=` restricts the donor pool. Donors
with a missing period are dropped with a warning; at least 2 must remain.

## When to use an interrupted time series

`interrupted_time_series(data, outcome, *, at, seasonality=None, hac_lags=None)`
needs no donors. It fits a level, a trend, a jump at `at` and a change in slope,
and reports the mean gap between the fit and the projected pre-trend. Its standard
errors are Newey-West, which allow for autocorrelated residuals.

- `level_change` and `slope_change` hold the jump and the slope change, each with a
  standard error.
- `seasonality=12` adds Fourier terms for a monthly cycle.
- It assumes the pre-period trend would have continued and that nothing else
  changed at `at`. A national decline after 1989, as here, breaks that assumption.
  Prefer synthetic control when donors exist.

<details markdown="1">
<summary>Requirements and errors</summary>

- Synthetic control: `at` must be one of the panel's periods, with at least 2
  periods before and 1 after. The treated unit must be observed in every period.
- Interrupted time series: at least 3 points on each side of `at`. Data must be a
  `TimeSeriesData` from `timeseries(frame, time=...)`.
- Both raise on a missing outcome value. Both record their assumptions in
  `result.design.assumptions`.

</details>

## Next steps

<div class="cards" markdown="1">

- [Difference-in-differences and event studies](difference-in-differences.md): many treated units adopting over time.
- [Notebooks](notebooks.md): the synthetic-control case study, step by step.
- [Python reference](reference-inference.md): every option of `synthetic_control`.

</div>
