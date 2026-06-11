# Panel and time-series designs

Everything in the [inference guide](inference.md) identifies an effect by what you adjust for: the graph names the confounders, and the `Identification` record says why the comparison is fair. The designs on this page identify an effect by what comparison stands in for the counterfactual. Difference-in-differences uses the change among units that didn't adopt. Synthetic control uses a weighted basket of donor units. Interrupted time series uses the unit's own pre-period trend, projected past the break.

There's no graph in these calls, and that's deliberate. Each design rests on a claim about whole trajectories: untreated outcomes would have moved in parallel, the treated path lies in the convex span of the donors, the pre-trend would have continued. Edges and adjustment sets can't express claims like these. So the estimators read a `PanelData` or `TimeSeriesData` directly, skip `identify_effect`, and return a result that carries a `Design` record naming its assumptions in words, the role `Identification` plays on the graph path. Those assumptions are untestable from the data alone. They're what you defend when you present the number.

## Difference-in-differences

Two groups, two eras. Treated units change for two reasons, the calendar and the treatment; comparison units change for the calendar alone. Subtract the comparison group's change from the treated group's change and the calendar drops out, along with every fixed difference between the groups, however large. What's left is the effect of adoption on the adopters, provided both groups would have moved in parallel without it. That clause is parallel trends, and the rest of this section is machinery for using it and probing it.

```python
import numpy as np
import pandas as pd
from ergodic import panel, did

rng = np.random.default_rng(0)
n_stores, n_weeks = 120, 12
store = np.repeat(np.arange(n_stores), n_weeks)
week = np.tile(np.arange(n_weeks), n_stores)
adopter = np.arange(n_stores) < 40                 # the adopters sit 2.0 higher
level = rng.normal(size=n_stores) + 2.0 * adopter
season = rng.normal(scale=0.5, size=n_weeks)       # one calendar for every store
promo = (adopter[store] & (week >= 6)).astype(float)
sales = level[store] + season[week] + 3.0 * promo + rng.normal(scale=0.5, size=store.size)
data = panel(
    pd.DataFrame({"store": store, "week": week, "promo": promo, "sales": sales}),
    entity="store", time="week",
)

est = did(data, "sales", "promo", random_state=0)
est.flavor           # 'canonical': every adopter switched in the same week
est.value            # ~2.9, against a planted 3.0
est.interval()       # (~2.8, ~3.0)
est.design.estimand  # 'ATT identified by parallel trends against never_treated units'
```

The treatment column is the design's input: 0/1 or boolean, turning on at adoption and staying on. `did` reads each unit's adoption date off its first 1, takes the units that never show one as the comparison pool, and raises on a treatment that switches back off, because adoption timing needs an absorbing treatment. The naive cross-section here (promo stores minus the rest, post weeks only) lands near 4.7: the adopters were better stores before the promo existed. Differencing each store against its own past is what removes that.

`strategy="auto"` picks the flavor from the adoption dates it finds: one date gives `canonical`, the 2x2 above; more than one gives `group_time`, staggered adoption in the style of Callaway and Sant'Anna (2021). Group-time computes one ATT(g, t) per adoption cohort and period, differences each against the period just before that cohort adopts, and aggregates the post-adoption cells with cohort-size weights.

```python
rng = np.random.default_rng(0)
adopt = np.where(np.arange(n_stores) < 40, 4, np.where(np.arange(n_stores) < 80, 8, -1))
level = rng.normal(size=n_stores) + 2.0 * (adopt >= 0)
season = rng.normal(scale=0.5, size=n_weeks)
on = (adopt[store] >= 0) & (week >= adopt[store])
e = np.where(on, week - adopt[store], 0)           # weeks since adoption
sales = (
    level[store] + season[week]
    + np.where(on, 1.0 + 0.5 * e, 0.0)             # the effect grows as it runs
    + rng.normal(scale=0.5, size=store.size)
)
staggered = panel(
    pd.DataFrame({"store": store, "week": week, "promo": on.astype(float), "sales": sales}),
    entity="store", time="week",
)

est = did(staggered, "sales", "promo", random_state=0)
est.flavor            # 'group_time'
est.value             # ~2.3; the planted estimand is 2.25
est.by_group_time[0]  # GroupTimeEffect(group=4, period=0, event_time=-4, estimate=0.18, ...)
```

A third flavor, `strategy="twfe"`, is the static two-way fixed-effects regression. It's offered for reference and `auto` never picks it: under staggered adoption with effects that change over time, TWFE mixes the cell effects with weights that can go negative (Goodman-Bacon 2021), because already-treated cohorts end up serving as controls for later ones. On the staggered world above it returns ~1.8 where the estimand is 2.25, and the planted effect grows with time since adoption, which is exactly the case that breaks it.

When every unit eventually adopts, there's no never-treated pool and the default comparison raises. Pass `comparison="not_yet_treated"` to admit units that adopt only after the periods a cell reads:

```python
did(staggered, "sales", "promo", comparison="not_yet_treated")
```

Inference is a unit bootstrap throughout: resample units with replacement, recompute, keep the draws. Resampling whole units rather than rows is what respects the correlation inside each unit's series. The result is a `DiDEffect`: `value`, `std_error`, and `interval()` read the headline ATT, `by_group_time` holds the per-cell detail, `by_event_time` fills when `event_study` builds the result, and `design` carries the assumptions in words:

```python
est.design.assumptions
# ('parallel trends: untreated outcomes of treated and comparison units move in parallel',
#  'no anticipation: outcomes do not respond before adoption')
```

Controls enter as pre-period unit means (`controls=["x"]`), which weakens the requirement to parallel trends conditional on those covariates. Validation is loud: a missing value in the outcome or controls raises, a treatment that isn't 0/1 raises, and a canonical-flavor unit lacking a pre or a post observation is dropped with a warning and counted in `diagnostics["n_dropped_units"]`.

## Event studies

The same group-time cells, read along the event-time axis: e counts periods since adoption, with e = 0 the adoption period itself. `event_study` aggregates the cells that share an event time, cohort-size weighted, so the effect becomes a path instead of one number. `leads=` and `lags=` bound the reported window.

```python
from ergodic import event_study

es = event_study(staggered, "sales", "promo", leads=3, lags=4, random_state=0)
es.value    # the overall post-adoption ATT, same as did() under group_time
for entry in es.by_event_time:
    print(entry.event_time, round(entry.estimate, 2), entry.interval)
# -3   0.13  (-0.07, 0.34)
# -2  -0.07  (-0.28, 0.14)
#  0   1.10  ( 0.88, 1.32)
#  1   1.56  ( 1.33, 1.79)
#  2   1.99  ( 1.77, 2.21)
#  3   2.65  ( 2.46, 2.85)
#  4   3.05  ( 2.73, 3.44)
```

e = -1 is missing on purpose. Every cell differences against the period just before adoption, so the effect there is zero by construction; it's the reference, and reporting it would dress a definition up as a finding. The leads are the visible part of parallel trends: nobody is treated before adoption, so a lead whose interval sits away from zero means the groups were already diverging, and the design fails where you can watch it. Here both leads cover zero and the lags climb the way the planted effect does (the truth at event time e is 1 + 0.5e). Clean leads don't prove parallel trends, since trends can split exactly at adoption; a dirty lead does refute them.

## Synthetic control

One treated unit, and no single comparison is good enough: a region runs a campaign, and every other region differs from it in level or trend. Synthetic control builds the comparison instead of choosing it. Nonnegative donor weights that sum to one are fit so the weighted donor outcome tracks the treated unit over the pre-period, and the post-period gap between the observed path and the synthetic one is the effect.

```python
from ergodic import synthetic_control

# 25 markets over 40 months; austin starts a campaign in month 30 worth 4.0
sc = synthetic_control(markets, "revenue", treated="austin", at=30)
sc.value        # ~4.0: the mean post-period gap
sc.weights      # {'m01': 0.44, 'm12': 0.20, 'm19': 0.11, ...}
sc.pre_rmspe    # ~0.06: how well the synthetic tracked before the campaign
sc.p_value      # 0.04: the placebo rank
sc.std_error    # None, on purpose
```

You can name the treated unit and period directly, as above, or pass `treatment="campaign"`, an absorbing 0/1 column that marks exactly one unit; its first treated period becomes `at`. `donors=` restricts the pool, and by default every other unit donates. The fit is deterministic: projected gradient descent on the squared pre-period tracking error, over the simplex.

Judge `pre_rmspe` before believing anything else. The claim is that the synthetic path is what the treated unit would have done untreated, and the pre-period is the only place that claim ever touches data; a synthetic control that tracked the past badly has no standing to forecast the counterfactual. The simplex constraint is the assumption with teeth: the weights are nonnegative and sum to one, so the synthetic path can never leave the donors' range, and the treated unit's untreated path has to lie in their convex span. A unit that's the largest or fastest-growing of its kind has no convex representation, and the pre-period fit will usually tell you so.

There's no standard error because there's no sampling story: one treated unit, a fixed donor pool, nothing resampled. `std_error` stays `None` on purpose, and inference is the placebo permutation of Abadie, Diamond, and Hainmueller instead. Each donor is refit as if it were the treated unit (the real one excluded), every run gets a post-to-pre RMSPE ratio, and `p_value` ranks the treated ratio among the placebos. With 24 donors and the treated unit ranking first, p = 1/25 = 0.04, which is also as small as 24 donors can deliver. `interval()` is a placebo band rather than a sampling interval: its half width is the gap a unit with no treatment can show, read from the placebo runs.

## Interrupted time series

One series and one break you can date: a redesign ships, a policy starts. Segmented regression fits a level and a trend plus a jump and a slope change at the break, and the effect is the gap between the fitted post-period and the pre-trend projected forward.

```python
from ergodic import timeseries, interrupted_time_series

# 104 weeks of site visits with a quarterly cycle; a redesign ships in week 60
ts = timeseries(visits_frame, time="week")
est = interrupted_time_series(ts, "visits", at=60, seasonality=13)
est.level_change    # ~7.8: the jump at the break (planted: 8.0)
est.slope_change    # ~0.47: the change in trend per week (planted: 0.5)
est.value           # ~17.8: the ATT, the mean post-period gap to the projected pre-trend
est.std_error       # ~0.9
```

The design matrix is `[1, t, post, post * (t - t0)]`, plus Fourier sine and cosine pairs when you pass `seasonality=` (the cycle length in rows; 13 here for a quarterly cycle in weekly data), so a seasonal swing doesn't get billed to the break. Standard errors are Newey-West (HAC) with Bartlett weights, because the residuals of one series autocorrelate and plain OLS standard errors would be too small; `hac_lags` defaults to `0.75 * T**(1/3)`, rounded down. The `counterfactual` path zeroes the break terms and keeps the trend and seasonal ones, and both sides of the break need at least 3 points.

The caveat that should travel with every ITS number: there is no control series, so the design can't tell the break from anything else that happened at the same time. A co-timed shock, say a holiday or a competitor's outage in the same week, lands in `level_change` with nothing in the data to separate it. The `Design` record states it ("no co-timed shock at the break"), and it's the assumption you argue from facts about the week, because the series can't.

## Choosing a design

With many units and a clean adoption date, use difference-in-differences: the comparison units carry the calendar, the within-unit differencing carries the levels, and the event-study leads give the assumption a check you can show. With one treated unit and a pool of plausible donors, use synthetic control: it builds the comparison the data can support, and the placebo runs price the result. With one series and nothing else, interrupted time series is what remains, and it's the weakest of the three. Its counterfactual is an extrapolation of the unit's own past, its pre-period fit has no untreated benchmark, and a co-timed shock is invisible to it. Reach for it when the break is sharp, the series is long enough to pin the trend and the season, and you can rule out competing events from knowledge of the domain, because the data won't do it for you.
