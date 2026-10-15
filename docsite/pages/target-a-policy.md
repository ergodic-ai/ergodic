# Target a policy

Learn a short, readable rule that says whom to treat, and measure how much it gains
over treating no one. The example uses a simulation in which the
treatment helps some people and harms others.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - Per-unit effects from `cate`, as in
      [Estimate heterogeneous effects](heterogeneous-effects.md).

## Run the program

In the simulation a coaching programme helps only people with both `need` and `age`
above 0. Everyone else is slightly or clearly worse off. On average it harms. The
program learns the rule on one half of the people and measures it on the other.

```python file="target_policy.py"
import numpy as np
import pandas as pd
from ergodic import cate, dag, policy_tree, tabular

# 1. Simulate 4,000 people. The effect is +0.8 when need > 0 and age > 0, else negative.
rng = np.random.default_rng(0)
n = 4000
age = rng.normal(0, 1, n)
need = rng.normal(0, 1, n)
income = rng.normal(0, 1, n)
coached = rng.binomial(1, 1 / (1 + np.exp(-age)))
true_effect = 1.0 * (need > 0) + 1.0 * (age > 0) - 1.2
outcome = age + need + income + true_effect * coached + rng.normal(0, 1, n)
frame = pd.DataFrame(
    {"age": age, "need": need, "income": income, "coached": coached, "outcome": outcome}
)
data = tabular(frame, types={"coached": "categorical"})
graph = dag(
    [
        "age -> coached",
        "income -> coached",
        "age -> outcome",
        "need -> outcome",
        "income -> outcome",
        "coached -> outcome",
    ]
)

# 2. Split the rows in half: one half learns the rule, the other evaluates it.
learn = np.arange(n) < n // 2
train, test = data.take(np.flatnonzero(learn)), data.take(np.flatnonzero(~learn))

# 3. Learn the best depth-2 rule from the per-unit effects on the first half.
effects = cate(graph, train, "coached", "outcome", random_state=0, uncertainty=False)
tree = policy_tree(effects, depth=2)
print(tree.describe())
print(f"gain over treating no one, in sample: {tree.value:.3f}")
print(f"best possible gain, learning half: {np.mean(np.maximum(true_effect[learn], 0)):.3f}")

# 4. Evaluate the rule on the held-out half, with fresh doubly robust scores.
held_out = cate(graph, test, "coached", "outcome", random_state=0, uncertainty=False)
value = tree.evaluate(held_out)
low, high = value.interval(0.95)
print(f"gain over treating no one, held out: {value.value:.3f} [{low:.3f}, {high:.3f}]")
x_test = test.to_pandas()[list(tree.covariates)].to_numpy()
treat = tree.predict(x_test)
print(f"held-out share treated: {treat.mean():.2f}")

# 5. Check the rule against the truth on the held-out half.
truth = true_effect[~learn]
print(f"true gain: {np.mean(treat * truth):.3f}")
print(f"best possible gain, held-out half: {np.mean(np.maximum(truth, 0)):.3f}")
```

```output from="target_policy.py"
if need <= -0.01998:
  if age <= -2.295:
    -> treat=1 (n=13)
  else:  # age > -2.295
    -> treat=0 (n=982)
else:  # need > -0.01998
  if age <= -0.04976:
    -> treat=0 (n=502)
  else:  # age > -0.04976
    -> treat=1 (n=503)
gain over treating no one, in sample: 0.204
best possible gain, learning half: 0.190
gain over treating no one, held out: 0.143 [0.095, 0.192]
held-out share treated: 0.26
true gain: 0.188
best possible gain, held-out half: 0.195
```

- **The rule.** The tree splits on `need` near 0 and then on `age`, the two
  variables that drive the effect. One small leaf, 13 people with very low `age`,
  is noise.
- **In sample.** On the half it was learned from, the rule seems to add 0.20 per
  person over treating no one. That is above even the best possible gain on that
  half, 0.190, because the tree was chosen to make exactly those scores look good.
- **Held out.** On the other half the estimate is 0.14, with a 95% interval
  from 0.10 to 0.19. Use this number.
- **Truth.** The rule's true gain on the held-out half is 0.19, inside the
  interval and close to the best any rule could reach.

## Measure a rule on held-out data

The tree is picked to maximize its own scores, so its value on those scores is
optimistic: a winner's curse. `tree.value` is the gain over treating no one in
outcome units, but `tree.value` and `tree.interval()` are valid only for a rule fixed
in advance. To measure a learned rule, score it on rows it never saw:

```python
held_out = cate(graph, test, "coached", "outcome", random_state=0, uncertainty=False)
value = tree.evaluate(held_out)
low, high = value.interval(0.95)
```

The held-out `cate` result carries one doubly robust score per person: a noisy signal
of their effect. `tree.evaluate` averages the score over the people the rule treats,
with 0 for the rest, which estimates the gain over treating no one, in outcome units.
It returns a `PolicyValue` with `value`, `std_error`, `interval()`, `treat_all_value`
and `regret()`. Its standard error is valid because the rule is fixed before these
rows are used. The held-out scores must carry the tree's split covariates; `evaluate`
picks them by name.

## How the tree is chosen

`policy_tree(effects, covariates=None, *, depth=2)` reads the doubly robust scores
that `cate` stores on its result (Athey and Wager, 2021). A leaf treats its people
when their scores sum to more than 0. The tree maximizes the total over leaves.

- At `depth=1` or `depth=2` the search tries every split on every covariate, so the
  tree is the best of its depth on the learning rows (Zhou, Athey and Wager, 2023).
  `tree.exhaustive` is `True`. The search time grows quickly with rows and
  covariates.
- At `depth=3` or more the search is greedy, one split at a time, and may miss the
  best tree.
- `covariates=` limits the splits to named columns, for example
  `policy_tree(effects, ["need"], depth=1)`.
- The splits are a point estimate with no interval. Prefer shallow trees, and check
  their splits with domain knowledge.

<details markdown="1">
<summary>Apply and inspect the rule</summary>

- `tree.assign()` returns the action (0 or 1) for each unit the tree was fit on.
- `tree.predict(x)` takes a NumPy array with one column per name in
  `tree.covariates`, in that order.
- `tree.leaves()` lists the leaves with their action and size.
- `tree.regret("treat_all")`, `("treat_none")` or `("best_trivial")` returns
  `value` minus that baseline's value, in outcome units and in sample (so optimistic,
  like `value`). `tree.treat_none_value` is 0 and `tree.treat_all_value` is the
  average effect.
- `effects.explain().policy_tree(depth=2)` builds the same tree.

</details>

## Next steps

<div class="cards" markdown="1">

- [Estimate heterogeneous effects](heterogeneous-effects.md): the per-unit effects the tree reads.
- [Refute an estimate](refute-an-estimate.md): stress-test the average effect first.
- [Python reference](reference-inference.md): every option of `policy_tree`.

</div>
