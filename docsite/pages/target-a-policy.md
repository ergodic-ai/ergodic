# Target a policy

Learn a short, readable rule that says whom to treat, and estimate how much better
it does than treating everyone or no one. The example uses a simulation in which the
treatment helps some people and harms others.

!!! need "You'll need"
    - ergodic, installed with `pip install "ergodic"`.
    - Per-unit effects from `cate`, as in
      [Estimate heterogeneous effects](heterogeneous-effects.md).

## Run the program

In the simulation a coaching programme helps only people with both `need` and `age`
above 0. Everyone else is slightly or clearly worse off. On average it harms.

```python file="target_policy.py"
import numpy as np
import pandas as pd
from ergodic import cate, dag, policy_tree, tabular

# 1. Simulate 2,000 people. The effect is +0.8 when need > 0 and age > 0, else negative.
rng = np.random.default_rng(0)
n = 2000
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

# 2. Estimate the per-unit effects. The tree needs their scores, not their intervals.
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
effects = cate(graph, data, "coached", "outcome", random_state=0, uncertainty=False)

# 3. Learn the best depth-2 rule.
tree = policy_tree(effects, depth=2)
print(tree.describe())

# 4. Read its value against the two trivial rules.
low, high = tree.interval()
print(f"rule:        {tree.value:.3f}  (90% interval {low:.3f} to {high:.3f})")
print(f"treat all:   {tree.treat_all_value:.3f}")
print(f"treat none:  {tree.treat_none_value:.3f}")
gain = (tree.value - tree.treat_none_value) / 2
print(f"estimated gain over treating no one: {gain:.3f} per person")

# 5. Check the rule against the truth.
treat = tree.assign()
print(f"share treated: {treat.mean():.2f}")
print(f"true gain: {np.mean(treat * true_effect):.3f}")
print(f"best possible gain: {np.mean(np.maximum(true_effect, 0)):.3f}")
```

```output from="target_policy.py"
if need <= -0.0101:
  if age <= 1.358:
    -> treat=0 (n=921)
  else:  # age > 1.358
    -> treat=1 (n=93)
else:  # need > -0.0101
  if age <= -0.1285:
    -> treat=0 (n=465)
  else:  # age > -0.1285
    -> treat=1 (n=521)
rule:        0.660  (90% interval 0.574 to 0.745)
treat all:   -0.238
treat none:  0.238
estimated gain over treating no one: 0.211 per person
share treated: 0.31
true gain: 0.172
best possible gain: 0.187
```

- **The rule.** The tree splits on `need` near 0 and then on `age`, the two
  variables that drive the effect. It treats about 3 people in 10.
- **Value.** `value` takes each person's doubly robust score, a noisy
  signal of their effect, with a plus sign when the rule treats them and a minus
  sign when it does not, then averages. Treating
  everyone scores the average effect; treating no one scores its negative.
- **Gain.** Half the difference between two values is the gain in outcome units.
  The rule adds about 0.21 per person over treating no one.
- **Truth.** The rule's true gain is close to the best any rule could reach.

## How the tree is chosen

`policy_tree(effects, covariates=None, *, depth=2)` reads the doubly robust scores
that `cate` stores on its result. A leaf treats its people when their scores sum
to more than 0. The tree maximizes the total over leaves.

- At `depth=1` or `depth=2` the search tries every split on every covariate, so the
  tree is the best of its depth. `tree.exhaustive` is `True`. The search time grows
  quickly with rows and covariates.
- At `depth=3` or more the search is greedy, one split at a time, and may miss the
  best tree.
- `covariates=` limits the splits to named columns, for example
  `policy_tree(effects, ["need"], depth=1)`.

!!! caution "Trust the value, not the exact splits"
    The value and its interval are averages of per-person scores, so they have a
    valid standard error. The splits do not. Here one leaf treats 93 people with
    `need` below 0 and `age` above 1.36, where the true effect is negative: the
    tree fit noise. Prefer shallow trees and check them with domain knowledge.

<details markdown="1">
<summary>Apply and inspect the rule</summary>

- `tree.assign()` returns the action (0 or 1) for each unit the tree was fit on.
- `tree.predict(x)` takes a NumPy array with one column per name in
  `tree.covariates`, in that order.
- `tree.leaves()` lists the leaves with their action and size.
- `tree.regret("treat_all")`, `("treat_none")` or `("best_trivial")` returns
  `value` minus that baseline's value, on the same doubled scale as the values.
- `effects.explain().policy_tree(depth=2)` builds the same tree.

</details>

## Next steps

<div class="cards" markdown="1">

- [Estimate heterogeneous effects](heterogeneous-effects.md): the per-unit effects the tree reads.
- [Refute an estimate](refute-an-estimate.md): stress-test the average effect first.
- [Python reference](reference-inference.md): every option of `policy_tree`.

</div>
