# SPDX-License-Identifier: Apache-2.0
"""Simpson's paradox: did Berkeley graduate admissions favor men in 1973?

Pooled across departments, men were admitted at a higher rate than women, which
reads as bias. But department is a confounder: women applied more often to the
competitive departments. Condition on it and the gap reverses to slightly favor
women. This is the textbook reason to adjust for a pre-treatment common cause.

Run with: uv run python examples/confounding_berkeley.py
"""

from ergodic import dag, estimate_effect
from ergodic.datasets import load_berkeley


def main() -> None:
    ds = load_berkeley()

    naive = estimate_effect(
        dag([f"{ds.treatment} -> {ds.outcome}"]),
        ds.data,
        ds.treatment,
        ds.outcome,
        treated=ds.treated,
    )
    adjusted = estimate_effect(
        dag(["dept -> gender", "dept -> admit", "gender -> admit"]),
        ds.data,
        ds.treatment,
        ds.outcome,
        treated=ds.treated,
    )

    print(ds.DESCR)
    print(f"pooled gap (Female minus Male):  {naive.value:+.3f}")
    print(f"adjusted for department:         {adjusted.value:+.3f}")
    print(f"the sign flips: {naive.value:+.3f} to {adjusted.value:+.3f}")


if __name__ == "__main__":
    main()
