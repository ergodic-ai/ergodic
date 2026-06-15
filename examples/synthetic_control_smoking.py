# SPDX-License-Identifier: Apache-2.0
"""Synthetic control: did California's Proposition 99 cut cigarette sales?

There is one treated unit and no control group to difference against, so we build
a synthetic California out of a weighted combination of other states that tracks
the real series before 1989. The post-1989 gap is the effect, and a placebo
permutation over the donor states tells us whether it is large by chance.

Run with: uv run python examples/synthetic_control_smoking.py
"""

from ergodic import synthetic_control
from ergodic.datasets import load_smoking


def main() -> None:
    ds = load_smoking()
    sc = synthetic_control(ds.data, ds.outcome, treated=ds.treated, at=ds.at)

    print(ds.DESCR)
    print(f"mean post-period gap:  {sc.value:6.1f} packs per capita")
    print(f"pre-period fit (rmspe): {sc.pre_rmspe:5.2f}")
    print(f"placebo p-value:       {sc.p_value:6.3f}")
    top = sorted(sc.weights.items(), key=lambda kv: -kv[1])[:5]
    print("top donor weights:     " + ", ".join(f"{unit} {weight:.2f}" for unit, weight in top))


if __name__ == "__main__":
    main()
