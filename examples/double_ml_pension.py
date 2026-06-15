# SPDX-License-Identifier: Apache-2.0
"""Double machine learning: the effect of 401(k) eligibility on net wealth.

Eligibility is as good as random once you condition on income, but income also
drives wealth, so the naive gap of about $19,500 is badly inflated. Double machine
learning with a flexible learner partials income out and lands near $9,000, the
published figure. A linear nuisance under-adjusts (closer to $6,000) because the
income effect is nonlinear.

The same table also supports an instrumental-variables reading, with eligibility
as the instrument for actual participation (ds.instrument and ds.participation).

Run with: uv run python examples/double_ml_pension.py
"""

from ergodic import dag, estimate_effect
from ergodic.datasets import load_pension401k


def main() -> None:
    ds = load_pension401k()

    edges = [f"{c} -> {ds.treatment}" for c in ds.confounders]
    edges += [f"{c} -> {ds.outcome}" for c in ds.confounders]
    edges.append(f"{ds.treatment} -> {ds.outcome}")
    graph = dag(edges)

    naive = estimate_effect(
        dag([f"{ds.treatment} -> {ds.outcome}"]), ds.data, ds.treatment, ds.outcome
    )
    dml = estimate_effect(
        graph,
        ds.data,
        ds.treatment,
        ds.outcome,
        strategy="dml",
        learner="knn",
        random_state=0,
    )

    print(ds.DESCR)
    print(f"naive gap:                {naive.value:9.0f} USD")
    print(f"double ML (eligibility):  {dml.value:9.0f} USD")
    print(f"literature target:        {ds.target['value']:9.0f} USD  ({ds.target['source']})")


if __name__ == "__main__":
    main()
