# SPDX-License-Identifier: Apache-2.0
"""Backdoor adjustment: does maternal smoking lower infant birthweight?

The naive difference in means overstates the effect, because smoking mothers
differ in ways that also affect birthweight. Adjusting for those confounders with
a doubly-robust estimator moves the number toward the literature figure of about
-203 grams.

Run with: uv run python examples/adjustment_birthweight.py
"""

from ergodic import dag, estimate_effect
from ergodic.datasets import load_birthweight


def main() -> None:
    ds = load_birthweight()

    # The adjustment graph: every confounder points at the treatment and the
    # outcome, and smoking points at birthweight.
    edges = [f"{c} -> {ds.treatment}" for c in ds.confounders]
    edges += [f"{c} -> {ds.outcome}" for c in ds.confounders]
    edges.append(f"{ds.treatment} -> {ds.outcome}")
    graph = dag(edges)

    naive = estimate_effect(
        dag([f"{ds.treatment} -> {ds.outcome}"]),
        ds.data,
        ds.treatment,
        ds.outcome,
        treated=ds.treated,
    )
    adjusted = estimate_effect(
        graph, ds.data, ds.treatment, ds.outcome, treated=ds.treated, strategy="aipw"
    )

    print(ds.DESCR)
    print(f"naive difference:   {naive.value:8.1f} g")
    print(f"adjusted (AIPW):    {adjusted.value:8.1f} g")
    print(f"literature target:  {ds.target['value']:8.1f} g  ({ds.target['source']})")


if __name__ == "__main__":
    main()
