# SPDX-License-Identifier: Apache-2.0
"""Root-cause attribution: which upstream step is to blame for one bad outcome.

A single unit comes out wrong: a batch fails, a day's revenue craters, a request runs
slow. The metric you can see is the one at the end of the chain, but the cause is
somewhere upstream, and the symptom has already propagated. Correlation drill-downs
light up every downstream step at once, because they all moved together.

The SCM attributes the anomaly to the step whose *own* shock caused it, by a
counterfactual that cancels one step's noise at a time. For each node it asks: had this
node taken its normal value given its parents (its own idiosyncratic shock removed, the
upstream left as it was), how much of the bad outcome goes away? The node that
restores the outcome is the culprit. A node that is merely a victim of an upstream shock
does not, because cancelling its own (normal) noise changes nothing.

Here a four-step chain A -> B -> C -> Output runs normally, except one unit takes a large
negative shock at B. B and C and the Output are all low for that unit, so a correlational
view blames all three. The counterfactual cleanly fingers B.

"Cancel the shock at a node" here means setting it to its normal value given its factual
parents, which removes that node's own residual while leaving the upstream as it was. The
separation is starkest when the anomaly's root sits at its normal value, which this unit
does by construction; the method still points at the true culprit otherwise, less sharply.

Run with: uv run python examples/scm/09_root_cause_attribution.py
"""

from __future__ import annotations

import logging

import numpy as np

from ergodic import dag, tabular
from ergodic.scm import SCM, SCMResult


def _quiet_pymc() -> None:
    for name in ("pymc", "pymc.sampling"):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_pymc()

    rng = np.random.default_rng(9)
    n = 3000
    a = rng.normal(0, 1, n)
    b = 1.0 * a + rng.normal(0, 0.4, n)
    c = 1.0 * b + rng.normal(0, 0.4, n)
    output = 1.0 * c + rng.normal(0, 0.4, n)

    # inject a large negative shock at B for ONE unit; it propagates to C and Output.
    # pick a unit whose root A sits at its normal value, so the anomaly is purely B's.
    culprit = int(np.argmin(np.abs(a)))
    shock = -5.0
    b[culprit] += shock
    c[culprit] += shock  # B's shock flows downstream...
    output[culprit] += shock  # ...all the way to the Output.

    data = tabular({"A": a, "B": b, "C": c, "Output": output})
    graph = dag(["A -> B", "B -> C", "C -> Output"])
    scm = SCM(graph, data, draws=400, tune=400, chains=2, random_state=9).fit()

    unit = data.take([culprit])
    factual = {"A": float(a[culprit]), "B": float(b[culprit]), "C": float(c[culprit])}
    output_obs = float(output[culprit])
    normal_output = float(np.median(output))

    parents = {"A": [], "B": ["A"], "C": ["B"], "Output": ["C"]}

    print(__doc__.strip().splitlines()[0])
    print("=" * 70)
    print(f"Unit #{culprit}: Output = {output_obs:.2f}, normal is about {normal_output:.2f}\n")
    print(f"{'cancel the shock at':<22}{'restores Output to':>20}{'recovered':>12}")
    print("-" * 70)
    recovery = {}
    for node in ("A", "B", "C"):
        # the node's NORMAL value given its actual parents (its own shock removed).
        pa = {p: factual[p] for p in parents[node]}
        denoised = (
            scm.interventional(node, pa).mean() if pa else scm.interventional(node, {}).mean()
        )
        cf = scm.counterfactual(unit, {node: denoised}, "Output", use_draws=False)
        assert isinstance(cf, SCMResult)
        recovery[node] = cf.mean() - output_obs
        print(f"{node:<22}{cf.mean():>20.2f}{recovery[node]:>+12.2f}")
    print("=" * 70)
    culprit_node = max(recovery, key=recovery.get)
    runner_up = sorted(recovery.values())[-2]
    rec_b = recovery["B"]
    print(
        f"Root cause: {culprit_node}. Cancelling B's shock recovers {rec_b:+.2f} of Output;\n"
        f"cancelling A or C recovers at most {runner_up:+.2f}. C is low too, but only because it\n"
        f"inherited B's shock, so de-noising C (a normal C-shock) changes nothing. The\n"
        f"counterfactual separates the cause from its symptoms."
    )


if __name__ == "__main__":
    main()
